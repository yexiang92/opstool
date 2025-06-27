from typing import Optional

import numpy as np
import openseespy.opensees as ops
import xarray as xr

from ...utils import suppress_ops_print
from ._response_base import ResponseBase, _expand_to_uniform_array

ELASTIC_BEAM_CLASSES = [3, 4, 5, 5001, 145, 146, 63, 631]


class FrameRespStepData(ResponseBase):
    def __init__(
        self,
        ele_tags=None,
        ele_load_data=None,
        elastic_frame_sec_points: int = 7,
        model_update: bool = False,
        dtype: Optional[dict] = None,
    ):
        self.resp_names = [
            "localForces",
            "basicForces",
            "basicDeformations",
            "plasticDeformation",
            "sectionForces",
            "sectionDeformations",
            "sectionLocs",
        ]
        self.resp_steps = None
        self.resp_steps_list = []  # for model update
        self.resp_steps_dict = {}  # for non-update
        self.step_track = 0
        self.ele_tags = ele_tags
        self.ele_load_data = ele_load_data
        self.times = []

        self.elastic_frame_sec_points = elastic_frame_sec_points
        self.model_update = model_update
        self.dtype = {"int": np.int32, "float": np.float32}
        if isinstance(dtype, dict):
            self.dtype.update(dtype)

        self.localDofs = ["FX1", "FY1", "FZ1", "MX1", "MY1", "MZ1", "FX2", "FY2", "FZ2", "MX2", "MY2", "MZ2"]
        self.basicDofs = ["N", "MZ1", "MZ2", "MY1", "MY2", "T"]
        self.secDofs = ["N", "MZ", "VY", "MY", "VZ", "T"]
        self.secPoints = None
        self.sec_loc_dofs = None
        self.attrs = {
            "localDofs": "local coord system dofs at end 1 and end 2",
            "basicDofs": "basic coord system dofs at end 1 and end 2",
            "secPoints": "section points No.",
            "secDofs": "section forces and deformations Dofs. "
            "Note that the section DOFs are only valid for <Elastic Section>, "
            "<Elastic Shear Section>, and <Fiber Section>. "
            "For <Aggregator Section>, you should carefully check the data, "
            "as it may not correspond directly to the DOFs.",
            "Notes": "Note that the deformations are displacements and rotations in the basicDofs;"
            "And strains and curvatures in the secDofs",
        }

        self.initialize()

    def initialize(self):
        self.resp_steps = None
        self.resp_steps_list = []
        for name in self.resp_names:
            self.resp_steps_dict[name] = []
        self.secPoints = None
        self.sec_loc_dofs = None

        self.add_data_one_step(self.ele_tags, self.ele_load_data)
        self.times = [0.0]
        self.step_track = 0

    def reset(self):
        self.initialize()

    def add_data_one_step(self, ele_tags, ele_load_data):
        local_forces = _get_beam_local_force(ele_tags, ("localForces", "localForce"), dtype=self.dtype)
        basic_forces = _get_beam_basic_resp(ele_tags, ("basicForce", "basicForces"), dtype=self.dtype)
        basic_defos = _get_beam_basic_resp(
            ele_tags,
            (
                "basicDeformation",
                "basicDeformations",
                "chordRotation",
                "chordDeformation",
                "deformations",
            ),
            dtype=self.dtype,
        )
        plastic_defos = _get_beam_basic_resp(ele_tags, ("plasticRotation", "plasticDeformation"), dtype=self.dtype)
        sec_f, sec_d, sec_locs = _get_beam_sec_resp(
            ele_tags, ele_load_data, local_forces, basic_defos, self.elastic_frame_sec_points, dtype=self.dtype
        )
        if self.sec_loc_dofs is None:
            if sec_locs.shape[-1] == 2:
                self.sec_loc_dofs = ["alpha", "X"]
            elif sec_locs.shape[-1] == 3:
                self.sec_loc_dofs = ["alpha", "X", "Y"]
            elif sec_locs.shape[-1] == 4:
                self.sec_loc_dofs = ["alpha", "X", "Y", "Z"]
            else:
                self.sec_loc_dofs = [f"loc{i + 1}" for i in range(sec_locs.shape[-1])]
        if self.secPoints is None:
            self.secPoints = np.arange(sec_locs.shape[1]) + 1

        if self.model_update:
            data_vars = {}
            data_vars["localForces"] = (["eleTags", "localDofs"], local_forces)
            data_vars["basicForces"] = (["eleTags", "basicDofs"], basic_forces)
            data_vars["basicDeformations"] = (["eleTags", "basicDofs"], basic_defos)
            data_vars["plasticDeformation"] = (["eleTags", "basicDofs"], plastic_defos)
            data_vars["sectionForces"] = (["eleTags", "secPoints", "secDofs"], sec_f)
            data_vars["sectionDeformations"] = (["eleTags", "secPoints", "secDofs"], sec_d)
            data_vars["sectionLocs"] = (["eleTags", "secPoints", "locs"], sec_locs)
            ds = xr.Dataset(
                data_vars=data_vars,
                coords={
                    "eleTags": ele_tags,
                    "localDofs": self.localDofs,
                    "basicDofs": self.basicDofs,
                    "secPoints": self.secPoints,
                    "secDofs": self.secDofs,
                    "locs": self.sec_loc_dofs,
                },
                attrs=self.attrs,
            )
            self.resp_steps_list.append(ds)
        else:
            datas = [local_forces, basic_forces, basic_defos, plastic_defos, sec_f, sec_d, sec_locs]
            for name, da in zip(self.resp_names, datas):
                self.resp_steps_dict[name].append(da)

        self.times.append(ops.getTime())
        self.step_track += 1

    def _to_xarray(self):
        if self.model_update:
            self.resp_steps = xr.concat(self.resp_steps_list, dim="time", join="outer")
            self.resp_steps.coords["time"] = self.times
        else:
            data_vars = {}
            data_vars["localForces"] = (["time", "eleTags", "localDofs"], self.resp_steps_dict["localForces"])
            data_vars["basicForces"] = (["time", "eleTags", "basicDofs"], self.resp_steps_dict["basicForces"])
            data_vars["basicDeformations"] = (
                ["time", "eleTags", "basicDofs"],
                self.resp_steps_dict["basicDeformations"],
            )
            data_vars["plasticDeformation"] = (
                ["time", "eleTags", "basicDofs"],
                self.resp_steps_dict["plasticDeformation"],
            )
            data_vars["sectionForces"] = (
                ["time", "eleTags", "secPoints", "secDofs"],
                self.resp_steps_dict["sectionForces"],
            )
            data_vars["sectionDeformations"] = (
                ["time", "eleTags", "secPoints", "secDofs"],
                self.resp_steps_dict["sectionDeformations"],
            )
            data_vars["sectionLocs"] = (["time", "eleTags", "secPoints", "locs"], self.resp_steps_dict["sectionLocs"])
            self.resp_steps = xr.Dataset(
                data_vars=data_vars,
                coords={
                    "time": self.times,
                    "eleTags": self.ele_tags,
                    "localDofs": self.localDofs,
                    "basicDofs": self.basicDofs,
                    "secPoints": self.secPoints,
                    "secDofs": self.secDofs,
                    "locs": self.sec_loc_dofs,
                },
                attrs=self.attrs,
            )

    def get_data(self):
        return self.resp_steps

    def get_track(self):
        return self.step_track

    def save_file(self, dt: xr.DataTree):
        self._to_xarray()
        dt["/FrameResponses"] = self.resp_steps
        return dt

    @staticmethod
    def read_file(dt: xr.DataTree, unit_factors: Optional[dict] = None):
        resp_steps = dt["/FrameResponses"].to_dataset()
        if unit_factors is not None:
            resp_steps = FrameRespStepData._unit_transform(resp_steps, unit_factors)
        return resp_steps

    @staticmethod
    def _unit_transform(resp_steps, unit_factors):
        force_factor = unit_factors["force"]
        moment_factor = unit_factors["moment"]
        curvature_factor = unit_factors["curvature"]
        disp_factor = unit_factors["disp"]

        # ---------------------------------------------------------
        resp_steps["localForces"].loc[{"localDofs": ["FX1", "FY1", "FZ1", "FX2", "FY2", "FZ2"]}] *= force_factor
        resp_steps["localForces"].loc[{"localDofs": ["MX1", "MY1", "MZ1", "MX2", "MY2", "MZ2"]}] *= moment_factor
        # ---------------------------------------------------------
        resp_steps["basicForces"].loc[{"basicDofs": ["N"]}] *= force_factor
        resp_steps["basicForces"].loc[{"basicDofs": ["MZ1", "MZ2", "MY1", "MY2", "T"]}] *= moment_factor
        resp_steps["basicDeformations"].loc[{"basicDofs": ["N"]}] *= disp_factor
        resp_steps["plasticDeformation"].loc[{"basicDofs": ["N"]}] *= disp_factor
        # ---------------------------------------------------------
        resp_steps["sectionForces"].loc[{"secDofs": ["N", "VY", "VZ"]}] *= force_factor
        resp_steps["sectionForces"].loc[{"secDofs": ["MZ", "MY", "T"]}] *= moment_factor
        resp_steps["sectionDeformations"].loc[{"secDofs": ["MZ", "MY", "T"]}] *= curvature_factor
        # --------------------------------------------------------
        resp_steps["sectionLocs"].loc[{"locs": ["X"]}] *= disp_factor
        if "Y" in resp_steps["sectionLocs"].coords["locs"]:
            resp_steps["sectionLocs"].loc[{"locs": ["Y"]}] *= disp_factor
        if "Z" in resp_steps["sectionLocs"].coords["locs"]:
            resp_steps["sectionLocs"].loc[{"locs": ["Z"]}] *= disp_factor

        return resp_steps

    @staticmethod
    def read_response(
        dt: xr.DataTree, resp_type: Optional[str] = None, ele_tags=None, unit_factors: Optional[dict] = None
    ):
        ds = FrameRespStepData.read_file(dt, unit_factors=unit_factors)
        if resp_type is None:
            if ele_tags is None:
                return ds
            else:
                return ds.sel(eleTags=ele_tags)
        else:
            if resp_type not in list(ds.keys()):
                raise ValueError(f"resp_type {resp_type} not found in {list(ds.keys())}")  # noqa: TRY003
            if ele_tags is not None:
                return ds[resp_type].sel(eleTags=ele_tags)
            else:
                return ds[resp_type]


def _get_beam_local_force(beam_tags, resp_types, dtype):
    local_forces = []
    for eletag in beam_tags:
        eletag = int(eletag)
        forces = []
        for name in resp_types:
            forces = ops.eleResponse(eletag, name)
            if len(forces) > 0:
                break
        if len(forces) == 0:
            forces = [0.0] * 12
        elif len(forces) == 6:
            forces = [
                forces[0],  # Fx
                forces[1],  # Fy
                0.0,  # Fz
                0.0,  # Mx
                0.0,  # My
                forces[2],  # Mz
                forces[3],
                forces[4],
                0.0,
                0.0,
                0.0,
                forces[5],
            ]
        local_forces.append(forces)
    return np.array(local_forces, dtype=dtype["float"])


def _get_beam_basic_resp(beam_tags, resp_types, dtype):
    basic_resps = []
    for ele_tag in beam_tags:
        ele_tag = int(ele_tag)
        resp = []
        for name in resp_types:
            resp = ops.eleResponse(ele_tag, name)
            if len(resp) > 0:
                break
        if len(resp) == 0:
            resp = [0.0] * 6
        elif len(resp) == 3:
            resp = [
                resp[0],  # N
                resp[1],  # MZ1
                resp[2],  # Mz2
                0.0,  # My1
                0.0,  # My2
                0.0,  # T
            ]
        basic_resps.append(resp)
    return np.array(basic_resps, dtype=dtype["float"])


def _get_beam_sec_resp(beam_tags, ele_load_data, local_forces, basic_disp, n_secs_elastic_beam, dtype):
    pattern_tags, load_eletags = _extract_pattern_info(ele_load_data)
    beam_secF, beam_secD, beam_locs = [], [], []

    beam_lengths, start_coords, end_coords = _get_ele_length(beam_tags)

    for eletag, length, local_f, basic_d in zip(beam_tags, beam_lengths, local_forces, basic_disp):
        eletag = int(eletag)
        if ops.getEleClassTags(eletag)[0] in ELASTIC_BEAM_CLASSES:
            xlocs = np.linspace(0, 1.0, n_secs_elastic_beam)
            sec_f = _get_elastic_sec_forces(eletag, length, ele_load_data, pattern_tags, load_eletags, local_f, xlocs)
            sec_d = _get_elastic_sec_defo(eletag, sec_f, basic_d, length, xlocs)
        else:
            xlocs, sec_f, sec_d = _get_nonlinear_section_response(eletag, length)

        beam_locs.append(np.array(xlocs))
        beam_secF.append(np.array(sec_f))
        beam_secD.append(np.array(sec_d))

    beam_locs = _expand_to_uniform_array(beam_locs, dtype=dtype["float"])
    beam_secF = _expand_to_uniform_array(beam_secF, dtype=dtype["float"])
    beam_secD = _expand_to_uniform_array(beam_secD, dtype=dtype["float"])
    beam_sec_locs = _get_ele_sec_coords(start_coords, end_coords, beam_locs)

    return beam_secF, beam_secD, beam_sec_locs.astype(dtype["float"])


def _extract_pattern_info(ele_load_data):
    pattern_tags, load_eletags = [], []
    if len(ele_load_data) > 0:
        petags = ele_load_data.coords["PatternEleTags"].values
        for item in petags:
            num1, num2 = item.split("-")
            pattern_tags.append(int(num1))
            load_eletags.append(int(num2))
    return np.array(pattern_tags), np.array(load_eletags)


def _get_nonlinear_section_response(eletag, length):
    xlocs, sec_f, sec_d = [], [], []
    for i, loc in enumerate(ops.sectionLocation(eletag) / length):
        xlocs.append(loc)
        forces = _format_six_component(ops.sectionForce(eletag, i + 1))
        defos = _format_six_component(ops.sectionDeformation(eletag, i + 1))
        sec_f.append(forces)
        sec_d.append(defos)
    return xlocs, sec_f, sec_d


def _format_six_component(vec):
    if len(vec) == 0:
        return [0.0] * 6
    if len(vec) == 2:
        return [vec[0], vec[1], 0.0, 0.0, 0.0, 0.0]
    if len(vec) == 3:
        return [vec[0], vec[1], vec[2], 0.0, 0.0, 0.0]
    if len(vec) == 4:
        return [vec[0], vec[1], 0.0, vec[2], 0.0, vec[3]]
    if len(vec) == 5:
        return [vec[0], vec[1], vec[4], vec[2], 0.0, vec[3]]
    return vec[:6]


def _get_param_value(eletag, param_name):
    paramTag = 1
    paramTags = ops.getParamTags()
    if len(paramTags) > 0:
        paramTag = max(paramTags) + 1
    ops.parameter(paramTag, "element", eletag, param_name)
    value = ops.getParamValue(paramTag)
    ops.remove("parameter", paramTag)
    return value


def _get_elastic_sec_defo(eletag, sec_force, basic_d, length, xlocs):
    with suppress_ops_print():
        # A = _get_param_value(eletag, "A")
        E = _get_param_value(eletag, "E")
        Iz = _get_param_value(eletag, "Iz")
        Iy = _get_param_value(eletag, "Iy")
        G = _get_param_value(eletag, "G")
        # J = _get_param_value(eletag, "J")
        Avy = _get_param_value(eletag, "Avy")
        Avz = _get_param_value(eletag, "Avz")

    # N1, Mz1, Vy1, My1, Vz1, T1
    eps = 1e-10
    sec_d = np.zeros_like(sec_force)
    oneOverL, xi6 = 1.0 / length, 6 * xlocs
    sec_d[:, 0] = basic_d[0] * oneOverL  # N
    sec_d[:, 5] = basic_d[5] * oneOverL  # T
    if E * Iz > eps:
        sec_d[:, 1] = sec_force[:, 1] / E / Iz  # MZ
    else:
        sec_d[:, 1] = oneOverL * ((xi6 - 4.0) * basic_d[1] + (xi6 - 2.0) * basic_d[2])  # MZ
    if E * Iy > eps:
        sec_d[:, 3] = sec_force[:, 3] / E / Iy  # MY
    else:
        sec_d[:, 3] = oneOverL * ((xi6 - 4.0) * basic_d[3] + (xi6 - 2.0) * basic_d[4])  # MY
    if G * Avy > eps:
        sec_d[:, 2] = sec_force[:, 2] / G / Avy
    if G * Avz > eps:
        sec_d[:, 4] = sec_force[:, 4] / G / Avz
    return sec_d


def _get_elastic_sec_forces(ele_tag, length, ele_load_data, pattern_tags, load_eletags, local_force, xlocs):
    sec_locs = xlocs
    sec_x = sec_locs * length
    sec_f = np.full((len(xlocs), 6), 0.0)
    # N1, Mz1, Vy1, My1, Vz1, T1
    sec_f[:, 0] = -local_force[0]
    sec_f[:, 1] = -local_force[5] + local_force[1] * sec_x
    sec_f[:, 2] = local_force[1]
    sec_f[:, 3] = -local_force[4] - local_force[2] * sec_x
    sec_f[:, 4] = -local_force[2]
    sec_f[:, 5] = -local_force[3]
    if ele_tag in load_eletags:
        idx = np.abs(load_eletags - ele_tag) < 1e-4
        load_data = ele_load_data[idx, 2:].data
        ptags = pattern_tags[idx]
        factors = [ops.getLoadFactor(int(ptag)) for ptag in ptags]
    else:
        load_data = [[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]]
        factors = [0.0]
    for ldata, factor in zip(load_data, factors):
        wya, wyb, wza, wzb, wxa, wxb, xa, xb = ldata
        if xb > xa and np.abs(xb - xa - 1) < 1e-2:  # Full uniform load
            wx, wy, wz = wxa * factor, wya * factor, wza * factor
            sec_f[:, 0] += -wx * sec_x
            sec_f[:, 1] += 0.5 * wy * sec_x**2
            sec_f[:, 2] += wy * sec_x
            sec_f[:, 3] += -0.5 * wz * sec_x * sec_x
            sec_f[:, 4] += -wz * sec_x
        elif xb < xa:  # Point Load
            px, py, pz = wxa * factor, wya * factor, wza * factor
            xa = xa * length
            idx = sec_x > xa
            sec_f[idx, 0] += -px
            sec_f[idx, 1] += py * (sec_x[idx] - xa)
            sec_f[idx:, 2] += py
            sec_f[idx:, 3] += -pz * (sec_x[idx] - xa)
            sec_f[idx:, 4] += -pz
        elif xb > xa and np.abs(xb - xa - 1) > 1e-2:  # Partial uniform load
            wx, wy, wz = wxa * factor, wya * factor, wza * factor
            xa = xa * length
            xb = xb * length
            idx2 = sec_x > xa & sec_x < xb
            idx3 = sec_x >= xb
            sec_f[idx2, 0] += -wx * (sec_x[idx2] - xa)
            sec_f[idx2, 1] += 0.5 * wy * (sec_x[idx2] - xa) ** 2
            sec_f[idx2, 2] += wy * (sec_x[idx2] - xa)
            sec_f[idx2, 3] += -0.5 * wz * (sec_x[idx2] - xa) ** 2
            sec_f[idx2, 4] += -wz * (sec_x[idx2] - xa)
            # ------------------------------
            sec_f[idx3, 0] += -wx * (xb - xa)
            sec_f[idx3, 1] += wy * (xb - xa) * (sec_x[idx3] - 0.5 * (xb + xa))
            sec_f[idx3, 2] += wy * (xb - xa)
            sec_f[idx3, 3] += -wz * (xb - xa) * (sec_x[idx3] - 0.5 * (xb + xa))
            sec_f[idx3, 4] += -wz * (xb - xa)
    return sec_f


def _get_ele_length(ele_tags):
    coords1, coords2 = [], []
    for ele_tag in ele_tags:
        ele_tag = int(ele_tag)
        nodes = ops.eleNodes(ele_tag)
        coords1.append(ops.nodeCoord(nodes[0]))
        coords2.append(ops.nodeCoord(nodes[1]))
    start = np.array(coords1)
    end = np.array(coords2)
    return np.linalg.norm(end - start, axis=1), start, end


def _get_ele_sec_coords(start, end, sec_locs):
    coords = start[:, None, :] + (end - start)[:, None, :] * sec_locs[..., None]
    locs_expanded = sec_locs[..., None]
    return np.concatenate([locs_expanded, coords], axis=-1)
