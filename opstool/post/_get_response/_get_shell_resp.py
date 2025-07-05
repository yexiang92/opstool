from collections import defaultdict
from typing import Optional

import numpy as np
import openseespy.opensees as ops
import xarray as xr

from ...utils import get_shell_gp2node_func, suppress_ops_print
from ._response_base import ResponseBase, _expand_to_uniform_array


class ShellRespStepData(ResponseBase):
    def __init__(
        self,
        ele_tags=None,
        model_update: bool = False,
        compute_nodal_resp: Optional[str] = None,
        dtype: Optional[dict] = None,
    ):
        self.resp_names = [
            "sectionForces",
            "sectionDeformations",
            "Stresses",
            "Strains",
            "sectionForcesAtNodes",
            "sectionDeformationsAtNodes",
            "StressesAtNodes",
            "StrainsAtNodes",
        ]
        self.resp_steps = None
        self.resp_steps_list = []  # for model update
        self.resp_steps_dict = {}  # for non-update
        self.step_track = 0
        self.ele_tags = ele_tags
        self.times = []

        self.node_tags = None
        self.compute_nodal_resp = compute_nodal_resp
        self.nodal_resp_method = compute_nodal_resp
        self.model_update = model_update
        self.dtype = {"int": np.int32, "float": np.float32}
        if isinstance(dtype, dict):
            self.dtype.update(dtype)

        self.attrs = {
            "FXX,FYY,FXY": "Membrane (in-plane) forces or deformations.",
            "MXX,MYY,MXY": "Bending moments or rotations (out-plane) of plate.",
            "VXZ,VYZ": "Shear forces or deformations.",
            "sigma11, sigma22": "Normal stress (strain) along local x, y",
            "sigma12, sigma23, sigma13": "Shear stress (strain).",
        }
        self.GaussPoints = None
        self.secDOFs = ["FXX", "FYY", "FXY", "MXX", "MYY", "MXY", "VXZ", "VYZ"]
        self.fiberPoints = None
        self.stressDOFs = ["sigma11", "sigma22", "sigma12", "sigma23", "sigma13"]

        self.initialize()

    def initialize(self):
        self.resp_steps = None
        self.resp_steps_list = []
        for name in self.resp_names:
            self.resp_steps_dict[name] = []
        self.add_data_one_step(self.ele_tags)
        self.times = [0.0]
        self.step_track = 0

    def reset(self):
        self.initialize()

    def add_data_one_step(self, ele_tags):
        sec_forces, sec_defos, stresses, strains = _get_shell_resp_one_step(ele_tags, dtype=self.dtype)

        if self.compute_nodal_resp:
            method = self.nodal_resp_method
            node_sec_forces_avg, node_tags = _get_nodal_resp(ele_tags, sec_forces, method=method, dtype=self.dtype)
            node_sec_defo_avg, node_tags = _get_nodal_resp(ele_tags, sec_defos, method=method, dtype=self.dtype)
            node_stresses_avg, node_tags = _get_nodal_resp(ele_tags, stresses, method=method, dtype=self.dtype)
            node_strains_avg, node_tags = _get_nodal_resp(ele_tags, strains, method=method, dtype=self.dtype)
            self.node_tags = node_tags

        if self.GaussPoints is None:
            self.GaussPoints = np.arange(sec_forces.shape[1]) + 1
        if self.fiberPoints is None:
            self.fiberPoints = np.arange(stresses.shape[2]) + 1

        if self.model_update:
            data_vars = {}
            data_vars["sectionForces"] = (["eleTags", "GaussPoints", "secDOFs"], sec_forces)
            data_vars["sectionDeformations"] = (["eleTags", "GaussPoints", "secDOFs"], sec_defos)
            data_vars["Stresses"] = (["eleTags", "GaussPoints", "fiberPoints", "stressDOFs"], stresses)
            data_vars["Strains"] = (["eleTags", "GaussPoints", "fiberPoints", "stressDOFs"], strains)
            coords = {
                "eleTags": ele_tags,
                "GaussPoints": self.GaussPoints,
                "secDOFs": self.secDOFs,
                "fiberPoints": self.fiberPoints,
                "stressDOFs": self.stressDOFs,
            }
            if self.compute_nodal_resp:
                data_vars["sectionForcesAtNodes"] = (["nodeTags", "secDOFs"], node_sec_forces_avg)
                data_vars["sectionDeformationsAtNodes"] = (["nodeTags", "secDOFs"], node_sec_defo_avg)
                if len(node_stresses_avg) > 0:
                    data_vars["StressesAtNodes"] = (["nodeTags", "fiberPoints", "stressDOFs"], node_stresses_avg)
                if len(node_strains_avg) > 0:
                    data_vars["StrainsAtNodes"] = (["nodeTags", "fiberPoints", "stressDOFs"], node_strains_avg)
                coords["nodeTags"] = node_tags
            ds = xr.Dataset(data_vars=data_vars, coords=coords, attrs=self.attrs)
            self.resp_steps_list.append(ds)
        else:
            self.resp_steps_dict["sectionForces"].append(sec_forces)
            self.resp_steps_dict["sectionDeformations"].append(sec_defos)
            self.resp_steps_dict["Stresses"].append(stresses)
            self.resp_steps_dict["Strains"].append(strains)
            if self.compute_nodal_resp:
                self.resp_steps_dict["sectionForcesAtNodes"].append(node_sec_forces_avg)
                self.resp_steps_dict["sectionDeformationsAtNodes"].append(node_sec_defo_avg)
                if len(node_stresses_avg) > 0:
                    self.resp_steps_dict["StressesAtNodes"].append(node_stresses_avg)
                if len(node_strains_avg) > 0:
                    self.resp_steps_dict["StrainsAtNodes"].append(node_strains_avg)

        self.times.append(ops.getTime())
        self.step_track += 1

    def _to_xarray(self):
        self.times = np.array(self.times, dtype=self.dtype["float"])
        if self.model_update:
            self.resp_steps = xr.concat(self.resp_steps_list, dim="time", join="outer")
            self.resp_steps.coords["time"] = self.times
        else:
            data_vars = {}
            data_vars["sectionForces"] = (
                ["time", "eleTags", "GaussPoints", "secDOFs"],
                self.resp_steps_dict["sectionForces"],
            )
            data_vars["sectionDeformations"] = (
                ["time", "eleTags", "GaussPoints", "secDOFs"],
                self.resp_steps_dict["sectionDeformations"],
            )
            data_vars["Stresses"] = (
                ["time", "eleTags", "GaussPoints", "fiberPoints", "stressDOFs"],
                self.resp_steps_dict["Stresses"],
            )
            data_vars["Strains"] = (
                ["time", "eleTags", "GaussPoints", "fiberPoints", "stressDOFs"],
                self.resp_steps_dict["Strains"],
            )
            coords = {
                "time": self.times,
                "eleTags": self.ele_tags,
                "GaussPoints": self.GaussPoints,
                "secDOFs": self.secDOFs,
                "fiberPoints": self.fiberPoints,
                "stressDOFs": self.stressDOFs,
            }
            if self.compute_nodal_resp:
                data_vars["sectionForcesAtNodes"] = (
                    ["time", "nodeTags", "secDOFs"],
                    self.resp_steps_dict["sectionForcesAtNodes"],
                )
                data_vars["sectionDeformationsAtNodes"] = (
                    ["time", "nodeTags", "secDOFs"],
                    self.resp_steps_dict["sectionDeformationsAtNodes"],
                )
                if len(self.resp_steps_dict["StressesAtNodes"]) > 0:
                    data_vars["StressesAtNodes"] = (
                        ["time", "nodeTags", "fiberPoints", "stressDOFs"],
                        self.resp_steps_dict["StressesAtNodes"],
                    )
                if len(self.resp_steps_dict["StrainsAtNodes"]) > 0:
                    data_vars["StrainsAtNodes"] = (
                        ["time", "nodeTags", "fiberPoints", "stressDOFs"],
                        self.resp_steps_dict["StrainsAtNodes"],
                    )
                coords["nodeTags"] = self.node_tags
            self.resp_steps = xr.Dataset(data_vars=data_vars, coords=coords, attrs=self.attrs)

    def get_data(self):
        return self.resp_steps

    def get_track(self):
        return self.step_track

    def save_file(self, dt: xr.DataTree):
        self._to_xarray()
        dt["/ShellResponses"] = self.resp_steps
        return dt

    @staticmethod
    def read_file(dt: xr.DataTree, unit_factors: Optional[dict] = None):
        resp_steps = dt["/ShellResponses"].to_dataset()
        if unit_factors is not None:
            resp_steps = ShellRespStepData._unit_transform(resp_steps, unit_factors)
        return resp_steps

    @staticmethod
    def _unit_transform(resp_steps, unit_factors):
        force_per_length_factor = unit_factors["force_per_length"]
        moment_per_length_factor = unit_factors["moment_per_length"]
        stress_factor = unit_factors["stress"]

        resp_steps["sectionForces"].loc[{"secDOFs": ["FXX", "FYY", "FXY", "VXZ", "VYZ"]}] *= force_per_length_factor
        resp_steps["sectionForces"].loc[{"secDOFs": ["MXX", "MYY", "MXY"]}] *= moment_per_length_factor
        resp_steps["Stresses"] *= stress_factor

        if "sectionForcesAtNodes" in resp_steps.data_vars:
            resp_steps["sectionForcesAtNodes"].loc[{"secDOFs": ["FXX", "FYY", "FXY", "VXZ", "VYZ"]}] *= (
                force_per_length_factor
            )
            resp_steps["sectionForcesAtNodes"].loc[{"secDOFs": ["MXX", "MYY", "MXY"]}] *= moment_per_length_factor
        if "StressesAtNodes" in resp_steps.data_vars:
            resp_steps["StressesAtNodes"] *= stress_factor

        return resp_steps

    @staticmethod
    def read_response(
        dt: xr.DataTree, resp_type: Optional[str] = None, ele_tags=None, unit_factors: Optional[dict] = None
    ):
        ds = ShellRespStepData.read_file(dt, unit_factors=unit_factors)
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


def _get_shell_resp_one_step(ele_tags, dtype):
    sec_forces, sec_defos = [], []
    stresses, strains = [], []
    for _i, etag in enumerate(ele_tags):
        etag = int(etag)
        forces = ops.eleResponse(etag, "stresses")
        defos = ops.eleResponse(etag, "strains")
        sec_forces.append(_reorder_by_ele_type(etag, np.reshape(forces, (-1, 8))))
        sec_defos.append(_reorder_by_ele_type(etag, np.reshape(defos, (-1, 8))))
        # stress and strains
        num_sec = int(len(forces) / 8)
        sec_stress, sec_strain = [], []
        for j in range(num_sec):
            for k in range(100000000000000000):  # ugly but useful, loop for fiber layers
                stress = ops.eleResponse(etag, "Material", f"{j + 1}", "fiber", f"{k + 1}", "stresses")
                strain = ops.eleResponse(etag, "Material", f"{j + 1}", "fiber", f"{k + 1}", "strains")
                if len(stress) == 0 or len(strain) == 0:
                    break
                sec_stress.extend(stress)
                sec_strain.extend(strain)
        if len(sec_stress) == 0:  # elastic section response
            sec_stress, sec_strain = _get_elastic_section_stress(etag, sec_forces[-1])
        sec_stress = np.reshape(sec_stress, (num_sec, -1, 5))
        sec_strain = np.reshape(sec_strain, (num_sec, -1, 5))
        stresses.append(_reorder_by_ele_type(etag, sec_stress))
        strains.append(_reorder_by_ele_type(etag, sec_strain))
    sec_forces = _expand_to_uniform_array(sec_forces, dtype=dtype["float"])
    sec_defos = _expand_to_uniform_array(sec_defos, dtype=dtype["float"])
    stresses = _expand_to_uniform_array(stresses, dtype=dtype["float"])
    strains = _expand_to_uniform_array(strains, dtype=dtype["float"])
    return sec_forces, sec_defos, stresses, strains


def _reorder_by_ele_type(etag, resp):
    ele_class_tag = ops.getEleClassTags(etag)[0]
    if ele_class_tag == 54 and len(resp) == 9:  # "ShellMITC9", 9 gps
        idx = [0, 2, 4, 6, 1, 3, 5, 7, 8]
    else:
        return resp
    return np.array([resp[i] for i in idx])


gp2node_type = {3: "tri", 6: "tri", 4: "quad", 8: "quad", 9: "quad"}


# Get nodal stresses and strains from the Gauss points of elements.
def _get_nodal_resp(ele_tags, ele_gp_resp, method, dtype):
    node_resp = defaultdict(list)
    for etag, gp_resp in zip(ele_tags, ele_gp_resp):
        etag = int(etag)
        ntags = ops.eleNodes(etag)
        gp_resp = drop_all_nan_rows(gp_resp)  # drop rows where all values are NaN
        if len(gp_resp) == 0:
            continue
        gp2node_func = get_shell_gp2node_func(ele_type=gp2node_type[len(ntags)], n=len(ntags), gp=len(gp_resp))
        if gp2node_func:
            resp = gp2node_func(method=method, gp_resp=gp_resp)
        else:
            resp = np.zeros((len(ntags), *gp_resp.shape[1:]), dtype=dtype["float"])
        for i, ntag in enumerate(ntags):
            node_resp[ntag].append(resp[i])
    # node_resp = dict(sorted(node_resp.items()))
    node_avg = {}

    for nid, vals in node_resp.items():
        arr = np.stack(vals, axis=0)  # shape: (k, m), k=num_samples, m=DOFs
        node_avg[nid] = np.nanmean(arr, axis=0)  # mean value
    node_avg = np.array(list(node_avg.values()), dtype=dtype["float"])
    node_tags = list(node_resp.keys())
    return node_avg, node_tags


def drop_all_nan_rows(arr: np.ndarray) -> np.ndarray:
    axis_to_check = tuple(range(1, arr.ndim))
    mask = ~np.isnan(arr).all(axis=axis_to_check)
    return arr[mask]


def _get_elastic_section_stress(eletag, sec_forces):
    with suppress_ops_print():
        E = _get_param_value(eletag, "E")
        nu = _get_param_value(eletag, "nu")
        h = _get_param_value(eletag, "h")
        # Ep_mod = _get_param_value(eletag, "Ep_mod")
        # rho = _get_param_value(eletag, "rho")
    if E > 0 and nu >= 0 and h > 0:
        sigmas, epses = [], []
        G = 0.5 * E / (1.0 + nu)
        xs = np.linspace(-h / 2, h / 2, 5)
        w = 12 / (h * h * h)
        for f11, f22, f12, m11, m22, m12, v13, v23 in sec_forces:
            sigma11 = f11 / h - w * m11 * xs
            sigma22 = f22 / h - w * m22 * xs
            sigma12 = f12 / h - w * m12 * xs
            sigma13 = v13 / h + np.zeros_like(xs)
            sigma23 = v23 / h + np.zeros_like(xs)
            eps11 = sigma11 / E
            eps22 = sigma22 / E
            eps12 = sigma12 / G
            eps13 = sigma13 / G
            eps23 = sigma23 / G
            sigma = np.array([sigma11, sigma22, sigma12, sigma23, sigma13]).T
            eps = np.array([eps11, eps22, eps12, eps23, eps13]).T
            sigmas.append(sigma)
            epses.append(eps)
        sigmas = np.array(sigmas)
        epses = np.array(epses)
    else:
        sigmas = np.full((len(sec_forces), 1, 5), np.nan)
        epses = np.full((len(sec_forces), 1, 5), np.nan)
    return sigmas, epses


def _get_param_value(eletag, param_name):
    paramTag = 1
    paramTags = ops.getParamTags()
    if len(paramTags) > 0:
        paramTag = max(paramTags) + 1
    ops.parameter(paramTag, "element", eletag, param_name)
    value = ops.getParamValue(paramTag)
    ops.remove("parameter", paramTag)
    return value
