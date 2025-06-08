from typing import Optional

import numpy as np
import openseespy.opensees as ops
import xarray as xr

from ._response_base import ResponseBase, _expand_to_uniform_array

# from ._response_extrapolation import (
#     resp_extrap_tri3,
#     resp_extrap_quad4,
#     resp_extrap_quad9,
# )
# from ._get_plane_resp import _get_principal_resp


class ShellRespStepData(ResponseBase):
    def __init__(self, ele_tags=None, model_update: bool = False, dtype: dict = None):
        self.resp_names = [
            "sectionForces",
            "sectionDeformations",
            "Stresses",
            "Strains",
        ]
        self.resp_steps = None
        self.resp_steps_list = []  # for model update
        self.resp_steps_dict = {}  # for non-update
        self.step_track = 0
        self.ele_tags = ele_tags
        self.times = []

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
            ds = xr.Dataset(
                data_vars=data_vars,
                coords={
                    "eleTags": ele_tags,
                    "GaussPoints": self.GaussPoints,
                    "secDOFs": self.secDOFs,
                    "fiberPoints": self.fiberPoints,
                    "stressDOFs": self.stressDOFs,
                },
                attrs=self.attrs,
            )
            self.resp_steps_list.append(ds)
        else:
            self.resp_steps_dict["sectionForces"].append(sec_forces)
            self.resp_steps_dict["sectionDeformations"].append(sec_defos)
            self.resp_steps_dict["Stresses"].append(stresses)
            self.resp_steps_dict["Strains"].append(strains)

        self.times.append(ops.getTime())
        self.step_track += 1

    def _to_xarray(self):
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
            self.resp_steps = xr.Dataset(
                data_vars=data_vars,
                coords={
                    "time": self.times,
                    "eleTags": self.ele_tags,
                    "GaussPoints": self.GaussPoints,
                    "secDOFs": self.secDOFs,
                    "fiberPoints": self.fiberPoints,
                    "stressDOFs": self.stressDOFs,
                },
                attrs=self.attrs,
            )

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
        sec_forces.append(np.reshape(forces, (-1, 8)))
        sec_defos.append(np.reshape(defos, (-1, 8)))
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
        if len(sec_stress) == 0:
            sec_stress.extend([np.nan, np.nan, np.nan, np.nan, np.nan] * num_sec)
        if len(sec_strain) == 0:
            sec_strain.extend([np.nan, np.nan, np.nan, np.nan, np.nan] * num_sec)
        sec_stress = np.reshape(sec_stress, (num_sec, -1, 5))
        sec_strain = np.reshape(sec_strain, (num_sec, -1, 5))
        stresses.append(sec_stress)
        strains.append(sec_strain)
    sec_forces = _expand_to_uniform_array(sec_forces, dtype=dtype["float"])
    sec_defos = _expand_to_uniform_array(sec_defos, dtype=dtype["float"])
    stresses = _expand_to_uniform_array(stresses, dtype=dtype["float"])
    strains = _expand_to_uniform_array(strains, dtype=dtype["float"])
    return sec_forces, sec_defos, stresses, strains


# --------------------------------------------------------------------------
# --------------------------------------------------------------------------
# def _get_shell_resp(ele_tags):
#     all_forces, all_defos = dict(), dict()
#     all_stress, all_strain = dict(), dict()
#     for etag in ele_tags:
#         ntags = ops.eleNodes(etag)
#         if len(ntags) == 3:
#             nodal_forces, nodal_defos, nodal_stress, nodal_strain = _get_resp_tri3(etag)
#         elif len(ntags) == 4:
#             nodal_forces, nodal_defos, nodal_stress, nodal_strain = _get_resp_quad4(
#                 etag
#             )
#         elif len(ntags) == 9:
#             nodal_forces, nodal_defos, nodal_stress, nodal_strain = _get_resp_quad9(
#                 etag
#             )
#         else:
#             raise RuntimeError("Unsupported planar element type!")
#         all_forces[etag], all_defos[etag] = nodal_forces, nodal_defos
#         all_stress[etag], all_strain[etag] = nodal_stress, nodal_strain
#     return all_forces, all_defos, all_stress, all_strain
#
#
# def _get_resp_tri3(etag):
#     forces = ops.eleResponse(etag, "stresses")
#     defos = ops.eleResponse(etag, "strains")
#     nodal_forces = resp_extrap_tri3(forces)
#     nodal_defos = resp_extrap_tri3(defos)
#     ng = 1
#     stresses, strains = _get_gaussian_stress_strain(etag, ng)
#     nlayer = len(stresses[0])
#     nodal_stress, nodal_strain = np.zeros((nlayer, 3, 13)), np.zeros((nlayer, 3, 13))
#
#     for i in range(nlayer):
#         stress, strain = [], []
#         for j in range(ng):
#             stress.append(stresses[j][i])
#             strain.append(strains[j][i])
#         nodal_stress[i] = resp_extrap_tri3(stress)
#         nodal_strain[i] = resp_extrap_tri3(strain)
#
#     return nodal_forces, nodal_defos, nodal_stress, nodal_strain
#
#
# def _get_resp_quad4(etag):
#     forces = ops.eleResponse(etag, "stresses")
#     defos = ops.eleResponse(etag, "strains")
#     forces = np.reshape(forces, (-1, 8))
#     defos = np.reshape(defos, (-1, 8))
#     nodal_forces = resp_extrap_quad4(forces)
#     nodal_defos = resp_extrap_quad4(defos)
#     ng = 4
#     stresses, strains = _get_gaussian_stress_strain(etag, ng)
#     nlayer = len(stresses[0])
#     nodal_stress, nodal_strain = np.zeros((nlayer, 4, 13)), np.zeros((nlayer, 4, 13))
#
#     for i in range(nlayer):
#         stress, strain = [], []
#         for j in range(ng):
#             stress.append(stresses[j][i])
#             strain.append(strains[j][i])
#         nodal_stress[i] = resp_extrap_quad4(stress)
#         nodal_strain[i] = resp_extrap_quad4(strain)
#
#     return nodal_forces, nodal_defos, nodal_stress, nodal_strain
#
#
# def _get_resp_quad9(etag):
#     forces = ops.eleResponse(etag, "stresses")
#     defos = ops.eleResponse(etag, "strains")
#     forces = np.reshape(forces, (-1, 8))
#     defos = np.reshape(defos, (-1, 8))
#     nodal_forces = resp_extrap_quad9(forces)
#     nodal_defos = resp_extrap_quad9(defos)
#     ng = 9
#     stresses, strains = _get_gaussian_stress_strain(etag, ng)
#     nlayer = len(stresses[0])
#     nodal_stress, nodal_strain = np.zeros((nlayer, 9, 13)), np.zeros((nlayer, 9, 13))
#
#     for i in range(nlayer):
#         stress, strain = [], []
#         for j in range(ng):
#             stress.append(stresses[j][i])
#             strain.append(strains[j][i])
#         nodal_stress[i] = resp_extrap_quad9(stress)
#         nodal_strain[i] = resp_extrap_quad9(strain)
#
#     return nodal_forces, nodal_defos, nodal_stress, nodal_strain
#
#
# def _get_gaussian_stress_strain(etag, ng):
#     stresses, strains = [], []
#     for i in range(ng):
#         j = 0
#         stress, strain = [], []
#         while True:
#             s = ops.eleResponse(
#                 etag, "material", f"{i + 1}", "fiber", f"{j + 1}", "stress"
#             )
#             d = ops.eleResponse(
#                 etag, "material", f"{i + 1}", "fiber", f"{j + 1}", "strain"
#             )
#             if j == 0 and len(s) == 0:
#                 s, d = [0.0] * 6, [0.0] * 6
#             if len(s) == 0:
#                 break
#             s, d = _get_all_stress_strain(s, d)
#             stress.extend(s)
#             strain.extend(d)
#             j += 1
#         stresses.append(stress)
#         strains.append(strain)
#     return stresses, strains
#
#
# def _get_all_stress_strain(stress, strain):
#     stress2 = _get_principal_resp(stress)
#     stress += stress2
#     strain2 = _get_principal_resp(strain)
#     strain += strain2
#     return stress, strain
