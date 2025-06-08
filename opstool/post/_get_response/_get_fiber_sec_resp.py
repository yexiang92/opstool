import warnings
from typing import ClassVar, Optional, Union

import numpy as np
import openseespy.opensees as ops
import xarray as xr

from ._response_base import ResponseBase, _expand_to_uniform_array


class FiberSecData:
    ELE_SEC_KEYS: ClassVar[dict] = {}  # key: ele_tag, value: sec_num

    @classmethod
    def add_data(cls, ele_tags=None):
        if ele_tags is None:
            ele_tags = []
        elif isinstance(ele_tags, str) and ele_tags.lower() == "all":
            ele_tags = ops.getEleTags()
        else:
            ele_tags = np.atleast_1d(ele_tags)

        for ele_tag in ele_tags:
            ele_tag = int(ele_tag)
            sec_locs = ops.sectionLocation(ele_tag)
            if isinstance(sec_locs, list) and len(sec_locs) > 0:
                cls.ELE_SEC_KEYS[ele_tag] = len(sec_locs)

    @classmethod
    def get_ele_sec_keys(cls):
        return cls.ELE_SEC_KEYS


def _set_fiber_sec_data(fiber_ele_tags: Optional[Union[str, list, tuple]] = None):
    FiberSecData.add_data(fiber_ele_tags)


class FiberSecRespStepData(ResponseBase):
    def __init__(self, fiber_ele_tags: Optional[Union[str, list, tuple]] = None, dtype: Optional[dict] = None):
        _set_fiber_sec_data(fiber_ele_tags)
        self.ELE_SEC_KEYS = FiberSecData.get_ele_sec_keys()

        self.resp_names = ["Stresses", "Strains", "secForce", "secDefo"]
        self.resp_steps = None
        self.resp_steps_dict = {}
        self.times = []
        self.step_track = 0

        self.dtype = {"int": np.int32, "float": np.float32}
        if isinstance(dtype, dict):
            self.dtype.update(dtype)

        self.secPoints = None
        self.fiberPoints = None
        self.DOFs = ["P", "Mz", "My", "T"]

        self.initialize()

    def initialize(self):
        self.resp_steps = None
        for name in self.resp_names:
            self.resp_steps_dict[name] = []
        self.add_data_one_step()
        self.step_track = 0
        self.times = [0.0]

    def reset(self):
        self.initialize()

    def add_data_one_step(self):
        stress, strain, defo, force = _get_fiber_sec_resp(self.ELE_SEC_KEYS, dtype=self.dtype)
        self.resp_steps_dict["Stresses"].append(stress)
        self.resp_steps_dict["Strains"].append(strain)
        self.resp_steps_dict["secForce"].append(force)
        self.resp_steps_dict["secDefo"].append(defo)

        if self.secPoints is None:
            self.secPoints = np.arange(stress.shape[1]) + 1
            self.fiberPoints = np.arange(stress.shape[2]) + 1

        self.step_track += 1
        self.times.append(ops.getTime())

    def _get_fiber_geo_data(self):
        all_ys, all_zs, all_mats, all_areas = [], [], [], []
        for ele_tag, sec_num in self.ELE_SEC_KEYS.items():
            ele_tag = int(ele_tag)
            sec_num = int(sec_num)
            ys, zs, areas, mats = [], [], [], []
            for i in range(sec_num):
                fiber_data = _get_fiber_sec_data(ele_tag, i + 1)
                ys.append(fiber_data[:, 0])
                zs.append(fiber_data[:, 1])
                areas.append(fiber_data[:, 2])
                mats.append(fiber_data[:, 3])
            all_ys.append(_expand_to_uniform_array(ys))
            all_zs.append(_expand_to_uniform_array(zs))
            all_areas.append(_expand_to_uniform_array(areas))
            all_mats.append(_expand_to_uniform_array(mats))
        all_ys = _expand_to_uniform_array(all_ys)
        all_zs = _expand_to_uniform_array(all_zs)
        all_areas = _expand_to_uniform_array(all_areas)
        all_mats = _expand_to_uniform_array(all_mats)

        self.resp_steps["ys"] = (("eleTags", "secPoints", "fiberPoints"), all_ys)
        self.resp_steps["zs"] = (("eleTags", "secPoints", "fiberPoints"), all_zs)
        self.resp_steps["areas"] = (("eleTags", "secPoints", "fiberPoints"), all_areas)
        self.resp_steps["matTags"] = (("eleTags", "secPoints", "fiberPoints"), all_mats)

    def _to_xarray(self):
        # self.resp_steps = xr.concat(self.resp_steps, dim="time", join="outer")
        # self.resp_steps.coords["time"] = self.times
        data_vars = {
            "Stresses": (("time", "eleTags", "secPoints", "fiberPoints"), self.resp_steps_dict["Stresses"]),
            "Strains": (("time", "eleTags", "secPoints", "fiberPoints"), self.resp_steps_dict["Strains"]),
            "secDefo": (("time", "eleTags", "secPoints", "DOFs"), self.resp_steps_dict["secDefo"]),
            "secForce": (("time", "eleTags", "secPoints", "DOFs"), self.resp_steps_dict["secForce"]),
        }
        self.resp_steps = xr.Dataset(
            data_vars=data_vars,
            coords={
                "time": self.times,
                "eleTags": list(self.ELE_SEC_KEYS.keys()),
                "secPoints": self.secPoints,
                "fiberPoints": self.fiberPoints,
                "DOFs": self.DOFs,
            },
        )

        # add geo data
        self._get_fiber_geo_data()

    def get_data(self):
        return self.resp_steps

    def get_track(self):
        return self.step_track

    def save_file(self, dt: xr.DataTree):
        self._to_xarray()
        dt["/FiberSectionResponses"] = self.resp_steps
        return dt

    @staticmethod
    def read_file(dt: xr.DataTree, unit_factors: Optional[dict] = None):
        resp_steps = dt["/FiberSectionResponses"].to_dataset()
        if unit_factors is not None:
            resp_steps = FiberSecRespStepData._unit_transform(resp_steps, unit_factors)
        return resp_steps

    @staticmethod
    def _unit_transform(resp_steps, unit_factors):
        force_factor = unit_factors["force"]
        moment_factor = unit_factors["moment"]
        curvature_factor = unit_factors["curvature"]
        disp_factor = unit_factors["disp"]
        stress_factor = unit_factors["stress"]

        # ---------------------------------------------------------
        resp_steps["Stresses"] *= stress_factor
        # ---------------------------------------------------------
        resp_steps["secForce"].loc[{"DOFs": ["P"]}] *= force_factor
        resp_steps["secForce"].loc[{"DOFs": ["Mz", "My", "T"]}] *= moment_factor
        resp_steps["secDefo"].loc[{"DOFs": ["Mz", "My", "T"]}] *= curvature_factor
        # --------------------------------------------------------
        resp_steps["ys"] *= disp_factor
        resp_steps["zs"] *= disp_factor
        resp_steps["areas"] *= disp_factor**2

        return resp_steps

    @staticmethod
    def read_response(
        dt: xr.DataTree, resp_type: Optional[str] = None, ele_tags=None, unit_factors: Optional[dict] = None
    ):
        ds = FiberSecRespStepData.read_file(dt, unit_factors=unit_factors)
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


def _get_fiber_sec_resp(ele_secs: dict, dtype: dict):
    """Get the fiber section responses one step.

    Parameters
    -----------
    ele_secs: Union[list, tuple]
        [(ele_tag1, sec_tag1), (ele_tag1, sec_tag2), …, (ele_tag2, sec_tag1), …]
    """
    all_stress = []
    all_strains = []
    for ele_tag, sec_num in ele_secs.items():
        ele_tag = int(ele_tag)
        sec_num = int(sec_num)
        stress, strain, defo, force = [], [], [], []
        for i in range(sec_num):
            fiber_data = _get_fiber_sec_data(ele_tag, i + 1, dtype=dtype)
            stress.append(fiber_data[:, -2])
            strain.append(fiber_data[:, -1])
        all_stress.append(_expand_to_uniform_array(stress))
        all_strains.append(_expand_to_uniform_array(strain))
    all_stress = _expand_to_uniform_array(all_stress, dtype=dtype["float"])
    all_strains = _expand_to_uniform_array(all_strains, dtype=dtype["float"])

    # -----------------------------------------------------------------------
    all_defo = []
    all_force = []
    for ele_tag, sec_num in ele_secs.items():
        ele_tag = int(ele_tag)
        sec_num = int(sec_num)
        defo, force = [], []
        for i in range(sec_num):
            defo_forces = ops.eleResponse(ele_tag, "section", f"{i + 1}", "forceAndDeformation")
            if len(defo_forces) == 4:
                defo_forces = [
                    defo_forces[0],  # epsilon
                    defo_forces[1],  # kappaz
                    0.0,  # kappay
                    0.0,  # theta
                    defo_forces[2],  # P
                    defo_forces[3],  # Mz
                    0.0,  # My
                    0.0,  # T
                ]
            elif len(defo_forces) == 0:
                defo_forces = [0.0] * 8
            defo.append(defo_forces[:4])
            force.append(defo_forces[4:])
        all_defo.append(np.array(defo))
        all_force.append(np.array(force))
    all_defo = _expand_to_uniform_array(all_defo, dtype=dtype["float"])
    all_force = _expand_to_uniform_array(all_force, dtype=dtype["float"])

    return all_stress, all_strains, all_defo, all_force


def _get_fiber_sec_data(ele_tag: int, sec_num: int = 1, dtype: Optional[dict] = None):
    """Get the fiber sec data for a beam element.

    Parameters
    ----------
    ele_tag: int
        The element tag to which the sec is to be displayed.
    sec_num: int
        Which integration point sec is displayed, tag from 1 from segment i to j.

    Returns
    -------
    fiber_data: ArrayLike
    """
    if dtype is None:
        dtype = {"float": np.float32, "int": np.int32}
    # Extract fiber data using eleResponse() command
    sec_loc = ops.sectionLocation(ele_tag)
    if len(sec_loc) == 0:
        raise ValueError(f"eleTag {ele_tag} have no fiber sec!")  # noqa: TRY003
    if sec_num > len(sec_loc):
        warnings.warn(
            f"Section number {sec_num} larger than max number {len(sec_loc)} of elemeng with tag {ele_tag}!"
            f"Section number {sec_num} set to {len(sec_loc)}!",
            stacklevel=2,
        )
        sec_num = len(sec_loc)
    ele_tag = int(ele_tag)
    fiber_data = ops.eleResponse(ele_tag, "section", f"{sec_num}", "fiberData2")
    if len(fiber_data) == 0:
        fiber_data = ops.eleResponse(ele_tag, "section", "fiberData2")
    # From column 1 to 6: "yCoord", "zCoord", "area", 'mat', "stress", "strain"
    fiber_data = np.reshape(fiber_data, (-1, 6))  # to six columns
    return fiber_data.astype(dtype["float"])
