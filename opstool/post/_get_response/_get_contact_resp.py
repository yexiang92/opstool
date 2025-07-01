from typing import Optional

import numpy as np
import openseespy.opensees as ops
import xarray as xr

from ...utils import suppress_ops_print
from ._response_base import ResponseBase


class ContactRespStepData(ResponseBase):
    def __init__(self, ele_tags=None, model_update: bool = False, dtype: Optional[dict] = None):
        self.resp_names = ["globalForces", "localForces", "localDisp", "slips"]
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
            "Px": "Global force in the x-direction on the constrained node",
            "Py": "Global force in the y-direction on the constrained node",
            "Pz": "Global force in the z-direction on the constrained node",
            "N": "Normal force or deformation",
            "Tx": "Tangential force or deformation in the x-direction",
            "Ty": "Tangential force or deformation in the y-direction",
        }

        self.initialize()

    def initialize(self):
        self.resp_steps = None
        self.resp_steps_list = []
        for name in self.resp_names:
            self.resp_steps_dict[name] = []
        self.add_data_one_step(self.ele_tags)
        self.step_track = 0
        self.times = [0.0]

    def reset(self):
        self.initialize()

    def add_data_one_step(self, ele_tags):
        with suppress_ops_print():
            global_forces, forces, defos, slips = _get_contact_resp(ele_tags, dtype=self.dtype)

        if self.model_update:
            data_vars = {}
            if len(ele_tags) > 0:
                data_vars["globalForces"] = (["eleTags", "globalDOFs"], global_forces)
                data_vars["localForces"] = (["eleTags", "localDOFs"], forces)
                data_vars["localDisp"] = (["eleTags", "localDOFs"], defos)
                data_vars["slips"] = (["eleTags", "slipDOFs"], slips)
                ds = xr.Dataset(
                    data_vars=data_vars,
                    coords={
                        "eleTags": ele_tags,
                        "globalDOFs": ["Px", "Py", "Pz"],
                        "localDOFs": ["N", "Tx", "Ty"],
                        "slipDOFs": ["Tx", "Ty"],
                    },
                    attrs=self.attrs,
                )
            else:
                data_vars["globalForces"] = xr.DataArray([])
                data_vars["localForces"] = xr.DataArray([])
                data_vars["localDisp"] = xr.DataArray([])
                data_vars["slips"] = xr.DataArray([])
                ds = xr.Dataset(data_vars=data_vars)
            self.resp_steps_list.append(ds)
        else:
            datas = [global_forces, forces, defos, slips]
            for name, da in zip(self.resp_names, datas):
                self.resp_steps_dict[name].append(da)

        self.times.append(ops.getTime())
        self.step_track += 1

    def _to_xarray(self):
        self.times = np.array(self.times, dtype=self.dtype["float"])
        if self.model_update:
            self.resp_steps = xr.concat(self.resp_steps_list, dim="time", join="outer")
            self.resp_steps.coords["time"] = self.times
        else:
            data_vars = {}
            data_vars["globalForces"] = (["time", "eleTags", "globalDOFs"], self.resp_steps_dict["globalForces"])
            data_vars["localForces"] = (["time", "eleTags", "localDOFs"], self.resp_steps_dict["localForces"])
            data_vars["localDisp"] = (["time", "eleTags", "localDOFs"], self.resp_steps_dict["localDisp"])
            data_vars["slips"] = (["time", "eleTags", "slipDOFs"], self.resp_steps_dict["slips"])
            self.resp_steps = xr.Dataset(
                data_vars=data_vars,
                coords={
                    "time": self.times,
                    "eleTags": self.ele_tags,
                    "globalDOFs": ["Px", "Py", "Pz"],
                    "localDOFs": ["N", "Tx", "Ty"],
                    "slipDOFs": ["Tx", "Ty"],
                },
                attrs=self.attrs,
            )

    def get_data(self):
        return self.resp_steps

    def get_track(self):
        return self.step_track

    def save_file(self, dt: xr.DataTree):
        self._to_xarray()
        dt["/ContactResponses"] = self.resp_steps
        return dt

    @staticmethod
    def read_file(dt: xr.DataTree, unit_factors: Optional[dict] = None):
        resp_steps = dt["/ContactResponses"].to_dataset()
        if unit_factors is not None:
            resp_steps = ContactRespStepData._unit_transform(resp_steps, unit_factors)
        return resp_steps

    @staticmethod
    def _unit_transform(resp_steps, unit_factors):
        force_factor = unit_factors["force"]
        disp_factor = unit_factors["disp"]

        resp_steps["globalForces"] *= force_factor
        resp_steps["localForces"] *= force_factor
        resp_steps["localDisp"] *= disp_factor
        resp_steps["slips"] *= disp_factor

        return resp_steps

    @staticmethod
    def read_response(
        dt: xr.DataTree, resp_type: Optional[str] = None, ele_tags=None, unit_factors: Optional[dict] = None
    ):
        ds = ContactRespStepData.read_file(dt, unit_factors=unit_factors)
        if resp_type is None:
            if ele_tags is None:
                return ds
            else:
                return ds.sel(eleTags=ele_tags)
        else:
            if resp_type not in list(ds.keys()):
                raise ValueError(  # noqa: TRY003
                    f"resp_type {resp_type} not found in {list(ds.keys())}"
                )
            if ele_tags is not None:
                return ds[resp_type].sel(eleTags=ele_tags)
            else:
                return ds[resp_type]


def _get_contact_resp(link_tags, dtype):
    defos, forces, slips, global_forces = [], [], [], []
    for etag in link_tags:
        etag = int(etag)
        global_fo = _get_contact_resp_by_type(etag, ("force", "forces"), type_="global")
        defo = _get_contact_resp_by_type(etag, ("localDisplacement", "localDispJump"), type_="local")
        force = _get_contact_resp_by_type(
            etag, ("localForce", "localForces", "forcescalars", "forcescalar"), type_="local"
        )
        slip = _get_contact_resp_by_type(etag, ("slip",), type_="slip")
        global_forces.append(global_fo)
        defos.append(defo)
        forces.append(force)
        slips.append(slip)
    defos = np.array(defos, dtype=dtype["float"])
    forces = np.array(forces, dtype=dtype["float"])
    slips = np.array(slips, dtype=dtype["float"])
    global_forces = np.array(global_forces, dtype=dtype["float"])
    return global_forces, forces, defos, slips


def _get_contact_resp_by_type(etag, etypes, type_="local"):
    etag = int(etag)
    resp = _get_valid_ele_response(etag, etypes)

    if type_ == "local":
        return _format_local_response(resp)
    elif type_ == "global":
        return _format_global_response(resp)
    elif type_ == "slip":
        return _format_slip_response(resp)
    else:
        raise ValueError(f"Unsupported response type: {type_}")  # noqa: TRY003


def _get_valid_ele_response(etag, etypes):
    for name in etypes:
        resp = ops.eleResponse(etag, name)
        if resp:
            return resp
    return []


def _format_local_response(resp):
    if len(resp) == 0:
        return [0.0, 0.0, 0.0]
    if len(resp) == 2:
        return [resp[0], resp[1], 0.0]
    return resp[:3]


def _format_global_response(resp):
    if len(resp) == 0:
        return [0.0, 0.0, 0.0]
    if len(resp) == 2:
        return [resp[0], resp[1], 0.0]
    if len(resp) in (4, 6):
        return resp[-3:]
    return resp[-3:]


def _format_slip_response(resp):
    if len(resp) == 0:
        return [0.0, 0.0]
    if len(resp) == 1:
        return [resp[0], resp[0]]
    return resp[:2]
