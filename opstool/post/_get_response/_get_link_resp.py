from typing import Optional

import numpy as np
import openseespy.opensees as ops
import xarray as xr

from ._response_base import ResponseBase


class LinkRespStepData(ResponseBase):
    def __init__(self, ele_tags=None, model_update: bool = False, dtype: Optional[dict] = None):
        self.resp_names = ["basicDeformation", "basicForce"]
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
            "DOFs": "The DOFs are aligned with the local coordinate system. "
            "Note that these DOFs are not necessarily valid unless all degrees of freedom are "
            "assigned to the material (e.g., all six DOFs in 3D). "
            "For cases where the material is assigned to only partial DOFs, "
            "the actual DOFs are arranged sequentially, with the remaining ones padded with zeros."
        }
        self.DOFs = ["UX", "UY", "UZ", "RX", "RY", "RZ"]

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
        data = _get_link_resp(ele_tags, dtype=self.dtype)

        if self.model_update:
            data_vars = {}
            if len(ele_tags) > 0:
                for name, data_ in zip(self.resp_names, data):
                    data_vars[name] = (["eleTags", "DOFs"], data_)
                ds = xr.Dataset(
                    data_vars=data_vars,
                    coords={
                        "eleTags": ele_tags,
                        "DOFs": self.DOFs,
                    },
                    attrs=self.attrs,
                )
            else:
                for name in self.resp_names:
                    data_vars[name] = xr.DataArray([])
                ds = xr.Dataset(data_vars=data_vars)
            self.resp_steps_list.append(ds)
        else:
            for name, data_ in zip(self.resp_names, data):
                self.resp_steps_dict[name].append(data_)

        self.times.append(ops.getTime())
        self.step_track += 1

    def _to_xarray(self):
        self.times = np.array(self.times, dtype=self.dtype["float"])
        if self.model_update:
            self.resp_steps = xr.concat(self.resp_steps_list, dim="time", join="outer")
            self.resp_steps.coords["time"] = self.times
        else:
            data_vars = {}
            for name, data_ in self.resp_steps_dict.items():
                data_vars[name] = (["time", "eleTags", "DOFs"], data_)
            self.resp_steps = xr.Dataset(
                data_vars=data_vars,
                coords={
                    "time": self.times,
                    "eleTags": self.ele_tags,
                    "DOFs": self.DOFs,
                },
                attrs=self.attrs,
            )

    def get_data(self):
        return self.resp_steps

    def get_track(self):
        return self.step_track

    def save_file(self, dt: xr.DataTree):
        self._to_xarray()
        dt["/LinkResponses"] = self.resp_steps
        return dt

    @staticmethod
    def read_file(dt: xr.DataTree, unit_factors: Optional[dict] = None):
        resp_steps = dt["/LinkResponses"].to_dataset()
        if unit_factors is not None:
            resp_steps = LinkRespStepData._unit_transform(resp_steps, unit_factors)
        return resp_steps

    @staticmethod
    def _unit_transform(resp_steps, unit_factors):
        force_factor = unit_factors["force"]
        moment_factor = unit_factors["moment"]
        disp_factor = unit_factors["disp"]

        # ---------------------------------------------------------
        resp_steps["basicForce"].loc[{"DOFs": ["UX", "UY", "UZ"]}] *= force_factor
        resp_steps["basicForce"].loc[{"DOFs": ["RX", "RY", "RZ"]}] *= moment_factor
        # ---------------------------------------------------------
        resp_steps["basicDeformation"].loc[{"DOFs": ["UX", "UY", "UZ"]}] *= disp_factor

        return resp_steps

    @staticmethod
    def read_response(
        dt: xr.DataTree, resp_type: Optional[str] = None, ele_tags=None, unit_factors: Optional[dict] = None
    ):
        ds = LinkRespStepData.read_file(dt, unit_factors=unit_factors)
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


def _get_link_resp(link_tags, dtype):
    defos, forces = [], []
    for etag in link_tags:
        etag = int(etag)
        defo = _get_link_resp_by_type(
            etag,
            (
                "basicDeformations",
                "basicDeformation",
                "deformations",
                "deformation",
                "basicDisplacements",
                "basicDisplacement",
            ),
        )
        force = _get_link_resp_by_type(etag, ("basicForces", "basicForce"))
        defos.append(defo)
        forces.append(force)
    defos = np.array(defos, dtype=dtype["float"])
    forces = np.array(forces, dtype=dtype["float"])
    return defos, forces


def _get_link_resp_by_type(etag, etypes):
    etag = int(etag)
    ntags = ops.eleNodes(etag)
    ndim = len(ops.nodeCoord(ntags[0]))
    resp = []
    for name in etypes:
        resp = ops.eleResponse(etag, name)
        if len(resp) > 0:
            break
    if len(resp) == 0:
        resp = [0.0] * 6
    elif ndim == 2 and len(resp) == 3:
        resp = [resp[0], resp[1], 0.0, 0.0, 0.0, resp[2]]
    elif len(resp) < 6:  # don't know dofs
        resp = resp + [0.0] * (6 - len(resp))
    elif len(resp) > 6:
        resp = resp[:6]
    return resp
