from typing import Optional

import numpy as np
import openseespy.opensees as ops
import xarray as xr

from ._response_base import ResponseBase


class TrussRespStepData(ResponseBase):
    def __init__(self, ele_tags=None, model_update: bool = False, dtype: dict = None):
        self.resp_names = ["axialForce", "axialDefo", "Stress", "Strain"]
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
        data = _get_truss_resp(ele_tags, dtype=self.dtype)

        if self.model_update:
            data_vars = {}
            if len(ele_tags) > 0:
                for name, data_ in zip(self.resp_names, data):
                    data_vars[name] = (["eleTags"], data_)
                ds = xr.Dataset(data_vars=data_vars, coords={"eleTags": ele_tags})
            else:
                for name in self.resp_names, data:
                    data_vars[name] = xr.DataArray([])
                ds = xr.Dataset(data_vars=data_vars)
            self.resp_steps_list.append(ds)
        else:
            for name, data_ in zip(self.resp_names, data):
                self.resp_steps_dict[name].append(data_)

        self.times.append(ops.getTime())
        self.step_track += 1

    def _to_xarray(self):
        if self.model_update:
            self.resp_steps = xr.concat(self.resp_steps_list, dim="time", join="outer")
            self.resp_steps.coords["time"] = self.times
        else:
            data_vars = {}
            for name, data in self.resp_steps_dict.items():
                data_vars[name] = (["time", "eleTags"], data)
            self.resp_steps = xr.Dataset(
                data_vars=data_vars,
                coords={"time": self.times, "eleTags": self.ele_tags},
            )

    def get_data(self):
        return self.resp_steps

    def get_track(self):
        return self.step_track

    def save_file(self, dt: xr.DataTree):
        self._to_xarray()
        dt["/TrussResponses"] = self.resp_steps
        return dt

    @staticmethod
    def read_file(dt: xr.DataTree, unit_factors: Optional[dict] = None):
        resp_steps = dt["/TrussResponses"].to_dataset()
        if unit_factors is not None:
            resp_steps = TrussRespStepData._unit_transform(resp_steps, unit_factors)
        return resp_steps

    @staticmethod
    def _unit_transform(resp_steps, unit_factors):
        force_factor = unit_factors["force"]
        disp_factor = unit_factors["disp"]
        stress_factor = unit_factors["stress"]

        resp_steps["axialForce"] *= force_factor
        resp_steps["axialDefo"] *= disp_factor
        resp_steps["Stress"] *= stress_factor

        return resp_steps

    @staticmethod
    def read_response(
        dt: xr.DataTree, resp_type: Optional[str] = None, ele_tags=None, unit_factors: Optional[dict] = None
    ):
        ds = TrussRespStepData.read_file(dt, unit_factors=unit_factors)
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


def _get_truss_resp(truss_tags, dtype: dict):
    forces, defos, stressss, strains = [], [], [], []
    for etag in truss_tags:
        etag = int(etag)
        force = ops.eleResponse(etag, "axialForce")
        force = _reshape_resp(force)
        defo = ops.eleResponse(etag, "basicDeformation")
        defo = _reshape_resp(defo)
        stress = ops.eleResponse(etag, "material", "1", "stress")
        stress = _reshape_resp(stress)

        strain = ops.eleResponse(etag, "material", "1", "strain")
        if len(strain) == 0:
            strain = ops.eleResponse(etag, "section", "1", "deformation")
        strain = _reshape_resp(strain)

        forces.append(force)
        defos.append(defo)
        stressss.append(stress)
        strains.append(strain)

    forces = np.array(forces, dtype=dtype["float"])
    defos = np.array(defos, dtype=dtype["float"])
    stressss = np.array(stressss, dtype=dtype["float"])
    strains = np.array(strains, dtype=dtype["float"])
    return forces, defos, stressss, strains


def _reshape_resp(data):
    if len(data) == 0:
        return 0.0
    else:
        return data[0]
