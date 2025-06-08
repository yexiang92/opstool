from typing import Optional

import openseespy.opensees as ops
import xarray as xr

from ..model_data import GetFEMData
from ._response_base import ResponseBase


class ModelInfoStepData(ResponseBase):
    def __init__(self, model_update: bool = False):
        self.model_update = model_update
        self.model_info_steps = {}
        # -----------------------------------------
        self.times = None
        self.step_track = 0
        self.init = False
        self.initialize()

    def is_model_update(self):
        return self.model_update

    def initialize(self):
        self.times = [0.0]
        # --------------------------------------------------------
        model_info, _ = GetFEMData().get_model_info()
        # ------------------------------------------------------------
        for key, value in model_info.items():
            self.model_info_steps[key] = [value]
        # ------------------------------------------------------------------
        self.init = True
        self.step_track = 0

    def reset(self):
        self.initialize()

    def add_data_one_step(self):
        if self.model_update:
            model_info, _ = GetFEMData().get_model_info()
            for key, value in model_info.items():
                self.model_info_steps[key].append(value)
            self.times.append(ops.getTime())
        self.step_track += 1

    def _to_xarray(self):
        for key, data in self.model_info_steps.items():
            new_data = xr.concat(data, dim="time", join="outer")
            new_data.coords["time"] = self.times
            self.model_info_steps[key] = new_data
        model_update = 1 if self.model_update else 0
        self.model_info_steps["ModelUpdate"] = xr.DataArray(model_update, name="ModelUpdate")

    def get_current_node_tags(self):
        da = self.model_info_steps["NodalData"][-1]
        node_tags = list(da.coords["nodeTags"].data)
        unused_node_tags = da.attrs["unusedNodeTags"]
        for tag in unused_node_tags:
            if tag in node_tags:
                node_tags.remove(tag)
        return node_tags

    def get_current_truss_tags(self):
        da = self.model_info_steps["TrussData"][-1]
        if len(da) > 0:
            return da.coords["eleTags"].values
        return []

    def get_current_frame_tags(self):
        da = self.model_info_steps["BeamData"][-1]
        if len(da) > 0:
            return da.coords["eleTags"].values
        return []

    def get_current_link_tags(self):
        da = self.model_info_steps["LinkData"][-1]
        if len(da) > 0:
            return da.coords["eleTags"].values
        return []

    def get_current_shell_tags(self):
        da = self.model_info_steps["ShellData"][-1]
        if len(da) > 0:
            return da.coords["eleTags"].values
        return []

    def get_current_plane_tags(self):
        da = self.model_info_steps["PlaneData"][-1]
        if len(da) > 0:
            return da.coords["eleTags"].values
        return []

    def get_current_brick_tags(self):
        da = self.model_info_steps["BrickData"][-1]
        if len(da) > 0:
            return da.coords["eleTags"].values
        return []

    def get_current_contact_tags(self):
        da = self.model_info_steps["ContactData"][-1]
        if len(da) > 0:
            return da.coords["eleTags"].values
        return []

    def get_current_frame_load_data(self):
        da = self.model_info_steps["EleLoadData"][-1]
        if len(da) > 0:
            return da
        return []

    def get_data(self):
        return self.model_info_steps

    def get_track(self):
        return self.step_track

    def save_file(self, dt: xr.DataTree):
        self._to_xarray()
        model_info_steps = self.model_info_steps
        for data in model_info_steps.values():
            dt[f"ModelInfo/{data.name}"] = xr.Dataset({data.name: data})
        return dt

    @staticmethod
    def read_file(dt: xr.DataTree, unit_factors: Optional[dict] = None):
        model_info = {}
        for key, value in dt["ModelInfo"].items():
            model_info[key] = value[key]
        model_update = int(model_info["ModelUpdate"])
        model_update = model_update == 1

        if unit_factors:
            model_info = ModelInfoStepData._unit_transform(model_info, unit_factors)

        return model_info, model_update

    @staticmethod
    def _unit_transform(model_info, unit_factors):
        disp_factor = unit_factors["disp"]

        model_info["NodalData"] *= disp_factor
        model_info["NodalData"].attrs["minBoundSize"] *= disp_factor
        model_info["NodalData"].attrs["maxBoundSize"] *= disp_factor
        bounds = model_info["NodalData"].attrs["bounds"]
        model_info["NodalData"].attrs["bounds"] = (data * disp_factor for data in bounds)

        if "info" in model_info["FixedNodalData"].coords:
            model_info["FixedNodalData"].loc[{"info": ["x", "y", "z"]}] *= disp_factor
        if "info" in model_info["MPConstraintData"].coords:
            model_info["MPConstraintData"].loc[{"info": ["xo", "yo", "zo"]}] *= disp_factor
        model_info["eleCenters"] *= disp_factor

        return model_info

    @staticmethod
    def read_data(dt: xr.DataTree, data_type: str):
        """Read data from the data tree

        Parameters:
        -----------
        dt: xr.DataTree
            The data tree.
        data_type: str
            The data type to read.
        """
        model_update = int(dt["ModelInfo"]["ModelUpdate"]["ModelUpdate"])
        data = dt["ModelInfo"][data_type][data_type]
        if model_update == 1:
            return data
        return data.isel(time=0)
