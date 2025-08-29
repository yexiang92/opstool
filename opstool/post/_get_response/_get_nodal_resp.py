from typing import Optional

import numpy as np
import openseespy.opensees as ops
import xarray as xr

from ._response_base import ResponseBase


class NodalRespStepData(ResponseBase):
    def __init__(self, node_tags=None, model_update: bool = False, dtype: Optional[dict] = None):
        self.resp_names = [
            "disp",
            "vel",
            "accel",
            "reaction",
            "reactionIncInertia",
            "rayleighForces",
            "pressure",
        ]
        self.node_tags = node_tags if node_tags is not None else ops.getNodeTags()
        self.resp_steps = None
        self.resp_steps_list = []  # for model update
        self.resp_steps_dict = {}  # for non-update
        self.times = []
        self.step_track = 0

        self.model_update = model_update
        self.dtype = {"int": np.int32, "float": np.float32}
        if isinstance(dtype, dict):
            self.dtype.update(dtype)

        self.attrs = {
            "UX": "Displacement in X direction",
            "UY": "Displacement in Y direction",
            "UZ": "Displacement in Z direction",
            "RX": "Rotation about X axis",
            "RY": "Rotation about Y axis",
            "RZ": "Rotation about Z axis",
        }

        self.initialize()

    def initialize(self):
        self.resp_steps = None
        self.resp_steps_list = []
        for name in self.resp_names:
            self.resp_steps_dict[name] = []
        self.add_data_one_step(self.node_tags)
        self.times = [0.0]
        self.step_track = 0

    def reset(self):
        self.initialize()

    def add_data_one_step(self, node_tags):
        # node_tags = ops.getNodeTags()
        disp, vel, accel, pressure = _get_nodal_resp(node_tags, dtype=self.dtype)
        reacts, reacts_inertia, rayleigh_forces = _get_nodal_react(node_tags, dtype=self.dtype)
        if self.model_update:
            datas = [disp, vel, accel, reacts, reacts_inertia, rayleigh_forces]
            data_vars = {}
            for name, data_ in zip(self.resp_names, datas):
                data_vars[name] = (["nodeTags", "DOFs"], data_)
            data_vars["pressure"] = (["nodeTags"], pressure)
            # can have different dimensions and coordinates
            ds = xr.Dataset(
                data_vars=data_vars,
                coords={
                    "nodeTags": node_tags,
                    "DOFs": ["UX", "UY", "UZ", "RX", "RY", "RZ"],
                },
                attrs=self.attrs,
            )
            self.resp_steps_list.append(ds)
        else:  # non-update
            datas = [disp, vel, accel, reacts, reacts_inertia, rayleigh_forces, pressure]
            for name, data_ in zip(self.resp_names, datas):
                self.resp_steps_dict[name].append(data_)
        self.times.append(ops.getTime())
        self.step_track += 1

    def get_data(self):
        return self.resp_steps

    def get_track(self):
        return self.step_track

    def _to_xarray(self):
        self.times = np.array(self.times, dtype=self.dtype["float"])
        if self.model_update:
            self.resp_steps = xr.concat(self.resp_steps_list, dim="time", join="outer")
            self.resp_steps.coords["time"] = self.times
        else:
            data_vars = {}
            for name in self.resp_names[:-1]:
                data_vars[name] = (["time", "nodeTags", "DOFs"], self.resp_steps_dict[name])
            data_vars["pressure"] = (["time", "nodeTags"], self.resp_steps_dict["pressure"])
            self.resp_steps = xr.Dataset(
                data_vars=data_vars,
                coords={
                    "time": self.times,
                    "nodeTags": self.node_tags,
                    "DOFs": ["UX", "UY", "UZ", "RX", "RY", "RZ"],
                },
                attrs=self.attrs,
            )

    def add_to_datatree(self, dt: xr.DataTree):
        self._to_xarray()
        dt["/NodalResponses"] = self.resp_steps
        return dt

    @staticmethod
    def read_datatree(dt: xr.DataTree, unit_factors: Optional[dict] = None):
        # (eleTag, steps, resp_type)
        resp_steps = dt["/NodalResponses"].to_dataset()
        if unit_factors is not None:
            resp_steps = NodalRespStepData._unit_transform(resp_steps, unit_factors)
        return resp_steps

    @staticmethod
    def _unit_transform(resp_steps, unit_factors):
        disp_factor = unit_factors["disp"]
        vel_factor = unit_factors["vel"]
        accel_factor = unit_factors["accel"]
        angular_vel_fact = unit_factors["angular_vel"]
        angular_accel_fact = unit_factors["angular_accel"]
        force_factor = unit_factors["force"]
        moment_factor = unit_factors["moment"]
        stress_factor = unit_factors["stress"]

        resp_steps["disp"].loc[{"DOFs": ["UX", "UY", "UZ"]}] *= disp_factor
        resp_steps["vel"].loc[{"DOFs": ["UX", "UY", "UZ"]}] *= vel_factor
        resp_steps["vel"].loc[{"DOFs": ["RX", "RY", "RZ"]}] *= angular_vel_fact
        resp_steps["accel"].loc[{"DOFs": ["UX", "UY", "UZ"]}] *= accel_factor
        resp_steps["accel"].loc[{"DOFs": ["RX", "RY", "RZ"]}] *= angular_accel_fact

        resp_steps["reaction"].loc[{"DOFs": ["UX", "UY", "UZ"]}] *= force_factor
        resp_steps["reaction"].loc[{"DOFs": ["RX", "RY", "RZ"]}] *= moment_factor
        resp_steps["reactionIncInertia"].loc[{"DOFs": ["UX", "UY", "UZ"]}] *= force_factor
        resp_steps["reactionIncInertia"].loc[{"DOFs": ["RX", "RY", "RZ"]}] *= moment_factor
        resp_steps["rayleighForces"].loc[{"DOFs": ["UX", "UY", "UZ"]}] *= force_factor
        resp_steps["rayleighForces"].loc[{"DOFs": ["RX", "RY", "RZ"]}] *= moment_factor
        resp_steps["pressure"] *= stress_factor

        return resp_steps

    @staticmethod
    def read_response(
        dt: xr.DataTree, resp_type: Optional[str] = None, node_tags=None, unit_factors: Optional[dict] = None
    ):
        ds = NodalRespStepData.read_datatree(dt, unit_factors=unit_factors)
        if resp_type is None:
            if node_tags is None:
                return ds
            else:
                return ds.sel(nodeTags=node_tags)
        else:
            if resp_type not in list(ds.keys()):
                raise ValueError(f"resp_type {resp_type} not found in {list(ds.keys())}")  # noqa: TRY003
            if node_tags is not None:
                return ds[resp_type].sel(nodeTags=node_tags)
            else:
                return ds[resp_type]


def handle_1d(disp, vel, accel):
    return (
        [*disp, 0.0, 0.0, 0.0, 0.0, 0.0],
        [*vel, 0.0, 0.0, 0.0, 0.0, 0.0],
        [*accel, 0.0, 0.0, 0.0, 0.0, 0.0],
    )


def handle_2d(disp, vel, accel):
    if len(disp) == 1:
        return handle_1d(disp, vel, accel)
    elif len(disp) == 2:
        return (
            [*disp, 0.0, 0.0, 0.0, 0.0],
            [*vel, 0.0, 0.0, 0.0, 0.0],
            [*accel, 0.0, 0.0, 0.0, 0.0],
        )
    elif len(disp) >= 3:
        # Assume (ux, uy, rz)
        return (
            [disp[0], disp[1], 0.0, 0.0, 0.0, disp[2]],
            [vel[0], vel[1], 0.0, 0.0, 0.0, vel[2]],
            [accel[0], accel[1], 0.0, 0.0, 0.0, accel[2]],
        )


def handle_3d(disp, vel, accel):
    if len(disp) == 3:
        return (
            [*disp, 0.0, 0.0, 0.0],
            [*vel, 0.0, 0.0, 0.0],
            [*accel, 0.0, 0.0, 0.0],
        )
    elif len(disp) == 4:
        return (
            [disp[0], disp[1], disp[2], 0.0, 0.0, disp[3]],
            [vel[0], vel[1], vel[2], 0.0, 0.0, vel[3]],
            [accel[0], accel[1], accel[2], 0.0, 0.0, accel[3]],
        )
    elif len(disp) < 6:
        pad_len = 6 - len(disp)
        return (
            disp + [0.0] * pad_len,
            vel + [0.0] * pad_len,
            accel + [0.0] * pad_len,
        )
    else:
        return (
            disp[:6],
            vel[:6],
            accel[:6],
        )


def _get_nodal_resp(node_tags, dtype: dict):
    node_disp, node_vel, node_accel, node_pressure = [], [], [], []
    all_node_tags = set(ops.getNodeTags())

    for tag in map(int, node_tags):
        if tag in all_node_tags:
            coord = ops.nodeCoord(tag)
            ndim = len(coord)
            disp = ops.nodeDisp(tag)
            vel = ops.nodeVel(tag)
            accel = ops.nodeAccel(tag)

            if ndim == 1:
                d, v, a = handle_1d(disp, vel, accel)
            elif ndim == 2:
                d, v, a = handle_2d(disp, vel, accel)
            else:
                d, v, a = handle_3d(disp, vel, accel)
        else:
            d = v = a = [np.nan] * 6

        node_disp.append(d)
        node_vel.append(v)
        node_accel.append(a)
        node_pressure.append(ops.nodePressure(tag))

    return (
        np.array(node_disp, dtype=dtype["float"]),
        np.array(node_vel, dtype=dtype["float"]),
        np.array(node_accel, dtype=dtype["float"]),
        np.array(node_pressure, dtype=dtype["float"]),
    )


def _get_react(tags):
    forces = []  # 6 data each row, Ux, Uy, Uz, Rx, Ry, Rz
    for tag in tags:
        tag = int(tag)
        if tag in ops.getNodeTags():
            coord = ops.nodeCoord(tag)
            fo = ops.nodeReaction(tag)
            ndim, ndf = len(coord), len(fo)
            if ndim == 1 or (ndim == 2 and ndf == 1):
                fo.extend([0.0, 0.0, 0.0, 0.0, 0.0])
            elif ndim == 2 and ndf == 2:
                fo.extend([0.0, 0.0, 0.0, 0.0])
            elif ndim == 2 and ndf >= 3:
                fo = [fo[0], fo[1], 0.0, 0.0, 0.0, fo[2]]
            elif ndim == 3 and ndf == 3:
                fo.extend([0.0, 0.0, 0.0])
            elif ndim == 3 and ndf < 6:  # 3 ndim 6 dof
                fo.extend([0] * (6 - len(fo)))
            elif ndim == 3 and ndf > 6:
                fo = fo[:6]
        else:
            fo = [np.nan] * 6
        forces.append(fo)
    return forces


def _get_nodal_react(node_tags, dtype: dict):
    ops.reactions()
    reacts = np.array(_get_react(node_tags), dtype=dtype["float"])
    # rayleighForces
    ops.reactions("-rayleigh")
    rayleigh_forces = np.array(_get_react(node_tags), dtype=dtype["float"])
    # Include Inertia
    ops.reactions("-dynamic")
    reacts_inertia = np.array(_get_react(node_tags), dtype=dtype["float"])
    return reacts, reacts_inertia, rayleigh_forces
