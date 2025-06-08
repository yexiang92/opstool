from typing import Optional

import numpy as np
import openseespy.opensees as ops
import xarray as xr

from ._response_base import ResponseBase, _expand_to_uniform_array

# from ...utils import OPS_ELE_TAGS


class BrickRespStepData(ResponseBase):
    def __init__(self, ele_tags=None, compute_measures: bool = True, model_update: bool = False, dtype: dict = None):
        self.resp_names = [
            "Stresses",
            "Strains",
        ]
        self.resp_steps = None
        self.resp_steps_list = []  # for model update
        self.resp_steps_dict = {}  # for non-update
        self.step_track = 0
        self.ele_tags = ele_tags
        self.times = []

        self.compute_measures = compute_measures
        self.model_update = model_update
        self.dtype = {"int": np.int32, "float": np.float32}
        if isinstance(dtype, dict):
            self.dtype.update(dtype)

        self.attrs = {
            "sigma11, sigma22, sigma33": "Normal stress (strain) along x, y, z.",
            "sigma12, sigma23, sigma13": "Shear stress (strain).",
            "p1, p2, p3": "Principal stresses (strains).",
            "eta_r": "Ratio between the shear (deviatoric) stress and peak shear strength at the current confinement",
            "sigma_vm": "Von Mises stress.",
            "tau_max": "Maximum shear stress (strains).",
            "sigma_oct": "Octahedral normal stress (strains).",
            "tau_oct": "Octahedral shear stress (strains).",
        }
        self.GaussPoints = None
        self.stressDOFs = None
        self.strainDOFs = ["eps11", "eps22", "eps33", "eps12", "eps23", "eps13"]

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
        stresses, strains = _get_gauss_resp(ele_tags, dtype=self.dtype)

        if self.stressDOFs is None:
            if stresses.shape[-1] == 6:
                self.stressDOFs = ["sigma11", "sigma22", "sigma33", "sigma12", "sigma23", "sigma13"]
            elif stresses.shape[-1] == 7:
                self.stressDOFs = ["sigma11", "sigma22", "sigma33", "sigma12", "sigma23", "sigma13", "eta_r"]
            else:
                self.stressDOFs = [f"sigma{i + 1}" for i in stresses.shape[-1]]
        if self.GaussPoints is None:
            self.GaussPoints = np.arange(stresses.shape[1]) + 1

        if self.model_update:
            data_vars = {}
            data_vars["Stresses"] = (["eleTags", "GaussPoints", "stressDOFs"], stresses)
            data_vars["Strains"] = (["eleTags", "GaussPoints", "strainDOFs"], strains)
            ds = xr.Dataset(
                data_vars=data_vars,
                coords={
                    "eleTags": ele_tags,
                    "GaussPoints": self.GaussPoints,
                    "stressDOFs": self.stressDOFs,
                    "strainDOFs": self.strainDOFs,
                },
                attrs=self.attrs,
            )
            self.resp_steps_list.append(ds)
        else:
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
            data_vars["Stresses"] = (["time", "eleTags", "GaussPoints", "stressDOFs"], self.resp_steps_dict["Stresses"])
            data_vars["Strains"] = (["time", "eleTags", "GaussPoints", "strainDOFs"], self.resp_steps_dict["Strains"])
            self.resp_steps = xr.Dataset(
                data_vars=data_vars,
                coords={
                    "time": self.times,
                    "eleTags": self.ele_tags,
                    "GaussPoints": self.GaussPoints,
                    "stressDOFs": self.stressDOFs,
                    "strainDOFs": self.strainDOFs,
                },
                attrs=self.attrs,
            )

        if self.compute_measures:
            self._compute_measures_()

    def _compute_measures_(self):
        stresses = self.resp_steps["Stresses"]
        strains = self.resp_steps["Strains"]

        stress_measures = _calculate_stresses_measures_4D(stresses.data, dtype=self.dtype)
        strain_measures = _calculate_stresses_measures_4D(strains.data, dtype=self.dtype)

        dims = ["time", "eleTags", "GaussPoints", "measures"]
        coords = {
            "time": stresses.coords["time"],
            "eleTags": stresses.coords["eleTags"],
            "GaussPoints": stresses.coords["GaussPoints"],
            "measures": ["p1", "p2", "p3", "sigma_vm", "tau_max", "sigma_oct", "tau_oct"],
        }

        self.resp_steps["stressMeasures"] = xr.DataArray(
            stress_measures,
            dims=dims,
            coords=coords,
            name="stressMeasures",
        )
        self.resp_steps["strainMeasures"] = xr.DataArray(
            strain_measures,
            dims=dims,
            coords=coords,
            name="strainMeasures",
        )

    def get_data(self):
        return self.resp_steps

    def get_track(self):
        return self.step_track

    def save_file(self, dt: xr.DataTree):
        self._to_xarray()
        dt["/SolidResponses"] = self.resp_steps
        return dt

    @staticmethod
    def read_file(dt: xr.DataTree, unit_factors: Optional[dict] = None):
        resp_steps = dt["/SolidResponses"].to_dataset()
        if unit_factors is not None:
            resp_steps = BrickRespStepData._unit_transform(resp_steps, unit_factors)
        return resp_steps

    @staticmethod
    def _unit_transform(resp_steps, unit_factors):
        stress_factor = unit_factors["stress"]

        resp_steps["Stresses"].loc[
            {"stressDOFs": ["sigma11", "sigma22", "sigma33", "sigma12", "sigma23", "sigma13"]}
        ] *= stress_factor

        resp_steps["stressMeasures"] *= stress_factor

        return resp_steps

    @staticmethod
    def read_response(
        dt: xr.DataTree, resp_type: Optional[str] = None, ele_tags=None, unit_factors: Optional[dict] = None
    ):
        ds = BrickRespStepData.read_file(dt, unit_factors=unit_factors)
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


def _get_gauss_resp(ele_tags, dtype: dict):
    all_stresses, all_strains = [], []
    for etag in ele_tags:
        etag = int(etag)
        integr_point_stress = []
        integr_point_strain = []
        for i in range(100000000):  # Ugly but useful
            # loop for integrPoint
            stress_ = ops.eleResponse(etag, "material", f"{i + 1}", "stresses")
            if len(stress_) == 0:
                stress_ = ops.eleResponse(etag, "integrPoint", f"{i + 1}", "stresses")
            strain_ = ops.eleResponse(etag, "material", f"{i + 1}", "strains")
            if len(strain_) == 0:
                strain_ = ops.eleResponse(etag, "integrPoint", f"{i + 1}", "strains")
            if len(stress_) == 0 or len(strain_) == 0:
                break
            integr_point_stress.append(stress_)
            integr_point_strain.append(strain_)
        # Call material response directly
        if len(integr_point_stress) == 0 or len(integr_point_strain) == 0:
            stress = ops.eleResponse(etag, "stresses")
            strain = ops.eleResponse(etag, "strains")
            if len(stress) > 0:
                integr_point_stress.append(stress)
            if len(strain) > 0:
                integr_point_strain.append(strain)
        # Finally, if void set to 0.0
        if len(integr_point_stress) == 0:
            integr_point_stress.append([np.nan, np.nan, np.nan, np.nan, np.nan, np.nan])
        if len(integr_point_strain) == 0:
            integr_point_strain.append([np.nan, np.nan, np.nan, np.nan, np.nan, np.nan])

        all_stresses.append(np.array(integr_point_stress))
        all_strains.append(np.array(integr_point_strain))
    stresses = _expand_to_uniform_array(all_stresses, dtype=dtype["float"])
    strains = _expand_to_uniform_array(all_strains, dtype=dtype["float"])
    return stresses, strains


def _calculate_stresses_measures_4D(stress_array, dtype):
    """
    Calculate various stresses from the stress values at Gaussian points.

    Parameters:
    stress_array (numpy.ndarray): A 4D array with shape (num_elements, num_gauss_points, num_stresses).

    Returns:
    dict: A dictionary containing the calculated stresses for each element.
    """
    # Extract individual stress components
    sig11 = stress_array[..., 0]  # Normal stress in x-direction
    sig22 = stress_array[..., 1]  # Normal stress in y-direction
    sig33 = stress_array[..., 2]  # Normal stress in z-direction
    sig12 = stress_array[..., 3]  # Shear stress in xy-plane
    sig23 = stress_array[..., 4]  # Shear stress in yz-plane
    sig13 = stress_array[..., 5]  # Shear stress in xz-plane

    # Calculate von Mises stress for each Gauss point
    sig_vm = np.sqrt(
        0.5 * ((sig11 - sig22) ** 2 + (sig22 - sig33) ** 2 + (sig33 - sig11) ** 2)
        + 3 * (sig12**2 + sig13**2 + sig23**2)
    )

    # Calculate principal stresses
    # Using the stress tensor to calculate eigenvalues
    stress_tensor = _assemble_stress_tensor_4D(stress_array)
    # Calculate principal stresses (eigenvalues)
    principal_stresses = np.linalg.eigvalsh(stress_tensor)  # Returns sorted eigenvalues
    p1 = principal_stresses[..., 2]  # Maximum principal stress
    p2 = principal_stresses[..., 1]  # Intermediate principal stress
    p3 = principal_stresses[..., 0]  # Minimum principal stress

    # Calculate maximum shear stress
    tau_max = np.abs(p1 - p3) / 2

    # Calculate octahedral normal stress
    sig_oct = (p1 + p2 + p3) / 3

    # Calculate octahedral shear stress
    tau_oct = 1 / 3 * np.sqrt((p1 - p2) ** 2 + (p2 - p3) ** 2 + (p3 - p1) ** 2)

    data = np.stack([p1, p2, p3, sig_vm, tau_max, sig_oct, tau_oct], axis=-1)

    return data.astype(dtype["float"])


def _assemble_stress_tensor_3D(stress_array):
    """
    Assemble a 3D stress array into a 4D stress tensor array.
    """
    num_elements, num_gauss_points, _ = stress_array.shape
    stress_tensor = np.full((num_elements, num_gauss_points, 3, 3), np.nan)
    for i in range(num_elements):
        for j in range(num_gauss_points):
            sig11 = stress_array[i, j, 0]  # Normal stress in x-direction
            sig22 = stress_array[i, j, 1]  # Normal stress in y-direction
            sig33 = stress_array[i, j, 2]  # Normal stress in z-direction
            tau12 = stress_array[i, j, 3]  # Shear stress in xy-plane
            tau23 = stress_array[i, j, 4]  # Shear stress in yz-plane
            tau13 = stress_array[i, j, 5]  # Shear stress in xz-plane
            st = np.array([
                [sig11, tau12, tau13],
                [tau12, sig22, tau23],
                [tau13, tau23, sig33],
            ])
            stress_tensor[i, j, ...] = st
    return np.nan_to_num(stress_tensor, nan=0.0)


def _assemble_stress_tensor_4D(stress_array):
    """
    Assemble a 4D stress array [time, eleTags, GaussPoints, 6]
    into a 5D stress tensor array [time, eleTags, GaussPoints, 3, 3].
    Handles NaNs safely (returns 0.0 where data is missing).

    Parameters:
        stress_array (np.ndarray): shape (time, eleTags, GaussPoints, 6)

    Returns:
        np.ndarray: shape (time, eleTags, GaussPoints, 3, 3)
    """
    num_time, num_elements, num_gauss_points, _ = stress_array.shape
    stress_tensor = np.full((num_time, num_elements, num_gauss_points, 3, 3), np.nan)

    for t in range(num_time):
        for i in range(num_elements):
            for j in range(num_gauss_points):
                sig11 = stress_array[t, i, j, 0]
                sig22 = stress_array[t, i, j, 1]
                sig33 = stress_array[t, i, j, 2]
                tau12 = stress_array[t, i, j, 3]
                tau23 = stress_array[t, i, j, 4]
                tau13 = stress_array[t, i, j, 5]

                if np.any(np.isnan([sig11, sig22, sig33, tau12, tau23, tau13])):
                    continue  # skip invalid tensor

                stress_tensor[t, i, j, ...] = np.array([
                    [sig11, tau12, tau13],
                    [tau12, sig22, tau23],
                    [tau13, tau23, sig33],
                ])

    return np.nan_to_num(stress_tensor, nan=0.0)
