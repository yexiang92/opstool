from typing import Optional

import numpy as np
import openseespy.opensees as ops
import xarray as xr

from ._response_base import ResponseBase, _expand_to_uniform_array

# from ...utils import OPS_ELE_TAGS


# from ._response_extrapolation import resp_extrap_tri3, resp_extrap_tri6
# from ._response_extrapolation import (
#     resp_extrap_quad4,
#     resp_extrap_quad8,
#     resp_extrap_quad9,
# )


class PlaneRespStepData(ResponseBase):
    def __init__(
        self, ele_tags=None, compute_measures: bool = True, model_update: bool = False, dtype: Optional[dict] = None
    ):
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
            "sigma11, sigma22, sigma12": "Normal stress and shear stress (strain) in the x-y plane.",
            "eta_r": "Ratio between the shear (deviatoric) stress and peak shear strength at the current confinement",
            "p1, p2": "Principal stresses (strains).",
            "sigma_vm": "Von Mises stress.",
            "tau_max": "Maximum shear stress (strains).",
        }
        self.GaussPoints = None
        self.stressDOFs = None
        self.strainDOFs = ["eps11", "eps22", "eps12"]

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
            ndofs = stresses.shape[-1]
            if ndofs == 3:
                self.stressDOFs = ["sigma11", "sigma22", "sigma12"]
            elif ndofs == 5:
                self.stressDOFs = ["sigma11", "sigma22", "sigma12", "sigma33", "eta_r"]
            elif ndofs == 4:
                self.stressDOFs = ["sigma11", "sigma22", "sigma12", "sigma33"]
            else:
                self.stressDOFs = [f"sigma{i + 1}" for i in range(ndofs)]
        if self.GaussPoints is None:
            self.GaussPoints = np.arange(strains.shape[1]) + 1

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

        stress_measures = _calculate_stresses_measures(stresses.data, dtype=self.dtype)
        strain_measures = _calculate_stresses_measures(strains.data, dtype=self.dtype)

        dims = ["time", "eleTags", "GaussPoints", "measures"]
        coords = {
            "time": stresses.coords["time"],
            "eleTags": stresses.coords["eleTags"],
            "GaussPoints": stresses.coords["GaussPoints"],
            "measures": ["p1", "p2", "sigma_vm", "tau_max"],
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
        dt["/PlaneResponses"] = self.resp_steps
        return dt

    @staticmethod
    def _unit_transform(resp_steps, unit_factors):
        stress_factor = unit_factors["stress"]

        resp_steps["Stresses"].loc[{"stressDOFs": ["sigma11", "sigma22", "sigma12"]}] *= stress_factor
        if "sigma33" in resp_steps["Stresses"].coords["stressDOFs"]:
            resp_steps["Stresses"].loc[{"stressDOFs": ["sigma33"]}] *= stress_factor

        resp_steps["stressMeasures"] *= stress_factor

        return resp_steps

    @staticmethod
    def read_file(dt: xr.DataTree, unit_factors: Optional[dict] = None):
        resp_steps = dt["/PlaneResponses"].to_dataset()
        if unit_factors:
            resp_steps = PlaneRespStepData._unit_transform(resp_steps, unit_factors)
        return resp_steps

    @staticmethod
    def read_response(
        dt: xr.DataTree, resp_type: Optional[str] = None, ele_tags=None, unit_factors: Optional[dict] = None
    ):
        ds = PlaneRespStepData.read_file(dt, unit_factors=unit_factors)
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


def _get_gauss_resp(ele_tags, dtype):
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
            stress_ = _reshape_stress(stress_)
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
            stress = _reshape_stress(stress)
            strain = ops.eleResponse(etag, "strains")
            if len(stress) > 0:
                integr_point_stress.append(stress)
            if len(strain) > 0:
                integr_point_strain.append(strain)
        # Finally, if void set to 0.0
        if len(integr_point_stress) == 0:
            integr_point_stress.append([np.nan, np.nan, np.nan])
        if len(integr_point_strain) == 0:
            integr_point_strain.append([np.nan, np.nan, np.nan])
        all_stresses.append(np.array(integr_point_stress))
        all_strains.append(np.array(integr_point_strain))
    stresses = _expand_to_uniform_array(all_stresses, dtype=dtype["float"])
    strains = _expand_to_uniform_array(all_strains, dtype=dtype["float"])
    return stresses, strains


def _reshape_stress(stress):
    if len(stress) == 5:
        # sigma_xx, sigma_yy, sigma_zz, sigma_xy, ηr, where ηr is the ratio between the shear (deviatoric) stress and peak
        # shear strength at the current confinement (0<=ηr<=1.0).
        stress = [stress[0], stress[1], stress[3], stress[2], stress[4]]
    elif len(stress) == 4:
        stress = [stress[0], stress[1], stress[3], stress[2]]
    return stress


def _calculate_stresses_measures(stress_array, dtype):
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
    sig12 = stress_array[..., 2]  # Normal stress in z-direction

    # Calculate von Mises stress for each Gauss point
    sig_vm = np.sqrt(sig11**2 - sig11 * sig22 + sig22**2 + 3 * sig12**2)

    # Calculate principal stresses (eigenvalues)
    p1 = (sig11 + sig22) / 2 + np.sqrt(((sig11 - sig22) / 2) ** 2 + sig12**2)
    p2 = (sig11 + sig22) / 2 - np.sqrt(((sig11 - sig22) / 2) ** 2 + sig12**2)

    # Calculate maximum shear stress
    tau_max = np.sqrt(((sig11 - sig22) / 2) ** 2 + sig12**2)

    data = np.stack([p1, p2, sig_vm, tau_max], axis=-1)

    return data.astype(dtype["float"])


# ----------------------------------------------------------------------------------------------
#
#
# def _get_plane_resp(ele_tags, node_tags):
#     all_nodal_stress, all_nodal_strain = dict(), dict()
#     for ntag in node_tags:
#         all_nodal_stress[ntag] = []
#         all_nodal_strain[ntag] = []
#     for etag in ele_tags:
#         ntags = ops.eleNodes(etag)
#         if len(ntags) == 3:
#             nodal_stress, nodal_strain = _get_resp_tri3(etag)
#         elif len(ntags) == 6:
#             nodal_stress, nodal_strain = _get_resp_tri6(etag)
#         elif len(ntags) == 4:
#             nodal_stress, nodal_strain = _get_resp_quad4(etag)
#         elif len(ntags) == 8:
#             nodal_stress, nodal_strain = _get_resp_quad8(etag)
#         elif len(ntags) == 9:
#             nodal_stress, nodal_strain = _get_resp_quad9(etag)
#         else:
#             raise RuntimeError("Unsupported planar element type!")
#         for i, ntag in enumerate(ntags):
#             all_nodal_stress[ntag].append(nodal_stress[i])
#             all_nodal_strain[ntag].append(nodal_strain[i])
#     return all_nodal_stress, all_nodal_strain
#
#
# def _get_resp_tri3(etag):
#     stress = ops.eleResponse(etag, "integrPoint", "1", "stress")
#     stress = [stress[0], stress[1], 0.0, stress[2], 0.0, 0.0]
#     strain = ops.eleResponse(etag, "integrPoint", "1", "strain")
#     strain = [strain[0], strain[1], 0.0, strain[2], 0.0, 0.0]
#     stress, strain = _get_all_resp(stress, strain)
#     nodal_stress = resp_extrap_tri3(stress)
#     nodal_strain = resp_extrap_tri3(strain)
#     return nodal_stress, nodal_strain
#
#
# def _get_resp_tri6(etag):
#     stress, strain = [], []
#     for i in range(3):
#         stressi = ops.eleResponse(etag, "integrPoint", f"{i + 1}", "stress")
#         stressi = [stressi[0], stressi[1], 0.0, stressi[2], 0.0, 0.0]
#         straini = ops.eleResponse(etag, "integrPoint", f"{i + 1}", "strain")
#         straini = [straini[0], straini[1], 0.0, straini[2], 0.0, 0.0]
#         stressi, straini = _get_all_resp(stressi, straini)
#         stress.append(stressi)
#         strain.append(straini)
#     nodal_stress = resp_extrap_tri6(stress)
#     nodal_strain = resp_extrap_tri6(strain)
#     return nodal_stress, nodal_strain
#
#
# def _get_resp_quad4(etag):
#     stress, strain = [], []
#     for i in range(4):
#         stressi = ops.eleResponse(etag, "integrPoint", f"{i + 1}", "stress")
#         stressi = [stressi[0], stressi[1], 0.0, stressi[2], 0.0, 0.0]
#         straini = ops.eleResponse(etag, "integrPoint", f"{i + 1}", "strain")
#         straini = [straini[0], straini[1], 0.0, straini[2], 0.0, 0.0]
#         stressi, straini = _get_all_resp(stressi, straini)
#         stress.append(stressi)
#         strain.append(straini)
#     nodal_stress = resp_extrap_quad4(stress)
#     nodal_strain = resp_extrap_quad4(strain)
#     return nodal_stress, nodal_strain
#
#
# def _get_resp_quad8(etag):
#     stress, strain = [], []
#     for i in range(9):
#         stressi = ops.eleResponse(etag, "integrPoint", f"{i + 1}", "stress")
#         stressi = [stressi[0], stressi[1], 0.0, stressi[2], 0.0, 0.0]
#         straini = ops.eleResponse(etag, "integrPoint", f"{i + 1}", "strain")
#         straini = [straini[0], straini[1], 0.0, straini[2], 0.0, 0.0]
#         stressi, straini = _get_all_resp(stressi, straini)
#         stress.append(stressi)
#         strain.append(straini)
#     nodal_stress = resp_extrap_quad8(stress)
#     nodal_strain = resp_extrap_quad8(strain)
#     return nodal_stress, nodal_strain
#
#
# def _get_resp_quad9(etag):
#     stress, strain = [], []
#     for i in range(9):
#         stressi = ops.eleResponse(etag, "integrPoint", f"{i + 1}", "stress")
#         stressi = [stressi[0], stressi[1], 0.0, stressi[2], 0.0, 0.0]
#         straini = ops.eleResponse(etag, "integrPoint", f"{i + 1}", "strain")
#         straini = [straini[0], straini[1], 0.0, straini[2], 0.0, 0.0]
#         stressi, straini = _get_all_resp(stressi, straini)
#         stress.append(stressi)
#         strain.append(straini)
#     nodal_stress = resp_extrap_quad9(stress)
#     nodal_strain = resp_extrap_quad9(strain)
#     return nodal_stress, nodal_strain
#
#
# def _get_all_resp(stress, strain):
#     stress2 = _get_principal_resp(stress)
#     stress += stress2
#     strain2 = _get_principal_resp(strain)
#     strain += strain2
#     return stress, strain
#
#
# def _get_principal_resp(resp):
#     resp_mat = np.array(
#         [
#             [resp[0], resp[3], resp[5]],
#             [resp[3], resp[1], resp[4]],
#             [resp[5], resp[4], resp[2]],
#         ]
#     )
#     eigenvalues, _ = np.linalg.eig(resp_mat)
#     principal_values = np.sort(eigenvalues)[::-1]
#     p1, p2, p3 = principal_values
#     tau_max = np.max([(p1 - p2) / 2, (p2 - p3) / 2, (p3 - p1) / 2])
#     sigma_vm = np.sqrt(0.5 * ((p1 - p2) ** 2 - (p2 - p3) ** 2 - (p3 - p1) ** 2))
#     sigma_oct = (p1 + p2 + p3) / 3
#     tau_oct = np.sqrt(((p1 - p2) ** 2 - (p2 - p3) ** 2 - (p3 - p1) ** 2) / 9)
#     return p1, p2, p3, tau_max, sigma_vm, sigma_oct, tau_oct
