from collections import defaultdict
from typing import Optional

import numpy as np
import openseespy.opensees as ops
import xarray as xr

from ...utils import get_gp2node_func
from ._response_base import ResponseBase, _expand_to_uniform_array


class PlaneRespStepData(ResponseBase):
    def __init__(
        self,
        ele_tags=None,
        compute_measures: bool = True,
        compute_nodal_resp: Optional[str] = None,
        model_update: bool = False,
        dtype: Optional[dict] = None,
    ):
        self.resp_names = [
            "Stresses",
            "Strains",
            "StressesAtNodes",
            "StressAtNodesErr",
            "StrainsAtNodes",
            "StrainsAtNodesErr",
        ]
        self.resp_steps = None
        self.resp_steps_list = []  # for model update
        self.resp_steps_dict = {}  # for non-update
        self.step_track = 0
        self.ele_tags = ele_tags
        self.times = []

        self.node_tags = None
        self.compute_measures = compute_measures
        self.compute_nodal_resp = compute_nodal_resp
        self.nodal_resp_method = compute_nodal_resp
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

        if self.compute_nodal_resp:
            node_stress_avg, node_stress_rel_error, node_tags = _get_nodal_resp(
                ele_tags, stresses, method=self.nodal_resp_method, dtype=self.dtype
            )
            node_strain_avg, node_strain_rel_error, node_tags = _get_nodal_resp(
                ele_tags, strains, method=self.nodal_resp_method, dtype=self.dtype
            )
            self.node_tags = node_tags

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
            coords = {
                "eleTags": ele_tags,
                "GaussPoints": self.GaussPoints,
                "stressDOFs": self.stressDOFs,
                "strainDOFs": self.strainDOFs,
            }
            if self.compute_nodal_resp:
                data_vars["StressesAtNodes"] = (["nodeTags", "stressDOFs"], node_stress_avg)
                data_vars["StrainsAtNodes"] = (["nodeTags", "strainDOFs"], node_strain_avg)
                data_vars["StressAtNodesErr"] = (["nodeTags", "stressDOFs"], node_stress_rel_error)
                data_vars["StrainsAtNodesErr"] = (["nodeTags", "strainDOFs"], node_strain_rel_error)
                coords["nodeTags"] = node_tags

            ds = xr.Dataset(data_vars=data_vars, coords=coords, attrs=self.attrs)
            self.resp_steps_list.append(ds)
        else:
            self.resp_steps_dict["Stresses"].append(stresses)
            self.resp_steps_dict["Strains"].append(strains)
            if self.compute_nodal_resp:
                self.resp_steps_dict["StressesAtNodes"].append(node_stress_avg)
                self.resp_steps_dict["StrainsAtNodes"].append(node_strain_avg)
                self.resp_steps_dict["StressAtNodesErr"].append(node_stress_rel_error)
                self.resp_steps_dict["StrainsAtNodesErr"].append(node_strain_rel_error)

        self.times.append(ops.getTime())
        self.step_track += 1

    def _to_xarray(self):
        self.times = np.array(self.times, dtype=self.dtype["float"])
        if self.model_update:
            self.resp_steps = xr.concat(self.resp_steps_list, dim="time", join="outer")
            self.resp_steps.coords["time"] = self.times
        else:
            data_vars = {}
            data_vars["Stresses"] = (["time", "eleTags", "GaussPoints", "stressDOFs"], self.resp_steps_dict["Stresses"])
            data_vars["Strains"] = (["time", "eleTags", "GaussPoints", "strainDOFs"], self.resp_steps_dict["Strains"])
            coords = {
                "time": self.times,
                "eleTags": self.ele_tags,
                "GaussPoints": self.GaussPoints,
                "stressDOFs": self.stressDOFs,
                "strainDOFs": self.strainDOFs,
            }
            if self.compute_nodal_resp:
                data_vars["StressesAtNodes"] = (
                    ["time", "nodeTags", "stressDOFs"],
                    self.resp_steps_dict["StressesAtNodes"],
                )
                data_vars["StrainsAtNodes"] = (
                    ["time", "nodeTags", "strainDOFs"],
                    self.resp_steps_dict["StrainsAtNodes"],
                )
                data_vars["StressAtNodesErr"] = (
                    ["time", "nodeTags", "stressDOFs"],
                    self.resp_steps_dict["StressAtNodesErr"],
                )
                data_vars["StrainsAtNodesErr"] = (
                    ["time", "nodeTags", "strainDOFs"],
                    self.resp_steps_dict["StrainsAtNodesErr"],
                )
                coords["nodeTags"] = self.node_tags
            self.resp_steps = xr.Dataset(data_vars=data_vars, coords=coords, attrs=self.attrs)

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

        self.resp_steps["StressMeasures"] = xr.DataArray(
            stress_measures,
            dims=dims,
            coords=coords,
            name="StressMeasures",
        )
        self.resp_steps["StrainMeasures"] = xr.DataArray(
            strain_measures,
            dims=dims,
            coords=coords,
            name="StrainMeasures",
        )

        if self.compute_nodal_resp:
            node_stress_measures = _calculate_stresses_measures(
                self.resp_steps["StressesAtNodes"].data, dtype=self.dtype
            )
            node_strain_measures = _calculate_stresses_measures(
                self.resp_steps["StrainsAtNodes"].data, dtype=self.dtype
            )
            dims = ["time", "nodeTags", "measures"]
            coords = {
                "time": stresses.coords["time"],
                "nodeTags": self.resp_steps["StressesAtNodes"].coords["nodeTags"],
                "measures": ["p1", "p2", "sigma_vm", "tau_max"],
            }
            self.resp_steps["StressMeasuresAtNodes"] = xr.DataArray(
                node_stress_measures, dims=dims, coords=coords, name="StressMeasuresAtNodes"
            )
            self.resp_steps["StrainMeasuresAtNodes"] = xr.DataArray(
                node_strain_measures, dims=dims, coords=coords, name="StrainMeasuresAtNodes"
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
        if "StressMeasures" in resp_steps.data_vars:
            resp_steps["StressMeasures"] *= stress_factor
        if "StressMeasuresAtNodes" in resp_steps.data_vars:
            resp_steps["StressMeasuresAtNodes"] *= stress_factor

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


gp2node_type = {3: "tri", 6: "tri", 4: "quad", 8: "quad", 9: "quad"}


# Get nodal stresses and strains from the Gauss points of elements.
def _get_nodal_resp(ele_tags, ele_gp_resp, method, dtype):
    node_resp = defaultdict(list)
    for etag, gp_resp in zip(ele_tags, ele_gp_resp):
        etag = int(etag)
        ntags = ops.eleNodes(etag)
        gp_resp = gp_resp[~np.isnan(gp_resp).any(axis=1)]
        if len(gp_resp) == 0:
            continue
        gp2node_func = get_gp2node_func(ele_type=gp2node_type[len(ntags)], n=len(ntags), gp=len(gp_resp))
        if gp2node_func:
            resp = gp2node_func(method=method, gp_resp=gp_resp)
        else:
            resp = np.zeros((len(ntags), gp_resp.shape[1]), dtype=dtype["float"])
        for i, ntag in enumerate(ntags):
            node_resp[ntag].append(resp[i])
    # node_resp = dict(sorted(node_resp.items()))
    node_avg = {}
    # node_max = {}
    # node_min = {}
    node_ptp = {}  # Peak-to-peak: max - min
    # node_std = {}
    node_rel_error = {}

    for nid, vals in node_resp.items():
        arr = np.stack(vals, axis=0)  # shape: (k, m), k=num_samples, m=DOFs
        node_avg[nid] = np.nanmean(arr, axis=0)  # mean value
        # node_max[nid] = np.nanmax(arr, axis=0)  # maximum value
        # node_min[nid] = np.nanmin(arr, axis=0)  # minimum value
        node_ptp[nid] = np.nanmax(arr, axis=0) - np.nanmin(arr, axis=0)
        # node_std[nid] = np.nanstd(arr, axis=0)  # standard deviation

        node_rel_error[nid] = node_ptp[nid] / (np.abs(node_avg[nid]) + 1e-8)  # avoid division by zero
        node_rel_error[nid][np.abs(node_avg[nid]) < 1e-8] = 0.0  # if avg is close to zero, set rel error to zero
    node_avg = np.array(list(node_avg.values()), dtype=dtype["float"])
    node_rel_error = np.array(list(node_rel_error.values()), dtype=dtype["float"])
    node_tags = list(node_resp.keys())
    return node_avg, node_rel_error, node_tags


#  Get Gauss point stresses and strains for all elements in the model.
def _get_gauss_resp(ele_tags, dtype):
    """Collect Gauss point stresses and strains for each element."""
    all_stresses, all_strains = [], []

    for etag in map(int, ele_tags):
        stresses, strains = _collect_element_responses(etag)
        stresses, strains = _reorder_by_element_type(etag, stresses, strains)
        all_stresses.append(np.array(stresses))
        all_strains.append(np.array(strains))

    return (
        _expand_to_uniform_array(all_stresses, dtype=dtype["float"]),
        _expand_to_uniform_array(all_strains, dtype=dtype["float"]),
    )


def _collect_element_responses(etag):
    stresses, strains = [], []
    for i in range(100000000):
        s = _try_fetch(etag, i + 1, "stresses")
        e = _try_fetch(etag, i + 1, "strains")
        if not s or not e:
            break
        stresses.append(_reshape_stress(s))
        strains.append(e)

    if not stresses and not strains:
        s = _reshape_stress(ops.eleResponse(etag, "stresses"))
        e = ops.eleResponse(etag, "strains")
        if s:
            stresses.append(s)
        if e:
            strains.append(e)

    if not stresses:
        stresses.append([np.nan, np.nan, np.nan])
    if not strains:
        strains.append([np.nan, np.nan, np.nan])
    return stresses, strains


def _try_fetch(etag, idx, key):
    """Try multiple ops.eleResponse paths to fetch value."""
    for prefix in ["material", "integrPoint"]:
        val = ops.eleResponse(etag, prefix, str(idx), key)
        if val:
            return val
    return []


def _reorder_by_element_type(etag, stress, strain):
    ele_class_tag = ops.getEleClassTags(etag)[0]
    if ele_class_tag == 209 and len(stress) == 3:  # SixNodeTri, 3 gps
        idx = [2, 0, 1]
    elif ele_class_tag == 61 and len(stress) == 9:  # NineNodeMixedQuad, 9 gps
        idx = [0, 6, 8, 2, 3, 7, 5, 1, 4]
    else:
        return stress, strain
    return [stress[i] for i in idx], [strain[i] for i in idx]


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
