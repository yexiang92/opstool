from __future__ import annotations

import os
from pathlib import Path
import shutil
import time
from typing import Literal

import numpy as np
import xarray as xr

from ..utils import CONFIGS, get_opensees_module, get_random_color
from ._post_utils import Beam3DDispInterpolator, generate_chunk_encoding_for_datatree
from .model_data import GetFEMData

ops = get_opensees_module()


def _get_modal_properties(mode_tag: int = 1, solver: str = "-genBandArpack"):
    """Get modal properties' data.

    Parameters
    ----------
    mode_tag : int, optional,
        Modal tag, all modal data smaller than this modal tag will be saved, by default 1
    solver : str, optional,
       OpenSees' eigenvalue analysis solver, by default "-genBandArpack".
       See `eigen Command <https://opensees.github.io/OpenSeesDocumentation/user/manual/analysis/eigen.html>`_

    Returns
    --------
    Data: xr.DataArray, modal properties' data.
    """
    ops.wipeAnalysis()
    if mode_tag == 1:
        ops.eigen(solver, 2)
    else:
        ops.eigen(solver, mode_tag)
    modal_props = ops.modalProperties("-return")
    # ------------------------------------------------------------------------
    attrs_names = ["domainSize", "totalMass", "totalFreeMass", "centerOfMass"]
    attrs = {name: tuple(modal_props[name]) for name in attrs_names}
    for key, value in attrs.items():
        if key == "domainSize":
            value = [int(v) for v in value]
        if len(value) == 1:
            value = value[0]
        attrs[key] = value
    # ------------------------------------------------------------------------
    column_names = [name for name in modal_props if name not in attrs_names]
    columns = [modal_props[name] for name in column_names]
    data = np.vstack(columns).transpose()[:mode_tag]
    data = xr.DataArray(
        data,
        coords={
            "modeTags": np.arange(1, mode_tag + 1),
            "Properties": column_names,
        },
        dims=("modeTags", "Properties"),
        attrs=attrs,
        name="ModalProps",
    )
    return data


def _get_eigen_vectors(mode_tag: int = 1, node_tags: list | None = None):
    eigenvectors = []
    for mode_tag_i in range(1, mode_tag + 1):
        eigen_vector = np.zeros((len(node_tags), 6))
        for i, Tag in enumerate(node_tags):
            coord = ops.nodeCoord(Tag)
            eigen = ops.nodeEigenvector(Tag, mode_tag_i)
            ndm, ndf = len(coord), len(eigen)
            if ndm == 1:
                eigen.extend([0, 0, 0, 0, 0])
            elif ndm == 2 and ndf == 3:
                eigen = [eigen[0], eigen[1], 0, 0, 0, eigen[2]]
            elif ndm == 2 and ndf == 2:
                eigen = [eigen[0], eigen[1], 0, 0, 0, 0]
            elif ndm == 2 and ndf == 1:
                eigen.extend([0, 0, 0, 0, 0])
            elif ndm == 3 and ndf == 3:
                eigen.extend([0, 0, 0])
            elif ndm == 3 and ndf in [4, 5]:
                eigen = [eigen[0], eigen[1], eigen[2], 0, 0, 0]
            elif ndm == 3 and ndf > 6:
                eigen = eigen[:6]
            eigen_vector[i] = eigen
        eigenvectors.append(eigen_vector)
    return np.array(eigenvectors)


def _get_eigen_info(
    mode_tag: int = 1,
    solver: str = "-genBandArpack",
):
    """Get modal properties' data.

    Parameters
    ----------
    mode_tag : int, optional,
        Modal tag, all modal data smaller than this modal tag will be saved, by default 1
    solver : str, optional,
       OpenSees' eigenvalue analysis solver, by default "-genBandArpack".
       See `eigen Command <https://opensees.github.io/OpenSeesDocumentation/user/manual/analysis/eigen.html>`_

    Returns
    --------
    modal_props: xr.DataArray
        Modal properties' data.
    eigenvectors: xr.DataArray
        Eigen vectors data.
    """
    modal_props = _get_modal_properties(mode_tag, solver)
    node_tags = ops.getNodeTags()
    eigenvectors = _get_eigen_vectors(mode_tag, node_tags)

    eigenvectors = xr.DataArray(
        eigenvectors,
        dims=["modeTags", "nodeTags", "DOFs"],
        coords={
            "modeTags": np.arange(1, mode_tag + 1),
            "nodeTags": node_tags,
            "DOFs": ["UX", "UY", "UZ", "RX", "RY", "RZ"],
        },
        name="EigenVectors",
    )
    return modal_props, eigenvectors


def _interpolator_eigenvectors(model_info: dict | None = None, eigen_vectors: xr.DataArray | None = None) -> xr.Dataset:
    node_coord = model_info["NodalData"].values
    node_tags = model_info["NodalData"]["nodeTags"].values
    axs, ays, azs, cells = [], [], [], []
    beam_data = model_info.get("BeamData", [])
    if len(beam_data) == 0:
        return xr.Dataset()
    link_data = model_info.get("LinkData", [])
    for data in [beam_data, link_data]:
        if len(data) > 0:
            cell = data.sel(info=["nodeI", "nodeJ"]).values
            ax = data.sel(info=["xaxis-x", "xaxis-y", "xaxis-z"]).values
            ay = data.sel(info=["yaxis-x", "yaxis-y", "yaxis-z"]).values
            az = data.sel(info=["zaxis-x", "zaxis-y", "zaxis-z"]).values
            cells.append(cell)
            axs.append(ax)
            ays.append(ay)
            azs.append(az)
    truss_data = model_info.get("TrussData", [])
    if len(truss_data) > 0:
        cell = truss_data.sel(cells=["nodeI", "nodeJ"]).values
        ax = np.zeros((cell.shape[0], 3))
        ay = np.zeros((cell.shape[0], 3))
        az = np.zeros((cell.shape[0], 3))
        cells.append(cell)
        axs.append(ax)
        ays.append(ay)
        azs.append(az)
    cells = np.vstack(cells)
    axs = np.vstack(axs)
    ays = np.vstack(ays)
    azs = np.vstack(azs)
    eigen_vec = eigen_vectors.sel(nodeTags=node_tags).values
    interp = Beam3DDispInterpolator(node_coord, cells, axs, ays, azs, one_based_node_id=False)
    local_eigen_vec = interp.global_to_local_ends(eigen_vec)
    points, response, cells = interp.interpolate(local_eigen_vec, npts_per_ele=11)
    ds = xr.Dataset(
        {
            "points": (("pointID", "coords"), points),
            "eigenVectors": (("modeTags", "pointID", "DOFs"), response),
            "cells": (("lines", "info"), cells),
        },
        coords={
            "modeTags": np.arange(1, len(eigen_vectors) + 1),
            "pointID": np.linspace(0, 1, points.shape[0]),
            "coords": ["x", "y", "z"],
            "DOFs": ["UX", "UY", "UZ"],
            "lines": np.arange(cells.shape[0]),
            "info": ["numNodes", "nodeI", "nodeJ"],
        },
    )
    return ds


def save_eigen_data(
    odb_tag: str | int = 1,
    mode_tag: int = 1,
    solver: Literal["-genBandArpack", "-fullGenLapack"] = "-genBandArpack",
    interpolate_beam: bool = True,
):
    """Save modal analysis data.

    Parameters
    ----------
    odb_tag: Union[str, int], default = 1
        Output database tag, the data will be saved in ``EigenData-{odb_tag}.nc``.
    mode_tag : int, optional,
        Modal tag, all modal data smaller than this modal tag will be saved, by default 1
    interpolate_beam : bool, optional,
        Whether to interpolate beam eigenvectors for better visualization, by default True
        If True, the eigenvectors of beam elements will be interpolated by shape functions,
        i.e., more points will be generated along the beam elements.
    solver : str, optional,
       OpenSees' eigenvalue analysis solver, by default "-genBandArpack".
       See `eigen Command <https://opensees.github.io/OpenSeesDocumentation/user/manual/analysis/eigen.html>`_
    """
    RESULTS_DIR = CONFIGS.get_output_dir()
    CONSOLE = CONFIGS.get_console()
    PKG_PREFIX = CONFIGS.get_pkg_prefix()
    EIGEN_FILE_NAME = CONFIGS.get_eigen_filename()
    odb_format, _ = CONFIGS.get_odb_format()

    output_filename = str(Path(RESULTS_DIR) / f"{EIGEN_FILE_NAME}-{odb_tag}.{odb_format}")
    # -----------------------------------------------------------------
    model_info, _ = GetFEMData().get_model_info()
    if model_info == {}:
        raise ValueError("No model data found, please check your model!")  # noqa: TRY003
    modal_props, eigen_vectors = _get_eigen_info(mode_tag, solver)
    eigen_data = {}
    for key in model_info:
        eigen_data[f"ModelInfo/{key}"] = xr.Dataset({key: model_info[key]})
    eigen_data["Eigen/ModalProps"] = xr.Dataset({modal_props.name: modal_props})
    eigen_data["Eigen/EigenVectors"] = xr.Dataset({eigen_vectors.name: eigen_vectors})
    # interpolated beam eigenvectors
    if interpolate_beam:
        interp_eigenvectors = _interpolator_eigenvectors(model_info, eigen_vectors)
        eigen_data["Eigen/InterpolatedEigenVectors"] = interp_eigenvectors
    else:
        eigen_data["Eigen/InterpolatedEigenVectors"] = xr.Dataset()
    dt = xr.DataTree.from_dict(eigen_data, name=f"{EIGEN_FILE_NAME}")

    max_retries = 5
    retry_delay = 1

    for attempt in range(max_retries + 1):
        try:
            # try to remove existing directory before writing
            if attempt > 0 and os.path.exists(output_filename):
                shutil.rmtree(output_filename)
                # Windows may take some time to release file locks
                time.sleep(0.5)
            if odb_format.lower() == "zarr":
                encoding = generate_chunk_encoding_for_datatree(dt, target_chunk_mb=10.0)
                dt.to_zarr(output_filename, mode="w", consolidated=True, encoding=encoding, zarr_format=2)
            else:
                dt.to_netcdf(output_filename, mode="w", engine="netcdf4")
            break

        except PermissionError:
            if attempt < max_retries:
                # Wait and retry
                time.sleep(retry_delay)
                retry_delay *= 1.5
            else:
                raise
    dt.close()
    del dt
    # -----------------------------------------------------------------
    color = get_random_color()
    CONSOLE.print(f"{PKG_PREFIX} Eigen data has been saved to [bold {color}]{output_filename}[/]!")


def load_eigen_data(
    odb_tag: str | int = 1,
    mode_tag: int = 1,
    solver: str = "-genBandArpack",
    resave: bool = True,
    interpolate_beam: bool = True,
):
    """Get the eigenvalue data from the saved file."""
    RESULTS_DIR = CONFIGS.get_output_dir()
    CONSOLE = CONFIGS.get_console()
    PKG_PREFIX = CONFIGS.get_pkg_prefix()
    EIGEN_FILE_NAME = CONFIGS.get_eigen_filename()
    odb_format, odb_engine = CONFIGS.get_odb_format()
    kargs = {"consolidated": False} if odb_format.lower() == "zarr" else {}

    filename = str(Path(RESULTS_DIR) / f"{EIGEN_FILE_NAME}-{odb_tag}.{odb_format}")
    if not os.path.exists(filename):
        resave = True
    if resave:
        save_eigen_data(odb_tag=odb_tag, mode_tag=mode_tag, solver=solver, interpolate_beam=interpolate_beam)
    else:
        color = get_random_color()
        CONSOLE.print(f"{PKG_PREFIX} Loading eigen data from [bold {color}]{filename}[/] ...")
    with xr.open_datatree(filename, engine=odb_engine, **kargs).load() as dt:
        model_info = {}
        for key, value in dt["ModelInfo"].items():
            model_info[key] = value[key]
        if model_info == {}:
            raise ValueError("No model data found, please check your model!")  # noqa: TRY003
        model_props = dt["Eigen/ModalProps"]["ModalProps"]
        eigen_vectors = dt["Eigen/EigenVectors"]["EigenVectors"]
        interp_eigenvectors = dt["Eigen/InterpolatedEigenVectors"]
        if len(interp_eigenvectors) == 0:
            interp_eigenvectors = None
        dt.close()
    return model_props, eigen_vectors, interp_eigenvectors, model_info


def get_eigen_data(odb_tag: str | int | None = None):
    """Get the eigenvalue data from the saved file.

    Parameters
    ----------
    odb_tag: Union[int, str], default: None
        Tag of output databases (ODB) have been saved.

    Returns
    --------
    modal_props: xr.DataArray
        Modal properties' data.
    eigenvectors: xr.DataArray
        Eigen vectors data.
    """
    model_props, eigen_vectors, _, _ = load_eigen_data(odb_tag, resave=False)
    return model_props, eigen_vectors
