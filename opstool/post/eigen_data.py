import os
from typing import Literal, Optional, Union

import numpy as np
import openseespy.opensees as ops
import xarray as xr

from ..utils import CONFIGS, get_random_color
from .model_data import GetFEMData


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
    eigenvectors = []
    node_tags = ops.getNodeTags()
    for mode_tag_i in range(1, mode_tag + 1):
        eigen_vector = np.zeros((len(node_tags), 6))
        for i, Tag in enumerate(node_tags):
            coord = ops.nodeCoord(Tag)
            eigen = ops.nodeEigenvector(Tag, mode_tag_i)
            if len(coord) == 1:
                coord.extend([0, 0])
                eigen.extend([0, 0, 0, 0, 0])
            elif len(coord) == 2:
                coord.extend([0])
                if len(eigen) == 3:
                    eigen = [eigen[0], eigen[1], 0, 0, 0, eigen[2]]
                elif len(eigen) == 2:
                    eigen = [eigen[0], eigen[1], 0, 0, 0, 0]
                elif len(eigen) == 1:
                    eigen.extend([0, 0, 0, 0, 0])
            else:
                if len(eigen) == 3:
                    eigen.extend([0, 0, 0])
                elif len(eigen) > 6:
                    eigen = eigen[:6]
            eigen_vector[i] = eigen
        eigenvectors.append(eigen_vector)
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


def save_eigen_data(
    odb_tag: Union[str, int] = 1,
    mode_tag: int = 1,
    solver: Literal["-genBandArpack", "-fullGenLapack"] = "-genBandArpack",
):
    """Save modal analysis data.

    Parameters
    ----------
    odb_tag: Union[str, int], default = 1
        Output database tag, the data will be saved in ``EigenData-{odb_tag}.nc``.
    mode_tag : int, optional,
        Modal tag, all modal data smaller than this modal tag will be saved, by default 1
    solver : str, optional,
       OpenSees' eigenvalue analysis solver, by default "-genBandArpack".
       See `eigen Command <https://opensees.github.io/OpenSeesDocumentation/user/manual/analysis/eigen.html>`_
    """
    RESULTS_DIR = CONFIGS.get_output_dir()
    CONSOLE = CONFIGS.get_console()
    PKG_PREFIX = CONFIGS.get_pkg_prefix()
    EIGEN_FILE_NAME = CONFIGS.get_eigen_filename()

    output_filename = RESULTS_DIR + "/" + f"{EIGEN_FILE_NAME}-{odb_tag}.nc"
    # -----------------------------------------------------------------
    model_info, _ = GetFEMData().get_model_info()
    modal_props, eigen_vectors = _get_eigen_info(mode_tag, solver)
    eigen_data = {}
    for key in model_info:
        eigen_data[f"ModelInfo/{key}"] = xr.Dataset({key: model_info[key]})
    eigen_data["Eigen/ModalProps"] = xr.Dataset({modal_props.name: modal_props})
    eigen_data["Eigen/EigenVectors"] = xr.Dataset({eigen_vectors.name: eigen_vectors})
    dt = xr.DataTree.from_dict(eigen_data, name=f"{EIGEN_FILE_NAME}")
    dt.to_netcdf(output_filename, mode="w", engine="netcdf4")
    # /////////////////////////////////////
    color = get_random_color()
    CONSOLE.print(f"{PKG_PREFIX} Eigen data has been saved to [bold {color}]{output_filename}[/]!")


def load_eigen_data(
    odb_tag: Union[str, int] = 1,
    mode_tag: int = 1,
    solver: str = "-genBandArpack",
    resave: bool = True,
):
    """Get the eigenvalue data from the saved file."""
    RESULTS_DIR = CONFIGS.get_output_dir()
    CONSOLE = CONFIGS.get_console()
    PKG_PREFIX = CONFIGS.get_pkg_prefix()
    EIGEN_FILE_NAME = CONFIGS.get_eigen_filename()

    filename = f"{RESULTS_DIR}/" + f"{EIGEN_FILE_NAME}-{odb_tag}.nc"
    if not os.path.exists(filename):
        resave = True
    if resave:
        save_eigen_data(odb_tag=odb_tag, mode_tag=mode_tag, solver=solver)
    else:
        color = get_random_color()
        CONSOLE.print(f"{PKG_PREFIX} Loading eigen data from [bold {color}]{filename}[/] ...")
    with xr.open_datatree(filename, engine="netcdf4").load() as dt:
        model_info = {}
        for key, value in dt["ModelInfo"].items():
            model_info[key] = value[key]
        model_props = dt["Eigen/ModalProps"]["ModalProps"]
        eigen_vectors = dt["Eigen/EigenVectors"]["EigenVectors"]
    return model_props, eigen_vectors, model_info


def get_eigen_data(
    odb_tag: Optional[Union[str, int]] = None,
):
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
    model_props, eigen_vectors, _ = load_eigen_data(odb_tag, resave=False)
    return model_props, eigen_vectors
