from typing import Optional, Union
from pathlib import Path
import numpy as np
import scipy.linalg as slin
import scipy.sparse as sp
import xarray as xr

from ..utils import CONFIGS, get_opensees_module, get_random_color
from ._post_utils import generate_chunk_encoding_for_datatree
from .model_data import GetFEMData

ops = get_opensees_module()


def _is_symmetric(a, tol=1e-10):
    return np.allclose(a, a.T, atol=tol)


def _get_dof_by_ntag(items, ntag):
    target = str(ntag)
    output = [item.split("-")[1] for item in items if item.split("-")[0] == target]
    return [int(i) for i in output] if output else []


def _get_linear_buckling_data(kmat: xr.DataArray, kgeo: xr.DataArray, n_modes=1):
    """Compute linear buckling eigenvalues and vectors."""
    symmetric = _is_symmetric(kmat) and _is_symmetric(kgeo)
    if symmetric:
        # use sparse solver for large systems
        kmat_sparse = sp.csr_matrix(np.array(kmat))
        kgeo_sparse = sp.csr_matrix(np.array(kgeo))
        k_request = n_modes * 2  # request more to account for possible complex pairs
        eigvals, eigvecs = sp.linalg.eigsh(
            A=kmat_sparse,
            k=k_request,
            M=kgeo_sparse,
            sigma=1e-3,
            which="LA",
            return_eigenvectors=True,
            maxiter=10000,
            tol=1e-12,
            mode="buckling",
        )
    else:
        # use dense solver for non-symmetric systems, may be slower for large systems
        eigvals, eigvecs = slin.eig(np.array(kmat), np.array(kgeo))

    # to real
    eigvals = np.real(eigvals)
    eigvecs = np.real(eigvecs)

    # filter positive eigenvalues
    mask = eigvals > 1e-10  # use a small positive threshold to avoid numerical errors
    eigvals = eigvals[mask]
    eigvecs = eigvecs[:, mask]

    # sort
    idx = np.argsort(eigvals)[:n_modes]
    eigvals = eigvals[idx]
    eigvecs = eigvecs[:, idx]

    # Node info
    node_tags = ops.getNodeTags()
    n_nodes = len(node_tags)
    node_tags_dofs = kmat.coords["nodeTagsDofs-1"].values

    # DOF position map
    dof_pos = {
        (1, 1): [0],
        (2, 2): [0, 1],
        (2, 3): [0, 1, 5],
        (3, 3): [0, 1, 2],
        (3, 6): [0, 1, 2, 3, 4, 5],
    }

    # Extract eigenvectors
    mode_vectors = []
    for i in range(len(eigvals)):
        vec = np.zeros((n_nodes, 6))
        for j, tag in enumerate(node_tags):
            ndm = ops.getNDM(tag)[0]
            ndf = ops.getNDF(tag)[0]
            pos = dof_pos.get((ndm, ndf), list(range(min(ndf, 6))))
            dofs = _get_dof_by_ntag(node_tags_dofs, tag)
            vec[j, pos] = eigvecs[dofs, i]
        mode_vectors.append(vec)

    # xarray outputs
    eigenvectors = xr.DataArray(
        np.stack(mode_vectors),
        dims=["modeTags", "nodeTags", "DOFs"],
        coords={
            "modeTags": np.arange(1, len(eigvals) + 1),
            "nodeTags": node_tags,
            "DOFs": ["UX", "UY", "UZ", "RX", "RY", "RZ"],
        },
        name="BucklingVectors",
    )
    eigenvalues = xr.DataArray(
        eigvals, dims=["modeTags"], coords={"modeTags": np.arange(1, len(eigvals) + 1)}, name="BucklingValues"
    )
    return eigenvalues, eigenvectors


def save_linear_buckling_data(
    kmat: xr.DataArray,
    kgeo: xr.DataArray,
    n_modes: int = 1,
    odb_tag: Union[str, int] = 1,
):
    """Save linear buckling analysis data. Added in v0.1.15.

    .. Note::
        * Currently you must use the matrix returned by :func:`opstool.pre.get_mck` to input `kmat` and `kgeo`.
        * Currently `scipy.sparse.linalg.eigsh <https://scipy.github.io/devdocs/reference/generated/scipy.sparse.linalg.eigsh.html#scipy.sparse.linalg.eigsh>`_ is used to solve the eigenvalue analysis for symmetric systems, which is more efficient for large models.
        * Currently `scipy.linalg.eig <https://docs.scipy.org/doc/scipy/reference/generated/scipy.linalg.eig.html>`_ is called to solve the eigenvalue analysis for non-symmetric systems, which can be slow for models with large degrees of freedom.

    Parameters
    ----------
    kmat : xr.DataArray
        Material stiffness matrix.
    kgeo : xr.DataArray
        Geometric stiffness matrix.
    n_modes : int, optional
        Number of modes to return, by default 1
    odb_tag: Union[str, int], default = 1
        Output database tag, the data will be saved in ``LinearBucklingData-{odb_tag}.nc``.
    """
    RESULTS_DIR = CONFIGS.get_output_dir()
    CONSOLE = CONFIGS.get_console()
    PKG_PREFIX = CONFIGS.get_pkg_prefix()
    BUCKLING_FILE_NAME = "LinearBucklingData"
    odb_format, _ = CONFIGS.get_odb_format()

    if not isinstance(kmat, xr.DataArray) or not isinstance(kgeo, xr.DataArray):
        raise TypeError("kmat and kgeo must be xarray.DataArray objects.")  # noqa: TRY003

    output_filename = str(Path(RESULTS_DIR) / f"{BUCKLING_FILE_NAME}-{odb_tag}.{odb_format}")
    # -----------------------------------------------------------------
    model_info, _ = GetFEMData().get_model_info()
    if model_info == {}:
        raise ValueError("No model data found, please check your model!")  # noqa: TRY003
    eigenvalues, eigenvectors = _get_linear_buckling_data(kmat=kmat, kgeo=kgeo, n_modes=n_modes)
    eigen_data = {}
    for key in model_info:
        eigen_data[f"ModelInfo/{key}"] = xr.Dataset({key: model_info[key]})
    eigen_data[f"LinearBuckling/{eigenvalues.name}"] = xr.Dataset({eigenvalues.name: eigenvalues})
    eigen_data[f"LinearBuckling/{eigenvectors.name}"] = xr.Dataset({eigenvectors.name: eigenvectors})
    dt = xr.DataTree.from_dict(eigen_data, name=BUCKLING_FILE_NAME)
    if odb_format.lower() == "zarr":
        encoding = generate_chunk_encoding_for_datatree(dt, target_chunk_mb=10.0)
        dt.to_zarr(output_filename, mode="w", consolidated=True, encoding=encoding, zarr_format=2)
    else:
        dt.to_netcdf(output_filename, mode="w", engine="netcdf4")
    # -----------------------------------------------------------------
    color = get_random_color()
    CONSOLE.print(f"{PKG_PREFIX} Linear Buckling data has been saved to [bold {color}]{output_filename}[/]!")


def load_linear_buckling_data(odb_tag: Union[str, int]):
    """Get the eigenvalue data from the saved file."""
    RESULTS_DIR = CONFIGS.get_output_dir()
    CONSOLE = CONFIGS.get_console()
    PKG_PREFIX = CONFIGS.get_pkg_prefix()
    BUCKLING_FILE_NAME = "LinearBucklingData"
    odb_format, odb_engine = CONFIGS.get_odb_format()
    kargs = {"consolidated": False} if odb_format.lower() == "zarr" else {}

    filename = str(Path(RESULTS_DIR) / f"{BUCKLING_FILE_NAME}-{odb_tag}.{odb_format}")
    color = get_random_color()
    CONSOLE.print(f"{PKG_PREFIX} Loading Linear Buckling data from [bold {color}]{filename}[/] ...")
    with xr.open_datatree(filename, engine=odb_engine, **kargs).load() as dt:
        model_info = {}
        for key, value in dt["ModelInfo"].items():
            model_info[key] = value[key]
        eigenvalues = dt["LinearBuckling/BucklingValues"]["BucklingValues"]
        eigenvectors = dt["LinearBuckling/BucklingVectors"]["BucklingVectors"]
        dt.close()
    return eigenvalues, eigenvectors, model_info


def get_linear_buckling_data(odb_tag: Optional[Union[str, int]] = None):
    """Get the linear buckling analysis data from the saved file. Added in v0.1.15.

    Parameters
    ----------
    odb_tag: Union[int, str], default: None
        Tag of output databases (ODB) have been saved.

    Returns
    --------
    BucklingValues: xr.DataArray
        Eigenvalues of the linear buckling analysis.
    BucklingVectors: xr.DataArray
        Eigenvectors of the linear buckling analysis.
    """
    eigenvalues, eigenvectors, _ = load_linear_buckling_data(odb_tag=odb_tag)
    return eigenvalues, eigenvectors
