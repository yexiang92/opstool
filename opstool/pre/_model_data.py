from typing import Union

import numpy as np
import openseespy.opensees as ops
import xarray as xr

from ..utils import suppress_ops_print


def get_node_mass() -> dict:
    """Get nodal mass data from the OpenSees model, including the mass from the nodes and elements.
    Added in v0.1.15.

    Returns:
    ---------
    node_mass : dict
        Dictionary with node tags as keys and mass values as lists.

    Examples:
    ---------
    >>> node_mass = get_node_mass()
    >>> print(node_mass)
    {1: [1.0, 1.0, 1.0], 2: [1.0, 1.0, 1.0], ...}
    """
    M = get_mck(
        "m", constraints_args=("Penalty", 0.0, 0.0), system_args=("Diagonal", "lumped"), numberer_args=("Plain",)
    ).to_numpy()
    node_mass = {}
    i = 0
    for ntag in ops.getNodeTags():
        ndofs = ops.getNDF(ntag)[0]
        mass = []
        for _ in range(ndofs):
            mass.append(M[i])
            i += 1
        node_mass[ntag] = mass
    return node_mass


def get_node_coord() -> xr.DataArray:
    """Get nodal data from the OpenSees model.

    Returns:
    ---------
    node_data : xarray.DataArray
        Nodal data array with coordinates and tags.

    Examples:
    ---------
    >>> node_coord = get_node_coord()
    >>> print(node_coord.coords)
    {'nodeTags': array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]), 'coords': ['Xloc', 'Yloc', 'Zloc']}
    """
    node_ndims, node_ndofs, node_coords = [], [], []
    node_tags = ops.getNodeTags()
    for _i, tag in enumerate(node_tags):
        coord = ops.nodeCoord(tag)
        ndim = ops.getNDM(tag)[0]
        ndof = ops.getNDF(tag)[0]
        if ndim == 1:
            coord.extend([0, 0])
        elif ndim == 2:
            coord.extend([0])
        node_ndims.append(ndim)
        node_ndofs.append(ndof)
        node_coords.append(coord)
    if len(node_coords) > 0:
        node_data = xr.DataArray(
            node_coords,
            coords={
                "nodeTags": node_tags,
                "coords": ["Xloc", "Yloc", "Zloc"],
            },
            dims=["nodeTags", "coords"],
            attrs={
                "numNodes": len(node_tags),
                "ndofs": tuple(node_ndofs),
                "ndims": tuple(node_ndims),
            },
        )
    else:
        node_data = xr.DataArray(node_coords)
    node_data.name = "NodalData"
    return node_data


def get_mck(
    matrix_type: str,
    constraints_args: Union[list, tuple],
    system_args: Union[list, tuple] = ("FullGeneral",),
    numberer_args: Union[list, tuple] = ("Plain",),
) -> xr.DataArray:
    """Get the mass, stiffness, or damping matrix from the OpenSees model.
    Added in v0.1.15.

    Parameters:
    -----------
    matrix_type : str
        Type of matrix to extract, must be one of 'm', 'k', 'ki', or 'c'.
    constraints_args : Union[list, tuple]
        Arguments for the constraints type, e.g., ("Penalty", 1e10, 1e10).
    system_args : Union[list, tuple], optional
        Arguments for the system type, default is ("FullGeneral",).
    numberer_args : Union[list, tuple], optional
        Arguments for numberer to use, default is ("Plain",).

    Returns:
    --------
    matrix : xarray.DataArray
        The extracted matrix, either mass, stiffness, or damping.
        The coordinates of the DataArray will be the node tags and degrees of freedom--"{ntag}-{dof}".

    Examples:
    ---------
    >>> K = get_mck("k", ["Penalty", 1e10, 1e10], ["FullGeneral"], ["Plain"])
    >>> print(K.to_numpy())
    [[1.0 0.0 0.0]
     [0.0 1.0 0.0]
     [0.0 0.0 1.0]]
    """
    with suppress_ops_print():
        return _get_mck(
            matrix_type=matrix_type,
            constraints_args=constraints_args,
            system_args=system_args,
            numberer_args=numberer_args,
        )


def _get_mck(
    matrix_type: str,
    constraints_args: Union[list, tuple],
    system_args: Union[list, tuple] = ("FullGeneral",),
    numberer_args: Union[list, tuple] = ("Plain",),
):
    ops.wipeAnalysis()  # Clear any previous analysis
    ops.numberer(*numberer_args)  # Set the numberer type, must be "Plain" for mass matrix extraction
    ops.constraints(*constraints_args)
    ops.system(*system_args)  # Set the system type, must be "Diagonal" for mass matrix extraction
    ops.algorithm("Linear")  # Use linear algorithm
    ops.test("NormDispIncr", 1, 10, 0)  # Set convergence test
    ops.analysis("Transient", "-noWarnings")
    if matrix_type.lower() == "m":
        ops.integrator("GimmeMCK", 1.0, 0.0, 0.0, 0.0)
    elif matrix_type.lower() == "c":
        ops.integrator("GimmeMCK", 0.0, 1.0, 0.0, 0.0)
    elif matrix_type.lower() == "k":
        ops.integrator("GimmeMCK", 0.0, 0.0, 1.0, 0.0)
    elif matrix_type.lower() == "ki":
        ops.integrator("GimmeMCK", 0.0, 0.0, 0.0, 1.0)
    else:
        raise ValueError("matrix_type must be 'm', 'k', 'ki', or 'c'.")  # noqa: TRY003
    ops.analyze(1, 0.0)

    matrix = np.array(ops.printA("-ret"))
    n = ops.systemSize()
    matrix = np.reshape(matrix, (n, n)) if len(matrix) == n**2 else matrix

    # nodedofs
    coords = [None] * n
    nodetags = ops.getNodeTags()
    for ntag in nodetags:
        dofs = ops.nodeDOFs(ntag)
        for dof in dofs:
            if dof >= 0:
                coords[dof] = f"{ntag}-{dof}"
    if matrix.ndim == 2:
        matrix = xr.DataArray(
            matrix,
            coords={"nodeTagsDofs-1": coords, "nodeTagsDofs-2": coords},
            dims=["nodeTagsDofs-1", "nodeTagsDofs-2"],
        )
    else:
        matrix = xr.DataArray(
            matrix,
            coords={"nodeTagsDofs": coords},
            dims=["nodeTagsDofs"],
        )

    ops.wipeAnalysis()
    return matrix
