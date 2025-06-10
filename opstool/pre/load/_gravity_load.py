from typing import Optional

import openseespy.opensees as ops

from .._model_data import get_node_mass


def create_gravity_load(
    exclude_nodes: Optional[list] = None,
    direction: str = "Z",
    factor: float = -9.81,
) -> dict[int, list[float]]:
    """Applying the gravity loads.
    The mass values are derived from the extracted mass matrix, including the masses from the nodes and elements.
    See the blog `Do It Your Self-Weight <https://portwooddigital.com/2023/11/05/do-it-your-self-weight/>`_ for more details.

    .. Note::
        * This function is a modification of the method described in the previous blog post, which relaxes the restrictions on imposing constraints. This means you can impose constraints at any position.
        * This function invokes the OpenSees ``load`` command, which applies the loads to the nodes in the model, and the function does not check if the ``timeSeries`` and ``pattern`` are defined, so make sure to define them before calling this function.
        * The mass values are derived from the mass matrix, which is obtained from the :func:`opstool.pre.get_node_mass`. The function assumes that the mass matrix is defined and available in the model.

    Parameters
    -----------
    exclude_nodes: list, default=None
        Excluded node tags, whose masses will not be used to generate gravity loads.
    direction: str, default="Z"
        The gravity load direction.
    factor: float, default=-9.81
        The factor applied to the mass values, it should be the multiplication of gravitational acceleration
        and directional indicators, e.g., -9.81, where 9.81 is the gravitational acceleration
        and -1 indicates along the negative Z axis.
        Of course, it can be multiplied by an additional factor to account for additional constant loads,
        e.g., 1.05 * (-9.81).

    Returns
    --------
    node_loads : dict[int, list[float]]
        A dictionary where keys are node tags and values are the gravity loads applied to those nodes.
        The loads are in the form of a list, with the load in the specified direction and zeros in other directions.

    Examples
    ---------
    >>> ops.timeSeries("Constant", 1)  # Define a constant time series
    >>> ops.pattern("Plain", 1, 1)  # Define a load pattern
    >>> node_loads = create_gravity_load(direction='Z', factor=-9.81)
    """
    direction = direction.upper()
    if direction not in ["X", "Y", "Z"]:
        raise ValueError(f"Invalid direction {direction}. Must be one of 'X', 'Y', or 'Z'.")  # noqa: TRY003
    elif direction == "X":
        gravDOF = 0
    elif direction == "Y":
        gravDOF = 1
    elif direction == "Z":
        gravDOF = 2
    node_tags = ops.getNodeTags()
    if exclude_nodes is not None:
        node_tags = [tag for tag in node_tags if tag not in exclude_nodes]

    node_mass = get_node_mass()
    node_loads = {}
    for ntag in node_tags:
        if ntag in node_mass:
            mass = node_mass[ntag]
            gravload = [0.0] * len(mass)
            p = factor * mass[gravDOF]
            if p != 0.0:
                gravload[gravDOF] = p
                ops.load(ntag, *gravload)
                node_loads[ntag] = gravload
    return node_loads


gen_grav_load = create_gravity_load
