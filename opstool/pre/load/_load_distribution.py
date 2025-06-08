from typing import Optional, Union

import matplotlib.pyplot as plt
import openseespy.opensees as ops


def apply_load_distribution(
    node_tags: Optional[Union[list, tuple, int]] = None,
    coord_axis: str = "z",
    load_axis: str = "x",
    dist_type: str = "triangle",
    sum_normalized: bool = True,
    plot: bool = False,
):
    """
    Apply load distribution along specified coordinate axis.

    .. Note::
        The load is applied to the ``OpenSeesPy`` domain.
        If sum_normalized=True, The sum of the loads for all nodes is 1.0.
        If sum_normalized=False, The maximum load is set to 1.0.

    Parameters:
    -----------
    node_tags : list, tuple, int, optional
        The node tags where the load will be applied.
        If None, the function will apply the load to all nodes.
    coord_axis : str, default='z'
        The coordinate axis along which the load is distributed ('x', 'y', or 'z').
    load_axis : str, default='x'
        The load direction ('x', 'y', or 'z').
    dist_type : str, default='triangle'
        Type of distribution ('triangle', "parabola", 'half_parabola_concave', 'half_parabola_convex', 'uniform').
    sum_normalized : bool, default=True
        If True, the loads are normalized to ensure their sum is 1.0.
        If False, the maximum load is set to 1.0.
    plot : bool, optional
        If True, plots the load distribution graph.

    Returns:
    --------
    node_loads : dict
        A dictionary containing the node tags and the corresponding normalized loads.
    """
    if isinstance(node_tags, int):
        node_tags = [node_tags]
    elif node_tags is None:
        node_tags = ops.getNodeTags()

    axis_idx = {"x": 0, "y": 1, "z": 2}[coord_axis.lower()]
    load_idx = {"x": 0, "y": 1, "z": 2}[load_axis.lower()]

    coords = {n: ops.nodeCoord(n)[axis_idx] for n in node_tags}
    sorted_items = sorted(coords.items(), key=lambda x: x[1])
    min_coord = sorted_items[0][1]
    rel_coords = [v - min_coord for _, v in sorted_items]
    max_coord = max(rel_coords) or 1.0  # avoid div-zero
    tags = [n for n, _ in sorted_items]

    dist_func = {
        "triangle": lambda x: x / max_coord,
        "parabola": lambda x: (x / max_coord) * (1 - x / max_coord),
        "half_parabola_concave": lambda x: 4 * (x / max_coord) ** 2,
        "half_parabola_convex": lambda x: -1 / max_coord**2 * (x - max_coord) ** 2 + 1,
        "uniform": lambda x: 1.0,
    }.get(dist_type.lower())

    if not dist_func:
        raise ValueError(f"Unsupported dist_type '{dist_type}'")  # noqa: TRY003

    raw_loads = [dist_func(x) for x in rel_coords]
    norm = sum(raw_loads) if sum_normalized else max(raw_loads)
    norm_loads = [v / norm for v in raw_loads]

    for tag, load in zip(tags, norm_loads):
        ndf = ops.getNDF(tag)[0]
        load_vec = [0.0] * ndf
        load_vec[load_idx] = load
        ops.load(tag, *load_vec)

    if plot:
        max_load = max(norm_loads)
        aspect_ratio = max_load / max_coord
        _plot_distribution(norm_loads, [c + min_coord for c in rel_coords], coord_axis, dist_type, aspect_ratio)

    return dict(zip(tags, norm_loads))


def _plot_distribution(loads, coords, coord_axis, dist_type, aspect_ratio=1.0):
    """Helper function to plot load distribution."""
    plt.figure(figsize=(8, 5))
    plt.plot(loads, coords, color="#c0737a", label="Load Distribution")
    for load, c in zip(loads, coords):
        plt.plot([0, load], [c, c], color="#c0737a", alpha=0.6)
    plt.scatter([0] * len(coords), coords, color="#2c6fbb", zorder=5)
    plt.xlabel("Normalized Load")
    plt.ylabel(f"{coord_axis}-coordinate")
    plt.grid(True)
    plt.title(f"{dist_type.capitalize()} Distribution")
    plt.tight_layout()
    plt.gca().set_aspect(aspect_ratio * 5)
    plt.show()
