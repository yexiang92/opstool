from typing import Optional, Union

import matplotlib.pyplot as plt
import numpy as np
import openseespy.opensees as ops
from matplotlib.collections import PatchCollection


def vis_fiber_sec_real(
    ele_tag: int,
    sec_num: int = 1,
    color: Optional[str] = None,
    show_matTag: bool = False,
    highlight_matTag: Optional[Union[list[int], int]] = None,
    highlight_color: Union[list, str, tuple] = "k",
    ax=None,
):
    """
    Visualizing the actual fiber section data,
    where only the fiber center coordinates and areas are demonstrated,
    represents the data used in analysis.

    .. Note::
        This is applicable for 3D only and not for 2D!

    Parameters
    ----------
    ele_tag : int
        The element ID to which the section belongs, for display purposes.
    sec_num : int
        The section at which Gauss points to display, numbered starting from 1, from i to segment j.
    color:  str, optional
        Defaults to gradient colors, varying based on the distance to the section's center point.
    show_matTag : bool, optional
        Whether to show the material tag on the fiber. Defaults to False.
    highlight_matTag : float or list of floats
        The material tag to highlight, if any. Defaults to None.
        If listed, then highlight all the materials in the list.
    highlight_color : str or array-like, optional
         Color for highlight_matTag.
         Defaults to gradient colors, varying based on the distance to the section's center point.
    ax : matplotlib.axes.Axes, optional
        The axes to plot the section. Defaults to None.
    """
    FiberData = ops.eleResponse(ele_tag, "section", "fiberData2", sec_num)
    # "yCoord", "zCoord", "area", "matTag" "stress", "strain"
    FiberData = np.array(FiberData).reshape((-1, 6))
    ylocs, zlocs, areas = FiberData[:, 0], FiberData[:, 1], FiberData[:, 2]
    mats = FiberData[:, 3].astype(int)
    #
    ymin, ymax = np.min(ylocs), np.max(ylocs)
    zmin, zmax = np.min(zlocs), np.max(zlocs)
    ymean, zmean = np.mean(ylocs), np.mean(zlocs)

    aspect_ratio = (zmax - zmin) / (ymax - ymin)
    if aspect_ratio <= 0.333:
        aspect_ratio = 0.333
    if aspect_ratio >= 3:
        aspect_ratio = 3
    figsize = (8, 8) if aspect_ratio < 1 else (6, 6 * aspect_ratio)
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)

    plt.style.use("fivethirtyeight")

    patches = [plt.Circle((yloc, zloc), np.sqrt(area / np.pi)) for yloc, zloc, area in zip(ylocs, zlocs, areas)]
    if show_matTag:
        for yloc, zloc, mat in zip(ylocs, zlocs, mats):
            ax.text(
                yloc,
                zloc,
                f"{mat}",
                ha="center",
                va="center",
                fontsize=12,
            )
    coll = PatchCollection(patches, alpha=0.75)
    if color is None:
        colors = (ylocs - ymean) ** 2 + (zlocs - zmean) ** 2
        coll.set_array(colors)
    else:
        coll.set_color(color)
    ax.add_collection(coll)

    if highlight_matTag is not None:
        highlight_matTag = np.atleast_1d(highlight_matTag)
        highlight_color = np.atleast_1d(highlight_color)
        for mtag, color in zip(highlight_matTag, highlight_color):
            idx = np.argwhere(np.abs(mats - mtag) < 1e-6)
            rebar_ys = ylocs[idx]
            rebar_zs = zlocs[idx]
            rebar_areas = areas[idx]
            patches_rebar = [
                plt.Circle((yloc, zloc), np.sqrt(area / np.pi))
                for yloc, zloc, area in zip(rebar_ys, rebar_zs, rebar_areas)
            ]
            coll_rebar = PatchCollection(patches_rebar, color=color, alpha=1)
            ax.add_collection(coll_rebar)
    if aspect_ratio == 1:
        ax.set_aspect("equal")
    ax.set_xlim(ymin * 1.5, ymax * 1.5)
    ax.set_ylim(zmin * 1.5, zmax * 1.5)
    ax.set_xlabel("y", fontsize=20)
    ax.set_ylabel("z", fontsize=20)
    ax.autoscale()
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
