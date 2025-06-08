import copy
from collections import defaultdict
from functools import wraps
from typing import ClassVar, Optional, Union

import matplotlib.patches as mpathes
import matplotlib.pyplot as plt
import numpy as np
import openseespy.opensees as ops
from matplotlib.collections import LineCollection

from ...utils import get_cycle_color

COLORS = get_cycle_color()

plt.rcParams["xtick.direction"] = "in"
plt.rcParams["ytick.direction"] = "in"


class FiberSecPlot:
    patches: ClassVar = defaultdict(lambda: [])
    line2Ds: ClassVar = defaultdict(lambda: [])
    sec_tag = 1

    @classmethod
    def set_sec_tag(cls, tag):
        cls.sec_tag = tag

    @classmethod
    def get_sec_tag(cls):
        return cls.sec_tag

    @classmethod
    def add_fiber_points(cls, args: Union[list, tuple], color: Union[str, tuple, list], alpha: float):
        y, z, area = args[0], args[1], args[2]
        r = np.sqrt(area / np.pi)
        circle = mpathes.Circle((y, z), r, color=color, alpha=alpha)
        cls.patches[cls.sec_tag].append(circle)

    @classmethod
    def add_rect(cls, args: Union[list, tuple], color: Union[str, tuple, list], alpha: float):
        numy, numz = args[1], args[2]
        yi, zi, yj, zj = args[3:]
        rect = mpathes.Rectangle((yi, zi), yj - yi, zj - zi, color=color, alpha=alpha)
        cls.patches[cls.sec_tag].append(rect)

        delta_y = np.abs(yj - yi) / numy
        delta_z = np.abs(zj - zi) / numz
        yg = np.arange(yi, yj + 0.1 * delta_y, delta_y)
        zg = np.arange(zi, zj + 0.1 * delta_z, delta_z)
        for y_ in yg:
            cls.line2Ds[cls.sec_tag].append([(y_, zi), (y_, zj)])
        for z_ in zg:
            cls.line2Ds[cls.sec_tag].append([(yi, z_), (yj, z_)])

    @classmethod
    def add_quad(cls, args: Union[list, tuple], color: Union[str, tuple, list], alpha: float):
        numij, numjk = args[1], args[2]
        yi, zi, yj, zj, yk, zk, yl, zl = args[3:]
        yz = np.array([[yi, zi], [yj, zj], [yk, zk], [yl, zl]])
        yzij = _line_mesh(yi, zi, yj, zj, numij)
        yzjk = _line_mesh(yj, zj, yk, zk, numjk)
        yzkl = _line_mesh(yk, zk, yl, zl, numij)
        yzli = _line_mesh(yl, zl, yi, zi, numjk)

        polygon = mpathes.Polygon(yz, color=color, alpha=alpha)
        cls.patches[cls.sec_tag].append(polygon)

        for i in range(len(yzij)):
            yz1 = yzij[i]
            yz2 = yzkl[::-1][i]
            cls.line2Ds[cls.sec_tag].append([(yz1[0], yz1[1]), (yz2[0], yz2[1])])
        for i in range(len(yzjk)):
            yz1 = yzjk[i]
            yz2 = yzli[::-1][i]
            cls.line2Ds[cls.sec_tag].append([(yz1[0], yz1[1]), (yz2[0], yz2[1])])

    @classmethod
    def add_circ(cls, args: Union[list, tuple], color: Union[str, tuple, list], alpha: float):
        num_circ, num_rad = args[1], args[2]
        yc, zc = args[3], args[4]
        r_in, r_ex = args[5], args[6]
        if len(args) > 7:
            ang_s, ang_e = args[7], args[8]
        else:
            ang_s, ang_e = 0.0, 360
        node_ex = _arc_mesh(ang_s, ang_e, r_ex, num_circ, yc, zc)
        node_in = _arc_mesh(ang_s, ang_e, r_in, num_circ, yc, zc)

        wedge = mpathes.Wedge((yc, zc), r_ex, ang_s, ang_e, width=r_ex - r_in, color=color, alpha=alpha)
        cls.patches[cls.sec_tag].append(wedge)

        for i in range(num_circ + 1):
            yz_ex = node_ex[i]
            yz_in = node_in[i]
            cls.line2Ds[cls.sec_tag].append([(yz_in[0], yz_in[1]), (yz_ex[0], yz_ex[1])])

        for i in range(num_rad + 1):
            delta_r = (r_ex - r_in) / num_rad
            arc = mpathes.Arc(
                (yc, zc),
                2 * (r_in + i * delta_r),
                2 * (r_in + i * delta_r),
                theta1=ang_s,
                theta2=ang_e,
                lw=0.5,
                color="k",
                alpha=alpha,
            )
            cls.patches[cls.sec_tag].append(arc)

    @classmethod
    def add_straight_layer(cls, args: Union[list, tuple], color: Union[str, tuple, list], alpha: float):
        num, area = args[1], args[2]
        r = np.sqrt(area / np.pi)
        yi, zi, yj, zj = args[3:]
        node = _line_mesh(yi, zi, yj, zj, num - 1)
        for i in range(num):
            circle = mpathes.Circle(node[i], r, color=color, alpha=alpha)
            cls.patches[cls.sec_tag].append(circle)

    @classmethod
    def add_circ_layer(cls, args: Union[list, tuple], color: Union[str, tuple, list], alpha: float):
        num, area = args[1], args[2]
        yc, zc, r = args[3], args[4], args[5]
        if len(args) > 6:
            ang_s, ang_e = args[6], args[7]
        else:
            ang_s, ang_e = 0.0, 360.0 - 360 / num
        node = _arc_mesh(ang_s, ang_e, r, num - 1, yc, zc)
        for i in range(num):
            circle = mpathes.Circle(node[i], np.sqrt(area / np.pi), color=color, alpha=alpha)
            cls.patches[cls.sec_tag].append(circle)

    @classmethod
    def plot_sec(
        cls,
        sec_tag: int,
        title: str = "My Section",
        label_size: int = 15,
        tick_size: int = 12,
        title_size: int = 18,
    ):
        _, ax = plt.subplots()
        # patches
        for pat in cls.patches[sec_tag]:
            ax.add_patch(copy.copy(pat))
        # mesh lines
        lc = LineCollection(cls.line2Ds[sec_tag], linewidths=0.5, colors="k")
        ax.add_collection(lc)
        # ----------------------------------------
        ax.set_xlabel("y", fontsize=label_size)
        ax.set_ylabel("z", fontsize=label_size)
        ax.tick_params(axis="both", which="major", width=1.2, length=5, labelsize=tick_size)
        ax.tick_params(axis="both", which="minor", width=0.8, length=3)
        ax.set_title(title, fontsize=title_size)
        ax.grid(False)
        ax.autoscale()
        plt.minorticks_on()
        for loc in ["bottom", "top", "left", "right"]:
            ax.spines[loc].set_linewidth(1.0)
        # plt.gcf().subplots_adjust(bottom=0.15)
        plt.axis("equal")
        plt.tight_layout()
        plt.show()


def fetch_fiber_plot_data(func):  # noqa: C901
    """Fetch data from fiber creation commands for visualization; commands include
    'fiber', 'patch', 'layer'. Note that this is a decorator.
    """
    fiber_type = func.__name__

    @wraps(func)
    def wrapper(*args, color: Optional[Union[str, tuple, list]] = None, opacity: float = 1.0):
        if color is None:
            color = next(COLORS)
        if fiber_type == "fiber":
            FiberSecPlot.add_fiber_points(args, color=color, alpha=opacity)
        elif fiber_type == "patch":
            if args[0] in ["quad", "quadr", "quadrilateral"]:
                FiberSecPlot.add_quad(args[1:], color=color, alpha=opacity)
            elif args[0] in ["rect", "rectangular"]:
                FiberSecPlot.add_rect(args[1:], color=color, alpha=opacity)
            elif args[0] in ["circ", "circular"]:
                FiberSecPlot.add_circ(args[1:], color=color, alpha=opacity)
        elif fiber_type == "layer":
            if args[0] == "straight":
                FiberSecPlot.add_straight_layer(args[1:], color=color, alpha=opacity)
            elif args[0] in ["circ", "circular"]:
                FiberSecPlot.add_circ_layer(args[1:], color=color, alpha=opacity)
        return func(*args)

    return wrapper


def section(*args):
    """``args`` see `section command <https://openseespydoc.readthedocs.io/en/latest/src/section.html#section>`_"""
    sec_tag = args[1]
    FiberSecPlot.set_sec_tag(sec_tag)
    ops.section(*args)


@fetch_fiber_plot_data
def fiber(*args):
    """``args`` see `fiber command <https://openseespydoc.readthedocs.io/en/latest/src/fiber.html>`_"""
    ops.fiber(*args)


@fetch_fiber_plot_data
def patch(*args):
    """``args`` see `patch command <https://openseespydoc.readthedocs.io/en/latest/src/patch.html>`_"""
    ops.patch(*args)


@fetch_fiber_plot_data
def layer(*args):
    """``args`` see `layer command <https://openseespydoc.readthedocs.io/en/latest/src/layer.html>`_"""
    ops.layer(*args)


def plot_fiber_sec_cmds(
    sec_tag: int,
    title: str = "My Section",
    label_size: int = 18,
    tick_size: int = 15,
    title_size: int = 20,
):
    """Plot the fiber section by section tag.

    Parameters
    -----------
    sec_tag : int
        section tag in the ``OpenSeesPy`` domain.
    title : str, optional
        Title of plot, by default "My Section"
    label_size : int, optional
        Axis label size, by default 18
    tick_size : int, optional
        Axis tick size, by default 15
    title_size : int, optional
        Title size, by default 20
    """
    FiberSecPlot.plot_sec(
        sec_tag,
        title=title,
        label_size=label_size,
        tick_size=tick_size,
        title_size=title_size,
    )


def _line_mesh(y1, z1, y2, z2, num):
    length = np.sqrt((y2 - y1) ** 2 + (z2 - z1) ** 2)
    delta_l = length / num
    cos_alpha = np.abs(y2 - y1) / length
    sin_alpha = np.abs(z2 - z1) / length
    delta_y = np.sign(y2 - y1) * delta_l * cos_alpha
    delta_z = np.sign(z2 - z1) * delta_l * sin_alpha
    nodes = []
    for i in range(num + 1):
        nodes.append((y1 + i * delta_y, z1 + i * delta_z))
    return nodes


def _arc_mesh(ang0, ang1, r, num_c, yc, zc):
    ang0 = np.deg2rad(ang0)
    ang1 = np.deg2rad(ang1)
    delta_ang = (ang1 - ang0) / num_c
    nodes = []
    for i in range(num_c + 1):
        nodes.append((
            r * np.cos(ang0 + i * delta_ang) + yc,
            r * np.sin(ang0 + i * delta_ang) + zc,
        ))
    return nodes


if __name__ == "__main__":
    import openseespy.opensees as ops

    import opstool as opst

    ops.wipe()
    ops.model("basic", "-ndm", 3, "-ndf", 6)
    ops.uniaxialMaterial("Elastic", 1, 1000)
    # The following commands, like those in `OpenSeesPy <https://openseespydoc.readthedocs.io/en/latest/>`_,
    # create objects in the domain.
    sectag = 1
    opst.pre.section.section("Fiber", sectag, "-GJ", 1.0e6)
    opst.pre.section.patch("circ", 1, 40, 1, 0, 0, 1.9, 2, 0, 360, color="blue", opacity=0.75)
    opst.pre.section.patch("circ", 1, 40, 5, 0, 0, 1, 1.9, 0, 360, color="green", opacity=0.35)
    opst.pre.section.layer("circ", 1, 40, np.pi * 0.016**2, 0, 0, 1.9 - 0.016, 0.0, 360.0, color="red")
    # plot
    opst.pre.section.plot_fiber_sec_cmds(sec_tag=1)

    sectag = 2
    opst.pre.section.section("Fiber", sectag, "-GJ", 1.0e6)
    opst.pre.section.patch("quad", 1, 20, 20, -1, -1, 1, -1, 2, 3, -2, 3, color="blue", opacity=0.25)
    opst.pre.section.layer("straight", 1, 20, np.pi * 0.02**2, *[-0.9, -0.9], *[1.9, 2.9], color="black")
    opst.pre.section.fiber(0, 1, np.pi * 0.05**2, 1, color="red")
    # plot
    opst.pre.section.plot_fiber_sec_cmds(sec_tag=2)
