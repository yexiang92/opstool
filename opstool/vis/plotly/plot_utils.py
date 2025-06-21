from types import SimpleNamespace
from typing import TypedDict

import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objs as go
from matplotlib.colors import to_hex
from typing_extensions import Unpack

from ...utils import OPS_ELE_TYPES

default_cmap = [
    [0.0, "rgb(165,0,38)"],
    [0.1111111111111111, "rgb(215,48,39)"],
    [0.2222222222222222, "rgb(244,109,67)"],
    [0.3333333333333333, "rgb(253,174,97)"],
    [0.4444444444444444, "rgb(254,224,144)"],
    [0.5555555555555556, "rgb(224,243,248)"],
    [0.6666666666666666, "rgb(171,217,233)"],
    [0.7777777777777778, "rgb(116,173,209)"],
    [0.8888888888888888, "rgb(69,117,180)"],
    [1.0, "rgb(49,54,149)"],
]
default_cmap = [[1.0 - x[0], x[1]] for x in default_cmap[::-1]]


class _PLOT_ARGS_TYPES1(TypedDict):
    point_size: float
    line_width: float
    theme: str
    scale_factor: float
    show_mesh_edges: bool
    mesh_edge_color: str
    mesh_edge_width: float
    mesh_opacity: float
    font_size: int
    title_font_size: int
    font_family: str
    window_size: tuple[int | None, int | None]
    cmap: list | None | str  # or str for named colorscale, default is default_cmap
    cmap_model: list | None | str  # or str for named colorscale, default is None


class _PLOT_ARGS_TYPES2(TypedDict):
    point: str
    frame: str
    beam: str
    truss: str
    link: str
    shell: str
    plane: str
    brick: str
    tet: str
    joint: str
    contact: str
    pfem: str
    constraint: str
    bc: str
    cmap: list | None | str  # or str for named colorscale, default is default_cmap
    cmap_model: list | None | str  # or str for named colorscale, default is None


PLOT_ARGS = SimpleNamespace(
    point_size=3.0,
    line_width=5.0,
    theme="plotly",
    scale_factor=1 / 15,
    show_mesh_edges=True,
    mesh_edge_color="black",
    mesh_edge_width=1.0,
    mesh_opacity=1.0,
    font_size=15,
    title_font_size=18,
    font_family="Arial, sans-serif",
    window_size=(None, None),
    # ------------------------------
    color_point="#580f41",
    color_frame="#0652ff",
    color_beam="#0652ff",
    color_truss="#FF8C00",
    color_link="#39FF14",
    color_shell="#76b852",
    color_plane="#00FFFF",
    color_brick="#FF4500",
    color_tet="#FFFF33",
    color_joint="#7FFF00",
    color_contact="#ff9408",
    color_pfem="#8080FF",
    color_constraint="#FF1493",
    color_bc="#15b01a",
    cmap=default_cmap,  # "plasma",
    cmap_model=None,
)


def set_plot_props(**kwargs: Unpack[_PLOT_ARGS_TYPES1]):
    """
    Set ploting properties.

    Parameters
    ----------
    kwargs: optional keyword arguments, including:
        * point_size : float, optional
            Point size of any nodes. Default ``5.0``
        * line_width : float, optional
            Thickness of line elements.  Only valid for wireframe and surface
            representations.  Default ``3.0``.
        * cmap : str, list, optional, default: None
            One of the following named colorscales: ["aggrnyl", "agsunset", "algae", "amp", "armyrose",
            "balance", "blackbody", "bluered", "blues", "blugrn", "bluyl", "brbg", "brwnyl", "bugn",
            "bupu", "burg", "burgyl", "cividis", "curl", "darkmint", "deep", "delta", "dense",
            "earth", "edge", "electric", "emrld", "fall", "geyser", "gnbu", "gray", "greens", "greys",
            "haline", "hot", "hsv", "ice", "icefire", "inferno", "jet", "magenta", "magma", "matter",
            "mint", "mrybm", "mygbm", "oranges", "orrd", "oryel", "oxy", "peach", "phase", "picnic",
            "pinkyl", "piyg", "plasma", "plotly3", "portland", "prgn", "pubu", "pubugn", "puor", "purd",
            "purp", "purples", "purpor", "rainbow", "rdbu", "rdgy", "rdpu", "rdylbu", "rdylgn", "redor",
            "reds", "solar", "spectral", "speed", "sunset", "sunsetdark", "teal", "tealgrn", "tealrose", "tempo",
            "temps", "thermal", "tropic", "turbid", "turbo", "twilight",
            "viridis", "ylgn", "ylgnbu", "ylorbr", "ylorrd"].

            Appending "_r" to a named colorscale reverses it.
        * cmap_model : str, list, optional, default=None
            Matplotlib colormap used for geometry model visualization.
            Same as ``cmap``, except that this parameter will be used
            for geometry model visualization and will be automatically mapped
            according to different element types.
            If None, If None, the color specified in the function``set_plot_colors``
            will be used.

            Available color maps are shown in
            `Colormaps in Matplotlib <https://matplotlib.org/stable/users/explain/colors/colormaps.html>`_
        * theme : str, optional, default: "plotly"
            Available theme templates for plotly:
            ['ggplot2', 'seaborn', 'simple_white', 'plotly', 'plotly_white',
            'plotly_dark', 'presentation', 'xgridoff', 'ygridoff', 'gridon', 'none']
        * window_size : list, optional
            Window size in pixels. Default to ``(None, None)``
        * show_mesh_edges: bool, default: True
            Whether to display the mesh edges of ``planes``, ``plates``, ``shells``, and ``solid`` elements.
        * mesh_edge_color: str, default: black
            Color of the mesh edges for ``planes``, ``plates``, ``shells``, and ``solid`` elements.
        * mesh_edge_width: float, default: 1.0
            Width of the mesh edges for ``planes``, ``plates``, ``shells``, and ``solid`` elements.
        * mesh_opacity: float, default: 1.0
            Display opacity for ``planes``, ``plates``, ``shells``, and ``solid`` elements.
        * font_family : str, optional, default: "Arial, sans-serif"
            HTML font family - the typeface that will be applied by the web browser.
            The web browser will only be able to apply a font if it is available on the system which it operates.
            Provide multiple font families, separated by commas, to indicate the preference in which to apply fonts
            if they aren"t available on the system.
            The Chart Studio Cloud (at https://chart-studio.plotly.com or on-premise) generates images on a server,
            where only a select number of fonts are installed and supported.
            These include “Arial”, “Balto”, “Courier New”, “Droid Sans”, “Droid Serif”, “Droid Sans Mono”, “Gravitas One”,
            “Old Standard TT”, “Open Sans”, “Overpass”, “PT Sans Narrow”, “Raleway”, “Times New Roman”.
        * scale_factor : float, optional
            Scale factor between the maximum deformation of the model and the maximum boundary size.
            Default ``1 / 20``.
        * font_size: int, default: 15
            Font size of labels.
        * title_font_size: int, default: 18
            Font size of title.

    Returns
    -------
    None
    """
    for key, value in kwargs.items():
        setattr(PLOT_ARGS, key, value)


def set_plot_colors(**kwargs: Unpack[_PLOT_ARGS_TYPES2]):
    """
    Set the display color of various element types.

    Parameters
    ----------
    kwargs : optional keyword arguments, including:
        * point : str, list[int, int, int], optional
            Color for nodal points.
            Either a string, RGB list, or hex color string.  For example,
            ``point='white'``, ``point='w'``, ``point=[1, 1, 1]``, or
            ``point='#FFFFFF'``.
            frame : str, list[int, int, int], optional
            Color for frame elements.
        * frame : str, list[int, int, int], optional
            Color for frame elements.
        * truss : str, list[int, int, int], optional
            Color for truss elements.
        * link : str, list[int, int, int], optional
            Color for link elements.
        * shell : str, list[int, int, int], optional
            Color for shell elements.
        * plane : str, list[int, int, int], optional
            Color for plane elements.
        * brick : str, list[int, int, int], optional
            Color for brick (solid) elements.
        * tet : str, list[int, int, int], optional
            Color for tetrahedral (solid) elements.
        * joint : str, list[int, int, int], optional
            Color for beam-column joint elements.
        * contact : str, list[int, int, int], optional
            Color for contact elements.
        * pfem : str, list[int, int, int], optional
            Color for PFEM elements.
        * constraint : str, list[int, int, int], optional
            Color for constraint.
        * bc : str, list[int, int, int], optional
            Color for boundary conditions.
        * cmap : str, list, optional, default: "plasma"
            One of the following named colorscales: ["aggrnyl", "agsunset", "algae", "amp", "armyrose",
            "balance", "blackbody", "bluered", "blues", "blugrn", "bluyl", "brbg", "brwnyl", "bugn",
            "bupu", "burg", "burgyl", "cividis", "curl", "darkmint", "deep", "delta", "dense",
            "earth", "edge", "electric", "emrld", "fall", "geyser", "gnbu", "gray", "greens", "greys",
            "haline", "hot", "hsv", "ice", "icefire", "inferno", "jet", "magenta", "magma", "matter",
            "mint", "mrybm", "mygbm", "oranges", "orrd", "oryel", "oxy", "peach", "phase", "picnic",
            "pinkyl", "piyg", "plasma", "plotly3", "portland", "prgn", "pubu", "pubugn", "puor", "purd",
            "purp", "purples", "purpor", "rainbow", "rdbu", "rdgy", "rdpu", "rdylbu", "rdylgn", "redor",
            "reds", "solar", "spectral", "speed", "sunset", "sunsetdark", "teal", "tealgrn", "tealrose", "tempo",
            "temps", "thermal", "tropic", "turbid", "turbo", "twilight",
            "viridis", "ylgn", "ylgnbu", "ylorbr", "ylorrd"].

            Appending "_r" to a named colorscale reverses it.
        * cmap_model : str, list, optional, default=None
            Matplotlib colormap used for geometry model visualization.
            Same as ``cmap``, except that this parameter will be used
            for geometry model visualization and will be automatically mapped
            according to different element types.
            If None, If None, the color specified in the function``set_plot_colors``
            will be used.

            Available color maps are shown in
            `Colormaps in Matplotlib <https://matplotlib.org/stable/users/explain/colors/colormaps.html>`_

    Returns
    -------
    None

    """
    for key, value in kwargs.items():
        if key in ["cmap", "cmap_model"]:
            setattr(PLOT_ARGS, key, value)
        else:
            setattr(PLOT_ARGS, "color_" + key, value)
    if "frame" in kwargs:
        PLOT_ARGS.color_beam = kwargs["frame"]


def _get_ele_color(ele_types: list[str]):
    if PLOT_ARGS.cmap_model:
        cmap = plt.get_cmap(PLOT_ARGS.cmap_model)
        colors = cmap(np.linspace(0, 1, len(ele_types)))
        colors = [to_hex(color) for color in colors]
    else:
        colors = ["#01153e"] * len(ele_types)
        for i, ele_type in enumerate(ele_types):
            if ele_type in OPS_ELE_TYPES.Beam:
                colors[i] = PLOT_ARGS.color_frame
            elif ele_type in OPS_ELE_TYPES.Truss:
                colors[i] = PLOT_ARGS.color_truss
            elif ele_type in OPS_ELE_TYPES.Link:
                colors[i] = PLOT_ARGS.color_link
            elif ele_type in OPS_ELE_TYPES.Plane:
                colors[i] = PLOT_ARGS.color_plane
            elif ele_type in OPS_ELE_TYPES.Shell:
                colors[i] = PLOT_ARGS.color_shell
            elif ele_type in OPS_ELE_TYPES.Tet:
                colors[i] = PLOT_ARGS.color_tet
            elif ele_type in OPS_ELE_TYPES.Brick:
                colors[i] = PLOT_ARGS.color_brick
            elif ele_type in OPS_ELE_TYPES.PFEM:
                colors[i] = PLOT_ARGS.color_pfem
            elif ele_type in OPS_ELE_TYPES.Joint:
                colors[i] = PLOT_ARGS.color_joint
            elif ele_type in OPS_ELE_TYPES.Contact:
                colors[i] = PLOT_ARGS.color_contact
    return colors


# -------------------------------------------------------------------------
# ---------------------------- Plotting Functions -------------------------
# -------------------------------------------------------------------------


class _VTKElementTriangulator:
    # https://www.weiy.city/2019/10/check-cells-in-model/
    # https://examples.vtk.org/site/Cxx/GeometricObjects/IsoparametricCellsDemo/
    def __init__(self, points, scalars=None, scalars_by_element=False):
        self.points = points
        self.face_points = []
        self.face_line_points = []
        self.face_mid_points = []
        self.veci = []
        self.vecj = []
        self.veck = []
        self.scalars = scalars
        self.face_scalars = []
        self.face_line_scalars = []
        self.by_ele = scalars_by_element
        self.scalars_idx_by_ele = 0

    def add_cell(self, cell_type, cell):
        data = self.points[cell[1:], :]
        self.face_mid_points.append(np.mean(data, axis=0))
        self.face_points.extend(data)
        if self.scalars is not None:
            if self.by_ele:
                idx = self.scalars_idx_by_ele
                self.face_scalars.extend([self.scalars[idx]] * len(data))
                self.scalars_idx_by_ele += 1
            else:
                self.face_scalars.extend(self.scalars[cell[1:]])
        self._add_vectors(cell_type, len(self.face_points) - len(data))

    def _add_vectors(self, cell_type, base_idx):
        if cell_type == 5:  # VTK_TRIANGLE
            self._add_triangle(base_idx)
        elif cell_type == 22:  # QUADRATIC_TRIANGLE
            self._add_quadratic_triangle(base_idx)
        elif cell_type == 34:  # BIQUADRATIC_TRIANGLE
            self._add_biquadratic_triangle(base_idx)
        elif cell_type == 9:  # VTK_QUAD
            self._add_quad(base_idx)
        elif cell_type == 23:  # QUADRATIC_QUAD
            self._add_quadratic_quad(base_idx)
        elif cell_type == 28:  # BIQUADRATIC_QUAD
            self._add_biquadratic_quad(base_idx)
        elif cell_type == 10:  # VTK_TETRA
            self._add_tetra(base_idx)
        elif cell_type == 24:  # QUADRATIC_TETRA
            self._add_quadratic_tetra(base_idx)
        elif cell_type == 12:  # VTK_HEXAHEDRON
            self._add_hexahedron(base_idx)
        elif cell_type == 25:  # QUADRATIC_HEXAHEDRON
            self._add_quadratic_hexahedron(base_idx)
        elif cell_type == 29:  # TRIQUADRATIC_HEXAHEDRON
            self._add_triquadratic_hexahedron(base_idx)

    def _add_triangle(self, idx):
        self._add_vectors_from_tuples(idx, [(0, 1, 2)])
        self._add_line_points(idx, [(0, 1, 2, 0)])

    def _add_quadratic_triangle(self, idx):
        connections = [(0, 3, 5), (1, 4, 3), (2, 5, 4), (3, 4, 5)]
        self._add_vectors_from_tuples(idx, connections)
        self._add_line_points(idx, [(0, 3, 1, 4, 2, 5, 0)])

    def _add_biquadratic_triangle(self, idx):
        connections = [(0, 3, 6), (3, 4, 6), (3, 1, 4), (0, 6, 5), (4, 5, 6), (2, 5, 4)]
        self._add_vectors_from_tuples(idx, connections)
        self._add_line_points(idx, [(0, 3, 1, 4, 2, 5, 0)])

    def _add_quad(self, idx):
        connections = [(0, 1, 2), (0, 2, 3)]
        self._add_vectors_from_tuples(idx, connections)
        self._add_line_points(idx, [(0, 1, 2, 3, 0)])

    def _add_quadratic_quad(self, idx):
        connections = [(0, 4, 7), (1, 5, 4), (2, 6, 5), (3, 7, 6), (4, 6, 7), (4, 5, 6)]
        self._add_vectors_from_tuples(idx, connections)
        self._add_line_points(idx, [(0, 4, 1, 5, 2, 6, 3, 7, 0)])

    def _add_biquadratic_quad(self, idx):
        connections = [
            (0, 4, 7),
            (1, 5, 4),
            (2, 6, 5),
            (3, 7, 6),
            (6, 7, 8),
            (5, 6, 8),
            (7, 4, 8),
            (4, 5, 8),
        ]
        self._add_vectors_from_tuples(idx, connections)
        self._add_line_points(idx, [(0, 4, 1, 5, 2, 6, 3, 7, 0)])

    def _add_tetra(self, idx):
        connections = [(0, 1, 2), (0, 1, 3), (0, 2, 3), (1, 2, 3)]
        self._add_vectors_from_tuples(idx, connections)
        self._add_line_points(idx, [(0, 1, 2, 0), (0, 1, 3, 0), (0, 2, 3, 0), (1, 2, 3, 1)])

    def _add_quadratic_tetra(self, idx):
        connections = [
            (0, 4, 7),
            (1, 8, 4),
            (3, 7, 8),
            (4, 8, 7),
            (1, 8, 5),
            (3, 9, 8),
            (2, 5, 9),
            (5, 8, 9),
            (0, 7, 6),
            (2, 6, 9),
            (3, 9, 7),
            (6, 7, 9),
            (0, 4, 6),
            (1, 5, 4),
            (2, 6, 5),
            (4, 5, 6),
        ]
        self._add_vectors_from_tuples(idx, connections)
        self._add_line_points(
            idx,
            [
                (0, 4, 1, 5, 2, 6, 0),
                (0, 4, 1, 8, 3, 7, 0),
                (0, 7, 3, 9, 2, 6, 0),
                (1, 8, 3, 9, 2, 5, 1),
            ],
        )

    def _add_hexahedron(self, idx):
        connections = [
            (0, 1, 2),
            (0, 2, 3),
            (0, 3, 7),
            (0, 7, 4),
            (0, 1, 5),
            (0, 5, 4),
            (1, 2, 6),
            (1, 6, 5),
            (2, 6, 3),
            (3, 6, 7),
            (4, 5, 6),
            (4, 6, 7),
        ]
        self._add_vectors_from_tuples(idx, connections)
        self._add_line_points(
            idx,
            [
                (0, 1, 2, 3, 0),
                (0, 1, 5, 4, 0),
                (0, 3, 7, 4, 0),
                (1, 2, 6, 5, 1),
                (2, 3, 7, 6, 2),
                (4, 5, 6, 7, 4),
            ],
        )

    def _add_quadratic_hexahedron(self, idx):
        connections = [
            (0, 8, 11),
            (1, 9, 8),
            (2, 10, 9),
            (3, 11, 10),
            (9, 10, 11),
            (8, 9, 11),
            (0, 16, 8),
            (4, 12, 16),
            (5, 17, 12),
            (1, 8, 17),
            (8, 12, 17),
            (8, 16, 12),
            (0, 16, 11),
            (4, 15, 16),
            (7, 19, 15),
            (3, 11, 19),
            (11, 19, 16),
            (15, 19, 16),
            (4, 12, 15),
            (5, 13, 12),
            (6, 14, 13),
            (7, 15, 14),
            (12, 14, 15),
            (12, 13, 14),
            (3, 19, 10),
            (7, 14, 19),
            (6, 18, 14),
            (2, 10, 18),
            (10, 19, 18),
            (14, 18, 19),
            (1, 17, 9),
            (5, 13, 17),
            (6, 18, 13),
            (2, 9, 18),
            (9, 13, 18),
            (9, 17, 13),
        ]
        self._add_vectors_from_tuples(idx, connections)
        self._add_line_points(
            idx,
            [
                (0, 8, 1, 9, 2, 10, 3, 11, 0),
                (0, 16, 4, 12, 5, 17, 1, 8, 0),
                (0, 16, 4, 15, 7, 19, 3, 11, 0),
                (4, 12, 5, 13, 6, 14, 7, 15, 4),
                (3, 19, 7, 14, 6, 18, 2, 10, 3),
                (1, 17, 5, 13, 6, 18, 2, 9, 1),
            ],
        )

    def _add_triquadratic_hexahedron(self, idx):
        connections = [
            # Bottom face (nodes: 0, 1, 2, 3, 8, 9, 10, 11, 24)
            (0, 8, 24),
            (8, 1, 24),
            (1, 9, 24),
            (9, 2, 24),
            (2, 10, 24),
            (10, 3, 24),
            (3, 11, 24),
            (11, 0, 24),
            # Top face (nodes: 4, 5, 6, 7, 12, 13, 14, 15, 25)
            (4, 12, 25),
            (12, 5, 25),
            (5, 13, 25),
            (13, 6, 25),
            (6, 14, 25),
            (14, 7, 25),
            (7, 15, 25),
            (15, 4, 25),
            # Front face (nodes: 0, 1, 5, 4, 8, 16, 12, 17, 26)
            (0, 8, 26),
            (8, 1, 26),
            (1, 16, 26),
            (16, 5, 26),
            (5, 12, 26),
            (12, 4, 26),
            (4, 17, 26),
            (17, 0, 26),
            # Right face (nodes: 1, 2, 6, 5, 9, 18, 13, 16, 26)
            (1, 9, 26),
            (9, 2, 26),
            (2, 18, 26),
            (18, 6, 26),
            (6, 13, 26),
            (13, 5, 26),
            (5, 16, 26),
            (16, 1, 26),
            # Back face (nodes: 2, 3, 7, 6, 10, 19, 14, 18, 26)
            (2, 10, 26),
            (10, 3, 26),
            (3, 19, 26),
            (19, 7, 26),
            (7, 14, 26),
            (14, 6, 26),
            (6, 18, 26),
            (18, 2, 26),
            # Left face (nodes: 3, 0, 4, 7, 11, 17, 15, 19, 26)
            (3, 11, 26),
            (11, 0, 26),
            (0, 17, 26),
            (17, 4, 26),
            (4, 15, 26),
            (15, 7, 26),
            (7, 19, 26),
            (19, 3, 26),
        ]
        self._add_vectors_from_tuples(idx, connections)
        self._add_line_points(
            idx,
            [
                (0, 8, 1, 9, 2, 10, 3, 11, 0),
                (0, 16, 4, 12, 5, 17, 1, 8, 0),
                (0, 16, 4, 15, 7, 19, 3, 11, 0),
                (4, 12, 5, 13, 6, 14, 7, 15, 4),
                (3, 19, 7, 14, 6, 18, 2, 10, 3),
                (1, 17, 5, 13, 6, 18, 2, 9, 1),
            ],
        )

    def _add_vectors_from_tuples(self, idx, connections):
        for i, j, k in connections:
            self.veci.append(idx + i)
            self.vecj.append(idx + j)
            self.veck.append(idx + k)

    def _add_line_points(self, idx, connections):
        for cell in np.array(connections):
            data0 = [self.face_points[idx + cell_] for cell_ in cell]
            data1 = [[np.nan, np.nan, np.nan]]
            self.face_line_points.extend(data0 + data1)
            if self.scalars is not None:
                data0 = [self.face_scalars[idx + cell_] for cell_ in cell]
                data1 = [np.nan]
                self.face_line_scalars.extend(data0 + data1)

    def _get_results(self):
        if self.scalars is None:
            return (
                self.face_points,
                self.face_line_points,
                self.face_mid_points,
                self.veci,
                self.vecj,
                self.veck,
            )
        else:
            return (
                self.face_points,
                self.face_line_points,
                self.face_mid_points,
                self.veci,
                self.vecj,
                self.veck,
                self.face_scalars,
                self.face_line_scalars,
            )

    def get_results(self):
        output = self._get_results()
        output = [np.array(data) for data in output]
        return output


# Usage
# grid = VTKUnstructuredGrid(points)
# for cell_type, cell in zip(cell_types, cells):
#     grid.add_cell(cell_type, cell)
# face_points, face_line_points, face_mid_points, veci, vecj, veck = grid.get_results()


def _make_lines_plotly(points, cells, scalars=None):
    line_points = []
    line_mid_points = []
    line_scalars = []
    for cell in cells:
        data0 = points[cell[1:], :]
        data1 = [np.nan, np.nan, np.nan]
        data = np.vstack([data0, data1])
        line_points.extend(data)
        line_mid_points.append(np.mean(data0, axis=0))
        if scalars is not None:
            line_scalars.extend(scalars[cell[1:]])
            line_scalars.append(np.nan)
    line_points = np.array(line_points)
    line_mid_points = np.array(line_mid_points)
    line_scalars = np.array(line_scalars)
    if scalars is None:
        return line_points, line_mid_points
    else:
        return line_points, line_mid_points, line_scalars


def _plot_points(
    plotter: list,
    pos,
    color: str = "black",
    size: float = 3.0,
    symbol: str = "circle",
    name="Node",
    customdata=None,
    hovertemplate=None,
):
    if len(pos) > 0 and size > 0:
        x, y, z = [pos[:, j] for j in range(3)]
        point_plot = go.Scatter3d(
            x=x,
            y=y,
            z=z,
            marker={"size": size, "color": color, "symbol": symbol},
            mode="markers",
            name=name,
            customdata=customdata,
            hovertemplate=hovertemplate,
            # hovertemplate="<b>x: %{x}</b><br>y: %{y}<br>z: %{z} <br>tag: %{customdata}",
        )
        plotter.append(point_plot)
        return point_plot
    else:
        return None


def _plot_points_cmap(
    plotter: list,
    pos,
    scalars,
    clim=None,
    coloraxis=None,
    size: float = 3.0,
    name="",
    show_hover: bool = True,
    color=None,
):
    if len(pos) > 0 and size > 0:
        if clim is None:
            clim = [np.min(scalars), np.max(scalars)]
        if show_hover:
            hover_kargs = {"customdata": scalars, "hovertemplate": "<b>%{customdata:.4E}</b>"}
        else:
            hover_kargs = {"hoverinfo": "skip"}
        if color is None:
            marker = {"size": size, "color": scalars, "coloraxis": coloraxis, "cmin": clim[0], "cmax": clim[1]}
        else:
            marker = {"size": size, "color": color}
        point_plot = go.Scatter3d(
            x=pos[:, 0],
            y=pos[:, 1],
            z=pos[:, 2],
            marker=marker,
            mode="markers",
            name=name,
            **hover_kargs,
        )
        plotter.append(point_plot)
        return point_plot
    else:
        return None


def _plot_lines(
    plotter: list,
    pos,
    width=1.0,
    color="blue",
    name="Line",
    customdata=None,
    hovertemplate=None,
    hoverinfo=None,
):
    if len(pos) > 0:
        x, y, z = [pos[:, j] for j in range(3)]
        line_plot = go.Scatter3d(
            x=x,
            y=y,
            z=z,
            line={"color": color, "width": width},
            mode="lines",
            name=name,
            customdata=customdata,
            hovertemplate=hovertemplate,
            connectgaps=False,
            hoverinfo=hoverinfo,
            # hoverinfo="skip",
        )
        plotter.append(line_plot)
        return line_plot
    else:
        return None


def _plot_lines_cmap(plotter: list, pos, scalars, coloraxis=None, width=1.0, clim=None, color=None):
    if len(pos) > 0:
        if clim is None:
            clim = [scalars.min(), scalars.max()]
        if color is None:
            line_dict = {
                "color": scalars,
                "width": width,
                "cmin": clim[0],
                "cmax": clim[1],
                "coloraxis": coloraxis,
            }
        else:
            line_dict = {"color": color, "width": width}
        line_plot = go.Scatter3d(
            x=pos[:, 0],
            y=pos[:, 1],
            z=pos[:, 2],
            line=line_dict,
            mode="lines",
            connectgaps=False,
            hoverinfo="skip",
        )
        plotter.append(line_plot)
        return line_plot
    else:
        return None


def _plot_unstru(
    plotter: list,
    pos,
    veci,
    vecj,
    veck,
    color="gray",
    name=None,
    customdata=None,
    hovertemplate=None,
    hoverinfo=None,
    opacity=1.0,
    style="surface",
    line_width: float = 2.0,
    show_edges: bool = True,
    edge_color: str = " black",
    edge_width: float = 1.0,
    edge_points: np.ndarray = None,
):
    """plot the unstructured grid."""
    if style.lower() == "surface":
        if len(pos) > 0:
            x, y, z = [pos[:, j] for j in range(3)]
            grid = go.Mesh3d(
                x=x,
                y=y,
                z=z,
                i=veci,
                j=vecj,
                k=veck,
                name=name,
                color=color,
                opacity=opacity,
                customdata=customdata,
                hovertemplate=hovertemplate,
                hoverinfo=hoverinfo,
                # hoverinfo="skip",
            )
            plotter.append(grid)
        if show_edges:
            _plot_lines(
                plotter,
                pos=edge_points,
                color=edge_color,
                width=edge_width,
                name="Edge",
                hoverinfo="skip",
            )
    else:
        _plot_lines(
            plotter,
            pos=edge_points,
            color=color,
            width=line_width,
            name=name,
            customdata=customdata,
            hovertemplate=hovertemplate,
            hoverinfo=hoverinfo,
        )


def _plot_unstru_cmap(
    plotter: list,
    pos,
    veci,
    vecj,
    veck,
    scalars,
    clim=None,
    coloraxis=None,
    opacity=1.0,
    style="surface",
    color=None,
    line_width: float = 2.0,
    show_edges: bool = True,
    edge_color: str = "black",
    edge_width: float = 1.0,
    edge_scalars: np.ndarray = None,
    edge_points: np.ndarray = None,
):
    if clim is None:
        clim = [scalars.min(), scalars.max()]
    if style.lower() == "surface":
        if len(pos) > 0:
            if color is None:
                kargs = {
                    "text": scalars,
                    "intensity": scalars,
                    "cmin": clim[0],
                    "cmax": clim[1],
                    "coloraxis": coloraxis,
                }
            else:
                kargs = {"color": color}
            grid = go.Mesh3d(
                x=pos[:, 0],
                y=pos[:, 1],
                z=pos[:, 2],
                i=veci,
                j=vecj,
                k=veck,
                opacity=opacity,
                hoverinfo="skip",
                **kargs,
            )
            plotter.append(grid)
        if show_edges:
            _plot_lines(
                plotter,
                pos=edge_points,
                color=edge_color,
                width=edge_width,
                name="Edge",
                hoverinfo="skip",
            )
    else:
        _plot_lines_cmap(
            plotter,
            pos=edge_points,
            scalars=edge_scalars,
            clim=clim,
            coloraxis=coloraxis,
            width=line_width,
            color=color,
        )


def _plot_all_mesh(
    plotter: list,
    line_points,
    unstru_line_points,
    color="gray",
    width=1.5,
):
    if len(line_points) > 0:
        _plot_lines(
            plotter,
            pos=line_points,
            color=color,
            name="",
            width=width,
            hoverinfo="skip",
        )
    if len(unstru_line_points) > 0:
        _plot_lines(
            plotter,
            pos=unstru_line_points,
            color=color,
            name="",
            width=width,
            hoverinfo="skip",
        )


def _plot_all_mesh_cmap(
    plotter: list,
    points,
    point_scalars,
    line_points,
    line_scalars,
    unstru_points,
    unstru_scalars,
    unstru_veci,
    unstru_vecj,
    unstru_veck,
    coloraxis=None,
    clim=None,
    point_size=0,
    line_width=1.0,
    opacity=1.0,
    show_edges: bool = True,
    edge_color: str = "black",
    edge_width: float = 1.0,
    edge_scalars: np.ndarray = None,
    edge_points: np.ndarray = None,
):
    if point_size > 1e-3:
        _plot_points_cmap(
            plotter,
            pos=points,
            scalars=point_scalars,
            coloraxis=coloraxis,
            size=point_size,
        )
    _plot_lines_cmap(
        plotter,
        pos=line_points,
        scalars=line_scalars,
        coloraxis=coloraxis,
        clim=clim,
        width=line_width,
    )
    _plot_unstru_cmap(
        plotter,
        pos=unstru_points,
        veci=unstru_veci,
        vecj=unstru_vecj,
        veck=unstru_veck,
        scalars=unstru_scalars,
        clim=clim,
        coloraxis=coloraxis,
        opacity=opacity,
        show_edges=show_edges,
        edge_color=edge_color,
        edge_width=edge_width,
        edge_scalars=edge_scalars,
        edge_points=edge_points,
    )
    return None


def _get_line_cells(line_data):
    if len(line_data) > 0:
        line_cells = line_data.to_numpy().astype(int)
        line_tags = line_data.coords["eleTags"]
    else:
        line_cells, line_tags = [], []
    return line_cells, line_tags


def _get_unstru_cells(unstru_data):
    if len(unstru_data) > 0:
        unstru_tags = unstru_data.coords["eleTags"]
        unstru_cell_types = np.array(unstru_data[:, -1], dtype=int)
        unstru_cells = unstru_data.to_numpy()
        if not np.any(np.isnan(unstru_cells)):
            unstru_cells_new = unstru_cells[:, :-1].astype(int)
        else:
            unstru_cells_new = []
            for cell in unstru_cells:
                num = int(cell[0])
                data = [num] + [int(data) for data in cell[1 : 1 + num]]
                unstru_cells_new.extend(data)
    else:
        unstru_tags, unstru_cell_types, unstru_cells_new = [], [], []
    return unstru_tags, unstru_cell_types, unstru_cells_new


def _dropnan_by_time(da, model_update=False):
    dims = da.dims
    time_dim = dims[0]
    cleaned_dataarrays = []
    for t in range(da.sizes[time_dim]):
        da_2d = da.isel({time_dim: t})
        if da_2d.size == 0 or any(dim == 0 for dim in da_2d.shape):
            cleaned_dataarrays.append([])
        else:
            dim2 = dims[1]
            da_2d_cleaned = da_2d.dropna(dim=dim2, how="any") if model_update else da_2d
            cleaned_dataarrays.append(da_2d_cleaned)
    return cleaned_dataarrays


# def group_cells(cells):
#     line_cells, line_cells_type = [], []
#     unstru_cells, unstru_cells_type = [], []
#     cells_vtk = cells.ELE_CELLS["VTK"]
#     cells_type_vtk = cells.ELE_CELLS["VTKType"]
#     for name in cells_vtk.keys():
#         for cell_, cell_type_ in zip(cells_vtk[name], cells_type_vtk[name]):
#             if cell_[0][0] == 2:
#                 line_cells.append(cell_)
#                 line_cells_type.append(cell_type_)
#             else:
#                 unstru_cells.append(cell_)
#                 unstru_cells_type.append(cell_type_)
#     return line_cells, line_cells_type, unstru_cells, unstru_cells_type
