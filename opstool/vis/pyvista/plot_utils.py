from types import SimpleNamespace
from typing import Optional, TypedDict

import matplotlib.pyplot as plt
import numpy as np
import pyvista as pv
import vtk
from typing_extensions import Unpack

from ...utils import CONFIGS, OPS_ELE_TYPES

PKG_NAME = CONFIGS.get_pkg_name()
pv.global_theme.title = PKG_NAME

_scalar_bar_kargs = {
    "fmt": "%10.3e",
    "n_labels": 10,
    "bold": False,
    "width": 0.1,
    "height": 0.5,
    "vertical": True,
    "font_family": "courier",
    "label_font_size": None,
    "title_font_size": None,
    "position_x": 0.825,
    "position_y": 0.05,
}


class _PLOT_ARGS_TYPES1(TypedDict):
    point_size: float
    line_width: float
    theme: str
    window_size: tuple[int, int]
    render_points_as_spheres: bool
    render_lines_as_tubes: bool
    anti_aliasing: str
    msaa_multi_samples: int
    smooth_shading: Optional[bool]
    lighting: Optional[bool]
    line_smoothing: bool
    polygon_smoothing: bool
    notebook: bool
    jupyter_backend: str
    font_family: Optional[str]
    scale_factor: float
    show_mesh_edges: bool
    mesh_edge_color: str
    mesh_edge_width: float
    mesh_opacity: float
    font_size: int
    title_font_size: int
    off_screen: bool
    scalar_bar_kargs: dict
    cmap: list | None | str  # or str for named colorscale, default is default_cmap
    cmap_model: list | None | str  # or str for named colorscale, default is None


class _PLOT_ARGS_TYPES2(TypedDict):
    point: str | list[int]
    frame: str | list[int]
    beam: str | list[int]
    truss: str | list[int]
    link: str | list[int]
    shell: str | list[int]
    plane: str | list[int]
    brick: str | list[int]
    tet: str | list[int]
    joint: str | list[int]
    contact: str | list[int]
    pfem: str | list[int]
    constraint: str | list[int]
    bc: str | list[int]
    cmap: list | None | str  # or str for named colorscale, default is default_cmap
    cmap_model: list | None | str  # or str for named colorscale, default is None
    n_colors: int
    color_map: list | None | str
    nodal_label: str | list[int]
    ele_label: str | list[int]


PLOT_ARGS = SimpleNamespace(
    point_size=1.0,
    line_width=2.0,
    theme="default",
    window_size=(1024, 768),
    render_points_as_spheres=True,
    render_lines_as_tubes=True,
    anti_aliasing="msaa",
    msaa_multi_samples=16,
    smooth_shading=None,
    lighting=None,
    line_smoothing=True,
    polygon_smoothing=True,
    notebook=False,
    jupyter_backend="trame",
    font_family=None,
    scale_factor=1 / 15,
    show_mesh_edges=True,
    mesh_edge_color="black",
    mesh_edge_width=1.0,
    mesh_opacity=1.0,
    font_size=15,
    title_font_size=18,
    off_screen=False,
    scalar_bar_kargs=_scalar_bar_kargs,
    # --------------------------
    color_point="#FF0055",
    color_frame="#0652ff",
    color_beam="#0652ff",
    color_truss="#FF8C00",
    color_link="#39FF14",
    color_shell="#769958",
    color_plane="#00FFFF",
    color_brick="#FF4500",
    color_tet="#FFFF33",
    color_joint="#7FFF00",
    color_contact="#ff9408",
    color_pfem="#8080FF",
    color_constraint="#FF1493",
    color_bc="#15b01a",
    cmap="jet",
    cmap_model=None,
    n_colors=256,
    color_map="jet",
    color_nodal_label="#048243",
    color_ele_label="#650021",
)


def set_plot_props(**kwargs: Unpack[_PLOT_ARGS_TYPES1]) -> None:
    """
    Set ploting properties.

    Parameters
    ----------
    kwargs: optional keyword arguments, including:
        - point_size : float, optional
            Point size of any nodes. Default ``5.0``
        - line_width : float, optional
            Thickness of line elements.  Only valid for wireframe and surface
            representations.  Default ``3.0``.
        - cmap : str, list, optional
            Name of the Matplotlib colormap to us when mapping the
            scalars.  See available Matplotlib colormaps.  Only
            applicable for when displaying ``scalars``. Requires Matplotlib
            to be installed.  ``colormap`` is also an accepted alias for
            this. If ``colorcet`` or ``cmocean`` are installed, their
            colormaps can be specified by name.

            You can also specify a list of colors to override an
            existing colormap with a custom one.  For example, to
            create a three color colormap you might specify
            ``['green', 'red', 'blue']``.
        - cmap_model : str, list, optional, default=None
            Matplotlib colormap used for geometry model visualization.
            Same as ``cmap``, except that this parameter will be used
            for geometry model visualization and will be automatically mapped
            according to different element types.
            If None, If None, the color specified in the function``set_plot_colors``
            will be used.

            Available color maps are shown in
            `Colormaps in Matplotlib <https://matplotlib.org/stable/users/explain/colors/colormaps.html>`_
        - n_colors : int, optional
            Number of colors to use when displaying scalars. Default to 256.
            The scalar bar will also have this many colors.
        - theme : str, optional,
            Theme name. Either 'default', 'document', 'dark', or 'paraview'.
            Defaults to "default" theme.
        - window_size : list, optional
            Window size in pixels.  Defaults to ``[1024, 768]``
        - render_points_as_spheres : bool, optional
            Render points as spheres.
        - render_lines_as_tubes : bool, optional
            Renders lines as tubes.
        - anti_aliasing: str, optional, default="msaa"
            Enable or disable antialiasing.
            * ``"ssaa"`` - Super-Sample Anti-Aliasing
            * ``"msaa"`` - Multi-Sample Anti-Aliasing
            * ``"fxaa"`` - Fast Approximate Anti-Aliasing

            .. Note::

                SSAA, or Super-Sample Anti-Aliasing is a brute force method of
                antialiasing. It results in the best image quality but comes at a
                tremendous resource cost. SSAA works by rendering the scene at a higher
                resolution. The final image is produced by downsampling the
                massive source image using an averaging filter. This acts as a low pass
                filter which removes the high frequency components that would cause
                jaggedness.

                MSAA, or Multi-Sample Anti-Aliasing is an optimization of SSAA that
                reduces the number of pixel shader evaluations that need to be computed
                by focusing on overlapping regions of the scene. The result is
                antialiasing along edges that are on par with SSAA and less
                antialiasing along surfaces as these make up the bulk of SSAA
                computations. MSAA is substantially less computationally expensive than
                SSAA and results in comparable image quality.

                FXAA, or Fast Approximate Anti-Aliasing is an Anti-Aliasing technique
                performed entirely in post-processing. FXAA operates on the
                rasterized image rather than the scene geometry. As a consequence,
                forcing FXAA or using FXAA incorrectly can result in the FXAA filter
                smoothing out parts of the visual overlay that are usually kept sharp
                for reasons of clarity as well as smoothing out textures. FXAA is
                inferior to MSAA but is almost free computationally and is thus
                desirable on high-end platforms.

        - msaa_multi_samples : int, optional, default=16
            The number of multi-samples when ``anti_aliasing`` is ``"msaa"``. Note
            that using this setting automatically enables this for all
            renderers.
        - smooth_shading : bool, optional,
            Smoothly render curved surfaces when plotting.  Not helpful
            for all meshes.
        - line_smoothing : bool, default:  True
            If ``True``, enable line smoothing.
        - polygon_smoothing : bool, default: True
            If ``True``, enable polygon smoothing.
        - lighting : bool, optional
            Enable or disable view direction lighting. Default False.
        - notebook : bool, optional
            When True, the resulting plot is placed inline a jupyter
            notebook.  Assumes a jupyter console is active.  Automatically
            enables off_screen.
        - jupyter_backend : str, optional, default: "trame"
            Jupyter backend to use when plotting.  Must be one of the following:

            * ``'static'``: Display a single static image within the
              Jupyterlab environment.  It Still requires that a virtual
              framebuffer be set up when displaying on a headless server,
              but does not require any additional modules to be installed.

            * ``'client'``: Export/serialize the scene graph to be rendered
              with VTK.js client-side through ``trame``. Requires ``trame``
              and ``jupyter-server-proxy`` to be installed.

            * ``'server'``: Render remotely and stream the resulting VTK
              images back to the client using ``trame``. This replaces the
              ``'ipyvtklink'`` backend with better performance.
              Supports the most VTK features, but suffers from minor lag due
              to remote rendering. Requires that a virtual framebuffer be set
              up when displaying on a headless server. Must have at least ``trame``
              and ``jupyter-server-proxy`` installed for cloud/remote Jupyter
              instances. This mode is also aliased by ``'trame'``.

            * ``'trame'``: The full Trame-based backend that combines both
              ``'server'`` and ``'client'`` into one backend. This requires a
              virtual frame buffer.

            * ``'html'``: Export/serialize the scene graph to be rendered
              with the Trame client backend but in a static HTML file.

            * ``'none'``: Do not display any plots within jupyterlab,
              instead display using dedicated VTK render windows.  This
              will generate nothing on headless servers even with a
              virtual framebuffer.

        - font_family : str, optional
            Font family.  Must be either ``'courier'``, ``'times'``,
            or ``arial``.
        - scale_factor : float, optional
            Scale factor between the maximum deformation of the model and the maximum boundary size.
            Default ``1 / 20``.
        - show_mesh_edges: bool, default: True
            Whether to display the mesh edges of ``planes``, ``plates``, ``shells``, and ``solid`` elements.
        - mesh_edge_color: str, default: black
            Color of the mesh edges for ``planes``, ``plates``, ``shells``, and ``solid`` elements.
        - mesh_edge_width: float, default: 1.0
            Width of the mesh edges for ``planes``, ``plates``, ``shells``, and ``solid`` elements.
        - mesh_opacity: float, default: 1.0
            Display opacity of ``surface`` and ``solid`` elements.
        - font_size: int, default: 15
            Font size of labels.
        - title_font_size: int, default: 18
            Font size of title.
        - off_screen: bool, optional
            Renders off-screen when True. Useful for automated screenshots.
        - scalar_bar_kargs: dict
            Arguments to pass to
            `Plotter.add_scalar_bar <https://docs.pyvista.org/api/plotting/_autosummary/pyvista.plotter.add_scalar_bar#pyvista.Plotter.add_scalar_bar>`_
            For example, ``dict(fmt="%.3e", n_labels=10)``.

    Returns
    -------
    None
    """
    if "point_size" in kwargs and abs(kwargs["point_size"]) < 1e-3:
        kwargs["point_size"] = 1e-5
    if kwargs.get("notebook"):
        if "jupyter_backend" in kwargs:
            pv.set_jupyter_backend(kwargs["jupyter_backend"])
        else:
            pv.set_jupyter_backend("trame")
    for key, value in kwargs.items():
        if key.lower() == "scalar_bar_kargs":
            getattr(PLOT_ARGS, key.lower()).update(value)
        else:
            setattr(PLOT_ARGS, key, value)


def set_plot_colors(**kwargs: Unpack[_PLOT_ARGS_TYPES2]) -> None:
    """
    Set the display color of various element types.

    Parameters
    ----------
    kwargs: optional keyword arguments, including:
        - point : str, list[int, int, int], optional
            Color for nodal points.
            Either a string, RGB list, or hex color string.  For example,
            ``point='white'``, ``point='w'``, ``point=[1, 1, 1]``, or
            ``point='#FFFFFF'``.
        - frame : str, list[int, int, int], optional
            Color for frame elements.
        - truss : str, list[int, int, int], optional
            Color for truss elements.
        - link : str, list[int, int, int], optional
            Color for link elements.
        - shell : str, list[int, int, int], optional
            Color for shell elements.
        - plane : str, list[int, int, int], optional
            Color for plane elements.
        - brick : str, list[int, int, int], optional
            Color for brick (solid) elements.
        - tet : str, list[int, int, int], optional
            Color for tetrahedral (solid) elements.
        - joint : str, list[int, int, int], optional
            Color for beam-column joint elements.
        - contact : str, list[int, int, int], optional
            Color for contact elements.
        - pfem : str, list[int, int, int], optional
            Color for PFEM elements.
        - constraint : str, list[int, int, int], optional
            Color for constraint.
        - bc : str, list[int, int, int], optional
            Color for boundary conditions.
        - cmap : str, list, optional
            Name of the Matplotlib colormap to us when mapping the
            scalars.  See available Matplotlib colormaps.  Only
            applicable for when displaying ``scalars``. Requires Matplotlib
            to be installed.  ``colormap`` is also an accepted alias for
            this. If ``colorcet`` or ``cmocean`` are installed, their
            colormaps can be specified by name.

            You can also specify a list of colors to override an
            existing colormap with a custom one.  For example, to
            create a three color colormap you might specify
            ``['green', 'red', 'blue']``.
        - cmap_model : str, list, optional, default=None
            Matplotlib colormap used for geometry model visualization.
            Same as ``cmap``, except that this parameter will be used
            for geometry model visualization and will be automatically mapped
            according to different element types.
            If None, If None, the color specified in the function``set_plot_colors``
            will be used.

            Available color maps are shown in
            `Colormaps in Matplotlib <https://matplotlib.org/stable/users/explain/colors/colormaps.html>`_
        - nodal_label: str, default="#048243"
            Color for nodal label.
        - ele_label: str, default="#650021"
            Color for element label.

    Returns
    -------
    None
    """
    for key, value in kwargs.items():
        if key in ["cmap", "cmap_model"]:
            setattr(PLOT_ARGS, key, value)
        else:
            setattr(PLOT_ARGS, "color_" + key, value)
    if "cmap" in kwargs:
        PLOT_ARGS.color_map = kwargs["cmap"]
    if "frame" in kwargs:
        PLOT_ARGS.color_beam = kwargs["frame"]


def _get_ele_color(ele_types: list[str]):
    if PLOT_ARGS.cmap_model:
        cmap = plt.get_cmap(PLOT_ARGS.cmap_model)
        colors = cmap(np.linspace(0, 1, len(ele_types)))
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


def _plot_points(
    plotter,
    pos,
    color: str = "black",
    size: float = 3.0,
    render_points_as_spheres: bool = True,
):
    point_plot = pv.PolyData(pos)
    plotter.add_mesh(
        point_plot,
        color=color,
        point_size=size,
        render_points_as_spheres=render_points_as_spheres,
    )
    return point_plot


def _plot_points_cmap(
    plotter,
    pos,
    scalars,
    cmap: str = "jet",
    size: float = 3.0,
    clim: Optional[list] = None,
    show_scalar_bar=False,
    render_points_as_spheres=True,
):
    point_plot = pv.PolyData(pos)
    point_plot["scalars"] = scalars  # auto to point_data or cell_data
    if clim is None:
        clim = (np.min(scalars), np.max(scalars))
    plotter.add_mesh(
        point_plot,
        colormap=cmap,
        scalars="scalars",
        interpolate_before_map=True,
        clim=clim,
        point_size=size,
        render_points_as_spheres=render_points_as_spheres,
        show_scalar_bar=show_scalar_bar,
    )
    return point_plot


def _plot_lines(plotter, pos, cells, width=1.0, color="blue", render_lines_as_tubes=True, label=None):
    if len(cells) == 0:
        return None
    line_plot = pv.PolyData()
    line_plot.points = pos
    line_plot.lines = cells
    plotter.add_mesh(
        line_plot,
        color=color,
        render_lines_as_tubes=render_lines_as_tubes,
        line_width=width,
        label=label,
    )
    return line_plot


def _plot_lines_cmap(
    plotter,
    pos,
    cells,
    scalars,
    cmap="jet",
    width=1.0,
    clim=None,
    render_lines_as_tubes=True,
    show_scalar_bar=False,
):
    if len(cells) == 0:
        return None
    line_plot = pv.PolyData()
    line_plot.points = pos
    line_plot.lines = cells
    line_plot["scalars"] = scalars
    if clim is None:
        clim = (np.min(scalars), np.max(scalars))
    plotter.add_mesh(
        line_plot,
        colormap=cmap,
        scalars="scalars",
        interpolate_before_map=True,
        clim=clim,
        render_lines_as_tubes=render_lines_as_tubes,
        line_width=width,
        show_scalar_bar=show_scalar_bar,
    )
    return line_plot


def _plot_face(
    plotter,
    pos,
    cells,
    color="green",
    show_edges=True,
    edge_color="black",
    edge_width=1.0,
    opacity=1.0,
    style="surface",
    label=None,
):
    """plot the face grid."""
    if len(cells) == 0:
        return None
    surf = pv.PolyData(pos, faces=cells)
    plotter.add_mesh(
        surf,
        color=color,
        show_edges=show_edges,
        edge_color=edge_color,
        line_width=edge_width,
        opacity=opacity,
        style=style,
        label=label,
    )
    return surf


def _plot_face_cmap(
    plotter,
    pos,
    cells,
    scalars,
    cmap="jet",
    color=None,
    clim=None,
    show_edges=True,
    edge_color="black",
    edge_width=1.0,
    opacity=1.0,
    style="surface",
    show_scalar_bar=False,
):
    if len(cells) == 0:
        return None
    surf = pv.PolyData(pos, faces=cells)
    surf["scalars"] = scalars
    if clim is None:
        clim = (np.min(scalars), np.max(scalars))
    color_args = (
        {"color": color, "cmap": None, "scalars": None} if color is not None else {"cmap": cmap, "scalars": "scalars"}
    )
    plotter.add_mesh(
        surf,
        clim=clim,
        show_edges=show_edges,
        edge_color=edge_color,
        line_width=edge_width,
        opacity=opacity,
        interpolate_before_map=True,
        style=style,
        show_scalar_bar=show_scalar_bar,
        render_lines_as_tubes=True,
        **color_args,
    )
    return surf


def _plot_unstru(
    plotter,
    pos,
    cells,
    cell_types,
    color="green",
    show_edges=True,
    edge_color="black",
    edge_width=1.0,
    opacity=1.0,
    style="surface",
    label=None,
):
    """plot the unstructured grid."""
    if len(cells) == 0:
        return None
    grid = pv.UnstructuredGrid(cells, cell_types, pos)
    plotter.add_mesh(
        grid,
        color=color,
        show_edges=show_edges,
        edge_color=edge_color,
        line_width=edge_width,
        opacity=opacity,
        style=style,
        label=label,
    )
    return grid


def _plot_unstru_cmap(
    plotter,
    pos,
    cells,
    cell_types,
    scalars,
    cmap="jet",
    clim=None,
    show_edges=True,
    edge_color="black",
    edge_width=1.0,
    opacity=1.0,
    style="surface",
    show_scalar_bar=False,
    show_origin=False,
    pos_origin=None,
):
    if len(cells) == 0:
        return None
    grid = pv.UnstructuredGrid(cells, cell_types, pos)
    # grid.point_data["data0"] = scalars
    grid["scalars"] = scalars
    if clim is None:
        clim = (np.min(scalars), np.max(scalars))
    plotter.add_mesh(
        grid,
        colormap=cmap,
        scalars="scalars",
        clim=clim,
        show_edges=show_edges,
        edge_color=edge_color,
        line_width=edge_width,
        opacity=opacity,
        interpolate_before_map=True,
        style=style,
        show_scalar_bar=show_scalar_bar,
    )
    if show_origin:
        _plot_unstru(
            plotter,
            pos_origin,
            cells,
            cell_types,
            color="gray",
            style="wireframe",
            edge_width=edge_width,
        )
    return grid


def _plot_all_mesh(
    plotter,
    pos,
    line_cells,
    unstru_cells,
    unstru_celltypes,
    color="gray",
    edge_width=1.0,
    render_lines_as_tubes=True,
):
    line_plot = _plot_lines(
        plotter,
        pos,
        line_cells,
        color=color,
        render_lines_as_tubes=render_lines_as_tubes,
    )
    unstru_plot = _plot_unstru(
        plotter,
        pos,
        unstru_cells,
        unstru_celltypes,
        color=color,
        style="wireframe",
        edge_width=edge_width,
    )
    return line_plot, unstru_plot


def _plot_all_mesh_cmap(
    plotter,
    pos,
    line_cells,
    unstru_cells,
    unstru_celltypes,
    scalars,
    cmap="jet",
    clim=None,
    show_edges=True,
    edge_color="black",
    edge_width=1.0,
    point_size=0,
    opacity=1.0,
    style="surface",
    lw=1.0,
    render_lines_as_tubes=True,
    render_points_as_spheres=True,
    show_scalar_bar=False,
    show_origin=False,
    pos_origin=None,
):
    if show_origin:
        _plot_lines(
            plotter,
            pos_origin,
            line_cells,
            width=edge_width,
            color="gray",
            render_lines_as_tubes=render_lines_as_tubes,
        )
        _plot_unstru(
            plotter,
            pos_origin,
            unstru_cells,
            unstru_celltypes,
            color="gray",
            style="wireframe",
            edge_width=edge_width,
        )
    if point_size > 0:
        point_plot = _plot_points_cmap(
            plotter,
            pos,
            scalars,
            cmap=cmap,
            clim=clim,
            size=point_size,
            show_scalar_bar=show_scalar_bar,
            render_points_as_spheres=render_points_as_spheres,
        )
    else:
        point_plot = None
    line_plot = _plot_lines_cmap(
        plotter,
        pos,
        line_cells,
        scalars,
        cmap=cmap,
        width=lw,
        clim=clim,
        render_lines_as_tubes=render_lines_as_tubes,
        show_scalar_bar=show_scalar_bar,
    )
    unstru_plot = _plot_unstru_cmap(
        plotter,
        pos,
        unstru_cells,
        unstru_celltypes,
        scalars,
        cmap=cmap,
        show_edges=show_edges,
        edge_color=edge_color,
        edge_width=edge_width,
        opacity=opacity,
        style=style,
        clim=clim,
        show_scalar_bar=show_scalar_bar,
    )
    return point_plot, line_plot, unstru_plot


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


def _update_point_label_actor(
    label_actor: vtk.vtkActor2D,
    coords,
    labels,
    text_property,
    renderer: vtk.vtkRenderer = None,
    *,
    shape_opacity: float = 0.0,
    always_visible: bool = True,
    tolerance: float = 0.001,
):
    if len(coords) != len(labels):
        raise ValueError("coords and labels must have the same length")  # noqa: TRY003

        # Create new vtkPoints and label array
    vtk_points = vtk.vtkPoints()
    vtk_labels = vtk.vtkStringArray()
    vtk_labels.SetName("labels")

    for pt, text in zip(coords, labels):
        vtk_points.InsertNextPoint(*pt)
        vtk_labels.InsertNextValue(str(text))

    polydata = vtk.vtkPolyData()
    polydata.SetPoints(vtk_points)
    polydata.GetPointData().AddArray(vtk_labels)

    # Construct label hierarchy
    hierarchy = vtk.vtkPointSetToLabelHierarchy()
    hierarchy.SetLabelArrayName("labels")

    if always_visible or renderer is None:
        hierarchy.SetInputData(polydata)
    else:
        visible_filter = vtk.vtkSelectVisiblePoints()
        visible_filter.SetInputData(polydata)
        visible_filter.SetRenderer(renderer)
        visible_filter.SetTolerance(tolerance)
        hierarchy.SetInputConnection(visible_filter.GetOutputPort())

    # Text style
    hierarchy.SetTextProperty(text_property)

    hierarchy.Update()

    # Replace the mapper's input connection
    mapper = label_actor.GetMapper()
    mapper.SetBackgroundOpacity(shape_opacity)
    mapper.SetInputConnection(hierarchy.GetOutputPort())


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
