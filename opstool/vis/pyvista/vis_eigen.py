from functools import partial
from typing import Optional, Union

import numpy as np
import pyvista as pv

from ...post import load_eigen_data, load_linear_buckling_data
from ...utils import CONFIGS
from .plot_resp_base import slider_widget_args
from .plot_utils import (
    PLOT_ARGS,
    _get_line_cells,
    _get_unstru_cells,
    _plot_all_mesh_cmap,
)
from .vis_model import PlotModelBase

PKG_NAME = CONFIGS.get_pkg_name()
SHAPE_MAP = CONFIGS.get_shape_map()


class PlotEigenBase:
    def __init__(self, model_info, modal_props, eigen_vectors):
        self.nodal_data = model_info["NodalData"]
        self.nodal_tags = self.nodal_data.coords["nodeTags"]
        self.points = self.nodal_data.to_numpy()
        self.ndims = self.nodal_data.attrs["ndims"]
        self.bounds = self.nodal_data.attrs["bounds"]
        self.min_bound_size = self.nodal_data.attrs["minBoundSize"]
        self.max_bound_size = self.nodal_data.attrs["maxBoundSize"]
        self.show_zaxis = not np.max(self.ndims) <= 2
        self.slider_widget_args = slider_widget_args
        # -------------------------------------------------------------
        self.line_data = model_info["AllLineElesData"]
        self.line_cells, self.line_tags = _get_line_cells(self.line_data)
        # -------------------------------------------------------------
        self.unstru_data = model_info["UnstructuralData"]
        self.unstru_tags, self.unstru_cell_types, self.unstru_cells = _get_unstru_cells(self.unstru_data)
        # --------------------------------------------------
        self.pargs = PLOT_ARGS
        self.ModelInfo = model_info
        self.ModalProps = modal_props
        self.EigenVectors = eigen_vectors
        self.plot_model_base = PlotModelBase(model_info, {})
        pv.set_plot_theme(PLOT_ARGS.theme)

    def _get_eigen_points(self, step, alpha):
        eigen_vec = self.EigenVectors.to_numpy()[..., :3][step]
        value_ = np.max(np.sqrt(np.sum(eigen_vec**2, axis=1)))
        alpha_ = self.max_bound_size * self.pargs.scale_factor / value_
        alpha_ = alpha_ * alpha if alpha else alpha_
        eigen_points = self.points + eigen_vec * alpha_
        scalars = np.sqrt(np.sum(eigen_vec**2, axis=1))
        return eigen_points, scalars, alpha_

    def _get_bc_points(self, step, scale: float):
        fixed_node_data = self.ModelInfo["FixedNodalData"]
        if len(fixed_node_data) > 0:
            fix_tags = fixed_node_data["nodeTags"].values
            coords = self.nodal_data.sel({"nodeTags": fix_tags}).to_numpy()
            eigen_vec = self.EigenVectors.sel({"nodeTags": fix_tags}).to_numpy()
            vec = eigen_vec[..., :3][step]
            coords = coords + vec * scale
        else:
            coords = []
        return coords

    def _make_eigen_txt(self, step):
        fi = self.ModalProps.loc[:, "eigenFrequency"][step]
        txt = f"Mode {step + 1}\nperiod: {1 / fi:.6f} s; freq: {fi:.6f} Hz\n"
        if not self.show_zaxis:
            txt += "modal participation mass ratios (%)\n"
            mx = self.ModalProps.loc[:, "partiMassRatiosMX"][step]
            my = self.ModalProps.loc[:, "partiMassRatiosMY"][step]
            rmz = self.ModalProps.loc[:, "partiMassRatiosRMZ"][step]
            txt += f"{mx:7.3f} {my:7.3f} {rmz:7.3f}\n"
            txt += "cumulative modal participation mass ratios (%)\n"
            mx = self.ModalProps.loc[:, "partiMassRatiosCumuMX"][step]
            my = self.ModalProps.loc[:, "partiMassRatiosCumuMY"][step]
            rmz = self.ModalProps.loc[:, "partiMassRatiosCumuRMZ"][step]
            txt += f"{mx:7.3f} {my:7.3f} {rmz:7.3f}\n"
            txt += "{:>7} {:>7} {:>7}\n".format("X", "Y", "RZ")
        else:
            txt += "modal participation mass ratios (%)\n"
            mx = self.ModalProps.loc[:, "partiMassRatiosMX"][step]
            my = self.ModalProps.loc[:, "partiMassRatiosMY"][step]
            mz = self.ModalProps.loc[:, "partiMassRatiosMZ"][step]
            rmx = self.ModalProps.loc[:, "partiMassRatiosRMX"][step]
            rmy = self.ModalProps.loc[:, "partiMassRatiosRMY"][step]
            rmz = self.ModalProps.loc[:, "partiMassRatiosRMZ"][step]
            txt += f"{mx:7.3f} {my:7.3f} {mz:7.3f} {rmx:7.3f} {rmy:7.3f} {rmz:7.3f}\n"
            txt += "cumulative modal participation mass ratios (%)\n"
            mx = self.ModalProps.loc[:, "partiMassRatiosCumuMX"][step]
            my = self.ModalProps.loc[:, "partiMassRatiosCumuMY"][step]
            mz = self.ModalProps.loc[:, "partiMassRatiosCumuMZ"][step]
            rmx = self.ModalProps.loc[:, "partiMassRatiosCumuRMX"][step]
            rmy = self.ModalProps.loc[:, "partiMassRatiosCumuRMY"][step]
            rmz = self.ModalProps.loc[:, "partiMassRatiosCumuRMZ"][step]
            txt += f"{mx:7.3f} {my:7.3f} {mz:7.3f} {rmx:7.3f} {rmy:7.3f} {rmz:7.3f}\n"
            txt += "{:>7} {:>7} {:>7} {:>7} {:>7} {:>7}\n".format("X", "Y", "Z", "RX", "RY", "RZ")
        return txt

    def _make_eigen_subplots_txt(self, step):
        period = 1 / self.ModalProps.loc[:, "eigenFrequency"][step]
        txt = f"Mode {step + 1}  T = {period:.3E} s" if period < 0.001 else f"Mode {step + 1}  T = {period:.3f} s"
        return txt

    def _create_mesh(
        self,
        plotter,
        idx,
        idxi=None,
        idxj=None,
        alpha=1.0,
        style="surface",
        show_outline=False,
        show_origin=False,
        show_bc: bool = True,
        bc_scale: float = 1.0,
        show_mp_constraint: bool = True,
        cpos="iso",
    ):
        if idxi is not None and idxj is not None:
            plotter.subplot(idxi, idxj)
            subplots = True
        else:
            plotter.clear_actors()
            subplots = False
        step = round(idx) - 1
        eigen_points, scalars, alpha_ = self._get_eigen_points(step, alpha)
        point_plot, line_plot, solid_plot = _plot_all_mesh_cmap(
            plotter,
            eigen_points,
            self.line_cells,
            self.unstru_cells,
            self.unstru_cell_types,
            scalars=scalars,
            cmap=self.pargs.cmap,
            clim=None,
            lw=self.pargs.line_width,
            show_edges=self.pargs.show_mesh_edges,
            edge_color=self.pargs.mesh_edge_color,
            edge_width=self.pargs.mesh_edge_width,
            opacity=self.pargs.mesh_opacity,
            style=style,
            show_scalar_bar=False,
            point_size=self.pargs.point_size,
            render_lines_as_tubes=self.pargs.render_lines_as_tubes,
            render_points_as_spheres=self.pargs.render_lines_as_tubes,
            show_origin=show_origin,
            pos_origin=self.points,
        )
        if not subplots:
            txt = self._make_eigen_txt(step)
            plotter.add_text(
                txt,
                position="lower_right",
                font_size=self.pargs.font_size,
                font="courier",
            )
        else:
            txt = self._make_eigen_subplots_txt(step)
            plotter.add_text(
                txt,
                position="upper_left",
                font_size=self.pargs.font_size,
                font="courier",
            )
            # txt = self._make_eigen_txt(step)
            # plotter.add_text(txt, position="lower_right", font_size=label_size, font="courier")
        bc_plot = None
        if show_bc:
            bc_points = self._get_bc_points(step, scale=alpha_)
            bc_plot = self.plot_model_base.plot_bc(plotter, bc_scale, points_new=bc_points)
        mp_plot = None
        if show_mp_constraint:
            mp_plot = self.plot_model_base.plot_mp_constraint(plotter, points_new=eigen_points)
        if show_outline:
            self._plot_outline(plotter)
        self._update_plotter(plotter, cpos)
        return point_plot, line_plot, solid_plot, alpha_, bc_plot, mp_plot

    def subplots(self, plotter, modei, modej, link_views=True, **kargs):
        if modej - modei + 1 > 64:
            raise ValueError("When subplots True, mode_tag range must < 64 for clarify")  # noqa: TRY003
        shape = SHAPE_MAP[modej - modei + 1]
        for i, idx in enumerate(range(modei, modej + 1)):
            idxi = int(np.ceil((i + 1) / shape[1]) - 1)
            idxj = int(i - idxi * shape[1])
            self._create_mesh(plotter, idx, idxi, idxj, **kargs)
        if link_views:
            plotter.link_views()

    def plot_slides(self, plotter, modei, modej, **kargs):
        plotter.add_slider_widget(
            partial(self._create_mesh, plotter, **kargs), [modei, modej], value=modei, **self.slider_widget_args
        )

    def plot_anim(
        self,
        plotter,
        mode_tag: int = 1,
        n_cycle: int = 5,
        framerate: int = 3,
        savefig: str = "EigenAnimation.gif",
        alpha: float = 1.0,
        **kargs,
    ):
        # animation
        if savefig.endswith(".gif"):
            plotter.open_gif(savefig, fps=framerate, palettesize=64)
        else:
            plotter.open_movie(savefig, framerate=framerate, quality=7)
        alphas = [0.0] + [alpha, -alpha] * n_cycle
        for alpha in alphas:
            self._create_mesh(plotter, mode_tag, alpha=alpha, **kargs)
            plotter.write_frame()

    def _update_plotter(self, plotter: pv.Plotter, cpos):
        if isinstance(cpos, str):
            cpos = cpos.lower()
            viewer = {
                "xy": plotter.view_xy,
                "yx": plotter.view_yx,
                "xz": plotter.view_xz,
                "zx": plotter.view_zx,
                "yz": plotter.view_yz,
                "zy": plotter.view_zy,
                "iso": plotter.view_isometric,
            }
            if not self.show_zaxis and cpos not in ["xy", "yx"]:
                cpos = "xy"
                plotter.enable_2d_style()
                plotter.enable_parallel_projection()
            viewer[cpos]()

            if cpos == "iso":  # rotate camera
                plotter.camera.Azimuth(180)
        else:
            plotter.camera_position = cpos
            if not self.show_zaxis:
                plotter.view_xy()
                plotter.enable_2d_style()
                plotter.enable_parallel_projection()

        plotter.add_axes()
        return plotter


class PlotBucklingBase(PlotEigenBase):
    def __init__(self, model_info, eigen_values, eigen_vectors):
        super().__init__(model_info, eigen_values, eigen_vectors)

    def _make_eigen_txt(self, step):
        fi = self.ModalProps.isel(modeTags=step)
        return f"Mode {step + 1}\nk={float(fi):.3f}"

    def _make_eigen_subplots_txt(self, step):
        return self._make_eigen_txt(step)


def plot_eigen(
    mode_tags: Union[list, tuple, int],
    odb_tag: Optional[Union[int, str]] = None,
    subplots: bool = False,
    link_views: bool = True,
    scale: float = 1.0,
    show_outline: bool = False,
    show_origin: bool = False,
    style: str = "surface",
    cpos: str = "iso",
    show_bc: bool = True,
    bc_scale: float = 1.0,
    show_mp_constraint: bool = True,
    solver: str = "-genBandArpack",
    mode: str = "eigen",
) -> pv.Plotter:
    """Modal visualization.

    Parameters
    ----------
    mode_tags: Union[List, Tuple]
        The modal range to visualize, [mode i, mode j].
    odb_tag: Union[int, str], default: None
        Tag of output databases (ODB) to be visualized.
        If None, data will be saved automatically.
    subplots: bool, default: False
        If True, multiple subplots are used to present mode i to mode j.
        Otherwise, they are presented as slides.
    link_views: bool, default: True
        Link the views' cameras when subplots=True.
    scale: float, default: 1.0
        Zoom the presentation size of the mode shapes.
    show_outline: bool, default: False
        Whether to display the outline of the model.
    show_origin: bool, default: False
        Whether to show the undeformed shape.
    style: str, default: surface
        Visualization the mesh style of surfaces and solids.
        One of the following: style='surface', style='wireframe', style='points', style='points_gaussian'.
        Defaults to 'surface'. Note that 'wireframe' only shows a wireframe of the outer geometry.
    cpos: str, default: iso
        Model display perspective, optional: "iso", "xy", "yx", "xz", "zx", "yz", "zy".
        If 3d, defaults to "iso". If 2d, defaults to "xy".
    show_bc: bool, default: True
        Whether to display boundary supports.
    bc_scale: float, default: 1.0
        Scale the size of boundary support display.
    show_mp_constraint: bool, default: True
        Whether to show multipoint (MP) constraint.
    solver : str, optional,
        OpenSees' eigenvalue analysis solver, by default "-genBandArpack".
    mode: str, default: eigen
        The type of modal analysis, can be "eigen" or "buckling".
        If "eigen", it will plot the eigenvalues and eigenvectors.
        If "buckling", it will plot the buckling factors and modes.
        Added in v0.1.15.

    Returns
    -------
    Plotting object of PyVista to display vtk meshes or numpy arrays.
    See `pyvista.Plotter <https://docs.pyvista.org/api/plotting/_autosummary/pyvista.plotter>`_.

    You can use
    `Plotter.show <https://docs.pyvista.org/api/plotting/_autosummary/pyvista.plotter.show#pyvista.Plotter.show>`_.
    to display the plotting window.

    You can also use
    `Plotter.export_html <https://docs.pyvista.org/api/plotting/_autosummary/pyvista.plotter.export_html#pyvista.Plotter.export_html>`_.
    to export this plotter as an interactive scene to an HTML file.
    """
    if isinstance(mode_tags, int):
        mode_tags = [1, mode_tags]
    modei, modej = int(mode_tags[0]), int(mode_tags[1])
    if mode.lower() == "eigen":
        resave = odb_tag is None
        odb_tag = "Auto" if odb_tag is None else odb_tag
        modalProps, eigenvectors, MODEL_INFO = load_eigen_data(
            odb_tag=odb_tag, mode_tag=mode_tags[-1], solver=solver, resave=resave
        )
        plotbase = PlotEigenBase(MODEL_INFO, modalProps, eigenvectors)
    elif mode.lower() == "buckling":
        modalProps, eigenvectors, MODEL_INFO = load_linear_buckling_data(odb_tag=odb_tag)
        plotbase = PlotBucklingBase(MODEL_INFO, modalProps, eigenvectors)
    else:
        raise ValueError(f"Unsupported mode: {mode}. Use 'eigen' or 'buckling'.")  # noqa: TRY003
    if subplots:
        shape = SHAPE_MAP[modej - modei + 1]
        plotter = pv.Plotter(
            notebook=PLOT_ARGS.notebook,
            shape=shape,
            line_smoothing=PLOT_ARGS.line_smoothing,
            polygon_smoothing=PLOT_ARGS.polygon_smoothing,
            off_screen=PLOT_ARGS.off_screen,
        )
        plotbase.subplots(
            plotter,
            modei,
            modej,
            link_views=link_views,
            alpha=scale,
            style=style,
            show_outline=show_outline,
            show_origin=show_origin,
            show_bc=show_bc,
            bc_scale=bc_scale,
            show_mp_constraint=show_mp_constraint,
        )
    else:
        plotter = pv.Plotter(
            notebook=PLOT_ARGS.notebook,
            line_smoothing=PLOT_ARGS.line_smoothing,
            polygon_smoothing=PLOT_ARGS.polygon_smoothing,
        )
        plotbase.plot_slides(
            plotter,
            modei,
            modej,
            alpha=scale,
            style=style,
            show_outline=show_outline,
            show_origin=show_origin,
            show_bc=show_bc,
            bc_scale=bc_scale,
            show_mp_constraint=show_mp_constraint,
        )
    if PLOT_ARGS.anti_aliasing:
        plotter.enable_anti_aliasing(PLOT_ARGS.anti_aliasing)
    return plotbase._update_plotter(plotter, cpos)


def plot_eigen_animation(
    mode_tag: int,
    odb_tag: Optional[Union[int, str]] = None,
    n_cycle: int = 5,
    framerate: int = 3,
    savefig: str = "EigenAnimation.gif",
    off_screen: bool = True,
    cpos: str = "iso",
    solver: str = "-genBandArpack",
    alpha: float = 1.0,
    mode: str = "eigen",
    **kargs,
) -> pv.Plotter:
    """Modal animation visualization.

    Parameters
    ----------
    mode_tag: int
        The mode tag to display.
    odb_tag: Union[int, str], default: None
        Tag of output databases (ODB) to be visualized.
        If None, data will be saved automatically.
    n_cycle: int, default: five
        Number of cycles for the display.
    framerate: int, default: three
        Framerate for the display, i.e., the number of frames per second.
    savefig: str, default: EigenAnimation.gif
        Path to save the animation. The suffix can be ``.gif`` or ``.mp4``.
    off_screen: bool, default: True
        Whether to display the plotting window.
        If True, the plotting window will not be displayed.
    cpos: str, default: iso
        Model display perspective, optional: "iso", "xy", "yx", "xz", "zx", "yz", "zy".
        If 3d, defaults to "iso". If 2d, defaults to "xy".
    solver : str, optional,
        OpenSees' eigenvalue analysis solver, by default "-genBandArpack".
    alpha: float, default: 1.0
        Zoom the presentation size of the mode shapes.
    mode: str, default: eigen
        The type of modal analysis, can be "eigen" or "buckling".
        If "eigen", it will plot the eigenvalues and eigenvectors.
        If "buckling", it will plot the buckling factors and modes.
        Added in v0.1.15.
    kargs: dict, optional parameters,
        see ``plot_eigen``.

    Returns
    -------
    Plotting object of PyVista to display vtk meshes or numpy arrays.
    See `pyvista.Plotter <https://docs.pyvista.org/api/plotting/_autosummary/pyvista.plotter>`_.

    You can use
    `Plotter.show <https://docs.pyvista.org/api/plotting/_autosummary/pyvista.plotter.show#pyvista.Plotter.show>`_.
    to display the plotting window.

    You can also use
    `Plotter.export_html <https://docs.pyvista.org/api/plotting/_autosummary/pyvista.plotter.export_html#pyvista.Plotter.export_html>`_.
    to export this plotter as an interactive scene to an HTML file.
    """
    if mode.lower() == "eigen":
        resave = odb_tag is None
        modalProps, eigenvectors, MODEL_INFO = load_eigen_data(
            odb_tag=odb_tag, mode_tag=mode_tag, solver=solver, resave=resave
        )
        plotbase = PlotEigenBase(MODEL_INFO, modalProps, eigenvectors)
    elif mode.lower() == "buckling":
        modalProps, eigenvectors, MODEL_INFO = load_linear_buckling_data(odb_tag=odb_tag)
        plotbase = PlotBucklingBase(MODEL_INFO, modalProps, eigenvectors)
    else:
        raise ValueError(f"Unsupported mode: {mode}. Use 'eigen' or 'buckling'.")  # noqa: TRY003
    plotter = pv.Plotter(
        notebook=PLOT_ARGS.notebook,
        line_smoothing=PLOT_ARGS.line_smoothing,
        polygon_smoothing=PLOT_ARGS.polygon_smoothing,
        off_screen=off_screen,
    )
    plotbase.plot_anim(
        plotter,
        mode_tag,
        n_cycle=n_cycle,
        framerate=framerate,
        savefig=savefig,
        alpha=alpha,
        **kargs,
    )
    if PLOT_ARGS.anti_aliasing:
        plotter.enable_anti_aliasing(PLOT_ARGS.anti_aliasing)
    print(f"Animation has been saved to {savefig}!")
    return plotbase._update_plotter(plotter, cpos)
