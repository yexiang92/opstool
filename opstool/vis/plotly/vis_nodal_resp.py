from typing import Optional, Union

import numpy as np
import plotly.graph_objs as go

from ...post import loadODB
from .plot_resp_base import PlotResponseBase
from .plot_utils import (
    _get_line_cells,
    _get_unstru_cells,
    _plot_lines_cmap,
    _plot_points_cmap,
    _plot_unstru_cmap,
)


class PlotNodalResponse(PlotResponseBase):
    def __init__(
        self,
        model_info_steps,
        node_resp_steps,
        model_update,
    ):
        super().__init__(model_info_steps, node_resp_steps, model_update)
        self.FIGURE = go.Figure()

        self.resps_norm = None
        self.comp_type = None

        title = f"<b>{self.PKG_NAME} :: Nodal Responses 3D Viewer</b><br><br>"
        self.title = {"text": title, "font": {"size": self.pargs.title_font_size}}

    def set_comp_resp_type(self, resp_type, component):
        if resp_type.lower() in ["disp", "dispacement"]:
            self.resp_type = "disp"
        elif resp_type.lower() in ["vel", "velocity"]:
            self.resp_type = "vel"
        elif resp_type.lower() in ["accel", "acceleration"]:
            self.resp_type = "accel"
        elif resp_type.lower() in ["reaction", "reactionforce"]:
            self.resp_type = "reaction"
        elif resp_type.lower() in ["reactionincinertia", "reactionincinertiaforce"]:
            self.resp_type = "reactionIncInertia"
        elif resp_type.lower() in ["rayleighforces", "rayleigh"]:
            self.resp_type = "rayleighForces"
        elif resp_type.lower() in ["pressure"]:
            self.resp_type = "pressure"
        else:
            raise ValueError(  # noqa: TRY003
                f"Invalid response type: {resp_type}. "
                "Valid options are: disp, vel, accel, reaction, reactionIncInertia, rayleighForces, pressure."
            )
        if isinstance(component, str):
            self.component = component.upper()
        else:
            self.component = list(component)

    def _get_resp_clim_peak(self, idx="absMax"):
        resps = []
        for i in range(self.num_steps):
            da = self._get_resp_da(i, self.resp_type, self.component)
            resps.append(da)
        if self.ModelUpdate:
            resps_norm = resps if resps[0].ndim == 1 else [np.linalg.norm(resp, axis=1) for resp in resps]
        else:
            resps_norm = resps if resps[0].ndim == 1 else np.linalg.norm(resps, axis=2)
        if isinstance(idx, str):
            if idx.lower() == "absmax":
                resp = [np.max(np.abs(data)) for data in resps]
                step = np.argmax(resp)
            elif idx.lower() == "max":
                resp = [np.max(data) for data in resps]
                step = np.argmax(resp)
            elif idx.lower() == "absmin":
                resp = [np.min(np.abs(data)) for data in resps]
                step = np.argmin(resp)
            elif idx.lower() == "min":
                resp = [np.min(data) for data in resps]
                step = np.argmin(resp)
            else:
                raise ValueError("Invalid argument, one of [absMax, absMin, Max, Min]")  # noqa: TRY003
        else:
            step = int(idx)
        max_resps = [np.max(resp) for resp in resps_norm]
        min_resps = [np.min(resp) for resp in resps_norm]
        cmin, cmax = np.min(min_resps), np.max(max_resps)
        self.resps_norm = resps_norm
        self.clim = (cmin, cmax)
        return cmin, cmax, step

    def _make_txt(self, step, add_title=False):
        max_norm, min_norm = np.max(self.resps_norm[step]), np.min(self.resps_norm[step])
        title = "Nodal Responses"
        if self.resp_type == "disp":
            resp_type = "Displacement"
        elif self.resp_type == "vel":
            resp_type = "Velocity"
        elif self.resp_type == "accel":
            resp_type = "Acceleration"
        else:
            resp_type = f"{self.resp_type.capitalize()}"
        comp = ",".join(self.component) if isinstance(self.component, (list, tuple)) else self.component
        size_symbol = ("norm.min", "norm.max") if isinstance(self.component, (list, tuple)) else ("min", "max")

        t_ = self.time[step]
        title = f"<b>{self._set_txt_props(resp_type)} *</b><br>"
        title += f"<b>(DOF) {self._set_txt_props(comp)} *</b><br>"
        if self.unit_symbol:
            unit_txt = self._set_txt_props(self.unit_symbol)
            title += f"<b>(unit) {unit_txt}</b><br>"
        max_norm = self._set_txt_props(f"{max_norm:.3E}")
        min_norm = self._set_txt_props(f"{min_norm:.3E}")
        title += f"<b>{size_symbol[1]}:</b> {max_norm}<br><b>{size_symbol[0]}:</b> {min_norm}"
        step_txt = self._set_txt_props(f"{step}")
        title += f"<br><b>step:</b> {step_txt}; "
        t_txt = self._set_txt_props(f"{t_:.3f}")
        title += f"<b>time</b>: {t_txt}<br> <br>"
        if add_title:
            title = self.title["text"] + title
        txt = {
            "font": {"size": self.pargs.font_size},
            "text": title,
        }
        return txt

    def _create_mesh(
        self,
        plotter,
        value,
        alpha=1.0,
        clim=None,
        style="surface",
        coloraxis=None,
        show_origin=False,
        show_bc: bool = True,
        bc_scale: float = 1.0,
        show_mp_constraint: bool = True,
        show_hover: bool = True,
    ):
        step = round(value)
        line_cells, _ = _get_line_cells(self._get_line_da(step))
        _, unstru_cell_types, unstru_cells = _get_unstru_cells(self._get_unstru_da(step))
        node_defo_coords = np.array(self._get_defo_coord_da(step, alpha))
        node_resp = np.array(self._get_resp_da(step, self.resp_type, self.component))
        if self.resps_norm is not None:
            scalars = self.resps_norm[step]
        else:
            scalars = node_resp if node_resp.ndim == 1 else np.linalg.norm(node_resp, axis=1)

        if show_bc:
            self._plot_bc(plotter=plotter, step=step, defo_scale=alpha, bc_scale=bc_scale)
        if show_mp_constraint:
            self._plot_mp_constraint(plotter, step=step, defo_scale=alpha)
        if show_origin:
            self._plot_all_mesh(plotter=plotter, step=step)

        if len(unstru_cells) > 0:
            (
                face_points,
                face_line_points,
                face_mid_points,
                veci,
                vecj,
                veck,
                face_scalars,
                face_line_scalars,
            ) = self._get_plotly_unstru_data(node_defo_coords, unstru_cell_types, unstru_cells, scalars)
            _plot_unstru_cmap(
                plotter,
                face_points,
                veci=veci,
                vecj=vecj,
                veck=veck,
                scalars=face_scalars,
                clim=clim,
                coloraxis=coloraxis,
                style=style,
                line_width=self.pargs.line_width,
                opacity=self.pargs.mesh_opacity,
                show_edges=self.pargs.show_mesh_edges,
                edge_color=self.pargs.mesh_edge_color,
                edge_width=self.pargs.mesh_edge_width,
                edge_points=face_line_points,
                edge_scalars=face_line_scalars,
            )
        if len(line_cells) > 0:
            line_points, line_mid_points, line_scalars = self._get_plotly_line_data(
                node_defo_coords, line_cells, scalars
            )
            _plot_lines_cmap(
                plotter,
                line_points,
                scalars=line_scalars,
                coloraxis=coloraxis,
                clim=clim,
                width=self.pargs.line_width,
            )
        _plot_points_cmap(
            plotter,
            node_defo_coords,
            scalars=scalars,
            clim=clim,
            coloraxis=coloraxis,
            name=self.resp_type,
            size=self.pargs.point_size,
            show_hover=show_hover,
        )

    def plot_slide(
        self,
        alpha=1.0,
        show_defo=True,
        show_bc: bool = True,
        bc_scale: float = 1.0,
        show_mp_constraint: bool = True,
        style="surface",
        show_origin=False,
    ):
        cmin, cmax, _ = self._get_resp_clim_peak()
        clim = (cmin, cmax)
        alpha_ = alpha if show_defo else 0.0
        ndatas = []
        ndata_cum = 0
        for i in range(self.num_steps):
            plotter = []
            self._create_mesh(
                plotter,
                i,
                alpha=alpha_,
                clim=clim,
                coloraxis=f"coloraxis{i + 1}",
                show_bc=show_bc,
                bc_scale=bc_scale,
                show_mp_constraint=show_mp_constraint,
                style=style,
                show_origin=show_origin,
            )
            self.FIGURE.add_traces(plotter)
            ndatas.append(len(self.FIGURE.data) - ndata_cum)
            ndata_cum = len(self.FIGURE.data)
        self._update_slider_layout(ndatas=ndatas, clim=clim)

    def plot_peak_step(
        self,
        step="absMax",
        alpha=1.0,
        show_defo=True,
        show_bc: bool = True,
        bc_scale: float = 1.0,
        show_mp_constraint: bool = True,
        style="surface",
        show_origin=False,
    ):
        cmin, cmax, step = self._get_resp_clim_peak(idx=step)
        clim = (cmin, cmax)
        alpha_ = alpha if show_defo else 0.0
        plotter = []
        self._create_mesh(
            plotter=plotter,
            value=step,
            alpha=alpha_,
            clim=clim,
            coloraxis="coloraxis",
            show_bc=show_bc,
            bc_scale=bc_scale,
            show_mp_constraint=show_mp_constraint,
            style=style,
            show_origin=show_origin,
        )
        self.FIGURE.add_traces(plotter)
        txt = self._make_txt(step)
        self.FIGURE.update_layout(
            coloraxis={
                "colorscale": self.pargs.cmap,
                "cmin": cmin,
                "cmax": cmax,
                "colorbar": {"tickfont": {"size": self.pargs.font_size - 2}, "title": txt},
            },
            title=self.title,
        )

    def plot_anim(
        self,
        alpha=1.0,
        show_defo=True,
        framerate: Optional[int] = None,
        show_bc: bool = True,
        bc_scale: float = 1.0,
        show_mp_constraint: bool = True,
        style="surface",
        show_origin=False,
    ):
        if framerate is None:
            framerate = np.ceil(self.num_steps / 10)
        cmin, cmax, _ = self._get_resp_clim_peak()
        clim = (cmin, cmax)
        alpha_ = alpha if show_defo else 0.0
        nb_frames = self.num_steps
        duration = 1000 / framerate
        # -----------------------------------------------------------------------------
        # start plot
        frames = []
        for i in range(nb_frames):
            plotter = []
            self._create_mesh(
                plotter=plotter,
                value=i,
                alpha=alpha_,
                clim=clim,
                coloraxis="coloraxis",
                show_bc=show_bc,
                bc_scale=bc_scale,
                show_mp_constraint=show_mp_constraint,
                style=style,
                show_origin=show_origin,
                show_hover=False,
            )
            frames.append(go.Frame(data=plotter, name="step:" + str(i)))
        # Add data to be displayed before animation starts
        plotter0 = []
        self._create_mesh(
            plotter0,
            0,
            alpha=alpha_,
            coloraxis="coloraxis",
            show_bc=show_bc,
            bc_scale=bc_scale,
            show_mp_constraint=show_mp_constraint,
            style=style,
            show_origin=show_origin,
            show_hover=False,
        )
        self.FIGURE = go.Figure(frames=frames, data=plotter0)

        self._update_antimate_layout(duration=duration, cbar_title=self.resp_type.capitalize())


def plot_nodal_responses(
    odb_tag: Union[int, str] = 1,
    slides: bool = False,
    step: Union[int, str] = "absMax",
    scale: float = 1.0,
    show_defo: bool = True,
    resp_type: str = "disp",
    resp_dof: Union[list, tuple, str] = ("UX", "UY", "UZ"),
    unit_symbol: Optional[str] = None,
    unit_factor: Optional[float] = None,
    show_bc: bool = True,
    bc_scale: float = 1.0,
    show_mp_constraint: bool = False,
    show_undeformed: bool = False,
    style: str = "surface",
    show_outline: bool = False,
) -> go.Figure:
    """Visualizing Node Responses.

    Parameters
    ----------
    odb_tag: Union[int, str], default: 1
        Tag of output databases (ODB) to be visualized.
    slides: bool, default: False
        Display the response for each step in the form of a slideshow.
        Otherwise, show the step with the following ``step`` parameter.
    step: Union[int, str], default: "absMax"
        If slides = False, this parameter will be used as the step to plot.
        If str, Optional: [absMax, absMin, Max, Min].
        If int, this step will be demonstrated (counting from 0).
    scale: float, default: 1.0
        Scales the size of the deformation presentation.
    show_defo: bool, default: True
        Whether to display the deformed shape.
    resp_type: str, default: disp
        Type of response to be visualized.
        Optional: "disp", "vel", "accel", "reaction", "reactionIncInertia", "rayleighForces", "pressure".
    resp_dof: str, default: ("UX", "UY", "UZ")
        Component to be visualized.
        Optional: "UX", "UY", "UZ", "RX", "RY", "RZ".
        You can also pass on a list or tuple to display multiple dimensions, for example, ["UX", "UY"],
        ["UX", "UY", "UZ"], ["RX", "RY", "RZ"], ["RX", "RY"], ["RY", "RZ"], ["RX", "RZ"], and so on.

        .. Note::
            If the nodes include fluid pressure dof,
            such as those used for ...UP elements, the pore pressure should be extracted using ``resp_type="vel"``,
            and ``resp_dof="UZ"``.

    unit_symbol: str, default: None
        Unit symbol to be displayed in the plot.
        This feature is added since v1.0.15.
    unit_factor: float, default: None
        This feature is added since v1.0.15.
        The multiplier used to convert units.
        For example, if you want to visualize stress and the current data unit is kPa, you can set ``unit_symbol="kPa" and unit_factor=1.0``.
        If you want to visualize in MPa, you can set ``unit_symbol="MPa" and unit_factor=0.001``.
    show_bc: bool, default: True
        Whether to display boundary supports.
    bc_scale: float, default: 1.0
        Scale the size of boundary support display.
    show_mp_constraint: bool, default: False
        Whether to show multipoint (MP) constraint.
    show_undeformed: bool, default: False
        Whether to show the undeformed shape of the model.
        Set to False can improve the performance of the visualization.
    show_outline: bool, default: False
        Whether to display the outline of the model.
    style: str, default: surface
        Visualization mesh style of surfaces and solids.
        One of the following: style='surface', style='wireframe', style='points', style='points_gaussian'.
        Defaults to 'surface'. Note that 'wireframe' only shows a wireframe of the outer geometry.

    Returns
    -------
    fig: `plotly.graph_objects.Figure <https://plotly.com/python-api-reference/generated/plotly.graph_objects.Figure.html>`_
        You can use `fig.show()` to display,
        You can also use `fig.write_html("path/to/file.html")` to save as an HTML file, see
        `Interactive HTML Export in Python <https://plotly.com/python/interactive-html-export/>`_
    """
    model_info_steps, model_update, node_resp_steps = loadODB(odb_tag, resp_type="Nodal")
    plotbase = PlotNodalResponse(model_info_steps, node_resp_steps, model_update)
    plotbase.set_unit(symbol=unit_symbol, factor=unit_factor)
    plotbase.set_comp_resp_type(resp_type=resp_type, component=resp_dof)
    if slides:
        plotbase.plot_slide(
            alpha=scale,
            show_defo=show_defo,
            show_bc=show_bc,
            bc_scale=bc_scale,
            show_mp_constraint=show_mp_constraint,
            style=style,
            show_origin=show_undeformed,
        )
    else:
        plotbase.plot_peak_step(
            step=step,
            alpha=scale,
            show_defo=show_defo,
            show_bc=show_bc,
            bc_scale=bc_scale,
            show_mp_constraint=show_mp_constraint,
            style=style,
            show_origin=show_undeformed,
        )
    return plotbase.update_fig(show_outline=show_outline)


def plot_nodal_responses_animation(
    odb_tag: Union[int, str] = 1,
    framerate: Optional[int] = None,
    scale: float = 1.0,
    show_defo: bool = True,
    resp_type: str = "disp",
    resp_dof: Union[list, tuple, str] = ("UX", "UY", "UZ"),
    unit_symbol: Optional[str] = None,
    unit_factor: Optional[float] = None,
    show_bc: bool = True,
    bc_scale: float = 1.0,
    show_mp_constraint: bool = False,
    show_undeformed: bool = False,
    style: str = "surface",
    show_outline: bool = False,
) -> go.Figure:
    """Visualize node response animation.

    Parameters
    ----------
    odb_tag: Union[int, str], default: 1
        Tag of output databases (ODB) to be visualized.
    framerate: int, default: 5
        Framerate for the display, i.e., the number of frames per second.
        For example, if an earthquake analysis has 1000 steps and you want to complete the demonstration in ten seconds, you should set ``framerate = 1000/10 = 100``.
    scale: float, default: 1.0
        Scales the size of the deformation presentation.
    show_defo: bool, default: True
        Whether to display the deformed shape.
    resp_type: str, default: disp
        Type of response to be visualized.
        Optional: "disp", "vel", "accel", "reaction", "reactionIncInertia", "rayleighForces", "pressure".
    resp_dof: str, default: ("UX", "UY", "UZ")
        Component to be visualized.
        Optional: "UX", "UY", "UZ", "RX", "RY", "RZ".
        You can also pass on a list or tuple to display multiple dimensions, for example, ["UX", "UY"],
        ["UX", "UY", "UZ"], ["RX", "RY", "RZ"], ["RX", "RY"], ["RY", "RZ"], ["RX", "RZ"], and so on.
    unit_symbol: str, default: None
        Unit symbol to be displayed in the plot.
        This feature is added since v1.0.15.
    unit_factor: float, default: None
        This feature is added since v1.0.15.
        The multiplier used to convert units.
        For example, if you want to visualize stress and the current data unit is kPa, you can set ``unit_symbol="kPa" and unit_factor=1.0``.
        If you want to visualize in MPa, you can set ``unit_symbol="MPa" and unit_factor=0.001``.
    show_bc: bool, default: True
        Whether to display boundary supports.
    bc_scale: float, default: 1.0
        Scale the size of boundary support display.
    show_mp_constraint: bool, default: False
        Whether to show multipoint (MP) constraint.
    show_undeformed: bool, default: False
        Whether to show the undeformed shape of the model.
        Set to False can improve the performance of the visualization.
    show_outline: bool, default: False
        Whether to display the outline of the model.
    style: str, default: surface
        Visualization mesh style of surfaces and solids.
        One of the following: style='surface' or style='wireframe'
        Defaults to 'surface'. Note that 'wireframe' only shows a wireframe of the outer geometry.

    Returns
    -------
    fig: `plotly.graph_objects.Figure <https://plotly.com/python-api-reference/generated/plotly.graph_objects.Figure.html>`_
        You can use `fig.show()` to display,
        You can also use `fig.write_html("path/to/file.html")` to save as an HTML file, see
        `Interactive HTML Export in Python <https://plotly.com/python/interactive-html-export/>`_
    """
    model_info_steps, model_update, node_resp_steps = loadODB(odb_tag, resp_type="Nodal")
    plotbase = PlotNodalResponse(model_info_steps, node_resp_steps, model_update)
    plotbase.set_unit(symbol=unit_symbol, factor=unit_factor)
    plotbase.set_comp_resp_type(resp_type=resp_type, component=resp_dof)
    plotbase.plot_anim(
        alpha=scale,
        show_defo=show_defo,
        framerate=framerate,
        show_bc=show_bc,
        bc_scale=bc_scale,
        show_mp_constraint=show_mp_constraint,
        style=style,
        show_origin=show_undeformed,
    )
    return plotbase.update_fig(show_outline=show_outline)
