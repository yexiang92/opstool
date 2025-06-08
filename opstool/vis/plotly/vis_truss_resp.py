from typing import Optional, Union

import numpy as np
import plotly.graph_objs as go

from ...post import loadODB
from ...utils import gram_schmidt
from .plot_resp_base import PlotResponseBase
from .plot_utils import _plot_lines, _plot_points_cmap, _plot_unstru_cmap


class PlotTrussResponse(PlotResponseBase):
    def __init__(self, model_info_steps, truss_resp_step, model_update):
        super().__init__(model_info_steps, truss_resp_step, model_update)

        title = f"{self.PKG_NAME} :: Truss Responses 3D Viewer</b><br><br>"
        self.title = {"text": title, "font": {"size": self.pargs.title_font_size}}

    def _get_truss_data(self, step):
        return self._get_model_da("TrussData", step)

    def _set_resp_type(self, resp_type: str):
        if resp_type.lower() in ["axialforce", "force", "forces"]:
            resp_type = "axialForce"
        elif resp_type.lower() in ["axialdefo", "axialdeformation", "deformation", "deformations"]:
            resp_type = "axialDefo"
        elif resp_type.lower() in ["stress", "stresses", "axialstress", "axialstresses"]:
            resp_type = "Stress"
        elif resp_type.lower() in ["strain", "strains", "axialstrain", "axialstrains"]:
            resp_type = "Strain"
        else:
            raise ValueError(  # noqa: TRY003
                f"Not supported response type {resp_type}! Valid options are: axialForce, axialDefo, Stress, Strain."
            )
        self.resp_type = resp_type

    def _make_truss_info(self, ele_tags, step):
        pos = self._get_node_da(step).to_numpy()
        truss_data = self._get_truss_data(step)
        if ele_tags is None:
            truss_tags = truss_data.coords["eleTags"].values
            truss_cells = truss_data.to_numpy().astype(int)
        else:
            truss_tags = np.atleast_1d(ele_tags)
            truss_cells = truss_data.sel(eleTags=truss_tags).to_numpy().astype(int)
        truss_node_coords = []
        truss_cells_new = []
        for i, cell in enumerate(truss_cells):
            nodei, nodej = cell[1:]
            truss_node_coords.append(pos[nodei])
            truss_node_coords.append(pos[nodej])
            truss_cells_new.append([2, 2 * i, 2 * i + 1])
        truss_node_coords = np.array(truss_node_coords)
        return truss_tags, truss_node_coords, truss_cells_new

    def refactor_resp_step(self, resp_type: str, ele_tags):
        self._set_resp_type(resp_type)
        resps = []
        if self.ModelUpdate or ele_tags is not None:
            for i in range(self.num_steps):
                truss_tags, _, _ = self._make_truss_info(ele_tags, i)
                da = self._get_resp_da(i, self.resp_type)
                da = da.sel(eleTags=truss_tags)
                resps.append(da)
        else:
            for i in range(self.num_steps):
                da = self._get_resp_da(i, self.resp_type)
                resps.append(da)
        self.resp_step = resps  # update

    def _get_resp_peak(self, idx="absMax"):
        if isinstance(idx, str):
            if idx.lower() == "absmax":
                resp = [np.max(np.abs(data)) for data in self.resp_step]
                step = np.argmax(resp)
            elif idx.lower() == "max":
                resp = [np.max(data) for data in self.resp_step]
                step = np.argmax(resp)
            elif idx.lower() == "absmin":
                resp = [np.min(np.abs(data)) for data in self.resp_step]
                step = np.argmin(resp)
            elif idx.lower() == "min":
                resp = [np.min(data) for data in self.resp_step]
                step = np.argmin(resp)
            else:
                raise ValueError("Invalid argument, one of [absMax, absMin, Max, Min]")  # noqa: TRY003
        else:
            step = int(idx)
        resp = self.resp_step[step]
        maxv = np.amax(np.abs(resp))
        alpha_ = 0.0 if maxv == 0 else self.max_bound_size * self.pargs.scale_factor / maxv
        cmin, cmax = self._get_truss_resp_clim()
        self.clim = (cmin, cmax)
        return step, (cmin, cmax), float(alpha_)

    def _get_truss_resp_clim(self):
        maxv = [np.max(data) for data in self.resp_step]
        minv = [np.min(data) for data in self.resp_step]
        cmin, cmax = np.min(minv), np.max(maxv)
        return cmin, cmax

    def _get_mesh_data(self, step, ele_tags, alpha):
        truss_tags, truss_coords, truss_cells = self._make_truss_info(ele_tags, step)
        resps = self.resp_step[step].to_numpy()
        resp_points, resp_cells, resp_celltypes = [], [], []
        scalars = []
        n_segs = 9
        for cell, resp in zip(truss_cells, resps):
            if resp == 0:
                continue
            else:
                coord1 = truss_coords[cell[1]]
                coord2 = truss_coords[cell[2]]
                xaxis = coord2 - coord1
                length = np.linalg.norm(xaxis)
                xaxis = xaxis / length
                cos_theta = np.dot(xaxis, [0, 0, 1])
                if 1 - cos_theta**2 < 1e-4:
                    axis_up = [1, 0, 0]
                elif self.show_zaxis:
                    axis_up = [0, 0, 1]
                else:
                    axis_up = [0, 1, 0]
                _, plot_axis, _ = gram_schmidt(xaxis, axis_up)
                coord3 = coord1 + alpha * resp * plot_axis
                coord4 = coord2 + alpha * resp * plot_axis
                coord_upper = [coord3 + length * i * xaxis / n_segs for i in range(n_segs)]
                coord_upper += [coord4]
                coord_lower = [coord1 + length * i * xaxis / n_segs for i in range(n_segs)]
                coord_lower += [coord2]
                for i in range(len(coord_upper) - 1):
                    resp_cells.append([
                        4,
                        len(resp_points),
                        len(resp_points) + 1,
                        len(resp_points) + 2,
                        len(resp_points) + 3,
                    ])
                    resp_points.extend([coord_lower[i], coord_lower[i + 1], coord_upper[i + 1], coord_upper[i]])
                    scalars.extend([resp, resp, resp, resp])
                    resp_celltypes.append(9)  # 9 for quad
        resp_points = np.array(resp_points)
        scalars = np.array(scalars)
        return truss_coords, truss_cells, resp_points, resp_cells, resp_celltypes, scalars

    def _create_mesh(
        self,
        plotter,
        value,
        ele_tags=None,
        show_values=True,
        plot_all_mesh=True,
        clim=None,
        coloraxis="coloraxis",
        alpha=1.0,
        line_width=1.5,
        style="surface",
        opacity=1.0,
        show_bc: bool = True,
        bc_scale: float = 1.0,
        show_mp_constraint: bool = True,
        color=None,
    ):
        step = round(value)
        coords, cells, resp_points, resp_cells, resp_celltypes, scalars = self._get_mesh_data(step, ele_tags, alpha)

        #  ---------------------------------
        if plot_all_mesh:
            self._plot_all_mesh(plotter, step=step)
        # truss response
        (
            face_points,
            face_line_points,
            face_mid_points,
            veci,
            vecj,
            veck,
            face_scalars,
            face_line_scalars,
        ) = self._get_plotly_unstru_data(resp_points, resp_celltypes, resp_cells, scalars, scalars_by_element=False)
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
            line_width=line_width,
            opacity=opacity,
            show_edges=False,
            edge_points=face_line_points,
            edge_scalars=face_line_scalars,
            color=color,
        )
        _plot_points_cmap(
            plotter,
            resp_points,
            scalars=scalars,
            clim=clim,
            coloraxis=coloraxis,
            name=self.resp_type,
            size=self.pargs.point_size,
            show_hover=show_values,
            color=color,
        )

        # truss lines
        line_points, _ = self._get_plotly_line_data(coords, cells, scalars=None)
        _plot_lines(
            plotter,
            pos=line_points,
            width=self.pargs.line_width,
            color=self.pargs.color_truss,
            name="Truss",
            hoverinfo="skip",
        )

        if show_bc:
            self._plot_bc(plotter=plotter, step=step, defo_scale=0.0, bc_scale=bc_scale)
        if show_mp_constraint:
            self._plot_mp_constraint(plotter, step=step, defo_scale=0.0)

    def _make_txt(self, step, add_title=False):
        resp = self.resp_step[step].to_numpy()
        maxv = np.max(resp)
        minv = np.min(resp)
        t_ = self.time[step]

        title = f"<b>{self._set_txt_props(self.resp_type)} *</b><br>"
        if self.unit_symbol:
            unit_txt = self._set_txt_props(self.unit_symbol)
            title += f"<b>(unit) {unit_txt}</b><br>"
        maxv = self._set_txt_props(f"{maxv:.3E}")
        minv = self._set_txt_props(f"{minv:.3E}")
        title += f"<b>max:</b> {maxv}<br><b>min:</b> {minv}"
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

    def plot_slide(
        self,
        ele_tags=None,
        show_values=True,
        alpha=1.0,
        line_width=1.5,
        plot_all_mesh=True,
        style="surface",
        opacity=1.0,
        show_bc: bool = True,
        bc_scale: float = 1.0,
        show_mp_constraint: bool = True,
        color=None,
    ):
        _, clim, alpha_ = self._get_resp_peak(idx="absMax")
        ndatas = []
        ndata_cum = 0
        for i in range(self.num_steps):
            plotter = []
            self._create_mesh(
                plotter,
                i,
                alpha=alpha_ * alpha,
                ele_tags=ele_tags,
                clim=clim,
                coloraxis=f"coloraxis{i + 1}",
                show_values=show_values,
                plot_all_mesh=plot_all_mesh,
                line_width=line_width,
                style=style,
                opacity=opacity,
                show_bc=show_bc,
                bc_scale=bc_scale,
                show_mp_constraint=show_mp_constraint,
                color=color,
            )
            self.FIGURE.add_traces(plotter)
            ndatas.append(len(self.FIGURE.data) - ndata_cum)
            ndata_cum = len(self.FIGURE.data)
        showscale = color is None
        self._update_slider_layout(ndatas=ndatas, clim=clim, showscale=showscale)

    def plot_peak_step(
        self,
        ele_tags=None,
        step="absMax",
        show_values=True,
        alpha=1.0,
        line_width=1.5,
        plot_all_mesh=True,
        style="surface",
        opacity=1.0,
        show_bc: bool = True,
        bc_scale: float = 1.0,
        show_mp_constraint: bool = True,
        color=None,
    ):
        max_step, clim, alpha_ = self._get_resp_peak(idx=step)

        plotter = []
        self._create_mesh(
            plotter=plotter,
            value=max_step,
            ele_tags=ele_tags,
            alpha=alpha_ * alpha,
            clim=clim,
            coloraxis="coloraxis",
            show_values=show_values,
            plot_all_mesh=plot_all_mesh,
            line_width=line_width,
            style=style,
            opacity=opacity,
            show_bc=show_bc,
            bc_scale=bc_scale,
            show_mp_constraint=show_mp_constraint,
            color=color,
        )
        self.FIGURE.add_traces(plotter)
        txt = self._make_txt(max_step)
        showscale = color is None
        self.FIGURE.update_layout(
            coloraxis={
                "colorscale": self.pargs.cmap,
                "cmin": clim[0],
                "cmax": clim[1],
                "colorbar": {"tickfont": {"size": self.pargs.font_size - 2}, "title": txt},
                "showscale": showscale,
            },
        )
        if not showscale:
            self.title["text"] += txt["text"]

    def plot_anim(
        self,
        ele_tags=None,
        show_values=True,
        alpha=1.0,
        framerate=None,
        line_width=1.5,
        plot_all_mesh=True,
        style="surface",
        opacity=1.0,
        show_bc: bool = True,
        bc_scale: float = 1.0,
        show_mp_constraint: bool = True,
        color=None,
    ):
        if framerate is None:
            framerate = np.ceil(self.num_steps / 10)
        _, clim, alpha_ = self._get_resp_peak()
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
                ele_tags=ele_tags,
                alpha=alpha_ * alpha,
                clim=clim,
                coloraxis="coloraxis",
                show_values=show_values,
                plot_all_mesh=plot_all_mesh,
                line_width=line_width,
                style=style,
                opacity=opacity,
                show_bc=show_bc,
                bc_scale=bc_scale,
                show_mp_constraint=show_mp_constraint,
                color=color,
            )
            frames.append(go.Frame(data=plotter, name="step:" + str(i)))
        # Add data to be displayed before animation starts
        plotter0 = []
        self._create_mesh(
            plotter0,
            0,
            alpha=alpha_,
            ele_tags=ele_tags,
            coloraxis="coloraxis",
            clim=clim,
            show_values=show_values,
            plot_all_mesh=plot_all_mesh,
            line_width=line_width,
            style=style,
            opacity=opacity,
            show_bc=show_bc,
            bc_scale=bc_scale,
            show_mp_constraint=show_mp_constraint,
            color=color,
        )
        self.FIGURE = go.Figure(frames=frames, data=plotter0)

        showscale = color is None
        self._update_antimate_layout(duration=duration, cbar_title=self.resp_type.capitalize(), showscale=showscale)


def plot_truss_responses(
    odb_tag: Union[int, str] = 1,
    ele_tags: Optional[Union[int, list]] = None,
    slides: bool = False,
    step: Union[int, str] = "absMax",
    show_values: bool = True,
    resp_type: str = "axialForce",
    alpha: float = 1.0,
    show_outline: bool = False,
    line_width: float = 1.5,
    unit_symbol: Optional[str] = None,
    unit_factor: Optional[float] = None,
    style: str = "surface",
    color: Optional[str] = None,
    opacity: float = 1.0,
    show_bc: bool = False,
    bc_scale: float = 1.0,
    show_mp_constraint: bool = False,
    show_model: bool = False,
) -> go.Figure:
    """Visualizing Truss Response.

    Parameters
    ----------
    odb_tag: Union[int, str], default: 1
        Tag of output databases (ODB) to be visualized.
    ele_tags: Union[int, list], default: None
        The tags of truss elements to be visualized. If None, all truss elements are selected.
    slides: bool, default: False
        Display the response for each step in the form of a slideshow.
        Otherwise, show the step with the following ``step`` parameter.
    step: Union[int, str], default: "absMax"
        If slides = False, this parameter will be used as the step to plot.
        If str, Optional: [absMax, absMin, Max, Min].
        If int, this step will be demonstrated (counting from 0).
    show_values: bool, default: False
        Whether to display the response value by hover.
        Set to False can improve the performance of the visualization.
    resp_type: str, default: "axialForce"
        Response type, optional, one of ["axialForce", "axialDefo", "Stress", "Strain"].
    alpha: float, default: 1.0
        Scale the size of the response graph.

        .. Note::
            You can adjust the scale to make the response graph more visible.
            A negative number will reverse the direction.

    show_outline: bool, default: False
        Whether to display the outline of the model.
    line_width: float, default: 1.5.
        Line width of the response graph.
    unit_symbol: str, default: None
        Unit symbol to be displayed in the plot.
        This feature is added since v1.0.15.
    unit_factor: float, default: None
        This feature is added since v1.0.15.
        The multiplier used to convert units.
        For example, if you want to visualize stress and the current data unit is kPa, you can set ``unit_symbol="kPa" and unit_factor=1.0``.
        If you want to visualize in MPa, you can set ``unit_symbol="MPa" and unit_factor=0.001``.
    style: str, default: surface
        Visualization mesh style of surfaces and solids.
        One of the following: style='surface' or style='wireframe'
        Defaults to 'surface'. Note that 'wireframe' only shows a wireframe of the outer geometry.
    color: str, default: None
        Single color of the response graph.
        If None, the colormap will be used.
    opacity: float, default: 1.0
        Face opacity when style="surface".
    show_bc: bool, default: False
        Whether to display boundary supports.
        Set to False can improve the performance of the visualization.
    bc_scale: float, default: 1.0
        Scale the size of boundary support display.
    show_mp_constraint: bool, default: False
        Whether to show multipoint (MP) constraint.
        Set to False can improve the performance of the visualization.
    show_model: bool, default: False
        Whether to plot the all model or not.
        Set to False can improve the performance of the visualization.

    Returns
    -------
    fig: `plotly.graph_objects.Figure <https://plotly.com/python-api-reference/generated/plotly.graph_objects.Figure.html>`_
        You can use `fig.show()` to display,
        You can also use `fig.write_html("path/to/file.html")` to save as an HTML file, see
        `Interactive HTML Export in Python <https://plotly.com/python/interactive-html-export/>`_
    """
    model_info_steps, model_update, truss_resp_step = loadODB(odb_tag, resp_type="Truss")
    plotbase = PlotTrussResponse(model_info_steps, truss_resp_step, model_update)
    plotbase.set_unit(symbol=unit_symbol, factor=unit_factor)
    plotbase.refactor_resp_step(resp_type=resp_type, ele_tags=ele_tags)
    if slides:
        plotbase.plot_slide(
            ele_tags=ele_tags,
            show_values=show_values,
            alpha=alpha,
            line_width=line_width,
            plot_all_mesh=show_model,
            style=style,
            opacity=opacity,
            show_bc=show_bc,
            bc_scale=bc_scale,
            show_mp_constraint=show_mp_constraint,
            color=color,
        )
    else:
        plotbase.plot_peak_step(
            step=step,
            show_values=show_values,
            alpha=alpha,
            line_width=line_width,
            plot_all_mesh=show_model,
            style=style,
            opacity=opacity,
            show_bc=show_bc,
            bc_scale=bc_scale,
            show_mp_constraint=show_mp_constraint,
            color=color,
        )
    return plotbase.update_fig(show_outline=show_outline)


def plot_truss_responses_animation(
    odb_tag: Union[int, str] = 1,
    ele_tags: Optional[Union[int, list]] = None,
    framerate: Optional[int] = None,
    show_values: bool = False,
    resp_type: str = "axialForce",
    alpha: float = 1.0,
    show_outline: bool = False,
    line_width: float = 1.5,
    unit_symbol: Optional[str] = None,
    unit_factor: Optional[float] = None,
    style: str = "surface",
    color: Optional[str] = None,
    opacity: float = 1.0,
    show_bc: bool = False,
    bc_scale: float = 1.0,
    show_mp_constraint: bool = False,
    show_model: bool = False,
) -> go.Figure:
    """Truss response animation.

    Parameters
    ----------
    odb_tag: Union[int, str], default: 1
        Tag of output databases (ODB) to be visualized.
    ele_tags: Union[int, list], default: None
        The tags of truss elements to be visualized. If None, all truss elements are selected.
    framerate: int, default: None
        Framerate for the display, i.e., the number of frames per second.
    show_values: bool, default: False
        Whether to display the response value by hover.
        Set to False can improve the performance of the visualization.
    resp_type: str, default: "axialForce"
        Response type, optional, one of ["axialForce", "axialDefo", "Stress", "Strain"].
    alpha: float, default: 1.0
        Scale the size of the response graph.

        .. Note::
            You can adjust the scale to make the response graph more visible.
            A negative number will reverse the direction.

    show_outline: bool, default: False
        Whether to display the outline of the model.
    line_width: float, default: 1.5.
        Line width of the response graph.
    unit_symbol: str, default: None
        Unit symbol to be displayed in the plot.
        This feature is added since v1.0.15.
    unit_factor: float, default: None
        This feature is added since v1.0.15.
        The multiplier used to convert units.
        For example, if you want to visualize stress and the current data unit is kPa, you can set ``unit_symbol="kPa" and unit_factor=1.0``.
        If you want to visualize in MPa, you can set ``unit_symbol="MPa" and unit_factor=0.001``.
    style: str, default: surface
        Visualization mesh style of surfaces and solids.
        One of the following: style='surface' or style='wireframe'
        Defaults to 'surface'. Note that 'wireframe' only shows a wireframe of the outer geometry.
    color: str, default: None
        Single color of the response graph.
        If None, the colormap will be used.
    opacity: float, default: 1.0
        Face opacity when style="surface".
    show_bc: bool, default: False
        Whether to display boundary supports.
        Set to False can improve the performance of the visualization.
    bc_scale: float, default: 1.0
        Scale the size of boundary support display.
    show_mp_constraint: bool, default: False
        Whether to show multipoint (MP) constraint.
        Set to False can improve the performance of the visualization.
    show_model: bool, default: False
        Whether to plot the all model or not.
        Set to False can improve the performance of the visualization.

    Returns
    -------
    fig: `plotly.graph_objects.Figure <https://plotly.com/python-api-reference/generated/plotly.graph_objects.Figure.html>`_
        You can use `fig.show()` to display,
        You can also use `fig.write_html("path/to/file.html")` to save as an HTML file, see
        `Interactive HTML Export in Python <https://plotly.com/python/interactive-html-export/>`_
    """
    model_info_steps, model_update, truss_resp_step = loadODB(odb_tag, resp_type="Truss")
    plotbase = PlotTrussResponse(model_info_steps, truss_resp_step, model_update)
    plotbase.set_unit(symbol=unit_symbol, factor=unit_factor)
    plotbase.refactor_resp_step(resp_type=resp_type, ele_tags=ele_tags)
    plotbase.plot_anim(
        ele_tags=ele_tags,
        show_values=show_values,
        alpha=alpha,
        framerate=framerate,
        line_width=line_width,
        plot_all_mesh=show_model,
        style=style,
        opacity=opacity,
        show_bc=show_bc,
        bc_scale=bc_scale,
        show_mp_constraint=show_mp_constraint,
        color=color,
    )
    return plotbase.update_fig(show_outline=show_outline)
