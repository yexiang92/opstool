from functools import partial
from typing import Optional, Union

import numpy as np
import pyvista as pv

from ...post import loadODB
from ...utils import gram_schmidt
from .plot_resp_base import PlotResponseBase
from .plot_utils import (
    PLOT_ARGS,
    # _plot_lines_cmap,
    _plot_face_cmap,
    _plot_lines,
    _update_point_label_actor,
)


class PlotTrussResponse(PlotResponseBase):
    def __init__(self, model_info_steps, truss_resp_step, model_update):
        super().__init__(model_info_steps, truss_resp_step, model_update)

    def _get_truss_data(self, step):
        return self._get_model_da("TrussData", step)

    def _set_resp_type(self, resp_type: str):
        if resp_type.lower() in ["axialforce", "force"]:
            resp_type = "axialForce"
        elif resp_type.lower() in ["axialdefo", "axialdeformation", "deformation"]:
            resp_type = "axialDefo"
        elif resp_type.lower() in ["stress", "axialstress"]:
            resp_type = "Stress"
        elif resp_type.lower() in ["strain", "axialstrain"]:
            resp_type = "Strain"
        else:
            raise ValueError(  # noqa: TRY003
                f"Not supported response type {resp_type}!Valid options are: axialForce, axialDefo, Stress, Strain."
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
        return step, (cmin, cmax), float(alpha_)

    def _get_truss_resp_clim(self):
        maxv = [np.max(data) for data in self.resp_step]
        minv = [np.min(data) for data in self.resp_step]
        cmin, cmax = np.min(minv), np.max(maxv)
        return cmin, cmax

    def _make_title(self, scalars, step, time):
        if len(scalars) == 0:
            scalars = np.array([0.0])
        info = {
            "title": "Truss",
            "resp_type": self.resp_type.capitalize(),
            "min": np.min(scalars),
            "max": np.max(scalars),
            "step": step,
            "time": time,
        }
        lines = [
            f"* {info['title']} Responses",
            f"* {info['resp_type']}",
            f"{info['min']:.3E} (min)",
            f"{info['max']:.3E} (max)",
            f"{info['step']}(step); {info['time']:.3f}(time)",
        ]
        if self.unit_symbol:
            info["unit"] = self.unit_symbol
            lines.insert(2, f"{info['unit']} (unit)")
        max_len = max(len(line) for line in lines)
        padded_lines = [line.rjust(max_len) for line in lines]
        text = "\n".join(padded_lines)
        return text + "\n"

    def _get_mesh_data(self, step, alpha, ele_tags):
        n_segs = 10
        truss_tags, truss_coords, truss_cells = self._make_truss_info(ele_tags, step)
        resps = self.resp_step[step].to_numpy()
        resp_points, resp_cells = [], []
        scalars = []
        label_points, labels = [], []
        for cell, resp in zip(truss_cells, resps):
            coord1 = np.array(truss_coords[cell[1]])
            coord2 = np.array(truss_coords[cell[2]])
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
            label_points.append((coord3 + coord4) / 2)
            labels.append(resp)
        fmt = self.pargs.scalar_bar_kargs["fmt"]
        labels = [f"{fmt}" % label for label in labels]
        label_points = np.array(label_points)
        resp_points = np.array(resp_points)
        scalars = np.array(scalars)
        return truss_coords, truss_cells, scalars, resp_points, resp_cells, labels, label_points

    def _create_mesh(
        self,
        plotter,
        value,
        ele_tags=None,
        show_values=True,
        plot_all_mesh=True,
        clim=None,
        alpha=1.0,
        line_width=1.5,
        style="surface",
        color=None,
        opacity=1.0,
        cpos="iso",
        show_outline=False,
        show_bc: bool = True,
        bc_scale: float = 1.0,
        show_mp_constraint: bool = False,
    ):
        step = round(value)
        truss_coords, truss_cells, scalars, resp_points, resp_cells, labels, label_points = self._get_mesh_data(
            step, alpha, ele_tags
        )
        #  ---------------------------------
        plotter.clear_actors()  # !!!!!!
        if plot_all_mesh:
            self._plot_all_mesh(plotter, color="gray", step=step)
        line_grid = _plot_lines(
            plotter,
            pos=truss_coords,
            cells=truss_cells,
            width=self.pargs.line_width,
            color=self.pargs.color_truss,
            render_lines_as_tubes=self.pargs.render_lines_as_tubes,
        )
        opacity = 1.0 if style.lower() != "surface" else opacity
        resp_grid = _plot_face_cmap(
            plotter,
            resp_points,
            resp_cells,
            scalars,
            color=color,
            cmap=self.pargs.cmap,
            clim=clim,
            show_edges=False,
            edge_width=line_width,
            opacity=opacity,
            style=style,
            show_scalar_bar=False,
        )
        title = self._make_title(scalars, step, self.time[step])
        scalar_bar = plotter.add_scalar_bar(title=title, **self.pargs.scalar_bar_kargs) if color is None else None
        if scalar_bar:
            title_prop = scalar_bar.GetTitleTextProperty()
            title_prop.SetJustificationToRight()
            title_prop.BoldOn()

        title_grid = (
            None
            if scalar_bar
            else plotter.add_text(title, position="upper_right", font_size=self.pargs.font_size - 2, font="courier")
        )

        if show_values:
            label_grid = plotter.add_point_labels(
                label_points,
                labels,
                # text_color="white",
                font_size=self.pargs.font_size,
                font_family="courier",
                bold=False,
                always_visible=False,
                shape=None,
                shape_opacity=0.0,
                show_points=False,
            )
        else:
            label_grid = None

        if show_outline:
            self._plot_outline(plotter)
        bc_grid, mp_grid = None, None
        if show_bc:
            bc_grid = self._plot_bc(plotter, step, defo_scale=0.0, bc_scale=bc_scale)
        if show_mp_constraint:
            mp_grid = self._plot_mp_constraint(plotter, step, defo_scale=0.0)
        self._update_plotter(plotter, cpos=cpos)
        return line_grid, resp_grid, scalar_bar, title_grid, label_grid, bc_grid, mp_grid

    def _update_mesh(
        self,
        step,
        alpha,
        ele_tags,
        line_grid,
        resp_grid,
        scalar_bar,
        title_grid,
        label_grid,
        bc_grid,
        mp_grid,
        bc_scale,
        plotter,
    ):
        step = round(step)
        truss_coords, truss_cells, scalars, resp_points, resp_cells, labels, label_points = self._get_mesh_data(
            step, alpha, ele_tags
        )

        if line_grid:
            line_grid.points = truss_coords
            line_grid.lines = truss_cells

        if resp_grid:
            resp_grid.points = resp_points
            # resp_grid.lines = resp_cells
            resp_grid.faces = resp_cells
            resp_grid["scalars"] = scalars

        if scalar_bar:
            title = self._make_title(scalars, step, self.time[step])
            scalar_bar.SetTitle(title)

        if title_grid:
            title_grid.SetText(3, self._make_title(scalars, step, self.time[step]))

        if label_grid:
            # mapper = label_plot.GetMapper()
            text_property = pv.TextProperty(
                bold=False,
                font_size=self.pargs.font_size,
                font_family="courier",
                color=pv.global_theme.font.color,
            )
            _update_point_label_actor(
                label_grid,
                label_points,
                labels,
                text_property=text_property,
                renderer=plotter.renderer,
                shape_opacity=0.0,
                always_visible=False,
            )

        if mp_grid:
            self._plot_mp_constraint_update(mp_grid, step, defo_scale=0.0)
        if bc_grid:
            self._plot_bc_update(bc_grid, step, defo_scale=0.0, bc_scale=bc_scale)

    def plot_slide(
        self,
        plotter,
        ele_tags=None,
        show_values=True,
        alpha=1.0,
        line_width=1.5,
        style="surface",
        opacity=1.0,
        cpos="iso",
        plot_model=True,
        show_bc: bool = True,
        bc_scale: float = 1.0,
        show_mp_constraint: bool = True,
        show_outline=False,
        color=None,
    ):
        _, clim, alpha_ = self._get_resp_peak()
        if self.ModelUpdate:
            func = partial(
                self._create_mesh,
                plotter,
                ele_tags=ele_tags,
                clim=clim,
                plot_all_mesh=plot_model,
                show_values=show_values,
                alpha=alpha * alpha_,
                line_width=line_width,
                style=style,
                opacity=opacity,
                cpos=cpos,
                show_outline=show_outline,
                show_bc=show_bc,
                bc_scale=bc_scale,
                show_mp_constraint=show_mp_constraint,
                color=color,
            )
        else:
            line_grid, resp_grid, scalar_bar, title_grid, label_grid, bc_grid, mp_grid = self._create_mesh(
                plotter,
                self.num_steps - 1,
                ele_tags=ele_tags,
                clim=clim,
                plot_all_mesh=plot_model,
                show_values=show_values,
                alpha=alpha * alpha_,
                line_width=line_width,
                style=style,
                opacity=opacity,
                cpos=cpos,
                show_outline=show_outline,
                show_bc=show_bc,
                bc_scale=bc_scale,
                show_mp_constraint=show_mp_constraint,
                color=color,
            )

            func = partial(
                self._update_mesh,
                alpha=alpha * alpha_,
                ele_tags=ele_tags,
                line_grid=line_grid,
                resp_grid=resp_grid,
                scalar_bar=scalar_bar,
                title_grid=title_grid,
                label_grid=label_grid,
                bc_grid=bc_grid,
                mp_grid=mp_grid,
                bc_scale=bc_scale,
                plotter=plotter,
            )
        plotter.add_slider_widget(func, [0, self.num_steps - 1], value=self.num_steps - 1, **self.slider_widget_args)

    def plot_peak_step(
        self,
        plotter,
        ele_tags=None,
        step="absMax",
        show_values=True,
        alpha=1.0,
        line_width=1.5,
        style="surface",
        opacity=1.0,
        cpos="iso",
        plot_model=True,
        show_bc: bool = True,
        bc_scale: float = 1.0,
        show_mp_constraint: bool = True,
        show_outline=False,
        color=None,
    ):
        step, clim, alpha_ = self._get_resp_peak(idx=step)
        self._create_mesh(
            plotter=plotter,
            value=step,
            ele_tags=ele_tags,
            show_values=show_values,
            clim=clim,
            plot_all_mesh=plot_model,
            alpha=alpha * alpha_,
            line_width=line_width,
            style=style,
            opacity=opacity,
            cpos=cpos,
            show_outline=show_outline,
            show_bc=show_bc,
            bc_scale=bc_scale,
            show_mp_constraint=show_mp_constraint,
            color=color,
        )

    def plot_anim(
        self,
        plotter,
        ele_tags=None,
        show_values=True,
        alpha=1.0,
        framerate: Optional[int] = None,
        savefig: str = "TrussRespAnimation.gif",
        line_width=1.5,
        style="surface",
        opacity=1.0,
        cpos="iso",
        plot_model=True,
        show_bc: bool = True,
        bc_scale: float = 1.0,
        show_mp_constraint: bool = True,
        show_outline=False,
        color=None,
    ):
        if framerate is None:
            framerate = np.ceil(self.num_steps / 10)
        if savefig.endswith(".gif"):
            plotter.open_gif(savefig, fps=framerate)
        else:
            plotter.open_movie(savefig, framerate=framerate)
        _, clim, alpha_ = self._get_resp_peak()
        # plotter.write_frame()  # write initial data

        if self.ModelUpdate:
            for step in range(self.num_steps):
                self._create_mesh(
                    plotter,
                    step,
                    ele_tags=ele_tags,
                    show_values=show_values,
                    clim=clim,
                    plot_all_mesh=plot_model,
                    alpha=alpha * alpha_,
                    line_width=line_width,
                    style=style,
                    opacity=opacity,
                    cpos=cpos,
                    show_outline=show_outline,
                    show_bc=show_bc,
                    bc_scale=bc_scale,
                    show_mp_constraint=show_mp_constraint,
                    color=color,
                )
                plotter.write_frame()
        else:
            line_grid, resp_grid, scalar_bar, title_grid, label_grid, bc_grid, mp_grid = self._create_mesh(
                plotter,
                0,
                ele_tags=ele_tags,
                clim=clim,
                plot_all_mesh=plot_model,
                show_values=show_values,
                alpha=alpha * alpha_,
                line_width=line_width,
                style=style,
                opacity=opacity,
                cpos=cpos,
                show_outline=show_outline,
                show_bc=show_bc,
                bc_scale=bc_scale,
                show_mp_constraint=show_mp_constraint,
                color=color,
            )
            plotter.write_frame()
            for step in range(1, self.num_steps):
                self._update_mesh(
                    step=step,
                    alpha=alpha * alpha_,
                    ele_tags=ele_tags,
                    line_grid=line_grid,
                    resp_grid=resp_grid,
                    scalar_bar=scalar_bar,
                    title_grid=title_grid,
                    label_grid=label_grid,
                    bc_grid=bc_grid,
                    mp_grid=mp_grid,
                    bc_scale=bc_scale,
                    plotter=plotter,
                )
                plotter.write_frame()


def plot_truss_responses(
    odb_tag: Union[int, str] = 1,
    ele_tags: Optional[Union[int, list]] = None,
    slides: bool = False,
    step: Union[int, str] = "absMax",
    show_values: bool = True,
    resp_type: str = "axialForce",
    color: Optional[str] = None,
    unit_symbol: Optional[str] = None,
    unit_factor: Optional[float] = None,
    alpha: float = 1.0,
    style: str = "surface",
    line_width: float = 1.5,
    opacity: float = 1.0,
    show_bc: bool = True,
    bc_scale: float = 1.0,
    show_mp_constraint: bool = False,
    show_outline: bool = False,
    cpos: str = "iso",
    show_model: bool = True,
) -> pv.Plotter:
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
    show_values: bool, default: True
        Whether to display the response value.
    resp_type: str, default: "axialForce"
        Response type, optional, one of ["axialForce", "axialDefo", "Stress", "Strain"].
    color: str, default: None
        Single color of the response graph.
        If None, the colormap will be used.
    unit_symbol: str, default: None
        Unit symbol to be displayed in the plot.
    unit_factor: float, default: None
        The multiplier used to convert units.
        For example, if you want to visualize stress and the current data unit is kPa, you can set ``unit_symbol="kPa" and unit_factor=1.0``.
        If you want to visualize in MPa, you can set ``unit_symbol="MPa" and unit_factor=0.001``.
    alpha: float, default: 1.0
        Scale the size of the response graph.

        .. Note::
            You can adjust the scale to make the response graph more visible.
            A negative number will reverse the direction.

    style: str, default: "surface
        Display style for responses plot, optional, one of ["surface", "wireframe"]
    line_width: float, default: 1.5.
        Line width of the response graph when style="wireframe".
    opacity: float, default: 1.0
        Face opacity when style="surface".
    cpos: str, default: iso
        Model display perspective, optional: "iso", "xy", "yx", "xz", "zx", "yz", "zy".
        If 3d, defaults to "iso". If 2d, defaults to "xy".
    show_model: bool, default: True
        Whether to plot the all model or not.
    show_bc: bool, default: True
        Whether to display boundary supports.
    bc_scale: float, default: 1.0
        Scale the size of boundary support display.
    show_mp_constraint: bool, default: False
        Whether to show multipoint (MP) constraint.
    show_outline: bool, default: False
        Whether to display the outline of the model.

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
    model_info_steps, model_update, truss_resp_step = loadODB(odb_tag, resp_type="Truss")
    plotter = pv.Plotter(
        notebook=PLOT_ARGS.notebook, line_smoothing=PLOT_ARGS.line_smoothing, off_screen=PLOT_ARGS.off_screen
    )
    plotbase = PlotTrussResponse(model_info_steps, truss_resp_step, model_update)
    plotbase.set_unit(symbol=unit_symbol, factor=unit_factor)
    plotbase.refactor_resp_step(resp_type=resp_type, ele_tags=ele_tags)
    if slides:
        plotbase.plot_slide(
            plotter,
            ele_tags=ele_tags,
            show_values=show_values,
            alpha=alpha,
            line_width=line_width,
            style=style,
            opacity=opacity,
            cpos=cpos,
            plot_model=show_model,
            show_bc=show_bc,
            bc_scale=bc_scale,
            show_mp_constraint=show_mp_constraint,
            show_outline=show_outline,
            color=color,
        )
    else:
        plotbase.plot_peak_step(
            plotter,
            ele_tags=ele_tags,
            step=step,
            show_values=show_values,
            alpha=alpha,
            line_width=line_width,
            style=style,
            opacity=opacity,
            cpos=cpos,
            plot_model=show_model,
            show_bc=show_bc,
            bc_scale=bc_scale,
            show_mp_constraint=show_mp_constraint,
            show_outline=show_outline,
            color=color,
        )
    if PLOT_ARGS.anti_aliasing:
        plotter.enable_anti_aliasing(PLOT_ARGS.anti_aliasing)
    return plotbase._update_plotter(plotter, cpos)


def plot_truss_responses_animation(
    odb_tag: Union[int, str] = 1,
    ele_tags: Optional[Union[int, list]] = None,
    framerate: Optional[int] = None,
    savefig: str = "TrussRespAnimation.gif",
    off_screen: bool = True,
    show_values: bool = False,
    resp_type: str = "axialForce",
    color: Optional[str] = None,
    unit_symbol: Optional[str] = None,
    unit_factor: Optional[float] = None,
    alpha: float = 1.0,
    style: str = "surface",
    line_width: float = 1.5,
    opacity: float = 1.0,
    cpos: str = "iso",
    show_model: bool = True,
    show_bc: bool = True,
    bc_scale: float = 1.0,
    show_mp_constraint: bool = False,
    show_outline: bool = False,
) -> pv.Plotter:
    """Truss response animation.

    Parameters
    ----------
    odb_tag: Union[int, str], default: 1
        Tag of output databases (ODB) to be visualized.
    ele_tags: Union[int, list], default: None
        The tags of truss elements to be visualized. If None, all truss elements are selected.
    framerate: int, default: None
        Framerate for the display, i.e., the number of frames per second.
    savefig: str, default: TrussRespAnimation.gif
        Path to save the animation. The suffix can be ``.gif`` or ``.mp4``.
    off_screen: bool, default: True
        Whether to display the plotting window.
        If True, the plotting window will not be displayed.
    show_values: bool, default: False
        Whether to display the response value.
    resp_type: str, default: "axialForce"
        Response type, optional, one of ["axialForce", "axialDefo", "Stress", "Strain"].
    color: str, default: None
        Single color of the response graph.
        If None, the colormap will be used.
    unit_symbol: str, default: None
        Unit symbol to be displayed in the plot.
    unit_factor: float, default: None
        The multiplier used to convert units.
        For example, if you want to visualize stress and the current data unit is kPa, you can set ``unit_symbol="kPa" and unit_factor=1.0``.
        If you want to visualize in MPa, you can set ``unit_symbol="MPa" and unit_factor=0.001``.
    alpha: float, default: 1.0
        Scale the size of the response graph.

        .. Note::
            You can adjust the scale to make the response graph more visible.
            A negative number will reverse the direction.

    style: str, default: "surface
        Display style for responses plot, optional, one of ["surface", "wireframe"]
    line_width: float, default: 1.5.
        Line width of the response graph when style="wireframe".
    opacity: float, default: 1.0
        Face opacity when style="surface".
    cpos: str, default: iso
        Model display perspective, optional: "iso", "xy", "yx", "xz", "zx", "yz", "zy".
        If 3d, defaults to "iso". If 2d, defaults to "xy".
    show_model: bool, default: True
        Whether to plot the all model or not.
    show_bc: bool, default: True
        Whether to display boundary supports.
    bc_scale: float, default: 1.0
        Scale the size of boundary support display.
    show_mp_constraint: bool, default: False
        Whether to show multipoint (MP) constraint.
    show_outline: bool, default: False
        Whether to display the outline of the model.

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
    model_info_steps, model_update, truss_resp_step = loadODB(odb_tag, resp_type="Truss")
    plotter = pv.Plotter(notebook=PLOT_ARGS.notebook, line_smoothing=PLOT_ARGS.line_smoothing, off_screen=off_screen)
    plotbase = PlotTrussResponse(model_info_steps, truss_resp_step, model_update)
    plotbase.set_unit(symbol=unit_symbol, factor=unit_factor)
    plotbase.refactor_resp_step(resp_type=resp_type, ele_tags=ele_tags)
    plotbase.plot_anim(
        plotter,
        ele_tags=ele_tags,
        show_values=show_values,
        alpha=alpha,
        framerate=framerate,
        savefig=savefig,
        line_width=line_width,
        style=style,
        opacity=opacity,
        cpos=cpos,
        plot_model=show_model,
        show_bc=show_bc,
        bc_scale=bc_scale,
        show_mp_constraint=show_mp_constraint,
        show_outline=show_outline,
        color=color,
    )
    if PLOT_ARGS.anti_aliasing:
        plotter.enable_anti_aliasing(PLOT_ARGS.anti_aliasing)
    print(f"Animation has been saved as {savefig}!")
    return plotbase._update_plotter(plotter, cpos)
