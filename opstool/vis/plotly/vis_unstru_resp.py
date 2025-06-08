from typing import Optional, Union

import numpy as np
import plotly.graph_objs as go

from ...post import loadODB
from .plot_resp_base import PlotResponseBase
from .plot_utils import (
    _get_unstru_cells,
    _plot_points_cmap,
    _plot_unstru_cmap,
)


class PlotUnstruResponse(PlotResponseBase):
    def __init__(self, model_info_steps, resp_step, model_update, node_resp_steps=None):
        super().__init__(model_info_steps, resp_step, model_update, nodal_resp_steps=node_resp_steps)
        self.ele_type = "Shell"

    def _get_unstru_da(self, step):
        if self.ele_type.lower() == "shell":
            return self._get_model_da("ShellData", step)
        elif self.ele_type.lower() == "plane":
            return self._get_model_da("PlaneData", step)
        elif self.ele_type.lower() in ["brick", "solid"]:
            return self._get_model_da("BrickData", step)
        else:
            raise ValueError(f"Invalid element type {self.ele_type}! Valid options are: Shell, Plane, Brick.")  # noqa: TRY003

    def _set_comp_resp_type(self, ele_type, resp_type, component):
        self.ele_type = ele_type
        self.resp_type = resp_type
        self.component = component

        title = f"<b>{self.PKG_NAME}:: {self.ele_type.capitalize()} Responses 3D Viewer</b><br><br>"
        self.title = {"text": title, "font": {"size": self.pargs.title_font_size}}

    def _make_unstru_info(self, ele_tags, step, defo_scale):
        pos_defo = np.array(self._get_defo_coord_da(step, defo_scale))
        # pos = self._get_node_da(step).to_numpy()
        unstru_data = self._get_unstru_da(step)
        if ele_tags is None:
            tags, cell_types, cells = _get_unstru_cells(unstru_data)
        else:
            tags = np.atleast_1d(ele_tags)
            cells = unstru_data.sel(eleTags=tags)
            tags, cell_types, cells = _get_unstru_cells(cells)
        return tags, pos_defo, cells, cell_types

    def refactor_resp_step(self, ele_tags, ele_type, resp_type: str, component: str):
        self._set_comp_resp_type(ele_type, resp_type, component)
        resps = []
        if self.ModelUpdate or ele_tags is not None:
            for i in range(self.num_steps):
                tags, _, _, _ = self._make_unstru_info(ele_tags, i)
                da = self._get_resp_da(i, self.resp_type, self.component)
                da = da.sel(eleTags=tags)
                resps.append(da.mean(dim="GaussPoints", skipna=True))
        else:
            for i in range(self.num_steps):
                da = self._get_resp_da(i, self.resp_type, self.component)
                resps.append(da.mean(dim="GaussPoints", skipna=True))
        self.resp_step = resps

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
        cmin, cmax = self._get_resp_clim()
        self.clim = (cmin, cmax)
        return step, (cmin, cmax)

    def _get_resp_clim(self):
        maxv = [np.max(data) for data in self.resp_step]
        minv = [np.min(data) for data in self.resp_step]
        cmin, cmax = np.min(minv), np.max(maxv)
        return cmin, cmax

    def _create_mesh(
        self,
        plotter,
        value,
        ele_tags=None,
        plot_all_mesh=True,
        clim=None,
        coloraxis="coloraxis",
        style="surface",
        show_values=False,
        defo_scale: float = 1.0,
        show_bc: bool = True,
        bc_scale: float = 1.0,
        show_mp_constraint: bool = True,
    ):
        step = round(value)
        tags, pos, cells, cell_types = self._make_unstru_info(ele_tags, step, defo_scale=defo_scale)
        resps = self.resp_step[step].to_numpy()
        scalars = resps
        #  ---------------------------------
        if plot_all_mesh:
            self._plot_all_mesh(plotter=plotter, step=step)
        if show_bc:
            self._plot_bc(plotter=plotter, step=step, defo_scale=defo_scale, bc_scale=bc_scale)
        if show_mp_constraint:
            self._plot_mp_constraint(plotter, step=step, defo_scale=defo_scale)
        # ---------------------------------------------------------------
        (
            face_points,
            face_line_points,
            face_mid_points,
            veci,
            vecj,
            veck,
            face_scalars,
            face_line_scalars,
        ) = self._get_plotly_unstru_data(pos, cell_types, cells, scalars, scalars_by_element=True)
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
        _plot_points_cmap(
            plotter,
            face_points,
            scalars=face_scalars,
            clim=clim,
            coloraxis=coloraxis,
            name=self.resp_type,
            size=self.pargs.point_size,
            show_hover=show_values,
        )

    def _make_txt(self, step, add_title=False):
        resp = self.resp_step[step].to_numpy()
        maxv, minv = np.max(resp), np.min(resp)
        t_ = self.time[step]

        title = f"<b>{self._set_txt_props(self.resp_type.capitalize())} *</b><br>"
        comp = self.component if isinstance(self.component, str) else " ".join(self.component)
        title += f"<b>(DOF) {self._set_txt_props(comp)}</b><br>"
        if self.unit_symbol:
            unit_txt = self._set_txt_props(self.unit_symbol)
            title += f"<b>(unit) {unit_txt}</b><br>"
        maxv = self._set_txt_props(f"{maxv:.3E}")
        minv = self._set_txt_props(f"{minv:.3E}")
        title += f"<b>Max.:</b> {maxv}<br><b>Min.:</b> {minv}"
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
        style="surface",
        show_values=False,
        plot_all_mesh=False,
        show_defo=True,
        defo_scale: float = 1.0,
        show_bc: bool = True,
        bc_scale: float = 1.0,
        show_mp_constraint: bool = True,
    ):
        _, clim = self._get_resp_peak()
        alpha_ = defo_scale if show_defo else 0.0
        ndatas = []
        ndata_cum = 0
        for i in range(self.num_steps):
            plotter = []
            self._create_mesh(
                plotter,
                i,
                ele_tags=ele_tags,
                clim=clim,
                coloraxis=f"coloraxis{i + 1}",
                show_values=show_values,
                plot_all_mesh=plot_all_mesh,
                style=style,
                defo_scale=alpha_,
                show_bc=show_bc,
                bc_scale=bc_scale,
                show_mp_constraint=show_mp_constraint,
            )
            self.FIGURE.add_traces(plotter)
            ndatas.append(len(self.FIGURE.data) - ndata_cum)
            ndata_cum = len(self.FIGURE.data)

        self._update_slider_layout(ndatas=ndatas, clim=clim)

    def plot_peak_step(
        self,
        ele_tags=None,
        step="absMax",
        style="surface",
        show_values=False,
        plot_all_mesh=False,
        show_defo=True,
        defo_scale: float = 1.0,
        show_bc: bool = True,
        bc_scale: float = 1.0,
        show_mp_constraint: bool = True,
    ):
        max_step, clim = self._get_resp_peak(idx=step)
        alpha_ = defo_scale if show_defo else 0.0
        plotter = []
        self._create_mesh(
            plotter=plotter,
            value=max_step,
            ele_tags=ele_tags,
            clim=clim,
            coloraxis="coloraxis",
            show_values=show_values,
            plot_all_mesh=plot_all_mesh,
            style=style,
            defo_scale=alpha_,
            show_bc=show_bc,
            bc_scale=bc_scale,
            show_mp_constraint=show_mp_constraint,
        )
        self.FIGURE.add_traces(plotter)
        txt = self._make_txt(max_step)
        self.FIGURE.update_layout(
            coloraxis={
                "colorscale": self.pargs.cmap,
                "cmin": clim[0],
                "cmax": clim[1],
                "colorbar": {"tickfont": {"size": self.pargs.font_size - 2}, "title": txt},
            },
        )

    def plot_anim(
        self,
        ele_tags=None,
        framerate: Optional[int] = None,
        style="surface",
        show_values=False,
        plot_all_mesh=False,
        show_defo=True,
        defo_scale: float = 1.0,
        show_bc: bool = True,
        bc_scale: float = 1.0,
        show_mp_constraint: bool = True,
    ):
        if framerate is None:
            framerate = np.ceil(self.num_steps / 11)
        nb_frames = self.num_steps
        duration = 1000 / framerate
        # ---------------------------------------------
        _, clim = self._get_resp_peak()
        alpha_ = defo_scale if show_defo else 0.0
        # -----------------------------------------------------------------------------
        # start plot
        frames = []
        for i in range(nb_frames):
            plotter = []
            self._create_mesh(
                plotter=plotter,
                value=i,
                ele_tags=ele_tags,
                clim=clim,
                coloraxis="coloraxis",
                show_values=show_values,
                plot_all_mesh=plot_all_mesh,
                style=style,
                defo_scale=alpha_,
                show_bc=show_bc,
                bc_scale=bc_scale,
                show_mp_constraint=show_mp_constraint,
            )
            frames.append(go.Frame(data=plotter, name="step:" + str(i)))
        # Add data to be displayed before animation starts
        plotter0 = []
        self._create_mesh(
            plotter0,
            0,
            ele_tags=ele_tags,
            clim=clim,
            coloraxis="coloraxis",
            show_values=show_values,
            style=style,
            plot_all_mesh=plot_all_mesh,
            defo_scale=alpha_,
            show_bc=show_bc,
            bc_scale=bc_scale,
            show_mp_constraint=show_mp_constraint,
        )
        self.FIGURE = go.Figure(frames=frames, data=plotter0)
        # self.title = self._make_txt(0, add_title=True)

        self._update_antimate_layout(duration=duration, cbar_title=self.component)


def plot_unstruct_responses(
    odb_tag: Union[int, str] = 1,
    ele_type: str = "Shell",
    ele_tags: Optional[Union[int, list]] = None,
    slides: bool = False,
    step: Union[int, str] = "absMax",
    resp_type: str = "sectionForces",
    resp_dof: str = "MXX",
    unit_symbol: Optional[str] = None,
    unit_factor: Optional[float] = None,
    style: str = "surface",
    show_outline: bool = False,
    show_values: bool = False,
    show_defo: bool = False,
    defo_scale: float = 1.0,
    show_bc: bool = False,
    bc_scale: float = 1.0,
    show_mp_constraint: bool = False,
    show_model: bool = False,
) -> go.Figure:
    """Visualizing unstructured element (Shell, Plane, Brick) Response.

    .. Note::
        The responses at all Gaussian points are averaged.

    Parameters
    ----------
    odb_tag: Union[int, str], default: 1
        Tag of output databases (ODB) to be visualized.
    ele_tags: Union[int, list], default: None
        The tags of elements to be visualized.
        If None, all elements are selected.
    slides: bool, default: False
        Display the response for each step in the form of a slideshow.
        Otherwise, show the step with the following ``step`` parameter.
    step: Union[int, str], default: "absMax"
        If slides = False, this parameter will be used as the step to plot.
        If str, Optional: [absMax, absMin, Max, Min].
        If int, this step will be demonstrated (counting from 0).
    ele_type: str, default: "Shell"
        Element type, optional, one of ["Shell", "Plane", "Solid"].
    resp_type: str, default: None
        Response type, which dependents on the element type `ele_type`.

        #. For ``Shell`` elements, one of ["sectionForces", "sectionDeformations"].
            I.e., section forces and deformations at Gaussian integration points (per unit length).
            If None, defaults to "sectionForces".
        #. For ``Plane`` elements, one of ["stresses", "strains"].
            I.e., stresses and strains at Gaussian integration points.
            If None, defaults to "stresses".
        #. For ``Brick`` or ``Solid`` elements, one of ["stresses", "strains"].
            I.e., stresses and strains at Gaussian integration points.
            If None, defaults to "stresses".

    resp_dof: str, default: None
        Dof to be visualized, which dependents on the element type `ele_type`.

        .. Note::
            The `resp_dof` here is consistent with stress-strain (force-deformation),
            and whether it is stress or strain depends on the parameter `resp_type`.

        #. For ``Shell`` elements, one of ["FXX", "FYY", "FXY", "MXX", "MYY", "MXY", "VXZ", "VYZ"].
            If None, defaults to "MXX".
        #. For ``Plane`` elements, one of ["sigma11", "sigma22", "sigma12", "p1", "p2", "sigma_vm", "tau_max"].

            * "sigma11, sigma22, sigma12": Normal stress and shear stress (strain) in the x-y plane.
            * "p1, p2": Principal stresses (strains).
            * "sigma_vm": Von Mises stress.
            * "tau_max": Maximum shear stress (strains).
            * If None, defaults to "sigma_vm".

        #. For ``Brick`` or ``Solid`` elements, one of ["sigma11", "sigma22", "sigma33", "sigma12", "sigma23", "sigma13", "p1", "p2", "p3", "sigma_vm", "tau_max", "sigma_oct", "tau_oct"]

            * "sigma11, sigma22, sigma33": Normal stress (strain) along x, y, z.
            * "sigma12, sigma23, sigma13": Shear stress (strain).
            * "p1, p2, p3": Principal stresses (strains).
            * "sigma_vm": Von Mises stress.
            * "tau_max": Maximum shear stress (strains).
            * "sigma_oct": Octahedral normal stress (strains).
            * "tau_oct": Octahedral shear stress (strains).
            * If None, defaults to "sigma_vm".

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
    show_values: bool, default: False
        Whether to display the response value by hover.
        Set to False can improve the performance of the visualization.
    show_outline: bool, default: False
        Whether to display the outline of the model.
    show_defo: bool, default: False
        Whether to display the deformed shape.
    defo_scale: float, default: 1.0
        Scales the size of the deformation presentation when show_defo is True.
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
    ele_type, resp_type, resp_dof = _check_input(ele_type, resp_type, resp_dof)
    model_info_steps, model_update, resp_step = loadODB(odb_tag, resp_type=ele_type)
    if show_defo:
        _, _, node_resp_steps = loadODB(odb_tag, resp_type="Nodal", verbose=False)
    else:
        node_resp_steps = None
    plotbase = PlotUnstruResponse(model_info_steps, resp_step, model_update, node_resp_steps=node_resp_steps)
    plotbase.set_unit(symbol=unit_symbol, factor=unit_factor)
    plotbase.refactor_resp_step(ele_tags=ele_tags, ele_type=ele_type, resp_type=resp_type, component=resp_dof)
    if slides:
        plotbase.plot_slide(
            ele_tags=ele_tags,
            style=style,
            show_values=show_values,
            show_defo=show_defo,
            defo_scale=defo_scale,
            plot_all_mesh=show_model,
            show_bc=show_bc,
            bc_scale=bc_scale,
            show_mp_constraint=show_mp_constraint,
        )
    else:
        plotbase.plot_peak_step(
            ele_tags=ele_tags,
            step=step,
            style=style,
            show_values=show_values,
            show_defo=show_defo,
            defo_scale=defo_scale,
            plot_all_mesh=show_model,
            show_bc=show_bc,
            bc_scale=bc_scale,
            show_mp_constraint=show_mp_constraint,
        )
    return plotbase.update_fig(show_outline)


def plot_unstruct_responses_animation(
    odb_tag: Union[int, str] = 1,
    ele_tags: Optional[Union[int, list]] = None,
    framerate: Optional[int] = None,
    ele_type: str = "Shell",
    resp_type: Optional[str] = None,
    resp_dof: Optional[str] = None,
    unit_symbol: Optional[str] = None,
    unit_factor: Optional[float] = None,
    style: str = "surface",
    show_outline: bool = False,
    show_values: bool = False,
    show_defo: bool = True,
    defo_scale: float = 1.0,
    show_bc: bool = True,
    bc_scale: float = 1.0,
    show_mp_constraint: bool = False,
    show_model: bool = True,
) -> go.Figure:
    """Unstructured element (Shell, Plane, Brick) response animation.

    .. Note::
        The responses at all Gaussian points are averaged.

    Parameters
    ----------
    odb_tag: Union[int, str], default: 1
        Tag of output databases (ODB) to be visualized.
    ele_tags: Union[int, list], default: None
        The tags of truss elements to be visualized. If None, all truss elements are selected.
    ele_type: str, default: "Shell"
        Element type, optional, one of ["Shell", "Plane", "Solid"].
    framerate: int, default: None
        Framerate for the display, i.e., the number of frames per second.
    resp_type: str, default: None
        Response type, which dependents on the element type `ele_type`.

        #. For ``Shell`` elements, one of ["sectionForces", "sectionDeformations"].
            I.e., section forces and deformations at Gaussian integration points (per unit length).
            If None, defaults to "sectionForces".
        #. For ``Plane`` elements, one of ["stresses", "strains"].
            I.e., stresses and strains at Gaussian integration points.
            If None, defaults to "stresses".
        #. For ``Brick`` or ``Solid`` elements, one of ["stresses", "strains"].
            I.e., stresses and strains at Gaussian integration points.
            If None, defaults to "stresses".

    resp_dof: str, default: None
        Dof to be visualized, which dependents on the element type `ele_type`.

        .. Note::
            The `resp_dof` here is consistent with stress-strain (force-deformation),
            and whether it is stress or strain depends on the parameter `resp_type`.

        #. For ``Shell`` elements, one of ["FXX", "FYY", "FXY", "MXX", "MYY", "MXY", "VXZ", "VYZ"].
            If None, defaults to "MXX".
        #. For ``Plane`` elements, one of ["sigma11", "sigma22", "sigma12", "p1", "p2", "sigma_vm", "tau_max"].

            * "sigma11, sigma22, sigma12": Normal stress and shear stress (strain) in the x-y plane.
            * "p1, p2": Principal stresses (strains).
            * "sigma_vm": Von Mises stress.
            * "tau_max": Maximum shear stress (strains).
            * If None, defaults to "sigma_vm".

        #. For ``Brick`` or ``Solid`` elements, one of ["sigma11", "sigma22", "sigma33", "sigma12", "sigma23", "sigma13", "p1", "p2", "p3", "sigma_vm", "tau_max", "sigma_oct", "tau_oct"]

            * "sigma11, sigma22, sigma33": Normal stress (strain) along x, y, z.
            * "sigma12, sigma23, sigma13": Shear stress (strain).
            * "p1, p2, p3": Principal stresses (strains).
            * "sigma_vm": Von Mises stress.
            * "tau_max": Maximum shear stress (strains).
            * "sigma_oct": Octahedral normal stress (strains).
            * "tau_oct": Octahedral shear stress (strains).
            * If None, defaults to "sigma_vm".

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
    show_values: bool, default: False
        Whether to display the response value by hover.
        Set to False can improve the performance of the visualization.
    show_outline: bool, default: False
        Whether to display the outline of the model.
    show_defo: bool, default: False
        Whether to display the deformed shape.
    defo_scale: float, default: 1.0
        Scales the size of the deformation presentation when show_defo is True.
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
    ele_type, resp_type, resp_dof = _check_input(ele_type, resp_type, resp_dof)
    model_info_steps, model_update, resp_step = loadODB(odb_tag, resp_type=ele_type)
    if show_defo:
        _, _, node_resp_steps = loadODB(odb_tag, resp_type="Nodal", verbose=False)
    else:
        node_resp_steps = None
    plotbase = PlotUnstruResponse(model_info_steps, resp_step, model_update, node_resp_steps=node_resp_steps)
    plotbase.set_unit(symbol=unit_symbol, factor=unit_factor)
    plotbase.refactor_resp_step(ele_tags=ele_tags, ele_type=ele_type, resp_type=resp_type, component=resp_dof)
    plotbase.plot_anim(
        ele_tags=ele_tags,
        framerate=framerate,
        style=style,
        show_values=show_values,
        show_defo=show_defo,
        defo_scale=defo_scale,
        plot_all_mesh=show_model,
        show_bc=show_bc,
        bc_scale=bc_scale,
        show_mp_constraint=show_mp_constraint,
    )
    return plotbase.update_fig(show_outline)


def _check_input(ele_type, resp_type, resp_dof):
    if ele_type.lower() == "shell":
        ele_type = "Shell"
        resp_type, resp_dof = _check_input_shell(resp_type, resp_dof)
    elif ele_type.lower() == "plane":
        ele_type = "Plane"
        resp_type, resp_dof = _check_input_plane(resp_type, resp_dof)
    elif ele_type.lower() in ["brick", "solid"]:
        ele_type = "Brick"
        resp_type, resp_dof = _check_input_solid(resp_type, resp_dof)
    else:
        raise ValueError(f"Not supported element type {ele_type}! Valid options are: Shell, Plane, Brick.")  # noqa: TRY003
    return ele_type, resp_type, resp_dof


def _check_input_shell(resp_type, resp_dof):
    if resp_type is None:
        resp_type = "sectionForces"
    if resp_type.lower() in ["sectionforces", "forces", "sectionforce", "force"]:
        resp_type = "sectionForces"
    elif resp_type.lower() in [
        "sectiondeformations",
        "sectiondeformation",
        "secdeformations",
        "secdeformation",
        "deformations",
        "deformation",
        "defo",
        "secdefo",
    ]:
        resp_type = "sectionDeformations"
    else:
        raise ValueError(  # noqa: TRY003
            f"Not supported response type {resp_type}! Valid options are: sectionForces, sectionDeformations."
        )
    if resp_dof is None:
        resp_dof = "MXX"
    if resp_dof.lower() not in [
        "fxx",
        "fyy",
        "fxy",
        "mxx",
        "myy",
        "mxy",
        "vxz",
        "vyz",
    ]:
        raise ValueError(  # noqa: TRY003
            f"Not supported component {resp_dof}! Valid options are: FXX, FYY, FXY, MXX, MYY, MXY, VXZ, VYZ."
        )
    return resp_type, resp_dof


def _check_input_plane(resp_type, resp_dof):
    if resp_type is None:
        resp_type = "Stresses"
    if resp_type.lower() in ["stresses", "stress"]:
        if resp_dof is None:
            resp_dof = "sigma_vm"
        if resp_dof.lower() in ["p1", "p2", "sigma_vm", "tau_max"]:
            resp_type = "stressMeasures"
        elif resp_dof.lower() in ["sigma11", "sigma22", "sigma12"]:
            resp_type = "Stresses"
        else:
            raise ValueError(  # noqa: TRY003
                f"Not supported component {resp_dof}! "
                "Valid options are: sigma11, sigma22, sigma12, p1, p2, sigma_vm, tau_max."
            )
    elif resp_type.lower() in ["strains", "strain"]:
        if resp_dof is None:
            resp_dof = "sigma_vm"
        if resp_dof.lower() in ["p1", "p2", "sigma_vm", "tau_max"]:
            resp_type = "strainMeasures"
        elif resp_dof.lower() in ["sigma11", "sigma22", "sigma12"]:
            resp_type = "Strains"
            resp_dof = resp_dof.replace("sigma", "eps")
        else:
            raise ValueError(  # noqa: TRY003
                f"Not supported component {resp_dof}! "
                "Valid options are: sigma11, sigma22, sigma12, p1, p2, sigma_vm, tau_max."
            )
    else:
        raise ValueError(f"Not supported response type {resp_type}! Valid options are: Stresses, Strains.")  # noqa: TRY003
    return resp_type, resp_dof


def _check_input_solid(resp_type, resp_dof):
    if resp_type is None:
        resp_type = "Stresses"
    if resp_type.lower() in ["stresses", "stress"]:
        if resp_dof is None:
            resp_dof = "sigma_vm"
        if resp_dof.lower() in ["p1", "p2", "p3", "sigma_vm", "tau_max", "sigma_oct", "tau_oct"]:
            resp_type = "stressMeasures"
        elif resp_dof.lower() in ["sigma11", "sigma22", "sigma33", "sigma12", "sigma23", "sigma13"]:
            resp_type = "Stresses"
        else:
            raise ValueError(  # noqa: TRY003
                f"Not supported component {resp_dof}! "
                "Valid options are: sigma11, sigma22, sigma33, sigma12, sigma23, sigma13, "
                "p1, p2, p3, sigma_vm, tau_max, sigma_oct, tau_oct!"
            )
    elif resp_type.lower() in ["strains", "strain"]:
        if resp_dof is None:
            resp_dof = "sigma_vm"
        if resp_dof.lower() in ["p1", "p2", "p3", "sigma_vm", "tau_max", "sigma_oct", "tau_oct"]:
            resp_type = "strainMeasures"
        elif resp_dof.lower() in ["sigma11", "sigma22", "sigma33", "sigma12", "sigma23", "sigma13"]:
            resp_type = "Strains"
            resp_dof = resp_dof.replace("sigma", "eps")
        else:
            raise ValueError(  # noqa: TRY003
                f"Not supported component {resp_dof}! "
                "Valid options are: sigma11, sigma22, sigma12, p1, p2, sigma_vm, tau_max."
            )
    else:
        raise ValueError(f"Not supported response type {resp_type}! Valid options are: Stresses, Strains.")  # noqa: TRY003
    return resp_type, resp_dof
