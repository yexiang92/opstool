from typing import Optional, Union

import numpy as np
import plotly.graph_objs as go

from ...post import loadODB
from .plot_resp_base import PlotResponseBase
from .plot_utils import _plot_lines, _plot_points_cmap, _plot_unstru_cmap


class PlotFrameResponse(PlotResponseBase):
    def __init__(self, model_info_steps, beam_resp_step, model_update):
        super().__init__(model_info_steps, beam_resp_step, model_update)
        self.resp_factor = 1.0
        self.plot_axis = None
        self.plot_axis_sign = 1.0
        self.sec_locs = None

        self.component_type = None

        title = f"{self.PKG_NAME} :: Frame Responses 3D Viewer</span></b><br><br>"
        self.title = {"text": title, "font": {"size": self.pargs.title_font_size}}

    def _set_comp_resp_type(self, resp_type, component):
        if resp_type.lower() in ["localforces", "localforce"]:
            self.resp_type = "localForces"
        elif resp_type.lower() in ["basicforces", "basicforce"]:
            self.resp_type = "basicForces"
        elif resp_type.lower() in ["basicdeformations", "basicdeformation", "basicdefo"]:
            self.resp_type = "basicDeformations"
        elif resp_type.lower() in ["plasticdeformation", "plasticdeformations", "plasticdefo"]:
            self.resp_type = "plasticDeformation"
        elif resp_type.lower() in ["sectionforces", "sectionforce"]:
            self.resp_type = "sectionForces"
        elif resp_type.lower() in ["sectiondeformations", "sectiondeformation", "sectiondefo"]:
            self.resp_type = "sectionDeformations"
        else:
            raise ValueError(  # noqa: TRY003
                f"Invalid response type: {resp_type}. "
                "Valid options are: localForces, basicForces, basicDeformations, "
                "plasticDeformations, sectionForces, sectionDeformations."
            )
        # component type
        self.component_type = component.upper()
        if self.resp_type == "localForces":
            self._set_comp_type_local(component)
        elif self.resp_type in ["basicForces", "basicDeformations", "plasticDeformation"]:
            self._set_comp_type_basic(component)
        else:
            self._set_comp_type_section(component)

    def _set_comp_type_local(self, comp_type):
        if comp_type.upper() == "FX":
            self.component = ["FX1", "FX2"]
            self.resp_factor = np.array([-1.0, 1.0])
            self.plot_axis = "y"
            self.plot_axis_sign = 1.0
        elif comp_type.upper() == "FY":
            self.component = ["FY1", "FY2"]
            self.resp_factor = np.array([-1.0, 1.0])
            self.plot_axis = "y"
            self.plot_axis_sign = 1.0
        elif comp_type.upper() == "FZ":
            self.component = ["FZ1", "FZ2"]
            self.resp_factor = np.array([-1.0, 1.0])
            self.plot_axis = "z"
            self.plot_axis_sign = 1.0
        elif comp_type.upper() == "MX":
            self.component = ["MX1", "MX2"]
            self.resp_factor = np.array([-1.0, 1.0])
            self.plot_axis = "y"
            self.plot_axis_sign = 1.0
        elif comp_type.upper() == "MY":
            self.component = ["MY1", "MY2"]
            self.plot_axis = "z"
            self.plot_axis_sign = -1.0
            self.resp_factor = np.array([1.0, -1.0])
        elif comp_type.upper() == "MZ":
            self.component = ["MZ1", "MZ2"]
            self.resp_factor = np.array([-1.0, 1.0])
            self.plot_axis = "y"
            self.plot_axis_sign = -1.0
        else:
            raise ValueError(  # noqa: TRY003
                f"Invalid component type for localForces: {comp_type}. Valid options are: FX, FY, FZ, MX, MY, MZ."
            )

    def _set_comp_type_basic(self, comp_type):
        if comp_type.upper() == "N":
            self.component = ["N", "N"]
            self.plot_axis = "y"
            self.plot_axis_sign = 1.0
        elif comp_type.upper() == "MZ":
            self.component = ["MZ1", "MZ2"]
            self.resp_factor = np.array([-1.0, 1.0])
            self.plot_axis = "y"
            self.plot_axis_sign = -1.0
        elif comp_type.upper() == "MY":
            self.component = ["MY1", "MY2"]
            self.resp_factor = np.array([1.0, -1.0])
            self.plot_axis = "z"
            self.plot_axis_sign = -1.0
        elif comp_type.upper() == "T":
            self.component = ["T", "T"]
            self.plot_axis = "y"
            self.plot_axis_sign = 1.0
        else:
            raise ValueError(  # noqa: TRY003
                f"Invalid component type for {self.resp_type}: {comp_type}. Valid options are: N, MZ, MY, T."
            )

    def _set_comp_type_section(self, comp_type):
        if comp_type.upper() in ["N", "MZ", "VY", "T"]:
            self.component = comp_type.upper()
            self.plot_axis = "y"
            self.plot_axis_sign = -1.0
        elif comp_type.upper() in ["VZ", "MY"]:
            self.component = comp_type.upper()
            self.plot_axis = "z"
            self.plot_axis_sign = 1.0
        else:
            raise ValueError(  # noqa: TRY003
                f"Invalid component type for {self.resp_type}: {comp_type}. Valid options are: N, MZ, VY, MY, VZ, T."
            )

    def _get_beam_data(self, step):
        return self._get_model_da("BeamData", step)

    def _make_frame_info(self, ele_tags, step):
        pos = self._get_node_da(step).to_numpy()
        beam_data = self._get_beam_data(step)
        beam_node_coords = []
        beam_cells = []
        if ele_tags is None:
            beam_tags = beam_data.coords["eleTags"].values
            beam_cells_orign = beam_data.loc[:, ["numNodes", "nodeI", "nodeJ"]].to_numpy().astype(int)
            yaxis = beam_data.loc[:, ["yaxis-x", "yaxis-y", "yaxis-z"]]
            zaxis = beam_data.loc[:, ["zaxis-x", "zaxis-y", "zaxis-z"]]
            for i, cell in enumerate(beam_cells_orign):
                nodei, nodej = cell[1:]
                beam_node_coords.append(pos[int(nodei)])
                beam_node_coords.append(pos[int(nodej)])
                beam_cells.append([2, 2 * i, 2 * i + 1])
        else:
            beam_tags = np.atleast_1d(ele_tags)
            beam_info = beam_data.sel(eleTags=beam_tags)
            yaxis, zaxis = [], []
            for i, etag in enumerate(beam_tags):
                nodei, nodej = beam_info.loc[etag, ["nodeI", "nodeJ"]]
                beam_node_coords.append(pos[int(nodei)])
                beam_node_coords.append(pos[int(nodej)])
                beam_cells.append([2, 2 * i, 2 * i + 1])
                yaxis.append(beam_info.loc[etag, ["yaxis-x", "yaxis-y", "yaxis-z"]])
                zaxis.append(beam_info.loc[etag, ["zaxis-x", "zaxis-y", "zaxis-z"]])
        beam_node_coords = np.array(beam_node_coords)
        yaxis, zaxis = np.array(yaxis), np.array(zaxis)
        return beam_tags, beam_node_coords, beam_cells, yaxis, zaxis

    def _get_sec_loc(self, step):
        sec_loc = self._get_resp_da(step, "sectionLocs", "alpha")
        return sec_loc

    def refactor_resp_data(self, ele_tags, resp_type, component):
        self._set_comp_resp_type(resp_type, component)
        resps, sec_locs = [], []
        if self.ModelUpdate or ele_tags is not None:
            for i in range(self.num_steps):
                beam_tags, _, _, _, _ = self._make_frame_info(ele_tags, i)
                da = self._get_resp_da(i, self.resp_type, self.component)
                da = da.sel(eleTags=beam_tags) * self.resp_factor
                resps.append(da)
                sec_da = self._get_sec_loc(i)
                sec_locs.append(sec_da.sel(eleTags=beam_tags))
        else:
            for i in range(self.num_steps):
                da = self._get_resp_da(i, self.resp_type, self.component)
                resps.append(da * self.resp_factor)
                sec_da = self._get_sec_loc(i)
                sec_locs.append(sec_da)

        self.resp_step = resps
        self.sec_locs = [loc / self.unit_factor for loc in sec_locs] if self.unit_factor else sec_locs

    def _get_resp_scale_factor(self, idx="absMax"):
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
        cmin, cmax = self._get_resp_clim()
        return float(alpha_), step, (cmin, cmax)

    def _get_resp_clim(self):
        maxv = [np.max(data) for data in self.resp_step]
        minv = [np.min(data) for data in self.resp_step]
        cmin, cmax = np.min(minv), np.max(maxv)
        self.clim = (cmin, cmax)
        return cmin, cmax

    @staticmethod
    def _set_segment_mesh(
        pos_bot, pos_top, force, resp_points: list, resp_cells: list, resp_celltypes: list, scalars: list
    ):
        for i in range(len(force) - 1):
            if force[i] * force[i + 1] >= 0:
                resp_cells.append([
                    4,
                    len(resp_points),
                    len(resp_points) + 1,
                    len(resp_points) + 2,
                    len(resp_points) + 3,
                ])
                resp_points.extend([pos_bot[i], pos_bot[i + 1], pos_top[i + 1], pos_top[i]])
                scalars.extend([force[i], force[i + 1], force[i + 1], force[i]])
                resp_celltypes.append(9)  # 9 for quad
            else:
                t = force[i] / (force[i] - force[i + 1])
                coord0 = pos_bot[i] + t * (pos_bot[i + 1] - pos_bot[i])
                resp_cells.append([
                    4,
                    len(resp_points),
                    len(resp_points) + 1,
                    len(resp_points) + 1,
                    len(resp_points) + 4,
                ])
                resp_cells.append([
                    4,
                    len(resp_points) + 2,
                    len(resp_points) + 1,
                    len(resp_points) + 1,
                    len(resp_points) + 3,
                ])
                resp_points.extend([pos_bot[i], coord0, pos_bot[i + 1], pos_top[i + 1], pos_top[i]])
                scalars.extend([force[i], 0.0, force[i + 1], force[i + 1], force[i]])
                resp_celltypes.extend([9, 9])  # 9 for quad

    def _get_resp_mesh(self, beam_node_coords, beam_cells, sec_locs, resp, resp_scale, axis_data):
        resp_points, resp_cells, resp_celltypes, scalars = [], [], [], []
        plot_points, plot_scalars = [], []
        resp, resp_scale, sec_locs = (resp.to_numpy(), resp_scale.to_numpy(), sec_locs.to_numpy())
        for i, cell in enumerate(beam_cells):
            axis = axis_data[i]
            node1, node2 = cell[1:]
            coord1, coord2 = beam_node_coords[node1], beam_node_coords[node2]
            if self.resp_type in [
                "localForces",
                "basicForces",
                "basicDeformations",
                "plasticDeformation",
            ]:
                f1, f2 = resp_scale[i]
                f1_, f2_ = resp[i]
                locs = np.linspace(0, 1, 11)
                force = np.interp(locs, [0, 1], [f1_, f2_])
                force_scale = np.interp(locs, [0, 1], [f1, f2])
            else:
                locs = sec_locs[i][~np.isnan(sec_locs[i])]
                force = resp[i][~np.isnan(resp[i])]
                force_scale = resp_scale[i][~np.isnan(resp_scale[i])]
            pos1 = np.array([coord1 + loc * (coord2 - coord1) for loc in locs])
            pos2 = [coord + force_scale[i] * axis * self.plot_axis_sign for i, coord in enumerate(pos1)]
            pos2 = np.array(pos2)
            self._set_segment_mesh(pos1, pos2, force, resp_points, resp_cells, resp_celltypes, scalars)
            plot_points.extend(pos2)
            plot_scalars.extend(force)
        resp_points = np.array(resp_points)
        scalars = np.array(scalars)
        resp_cells = np.array(resp_cells)
        resp_celltypes = np.array(resp_celltypes)
        plot_points = np.array(plot_points)
        plot_scalars = np.array(plot_scalars)
        return resp_points, resp_cells, resp_celltypes, scalars, plot_points, plot_scalars

    def _create_mesh(
        self,
        plotter,
        value,
        ele_tags=None,
        alpha=1.0,
        show_values=True,
        plot_all_mesh=True,
        clim=None,
        line_width=1.0,
        coloraxis="coloraxis",
        style="surface",
        opacity=1.0,
        show_bc: bool = True,
        bc_scale: float = 1.0,
        show_mp_constraint: bool = True,
        color=None,
    ):
        step = round(value)
        resp = self.resp_step[step]
        resp_scale = resp * alpha
        beam_tags, beam_node_coords, beam_cells, yaxis, zaxis = self._make_frame_info(ele_tags, step)
        axis_data = yaxis if self.plot_axis == "y" else zaxis
        sec_locs = self.sec_locs[step]
        resp_points, resp_cells, resp_celltypes, scalars, plot_points, plot_scalars = self._get_resp_mesh(
            beam_node_coords, beam_cells, sec_locs, resp, resp_scale, axis_data
        )
        #  ---------------------------------
        if plot_all_mesh:
            self._plot_all_mesh(plotter, step=step)
        #  ---------------------------------
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
            plot_points,
            scalars=plot_scalars,
            clim=clim,
            coloraxis=coloraxis,
            name=self.resp_type,
            size=self.pargs.point_size,
            show_hover=show_values,
            color=color,
        )

        line_points, line_mid_points = self._get_plotly_line_data(beam_node_coords, beam_cells, scalars=None)
        _plot_lines(
            plotter,
            pos=line_points,
            width=self.pargs.line_width,
            color=self.pargs.color_beam,
            name="Frame",
            hoverinfo="skip",
        )

        if show_bc:
            self._plot_bc(plotter=plotter, step=step, defo_scale=0.0, bc_scale=bc_scale)
        if show_mp_constraint:
            self._plot_mp_constraint(plotter, step=step, defo_scale=0.0)

    def _make_txt(self, step, add_title=False):
        resp = self.resp_step[step].data
        maxv, minv = np.max(resp), np.min(resp)
        t_ = self.time[step]

        title = f"<b>{self._set_txt_props(self.resp_type)} *</b><br>"
        comp = self.component if isinstance(self.component, str) else " ".join(self.component)
        title += f"<b>(DOF) {self._set_txt_props(comp)} *</b><br>"
        if self.unit_symbol:
            unit_txt = self._set_txt_props(self.unit_symbol)
            title += f"<b>(unit) {unit_txt}</b><br>"
        maxv = self._set_txt_props(f"{maxv:.3E}")
        minv = self._set_txt_props(f"{minv:.3E}")
        title += f"<b>max:</b> {maxv}<br><b>min:</b> {minv}<br>"
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
        alpha=1.0,
        resp_type=None,
        component=None,
        show_values=True,
        line_width=1.0,
        plot_all_mesh=True,
        style="surface",
        opacity=1.0,
        show_bc: bool = True,
        bc_scale: float = 1.0,
        show_mp_constraint: bool = True,
        color=None,
    ):
        self.refactor_resp_data(ele_tags, resp_type, component)
        alpha_, maxstep, clim = self._get_resp_scale_factor(idx="absMax")
        ndatas = []
        ndata_cum = 0
        for i in range(self.num_steps):
            plotter = []
            self._create_mesh(
                plotter,
                i,
                alpha=alpha_ * alpha,
                clim=clim,
                ele_tags=ele_tags,
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
        alpha=1.0,
        resp_type=None,
        component=None,
        show_values=True,
        line_width=1.0,
        plot_all_mesh=True,
        style="surface",
        opacity=1.0,
        show_bc: bool = True,
        bc_scale: float = 1.0,
        show_mp_constraint: bool = True,
        color=None,
    ):
        self.refactor_resp_data(ele_tags, resp_type, component)
        alpha_, maxstep, clim = self._get_resp_scale_factor(idx=step)
        plotter = []
        self._create_mesh(
            plotter,
            maxstep,
            alpha=alpha_ * alpha,
            clim=clim,
            ele_tags=ele_tags,
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
        txt = self._make_txt(maxstep)
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
        alpha=1.0,
        resp_type=None,
        component=None,
        show_values=True,
        framerate: Optional[int] = None,
        line_width=1.0,
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
        self.refactor_resp_data(ele_tags, resp_type, component)
        alpha_, maxstep, clim = self._get_resp_scale_factor()
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
                alpha=alpha_ * alpha,
                clim=clim,
                ele_tags=ele_tags,
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
            ele_tags=ele_tags,
            alpha=alpha_,
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
        self._update_antimate_layout(duration=duration, cbar_title=self.component, showscale=showscale)


def plot_frame_responses(
    odb_tag: Union[int, str] = 1,
    ele_tags: Optional[Union[int, list]] = None,
    resp_type: str = "sectionForces",
    resp_dof: str = "MZ",
    slides: bool = False,
    step: Union[int, str] = "absMax",
    scale: float = 1.0,
    show_values: bool = False,
    line_width: float = 5.0,
    show_outline: bool = False,
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
    """Plot the responses of the frame element.

    Parameters
    ----------
    odb_tag: Union[int, str], default: 1
        Tag of output databases (ODB) to be visualized.
    ele_tags: Union[int, list], default: None
        The tags of frame elements to be visualized. If None, all frame elements are selected.
    resp_type: str, default: "sectionforces"
        Response type, optional, one of ["localForces", "basicForces", "basicDeformations",
        "plasticDeformation", "sectionForces", "sectionDeformations"].
    resp_dof: str, default: "MZ"
        Component type corrsponding to the resp_type.

        - For `localForces`: ["FX", "FY", "FZ", "MX", "MY", "MZ"]
        - For `basicForces`: ["N", "MZ", "MY", "T"]
        - For `basicDeformations`: ["N", "MZ", "MY", "T"]
        - For `plasticDeformation`: ["N", "MZ", "MY", "T"]
        - For `sectionForces`: ["N", "MZ", "VY", "MY", "VZ", "T"]
        - For `sectionDeformations`: ["N", "MZ", "VY", "MY", "VZ", "T"]

        .. Note::
            For `sectionForces` and `sectionDeformations`,
            not all sections include the shear dof VY and VZ.
            For instance, in the most commonly used 3D fiber cross-sections,
            only the axial force N, bending moments MZ and MY, and torsion T are available.

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
    scale: float, default: 1.0
        Scale the size of the response graph.

        .. Note::
            You can adjust the scale to make the response graph more visible.
            A negative number will reverse the direction.

    line_width: float, default: 1.5.
        Line width of the response graph.
    show_outline: bool, default: False
        Whether to display the outline of the model.
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
    model_info_steps, model_update, beam_resp_steps = loadODB(odb_tag, resp_type="Frame")
    plotbase = PlotFrameResponse(model_info_steps, beam_resp_steps, model_update)
    plotbase.set_unit(symbol=unit_symbol, factor=unit_factor)
    if slides:
        plotbase.plot_slide(
            ele_tags=ele_tags,
            alpha=scale,
            show_values=show_values,
            resp_type=resp_type,
            component=resp_dof,
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
            ele_tags=ele_tags,
            step=step,
            alpha=scale,
            show_values=show_values,
            resp_type=resp_type,
            component=resp_dof,
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


def plot_frame_responses_animation(
    odb_tag: Union[int, str] = 1,
    ele_tags: Optional[Union[int, list]] = None,
    resp_type: str = "sectionForces",
    resp_dof: str = "MZ",
    scale: float = 1.0,
    show_values: bool = False,
    framerate: Optional[int] = None,
    line_width: float = 1.5,
    show_outline: bool = False,
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
    """Animate the responses of frame elements.

    Parameters
    ----------
    odb_tag: Union[int, str], default: 1
        Tag of output databases (ODB) to be visualized.
    ele_tags: Union[int, list], default: None
        The tags of frame elements to be visualized. If None, all frame elements are selected.
    resp_type: str, default: "sectionforces"
        Response type, optional, one of ["localForces", "basicForces", "basicDeformations",
        "plasticDeformation", "sectionForces", "sectionDeformations"].
    resp_dof: str, default: "MZ"
        Component type corrsponding to the resp_type.

        - For `localForces`: ["FX", "FY", "FZ", "MX", "MY", "MZ"]
        - For `basicForces`: ["N", "MZ", "MY", "T"]
        - For `basicDeformations`: ["N", "MZ", "MY", "T"]
        - For `plasticDeformation`: ["N", "MZ", "MY", "T"]
        - For `sectionForces`: ["N", "MZ", "VY", "MY", "VZ", "T"]
        - For `sectionDeformations`: ["N", "MZ", "VY", "MY", "VZ", "T"]

        .. Note::
            For `sectionForces` and `sectionDeformations`,
            not all sections include the shear dof VY and VZ.
            For instance, in the most commonly used 3D fiber cross-sections,
            only the axial force N, bending moments MZ and MY, and torsion T are available.

    scale: float, default: 1.0
        Scale the size of the response graph.

        .. Note::
            You can adjust the scale to make the response graph more visible.
            A negative number will reverse the direction.

    show_values: bool, default: False
        Whether to display the response value by hover.
        Set to False can improve the performance of the visualization.
    framerate: int, default: None
        Framerate for the display, i.e., the number of frames per second.
    line_width: float, default: 1.5.
        Line width of the response graph.
    show_outline: bool, default: False
        Whether to display the outline of the model.
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
    model_info_steps, model_update, beam_resp_steps = loadODB(odb_tag, resp_type="Frame")
    plotbase = PlotFrameResponse(model_info_steps, beam_resp_steps, model_update)
    plotbase.set_unit(symbol=unit_symbol, factor=unit_factor)
    plotbase.plot_anim(
        ele_tags=ele_tags,
        alpha=scale,
        show_values=show_values,
        resp_type=resp_type,
        component=resp_dof,
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
