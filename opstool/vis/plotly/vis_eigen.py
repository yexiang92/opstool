from typing import Optional, Union

import numpy as np
import plotly.graph_objs as go
from plotly.subplots import make_subplots

from ...post import load_eigen_data, load_linear_buckling_data
from ...utils import CONFIGS
from .plot_resp_base import PlotResponseBase
from .plot_utils import (
    PLOT_ARGS,
    _get_line_cells,
    _get_unstru_cells,
    _plot_lines_cmap,
    _plot_unstru_cmap,
)
from .vis_model import PlotModelBase

PKG_NAME = CONFIGS.get_pkg_name()
SHAPE_MAP = CONFIGS.get_shape_map()


class PlotEigenBase(PlotResponseBase):
    def __init__(self, model_info, modal_props, eigen_vectors):
        self.nodal_data = model_info["NodalData"]
        self.nodal_tags = self.nodal_data.coords["nodeTags"]
        self.points = self.nodal_data.to_numpy()
        self.ndims = self.nodal_data.attrs["ndims"]
        self.bounds = self.nodal_data.attrs["bounds"]
        self.min_bound_size = self.nodal_data.attrs["minBoundSize"]
        self.max_bound_size = self.nodal_data.attrs["maxBoundSize"]
        self.show_zaxis = not np.max(self.ndims) <= 2
        # -------------------------------------------------------------
        self.line_data = model_info["AllLineElesData"]
        self.line_cells, self.line_tags = _get_line_cells(self.line_data)
        # -------------------------------------------------------------
        self.unstru_data = model_info["UnstructuralData"]
        self.unstru_tags, self.unstru_cell_types, self.unstru_cells = _get_unstru_cells(self.unstru_data)
        # --------------------------------------------------
        self.ModelInfo = model_info
        self.ModalProps = modal_props
        self.EigenVectors = eigen_vectors
        self.plot_model_base = PlotModelBase(model_info, {})

        # plotly
        self.pargs = PLOT_ARGS
        self.FIGURE = go.Figure()

        self.title = {
            "font": {"family": "courier", "size": self.pargs.title_font_size},
            "text": f"<b>{PKG_NAME} :: Eigen 3D Viewer</b><br><br>",
        }

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
        txt = f'<span style="font-weight:bold; font-size:{self.pargs.title_font_size}px">Mode {step + 1}</span>'
        # txt = f"<b>Mode {step + 1}</b>"
        period_txt = self._set_txt_props(f"{1 / fi:.6f}; ", color="blue")
        txt += f"<br><b>Period (s):</b> {period_txt}"
        fi_txt = self._set_txt_props(f"{fi:.6f};", color="blue")
        txt += f"<b>Frequency (Hz):</b> {fi_txt}"
        if not self.show_zaxis:
            txt += "<br><b>Modal participation mass ratios (%)</b><br>"
            mx = self.ModalProps.loc[:, "partiMassRatiosMX"][step]
            my = self.ModalProps.loc[:, "partiMassRatiosMY"][step]
            rmz = self.ModalProps.loc[:, "partiMassRatiosRMZ"][step]
            txt += self._set_txt_props(f"{mx:7.3f} {my:7.3f} {rmz:7.3f}", color="blue")
            txt += "<br><b>Cumulative modal participation mass ratios (%)</b><br>"
            mx = self.ModalProps.loc[:, "partiMassRatiosCumuMX"][step]
            my = self.ModalProps.loc[:, "partiMassRatiosCumuMY"][step]
            rmz = self.ModalProps.loc[:, "partiMassRatiosCumuRMZ"][step]
            txt += self._set_txt_props(f"{mx:7.3f} {my:7.3f} {rmz:7.3f}", color="blue")
            txt += "<br><b>{:>7} {:>7} {:>7}</b>".format("X", "Y", "RZ")
        else:
            txt += "<br><b>Modal participation mass ratios (%)</b><br>"
            mx = self.ModalProps.loc[:, "partiMassRatiosMX"][step]
            my = self.ModalProps.loc[:, "partiMassRatiosMY"][step]
            mz = self.ModalProps.loc[:, "partiMassRatiosMZ"][step]
            rmx = self.ModalProps.loc[:, "partiMassRatiosRMX"][step]
            rmy = self.ModalProps.loc[:, "partiMassRatiosRMY"][step]
            rmz = self.ModalProps.loc[:, "partiMassRatiosRMZ"][step]
            txt += self._set_txt_props(f"{mx:7.3f} {my:7.3f} {mz:7.3f} {rmx:7.3f} {rmy:7.3f} {rmz:7.3f}", color="blue")
            txt += "<br><b>Cumulative modal participation mass ratios (%)</b><br>"
            mx = self.ModalProps.loc[:, "partiMassRatiosCumuMX"][step]
            my = self.ModalProps.loc[:, "partiMassRatiosCumuMY"][step]
            mz = self.ModalProps.loc[:, "partiMassRatiosCumuMZ"][step]
            rmx = self.ModalProps.loc[:, "partiMassRatiosCumuRMX"][step]
            rmy = self.ModalProps.loc[:, "partiMassRatiosCumuRMY"][step]
            rmz = self.ModalProps.loc[:, "partiMassRatiosCumuRMZ"][step]
            txt += self._set_txt_props(f"{mx:7.3f} {my:7.3f} {mz:7.3f} {rmx:7.3f} {rmy:7.3f} {rmz:7.3f}", color="blue")
            txt += f"<br><b>{'X':>7} {'Y':>7} {'Z':>7} {'RX':>7} {'RY':>7} {'RZ':>7}</b>"
            # f'<span style="color:blue; font-weight:bold;">{"X":>7} {"Y":>7} {"Z":>7} {"RX":>7} {"RY":>7} {"RZ":>7}</span>'
        return txt

    def _make_eigen_subplots_txt(self, step):
        f = self.ModalProps.loc[:, "eigenFrequency"]
        mode = self._set_txt_props(f"{step + 1}", color="#8eab12")
        period = 1 / f[step]
        t = self._set_txt_props(f"{period:.3E}") if period < 0.001 else self._set_txt_props(f"{period:.3f}")
        txt = f"Mode <b>{mode}</b>: T = <b>{t}</b> s"
        return txt

    def _create_mesh(
        self,
        plotter: list,
        idx,
        coloraxis,
        alpha=1.0,
        style="surface",
        show_origin=False,
        show_bc: bool = True,
        bc_scale: float = 1.0,
        show_mp_constraint: bool = True,
    ):
        step = round(idx) - 1
        eigen_points, scalars, alpha_ = self._get_eigen_points(step, alpha)

        if show_origin:
            self.plot_model_base.plot_model_one_color(
                plotter,
                color="gray",
                style="wireframe",
            )

        if len(self.unstru_data) > 0:
            (
                face_points,
                face_line_points,
                face_mid_points,
                veci,
                vecj,
                veck,
                face_scalars,
                face_line_scalars,
            ) = self._get_plotly_unstru_data(eigen_points, self.unstru_cell_types, self.unstru_cells, scalars)
            _plot_unstru_cmap(
                plotter,
                face_points,
                veci=veci,
                vecj=vecj,
                veck=veck,
                scalars=face_scalars,
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
        if len(self.line_data) > 0:
            line_points, line_mid_points, line_scalars = self._get_plotly_line_data(
                eigen_points, self.line_cells, scalars
            )
            _plot_lines_cmap(
                plotter,
                line_points,
                scalars=line_scalars,
                coloraxis=coloraxis,
                width=self.pargs.line_width,
            )
        if show_bc:
            bc_points = self._get_bc_points(step, scale=alpha_)
            self.plot_model_base.plot_bc(plotter, bc_scale, points_new=bc_points)
        if show_mp_constraint:
            self.plot_model_base.plot_mp_constraint(
                plotter,
                points_new=eigen_points,
            )

    def subplots(self, modei, modej, show_outline, **kargs):
        if modej - modei + 1 > 64:
            raise ValueError("When subplots True, mode_tag range must < 64 for clarify")  # noqa: TRY003
        shape = SHAPE_MAP[modej - modei + 1]
        specs = [[{"is_3d": True} for _ in range(shape[1])] for _ in range(shape[0])]
        subplot_titles = []
        for i, idx in enumerate(range(modei, modej + 1)):  # noqa: B007
            txt = self._make_eigen_subplots_txt(idx - 1)
            subplot_titles.append(txt)
        self.FIGURE = make_subplots(
            rows=shape[0],
            cols=shape[1],
            specs=specs,
            figure=self.FIGURE,
            print_grid=False,
            subplot_titles=subplot_titles,
            horizontal_spacing=0.07 / shape[1],
            vertical_spacing=0.1 / shape[0],
            column_widths=[1] * shape[1],
            row_heights=[1] * shape[0],
        )
        for i, idx in enumerate(range(modei, modej + 1)):
            idxi = int(np.ceil((i + 1) / shape[1]) - 1)
            idxj = int(i - idxi * shape[1])
            plotter = []
            self._create_mesh(plotter, idx, coloraxis=f"coloraxis{i + 1}", **kargs)
            self.FIGURE.add_traces(plotter, rows=idxi + 1, cols=idxj + 1)
        if not self.show_zaxis:
            scene = self._get_plotly_dim_scene(mode="2d", show_outline=show_outline)
        else:
            scene = self._get_plotly_dim_scene(mode="3d", show_outline=show_outline)
        scenes = {}
        coloraxiss = {}
        for k in range(shape[0] * shape[1]):
            coloraxiss[f"coloraxis{k + 1}"] = {"showscale": False, "colorscale": self.pargs.cmap}
            if k >= 1:
                if not self.show_zaxis:
                    scenes[f"scene{k + 1}"] = self._get_plotly_dim_scene(mode="2d", show_outline=show_outline)
                else:
                    scenes[f"scene{k + 1}"] = self._get_plotly_dim_scene(mode="3d", show_outline=show_outline)
        self.FIGURE.update_layout(
            font={"family": self.pargs.font_family},
            template=self.pargs.theme,
            autosize=True,
            showlegend=False,
            coloraxis={"showscale": False, "colorscale": self.pargs.cmap},
            scene=scene,
            **scenes,
            **coloraxiss,
        )

        return self.FIGURE

    def plot_slides(self, modei, modej, **kargs):
        n_data = None
        for i, idx in enumerate(range(modei, modej + 1)):
            plotter = []
            self._create_mesh(plotter, idx, coloraxis=f"coloraxis{i + 1}", **kargs)
            self.FIGURE.add_traces(plotter)
            if i == 0:
                n_data = len(self.FIGURE.data)
        for i in range(n_data, len(self.FIGURE.data)):
            self.FIGURE.data[i].visible = False
        # Create and add slider
        steps = []
        for i, idx in enumerate(range(modei, modej + 1)):
            # txt = "Mode {}: T = {:.3f} s".format(idx, 1 / f[idx - 1])
            txt = self._make_eigen_txt(idx - 1)
            txt = {"font": {"family": "courier", "size": self.pargs.font_size}, "text": txt}
            step = {
                "method": "update",
                "args": [{"visible": [False] * len(self.FIGURE.data)}, {"title": txt}],  # layout attribute
                "label": str(idx),
            }
            step["args"][0]["visible"][n_data * i : n_data * (i + 1)] = [True] * n_data
            # Toggle i'th trace to "visible"
            steps.append(step)
        sliders = [
            {
                "active": 0,
                "currentvalue": {"prefix": "Mode: "},
                "pad": {"t": 50},
                "steps": steps,
            }
        ]
        coloraxiss = {}
        for i in range(modej - modei + 1):
            coloraxiss[f"coloraxis{i + 1}"] = {
                "colorscale": self.pargs.cmap,
                # cmin=cmins[i],
                # cmax=cmaxs[i],
                "showscale": False,
                "colorbar": {"tickfont": {"size": self.pargs.font_size - 2}},
            }
        self.FIGURE.update_layout(sliders=sliders, **coloraxiss)

        return self.FIGURE

    def plot_anim(
        self,
        mode_tag: int = 1,
        n_cycle: int = 5,
        framerate: int = 1,
        alpha: float = 1.0,
        **kargs,
    ):
        alphas = [0.0] + [alpha, -alpha] * n_cycle
        duration = 1000 / framerate  # convert to milliseconds
        # ---------------------frames--------------------------------------------------------
        # start plot
        frames = []
        for k, alpha in enumerate(alphas):
            plotter = []
            self._create_mesh(plotter, mode_tag, alpha=alpha, coloraxis="coloraxis", **kargs)
            frames.append(go.Frame(data=plotter, name="step:" + str(k + 1)))
        # Add data to be displayed before animation starts
        plotter0 = []
        self._create_mesh(plotter0, mode_tag, alpha=alpha, coloraxis="coloraxis", **kargs)

        self.FIGURE = go.Figure(data=plotter0, frames=frames)

        # Layout
        txt = self._make_eigen_txt(mode_tag - 1)
        self.title["text"] += txt
        self.FIGURE.update_layout(
            coloraxis={"colorscale": self.pargs.cmap, "showscale": False},
        )
        self._update_antimate_layout(duration=duration, is_response_step=False)

    def plot_props_table(self, modei, modej):
        df = self.ModalProps.to_pandas()[modei - 1 : modej]
        df = df.T
        fig = go.Figure(
            data=[
                go.Table(
                    header={"values": ["modeTags", *list(df.columns)]},
                    cells={
                        "values": [df.index] + [df[col].tolist() for col in df.columns],
                        "format": [""] + [".3E"] * len(df.columns),
                    },
                )
            ]
        )
        return fig


class PlotBucklingBase(PlotEigenBase):
    def __init__(self, model_info, eigen_values, eigen_vectors):
        super().__init__(model_info, eigen_values, eigen_vectors)

    def _make_eigen_txt(self, step):
        mode = self._set_txt_props(f"{step + 1}", color="#8eab12")
        fi = self.ModalProps.isel(modeTags=step)
        fi = self._set_txt_props(f"{fi:.3E}") if fi < 0.001 else self._set_txt_props(f"{fi:.3f}")
        txt = f"Mode <b>{mode}</b>:<br>k = <b>{fi}</b>"
        return txt

    def _make_eigen_subplots_txt(self, step):
        return self._make_eigen_txt(step)


def plot_eigen(
    mode_tags: Union[list, tuple, int],
    odb_tag: Optional[Union[int, str]] = None,
    subplots: bool = False,
    scale: float = 1.0,
    show_outline: bool = False,
    show_origin: bool = False,
    style: str = "surface",
    show_bc: bool = True,
    bc_scale: float = 1.0,
    show_mp_constraint: bool = True,
    solver: str = "-genBandArpack",
    mode: str = "eigen",
) -> go.Figure:
    """Modal visualization.

    Parameters
    ----------
    mode_tags: Union[List, Tuple]
        The modal range to visualize, [mode i, mode j].
    odb_tag: Union[int, str], default: None
        Tag of output databases (ODB) to be visualized.
        If None, the data will be saved automatically.
    subplots: bool, default: False
        If True, multiple subplots are used to present mode i to mode j.
        Otherwise, they are presented as slides.
    scale: float, default: 1.0
        Zoom the presentation size of the mode shapes.
    show_outline: bool, default: False
        Whether to display the outline of the model.
    show_origin: bool, default: False
        Whether to show the undeformed shape.
    style: str, default: surface
        Visualization mesh style of surfaces and solids.
        One of the following: style='surface' or style='wireframe'
        Defaults to 'surface'. Note that 'wireframe' only shows a wireframe of the outer geometry.
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
    fig: `plotly.graph_objects.Figure <https://plotly.com/python-api-reference/generated/plotly.graph_objects.Figure.html>`_
        You can use `fig.show()` to display,
        You can also use `fig.write_html("path/to/file.html")` to save as an HTML file, see
        `Interactive HTML Export in Python <https://plotly.com/python/interactive-html-export/>`_
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
        plotbase.subplots(
            modei,
            modej,
            show_outline=show_outline,
            # link_views=link_views,
            alpha=scale,
            style=style,
            show_origin=show_origin,
            show_bc=show_bc,
            bc_scale=bc_scale,
            show_mp_constraint=show_mp_constraint,
        )
    else:
        plotbase.plot_slides(
            modei,
            modej,
            alpha=scale,
            style=style,
            show_origin=show_origin,
            show_bc=show_bc,
            bc_scale=bc_scale,
            show_mp_constraint=show_mp_constraint,
        )

    return plotbase.update_fig(show_outline=show_outline)


def plot_eigen_animation(
    mode_tag: int,
    odb_tag: Optional[Union[int, str]] = None,
    n_cycle: int = 5,
    framerate: int = 3,
    scale: float = 1.0,
    solver: str = "-genBandArpack",
    show_outline: bool = False,
    show_origin: bool = False,
    style: str = "surface",
    show_bc: bool = True,
    bc_scale: float = 1.0,
    show_mp_constraint: bool = True,
    mode: str = "eigen",
) -> go.Figure:
    """Modal animation visualization.

    Parameters
    ----------
    mode_tag: int
        The mode tag to display.
    odb_tag: Union[int, str], default: None
        Tag of output databases (ODB) to be visualized.
        If None, the data will be saved automatically.
    n_cycle: int, default: five
        Number of cycles for the display.
    framerate: int, default: three
        Framerate for the display, i.e., the number of frames per second.
    scale: float, default: 1.0
        Zoom the presentation size of the mode shapes.
    solver : str, optional,
       OpenSees' eigenvalue analysis solver, by default "-genBandArpack".
    show_outline: bool, default: False
        Whether to display the outline of the model.
    show_origin: bool, default: False
        Whether to show the undeformed shape.
    style: str, default: surface
        Visualization mesh style of surfaces and solids.
        One of the following: style='surface' or style='wireframe'
        Defaults to 'surface'. Note that 'wireframe' only shows a wireframe of the outer geometry.
    show_bc: bool, default: True
        Whether to display boundary supports.
    bc_scale: float, default: 1.0
        Scale the size of boundary support display.
    show_mp_constraint: bool, default: True
        Whether to show multipoint (MP) constraint.
    mode: str, default: eigen
        The type of modal analysis, can be "eigen" or "buckling".
        If "eigen", it will plot the eigenvalues and eigenvectors.
        If "buckling", it will plot the buckling factors and modes.
        Added in v0.1.15.

    Returns
    -------
    fig: `plotly.graph_objects.Figure <https://plotly.com/python-api-reference/generated/plotly.graph_objects.Figure.html>`_
        You can use `fig.show()` to display,
        You can also use `fig.write_html("path/to/file.html")` to save as an HTML file, see
        `Interactive HTML Export in Python <https://plotly.com/python/interactive-html-export/>`_
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
    plotbase.plot_anim(
        mode_tag,
        n_cycle=n_cycle,
        framerate=framerate,
        alpha=scale,
        style=style,
        show_origin=show_origin,
        show_bc=show_bc,
        bc_scale=bc_scale,
        show_mp_constraint=show_mp_constraint,
    )
    return plotbase.update_fig(show_outline=show_outline)


def plot_eigen_table(
    mode_tags: Union[list, tuple, int],
    odb_tag: Union[int, str] = 1,
    solver: str = "-genBandArpack",
) -> go.Figure:
    """Plot Modal Properties Table.

    Parameters
    ----------
    mode_tags: Union[List, Tuple]
        The modal range to visualize, [mode i, mode j].
    odb_tag: Union[int, str], default: None
        Tag of output databases (ODB) to be visualized.
    solver : str, optional,
       OpenSees' eigenvalue analysis solver, by default "-genBandArpack".

    Returns
    -------
    fig: `plotly.graph_objects.Figure <https://plotly.com/python-api-reference/generated/plotly.graph_objects.Figure.html>`_
        You can use `fig.show()` to display,
        You can also use `fig.write_html("path/to/file.html")` to save as an HTML file, see
        `Interactive HTML Export in Python <https://plotly.com/python/interactive-html-export/>`_
    """
    resave = odb_tag is None
    if isinstance(mode_tags, int):
        mode_tags = [1, mode_tags]
    modalProps, eigenvectors, MODEL_INFO = load_eigen_data(
        odb_tag=odb_tag, mode_tag=mode_tags[-1], solver=solver, resave=resave
    )
    modei, modej = int(mode_tags[0]), int(mode_tags[1])
    plotbase = PlotEigenBase(MODEL_INFO, modalProps, eigenvectors)
    fig = plotbase.plot_props_table(modei, modej)
    return fig
