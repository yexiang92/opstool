import warnings
from typing import Optional, Union

import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objs as go
from matplotlib.colors import to_hex

from ...post import load_model_data
from ...utils import CONFIGS, gram_schmidt
from .plot_resp_base import PlotResponseBase, _make_lines_arrows, _plot_bc, _plot_mp_constraint
from .plot_utils import (
    PLOT_ARGS,
    _get_ele_color,
    _get_line_cells,
    _get_unstru_cells,
    _make_lines_plotly,
    _plot_lines,
    _plot_points,
    _plot_unstru,
)

PKG_NAME = CONFIGS.get_pkg_name()


class PlotModelBase(PlotResponseBase):
    def __init__(self, model_info: dict, cells: dict):
        # --------------------------------------------------------------
        self.nodal_data = model_info["NodalData"]
        if "nodeTags" in self.nodal_data.coords:
            self.nodal_tags = self.nodal_data.coords["nodeTags"].values
        else:
            raise ValueError("Model have no nodal data!")  # noqa: TRY003
        self.points = self.nodal_data.to_numpy()
        self.ndims = self.nodal_data.attrs["ndims"]
        self.show_zaxis = not np.max(self.ndims) <= 2
        self.bounds = self.nodal_data.attrs["bounds"]
        self.min_bound_size = self.nodal_data.attrs["minBoundSize"]
        self.max_bound_size = self.nodal_data.attrs["maxBoundSize"]
        # -------------------------------------------------------------
        self.ele_centers = model_info["eleCenters"]
        if "eleTags" in self.ele_centers.coords:
            self.ele_tags = self.ele_centers.coords["eleTags"]
        else:
            self.ele_tags = []
        # ---------------------------------------------------------------
        self.ele_data_types = cells
        self.ele_types = list(cells.keys())
        # -------------------------------------------------------------
        self.fixed_node_data = model_info["FixedNodalData"]
        self.nodal_load_data = model_info["NodalLoadData"]
        self.ele_load_data = model_info["EleLoadData"]
        self.mp_constraint_data = model_info["MPConstraintData"]
        # ------------------------------------------------------------
        self.beam_data = model_info["BeamData"]
        # -------------------------------------------------------------
        self.link_data = model_info["LinkData"]
        # -------------------------------------------------------------
        self.shell_data = model_info["ShellData"]
        # -------------------------------------------------------------
        self.line_data = model_info["AllLineElesData"]
        self.line_cells, self.line_tags = _get_line_cells(self.line_data)
        # -------------------------------------------------------------
        self.unstru_data = model_info["UnstructuralData"]
        self.unstru_tags, self.unstru_cell_types, self.unstru_cells = _get_unstru_cells(self.unstru_data)
        # -------------------------------------------------------------
        self.pargs = PLOT_ARGS
        self.FIGURE = go.Figure()

    def plot_model_one_color(
        self,
        plotter: list,
        color: str,
        style: str,
    ):
        if len(self.unstru_data) > 0:
            (
                face_points,
                face_line_points,
                face_mid_points,
                face_veci,
                face_vecj,
                face_veck,
            ) = self._get_plotly_unstru_data(self.points, self.unstru_cell_types, self.unstru_cells)
            _plot_unstru(
                plotter,
                pos=face_points,
                veci=face_veci,
                vecj=face_vecj,
                veck=face_veck,
                style=style,
                line_width=self.pargs.line_width,
                color=color,
                opacity=self.pargs.mesh_opacity,
                show_edges=self.pargs.show_mesh_edges,
                edge_color=self.pargs.mesh_edge_color,
                edge_width=self.pargs.mesh_edge_width,
                edge_points=face_line_points,
                hoverinfo="skip",
            )
        if len(self.line_data) > 0:
            line_points, line_mid_points = self._get_plotly_line_data(self.points, self.line_cells)
            _plot_lines(
                plotter,
                pos=line_points,
                color=color,
                width=self.pargs.line_width,
                hoverinfo="skip",
            )

    def plot_model(
        self,
        plotter: list,
        style: str,
        show_ele_hover: bool = True,
    ):
        if len(self.ele_data_types) > 0:
            colors = _get_ele_color(self.ele_types)
            for i, name in enumerate(self.ele_types):
                cell = np.array(self.ele_data_types[name][:, :-1], dtype=int)
                cell_type = np.array(self.ele_data_types[name][:, -1], dtype=int)
                if cell_type[0] in self.unstru_cell_types:
                    (
                        face_points,
                        face_line_points,
                        face_mid_points,
                        face_veci,
                        face_vecj,
                        face_veck,
                    ) = self._get_plotly_unstru_data(self.points, cell_type, cell)
                    _plot_unstru(
                        plotter,
                        pos=face_points,
                        veci=face_veci,
                        vecj=face_vecj,
                        veck=face_veck,
                        style=style,
                        color=colors[i],
                        name=name,
                        hoverinfo="skip",
                        line_width=self.pargs.line_width,
                        opacity=self.pargs.mesh_opacity,
                        show_edges=self.pargs.show_mesh_edges,
                        edge_color=self.pargs.mesh_edge_color,
                        edge_width=self.pargs.mesh_edge_width,
                        edge_points=face_line_points,
                    )
            for i, name in enumerate(self.ele_types):
                cell = np.array(self.ele_data_types[name][:, :-1], dtype=int)
                if cell[0, 0] == 2:
                    line_points, line_mid_points = self._get_plotly_line_data(self.points, cell)
                    _plot_lines(
                        plotter,
                        pos=line_points,
                        color=colors[i],
                        width=self.pargs.line_width,
                        name=name,
                        hoverinfo="skip",
                    )
            # add element hover data
            if show_ele_hover:
                for i, name in enumerate(self.ele_types):
                    cell = np.array(self.ele_data_types[name][:, :-1], dtype=int)
                    ele_tags = self.ele_data_types[name].coords["eleTags"].values
                    ele_centers, labels = [], []
                    for etag, cell_ in zip(ele_tags, cell):
                        ntags = self.nodal_tags[cell_[1:]]
                        ele_centers.append(np.mean(self.points[cell_[1:]], axis=0))
                        labels.append(f"eleTag: {etag}<br>connectedNodes:<br> {ntags}")
                    size = 2 if self.pargs.point_size < 2 else self.pargs.point_size
                    _plot_points(
                        plotter,
                        pos=np.array(ele_centers),
                        color=colors[i],
                        size=size,
                        name=name,
                        symbol="diamond-open",
                        customdata=labels,
                        hovertemplate="%{customdata}",
                    )
        node_labels = [str(i) for i in self.nodal_tags]
        _plot_points(
            plotter,
            pos=self.points,
            color=self.pargs.color_point,
            size=self.pargs.point_size,
            name="Nodes",
            customdata=node_labels,
            hovertemplate="x: %{x}<br>y: %{y}<br>z: %{z} <br>nodeTag: %{customdata}",
        )

    def plot_nodal_labels(
        self,
        plotter: list,
    ):
        if len(self.nodal_data) > 0:
            node_labels = [str(i) for i in self.nodal_tags]
            x, y, z = self.points[:, 0], self.points[:, 1], self.points[:, 2]
            txt_plot = go.Scatter3d(
                x=x,
                y=y,
                z=z,
                text=node_labels,
                textfont={"color": "#6e750e", "size": self.pargs.font_size},
                mode="text",
                name="Node Tag",
            )
            plotter.append(txt_plot)

    def plot_ele_labels(
        self,
        plotter: list,
    ):
        if len(self.ele_centers) > 0:
            ele_tags = self.ele_centers.coords["eleTags"].data
            ele_centers = self.ele_centers.to_numpy()
            ele_labels = ["E" + str(i) for i in ele_tags]
            txt_plot = go.Scatter3d(
                x=ele_centers[:, 0],
                y=ele_centers[:, 1],
                z=ele_centers[:, 2],
                text=ele_labels,
                textfont={"color": "#650021", "size": self.pargs.font_size},
                mode="text",
                name="Element Tag",
            )
            plotter.append(txt_plot)

    def plot_bc(self, plotter: list, alpha: float = 1.0, points_new=None):
        if len(self.fixed_node_data) > 0:
            fixed_data = self.fixed_node_data.to_numpy()
            fixed_dofs = fixed_data[:, -6:].astype(int)
            fixed_coords = points_new if points_new is not None else fixed_data[:, :3]
            s = (self.max_bound_size + self.min_bound_size) / 100 * alpha
            bc_plot = _plot_bc(
                plotter,
                fixed_dofs,
                fixed_coords,
                s,
                show_zaxis=self.show_zaxis,
                color=self.pargs.color_bc,
            )
            return bc_plot
        else:
            return None

    def plot_link(self, plotter: list):
        if len(self.link_data) == 0:
            return None
        cells = self.link_data[:, :3]
        points_zero = []
        points_nonzero = []
        cells_nonzero = []
        for cell in cells:
            idx1, idx2 = cell[1:]
            idx1, idx2 = int(idx1), int(idx2)
            coord1, coord2 = self.points[idx1], self.points[idx2]
            length = np.sqrt(np.sum((coord2 - coord1) ** 2))
            if np.abs(length) < 1e-8:
                points_zero.append(coord1)
            else:
                xaxis = np.array(coord2 - coord1)
                global_z = [0.0, 0.0, 1.0]
                cos_angle = xaxis.dot(global_z) / (np.linalg.norm(xaxis) * np.linalg.norm(global_z))
                if np.abs(1 - cos_angle**2) < 1e-10:
                    yaxis = np.cross([-1.0, 0.0, 0.0], xaxis)
                else:
                    yaxis = np.cross(global_z, xaxis)
                xaxis = xaxis / np.linalg.norm(xaxis)
                yaxis = yaxis / np.linalg.norm(yaxis)
                idx = len(points_nonzero)
                for i in range(5):
                    cells_nonzero.extend([2, idx + i, idx + i + 1])
                points_nonzero.extend([
                    coord1 + 0.25 * length * xaxis,
                    coord1 + 0.25 * length * xaxis + 0.25 * length * yaxis,
                    coord1 + 0.5 * length * xaxis - 0.25 * length * yaxis,
                    coord1 + 0.5 * length * xaxis + 0.25 * length * yaxis,
                    coord1 + 0.75 * length * xaxis - 0.25 * length * yaxis,
                    coord1 + 0.75 * length * xaxis,
                ])
        # plot
        if len(points_zero) > 0:
            points_zero = np.array(points_zero)
            plotter.append(
                go.Scatter3d(
                    x=points_zero[:, 0],
                    y=points_zero[:, 1],
                    z=points_zero[:, 2],
                    marker={"size": self.pargs.point_size * 2, "color": self.pargs.color_link},
                    mode="markers",
                    hoverinfo="skip",
                )
            )
        if len(points_nonzero) > 0:
            cells_nonzero = np.reshape(cells_nonzero, (-1, 3))
            points_nonzero = np.array(points_nonzero)
            line_points, _ = _make_lines_plotly(points_nonzero, cells_nonzero)
            x, y, z = line_points[:, 0], line_points[:, 1], line_points[:, 2]
            plotter.append(
                go.Scatter3d(
                    x=x,
                    y=y,
                    z=z,
                    line={"color": self.pargs.color_link, "width": self.pargs.line_width / 2},
                    mode="lines",
                    connectgaps=False,
                    hoverinfo="skip",
                )
            )

    @staticmethod
    def _plot_local_axis(plotter, xaxis, yaxis, zaxis, midpoints, lengths):
        if len(midpoints) > 0:
            arrow_heights = [0.4 * ll for ll in lengths]
            arrow_widths = [0.8 * h for h in arrow_heights]
            plotter.extend(
                _make_lines_arrows(
                    midpoints,
                    lengths,
                    xaxis,
                    yaxis,
                    zaxis,
                    "#cf6275",
                    "Local Axis",
                    ["x"] * len(midpoints),
                    1.0,
                    arrow_heights,
                    arrow_widths,
                )
            )
            plotter.extend(
                _make_lines_arrows(
                    midpoints,
                    lengths,
                    yaxis,
                    xaxis,
                    zaxis,
                    "#04d8b2",
                    "Local Axis",
                    ["y"] * len(midpoints),
                    1.0,
                    arrow_heights,
                    arrow_widths,
                )
            )
            plotter.extend(
                _make_lines_arrows(
                    midpoints,
                    lengths,
                    zaxis,
                    xaxis,
                    yaxis,
                    "#9aae07",
                    "Local Axis",
                    ["z"] * len(midpoints),
                    1.0,
                    arrow_heights,
                    arrow_widths,
                )
            )

        else:
            warnings.warn("Model has no frame elements when show_local_crd=True!", stacklevel=2)

    def plot_link_local_axes(self, plotter: list, alpha: float = 1.0):
        if len(self.link_data) == 0:
            return None
        lengths = self.link_data.loc[:, "length"].to_numpy()
        lengths = [np.mean(lengths) / 5 * alpha] * len(lengths)
        self._plot_local_axis(
            plotter,
            self.link_data.loc[:, ["xaxis-x", "xaxis-y", "xaxis-z"]].to_numpy(),
            self.link_data.loc[:, ["yaxis-x", "yaxis-y", "yaxis-z"]].to_numpy(),
            self.link_data.loc[:, ["zaxis-x", "zaxis-y", "zaxis-z"]].to_numpy(),
            self.link_data.loc[:, ["xo", "yo", "zo"]].to_numpy(),
            lengths,
        )

    def plot_beam_local_axes(self, plotter: list, alpha: float = 1.0):
        if len(self.beam_data) == 0:
            return None
        lengths = self.beam_data.loc[:, "length"].to_numpy()
        lengths = [np.mean(lengths) / 5 * alpha] * len(lengths)
        self._plot_local_axis(
            plotter,
            self.beam_data.loc[:, ["xaxis-x", "xaxis-y", "xaxis-z"]].to_numpy(),
            self.beam_data.loc[:, ["yaxis-x", "yaxis-y", "yaxis-z"]].to_numpy(),
            self.beam_data.loc[:, ["zaxis-x", "zaxis-y", "zaxis-z"]].to_numpy(),
            self.beam_data.loc[:, ["xo", "yo", "zo"]].to_numpy(),
            lengths,
        )

    def plot_shell_local_axes(self, plotter: list, alpha: float = 1.0):
        if len(self.shell_data) == 0:
            return None
        node_coords = self.points
        cells = self.shell_data.to_numpy()[:, :-1]
        xlocal, ylocal, zlocal, midpoints, lengths = [], [], [], [], []
        for cell in cells:
            num = len(cell) - 1
            coord = node_coords[cell[1:]]
            if num == 3:
                coord_ = coord
                v1, v2 = coord_[1] - coord_[0], coord_[2] - coord_[0]
            elif num == 6:
                coord_ = coord[[0, 2, 4]]
                v1, v2 = coord_[1] - coord_[0], coord_[2] - coord_[0]
            elif num == 4:
                coord_ = coord
                v1 = 0.5 * ((coord_[1] + coord_[2]) - (coord_[0] + coord_[3]))
                v2 = 0.5 * ((coord_[2] + coord_[3]) - (coord_[0] + coord_[1]))
            else:
                coord_ = coord[[0, 2, 4, 6]]
                v1 = 0.5 * ((coord_[1] + coord_[2]) - (coord_[0] + coord_[3]))
                v2 = 0.5 * ((coord_[2] + coord_[3]) - (coord_[0] + coord_[1]))
            x, y, z = gram_schmidt(v1, v2)
            xyzo = np.mean(coord, axis=0)
            xlocal.append(x)
            ylocal.append(y)
            zlocal.append(z)
            midpoints.append(xyzo)
            lengths.append((np.linalg.norm(v1) + np.linalg.norm(v2)) / 2)
        xaxis, yaxis, zaxis = np.array(xlocal), np.array(ylocal), np.array(zlocal)
        midpoints, lengths = np.array(midpoints), np.array(lengths)
        lengths = [np.mean(lengths) / 5 * alpha] * len(lengths)
        self._plot_local_axis(
            plotter,
            xaxis,
            yaxis,
            zaxis,
            midpoints,
            lengths,
        )

    def plot_node_load(self, plotter: list, alpha: float = 1.0):
        if len(self.nodal_load_data) == 0:
            return None
        pntags = self.nodal_load_data.coords["PatternNodeTags"].values
        patterntags, nodetags = [], []
        for item in pntags:
            num1, num2 = item.split("-")
            patterntags.append(int(num1))
            nodetags.append(int(num2))
        patterntags, nodetags = np.array(patterntags), np.array(nodetags)
        load_data = self.nodal_load_data.to_numpy()
        maxdata = np.max(np.abs(load_data))
        alpha_ = (self.max_bound_size + self.min_bound_size) / 20 / maxdata
        alpha_ *= alpha
        patterntags2 = np.unique(patterntags)
        cmap = plt.get_cmap("winter")
        colors = cmap(np.linspace(0, 1, len(patterntags2)))
        xyzlocals = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
        idxs = [[0, 1, 2], [1, 0, 2], [2, 0, 1]]
        for p, ptag in enumerate(patterntags2):
            idx = np.abs(patterntags - ptag) < 1e-4
            ntags = nodetags[idx]
            coords = self.nodal_data.loc[ntags, :].to_numpy()
            for i in range(3):
                data = np.ravel(load_data[idx, i])
                lengths = np.abs(data) * alpha_
                arrow_heights = [0.5 * ll for ll in lengths]
                arrow_widths = [0.8 * h for h in arrow_heights]
                xaxis = np.reshape(xyzlocals[idxs[i][0]] * len(coords), (-1, 3))
                yaxis = np.reshape(xyzlocals[idxs[i][1]] * len(coords), (-1, 3))
                zaxis = np.reshape(xyzlocals[idxs[i][2]] * len(coords), (-1, 3))
                new_coords = np.zeros_like(coords)
                for j in range(len(xaxis)):
                    xaxis[j] *= np.sign(data[j])
                    new_coords[j] = coords[j] - 1.4 * lengths[j] * xaxis[j]
                labels = [f"{d:.2e}" for d in data]
                color = colors[p]
                plotter.extend(
                    _make_lines_arrows(
                        new_coords,
                        lengths,
                        xaxis,
                        yaxis,
                        zaxis,
                        to_hex(color),
                        f"<b>Pattern {ptag}</b><br>Nodal Load",
                        labels,
                        self.pargs.line_width,
                        arrow_heights,
                        arrow_widths,
                    )
                )

    def plot_ele_load(self, plotter: list, alpha: float = 1.0):
        if len(self.ele_load_data) == 0:
            return None
        petags = self.ele_load_data.coords["PatternEleTags"].values
        patterntags, eletags = [], []
        for item in petags:
            num1, num2 = item.split("-")
            patterntags.append(int(num1))
            eletags.append(int(num2))
        patterntags, eletags = np.array(patterntags), np.array(eletags)
        patterntags2 = np.unique(patterntags)
        load_info = self.ele_load_data.to_numpy()
        new_points = []
        new_locals = []
        new_ptags = []
        load_data = []
        for i, ptag in enumerate(patterntags):
            node1, node2 = load_info[i, :2]
            coord1, coord2 = (
                self.nodal_data.loc[node1, :],
                self.nodal_data.loc[node2, :],
            )
            wya, wyb, wza, wzb, wxa, wxb, xa, xb = load_info[i, 2:]
            etag = eletags[i]
            local_axis = self.beam_data.loc[
                etag,
                [
                    "xaxis-x",
                    "xaxis-y",
                    "xaxis-z",
                    "yaxis-x",
                    "yaxis-y",
                    "yaxis-z",
                    "zaxis-x",
                    "zaxis-y",
                    "zaxis-z",
                ],
            ].to_numpy()

            if xb > xa:  # distributed load
                n = np.max([int((xb - xa) / 0.1) + 1, 6])
                xl = np.linspace(xa, xb, n)
                wz = np.interp(xl, [xa, xb], [wza, wzb])
                wy = np.interp(xl, [xa, xb], [wya, wyb])
                wx = np.interp(xl, [xa, xb], [wxa, wxb])
                localaxis = [local_axis] * n
                new_ptags.extend([ptag] * n)
            else:
                xl = [xa]
                wx, wy, wz = wxa, wya, wza
                localaxis = [local_axis]
                new_ptags.append(ptag)
            xs = np.interp(xl, [0, 1], [coord1[0], coord2[0]])
            ys = np.interp(xl, [0, 1], [coord1[1], coord2[1]])
            zs = np.interp(xl, [0, 1], [coord1[2], coord2[2]])
            new_points.append(np.column_stack([xs, ys, zs]))
            new_locals.append(localaxis)
            load_data.append(np.column_stack([wx, wy, wz]))
        new_points = np.vstack(new_points)
        new_locals = np.vstack(new_locals)
        load_data = np.vstack(load_data)
        new_ptags = np.array(new_ptags)
        maxdata = np.max(np.abs(load_data))
        alpha_ = (self.max_bound_size + self.min_bound_size) / 20 / maxdata
        alpha_ *= alpha
        cmap = plt.get_cmap("turbo_r")
        colors = cmap(np.linspace(0, 1, len(patterntags2)))
        idxs = [[0, 1, 2], [3, 4, 5], [6, 7, 8]]
        idxsidx = [[0, 1, 2], [1, 0, 2], [2, 0, 1]]
        for p, ptag in enumerate(patterntags2):
            idx = np.abs(new_ptags - ptag) < 1e-3
            coords = new_points[idx]
            for i in range(3):
                data = np.ravel(load_data[idx, i])
                lengths = np.abs(data) * alpha_
                arrow_heights = [0.5 * ll for ll in lengths]
                arrow_widths = [0.8 * h for h in arrow_heights]
                xaxis = new_locals[np.ix_(idx, idxs[idxsidx[i][0]])]
                yaxis = new_locals[np.ix_(idx, idxs[idxsidx[i][1]])]
                zaxis = new_locals[np.ix_(idx, idxs[idxsidx[i][2]])]
                new_coords = np.zeros_like(coords)
                for j in range(len(xaxis)):
                    xaxis[j] *= np.sign(data[j])
                    new_coords[j] = coords[j] - 1.4 * lengths[j] * xaxis[j]
                labels = [f"{d:.2e}" for d in data]
                plotter.extend(
                    _make_lines_arrows(
                        new_coords,
                        lengths,
                        xaxis,
                        yaxis,
                        zaxis,
                        to_hex(colors[p]),
                        f"<b>Pattern {ptag}</b><br>Element Load",
                        labels,
                        self.pargs.line_width,
                        arrow_heights,
                        arrow_widths,
                    )
                )

    def plot_mp_constraint(self, plotter: list, show_dofs=False, points_new=None):
        if len(self.mp_constraint_data) == 0:
            return None
        points = self.points if points_new is None else points_new
        cells = self.mp_constraint_data.to_numpy()[:, :3].astype(int)
        dofs = self.mp_constraint_data.to_numpy()[:, -6:].astype(int)
        # midcoords = self.mp_constraint_data.to_numpy()[:, 3:6]
        _plot_mp_constraint(
            plotter,
            points,
            cells,
            dofs,
            self.pargs.line_width / 2,
            self.pargs.color_constraint,
            show_dofs=show_dofs,
        )

    def update_fig(self, plotter: list, show_outline: bool = False):
        self.FIGURE.add_traces(plotter)
        if not self.show_zaxis:
            scene = self._get_plotly_dim_scene(mode="2d", show_outline=show_outline)
        else:
            scene = self._get_plotly_dim_scene(mode="3d", show_outline=show_outline)
        txt = f"<b>{PKG_NAME}</b>:: Num. Node: <b>{len(self.nodal_tags)}</b> Num. Ele: <b>{len(self.ele_tags)}</b>"
        self.FIGURE.update_layout(
            template=self.pargs.theme,
            autosize=True,
            showlegend=False,
            scene=scene,
            title={
                "font": {"family": "courier", "size": self.pargs.title_font_size},
                "text": txt,
            },
            width=self.pargs.window_size[0],
            height=self.pargs.window_size[1],
            font={"family": self.pargs.font_family},
        )
        return self.FIGURE


def plot_model(
    odb_tag: Optional[Union[int, str]] = None,
    show_node_numbering: bool = False,
    show_ele_numbering: bool = False,
    show_ele_hover: bool = True,
    style: str = "surface",
    color: Optional[str] = None,
    show_bc: bool = True,
    bc_scale: float = 1.0,
    show_link: bool = True,
    show_mp_constraint: bool = True,
    show_constraint_dofs: bool = False,
    show_nodal_loads: bool = False,
    show_ele_loads: bool = False,
    load_scale: float = 1.0,
    show_local_axes: bool = False,
    local_axes_scale: float = 1.0,
    show_outline: bool = True,
) -> go.Figure:
    """
    Geometric model visualization based on ``plotly``.

    Parameters
    ----------
    odb_tag: Union[int, str], default: None
        Tag of output databases (ODB) to be visualized.
        If None, data will be extracted from the current running memory..
    show_node_numbering: bool, default: False
        Whether to display node tag labels.
    show_ele_numbering: bool, default: False
        Whether to display element tag labels.
    show_ele_hover: bool, default: True
        Whether to display element tag labels when hovering over the element.
    style: str, default: surface
        Visualization mesh style of surfaces and solids.
        One of the following: style='surface' or style='wireframe'
        Defaults to 'surface'. Note that 'wireframe' only shows a wireframe of the outer geometry.
    color: str, default: black
        Model display color.
    show_bc: bool, default: True
        Whether to display boundary supports.
    bc_scale: float, default: 1.0
        Scale the size of boundary support display.
    show_link: bool, default: True
        Whether to show link elements.
    show_mp_constraint: bool, default: True
        Whether to show multipoint (MP) constraint.
    show_constraint_dofs: bool, default: False
        Whether to show dofs of mp-constraints.
    show_nodal_loads: bool, default: False
        Whether to show nodal loads.
    show_ele_loads: bool, default: False
        Whether to show element loads.
    load_scale: float, default: 1.0
        Scale the size of load arrow presentation.
    show_local_axes: bool, default: False
        Whether to display element local axes, including ``beam-column``, ``link``, and ``shell`` elements.
    local_axes_scale: float, default: 1.0
        Scales the presentation size of the local axes.
    show_outline: bool, default: False
        Whether to display the outline of the model.

    Returns
    -------
    fig: `plotly.graph_objects.Figure <https://plotly.com/python-api-reference/generated/plotly.graph_objects.Figure.html>`_
        You can use `fig.show()` to display,
        You can also use `fig.write_html("path/to/file.html")` to save as an HTML file, see
        `Interactive HTML Export in Python <https://plotly.com/python/interactive-html-export/>`_
    """
    resave = odb_tag is None
    model_info, cells = load_model_data(odb_tag, resave=resave)
    plotbase = PlotModelBase(model_info, cells)
    plotter = []
    if color:  # single color
        plotbase.plot_model_one_color(
            plotter,
            color,
            style,
        )
    else:
        plotbase.plot_model(plotter, style, show_ele_hover=show_ele_hover)
    if show_node_numbering:
        plotbase.plot_nodal_labels(plotter)
    if show_ele_numbering:
        plotbase.plot_ele_labels(plotter)
    if show_bc:
        plotbase.plot_bc(plotter, bc_scale)
    if show_mp_constraint:
        plotbase.plot_mp_constraint(plotter, show_constraint_dofs)
    if show_link:
        plotbase.plot_link(plotter)
    if show_local_axes:
        plotbase.plot_beam_local_axes(plotter, local_axes_scale)
        plotbase.plot_link_local_axes(plotter, local_axes_scale)
        plotbase.plot_shell_local_axes(plotter, local_axes_scale)
    if show_nodal_loads:
        plotbase.plot_node_load(plotter, load_scale)
    if show_ele_loads:
        plotbase.plot_ele_load(plotter, load_scale)
    return plotbase.update_fig(plotter, show_outline)
