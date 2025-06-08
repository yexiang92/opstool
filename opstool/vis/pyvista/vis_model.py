from typing import Optional, Union

import matplotlib.pyplot as plt
import numpy as np
import pyvista as pv
from numpy.linalg import norm

from ...post import load_model_data
from ...utils import CONFIGS, gram_schmidt
from .plot_resp_base import PlotResponseBase, _plot_bc, _plot_mp_constraint
from .plot_utils import (
    PLOT_ARGS,
    _get_ele_color,
    _get_line_cells,
    _get_unstru_cells,
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
        pv.set_plot_theme(PLOT_ARGS.theme)

    def plot_model_one_color(
        self,
        plotter,
        color: str,
        style: str,
    ):
        if len(self.unstru_data) > 0:
            _plot_unstru(
                plotter,
                pos=self.points,
                cells=self.unstru_cells,
                cell_types=self.unstru_cell_types,
                color=color,
                show_edges=self.pargs.show_mesh_edges,
                edge_color=self.pargs.mesh_edge_color,
                edge_width=self.pargs.mesh_edge_width,
                opacity=self.pargs.mesh_opacity,
                style=style,
            )
        if len(self.line_data) > 0:
            _plot_lines(
                plotter,
                pos=self.points,
                cells=self.line_cells,
                color=color,
                width=self.pargs.line_width,
                render_lines_as_tubes=self.pargs.render_lines_as_tubes,
            )

    def plot_model(self, plotter, style: str):
        if len(self.ele_data_types) > 0:
            colors = _get_ele_color(self.ele_types)
            for i, name in enumerate(self.ele_types):
                cell = np.array(self.ele_data_types[name][:, :-1], dtype=int)
                cell_type = np.array(self.ele_data_types[name][:, -1], dtype=int)
                if cell_type[0] in self.unstru_cell_types:
                    _plot_unstru(
                        plotter,
                        pos=self.points,
                        cells=cell,
                        cell_types=cell_type,
                        color=colors[i],
                        show_edges=self.pargs.show_mesh_edges,
                        edge_color=self.pargs.mesh_edge_color,
                        edge_width=self.pargs.mesh_edge_width,
                        opacity=self.pargs.mesh_opacity,
                        style=style,
                        label=name,
                    )
            for i, name in enumerate(self.ele_types):
                cell = np.array(self.ele_data_types[name][:, :-1], dtype=int)
                if cell[0, 0] == 2:
                    _plot_lines(
                        plotter,
                        pos=self.points,
                        cells=cell,
                        color=colors[i],
                        width=self.pargs.line_width,
                        render_lines_as_tubes=self.pargs.render_lines_as_tubes,
                        label=name,
                    )
        _plot_points(
            plotter,
            pos=self.points,
            color=self.pargs.color_point,
            size=self.pargs.point_size,
            render_points_as_spheres=self.pargs.render_points_as_spheres,
        )

    def plot_nodal_labels(self, plotter):
        if len(self.nodal_data) > 0:
            node_labels = ["N" + str(i) for i in self.nodal_tags]
            plotter.add_point_labels(
                self.points,
                node_labels,
                text_color=self.pargs.color_nodal_label,
                font_size=self.pargs.font_size,
                point_color=self.pargs.color_point,
                bold=True,
                render_points_as_spheres=True,
                point_size=1e-5,
                always_visible=True,
                shape_opacity=0.0,
            )

    def plot_ele_labels(self, plotter):
        if len(self.ele_centers) > 0:
            ele_tags = self.ele_centers.coords["eleTags"].data
            ele_centers = self.ele_centers.to_numpy()
            ele_labels = ["E" + str(i) for i in ele_tags]
            plotter.add_point_labels(
                ele_centers,
                ele_labels,
                text_color=self.pargs.color_ele_label,
                font_size=self.pargs.font_size,
                point_size=self.pargs.point_size + 2,
                bold=True,
                always_visible=True,
                shape_opacity=0.0,
            )

    def plot_outline(self, plotter):
        plotter.show_bounds(
            grid=False,
            location="outer",
            bounds=self.bounds,
            show_zaxis=self.show_zaxis,
        )

    def plot_bc(self, plotter, alpha: float = 1.0, points_new: Optional[np.ndarray] = None):
        if len(self.fixed_node_data) > 0:
            fixed_data = self.fixed_node_data.to_numpy()
            fixed_dofs = fixed_data[:, -6:].astype(int)
            fixed_coords = points_new if points_new is not None else fixed_data[:, :3]
            s = (self.max_bound_size + self.min_bound_size) / 75 * alpha
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

    def plot_link(self, plotter):
        if len(self.link_data) == 0:
            return None
        cells = self.link_data[:, :3]
        points_zero, points_nonzero, cells_nonzero = [], [], []
        for cell in cells:
            idx1, idx2 = cell[1:]
            idx1, idx2 = int(idx1), int(idx2)
            coord1, coord2 = self.points[idx1], self.points[idx2]
            length = np.sqrt(np.sum((coord2 - coord1) ** 2))
            if np.abs(length) < 1e-8:
                points_zero.append(coord1)
                points_zero.append(coord2)
            else:
                xaxis = np.array(coord2 - coord1, dtype=float)
                global_z = np.array([0.0, 0.0, 1.0], dtype=float)
                cos_angle = np.dot(xaxis, global_z) / (norm(xaxis) * norm(global_z))
                if np.abs(1 - cos_angle**2) > 1e-8:
                    yaxis = np.cross(global_z, xaxis)
                else:
                    yaxis = np.cross([-1.0, 0.0, 0.0], xaxis)
                xaxis = xaxis / norm(xaxis)
                yaxis = yaxis / norm(yaxis)
                idx = len(points_nonzero)
                for i in range(5):
                    cells_nonzero.extend([2, idx + i, idx + i + 1])
                points_nonzero.extend([
                    coord1 + 0.25 * length * xaxis,
                    coord1 + 0.25 * length * xaxis - 0.25 * length * yaxis,
                    coord1 + 0.5 * length * xaxis + 0.25 * length * yaxis,
                    coord1 + 0.5 * length * xaxis - 0.25 * length * yaxis,
                    coord1 + 0.75 * length * xaxis + 0.25 * length * yaxis,
                    coord1 + 0.75 * length * xaxis,
                ])
        if len(points_zero) > 0:
            return _plot_points(
                plotter,
                np.array(points_zero),
                self.pargs.color_link,
                self.pargs.point_size * 1.2,
            )
        if len(points_nonzero) > 0:
            points_nonzero = np.array(points_nonzero)
            return _plot_lines(
                plotter,
                points_nonzero,
                cells_nonzero,
                width=self.pargs.line_width / 2,
                color=self.pargs.color_link,
                render_lines_as_tubes=False,
                label="Link",
            )
        return None

    @staticmethod
    def _plot_local_axis(plotter, xaxis, yaxis, zaxis, midpoints, lengths, alpha, labelsize):
        if len(midpoints) > 0:
            length = np.mean(lengths) / 6 * alpha
            _ = plotter.add_arrows(midpoints, xaxis, mag=length, color="#cf6275")
            _ = plotter.add_arrows(midpoints, yaxis, mag=length, color="#04d8b2")
            _ = plotter.add_arrows(midpoints, zaxis, mag=length, color="#9aae07")
            plotter.add_point_labels(
                midpoints + length * xaxis,
                ["x"] * midpoints.shape[0],
                text_color="#cf6275",
                font_size=labelsize,
                bold=False,
                shape=None,
                render_points_as_spheres=True,
                point_size=1.0e-5,
                always_visible=True,
            )
            plotter.add_point_labels(
                midpoints + length * yaxis,
                ["y"] * midpoints.shape[0],
                text_color="#04d8b2",
                font_size=labelsize,
                bold=False,
                shape=None,
                render_points_as_spheres=True,
                point_size=1.0e-5,
                always_visible=True,
            )
            plotter.add_point_labels(
                midpoints + length * zaxis,
                ["z"] * midpoints.shape[0],
                text_color="#9aae07",
                font_size=labelsize,
                bold=False,
                shape=None,
                render_points_as_spheres=True,
                point_size=1.0e-5,
                always_visible=True,
            )

    def plot_link_local_axes(self, plotter, alpha: float = 1.0):
        if len(self.link_data) == 0:
            return None
        return self._plot_local_axis(
            plotter,
            self.link_data.loc[:, ["xaxis-x", "xaxis-y", "xaxis-z"]].to_numpy(),
            self.link_data.loc[:, ["yaxis-x", "yaxis-y", "yaxis-z"]].to_numpy(),
            self.link_data.loc[:, ["zaxis-x", "zaxis-y", "zaxis-z"]].to_numpy(),
            self.link_data.loc[:, ["xo", "yo", "zo"]].to_numpy(),
            self.link_data.loc[:, "length"].to_numpy(),
            alpha,
            self.pargs.font_size,
        )

    def plot_beam_local_axes(self, plotter, alpha: float = 1.0):
        if len(self.beam_data) == 0:
            return None
        return self._plot_local_axis(
            plotter,
            self.beam_data.loc[:, ["xaxis-x", "xaxis-y", "xaxis-z"]].to_numpy(),
            self.beam_data.loc[:, ["yaxis-x", "yaxis-y", "yaxis-z"]].to_numpy(),
            self.beam_data.loc[:, ["zaxis-x", "zaxis-y", "zaxis-z"]].to_numpy(),
            self.beam_data.loc[:, ["xo", "yo", "zo"]].to_numpy(),
            self.beam_data.loc[:, "length"].to_numpy(),
            alpha,
            self.pargs.font_size,
        )

    # def plot_beam_sec(self, plotter, paras):
    #     ext_points = self.MINFO["BeamSecExtPoints"]
    #     int_points = self.MINFO["BeamSecIntPoints"]
    #     sec_points = self.MINFO["BeamSecPoints"]
    #     ext_cells = self.CELLS["BeamSecExt"]
    #     int_cells = self.CELLS["BeamSecInt"]
    #     sec_cells = self.CELLS["BeamSec"]
    #     if paras["texture"]:
    #         texture = pv.read_texture(paras["texture"])
    #     else:
    #         texture = None
    #     if len(ext_cells) > 0:
    #         ext = pv.PolyData(ext_points, ext_cells)
    #         if texture is not None:
    #             ext.texture_map_to_plane(inplace=True)
    #         plotter.add_mesh(
    #             ext,
    #             show_edges=False,
    #             color=paras["color"],
    #             opacity=paras["opacity"],
    #             texture=texture,
    #         )
    #     if len(int_cells) > 0:
    #         intt = pv.PolyData(int_points, int_cells)
    #         if texture is not None:
    #             intt.texture_map_to_plane(inplace=True)
    #         plotter.add_mesh(
    #             intt,
    #             show_edges=False,
    #             color=paras["color"],
    #             opacity=paras["opacity"],
    #             texture=texture,
    #         )
    #     if len(sec_cells) > 0:
    #         sec = pv.PolyData(sec_points, sec_cells)
    #         if texture is not None:
    #             sec.texture_map_to_plane(inplace=True)
    #         plotter.add_mesh(
    #             sec,
    #             show_edges=False,
    #             color=paras["color"],
    #             opacity=paras["opacity"],
    #             texture=texture,
    #         )
    #
    # def plot_shell_thick(self, plotter, paras):
    #     points = self.MINFO["ShellThickPoints"]
    #     cells = self.CELLS["ShellThick"]
    #     if paras["texture"]:
    #         texture = pv.read_texture(paras["texture"])
    #     else:
    #         texture = None
    #     if len(cells) > 0:
    #         ext = pv.PolyData(points, cells)
    #         if texture is not None:
    #             ext.texture_map_to_plane(inplace=True)
    #         plotter.add_mesh(
    #             ext,
    #             show_edges=False,
    #             color=paras["color"],
    #             opacity=paras["opacity"],
    #             texture=texture,
    #         )

    def plot_shell_local_axes(self, plotter, alpha: float = 1.0):
        if len(self.shell_data) == 0:
            return None
        node_coords = self.points
        cells = self.shell_data.to_numpy()[:, :-1]
        xlocal, ylocal, zlocal, midpoints, lengths = [], [], [], [], []
        for cell_ in cells:
            n = cell_[0]
            cell = cell_[1 : n + 1]
            coord = node_coords[cell]
            if n == 3:
                coord_ = coord
                v1, v2 = coord_[1] - coord_[0], coord_[2] - coord_[0]
            elif n == 6:
                coord_ = coord[[0, 2, 4]]
                v1, v2 = coord_[1] - coord_[0], coord_[2] - coord_[0]
            elif n == 4:
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
        xlocal, ylocal, zlocal = np.array(xlocal), np.array(ylocal), np.array(zlocal)
        midpoints, lengths = np.array(midpoints), np.array(lengths)
        return self._plot_local_axis(
            plotter,
            xlocal,
            ylocal,
            zlocal,
            midpoints,
            lengths,
            alpha,
            self.pargs.font_size,
        )

    def plot_node_load(self, plotter, alpha: float = 1.0):
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
        geom = pv.Arrow(start=(-1.0, 0, 0), tip_length=0.25, tip_radius=0.1, shaft_radius=0.03)
        for p, ptag in enumerate(patterntags2):
            idx = np.abs(patterntags - ptag) < 1e-4
            ntags = nodetags[idx]
            coords = self.nodal_data.loc[ntags, :].to_numpy()
            for i in range(3):
                ply = pv.PolyData(coords)
                data = np.ravel(load_data[idx, i])
                ply["scalars"] = np.abs(data)
                ply["vectors"] = np.reshape(xyzlocals[i] * len(coords), (-1, 3))
                for j in range(len(ply["vectors"])):
                    ply["vectors"][j] *= np.sign(data[j])
                glyphs = ply.glyph(orient="vectors", scale="scalars", factor=alpha_, geom=geom)
                label = f"Nodal Load: patternTag={ptag}" if i == 0 else None
                plotter.add_mesh(glyphs, show_scalar_bar=False, color=colors[p], label=label)
        return None

    def plot_ele_load(self, plotter, alpha: float = 1.0):
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
        geom = pv.Arrow(start=(-0.0, 0, 0), tip_length=0.5, tip_radius=0.2, shaft_radius=0.05)
        for p, ptag in enumerate(patterntags2):
            idx = np.abs(new_ptags - ptag) < 1e-3
            coords = new_points[idx]
            for i in range(3):
                ply = pv.PolyData(coords)
                data = np.ravel(load_data[idx, i]) * alpha_
                ply["scalars"] = np.abs(data)
                ply["vectors"] = new_locals[np.ix_(idx, [3 * i, 3 * i + 1, 3 * i + 2])]
                for j in range(len(ply["vectors"])):
                    ply["vectors"][j] *= np.sign(data[j])
                glyphs = ply.glyph(orient="vectors", scale="scalars", factor=1.0, geom=geom)
                label = f"Element load: patternTag={ptag}" if i == 0 else None
                plotter.add_mesh(glyphs, show_scalar_bar=False, color=colors[p], label=label)
        return None

    def plot_mp_constraint(self, plotter, show_dofs=False, points_new=None):
        if len(self.mp_constraint_data) == 0:
            return None
        points = self.points if points_new is None else points_new
        cells = self.mp_constraint_data.to_numpy()[:, :3].astype(int)
        dofs = self.mp_constraint_data.to_numpy()[:, -6:].astype(int)
        midcoords = self.mp_constraint_data.to_numpy()[:, 3:6]
        pplot = _plot_mp_constraint(
            plotter,
            points,
            cells,
            dofs,
            midcoords,
            self.pargs.line_width / 2,
            self.pargs.color_constraint,
            show_dofs=show_dofs,
        )
        return pplot

    def add_model_info(self, plotter):
        txt = f"{PKG_NAME}:: Num. Node: {len(self.nodal_tags)} Num. Ele: {len(self.ele_tags)}"
        plotter.add_text(txt, position="lower_left", font_size=12, font="courier")


def plot_model(
    odb_tag: Optional[Union[int, str]] = None,
    show_node_numbering: bool = False,
    show_ele_numbering: bool = False,
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
    show_outline: bool = False,
    show_legend: bool = False,
    cpos: str = "iso",
) -> pv.Plotter:
    """
    Geometric model visualization based on ``pyvista``.

    Parameters
    ----------
    odb_tag: Union[int, str], default: None
        Tag of output databases (ODB) to be visualized.
        If None, data will be extracted from the current running memory.
    show_node_numbering: bool, default: False
        Whether to display node tag labels.
    show_ele_numbering: bool, default: False
        Whether to display element tag labels.
    style: str, default: surface
        Visualization mesh style of surfaces and solids.
        One of the following: style='surface', style='wireframe', style='points', style='points_gaussian'.
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
    show_legend: bool, default: False
        Whether to show legend.
    cpos: str, default: iso
        Model display perspective, optional: "iso", "xy", "yx", "xz", "zx", "yz", "zy".
        If 3d, defaults to "iso". If 2d, defaults to "xy".

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
    resave = odb_tag is None
    model_info, cells = load_model_data(odb_tag, resave=resave)
    plotbase = PlotModelBase(model_info, cells)
    plotter = pv.Plotter(
        notebook=PLOT_ARGS.notebook,
        line_smoothing=PLOT_ARGS.line_smoothing,
        polygon_smoothing=PLOT_ARGS.polygon_smoothing,
        off_screen=PLOT_ARGS.off_screen,
    )
    if color:  # single color
        plotbase.plot_model_one_color(
            plotter,
            color,
            style,
        )
    else:
        plotbase.plot_model(plotter, style)
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
    if show_outline:
        plotbase.plot_outline(plotter)
    if show_legend:
        plotter.add_legend(bcolor=None)
    if PLOT_ARGS.anti_aliasing:
        plotter.enable_anti_aliasing(PLOT_ARGS.anti_aliasing, multi_samples=PLOT_ARGS.msaa_multi_samples)
    plotbase.add_model_info(plotter)
    return plotbase._update_plotter(plotter, cpos)
