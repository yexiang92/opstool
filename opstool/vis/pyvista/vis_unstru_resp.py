from functools import partial
from typing import Optional, Union

import numpy as np
import pyvista as pv

from ...post import loadODB
from .plot_resp_base import PlotResponseBase
from .plot_utils import PLOT_ARGS, _get_unstru_cells, _plot_unstru_cmap


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

    def _set_comp_resp_type(self, ele_type, resp_type, component, fiber_point=None):
        self.ele_type = ele_type
        self.resp_type = resp_type
        self.component = component
        self.fiber_point = fiber_point

    def _make_unstru_info(self, ele_tags, step):
        pos = self._get_node_da(step)
        unstru_data = self._get_unstru_da(step)
        if ele_tags is None:
            tags, cell_types, cells = _get_unstru_cells(unstru_data)
        else:
            tags = np.atleast_1d(ele_tags)
            cells = unstru_data.sel(eleTags=tags)
            tags, cell_types, cells = _get_unstru_cells(cells)
        return tags, pos, cells, cell_types

    def refactor_resp_step(self, ele_tags, ele_type, resp_type: str, component: str, fiber_point: Optional[int] = None):
        self._set_comp_resp_type(ele_type, resp_type, component, fiber_point=fiber_point)
        resps = []

        for i in range(self.num_steps):
            da = self._get_resp_da(i, self.resp_type, self.component)

            if self.ModelUpdate or ele_tags is not None:
                tags, pos, _, _ = self._make_unstru_info(ele_tags, i)
                da = da.sel(eleTags=tags)
            else:
                pos = self._get_node_da(i)

            resps.append(self._process_scalar_from_da(da, pos, fiber_point))

        self.resp_step = resps

    def _process_scalar_from_da(self, da, pos, fiber_point):
        def _reset_fiber_point(fiber_point, da):
            if fiber_point == "top":
                fiber_point = da.coords["fiberPoints"].values[-1]
            elif fiber_point == "bottom":
                fiber_point = da.coords["fiberPoints"].values[0]
            elif fiber_point == "middle":
                fiber_point = da.coords["fiberPoints"].values[len(da.coords["fiberPoints"]) // 2]
            return fiber_point

        if "nodeTags" in da.dims:
            scalars = pos.sel(coords="x").copy() * 0
            if "fiberPoints" in da.dims:
                fiber_point = _reset_fiber_point(fiber_point, da)
                da = da.sel(fiberPoints=fiber_point)
            scalars.loc[{"nodeTags": da.coords["nodeTags"]}] = da
            return scalars

        if "fiberPoints" in da.dims and "GaussPoints" in da.dims:
            fiber_point = _reset_fiber_point(fiber_point, da)
            da = da.sel(fiberPoints=fiber_point)
            return da.sel(fiberPoints=fiber_point).mean(dim="GaussPoints", skipna=True)

        if "GaussPoints" in da.dims:
            return da.mean(dim="GaussPoints", skipna=True)

        return da  # fallback: return raw

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
        return step, (cmin, cmax)

    def _get_resp_clim(self):
        maxv = [np.max(data) for data in self.resp_step]
        minv = [np.min(data) for data in self.resp_step]
        cmin, cmax = np.min(minv), np.max(maxv)
        return cmin, cmax

    def _make_title(self, scalars, step, time):
        if self.resp_type.lower() in ["stressmeasures", "stressmeasuresatnodes"]:
            resp_type = "Stress Measures"
        elif self.resp_type.lower() in ["strainmeasures", "strainmeasuresatnodes"]:
            resp_type = "Strain Measures"
        elif self.resp_type.lower() in ["sectionforces", "sectionforcesatnodes"]:
            resp_type = "Section Forces"
        elif self.resp_type.lower() in ["sectiondeformations", "sectiondeformationsatnodes"]:
            resp_type = "Section Deformations"
        elif self.resp_type.lower() in ["stresses", "stressesatnodes"]:
            resp_type = "Stresses"
        elif self.resp_type.lower() in ["strains", "strainsatnodes"]:
            resp_type = "Strains"
        else:
            resp_type = self.resp_type.capitalize()
        info = {
            "title": self.ele_type.capitalize(),
            "resp_type": resp_type,
            "dof": self.component.capitalize(),
            "min": np.min(scalars),
            "max": np.max(scalars),
            "step": step,
            "time": time,
        }
        lines = [
            f"* {info['title']} Responses",
            f"* {info['resp_type']}",
            f"* {info['dof']} (DOF)",
            f"{info['min']:.3E} (min)",
            f"{info['max']:.3E} (max)",
            f"{info['step']} (step)",
            f"{info['time']:.3f} (time)",
        ]
        if self.unit_symbol:
            info["unit"] = self.unit_symbol
            lines.insert(3, f"{info['unit']} (unit)")
        if self.fiber_point and "Sec" not in resp_type and self.ele_type.lower() == "shell":
            info["fiber_point"] = self.fiber_point
            lines.insert(3, f"* {info['fiber_point']} (Fiber)")

        max_len = max(len(line) for line in lines)
        padded_lines = [line.rjust(max_len) for line in lines]
        text = "\n".join(padded_lines)
        return text + "\n"

    def _get_mesh_data(self, step, ele_tags, defo_scale):
        pos_defo = np.array(self._get_defo_coord_da(step, defo_scale))
        tags, pos, cells, cell_types = self._make_unstru_info(ele_tags, step)
        scalars = self.resp_step[step].to_numpy()
        return pos_defo, cells, cell_types, scalars

    def _create_mesh(
        self,
        plotter,
        value,
        ele_tags=None,
        plot_all_mesh=True,
        clim=None,
        style="surface",
        cpos="iso",
        defo_scale=1.0,
        show_outline=False,
        show_bc: bool = True,
        bc_scale: float = 1.0,
        show_mp_constraint: bool = False,
    ):
        step = round(value)
        node_no_deform_coords = np.array(self._get_node_da(step))
        pos, cells, cell_types, scalars = self._get_mesh_data(step, ele_tags, defo_scale)
        #  ---------------------------------
        plotter.clear_actors()  # !!!!!!
        if plot_all_mesh:
            self._plot_all_mesh(plotter, color="gray", step=step)
        resp_plot = _plot_unstru_cmap(
            plotter,
            pos=pos,
            cells=cells,
            cell_types=cell_types,
            scalars=scalars,
            cmap=self.pargs.cmap,
            clim=clim,
            show_scalar_bar=False,
            show_edges=self.pargs.show_mesh_edges,
            edge_color=self.pargs.mesh_edge_color,
            edge_width=self.pargs.mesh_edge_width,
            opacity=self.pargs.mesh_opacity,
            style=style,
            pos_origin=node_no_deform_coords,
        )

        title = self._make_title(scalars, step, self.time[step])
        scalar_bar = plotter.add_scalar_bar(title=title, **self.pargs.scalar_bar_kargs)
        if scalar_bar:
            # scalar_bar.SetTitle(title)
            title_prop = scalar_bar.GetTitleTextProperty()
            # title_prop.SetJustificationToRight()
            title_prop.BoldOn()

        if show_outline:
            self._plot_outline(plotter)
        bc_grid, mp_grid = None, None
        if show_bc:
            bc_grid = self._plot_bc(plotter, step, defo_scale=defo_scale, bc_scale=bc_scale)
        if show_mp_constraint:
            mp_grid = self._plot_mp_constraint(plotter, step, defo_scale=defo_scale)

        self._update_plotter(plotter, cpos)
        return resp_plot, scalar_bar, bc_grid, mp_grid

    def _update_mesh(
        self,
        step,
        ele_tags,
        resp_plot,
        scalar_bar,
        bc_grid=None,
        mp_grid=None,
        defo_scale=1.0,
        bc_scale: float = 1.0,
    ):
        step = round(step)
        pos, cells, cell_types, scalars = self._get_mesh_data(step, ele_tags, defo_scale)

        if resp_plot:
            resp_plot.points = pos
            resp_plot["scalars"] = scalars

        if scalar_bar:
            title = self._make_title(scalars, step, self.time[step])
            scalar_bar.SetTitle(title)

        if mp_grid:
            self._plot_mp_constraint_update(mp_grid, step, defo_scale=defo_scale)
        if bc_grid:
            self._plot_bc_update(bc_grid, step, defo_scale=defo_scale, bc_scale=bc_scale)

    def plot_slide(
        self,
        plotter,
        ele_tags=None,
        style="surface",
        plot_model=True,
        cpos="iso",
        show_defo=True,
        defo_scale: float = 1.0,
        show_bc: bool = True,
        bc_scale: float = 1.0,
        show_mp_constraint: bool = True,
        show_outline=False,
    ):
        _, clim = self._get_resp_peak()
        alpha_ = defo_scale if show_defo else 0.0
        if self.ModelUpdate:
            func = partial(
                self._create_mesh,
                plotter,
                ele_tags=ele_tags,
                clim=clim,
                plot_all_mesh=plot_model,
                style=style,
                cpos=cpos,
                defo_scale=alpha_,
                show_outline=show_outline,
                show_bc=show_bc,
                bc_scale=bc_scale,
                show_mp_constraint=show_mp_constraint,
            )
        else:
            resp_plot, scalar_bar, bc_grid, mp_grid = self._create_mesh(
                plotter,
                self.num_steps - 1,
                ele_tags=ele_tags,
                clim=clim,
                plot_all_mesh=plot_model,
                style=style,
                cpos=cpos,
                defo_scale=alpha_,
                show_outline=show_outline,
                show_bc=show_bc,
                bc_scale=bc_scale,
                show_mp_constraint=show_mp_constraint,
            )
            func = partial(
                self._update_mesh,
                ele_tags=ele_tags,
                resp_plot=resp_plot,
                scalar_bar=scalar_bar,
                bc_grid=bc_grid,
                mp_grid=mp_grid,
                bc_scale=bc_scale,
                defo_scale=alpha_,
            )
        plotter.add_slider_widget(func, [0, self.num_steps - 1], value=self.num_steps - 1, **self.slider_widget_args)

    def plot_peak_step(
        self,
        plotter,
        step="absMax",
        ele_tags=None,
        style="surface",
        plot_model=True,
        cpos="iso",
        defo_scale=1.0,
        show_defo=True,
        show_bc: bool = True,
        bc_scale: float = 1.0,
        show_mp_constraint: bool = True,
        show_outline=False,
    ):
        step, clim = self._get_resp_peak(idx=step)
        alpha_ = defo_scale if show_defo else 0.0
        self._create_mesh(
            plotter=plotter,
            value=step,
            ele_tags=ele_tags,
            clim=clim,
            plot_all_mesh=plot_model,
            style=style,
            cpos=cpos,
            defo_scale=alpha_,
            show_outline=show_outline,
            show_bc=show_bc,
            bc_scale=bc_scale,
            show_mp_constraint=show_mp_constraint,
        )

    def plot_anim(
        self,
        plotter,
        ele_tags=None,
        framerate: Optional[int] = None,
        savefig: str = "ShellRespAnimation.gif",
        style="surface",
        plot_model=True,
        cpos="iso",
        defo_scale=1.0,
        show_defo=True,
        show_bc: bool = True,
        bc_scale: float = 1.0,
        show_mp_constraint: bool = True,
        show_outline=False,
    ):
        if framerate is None:
            framerate = np.ceil(self.num_steps / 11)
        if savefig.endswith(".gif"):
            plotter.open_gif(savefig, fps=framerate)
        else:
            plotter.open_movie(savefig, framerate=framerate)
        _, clim = self._get_resp_peak()
        alpha_ = defo_scale if show_defo else 0.0
        # plotter.write_frame()  # write initial data

        if self.ModelUpdate:
            for step in range(self.num_steps):
                self._create_mesh(
                    plotter,
                    step,
                    ele_tags=ele_tags,
                    clim=clim,
                    plot_all_mesh=plot_model,
                    style=style,
                    cpos=cpos,
                    defo_scale=alpha_,
                    show_outline=show_outline,
                    show_bc=show_bc,
                    bc_scale=bc_scale,
                    show_mp_constraint=show_mp_constraint,
                )
                plotter.write_frame()
        else:
            resp_plot, scalar_bar, bc_grid, mp_grid = self._create_mesh(
                plotter,
                0,
                ele_tags=ele_tags,
                clim=clim,
                plot_all_mesh=plot_model,
                style=style,
                cpos=cpos,
                defo_scale=alpha_,
                show_outline=show_outline,
                show_bc=show_bc,
                bc_scale=bc_scale,
                show_mp_constraint=show_mp_constraint,
            )
            plotter.write_frame()
            for step in range(1, self.num_steps):
                self._update_mesh(
                    step=step,
                    ele_tags=ele_tags,
                    resp_plot=resp_plot,
                    scalar_bar=scalar_bar,
                    bc_grid=bc_grid,
                    mp_grid=mp_grid,
                    bc_scale=bc_scale,
                    defo_scale=alpha_,
                )
                plotter.write_frame()


def plot_unstruct_responses(
    odb_tag: Union[int, str] = 1,
    ele_type: str = "Shell",
    ele_tags: Optional[Union[int, list]] = None,
    slides: bool = False,
    step: Union[int, str] = "absMax",
    resp_type: str = "sectionForces",
    resp_dof: str = "MXX",
    shell_fiber_loc: Optional[Union[str, int]] = "top",
    style: str = "surface",
    unit_symbol: Optional[str] = None,
    unit_factor: Optional[float] = None,
    show_defo: bool = True,
    defo_scale: float = 1.0,
    show_bc: bool = True,
    bc_scale: float = 1.0,
    show_mp_constraint: bool = False,
    show_outline: bool = False,
    cpos: str = "iso",
    show_model: bool = True,
) -> pv.Plotter:
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

        #. For ``Shell`` elements, one of ["sectionForces", "sectionDeformations", "sectionForcesAtNodes", "sectionDeformationsAtNodes", "Stresses", "Strains", "StressesAtNodes", "StrainsAtNodes"].
            If it endswith `AtNodes`, responses at nodes will be displayed,
            else responses at Gaussian integration points will be averaged for each element (per unit length).
            If None, defaults to "sectionForces".
        #. For ``Plane`` elements, one of ["stresses", "strains", "stressesAtNodes", "strainsAtNodes"].
            If it endswith `AtNodes`, responses at nodes will be displayed,
            else responses at Gaussian integration points will be averaged for each element.
            If None, defaults to "stresses".
        #. For ``Brick`` or ``Solid`` elements, one of ["stresses", "strains", "stressesAtNodes", "strainsAtNodes"].
            If it endswith `AtNodes`, responses at nodes will be displayed,
            else responses at Gaussian integration points will be averaged for each element.
            If None, defaults to "stresses".

    resp_dof: str, default: None
        Dof to be visualized, which dependents on the element type `ele_type`.

        .. Note::
            The `resp_dof` here is consistent with stress-strain (force-deformation),
            and whether it is stress or strain depends on the parameter `resp_type`.

        #. For ``Shell`` elements, If resp_type is the section responses, one of ["FXX", "FYY", "FXY", "MXX", "MYY", "MXY", "VXZ", "VYZ"]. If resp_type is the stress or strain, one of ["sigma11", "sigma22", "sigma12", "sigma23", "sigma13"].
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

    shell_fiber_loc: Optional[Union[str, int]], default: "top", added in v1.0.16
        The location of the fiber point for shell elements.
        If str, one of ["top", "bottom", "middle"].
        If int, the index of the fiber point to be visualized, from 1 (bottom) to N (top).
        The fiber point is the fiber layer in the shell section.
        Note that this parameter is only valid for stresses and strains in shell elements.

    unit_symbol: str, default: None
        Unit symbol to be displayed in the plot.
    unit_factor: float, default: None
        The multiplier used to convert units.
        For example, if you want to visualize stress and the current data unit is kPa, you can set ``unit_symbol="kPa" and unit_factor=1.0``.
        If you want to visualize in MPa, you can set ``unit_symbol="MPa" and unit_factor=0.001``.
    style: str, default: surface
        Visualization the mesh style of surfaces and solids.
        One of the following: style='surface', style='wireframe', style='points', style='points_gaussian'.
        Defaults to 'surface'. Note that 'wireframe' only shows a wireframe of the outer geometry.
    show_defo: bool, default: True
        Whether to display the deformed shape.
    defo_scale: float, default: 1.0
        Scales the size of the deformation presentation when show_defo is True.
    show_bc: bool, default: True
        Whether to display boundary supports.
    bc_scale: float, default: 1.0
        Scale the size of boundary support display.
    show_mp_constraint: bool, default: False
        Whether to show multipoint (MP) constraint.
    show_outline: bool, default: False
        Whether to display the outline of the model.
    cpos: str, default: iso
        Model display perspective, optional: "iso", "xy", "yx", "xz", "zx", "yz", "zy".
        If 3d, defaults to "iso". If 2d, defaults to "xy".
    show_model: bool, default: True
        Whether to plot the all model or not.

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
    ele_type, resp_type, resp_dof, fiber_pts = _check_input(ele_type, resp_type, resp_dof, fiber_pts=shell_fiber_loc)
    model_info_steps, model_update, resp_step = loadODB(odb_tag, resp_type=ele_type)
    if show_defo:
        _, _, node_resp_steps = loadODB(odb_tag, resp_type="Nodal", verbose=False)
    else:
        node_resp_steps = None
    plotter = pv.Plotter(
        notebook=PLOT_ARGS.notebook,
        line_smoothing=PLOT_ARGS.line_smoothing,
        polygon_smoothing=PLOT_ARGS.polygon_smoothing,
        off_screen=PLOT_ARGS.off_screen,
    )
    plotbase = PlotUnstruResponse(model_info_steps, resp_step, model_update, node_resp_steps=node_resp_steps)
    plotbase.set_unit(symbol=unit_symbol, factor=unit_factor)
    plotbase.refactor_resp_step(
        ele_tags=ele_tags, ele_type=ele_type, resp_type=resp_type, component=resp_dof, fiber_point=fiber_pts
    )
    if slides:
        plotbase.plot_slide(
            plotter,
            ele_tags=ele_tags,
            style=style,
            cpos=cpos,
            plot_model=show_model,
            defo_scale=defo_scale,
            show_defo=show_defo,
            show_bc=show_bc,
            bc_scale=bc_scale,
            show_mp_constraint=show_mp_constraint,
            show_outline=show_outline,
        )
    else:
        plotbase.plot_peak_step(
            plotter,
            ele_tags=ele_tags,
            step=step,
            style=style,
            cpos=cpos,
            plot_model=show_model,
            defo_scale=defo_scale,
            show_defo=show_defo,
            show_bc=show_bc,
            bc_scale=bc_scale,
            show_mp_constraint=show_mp_constraint,
            show_outline=show_outline,
        )
    if PLOT_ARGS.anti_aliasing:
        plotter.enable_anti_aliasing(PLOT_ARGS.anti_aliasing)
    return plotbase._update_plotter(plotter, cpos)


def plot_unstruct_responses_animation(
    odb_tag: Union[int, str] = 1,
    ele_tags: Optional[Union[int, list]] = None,
    framerate: Optional[int] = None,
    ele_type: str = "Shell",
    resp_type: Optional[str] = None,
    resp_dof: Optional[str] = None,
    shell_fiber_loc: Optional[Union[str, int]] = "top",
    savefig: Optional[str] = None,
    off_screen: bool = True,
    style: str = "surface",
    unit_symbol: Optional[str] = None,
    unit_factor: Optional[float] = None,
    show_defo: bool = True,
    defo_scale: float = 1.0,
    show_bc: bool = True,
    bc_scale: float = 1.0,
    show_mp_constraint: bool = False,
    show_outline: bool = False,
    cpos: str = "iso",
    show_model: bool = True,
) -> pv.Plotter:
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
    savefig: str, default: None
        Path to save the animation. The suffix can be ``.gif`` or ``.mp4``.
    off_screen: bool, default: True
        Off-screen rendering, i.e., not showing the rendering window.
        If False, the rendering window will be displayed.
    resp_type: str, default: None
        Response type, which dependents on the element type `ele_type`.

        #. For ``Shell`` elements, one of ["sectionForces", "sectionDeformations", "sectionForcesAtNodes", "sectionDeformationsAtNodes", "Stresses", "Strains", "StressesAtNodes", "StrainsAtNodes"].
            If it endswith `AtNodes`, responses at nodes will be displayed,
            else responses at Gaussian integration points will be averaged for each element (per unit length).
            If None, defaults to "sectionForces".
        #. For ``Plane`` elements, one of ["stresses", "strains", "stressesAtNodes", "strainsAtNodes"].
            If it endswith `AtNodes`, responses at nodes will be displayed,
            else responses at Gaussian integration points will be averaged for each element.
            If None, defaults to "stresses".
        #. For ``Brick`` or ``Solid`` elements, one of ["stresses", "strains", "stressesAtNodes", "strainsAtNodes"].
            If it endswith `AtNodes`, responses at nodes will be displayed,
            else responses at Gaussian integration points will be averaged for each element.
            If None, defaults to "stresses".

    resp_dof: str, default: None
        Dof to be visualized, which dependents on the element type `ele_type`.

        .. Note::
            The `resp_dof` here is consistent with stress-strain (force-deformation),
            and whether it is stress or strain depends on the parameter `resp_type`.

        #. For ``Shell`` elements, If resp_type is the section responses, one of ["FXX", "FYY", "FXY", "MXX", "MYY", "MXY", "VXZ", "VYZ"]. If resp_type is the stress or strain, one of ["sigma11", "sigma22", "sigma12", "sigma23", "sigma13"].
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

    shell_fiber_loc: Optional[Union[str, int]], default: "top", added in v1.0.16
        The location of the fiber point for shell elements.
        If str, one of ["top", "bottom", "middle"].
        If int, the index of the fiber point to be visualized, from 1 (bottom) to N (top).
        The fiber point is the fiber layer in the shell section.
        Note that this parameter is only valid for stresses and strains in shell elements.

    unit_symbol: str, default: None
        Unit symbol to be displayed in the plot.
    unit_factor: float, default: None
        The multiplier used to convert units.
        For example, if you want to visualize stress and the current data unit is kPa, you can set ``unit_symbol="kPa" and unit_factor=1.0``.
        If you want to visualize in MPa, you can set ``unit_symbol="MPa" and unit_factor=0.001``.
    style: str, default: surface
        Visualization the mesh style of surfaces and solids.
        One of the following: style='surface', style='wireframe', style='points', style='points_gaussian'.
        Defaults to 'surface'. Note that 'wireframe' only shows a wireframe of the outer geometry.
    show_defo: bool, default: True
        Whether to display the deformed shape.
    defo_scale: float, default: 1.0
        Scales the size of the deformation presentation when show_defo is True.
    show_bc: bool, default: True
        Whether to display boundary supports.
    bc_scale: float, default: 1.0
        Scale the size of boundary support display.
    show_mp_constraint: bool, default: False
        Whether to show multipoint (MP) constraint.
    show_outline: bool, default: False
        Whether to display the outline of the model.
    cpos: str, default: iso
        Model display perspective, optional: "iso", "xy", "yx", "xz", "zx", "yz", "zy".
        If 3d, defaults to "iso". If 2d, defaults to "xy".
    show_model: bool, default: True
        Whether to plot the all model or not.

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
    ele_type, resp_type, resp_dof, fiber_point = _check_input(ele_type, resp_type, resp_dof, fiber_pts=shell_fiber_loc)
    if savefig is None:
        savefig = f"{ele_type.capitalize()}RespAnimation.gif"
    model_info_steps, model_update, resp_step = loadODB(odb_tag, resp_type=ele_type)
    if show_defo:
        _, _, node_resp_steps = loadODB(odb_tag, resp_type="Nodal", verbose=False)
    else:
        node_resp_steps = None
    plotter = pv.Plotter(
        notebook=PLOT_ARGS.notebook,
        line_smoothing=PLOT_ARGS.line_smoothing,
        polygon_smoothing=PLOT_ARGS.polygon_smoothing,
        off_screen=off_screen,
    )
    plotbase = PlotUnstruResponse(model_info_steps, resp_step, model_update, node_resp_steps=node_resp_steps)
    plotbase.set_unit(symbol=unit_symbol, factor=unit_factor)
    plotbase.refactor_resp_step(
        ele_tags=ele_tags, ele_type=ele_type, resp_type=resp_type, component=resp_dof, fiber_point=fiber_point
    )
    plotbase.plot_anim(
        plotter,
        ele_tags=ele_tags,
        framerate=framerate,
        savefig=savefig,
        style=style,
        cpos=cpos,
        plot_model=show_model,
        defo_scale=defo_scale,
        show_defo=show_defo,
        show_bc=show_bc,
        bc_scale=bc_scale,
        show_mp_constraint=show_mp_constraint,
        show_outline=show_outline,
    )
    if PLOT_ARGS.anti_aliasing:
        plotter.enable_anti_aliasing(PLOT_ARGS.anti_aliasing)
    print(f"Animation has been saved as {savefig}!")
    return plotbase._update_plotter(plotter, cpos)


def _check_input(ele_type, resp_type, resp_dof, fiber_pts=None):
    if ele_type.lower() == "shell":
        ele_type = "Shell"
        resp_type, resp_dof, fiber_pts = _check_input_shell(resp_type, resp_dof, fiber_pts)
    elif ele_type.lower() == "plane":
        ele_type = "Plane"
        resp_type, resp_dof = _check_input_plane(resp_type, resp_dof)
    elif ele_type.lower() in ["brick", "solid"]:
        ele_type = "Brick"
        resp_type, resp_dof = _check_input_solid(resp_type, resp_dof)
    else:
        raise ValueError(f"Not supported element type {ele_type}! Valid options are: Shell, Plane, Brick.")  # noqa: TRY003
    return ele_type, resp_type, resp_dof, fiber_pts


def _check_input_shell(resp_type, resp_dof, fiber_pts=None):
    if resp_type is None:
        resp_type = "sectionForces"
    resp_type_lower = resp_type.lower()

    valid_resp_map = {
        "sectionforces": "sectionForces",
        "sectiondeformations": "sectionDeformations",
        "sectionforcesatnodes": "sectionForcesAtNodes",
        "sectiondeformationsatnodes": "sectionDeformationsAtNodes",
        "stresses": "Stresses",
        "strains": "Strains",
        "stressesatnodes": "StressesAtNodes",
        "strainsatnodes": "StrainsAtNodes",
    }

    if resp_type_lower not in valid_resp_map:
        raise ValueError(
            f"Not supported GP response type {resp_type}! Valid options are: "
            + ", ".join(valid_resp_map.values())
            + "."
        )

    resp_type = valid_resp_map[resp_type_lower]

    if "section" in resp_type.lower():
        valid_dofs = {"fxx", "fyy", "fxy", "mxx", "myy", "mxy", "vxz", "vyz"}
        if resp_dof is None:
            resp_dof = "MXX"
    else:
        valid_dofs = {"sigma11", "sigma22", "sigma12", "sigma23", "sigma13"}
        if resp_dof is None:
            resp_dof = "sigma11"
        # fiber_pts check
        if fiber_pts is None:
            fiber_pts = "top"
        elif isinstance(fiber_pts, str):
            fiber_pts = fiber_pts.lower()
            if fiber_pts not in {"top", "bottom", "middle"}:
                raise ValueError(f"Not supported fiber points {fiber_pts}! Valid options are: top, bottom, middle.")  # noqa: TRY003
        else:
            fiber_pts = int(fiber_pts)

    if resp_dof.lower() not in valid_dofs:
        raise ValueError(  # noqa: TRY003
            f"Not supported component {resp_dof}! Valid options are: {', '.join(d.upper() for d in valid_dofs)}."
        )

    return resp_type, resp_dof, fiber_pts


def _check_input_plane(resp_type, resp_dof):
    if resp_type is None:
        resp_type = "Stresses"

    resp_type_lower = resp_type.lower()
    type_map = {
        "stresses": "Stresses",
        "stress": "Stresses",
        "stressesatnodes": "StressesAtNodes",
        "stressatnodes": "StressesAtNodes",
        "strains": "Strains",
        "strain": "Strains",
        "strainsatnodes": "StrainsAtNodes",
        "strainatnodes": "StrainsAtNodes",
    }

    if resp_type_lower not in type_map:
        raise ValueError(  # noqa: TRY003
            f"Not supported response type {resp_type}! "
            "Valid options are: Stresses, StressesAtNodes, Strains, StrainsAtNodes"
        )

    is_stress = "stress" in resp_type_lower
    is_node = "nodes" in resp_type_lower

    if resp_dof is None:
        resp_dof = "sigma_vm"

    resp_dof_lower = resp_dof.lower()
    tensor_dofs = {"sigma11", "sigma22", "sigma12"}
    measure_dofs = {"p1", "p2", "sigma_vm", "tau_max"}

    if resp_dof_lower in measure_dofs:
        resp_type = ("StressMeasures" if is_stress else "StrainMeasures") + ("AtNodes" if is_node else "")
    elif resp_dof_lower in tensor_dofs:
        resp_type = ("Stresses" if is_stress else "Strains") + ("AtNodes" if is_node else "")
        if not is_stress:
            resp_dof = resp_dof_lower.replace("sigma", "eps")
    else:
        raise ValueError(  # noqa: TRY003
            f"Not supported component {resp_dof}! "
            "Valid options are: sigma11, sigma22, sigma12, p1, p2, sigma_vm, tau_max."
        )

    return resp_type, resp_dof


def _check_input_solid(resp_type, resp_dof):
    if resp_type is None:
        resp_type = "Stresses"

    resp_type_lower = resp_type.lower()
    type_map = {
        "stresses": "Stresses",
        "stress": "Stresses",
        "stressesatnodes": "StressesAtNodes",
        "stressatnodes": "StressesAtNodes",
        "strains": "Strains",
        "strain": "Strains",
        "strainsatnodes": "StrainsAtNodes",
        "strainatnodes": "StrainsAtNodes",
    }

    if resp_type_lower not in type_map:
        raise ValueError(  # noqa: TRY003
            f"Not supported response type {resp_type}! "
            "Valid options are: Stresses, StressesAtNodes, Strains, StrainsAtNodes"
        )

    is_stress = "stress" in resp_type_lower
    is_node = "nodes" in resp_type_lower

    if resp_dof is None:
        resp_dof = "sigma_vm"

    resp_dof_lower = resp_dof.lower()
    tensor_dofs = {"sigma11", "sigma22", "sigma33", "sigma12", "sigma23", "sigma13"}
    measure_dofs = {"p1", "p2", "p3", "sigma_vm", "tau_max", "sigma_oct", "tau_oct"}

    if resp_dof_lower in measure_dofs:
        resp_type = ("StressMeasures" if is_stress else "StrainMeasures") + ("AtNodes" if is_node else "")
    elif resp_dof_lower in tensor_dofs:
        resp_type = ("Stresses" if is_stress else "Strains") + ("AtNodes" if is_node else "")
        if not is_stress:
            resp_dof = resp_dof_lower.replace("sigma", "eps")
    else:
        raise ValueError(  # noqa: TRY003
            f"Not supported component {resp_dof}! "
            "Valid options are: sigma11, sigma22, sigma33, sigma12, sigma23, sigma13, "
            "p1, p2, p3, sigma_vm, tau_max, sigma_oct, tau_oct."
        )

    return resp_type, resp_dof
