from __future__ import annotations

import os
from pathlib import Path
import shutil
import time
import warnings
from types import SimpleNamespace
from typing import Literal, TypedDict

import numpy as np
import xarray as xr
from typing_extensions import Unpack

from ..utils import CONFIGS, get_opensees_module, get_random_color
from ._get_response import (
    BrickRespStepData,
    ContactRespStepData,
    FiberSecRespStepData,
    FrameRespStepData,
    LinkRespStepData,
    ModelInfoStepData,
    NodalRespStepData,
    PlaneRespStepData,
    SensitivityRespStepData,
    ShellRespStepData,
    TrussRespStepData,
)
from ._post_utils import generate_chunk_encoding_for_datatree
from ._unit_postprocess import get_post_unit_multiplier, get_post_unit_symbol
from .eigen_data import save_eigen_data
from .model_data import save_model_data

ops = get_opensees_module()


class _POST_ARGS_TYPES(TypedDict, total=False):
    elastic_frame_sec_points: int
    interpolate_beam_disp: bool | int = False
    compute_mechanical_measures: bool
    project_gauss_to_nodes: str | None
    section_response_dof: dict[str, list[str]] | None
    # -------------------------------------------
    save_nodal_resp: bool
    save_frame_resp: bool
    save_truss_resp: bool
    save_link_resp: bool
    save_shell_resp: bool
    save_fiber_sec_resp: bool
    save_plane_resp: bool
    save_brick_resp: bool
    save_contact_resp: bool
    save_sensitivity_resp: bool
    # -------------------------------------------
    node_tags: list[int] | tuple[int, ...] | int | None
    frame_tags: list[int] | tuple[int, ...] | int | None
    truss_tags: list[int] | tuple[int, ...] | int | None
    link_tags: list[int] | tuple[int, ...] | int | None
    shell_tags: list[int] | tuple[int, ...] | int | None
    fiber_ele_tags: list[int] | str | None
    plane_tags: list[int] | tuple[int, ...] | int | None
    brick_tags: list[int] | tuple[int, ...] | int | None
    contact_tags: list[int] | tuple[int, ...] | int | None
    sensitivity_para_tags: list[int] | tuple[int, ...] | int | None
    # ----------------------------------------------------------------


_ELE_RESP_READERS = {
    "frame": FrameRespStepData,
    "beam": FrameRespStepData,
    "fibersec": FiberSecRespStepData,
    "fibersection": FiberSecRespStepData,
    "truss": TrussRespStepData,
    "link": LinkRespStepData,
    "shell": ShellRespStepData,
    "plane": PlaneRespStepData,
    "brick": BrickRespStepData,
    "solid": BrickRespStepData,
    "contact": ContactRespStepData,
}

_UNIT_SYSTEM = SimpleNamespace(unit_factors=None, unit_symbols=None)

_POST_ARGS_DEFAULT = {
    "elastic_frame_sec_points": 7,
    "interpolate_beam_disp": False,
    "section_response_dof": None,
    "compute_mechanical_measures": "All",
    "project_gauss_to_nodes": "copy",
    # ------------------------------
    "save_nodal_resp": True,
    "save_frame_resp": True,
    "save_truss_resp": True,
    "save_link_resp": True,
    "save_shell_resp": True,
    "save_fiber_sec_resp": False,
    "save_plane_resp": True,
    "save_brick_resp": True,
    "save_contact_resp": True,
    "save_sensitivity_resp": False,
    # ----------------------------------
    "node_tags": None,
    "frame_tags": None,
    "truss_tags": None,
    "link_tags": None,
    "shell_tags": None,
    "fiber_ele_tags": None,
    "plane_tags": None,
    "brick_tags": None,
    "contact_tags": None,
    "sensitivity_para_tags": None,
    # -----------------------------------
}


class CreateODB:
    """Create an output database (ODB) to save response data.

    Parameters
    ------------
    odb_tag: Union[int, str], default: 1
        Tag of output databases (ODB) to be saved.
        This tag can be used to identify the load case and is used in post-processing and visualization to identify which results are processed.
    model_update: bool, default: False
        Whether to update the model data.

        .. Note::
            If True, the model data will be updated at each step.
            If no nodes and elements are added or removed during the analysis of your model,
            keep this parameter set to **False**.
            Enabling model updates unnecessarily can increase memory usage and slow down performance.
            If some nodes or elements are deleted during the analysis, you should set this parameter to `True`.
    save_every: Optional[int], default: None
        Save response data at a fixed interval of analysis steps.

        Added since version 1.0.25.

        .. Note::
            If set to an integer value, response data will be written to disk every save_every steps.
            This is particularly useful for large models with many analysis steps, as it allows responses to be flushed periodically into multiple ODB files, significantly reducing peak memory usage.

            If set to None, all response data are accumulated in memory and written to a single ODB file at the end of the analysis.
            This option provides the best runtime performance due to minimal I/O overhead, but may require substantial memory for large-scale models.

            In practice, this parameter should be chosen based on the model size and available system memory.
            If sufficient memory is available, it is recommended to keep ``save_every=None``.
            Smaller values of save_every increase the frequency of disk writes, leading to higher disk usage and potentially slower data loading in subsequent post-processing stages.
    dtype: dict, default: dict(int=np.int32, float=np.float32)
        Set integer and floating point precision types.
    zlib: bool, default: False
        Whether to compress the response data file.
        Only works when odb_format is set to "nc" in `opstool.post.set_odb_format()`.
        Added since version 1.0.25.
    kwargs: Other post-processing parameters, optional:
        * elastic_frame_sec_points: int, default: 7
            The number of elastic frame elements section points.
            A larger number may result in a larger file size.
        * interpolate_beam_disp: Union[bool, int], default: False, added since version 1.0.25
            Whether to interpolate beam displacements for nodal response of beam elements.
            If True, shape functions will be used to interpolate the displacements of beam elements for a smoother visualization.
            If an integer n is provided, it specifies the number of interpolation points along each beam element.
            If you have a large number of beam elements, enabling this option may slow down the plotting process, and it is recommended to disable it.
            Interpolation will not have a significant effect when applied to sufficiently subdivided beam elements; instead,
            it will increase the data size and slow down the speed.
        * section_response_dof: Optional[dict], default: None
            A dictionary to specify the section response type for different section types.
            The keys are the section types, and the values are the response types.
            For example, to specify the response type for "SectionAggregator", you can set:

            ..  code-block:: python

                {
                    "SectionAggregator": ["P", "MZ", "MY", "T", "VY", "VZ"]  # means you use the section "P", "MZ", "MY", "T" and addtional "VY", "VZ" dof.
                }

            This is because for some section types, such as ``Aggregator``, the number and order of the degrees of freedom are specified by the user.
            For other sections, the program can determine them automatically.
        * compute_mechanical_measures: Union[bool, str, dict], default: "All"
            Whether to compute mechanical measures for ``solid and planar elements``,
            including principal stresses, principal strains, von Mises stresses, etc.
            If False, no mechanical measures will be computed.
            If True or "All", all mechanical measures will be computed.

            Otherwise, you can specify a dictionary to indicate which measures to compute.
            Dictionary keys are the names of mechanical measures, and values are parameters needed for computation.
            Only the measures specified in the dictionary will be computed.
            For example, the default setting is:

            ..  code-block:: python

                {
                    "principal": None,  # principal stresses
                    "von-mises": None,  # von Mises stress
                    "octahedral": None,  # octahedral stresses (sigma_oct, tau_oct)
                    "tau_max": None,  # maximum shear stress (Tresca)
                }

            where the keys are the names of mechanical measures to be computed, and the values are None because no additional parameters are needed for these measures.
            If you want to compute the Mohr-Coulomb or Drucker-Prager equivalent stress with specific parameters, you can set:

            ..  code-block:: python

                {
                    "principal": None,  # principal stresses
                    "von-mises": None,  # von Mises stress
                    "octahedral": None,  # octahedral stresses (sigma_oct, tau_oct)
                    "tau_max": None,  # maximum shear stress (Tresca)
                    # Mohr-Coulomb  with Tension-Compression Form
                    "mohr_coulomb_sy": {"syc": <compressive strength>, "syt": <tensile strength>},
                    # Mohr-Coulomb with Cohesion-Friction Form
                    "mohr_coulomb_c_phi": {"c": <cohesion>, "phi": <friction angle in degrees>},
                    # Drucker-Prager with Tension-Compression Form
                    "drucker_prager_sy": {"syc": <compressive strength>, "syt": <tensile strength>},
                    # Drucker-Prager with Cohesion-Friction Form
                    "drucker_prager_c_phi": {"c": <cohesion>, "phi": <friction angle in degrees>, "kind": <circumscribed/middle/inscribed>},
                }

            For detailed information, see :ref:`theory_stress_criteria`.

        * project_gauss_to_nodes: Optional[str], default: "copy"
            Method to project Gauss point responses to nodes. Options are:

            * ``"copy"``: Copy Gauss point responses to nodes, that is, the node's response comes from the nearest Gaussian point.
            * ``"average"``: Average Gauss point responses to nodes by integrate weight.
            * ``"extrapolate"``: Extrapolate Gauss point responses to nodes by element shape function.
            * ``None`` or ``False``: Do not project Gauss point responses to nodes.

        * Whether to save the responses:
            * save_nodal_resp: bool, default: True
                Whether to save nodal responses.
            * save_frame_resp: bool, default: True
                Whether to save frame element responses.
            * save_truss_resp: bool, default: True
                Whether to save truss element responses.
            * save_link_resp: bool, default: True
                Whether to save link element responses.
            * save_shell_resp: bool, default: True
                Whether to save shell element responses.
            * save_fiber_sec_resp: bool, default: True
                Whether to save fiber section responses.
            * save_plane_resp: bool, default: True
                Whether to save plane element responses.
            * save_brick_resp: bool, default: True
                Whether to save brick element responses.
            * save_contact_resp: bool, default: True
                Whether to save contact element responses.
            * save_sensitivity_resp: bool, default: False
                Whether to save sensitivity analysis responses.
        * Nodes or elements that need to be saved:
            * node_tags: Union[list, tuple, int], default: None
                Node tags to be saved.
                If None, save all nodes' responses.
            * frame_tags: Union[list, tuple, int], default: None
                Frame element tags to be saved.
                If None, save all frame elements' responses.
            * truss_tags: Union[list, tuple, int], default: None
                Truss element tags to be saved.
                If None, save all truss elements' responses.
            * link_tags: Union[list, tuple, int], default: None
                Link element tags to be saved.
                If None, save all link elements' responses.
            * shell_tags: Union[list, tuple, int], default: None
                Shell element tags to be saved.
                If None, save all shell elements' responses.
            * fiber_ele_tags: Union[list, str], default: None
                Element tags that contain fiber sections to be saved.
                If "all", save all fiber section elements responses.
                If None, save nothing.
            * plane_tags: Union[list, tuple, int], default: None
                Plane element tags to be saved.
                If None, save all plane elements' responses.
            * brick_tags: Union[list, tuple, int], default: None
                Brick element tags to be saved.
                If None, save all brick elements' responses.
            * contact_tags: Union[list, tuple, int], default: None
                Contact element tags to be saved.
            * sensitivity_para_tags: Union[list, tuple, int], default: None
                Sensitivity parameter tags to be saved.

            .. Note::
                If you enter optional node and element tags to avoid saving all data,
                make sure these tags are not deleted during the analysis.
                Otherwise, unexpected behavior may occur.
    """

    def __init__(
        self,
        odb_tag: int | str = 1,
        model_update: bool = False,
        save_every: int | None = None,
        dtype: dict[str, np.dtype] | None = None,
        zlib: bool | None = False,
        **kwargs: Unpack[_POST_ARGS_TYPES],
    ):
        RESULTS_DIR = CONFIGS.get_output_dir()
        RESP_FILE_NAME = CONFIGS.get_resp_filename()

        self._odb_tag = odb_tag
        self._model_update = model_update
        self._odb_format = CONFIGS.get_odb_format()[0]
        self._flush_every = save_every
        self._store_path = str(Path(RESULTS_DIR) / f"{RESP_FILE_NAME}-{self._odb_tag}.odb")
        self._pending_steps = 0
        self._file_idx = 1
        self._zlib = zlib

        self.resp_kargs = {"model_update": model_update, "dtype": dtype}

        self._POST_ARGS = SimpleNamespace(**_POST_ARGS_DEFAULT)
        for key, value in kwargs.items():
            if key not in list(vars(self._POST_ARGS).keys()):
                raise KeyError(f"Incorrect parameter {key}, should be one of {list(vars(self._POST_ARGS).keys())}!")  # noqa: TRY003
            else:
                setattr(self._POST_ARGS, key, value)

        self._ModelInfo = None
        self._NodalResp = None
        self._FrameResp = None
        self._TrussResp = None
        self._LinkResp = None
        self._ShellResp = None
        self._FiberSecResp = None
        self._PlaneResp = None
        self._BrickResp = None
        self._ContactResp = None
        self._SensitivityResp = None

        self._set_resp()

        self._RESPS = [
            self._ModelInfo,
            self._NodalResp,
            self._FrameResp,
            self._TrussResp,
            self._LinkResp,
            self._ShellResp,
            self._FiberSecResp,
            self._PlaneResp,
            self._BrickResp,
            self._ContactResp,
            self._SensitivityResp,
        ]

        self._init_path()

    def _set_resp(self):
        self._set_model_info()
        self._set_node_resp()
        self._set_frame_resp()
        self._set_truss_resp()
        self._set_link_resp()
        self._set_shell_resp()
        self._set_fiber_sec_resp()
        self._set_plane_resp()
        self._set_brick_resp()
        self._set_contact_resp()
        self._set_sensitivity_resp()

    def _get_resp(self):
        return self._RESPS

    def _init_path(self):

        path = Path(self._store_path)

        if path.exists():
            for item in path.iterdir():
                if item.is_dir():
                    shutil.rmtree(item)
                else:
                    item.unlink()
        else:
            path.mkdir(parents=True, exist_ok=True)

    def _set_model_info(self):
        if self._ModelInfo is None:
            self._ModelInfo = ModelInfoStepData(**self.resp_kargs)
        else:
            self._ModelInfo.add_resp_data_one_step()

    def _set_node_resp(self):
        _save_nodal_resp = self._POST_ARGS.save_nodal_resp
        _node_tags = self._POST_ARGS.node_tags
        node_tags = _node_tags if _node_tags is not None else self._ModelInfo.get_current_node_tags()
        model_info = self._ModelInfo.get_current_model_info() if self._POST_ARGS.interpolate_beam_disp else None
        if node_tags is not None:
            node_tags = [int(tag) for tag in np.atleast_1d(node_tags)]  # Ensure tags are integers
        if len(node_tags) > 0 and _save_nodal_resp:
            if self._NodalResp is None:
                self._NodalResp = NodalRespStepData(
                    node_tags,
                    interpolate_beam=self._POST_ARGS.interpolate_beam_disp,
                    model_info=model_info,
                    **self.resp_kargs,
                )
            else:
                self._NodalResp.add_resp_data_one_step(node_tags, model_info=model_info)

    def _set_frame_resp(self):
        _save_frame_resp = self._POST_ARGS.save_frame_resp
        _frame_tags = self._POST_ARGS.frame_tags
        frame_tags = _frame_tags if _frame_tags is not None else self._ModelInfo.get_current_frame_tags()
        if frame_tags is not None:
            frame_tags = [int(tag) for tag in np.atleast_1d(frame_tags)]  # Ensure tags are integers
        frame_load_data = self._ModelInfo.get_current_frame_load_data()
        if len(frame_tags) > 0 and _save_frame_resp:
            if self._FrameResp is None:
                self._FrameResp = FrameRespStepData(
                    frame_tags,
                    frame_load_data,
                    elastic_frame_sec_points=self._POST_ARGS.elastic_frame_sec_points,
                    section_response_dof=self._POST_ARGS.section_response_dof,
                    **self.resp_kargs,
                )
            else:
                self._FrameResp.add_resp_data_one_step(frame_tags, frame_load_data)

    def _set_truss_resp(self):
        _save_truss_resp = self._POST_ARGS.save_truss_resp
        _truss_tags = self._POST_ARGS.truss_tags
        truss_tags = _truss_tags if _truss_tags is not None else self._ModelInfo.get_current_truss_tags()
        if truss_tags is not None:
            truss_tags = [int(tag) for tag in np.atleast_1d(truss_tags)]  # Ensure tags are integers
        if len(truss_tags) > 0 and _save_truss_resp:
            if self._TrussResp is None:
                self._TrussResp = TrussRespStepData(truss_tags, **self.resp_kargs)
            else:
                self._TrussResp.add_resp_data_one_step(truss_tags)

    def _set_link_resp(self):
        _save_link_resp = self._POST_ARGS.save_link_resp
        _link_tags = self._POST_ARGS.link_tags
        link_tags = _link_tags if _link_tags is not None else self._ModelInfo.get_current_link_tags()
        if link_tags is not None:
            link_tags = [int(tag) for tag in np.atleast_1d(link_tags)]
        if len(link_tags) > 0 and _save_link_resp:
            if self._LinkResp is None:
                self._LinkResp = LinkRespStepData(link_tags, **self.resp_kargs)
            else:
                self._LinkResp.add_resp_data_one_step(link_tags)

    def _set_shell_resp(self):
        _save_shell_resp = self._POST_ARGS.save_shell_resp
        _shell_tags = self._POST_ARGS.shell_tags
        shell_tags = _shell_tags if _shell_tags is not None else self._ModelInfo.get_current_shell_tags()
        if shell_tags is not None:
            shell_tags = [int(tag) for tag in np.atleast_1d(shell_tags)]
        if len(shell_tags) > 0 and _save_shell_resp:
            if self._ShellResp is None:
                self._ShellResp = ShellRespStepData(
                    shell_tags, compute_nodal_resp=self._POST_ARGS.project_gauss_to_nodes, **self.resp_kargs
                )
            else:
                self._ShellResp.add_resp_data_one_step(shell_tags)

    def _set_fiber_sec_resp(self):
        _save_fiber_sec_resp = self._POST_ARGS.save_fiber_sec_resp
        _fiber_ele_tags = self._POST_ARGS.fiber_ele_tags
        if _fiber_ele_tags is not None:
            if not isinstance(_fiber_ele_tags, str):
                _fiber_ele_tags = [int(tag) for tag in np.atleast_1d(_fiber_ele_tags)]
            else:
                if _fiber_ele_tags.lower() != "all":
                    _fiber_ele_tags = None
        if _fiber_ele_tags is not None:
            _save_fiber_sec_resp = True

        if _fiber_ele_tags is not None and _save_fiber_sec_resp:
            if self._FiberSecResp is None:
                self._FiberSecResp = FiberSecRespStepData(_fiber_ele_tags, **self.resp_kargs)
            else:
                self._FiberSecResp.add_resp_data_one_step()

    def _set_plane_resp(self):
        _save_plane_resp = self._POST_ARGS.save_plane_resp
        _plane_tags = self._POST_ARGS.plane_tags
        plane_tags = _plane_tags if _plane_tags is not None else self._ModelInfo.get_current_plane_tags()
        if plane_tags is not None:
            plane_tags = [int(tag) for tag in np.atleast_1d(plane_tags)]

        if len(plane_tags) > 0 and _save_plane_resp:
            if self._PlaneResp is None:
                self._PlaneResp = PlaneRespStepData(
                    plane_tags,
                    compute_measures=self._POST_ARGS.compute_mechanical_measures,
                    compute_nodal_resp=self._POST_ARGS.project_gauss_to_nodes,
                    **self.resp_kargs,
                )
            else:
                self._PlaneResp.add_resp_data_one_step(plane_tags)

    def _set_brick_resp(self):
        _save_brick_resp = self._POST_ARGS.save_brick_resp
        _brick_tags = self._POST_ARGS.brick_tags
        brick_tags = _brick_tags if _brick_tags is not None else self._ModelInfo.get_current_brick_tags()
        if brick_tags is not None:
            brick_tags = [int(tag) for tag in np.atleast_1d(brick_tags)]
        if len(brick_tags) > 0 and _save_brick_resp:
            if self._BrickResp is None:
                self._BrickResp = BrickRespStepData(
                    brick_tags,
                    compute_measures=self._POST_ARGS.compute_mechanical_measures,
                    compute_nodal_resp=self._POST_ARGS.project_gauss_to_nodes,
                    **self.resp_kargs,
                )
            else:
                self._BrickResp.add_resp_data_one_step(brick_tags)

    def _set_contact_resp(self):
        _save_contact_resp = self._POST_ARGS.save_contact_resp
        _contact_tags = self._POST_ARGS.contact_tags
        contact_tags = _contact_tags if _contact_tags is not None else self._ModelInfo.get_current_contact_tags()
        if contact_tags is not None:
            contact_tags = [int(tag) for tag in np.atleast_1d(contact_tags)]
        if len(contact_tags) > 0 and _save_contact_resp:
            if self._ContactResp is None:
                self._ContactResp = ContactRespStepData(contact_tags, **self.resp_kargs)
            else:
                self._ContactResp.add_resp_data_one_step(contact_tags)

    def _set_sensitivity_resp(self):
        _save_sensitivity_resp = self._POST_ARGS.save_sensitivity_resp
        _sensitivity_para_tags = self._POST_ARGS.sensitivity_para_tags
        sens_para_tags = _sensitivity_para_tags if _sensitivity_para_tags is not None else ops.getParamTags()

        _node_tags = self._POST_ARGS.node_tags
        node_tags = _node_tags if _node_tags is not None else self._ModelInfo.get_current_node_tags()
        if node_tags is not None:
            node_tags = [int(tag) for tag in np.atleast_1d(node_tags)]

        if len(node_tags) > 0 and len(sens_para_tags) > 0 and _save_sensitivity_resp:
            if self._SensitivityResp is None:
                self._SensitivityResp = SensitivityRespStepData(
                    node_tags=node_tags, ele_tags=None, sens_para_tags=sens_para_tags, **self.resp_kargs
                )
            else:
                self._SensitivityResp.add_resp_data_one_step(node_tags=node_tags, sens_para_tags=sens_para_tags)

    def reset(self):
        """Reset the ODB model."""
        for resp in self._get_resp():
            if resp is not None:
                resp.reset()

    def fetch_response_step(self, print_info: bool = False):
        """Extract response data for the current analysis step.

        Parameters
        ------------
        print_info: bool, optional
            print information, by default, False
        """

        self._set_resp()

        self._pending_steps += 1
        if self._flush_every is not None and self._pending_steps >= int(self._flush_every):
            self._flush_response_data()
            for resp in self._get_resp():
                if resp is not None:
                    resp.reset_resp_step_data()

        if print_info:
            CONSOLE = CONFIGS.get_console()
            PKG_PREFIX = CONFIGS.get_pkg_prefix()
            time = ops.getTime()
            color = get_random_color()
            CONSOLE.print(f"{PKG_PREFIX} The responses data at time [bold {color}]{time:.4f}[/] has been fetched!")

    def combine_response_spectrum(
        self,
        method: str = "srss",
        lambdas: list[float] | tuple[float, ...] | np.ndarray | None = None,
        damping: float | list[float] | tuple[float, ...] | np.ndarray = 0.05,
        scale: list[float] | tuple[float, ...] | np.ndarray | int | float = 1.0,
    ):
        """Combine modal responses data, only for response spectrum analysis.

        ..  note::
            This method only works when ``save_every=None`` in :py:func:`opstool.post.CreateODB`.

        Parameters
        ----------
        method : {"srss", "cqc"}
            Combination method.
        lambdas : array-like, optional
            Modal frequencies. Required for CQC.
        damping : float or array-like
            Modal damping ratios. Optional for CQC.
        scale : array-like or None
            Modal scaling factors. Optional for CQC.
        """
        from ._combine_response_spectrum import combine_response_spectrum

        if self._flush_every is not None:
            raise RuntimeError("combine_response_spectrum() only works when save_every=None in CreateODB!")  # noqa: TRY003

        for resp in self._get_resp()[1:]:  # Skip ModelInfo
            if resp is not None:
                resp_dataset = resp.resp_step_data
                exclude_vars = ["ys", "zs", "areas", "matTags", "sectionLocs", "lambdas"]
                rcombined = combine_response_spectrum(
                    resp_dataset,
                    method=method,
                    lambdas=lambdas,
                    damping=damping,
                    scale=scale,
                    time_dim="time",
                    exclude_vars=exclude_vars,
                )
                resp.resp_step_data = rcombined

    def _flush_response_data(self):
        """Flush pending response data to the ODB file."""
        if self._odb_format.lower() == "zarr":
            filename = self._store_path + f"/part_{self._file_idx}.zarr"
            self._save_response_zarr(filename)
        elif self._odb_format.lower() == "nc":
            filename = self._store_path + f"/part_{self._file_idx}.nc"
            self._save_response_nc(filename, zlib=self._zlib)

        self._file_idx += 1
        self._pending_steps = 0

    def save_response(self, zlib: bool | None = None):
        """
        Save all response data.
        """
        if zlib is not None:
            warnings.warn(
                "The zlib parameter in save_response() is deprecated. Please set zlib in CreateODB().",
                DeprecationWarning,
                stacklevel=2,
            )

        if self._pending_steps > 0:
            self._flush_response_data()

        # Print information
        CONSOLE = CONFIGS.get_console()
        PKG_PREFIX = CONFIGS.get_pkg_prefix()
        color = get_random_color()
        CONSOLE.print(
            f"{PKG_PREFIX} All responses data with _odb_tag = {self._odb_tag} saved in [bold {color}]{self._store_path}[/]!"
        )

    def _save_response_zarr(self, filename):
        """Save response data to a Zarr file."""
        with xr.DataTree() as dt:
            for resp in self._get_resp():
                if resp is not None:
                    resp.add_resp_data_to_datatree(dt)
            # Generate encoding
            encoding = generate_chunk_encoding_for_datatree(dt, target_chunk_mb=20.0)
            max_retries = 5
            retry_delay = 1

            for attempt in range(max_retries + 1):
                try:
                    # try to remove existing directory before writing
                    if attempt > 0 and os.path.exists(filename):
                        shutil.rmtree(filename)
                        # Windows may take some time to release file locks
                        time.sleep(0.5)

                    dt.to_zarr(filename, mode="w", consolidated=True, encoding=encoding, zarr_format=2)
                    break

                except PermissionError:
                    if attempt < max_retries:
                        # Wait and retry
                        time.sleep(retry_delay)
                        retry_delay *= 1.5
                    else:
                        raise

            dt.close()
            del dt

    def _save_response_nc(self, filename, zlib=False):
        """Save response data to a NetCDF file."""
        with xr.DataTree() as dt:
            for resp in self._get_resp():
                if resp is not None:
                    resp.add_resp_data_to_datatree(dt)

            if zlib:
                encoding = {}
                for path, node in dt.items():
                    if path == "ModelInfo":
                        for key, _value in node.items():
                            encoding[f"/{path}/{key}"] = {
                                key: {"_FillValue": -9999, "zlib": True, "complevel": 5, "dtype": "float32"}
                            }
                    else:
                        for key, _value in node.items():
                            encoding[f"/{path}"] = {
                                key: {"_FillValue": -9999, "zlib": True, "complevel": 5, "dtype": "float32"}
                            }
            else:
                encoding = None

            max_retries = 5
            retry_delay = 1

            for attempt in range(max_retries + 1):
                try:
                    # try to remove existing file before writing
                    if attempt > 0 and os.path.exists(filename):
                        os.remove(filename)
                        # Windows may take some time to release file locks
                        time.sleep(0.5)

                    dt.to_netcdf(filename, mode="w", engine="netcdf4", encoding=encoding)
                    break

                except PermissionError:
                    if attempt < max_retries:
                        # Wait and retry
                        time.sleep(retry_delay)
                        retry_delay *= 1.5
                    else:
                        raise

            dt.close()
            del dt

    def save_eigen_data(self, mode_tag: int = 1, solver: str = "-genBandArpack"):
        """Save modal analysis data.

        Parameters
        ----------
        mode_tag : int, optional,
            Modal tag, all modal data smaller than this modal tag will be saved, by default 1
        solver : str, optional,
           OpenSees' eigenvalue analysis solver, by default "-genBandArpack".
        """
        save_eigen_data(odb_tag=self._odb_tag, mode_tag=mode_tag, solver=solver)

    def save_model_data(self):
        """Save the model data from the current domain."""
        save_model_data(odb_tag=self._odb_tag)


def _load_parts_as_datatree(path) -> list[xr.DataTree]:
    """
    Read part_i DataTree stores under `folder` and return a list of DataTree(s).
    Supported:
      - part_0            (zarr store directory)
      - part_0.zarr       (zarr store directory)
      - part_0.nc         (netcdf file)
    Notes:
      - All parts are assumed to be the same format (all zarr dirs OR all .nc files).
      - Returns a list with 1 or more elements.
    """
    import re

    folder = Path(path)
    pat = re.compile(r"^part_(\d+)(?:\.nc|\.zarr)?$")

    parts = sorted(
        (
            (int(m.group(1)), p)
            for p in folder.iterdir()
            for m in [pat.match(p.name)]
            if m and (p.is_dir() or p.suffix.lower() == ".nc")
        ),
        key=lambda t: t[0],
    )
    if not parts:
        raise FileNotFoundError(f"No parts found in {folder} (expected part_i(.nc/.zarr)).")  # noqa: TRY003

    # uniform type (zarr dirs vs .nc files)
    is_zarr = parts[0][1].is_dir()
    if any(p.is_dir() != is_zarr for _, p in parts):
        raise ValueError(f"Mixed part types found in {folder}: zarr dirs and nc files.")  # noqa: TRY003

    engine = "zarr" if is_zarr else "netcdf4"
    open_kwargs = {"mask_and_scale": True}
    open_kwargs.update({"consolidated": True} if is_zarr else {})

    return [xr.open_datatree(p, engine=engine, **open_kwargs) for _, p in parts]


def loadODB(
    odb_tag,
    resp_type: Literal[
        "Nodal",
        "Frame",
        "FiberSec",
        "FiberSection",
        "Truss",
        "Link",
        "Shell",
        "Plane",
        "Brick",
        "Solid",
        "Contact",
        "Sensitivity",
    ] = "Nodal",
    lazy_load: bool = True,
    verbose: bool = True,
):
    """Load saved response data.

    Returns
    --------
    Relevant to a response type.
    """
    RESULTS_DIR = CONFIGS.get_output_dir()
    CONSOLE = CONFIGS.get_console()
    PKG_PREFIX = CONFIGS.get_pkg_prefix()
    RESP_FILE_NAME = CONFIGS.get_resp_filename()

    store_path = str(Path(RESULTS_DIR) / f"{RESP_FILE_NAME}-{odb_tag}.odb")
    dts = _load_parts_as_datatree(store_path)
    if verbose:
        color = get_random_color()
        CONSOLE.print(f"{PKG_PREFIX} Loading response data from [bold {color}]{store_path}[/] ...")
    model_info_steps, model_update = ModelInfoStepData.read_response(
        dts, unit_factors=_UNIT_SYSTEM.unit_factors, lazy=lazy_load
    )
    resp_type_lower = resp_type.lower()
    if resp_type_lower == "nodal":
        resp_step = NodalRespStepData.read_response(dts, unit_factors=_UNIT_SYSTEM.unit_factors, lazy=lazy_load)
    elif resp_type_lower == "sensitivity":
        resp_step = SensitivityRespStepData.read_response(dts, lazy=lazy_load)
    elif resp_type_lower in _ELE_RESP_READERS:
        resp_step = _ELE_RESP_READERS[resp_type_lower].read_response(
            dts, unit_factors=_UNIT_SYSTEM.unit_factors, lazy=lazy_load
        )
    else:
        raise ValueError(f"Unsupported response type {resp_type}!")  # noqa: TRY003

    return model_info_steps, model_update, resp_step


def get_model_data(
    odb_tag: int | str | None = None,
    data_type: Literal[
        "Nodal",
        "Frame",
        "Beam",
        "Link",
        "Truss",
        "Shell",
        "Plane",
        "Brick",
        "Solid",
        "Contact",
        "FixedNode",
        "NodalLoad",
        "FrameLoad",
        "MPConstraint",
    ]
    | None = "Nodal",
    from_responses: bool = False,
    lazy_load: bool = False,
    print_info: bool = True,
):
    """Read model data from a file.

    Parameters
    ----------
    odb_tag: Union[int, str], default: one
        Tag of output databases (ODB) to be read.
    data_type: Literal["Nodal", "Frame", "Beam", "Link", "Truss", "Shell", "Plane", "Brick", "Solid", "Contact", "FixedNode", "NodalLoad", "FrameLoad", "MPConstraint"], default: Nodal
        Type of data to be read.
        Optional: "Nodal", "Frame", "Link", "Truss", "Shell", "Plane", "Brick", "Contact", "FixedNode", "NodalLoad", "FrameLoad", "MPConstraint".

        ... Note::
            For element data, the cells represent the index of the nodes in "Nodal" data.
            You can use the ``.isel`` method of xarray to select node information by cell index.

    from_responses: bool, default: False
        Whether to read data from response data.
        If True, the data will be read from the response data file.
        This is useful when the model data is updated in an analysis process.

    lazy_load: bool, default: False, added since version 1.0.25.
        Whether to lazy load the data.
        If True, the data will be loaded on demand, which can save memory for large datasets.
        If False, the data will be fully loaded into memory.
        If you have enough memory, it is recommended to set this parameter to False for faster and safer data access.

    print_info: bool, default: True
        Whether to print information.

    Returns
    ---------
    ModelData: xarray.Dataset if model_update is True, otherwise xarray.DataArray
    """
    RESULTS_DIR = CONFIGS.get_output_dir()
    CONSOLE = CONFIGS.get_console()
    PKG_PREFIX = CONFIGS.get_pkg_prefix()
    RESP_FILE_NAME = CONFIGS.get_resp_filename()
    MODEL_FILE_NAME = CONFIGS.get_model_filename()

    if isinstance(data_type, str):
        if data_type.lower() == "nodal":
            data_type = "NodalData"
        elif data_type.lower() in ["frame", "beam"]:
            data_type = "BeamData"
        elif data_type.lower() == "link":
            data_type = "LinkData"
        elif data_type.lower() == "truss":
            data_type = "TrussData"
        elif data_type.lower() == "shell":
            data_type = "ShellData"
        elif data_type.lower() == "plane":
            data_type = "PlaneData"
        elif data_type.lower() in ["brick", "solid"]:
            data_type = "BrickData"
        elif data_type.lower() == "contact":
            data_type = "ContactData"
        elif data_type.lower() == "fixednode":
            data_type = "FixedNodalData"
        elif data_type.lower() == "nodalload":
            data_type = "NodalLoadData"
        elif data_type.lower() in ["frameload", "beamload"]:
            data_type = "FrameLoadData"
        elif data_type.lower() == "mpconstraint":
            data_type = "MPConstraintData"
        else:
            raise ValueError(f"Data type {data_type} not found.")  # noqa: TRY003

    if from_responses:
        filename = str(Path(RESULTS_DIR) / f"{RESP_FILE_NAME}-{odb_tag}.odb")
        dts = _load_parts_as_datatree(filename)
        data = ModelInfoStepData.read_data(dts, data_type, lazy=lazy_load)
    else:
        suffix, engine = CONFIGS.get_odb_format()
        kargs = {"consolidated": False} if suffix == "zarr" else {}
        filename = str(Path(RESULTS_DIR) / f"{MODEL_FILE_NAME}-{odb_tag}.{suffix}")
        with xr.open_datatree(filename, engine=engine, **kargs).load() as dt:
            if data_type not in dt["ModelInfo"]:
                raise ValueError(f"Data type {data_type} not found in model data.")  # noqa: TRY003
            data = dt["ModelInfo"][data_type][data_type]
            dt.close()
    if print_info:
        color = get_random_color()
        CONSOLE.print(f"{PKG_PREFIX} Loading {data_type} data from [bold {color}]{filename}[/] ...")
    return data


def get_nodal_responses(
    odb_tag: int | str,
    resp_type: Literal["disp", "vel", "accel", "reaction", "reactionIncInertia", "rayleighForces", "pressure"]
    | None = None,
    node_tags: list[int] | tuple[int, ...] | int | None = None,
    lazy_load: bool = False,
    print_info: bool = True,
) -> xr.Dataset:
    """Read nodal responses data from a file.

    .. important::
        You can use :func:`opstool.post.get_nodal_responses_info` to get valid response types and DOFs.

    Parameters
    ----------
    odb_tag: Union[int, str], default: one
        Tag of output databases (ODB) to be read.
    resp_type: str, default: None
        Type of response to be read.
        Optional:

        * "disp" - Displacement at the node.
        * "vel" - Velocity at the node.
        * "accel" - Acceleration at the node.
        * "reaction" - Reaction forces at the node.
        * "reactionIncInertia" - Reaction forces including inertial effects.
        * "rayleighForces" - Forces resulting from Rayleigh damping.
        * "pressure" - Pressure applied to the node.
        * If None, return all responses.

        .. Note::
            If the nodes include fluid pressure dof,
            such as those used for ``**UP`` elements, the pore pressure should be extracted using ``resp_type="vel"``,
            and the value is placed in the degree of freedom ``RZ``.

    node_tags: Union[list, tuple, int], default: None
        Node tags to be read.
        Such as [1, 2, 3] or numpy.array([1, 2, 3]) or 1.
        If None, return all nodal responses.

        .. Note::
            If some nodes are deleted during the analysis,
            their response data will be filled with `numpy.nan`.

    lazy_load: bool, default: False, added since version 1.0.25.
        Whether to lazy load the data.
        If True, the data will be loaded on demand, which can save memory for large datasets.
        If False, the data will be fully loaded into memory.
        If you have enough memory, it is recommended to set this parameter to False for faster and safer data access.

    print_info: bool, default: True
        Whether to print information

    Returns
    ---------
    NodalResp: `xarray.Dataset <https://docs.xarray.dev/en/stable/generated/xarray.Dataset.html>`_
        Nodal responses' data.

    .. note::
        The returned data can be viewed using ".data_vars,” `.dims`, `.coords`, and `.attrs` to view the
        dimension names and coordinates.
        You can further index or process the data.

    """
    RESULTS_DIR = CONFIGS.get_output_dir()
    CONSOLE = CONFIGS.get_console()
    PKG_PREFIX = CONFIGS.get_pkg_prefix()
    RESP_FILE_NAME = CONFIGS.get_resp_filename()

    store_path = str(Path(RESULTS_DIR) / f"{RESP_FILE_NAME}-{odb_tag}.odb")
    dts = _load_parts_as_datatree(store_path)

    nodal_resp = NodalRespStepData.read_response(
        dts, resp_type=resp_type, node_tags=node_tags, unit_factors=_UNIT_SYSTEM.unit_factors, lazy=lazy_load
    )

    if print_info:
        color = get_random_color()
        if resp_type is None:
            CONSOLE.print(f"{PKG_PREFIX} Loading all response data from [bold {color}]{store_path}[/] ...")
        else:
            CONSOLE.print(f"{PKG_PREFIX} Loading {resp_type} response data from [bold {color}]{store_path}[/] ...")
    return nodal_resp


def get_element_responses(
    odb_tag: int | str,
    ele_type: Literal["Frame", "FiberSection", "FiberSec", "Truss", "Link", "Shell", "Plane", "Solid", "Contact"],
    resp_type: str | None = None,
    ele_tags: list[int] | tuple[int, ...] | int | None = None,
    lazy_load: bool = False,
    print_info: bool = True,
) -> xr.Dataset:
    """Read nodal responses data from a file.

    .. important::
        You can use :func:`opstool.post.get_element_responses_info` to get valid response types and DOFs for each element type.

    Parameters
    ------------
    odb_tag: Union[int, str], default: one
        Tag of output databases (ODB) to be read.
    ele_type: str, default: Frame
        Type of element to be read.
        Optional: "Frame", "FiberSection", "Truss", "Link", "Shell", "Plane", "Solid", "Contact
    resp_type: str, default: disp
        The response type, which depends on the parameter `ele_type`.
        If None, return all responses to that `ele_type`.

        #. For `Frame`:
            * "localForces": Local forces in the element local coordinate system.
            * "basicForces": Basic forces in the element basic coordinate system.
            * "basicDeformations": Basic deformations in the element basic coordinate system.
            * "plasticDeformation": Plastic deformations in the element basic coordinate system.
            * "sectionForces": Section forces in the element coordinate system.
            * "sectionDeformations": Section deformations in the element coordinate system.
            * "sectionLocs": Section locations, 0.0 to 1.0.
        #. For `FiberSection`:
            * "Stresses": Stress.
            * "Strains": Strain.
            * "ys": y coords.
            * "zs": z coords.
            * "areas": Fiber point areas.
            * "matTags": Mat tags in OpenSees.
            * "secDefo": Section deformations.
            * "secForce": Section forces.
        #. For `Truss`:
            * "axialForce": Axial force.
            * "axialDefo": Axial deformation.
            * "Stress": Stress of material.
            * "Strain": Strain of material.
        #. For `Link`:
            * "basicDeformation": Basic deformation, i.e., pure deformation.
            * "basicForce": Basic force.
        #. For `Shell`:
            * "sectionForces": Sectional forces at Gauss points (per unit length).
            * "sectionDeformations": Sectional deformation at Gauss points (per unit length).
            * "Stresses": The stresses of each fiber layer at each Gauss point.
            * "Strains": The strains of each fiber layer at each Gauss point.
        #. For `Plane`:
            * "stresses": Stresses at Gauss points.
            * "strains": Strains at Gauss points.
        #. For `Brick` or 'Solid':
            * "stresses": Stresses at Gauss points.
            * "strains": Strains at Gauss points.
        #. For `Contact`:
            * "localForces": Local forces in the element local coordinate system (normal and tangential).
            * "localDisp": Local displacements in the element local coordinate system (normal and tangential).
            * "slips": Slips in the element local coordinate system (tangential).

    ele_tags: Union[list, tuple, int], default: None
        Element tags to be read.
        Such as [1, 2, 3] or numpy.array([1, 2, 3]) or 1.
        If None, return all nodal responses.

        .. note::
            If some elements are deleted during the analysis,
            their response data will be filled with `numpy.nan`.

    lazy_load: bool, default: False, added since version 1.0.25.
        Whether to lazy load the data.
        If True, the data will be loaded on demand, which can save memory for large datasets.
        If False, the data will be fully loaded into memory.
        If you have enough memory, it is recommended to set this parameter to False for faster and safer data access.

    print_info: bool, default: True
        Whether to print information.

    Returns
    ---------
    EleResp: `xarray.Dataset <https://docs.xarray.dev/en/stable/generated/xarray.Dataset.html>`_
        Element responses' data.

    .. note::
        The returned data can be viewed using ".data_vars,” `.dims`, `.coords`, and `.attrs` to view the
        dimension names and coordinates.
        You can further index or process the data.
    """
    RESULTS_DIR = CONFIGS.get_output_dir()
    CONSOLE = CONFIGS.get_console()
    PKG_PREFIX = CONFIGS.get_pkg_prefix()
    RESP_FILE_NAME = CONFIGS.get_resp_filename()

    store_path = str(Path(RESULTS_DIR) / f"{RESP_FILE_NAME}-{odb_tag}.odb")
    dts = _load_parts_as_datatree(store_path)

    ele_type_l = ele_type.lower()

    if ele_type_l in _ELE_RESP_READERS:
        reader_cls = _ELE_RESP_READERS[ele_type_l]
    else:
        raise ValueError(  # noqa: TRY003
            f"Unsupported element type {ele_type}, "
            "must be in [Frame, FiberSection, Truss, Link, Shell, Plane, Solid, Contact]!"
        )
    ele_resp = reader_cls.read_response(
        dts, resp_type=resp_type, ele_tags=ele_tags, unit_factors=_UNIT_SYSTEM.unit_factors, lazy=lazy_load
    )

    if print_info:
        color = get_random_color()
        if resp_type is None:
            CONSOLE.print(f"{PKG_PREFIX} Loading {ele_type} response data from [bold {color}]{store_path}[/] ...")
        else:
            CONSOLE.print(
                f"{PKG_PREFIX} Loading {ele_type} {resp_type} response data from [bold {color}]{store_path}[/] ..."
            )

    return ele_resp


def get_sensitivity_responses(
    odb_tag: int | str,
    resp_type: Literal["disp", "vel", "accel", "pressure", "lambda"] | None = None,
    print_info: bool = True,
    lazy_load: bool = False,
) -> xr.Dataset:
    """Read sensitivity responses data from a file.

    Parameters
    ------------
    odb_tag: Union[int, str], default: one
        Tag of output databases (ODB) to be read.
    resp_type: str, default: None
        Type of response to be read.
        Optional:

        * "disp" - Displacement at the node.
        * "vel" - Velocity at the node.
        * "accel" - Acceleration at the node.
        * "pressure" - Pressure applied to the node.
        * "lambda" - Multiplier in load patterns.
        * If None, return all responses.

    print_info: bool, default: True
        Whether to print information.
    lazy_load: bool, default: False, added since version 1.0.25.
        Whether to lazy load the data.
        If True, the data will be loaded on demand, which can save memory for large datasets.
        If False, the data will be fully loaded into memory.
        If you have enough memory, it is recommended to set this parameter to False for faster and safer data access.

    Returns
    ---------
    SensResp: `xarray.Dataset <https://docs.xarray.dev/en/stable/generated/xarray.Dataset.html>`_
        Sensitivity responses' data.
    """
    RESULTS_DIR = CONFIGS.get_output_dir()
    CONSOLE = CONFIGS.get_console()
    PKG_PREFIX = CONFIGS.get_pkg_prefix()
    RESP_FILE_NAME = CONFIGS.get_resp_filename()

    store_path = str(Path(RESULTS_DIR) / f"{RESP_FILE_NAME}-{odb_tag}.odb")
    dts = _load_parts_as_datatree(store_path)

    resp = SensitivityRespStepData.read_response(dts, resp_type=resp_type, lazy=lazy_load)

    if print_info:
        color = get_random_color()
        if resp_type is None:
            CONSOLE.print(f"{PKG_PREFIX} Loading response data from [bold {color}]{store_path}[/] ...")
        else:
            CONSOLE.print(f"{PKG_PREFIX} Loading {resp_type} response data from [bold {color}]{store_path}[/] ...")

    return resp


# -----------------------------------------------------------------------------------------------------
# Unit system for post-processing
# -----------------------------------------------------------------------------------------------------
def update_unit_system(
    pre: dict[str, str] | None = None,
    post: dict[str, str] | None = None,
):
    """Set the unit system will be used for post-processing.

    Parameters
    -----------
    pre: dict, default: None.
        Unit system used in pre-processing and analysis.

        Style: dict(force=force_unit, length=length_unit, time=time_unit)

        * force_unit: Optional ["lb"("lbf"), "kip", "n", "kn", "mn", "kgf", "tonf"].
        * length_unit: Optional ["inch", "ft", "mm", "cm", "m", "km"].
        * time_unit: Optional ["sec"].

    post: dict, default: None.
        Unit system will be used for post-processing.

        Style: dict(force=force_unit, length=length_unit, time=time_unit)

        When ``pre`` and ``post`` are specified together,
        the response data will be transformed from the ``pre`` unit system to the ``post`` unit system.
        This will affect its numerical size.

    Returns
    --------
    None
    """
    unit_factors, unit_syms = _parse_unit_factors(analysis_unit_system=pre, post_unit_system=post)
    _UNIT_SYSTEM.unit_factors = unit_factors
    _UNIT_SYSTEM.unit_symbols = unit_syms


def reset_unit_system():
    """Reset unit system for post-processing."""
    _UNIT_SYSTEM.unit_factors = None
    _UNIT_SYSTEM.unit_symbols = None


def _parse_unit_factors(analysis_unit_system, post_unit_system):
    if analysis_unit_system is None or post_unit_system is None:
        unit_factors = None
        unit_syms = None
    else:
        if not isinstance(analysis_unit_system, dict):
            raise ValueError("analysis_unit_system must be of type dict!")  # noqa: TRY003
        if not isinstance(post_unit_system, dict):
            raise ValueError("post_unit_system must be of type dict!")  # noqa: TRY003
        for key in analysis_unit_system:
            if key not in ["length", "force", "time"]:
                raise ValueError("key must be one of [length, force, time]!")  # noqa: TRY003
        for key in post_unit_system:
            if key not in ["length", "force", "time"]:
                raise ValueError("key must be one of [length, force, time]!")  # noqa: TRY003

        analysis_units_ = {"force": None, "length": None, "time": None}
        analysis_units_.update(analysis_unit_system)
        post_units_ = {"force": None, "length": None, "time": None}
        post_units_.update(post_unit_system)
        unit_factors = get_post_unit_multiplier(
            analysis_length=analysis_units_["length"],
            analysis_force=analysis_units_["force"],
            analysis_time=analysis_units_["time"],
            post_length=post_units_["length"],
            post_force=post_units_["force"],
            post_time=post_units_["time"],
        )
        unit_syms = get_post_unit_symbol(
            analysis_length=analysis_units_["length"],
            analysis_force=analysis_units_["force"],
            analysis_time=analysis_units_["time"],
            post_length=post_units_["length"],
            post_force=post_units_["force"],
            post_time=post_units_["time"],
        )
    return unit_factors, unit_syms
