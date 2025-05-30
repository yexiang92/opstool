from typing import Union
from types import SimpleNamespace

import numpy as np
import openseespy.opensees as ops
import xarray as xr

from ._get_response import (
    ModelInfoStepData,
    NodalRespStepData,
    TrussRespStepData,
    FrameRespStepData,
    LinkRespStepData,
    FiberSecRespStepData,
    ShellRespStepData,
    PlaneRespStepData,
    BrickRespStepData,
    ContactRespStepData,
    SensitivityRespStepData
)
from .eigen_data import save_eigen_data
from .model_data import save_model_data
from ._unit_postprocess import get_post_unit_multiplier, get_post_unit_symbol

from ..utils import get_random_color, CONSTANTS, ConfigUpdate

# RESULTS_DIR = CONSTANTS.get_output_dir()
# CONSOLE = CONSTANTS.get_console()
# PKG_PREFIX = CONSTANTS.get_pkg_prefix()
# RESP_FILE_NAME = CONSTANTS.get_resp_filename()
# MODEL_FILE_NAME = CONSTANTS.get_model_filename()

POST_ARGS = SimpleNamespace(
    elastic_frame_sec_points=7,
    compute_mechanical_measures=True,
    dtype=dict(int=np.int32, float=np.float32),
    # ------------------------------
    save_nodal_resp=True,
    save_frame_resp=True,
    save_truss_resp=True,
    save_link_resp=True,
    save_shell_resp=True,
    save_fiber_sec_resp=True,
    save_plane_resp=True,
    save_brick_resp=True,
    save_contact_resp=True,
    save_sensitivity_resp=False,
    # ----------------------------------
    node_tags=None,
    frame_tags=None,
    truss_tags=None,
    link_tags=None,
    shell_tags=None,
    fiber_ele_tags=None,
    plane_tags=None,
    brick_tags=None,
    contact_tags=None,
    sensitivity_para_tags=None,
    # -----------------------------------
    unit_factors = None,
    unit_symbols = None
)


class CreateODB:
    """Create an output database (ODB) to save response data.

    Parameters
    ------------
    odb_tag: Union[int, str], default: 1
        Tag of output databases (ODB) to be saved.
    model_update: bool, default: False
        Whether to update the model data.

        .. Note::
            If True, the model data will be updated at each step.
            If no nodes and elements are added or removed during the analysis of your model,
            keep this parameter set to **False**.
            Enabling model updates unnecessarily can increase memory usage and slow down performance.
            If some nodes or elements are deleted during the analysis, you should set this parameter to `True`.
    kwargs: Other post-processing parameters, optional:
        * elastic_frame_sec_points: int, default: 7
            The number of elastic frame elements section points.
            A larger number may result in a larger file size.
        * compute_mechanical_measures: bool, default: True
            Whether to compute mechanical measures for ``solid and planar elements``,
            including principal stresses, principal strains, von Mises stresses, etc.
        * dtype: dict, default: dict(int=np.int32, float=np.float32)
            Set integer and floating point precision types.
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
            odb_tag: Union[int, str] = 1,
            model_update: bool = False,
            **kwargs
    ):
        self._odb_tag = odb_tag
        self._model_update = model_update

        for key, value in kwargs.items():
            if key not in list(vars(POST_ARGS).keys()):
                raise KeyError(f"Incorrect parameter {key}, should be one of {list(vars(POST_ARGS).keys())}!")
            else:
                setattr(POST_ARGS, key, value)

        self._save_nodal_resp = POST_ARGS.save_nodal_resp
        self._save_frame_resp = POST_ARGS.save_frame_resp
        self._save_truss_resp = POST_ARGS.save_truss_resp
        self._save_link_resp = POST_ARGS.save_link_resp
        self._save_shell_resp = POST_ARGS.save_shell_resp
        self._save_fiber_sec_resp = POST_ARGS.save_fiber_sec_resp
        self._save_plane_resp = POST_ARGS.save_plane_resp
        self._save_brick_resp = POST_ARGS.save_brick_resp
        self._save_contact_resp = POST_ARGS.save_contact_resp
        self._save_sensitivity_resp = POST_ARGS.save_sensitivity_resp

        self._node_tags = POST_ARGS.node_tags
        self._frame_tags = POST_ARGS.frame_tags
        self._truss_tags = POST_ARGS.truss_tags
        self._link_tags = POST_ARGS.link_tags
        self._shell_tags = POST_ARGS.shell_tags
        self._fiber_ele_tags = POST_ARGS.fiber_ele_tags
        self._plane_tags = POST_ARGS.plane_tags
        self._brick_tags = POST_ARGS.brick_tags
        self._contact_tags = POST_ARGS.contact_tags
        self._sensitivity_para_tags = POST_ARGS.sensitivity_para_tags

        if self._node_tags is not None:
            self._node_tags = [int(tag) for tag in np.atleast_1d(self._node_tags)]
        if self._frame_tags is not None:
            self._frame_tags = [int(tag) for tag in np.atleast_1d(self._frame_tags)]
        if self._truss_tags is not None:
            self._truss_tags = [int(tag) for tag in np.atleast_1d(self._truss_tags)]
        if self._link_tags is not None:
            self._link_tags = [int(tag) for tag in np.atleast_1d(self._link_tags)]
        if self._shell_tags is not None:
            self._shell_tags = [int(tag) for tag in np.atleast_1d(self._shell_tags)]
        if self._fiber_ele_tags is not None:
            if not isinstance(self._fiber_ele_tags, str):
                self._fiber_ele_tags = [int(tag) for tag in np.atleast_1d(self._fiber_ele_tags)]
            else:
                if not self._fiber_ele_tags.lower() == "all":
                    self._fiber_ele_tags = None
        if self._plane_tags is not None:
            self._plane_tags = [int(tag) for tag in np.atleast_1d(self._plane_tags)]
        if self._brick_tags is not None:
            self._brick_tags = [int(tag) for tag in np.atleast_1d(self._brick_tags)]
        if self._contact_tags is not None:
            self._contact_tags = [int(tag) for tag in np.atleast_1d(self._contact_tags)]
        if self._sensitivity_para_tags is not None:
            self._sensitivity_para_tags = [int(tag) for tag in np.atleast_1d(self._sensitivity_para_tags)]

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
        output = [
            self._ModelInfo, self._NodalResp, self._FrameResp, self._TrussResp,
            self._LinkResp, self._ShellResp, self._FiberSecResp,
            self._PlaneResp, self._BrickResp, self._ContactResp, self._SensitivityResp
        ]
        return output

    def _set_model_info(self):
        if self._ModelInfo is None:
            self._ModelInfo = ModelInfoStepData(model_update=self._model_update)
        else:
            self._ModelInfo.add_data_one_step()

    def _set_node_resp(self):
        if self._node_tags is not None:
            node_tags = self._node_tags
        else:
            node_tags = self._ModelInfo.get_current_node_tags()
        if len(node_tags) > 0 and self._save_nodal_resp:
            if self._NodalResp is None:
                self._NodalResp = NodalRespStepData(
                    node_tags,
                    model_update=self._model_update,
                    dtype=POST_ARGS.dtype
                )
            else:
                self._NodalResp.add_data_one_step(node_tags)

    def _set_frame_resp(self):
        if self._frame_tags is not None:
            frame_tags = self._frame_tags
        else:
            frame_tags = self._ModelInfo.get_current_frame_tags()
        frame_load_data = self._ModelInfo.get_current_frame_load_data()
        if len(frame_tags) > 0 and self._save_frame_resp:
            if self._FrameResp is None:
                self._FrameResp = FrameRespStepData(
                    frame_tags,
                    frame_load_data,
                    elastic_frame_sec_points=POST_ARGS.elastic_frame_sec_points,
                    model_update=self._model_update,
                    dtype=POST_ARGS.dtype
                )
            else:
                self._FrameResp.add_data_one_step(frame_tags, frame_load_data)

    def _set_truss_resp(self):
        if self._truss_tags is not None:
            truss_tags = self._truss_tags
        else:
            truss_tags = self._ModelInfo.get_current_truss_tags()
        if len(truss_tags) > 0 and self._save_truss_resp:
            if self._TrussResp is None:
                self._TrussResp = TrussRespStepData(
                    truss_tags,
                    model_update=self._model_update,
                    dtype=POST_ARGS.dtype
                )
            else:
                self._TrussResp.add_data_one_step(truss_tags)

    def _set_link_resp(self):
        if self._link_tags is not None:
            link_tags = self._link_tags
        else:
            link_tags = self._ModelInfo.get_current_link_tags()
        if len(link_tags) > 0 and self._save_link_resp:
            if self._LinkResp is None:
                self._LinkResp = LinkRespStepData(
                    link_tags,
                    model_update=self._model_update,
                    dtype=POST_ARGS.dtype
                )
            else:
                self._LinkResp.add_data_one_step(link_tags)

    def _set_shell_resp(self):
        if self._shell_tags is not None:
            shell_tags = self._shell_tags
        else:
            shell_tags = self._ModelInfo.get_current_shell_tags()
        if len(shell_tags) > 0 and self._save_shell_resp:
            if self._ShellResp is None:
                self._ShellResp = ShellRespStepData(
                    shell_tags,
                    model_update=self._model_update,
                    dtype=POST_ARGS.dtype
                )
            else:
                self._ShellResp.add_data_one_step(shell_tags)

    def _set_fiber_sec_resp(self):
        if self._fiber_ele_tags is not None and self._save_fiber_sec_resp:
            if self._FiberSecResp is None:
                self._FiberSecResp = FiberSecRespStepData(
                    self._fiber_ele_tags,
                    dtype=POST_ARGS.dtype
                )
            else:
                self._FiberSecResp.add_data_one_step()

    def _set_plane_resp(self):
        if self._plane_tags is not None:
            plane_tags = self._plane_tags
        else:
            plane_tags = self._ModelInfo.get_current_plane_tags()
        if len(plane_tags) > 0 and self._save_plane_resp:
            if self._PlaneResp is None:
                self._PlaneResp = PlaneRespStepData(
                    plane_tags,
                    compute_measures=POST_ARGS.compute_mechanical_measures,
                    model_update=self._model_update,
                    dtype=POST_ARGS.dtype
                )
            else:
                self._PlaneResp.add_data_one_step(plane_tags)

    def _set_brick_resp(self):
        if self._brick_tags is not None:
            brick_tags = self._brick_tags
        else:
            brick_tags = self._ModelInfo.get_current_brick_tags()
        if len(brick_tags) > 0 and self._save_brick_resp:
            if self._BrickResp is None:
                self._BrickResp = BrickRespStepData(
                    brick_tags,
                    compute_measures=POST_ARGS.compute_mechanical_measures,
                    model_update=self._model_update,
                    dtype=POST_ARGS.dtype
                )
            else:
                self._BrickResp.add_data_one_step(brick_tags)

    def _set_contact_resp(self):
        if self._contact_tags is not None:
            contact_tags = self._contact_tags
        else:
            contact_tags = self._ModelInfo.get_current_contact_tags()
        if len(contact_tags) > 0 and self._save_contact_resp:
            if self._ContactResp is None:
                self._ContactResp = ContactRespStepData(
                    contact_tags,
                    model_update=self._model_update,
                    dtype=POST_ARGS.dtype
                )
            else:
                self._ContactResp.add_data_one_step(contact_tags)

    def _set_sensitivity_resp(self):
        if self._sensitivity_para_tags is not None:
            sens_para_tags = self._sensitivity_para_tags
        else:
            sens_para_tags = ops.getParamTags()
        if self._node_tags is not None:
            node_tags = self._node_tags
        else:
            node_tags = self._ModelInfo.get_current_node_tags()
        if len(node_tags) > 0 and len(sens_para_tags) > 0 and self._save_sensitivity_resp:
            if self._SensitivityResp is None:
                self._SensitivityResp = SensitivityRespStepData(
                    node_tags=node_tags,
                    ele_tags=None,
                    sens_para_tags=sens_para_tags,
                    model_update=self._model_update,
                    dtype=POST_ARGS.dtype
                )
            else:
                self._SensitivityResp.add_data_one_step(node_tags=node_tags, sens_para_tags=sens_para_tags)

    def reset(self):
        """Reset the ODB model.
        """
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
        model_config = ConfigUpdate()

        if print_info:
            time = ops.getTime()
            color = get_random_color()
            model_config.CONSOLE.print(
                f"{model_config.PKG_PREFIX} The responses data at time [bold {color}]{time:.4f}[/] has been fetched!"
            )

    def save_response(self, zlib: bool = False):
        """
        Save all response data to a file name ``RespStepData-{odb_tag}.nc``.

        Parameters
        -----------
        zlib: bool, optional, default: False
            If True, the data is saved compressed,
            which is useful when your result files are expected to be large,
            especially if model updating is turned on.
        """
        model_config = ConfigUpdate()
        filename = f"{model_config.RESULTS_DIR}/" + f"{model_config.RESP_FILE_NAME}-{self._odb_tag}.nc"
        with xr.DataTree(name=f"{model_config.RESP_FILE_NAME}") as dt:
            for resp in self._get_resp():
                if resp is not None:
                    resp.save_file(dt)

            if zlib:
                encoding = {}
                for path, node in dt.items():
                    if path == "ModelInfo":
                        for key, value in node.items():
                            encoding[f"/{path}/{key}"] = {
                                key: {
                                    "_FillValue": -9999,
                                    "zlib": True,
                                    "complevel": 5,
                                    "dtype": "float32"
                                }
                            }
                    else:
                        for key, value in node.items():
                            encoding[f"/{path}"] = {
                                key: {
                                    "_FillValue": -9999,
                                    "zlib": True,
                                    "complevel": 5,
                                    "dtype": "float32"
                                }
                            }
            else:
                encoding = None

            dt.to_netcdf(filename, mode="w", engine="netcdf4", encoding=encoding)

        color = get_random_color()
        model_config.CONSOLE.print(
            f"{model_config.PKG_PREFIX} All responses data with _odb_tag = {self._odb_tag} "
            f"saved in [bold {color}]{filename}[/]!"
        )

    def save_eigen_data(
            self,
            mode_tag: int = 1,
            solver: str = "-genBandArpack",
    ):
        """Save modal analysis data.

        Parameters
        ----------
        mode_tag : int, optional,
            Modal tag, all modal data smaller than this modal tag will be saved, by default 1
        solver : str, optional,
           OpenSees' eigenvalue analysis solver, by default "-genBandArpack".
        """
        save_eigen_data(odb_tag=self._odb_tag, mode_tag=mode_tag, solver=solver)

    def save_model_data(
            self,
    ):
        """Save the model data from the current domain."""
        save_model_data(odb_tag=self._odb_tag)


def loadODB(
        obd_tag,
        resp_type: str = "Nodal",
):
    """Load saved response data.

    Returns
    --------
    Relevant to a response type.
    """
    model_config = ConfigUpdate()
    filename = f"{model_config.RESULTS_DIR}/" + f"{model_config.RESP_FILE_NAME}-{obd_tag}.nc"
    with xr.open_datatree(filename, engine="netcdf4").load() as dt:
        color = get_random_color()
        model_config.CONSOLE.print(
            f"{model_config.PKG_PREFIX} Loading response data from [bold {color}]{filename}[/] ..."
        )
        model_info_steps, model_update = ModelInfoStepData.read_file(dt, unit_factors=POST_ARGS.unit_factors)
        if resp_type.lower() == "nodal":
            resp_step = NodalRespStepData.read_file(dt, unit_factors=POST_ARGS.unit_factors)
        elif resp_type.lower() == "frame":
            resp_step = FrameRespStepData.read_file(dt, unit_factors=POST_ARGS.unit_factors)
        elif resp_type.lower() == "fibersec":
            resp_step = FiberSecRespStepData.read_file(dt, unit_factors=POST_ARGS.unit_factors)
        elif resp_type.lower() == "truss":
            resp_step = TrussRespStepData.read_file(dt, unit_factors=POST_ARGS.unit_factors)
        elif resp_type.lower() == "link":
            resp_step = LinkRespStepData.read_file(dt, unit_factors=POST_ARGS.unit_factors)
        elif resp_type.lower() == "shell":
            resp_step = ShellRespStepData.read_file(dt, unit_factors=POST_ARGS.unit_factors)
        elif resp_type.lower() == "plane":
            resp_step = PlaneRespStepData.read_file(dt, unit_factors=POST_ARGS.unit_factors)
        elif resp_type.lower() in ["brick", "solid"]:
            resp_step = BrickRespStepData.read_file(dt, unit_factors=POST_ARGS.unit_factors)
        elif resp_type.lower() == "contact":
            resp_step = ContactRespStepData.read_file(dt, unit_factors=POST_ARGS.unit_factors)
        elif resp_type.lower() == "sensitivity":
            resp_step = SensitivityRespStepData.read_file(dt)
        else:
            raise ValueError(f"Unsupported response type {resp_type}!")

    return model_info_steps, model_update, resp_step


def get_model_data(
        odb_tag: Union[int, str] = None,
        data_type: str = "Nodal",
        from_responses: bool = False
):
    """Read model data from a file.

    Parameters
    ----------
    odb_tag: Union[int, str], default: one
        Tag of output databases (ODB) to be read.
    data_type: str, default: Nodal
        Type of data to be read.
        Optional: "Nodal", "Frame", "Link", "Truss", "Shell", "Plane", "Brick".
    from_responses: bool, default: False
        Whether to read data from response data.
        If True, the data will be read from the response data file.
        This is useful when the model data is updated in an analysis process.

    Returns
    ---------
    ModelData: xarray.Dataset if model_update is True, otherwise xarray.DataArray
    """
    model_config = ConfigUpdate()
    if data_type.lower() == "nodal":
        data_type = "NodalData"
    elif data_type.lower() in ["frame", "beam"]:
        data_type = "BeamData"
    elif data_type.lower() == "link":
        data_type = "LinkData"
    elif data_type.lower() == "truss":
        data_type = "TrussData"
    elif data_type.lower() == "link":
        data_type = "LinkData"
    elif data_type.lower() == "shell":
        data_type = "ShellData"
    elif data_type.lower() == "plane":
        data_type = "PlaneData"
    elif data_type.lower() in ["brick", "solid"]:
        data_type = "BrickData"
    else:
        raise ValueError(f"Data type {data_type} not found.")
    if from_responses:
        filename = f"{model_config.RESULTS_DIR}/" + f"{model_config.RESP_FILE_NAME}-{odb_tag}.nc"
        with xr.open_datatree(filename, engine="netcdf4").load() as dt:
            data = ModelInfoStepData.read_data(dt, data_type)
    else:
        filename = f"{model_config.RESULTS_DIR}/" + f"{model_config.MODEL_FILE_NAME}-{odb_tag}.nc"
        with xr.open_datatree(filename, engine="netcdf4").load() as dt:
            data = dt["ModelInfo"][data_type][data_type]
    color = get_random_color()
    model_config.CONSOLE.print(
        f"{model_config.PKG_PREFIX} Loading {data_type} data from [bold {color}]{filename}[/] ..."
    )
    return data


def get_nodal_responses(
        odb_tag: int,
        resp_type: str = None,
        node_tags: Union[list, tuple, int] = None,
        print_info: bool = True,
) -> xr.Dataset:
    """Read nodal responses data from a file.

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
            such as those used for ...UP elements, the pore pressure should be extracted using ``resp_type="vel"``,
            and the value is placed in the degree of freedom ``RZ``.

    node_tags: Union[list, tuple, int], default: None
        Node tags to be read.
        Such as [1, 2, 3] or numpy.array([1, 2, 3]) or 1.
        If None, return all nodal responses.

        .. Note::
            If some nodes are deleted during the analysis,
            their response data will be filled with `numpy.nan`.

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
    model_config = ConfigUpdate()
    filename = f"{model_config.RESULTS_DIR}/" + f"{model_config.RESP_FILE_NAME}-{odb_tag}.nc"
    with xr.open_datatree(filename, engine="netcdf4").load() as dt:
        if print_info:
            color = get_random_color()
            if resp_type is None:
                model_config.CONSOLE.print(
                    f"{model_config.PKG_PREFIX} Loading all response data from [bold {color}]{filename}[/] ..."
                )
            else:
                model_config.CONSOLE.print(
                    f"{model_config.PKG_PREFIX} Loading {resp_type} response data from [bold {color}]{filename}[/] ..."
                )

        nodal_resp = NodalRespStepData.read_response(
            dt,
            resp_type=resp_type,
            node_tags=node_tags,
            unit_factors=POST_ARGS.unit_factors
        )
    return nodal_resp


def get_element_responses(
        odb_tag: int,
        ele_type: str,
        resp_type: str = None,
        ele_tags: Union[list, tuple, int] = None,
        print_info: bool = True,
) -> xr.Dataset:
    """Read nodal responses data from a file.

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
    model_config = ConfigUpdate()
    filename = f"{model_config.RESULTS_DIR}/" + f"{model_config.RESP_FILE_NAME}-{odb_tag}.nc"
    with xr.open_datatree(filename, engine="netcdf4").load() as dt:
        if print_info:
            color = get_random_color()
            if resp_type is None:
                model_config.CONSOLE.print(
                    f"{model_config.PKG_PREFIX} Loading {ele_type} response data from [bold {color}]{filename}[/] ..."
                )
            else:
                model_config.CONSOLE.print(
                    f"{model_config.PKG_PREFIX} Loading {ele_type} {resp_type} response data from [bold {color}]{filename}[/] ..."
                )

        if ele_type.lower() == "frame":
            ele_resp = FrameRespStepData.read_response(
                dt, resp_type=resp_type, ele_tags=ele_tags, unit_factors=POST_ARGS.unit_factors
            )
        elif ele_type.lower() == "fibersection":
            ele_resp = FiberSecRespStepData.read_response(
                dt, resp_type=resp_type, ele_tags=ele_tags, unit_factors=POST_ARGS.unit_factors
            )
        elif ele_type.lower() == "truss":
            ele_resp = TrussRespStepData.read_response(
                dt, resp_type=resp_type, ele_tags=ele_tags, unit_factors=POST_ARGS.unit_factors
            )
        elif ele_type.lower() == "link":
            ele_resp = LinkRespStepData.read_response(
                dt, resp_type=resp_type, ele_tags=ele_tags, unit_factors=POST_ARGS.unit_factors
            )
        elif ele_type.lower() == "shell":
            ele_resp = ShellRespStepData.read_response(
                dt, resp_type=resp_type, ele_tags=ele_tags, unit_factors=POST_ARGS.unit_factors
            )
        elif ele_type.lower() == "plane":
            ele_resp = PlaneRespStepData.read_response(
                dt, resp_type=resp_type, ele_tags=ele_tags, unit_factors=POST_ARGS.unit_factors
            )
        elif ele_type.lower() in ["brick", "solid"]:
            ele_resp = BrickRespStepData.read_response(
                dt, resp_type=resp_type, ele_tags=ele_tags, unit_factors=POST_ARGS.unit_factors
            )
        elif ele_type.lower() == "contact":
            ele_resp = ContactRespStepData.read_response(
                dt, resp_type=resp_type, ele_tags=ele_tags, unit_factors=POST_ARGS.unit_factors)
        else:
            raise ValueError(
                f"Unsupported element type {ele_type}, "
                "must in [Frame, Truss, Link, Shell, Plane, Solid, Contact]!"
            )

    return ele_resp


def get_sensitivity_responses(
        odb_tag: int,
        resp_type: str = None,
        print_info: bool = True,
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

    Returns
    ---------
    SensResp: `xarray.Dataset <https://docs.xarray.dev/en/stable/generated/xarray.Dataset.html>`_
        Sensitivity responses' data.
    """
    model_config = ConfigUpdate()
    filename = f"{model_config.RESULTS_DIR}/" + f"{model_config.RESP_FILE_NAME}-{odb_tag}.nc"
    with xr.open_datatree(filename, engine="netcdf4").load() as dt:
        if print_info:
            color = get_random_color()
            if resp_type is None:
                model_config.CONSOLE.print(
                    f"{model_config.PKG_PREFIX} Loading response data from [bold {color}]{filename}[/] ..."
                )
            else:
                model_config.CONSOLE.print(
                    f"{model_config.PKG_PREFIX} Loading {resp_type} response data from [bold {color}]{filename}[/] ..."
                )

        resp = SensitivityRespStepData.read_response(dt, resp_type=resp_type)

    return resp


def update_unit_system(
        pre: dict[str, str] = None,
        post: dict[str, str] = None,
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
    setattr(POST_ARGS, "unit_factors", unit_factors)
    setattr(POST_ARGS, "unit_symbols", unit_syms)


def reset_unit_system():
    """Reset unit system for post-processing.
    """
    setattr(POST_ARGS, "unit_factors", None)
    setattr(POST_ARGS, "unit_symbols", None)


def _parse_unit_factors(analysis_unit_system, post_unit_system):
    if analysis_unit_system is None or post_unit_system is None:
        unit_factors = None
        unit_syms = None
    else:
        if not isinstance(analysis_unit_system, dict):
            raise ValueError("analysis_unit_system must be of type dict!")
        if not isinstance(post_unit_system, dict):
            raise ValueError("post_unit_system must be of type dict!")
        for key in analysis_unit_system.keys():
            if key not in ["length", "force", "time"]:
                raise ValueError("key must be one of [length, force, time]!")
        for key in post_unit_system.keys():
            if key not in ["length", "force", "time"]:
                raise ValueError("key must be one of [length, force, time]!")

        analysis_units_ = dict(force=None, length=None, time=None)
        analysis_units_.update(analysis_unit_system)
        post_units_ = dict(force=None, length=None, time=None)
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
