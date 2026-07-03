"""
This file contains functions to get data from the current domain of OpenSeesPy
"""

import os
from pathlib import Path
import shutil
import time
from typing import Union

import numpy as np
import xarray as xr

from ..utils import CONFIGS, get_random_color
from ._get_model_data_base import FEMData
from ._post_utils import generate_chunk_encoding_for_datatree


class GetFEMData(FEMData):
    def __init__(self):
        super().__init__()

    def get_nodal_data(self):
        self._make_nodal_info()
        self._make_bounds()
        if len(self.node_coords) > 0:
            node_data = xr.DataArray(
                self.node_coords,
                coords={"nodeTags": self.node_tags, "coords": ["x", "y", "z"]},
                dims=["nodeTags", "coords"],
            )
            node_data.name = "NodalData"
            node_data.attrs = {
                # "bounds": tuple(self.bounds),  # must tuple
                "numNodes": len(self.node_tags),
                "minBoundSize": self.min_bound,
                "maxBoundSize": self.max_bound,
                "ndofs": tuple(self.node_ndofs),
                "ndims": tuple(self.node_ndims),
            }
        else:
            node_data = None
        return node_data

    def get_node_fixed_data(self):
        self._make_node_fixed()
        if len(self.fixed_coords) > 0:
            data = np.hstack([self.fixed_coords, self.fixed_dofs])
            fixed_nodes = xr.DataArray(
                data,
                coords={
                    "nodeTags": self.fixed_node_tags,
                    # "dofs": ("tags", self.fixed_dofs),
                    "info": [
                        "x",
                        "y",
                        "z",
                        "dof1",
                        "dof2",
                        "dof3",
                        "dof4",
                        "dof5",
                        "dof6",
                    ],
                },
                dims=["nodeTags", "info"],
            )
            fixed_nodes.name = "FixedNodalData"
        else:
            fixed_nodes = None
        return fixed_nodes

    def get_nodal_load_data(self):
        self._make_nodal_load()
        pntags = [f"{tags[0]}-{tags[1]}" for tags in self.pattern_node_tags]
        if len(self.pattern_node_tags) > 0:
            node_load_data = xr.DataArray(
                self.node_load_data,
                coords={
                    "PatternNodeTags": pntags,
                    "loadData": ["Px", "Py", "Pz"],
                },
                dims=["PatternNodeTags", "loadData"],
            )
            node_load_data.name = "NodalLoadData"
        else:
            node_load_data = None
        return node_load_data

    def get_ele_load_data(self):
        self._make_ele_load()
        beam_petags = [f"{tags[0]}-{tags[1]}" for tags in self.beam_pattern_ele_tags]
        if len(self.beam_pattern_ele_tags) > 0:
            beam_load_data = xr.DataArray(
                self.beam_ele_load_data,
                coords={
                    "loadData": [
                        "nodeI",
                        "nodeJ",
                        "wya",
                        "wyb",
                        "wza",
                        "wzb",
                        "wxa",
                        "wxb",
                        "xa",
                        "xb",
                    ],
                    "PatternEleTags": beam_petags,
                },
                dims=["PatternEleTags", "loadData"],
            )
            beam_load_data.name = "FrameLoadData"
        else:
            beam_load_data = None
        return beam_load_data

    def get_mp_constraint_data(self):
        self._make_mp_constraint()
        node_tags = [f"{tags[0]}-{tags[1]}" for tags in self.mp_pair_nodes]
        if len(self.mp_cells) > 0:
            data = np.hstack([self.mp_cells, self.mp_centers, self.mp_dofs])
            mp_constraint = xr.DataArray(
                data,
                coords={
                    "info": [
                        "numNodes",
                        "nodeI",
                        "nodeJ",
                        "xo",
                        "yo",
                        "zo",
                        "dof1",
                        "dof2",
                        "dof3",
                        "dof4",
                        "dof5",
                        "dof6",
                    ],
                    "nodeTags": node_tags,
                },
                dims=["nodeTags", "info"],
            )
            mp_constraint.name = "MPConstraintData"
        else:
            mp_constraint = None
        return mp_constraint

    def get_truss_data(self):
        if len(self.truss_cells) > 0:
            truss = xr.DataArray(
                self.truss_cells,
                coords={"cells": ["numNodes", "nodeI", "nodeJ"], "eleTags": self.truss_tags},
                dims=["eleTags", "cells"],
            )
            truss.name = "TrussData"
        else:
            truss = None
        return truss

    def get_links_data(self):
        if len(self.link_cells) > 0:
            lengths = np.array(self.link_lengths).reshape(-1, 1)
            data = np.hstack((
                self.link_cells,
                lengths,
                self.link_centers,
                self.link_xaxis,
                self.link_yaxis,
                self.link_zaxis,
            ))
            links = xr.DataArray(
                data,
                coords={
                    "info": [
                        "numNodes",
                        "nodeI",
                        "nodeJ",
                        "length",
                        "xo",
                        "yo",
                        "zo",
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
                    "eleTags": self.link_tags,
                },
                dims=["eleTags", "info"],
            )
            links.name = "LinkData"
        else:
            links = None
        return links

    def get_beams_data(self):
        if len(self.beam_cells) > 0:
            lengths = np.array(self.beam_lengths).reshape(-1, 1)
            data = np.hstack((
                self.beam_cells,
                lengths,
                self.beam_centers,
                self.beam_xaxis,
                self.beam_yaxis,
                self.beam_zaxis,
            ))
            beams = xr.DataArray(
                data,
                coords={
                    "info": [
                        "numNodes",
                        "nodeI",
                        "nodeJ",
                        "length",
                        "xo",
                        "yo",
                        "zo",
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
                    "eleTags": self.beam_tags,
                },
                dims=["eleTags", "info"],
            )
            beams.name = "BeamData"
        else:
            beams = None
        return beams

    def get_all_lines_data(self):
        if len(self.all_line_cells) > 0:
            lines = xr.DataArray(
                self.all_line_cells,
                coords={"cells": ["numNodes", "nodeI", "nodeJ"], "eleTags": self.all_line_tags},
                dims=["eleTags", "cells"],
            )
            lines.name = "AllLineElesData"
        else:
            lines = None
        return lines

    def get_shell_data(self):
        if len(self.shell_cells) > 0:
            cell_types = np.reshape(self.shell_cells_type, (-1, 1))
            data = np.hstack([self.shell_cells, cell_types])
            names = ["numNodes"] + [f"node{i + 1}" for i in range(data.shape[1] - 2)] + ["cellType"]
            shell = xr.DataArray(
                data,
                coords={"cells": names, "eleTags": self.shell_tags},
                dims=["eleTags", "cells"],
            )
            shell.name = "ShellData"
        else:
            shell = None
        return shell

    def get_plane_date(self):
        if len(self.plane_cells) > 0:
            cell_types = np.reshape(self.plane_cells_type, (-1, 1))
            data = np.hstack([self.plane_cells, cell_types])
            names = ["numNodes"] + [f"node{i + 1}" for i in range(data.shape[1] - 2)] + ["cellType"]
            plane = xr.DataArray(
                data,
                coords={"cells": names, "eleTags": self.plane_tags},
                dims=["eleTags", "cells"],
            )
            plane.name = "PlaneData"
        else:
            plane = None
        return plane

    def get_brick_data(self):
        if len(self.brick_cells) > 0:
            cell_types = np.reshape(self.brick_cells_type, (-1, 1))
            data = np.hstack([self.brick_cells, cell_types])
            names = ["numNodes"] + [f"node{i + 1}" for i in range(data.shape[1] - 2)] + ["cellType"]
            brick = xr.DataArray(data, coords={"cells": names, "eleTags": self.brick_tags}, dims=["eleTags", "cells"])
            brick.name = "BrickData"
        else:
            brick = None
        return brick

    def get_unstru_data(self):
        if len(self.unstru_cells) > 0:
            unstru_cells_type = np.array(self.unstru_cells_type).reshape(-1, 1)
            data = np.hstack([self.unstru_cells, unstru_cells_type])
            names = ["numNodes"] + [f"node{i + 1}" for i in range(data.shape[1] - 2)] + ["cellType"]
            unstru = xr.DataArray(data, coords={"cells": names, "eleTags": self.unstru_tags}, dims=["eleTags", "cells"])
            unstru.name = "UnstructuralData"
        else:
            unstru = None
        return unstru

    def get_contact_data(self):
        if len(self.contact_cells) > 0:
            contact = xr.DataArray(
                self.contact_cells,
                coords={
                    "cells": ["numNodes", "nodeI", "nodeJ"] * (len(self.contact_cells[0]) // 3),
                    "eleTags": self.contact_tags,
                },
                dims=["eleTags", "cells"],
            )
            contact.name = "ContactData"
        else:
            contact = None
        return contact

    def get_ele_centers_data(self):
        if len(self.ele_centers) > 0:
            return xr.DataArray(
                self.ele_centers,
                coords={
                    "centers": ["xo", "yo", "zo"],
                    "eleTags": self.ele_tags,
                    "eleClassTags": ("eleTags", self.ele_class_tags),
                },
                dims=["eleTags", "centers"],
                name="eleCenters",
            )
        else:
            return None

    def get_ele_data(self):
        self._make_ele_info()  # This function is called to gather all element data
        # -----------------------------------------
        truss_data = self.get_truss_data()
        beam_data = self.get_beams_data()
        link_data = self.get_links_data()
        all_lines_data = self.get_all_lines_data()
        shell_data = self.get_shell_data()
        plane_data = self.get_plane_date()
        brick_data = self.get_brick_data()
        unstru_data = self.get_unstru_data()
        contact_data = self.get_contact_data()
        ele_centers = self.get_ele_centers_data()
        # --------------------------------------------------------------
        all_eles = {}  # all elements data, key is the element type, such as "ZeroLength"
        for key in self.ELE_CELLS_VTK:
            cells_type = np.array(self.ELE_CELLS_TYPE_VTK[key])
            cells_type = np.reshape(cells_type, (-1, 1))
            data = np.hstack([self.ELE_CELLS_VTK[key], cells_type])
            names = ["numNodes"] + [f"node{i + 1}" for i in range(data.shape[1] - 2)] + ["cellType"]
            all_eles[key] = xr.DataArray(
                data,
                coords={"info": names, "eleTags": self.ELE_CELLS_TAGS[key]},
                dims=["eleTags", "info"],
                name=key,
            )
        return (
            truss_data,
            beam_data,
            link_data,
            all_lines_data,
            shell_data,
            plane_data,
            brick_data,
            unstru_data,
            contact_data,
            ele_centers,
            all_eles,
        )

    def get_model_info(self):
        nodal_data = self.get_nodal_data()
        node_fixed_data = self.get_node_fixed_data()
        nodal_load_data = self.get_nodal_load_data()
        ele_load_data = self.get_ele_load_data()
        mp_constraint_data = self.get_mp_constraint_data()

        ele_data = self.get_ele_data()

        # ----------------------------------------------------------------
        # update and save the model info
        if nodal_data is not None:
            nodal_data.attrs["unusedNodeTags"] = tuple(self.unused_node_tags)
            self.MODEL_INFO[nodal_data.name] = nodal_data
        if node_fixed_data is not None:
            self.MODEL_INFO[node_fixed_data.name] = node_fixed_data
        if nodal_load_data is not None:
            self.MODEL_INFO[nodal_load_data.name] = nodal_load_data
        if ele_load_data is not None:
            self.MODEL_INFO[ele_load_data.name] = ele_load_data
        if mp_constraint_data is not None:
            self.MODEL_INFO[mp_constraint_data.name] = mp_constraint_data
        # -----------------------------------------------------------------
        for edata in ele_data[:-1]:
            if edata is not None:
                self.MODEL_INFO[edata.name] = edata

        self.ELE_CELLS = ele_data[-1]

        return self.MODEL_INFO, self.ELE_CELLS


def save_model_data(
    odb_tag: Union[str, int] = 1,
):
    """Save the model data from the current domain.

    .. Note::
       Since this package chooses `xarray <https://docs.xarray.dev/en/stable/index.html>`_
       as the data structure, it is saved in
       `netCDF <https://docs.xarray.dev/en/stable/user-guide/io.html>`_ format.

    Parameters
    ----------
    odb_tag: Union[str, int], default = 1
        Output database tag, the data will be saved in ``ModelData-{odb_tag}.nc``.
    """
    RESULTS_DIR = CONFIGS.get_output_dir()
    CONSOLE = CONFIGS.get_console()
    PKG_PREFIX = CONFIGS.get_pkg_prefix()
    MODEL_FILE_NAME = CONFIGS.get_model_filename()
    odb_format, _ = CONFIGS.get_odb_format()

    output_filename = str(Path(RESULTS_DIR) / f"{MODEL_FILE_NAME}-{odb_tag}.{odb_format}")
    model_data = GetFEMData()
    model_info, cells = model_data.get_model_info()
    model_data = {}
    for key in model_info:
        model_data[f"ModelInfo/{key}"] = xr.Dataset({key: model_info[key]})
    for key in cells:
        model_data[f"Cells/{key}"] = xr.Dataset({key: cells[key]})
    if model_data == {}:
        color = get_random_color()
        CONSOLE.print(f"{PKG_PREFIX} No model data to be saved!")
        raise RuntimeError()
    dt = xr.DataTree.from_dict(model_data, name=f"{MODEL_FILE_NAME}")

    max_retries = 5
    retry_delay = 1

    for attempt in range(max_retries + 1):
        try:
            # try to remove existing directory before writing
            if attempt > 0 and os.path.exists(output_filename):
                shutil.rmtree(output_filename)
                # Windows may take some time to release file locks
                time.sleep(0.5)
            if odb_format.lower() == "zarr":
                encoding = generate_chunk_encoding_for_datatree(dt, target_chunk_mb=10.0)
                dt.to_zarr(output_filename, mode="w", consolidated=True, encoding=encoding, zarr_format=2)
            else:
                dt.to_netcdf(output_filename, mode="w", engine="netcdf4")
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
    # --------------------------------------------------------------------------------------------------
    color = get_random_color()
    CONSOLE.print(f"{PKG_PREFIX} Model data has been saved to [bold {color}]{output_filename}[/]!")


def load_model_data(
    odb_tag: Union[str, int] = 1,
    resave: bool = False,
) -> tuple[dict[str, xr.DataArray], dict[str, xr.DataArray]]:
    """Get the model data from the saved file.

    Parameters
    ----------
    odb_tag: Union[str, int], default = 1
        Output database tag, the data that have been saved in ``ModelData-{odb_tag}.nc``.
    resave: bool, default=True
        Resave the model data.

    Returns
    --------
    model_info: dict[xarray.DataArray]
    cells: dict[xarray.DataArray]
    """
    RESULTS_DIR = CONFIGS.get_output_dir()
    CONSOLE = CONFIGS.get_console()
    PKG_PREFIX = CONFIGS.get_pkg_prefix()
    MODEL_FILE_NAME = CONFIGS.get_model_filename()

    if odb_tag is None:
        model_data = GetFEMData()
        model_info, cells = model_data.get_model_info()
    else:
        odb_format, odb_engine = CONFIGS.get_odb_format()
        kargs = {"consolidated": False} if odb_format.lower() == "zarr" else {}
        filename = str(Path(RESULTS_DIR) / f"{MODEL_FILE_NAME}-{odb_tag}.{odb_format}")
        if not os.path.exists(filename):
            resave = True
        if resave:
            save_model_data(odb_tag=odb_tag)
        else:
            color = get_random_color()
            CONSOLE.print(f"{PKG_PREFIX} Loading model data from [bold {color}]{filename}[/] ...")
        model_info, cells = {}, {}
        with xr.open_datatree(filename, engine=odb_engine, **kargs).load() as dt:
            if "ModelInfo" in dt:
                for key, value in dt["ModelInfo"].items():
                    model_info[key] = value[key]
            if "Cells" in dt:
                for key, value in dt["Cells"].items():
                    cells[key] = value[key]
            dt.close()
        if model_info == {} and cells == {}:
            raise RuntimeError(f"{PKG_PREFIX} No model data in the file {filename}!")  # noqa: TRY003
    return model_info, cells
