"""
This file contains functions to get data from the current domain of OpenSeesPy
"""

import os
import numpy as np
import xarray as xr
from typing import Union

from ..utils import CONSTANTS, get_random_color, ConfigUpdate
from ._get_model_data_base import FEMData

# RESULTS_DIR = CONSTANTS.get_output_dir()
# CONSOLE = CONSTANTS.get_console()
# PKG_PREFIX = CONSTANTS.get_pkg_prefix()
# MODEL_FILE_NAME = CONSTANTS.get_model_filename()

class GetFEMData(FEMData):

    def __init__(self):
        super().__init__()

    def get_nodal_data(self):
        self._make_nodal_info()
        self._make_bounds()
        if len(self.node_coords) > 0:
            node_data = xr.DataArray(
                self.node_coords,
                coords={
                    "tags": self.node_tags,
                    "coords": ["x", "y", "z"],
                },
                dims=["tags", "coords"]
            )
        else:
            node_data = xr.DataArray(
                self.node_coords,
                # coords={
                #     "tags": [],
                #     "coords": [],
                # },
                # dims=["tags", "coords"],
            )
        node_data.name = "NodalData"
        node_data.attrs = {
            "bounds": self.bounds,  # must tuple
            "numNodes": len(self.node_tags),
            "minBoundSize": self.min_bound,
            "maxBoundSize": self.max_bound,
            "ndofs": tuple(self.node_ndofs),
            "ndims": tuple(self.node_ndims),
        }
        return node_data

    def get_node_fixed_data(self):
        self._make_node_fixed()
        if len(self.fixed_coords) > 0:
            data = np.hstack([self.fixed_coords, self.fixed_dofs])
            fixed_nodes = xr.DataArray(
                data,
                coords={
                    "tags": self.fixed_node_tags,
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
                dims=["tags", "info"],
            )
        else:
            fixed_nodes = xr.DataArray(self.fixed_coords)
        fixed_nodes.name = "FixedNodalData"
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
        else:
            node_load_data = xr.DataArray(self.node_load_data)
        node_load_data.name = "NodalLoadData"
        return node_load_data

    def get_ele_load_data(self):
        self._make_ele_load()
        petags = [f"{tags[0]}-{tags[1]}" for tags in self.pattern_ele_tags]
        if len(self.pattern_ele_tags) > 0:
            ele_load_data = xr.DataArray(
                self.ele_load_data,
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
                    "PatternEleTags": petags,
                },
                dims=["PatternEleTags", "loadData"],
            )
        else:
            ele_load_data = xr.DataArray(self.ele_load_data)
        ele_load_data.name = "EleLoadData"
        return ele_load_data

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
        else:
            mp_constraint = xr.DataArray(self.mp_cells)
        mp_constraint.name = "MPConstraintData"
        return mp_constraint

    def get_truss_data(self):
        if len(self.truss_cells) > 0:
            truss = xr.DataArray(
                self.truss_cells,
                coords={
                    "cells": ["numNodes", "nodeI", "nodeJ"],
                    "eleTags": self.truss_tags,
                },
                dims=["eleTags", "cells"],
            )
        else:
            truss = xr.DataArray(self.truss_cells)
        truss.name = "TrussData"
        return truss

    def get_links_data(self):
        if len(self.link_cells) > 0:
            lengths = np.array(self.link_lengths).reshape(-1, 1)
            data = np.hstack(
                (
                    self.link_cells,
                    lengths,
                    self.link_centers,
                    self.link_xaxis,
                    self.link_yaxis,
                    self.link_zaxis,
                )
            )
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
        else:
            links = xr.DataArray(self.link_cells)
        links.name = "LinkData"
        return links

    def get_beams_data(self):
        if len(self.beam_cells) > 0:
            lengths = np.array(self.beam_lengths).reshape(-1, 1)
            data = np.hstack(
                (
                    self.beam_cells,
                    lengths,
                    self.beam_centers,
                    self.beam_xaxis,
                    self.beam_yaxis,
                    self.beam_zaxis,
                )
            )
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
        else:
            beams = xr.DataArray(self.beam_cells)
        beams.name = "BeamData"
        return beams

    def get_all_lines_data(self):
        if len(self.all_line_cells) > 0:
            lines = xr.DataArray(
                self.all_line_cells,
                coords={
                    "cells": ["numNodes", "nodeI", "nodeJ"],
                    "eleTags": self.all_line_tags,
                },
                dims=["eleTags", "cells"],
            )
        else:
            lines = xr.DataArray(self.all_line_cells)
        lines.name = "AllLineElesData"
        return lines

    def get_shell_data(self):
        if len(self.shell_cells) > 0:
            cell_types = np.reshape(self.shell_cells_type, (-1, 1))
            data = np.hstack([self.shell_cells, cell_types])
            shell = xr.DataArray(
                data,
                coords={
                    # "cells": ["numNodes"] + [f"node{i+1}" for i in range(num-1)],
                    "eleTags": self.shell_tags,
                },
                dims=["eleTags", "cells"],
            )
        else:
            shell = xr.DataArray(self.shell_cells)
        shell.name = "ShellData"
        return shell

    def get_plane_date(self):
        if len(self.plane_cells) > 0:
            cell_types = np.reshape(self.plane_cells_type, (-1, 1))
            data = np.hstack([self.plane_cells, cell_types])
            plane = xr.DataArray(
                data,
                coords={
                    "eleTags": self.plane_tags,
                },
                dims=["eleTags", "cells"],
            )
        else:
            plane = xr.DataArray(self.plane_cells)
        plane.name = "PlaneData"
        return plane

    def get_brick_data(self):
        if len(self.brick_cells) > 0:
            cell_types = np.reshape(self.brick_cells_type, (-1, 1))
            data = np.hstack([self.brick_cells, cell_types])
            brick = xr.DataArray(
                data,
                coords={
                    "eleTags": self.brick_tags,
                },
                dims=["eleTags", "cells"],
            )
        else:
            brick = xr.DataArray(self.brick_cells)
        brick.name = "BrickData"
        return brick

    def get_unstru_data(self):
        if len(self.unstru_cells) > 0:
            unstru_cells_type = np.array(self.unstru_cells_type).reshape(-1, 1)
            data = np.hstack([self.unstru_cells, unstru_cells_type])
            names = (
                    ["numNodes"]
                    + [f"node{i + 1}" for i in range(data.shape[1] - 2)]
                    + ["cellType"]
            )
            unstru = xr.DataArray(
                data,
                coords={
                    "cells": names,
                    "eleTags": self.unstru_tags,
                },
                dims=["eleTags", "cells"],
            )
        else:
            unstru = xr.DataArray(self.unstru_cells)
        unstru.name = "UnstructuralData"
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
        else:
            contact = xr.DataArray(self.contact_cells)
        contact.name = "ContactData"
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
            return xr.DataArray(self.ele_centers, name="eleCenters")

    def get_ele_data(self):
        self._make_ele_info()
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
        all_eles = dict()
        for key in self.ELE_CELLS_VTK.keys():
            cells_type = np.array(self.ELE_CELLS_TYPE_VTK[key])
            cells_type = np.reshape(cells_type, (-1, 1))
            data = np.hstack([self.ELE_CELLS_VTK[key], cells_type])
            names = (
                    ["numNodes"]
                    + [f"node{i + 1}" for i in range(data.shape[1] - 2)]
                    + ["cellType"]
            )
            all_eles[key] = xr.DataArray(
                data,
                coords={
                    "info": names,
                    "eleTags": self.ELE_CELLS_TAGS[key],
                },
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
        nodal_data.attrs["unusedNodeTags"] = tuple(self.unused_node_tags)
        self.MODEL_INFO[nodal_data.name] = nodal_data
        self.MODEL_INFO[node_fixed_data.name] = node_fixed_data
        self.MODEL_INFO[nodal_load_data.name] = nodal_load_data
        self.MODEL_INFO[ele_load_data.name] = ele_load_data
        self.MODEL_INFO[mp_constraint_data.name] = mp_constraint_data
        # -----------------------------------------------------------------
        for edata in ele_data[:-1]:
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
    model_config = ConfigUpdate()
    output_filename = model_config.RESULTS_DIR + "/" + f"{model_config.MODEL_FILE_NAME}-{odb_tag}.nc"
    model_data = GetFEMData()
    model_info, cells = model_data.get_model_info()
    model_data = dict()
    for key in model_info.keys():
        model_data[f"ModelInfo/{key}"] = xr.Dataset({key: model_info[key]})
    if len(cells) > 0:
        for key in cells.keys():
            model_data[f"Cells/{key}"] = xr.Dataset({key: cells[key]})
    else:
        model_data["Cells"] = xr.Dataset()
    dt = xr.DataTree.from_dict(model_data, name=f"{model_config.MODEL_FILE_NAME}")
    dt.to_netcdf(output_filename, mode="w", engine="netcdf4")
    # /////////////////////////////////////
    color = get_random_color()
    model_config.CONSOLE.print(
        f"{model_config.PKG_PREFIX} Model data has been saved to [bold {color}]{output_filename}[/]!"
    )


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
    model_config = ConfigUpdate()
    if odb_tag is None:
        model_data = GetFEMData()
        model_info, cells = model_data.get_model_info()
    else:
        filename = f"{model_config.RESULTS_DIR}/" + f"{model_config.MODEL_FILE_NAME}-{odb_tag}.nc"
        if not os.path.exists(filename):
            resave = True
        if resave:
            save_model_data(odb_tag=odb_tag)
        else:
            color = get_random_color()
            model_config.CONSOLE.print(
                f"{model_config.PKG_PREFIX} Loading model data from [bold {color}]{filename}[/] ..."
            )
        model_info, cells = dict(), dict()
        with xr.open_datatree(filename, engine="netcdf4").load() as dt:
            for key, value in dt["ModelInfo"].items():
                model_info[key] = value[key]
            for key, value in dt["Cells"].items():
                cells[key] = value[key]
    return model_info, cells

#
# def save_model_data(
#         odb_tag: Union[str, int] = 1,
# ):
#     """Save the model data from the current domain.
#
#     Parameters
#     ----------
#     odb_tag: Union[str, int], default = 1
#         Output database tag, the data will be saved in ``ModelData-{odb_tag}.hdf5``.
#     """
#     # --------------------------------
#     output_filename = RESULTS_DIR + "/" + f"ModelData-{odb_tag}.hdf5"
#     model_data = FEMData()
#     model_info, cells = model_data.get_model_info()
#     with h5py.File(output_filename, "w") as f:
#         grp = f.create_group("ModelInfo")
#         for name, value in model_info.items():
#             grp.create_dataset(name, data=value)
#         grp = f.create_group("EleCells")
#         for name, value in cells.items():
#             subgrp = grp.create_group(name)
#             for sub_name, sub_value in value.items():
#                 subgrp.create_dataset(sub_name, data=sub_value)
#     color = get_random_color()
#     CONSOLE.print(
#         f"{PKG_PREFIX} Model data has been saved to [bold {color}]{output_filename}[/]!"
#     )
#
#
# def get_model_data(odb_tag: Union[str, int] = 1):
#     """Get the model data from the saved file.
#
#     Parameters
#     ----------
#     odb_tag: Union[str, int], default = 1
#         Output database tag, the data that have be saved in ``ModelData-{odb_tag}.hdf5``.
#     """
#     filename = f"{RESULTS_DIR}/" + f"ModelData-{odb_tag}.hdf5"
#     if not os.path.exists(filename):
#         save_model_data(odb_tag=odb_tag)
#     model_info, cells = dict(), defaultdict(dict)
#     with h5py.File(filename, "r") as f:
#         grp = f["ModelInfo"]
#         for name, value in grp.items():
#             model_info[name] = value[...]
#         grp = f["EleCells"]
#         for name, value in grp.items():
#             for subname, subvalue in value.items():
#                 cells[name][subname] = subvalue[...]
#     return model_info, cells
