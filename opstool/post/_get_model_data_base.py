from collections import defaultdict

import numpy as np
import openseespy.opensees as ops

from opstool.utils import OPS_ELE_CLASSTAG2TYPE, OPS_ELE_TAGS

# from pyvista import CellType

# LINE_CELL_TYPE_VTK = {2: CellType.LINE, 3: CellType.QUADRATIC_EDGE}
# PLANE_CELL_TYPE_VTK = {
#     3: CellType.TRIANGLE, 4: CellType.QUAD, 6: CellType.QUADRATIC_TRIANGLE, 7: CellType.BIQUADRATIC_TRIANGLE,
#     8: CellType.QUADRATIC_QUAD, 9: CellType.BIQUADRATIC_QUAD
# }
# SOLID_CELL_TYPE_VTK = {
#     4: CellType.TETRA, 8: CellType.HEXAHEDRON, 10: CellType.QUADRATIC_TETRA, 20: CellType.QUADRATIC_HEXAHEDRON,
#     24: CellType.BIQUADRATIC_QUADRATIC_HEXAHEDRON, 27: CellType.TRIQUADRATIC_HEXAHEDRON
# }
LINE_CELL_TYPE_VTK = {2: 3, 3: 21}  # KEY : NUM.VERTEX; VALUE: VTK CELL TYPE
PLANE_CELL_TYPE_VTK = {3: 5, 4: 9, 6: 22, 7: 34, 8: 23, 9: 28}
SOLID_CELL_TYPE_VTK = {4: 10, 8: 12, 10: 24, 20: 25, 24: 33, 27: 29}

INT_TYPE = np.int32
FLOAT_TYPE = np.float32


class FEMData:
    """
    A class for collecting data in the current domain of OpenSeesPy.
    """

    def __init__(self) -> None:
        self.MODEL_INFO = {}
        self.ELE_CELLS = {}
        self.ELE_CELLS_VTK = defaultdict(list)  # key: EleClassName, value: Element cells in VTK
        self.ELE_CELLS_TYPE_VTK = defaultdict(list)  # key: EleClassName, value: Element cell type in VTK
        self.ELE_CELLS_TAGS = defaultdict(list)  # key: EleClassName, value: Element tags
        # ------------------------------------------------------------------------
        # --------------------------nodal info------------------------------------
        # ------------------------------------------------------------------------
        self.node_tags = ops.getNodeTags()
        self.unused_node_tags = []  # record unused nodal tags by this pacakge
        self.node_coords = []  # Nodal Coords
        self.node_index = {}  # Key: nodeTag, value: index in self.node_coords
        self.node_ndims, self.node_ndofs = [], []  # Nodal Dims, Nodal dofs
        self.bounds, self.min_bound, self.max_bound = (), 0, 0
        # Fixed node
        self.fixed_node_tags = ops.getFixedNodes()  # Fixed Nodal Tags, ie, fix() commands
        self.fixed_coords, self.fixed_dofs = [], []
        # Nodal load info
        self.pattern_tags = ops.getPatterns()
        self.node_load_data, self.pattern_node_tags = [], []
        # Ele load info
        self.ele_load_data, self.pattern_ele_tags = [], []
        # mp-constraints
        self.mp_cells, self.mp_centers, self.mp_dofs, self.mp_pair_nodes = (
            [],
            [],
            [],
            [],
        )
        # ------------------------------------------------------------------------
        # --------------------------Element info----------------------------------
        # ------------------------------------------------------------------------
        self.ele_tags = []
        self.ele_centers, self.ele_class_tags = [], []
        # -----------------------------------------------------------------------
        self.truss_tags, self.truss_cells = [], []
        # -----------------------------------------------------------------------
        self.beam_tags, self.beam_cells = [], []
        self.beam_centers, self.beam_lengths = [], []
        self.beam_xaxis, self.beam_yaxis, self.beam_zaxis = [], [], []
        # -----------------------------------------------------------------------
        self.other_line_tags = []
        self.all_line_tags, self.all_line_cells = [], []
        # -----------------------------------------------------------------------
        self.link_tags, self.link_cells = [], []
        self.link_centers, self.link_lengths = [], []
        self.link_xaxis, self.link_yaxis, self.link_zaxis = [], [], []
        # ------------------------------------------------------------------------
        self.contact_tags, self.contact_cells = [], []
        # ------------------------------------------------------------------------
        self.plane_tags, self.plane_cells, self.plane_cells_type = [], [], []
        self.shell_tags, self.shell_cells, self.shell_cells_type = [], [], []
        self.brick_tags, self.brick_cells, self.brick_cells_type = [], [], []
        self.joint_tags = []
        self.unstru_tags, self.unstru_cells, self.unstru_cells_type = [], [], []

    def get_node_tags(self):
        return self.node_tags

    def get_ele_tags(self):
        return self.ele_tags

    def _make_bounds(self):
        min_node = np.min(self.node_coords, axis=0)
        max_node = np.max(self.node_coords, axis=0)
        self.bounds = (
            min_node[0],
            max_node[0],
            min_node[1],
            max_node[1],
            min_node[2],
            max_node[2],
        )
        sizes = [
            max_node[0] - min_node[0],
            max_node[1] - min_node[1],
            max_node[2] - min_node[2],
        ]
        self.min_bound, self.max_bound = np.min(sizes), np.max(sizes)

    def _make_nodal_info(self):
        for i, tag in enumerate(self.node_tags):
            coord = ops.nodeCoord(tag)
            ndim = ops.getNDM(tag)[0]
            ndof = ops.getNDF(tag)[0]
            if ndim == 1:
                coord.extend([0, 0])
            elif ndim == 2:
                coord.extend([0])
            self.node_ndims.append(ndim)
            self.node_ndofs.append(ndof)
            self.node_coords.append(coord)
            self.node_index[tag] = i
        self.node_coords = np.array(self.node_coords, dtype=FLOAT_TYPE)

    def _make_node_fixed(self):
        for tag in self.fixed_node_tags:
            if tag in self.node_index:
                self.fixed_coords.append(self.node_coords[self.node_index[tag]])
                fixeddofs = ops.getFixedDOFs(tag)
                fixities = [0] * 6
                for dof in fixeddofs:
                    fixities[dof - 1] = 1
                self.fixed_dofs.append(fixities)
            else:
                self.fixed_coords.append([np.nan] * 3)
                self.fixed_dofs.append([np.nan] * 6)

    def _make_nodal_load(self):
        self.node_load_data, self.pattern_node_tags = [], []
        for pattern in self.pattern_tags:
            nodetags = ops.getNodeLoadTags(pattern)
            loaddata = ops.getNodeLoadData(pattern)
            loc = 0
            for tag in nodetags:
                self.pattern_node_tags.append([pattern, tag])
                ndm = ops.getNDM(tag)[0]
                ndf = ops.getNDF(tag)[0]
                data = loaddata[loc : loc + ndf]
                if ndm <= 2 and ndf <= 3:
                    self.node_load_data.append([data[0], data[1], 0])  # px, py, pz=0
                else:
                    self.node_load_data.append([data[0], data[1], data[2]])  # px, py, pz
                loc += ndf
        self.pattern_node_tags = np.array(self.pattern_node_tags)

    def _make_ele_load(self):
        self.ele_load_data, self.pattern_ele_tags = [], []
        for pattern in self.pattern_tags:
            eletags = ops.getEleLoadTags(pattern)
            eleclasstags = ops.getEleLoadClassTags(pattern)
            loaddata = ops.getEleLoadData(pattern)
            loc = 0
            for tag, classtag in zip(eletags, eleclasstags):
                ntags = ops.eleNodes(tag)
                if len(ntags) != 2:
                    continue
                wya, wyb, wza, wzb, wxa, wxb, xa, xb = (
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    1.0,
                )
                if classtag == 3:  # beamUniform2D, Wya <Wxa>
                    wy, wx = loaddata[loc : loc + 2]
                    wya, wyb, wxa, wxb = wy, wy, wx, wx
                    loc += 2
                elif classtag == 12:  # beamUniform2D, Wya <Wxa> <aL> <bL> <Wyb> <Wxb>
                    wya, wyb, wxa, wxb, xa, xb = loaddata[loc : loc + 6]
                    loc += 6
                elif classtag == 5:  # beamUniform3D wy, wz, wx
                    wy, wz, wx = loaddata[loc : loc + 3]
                    wya, wyb, wza, wzb, wxa, wxb = wy, wy, wz, wz, wx, wx
                    loc += 3
                elif classtag == 4:  # beamPoint2D, Py xL <Px>
                    wya, wxa, xa = loaddata[loc : loc + 3]
                    xb = -10000
                    loc += 3
                elif classtag == 6:  # beamPoint3D, Py, Pz, x, N
                    wya, wza, wxa, xa = loaddata[loc : loc + 4]
                    xb = -10000
                    loc += 4
                else:
                    pass
                data = [*ntags, wya, wyb, wza, wzb, wxa, wxb, xa, xb]
                self.ele_load_data.append(data)
                self.pattern_ele_tags.append([pattern, tag])
        self.pattern_ele_tags = np.array(self.pattern_ele_tags)

    def _make_mp_constraint(self):
        self.mp_cells, self.mp_centers, self.mp_dofs, self.mp_pair_nodes = (
            [],
            [],
            [],
            [],
        )
        retained_nodes = ops.getRetainedNodes()
        for tag in retained_nodes:
            constrained_nodes = ops.getConstrainedNodes(tag)
            for tag2 in constrained_nodes:
                self.mp_pair_nodes.append([tag, tag2])
                self.mp_cells.append([2, self.node_index[tag], self.node_index[tag2]])
                self.mp_centers.append(
                    (self.node_coords[self.node_index[tag]] + self.node_coords[self.node_index[tag2]]) / 2
                )
                dofs = ops.getRetainedDOFs(tag, tag2)
                fixities = [0] * 6
                for dof in dofs:
                    fixities[dof - 1] = 1
                self.mp_dofs.append(fixities)
        self.mp_pair_nodes = np.array(self.mp_pair_nodes)

    def _make_truss_info(self, ele_tag):
        self.truss_tags.append(ele_tag)
        ele_nodes = ops.eleNodes(ele_tag)
        node_i, node_j = ele_nodes[:2]
        idx_i, idx_j = self.node_index[node_i], self.node_index[node_j]
        # --------------------------------------------------------
        self.truss_cells.append([2, idx_i, idx_j])

    @staticmethod
    def _get_local_axis(ele_tag):
        xaxis = ops.eleResponse(ele_tag, "xaxis")
        yaxis = ops.eleResponse(ele_tag, "yaxis")
        zaxis = ops.eleResponse(ele_tag, "zaxis")
        if not xaxis:
            xaxis = ops.eleResponse(ele_tag, "xlocal")
        if not yaxis:
            yaxis = ops.eleResponse(ele_tag, "ylocal")
        if not zaxis:
            zaxis = ops.eleResponse(ele_tag, "zlocal")
        xaxis = np.array(xaxis) / np.linalg.norm(xaxis) if xaxis else np.array([0.0, 0.0, 0.0])
        yaxis = np.array(yaxis) / np.linalg.norm(yaxis) if yaxis else np.array([0.0, 0.0, 0.0])
        zaxis = np.array(zaxis) / np.linalg.norm(zaxis) if zaxis else np.array([0.0, 0.0, 0.0])
        return xaxis, yaxis, zaxis

    def _make_link_info(self, ele_tag):
        node_i, node_j = ops.eleNodes(ele_tag)[:2]
        idx_i, idx_j = self.node_index[node_i], self.node_index[node_j]
        coordi, coordj = self.node_coords[idx_i], self.node_coords[idx_j]
        # --------------------------------------------------------
        self.link_tags.append(ele_tag)
        self.link_cells.append([2, idx_i, idx_j])
        self.link_centers.append((coordi + coordj) / 2)
        self.link_lengths.append(np.linalg.norm(coordj - coordi))
        xaxis, yaxis, zaxis = self._get_local_axis(ele_tag)
        self.link_xaxis.append(xaxis)
        self.link_yaxis.append(yaxis)
        self.link_zaxis.append(zaxis)

    def _make_beam_info(self, ele_tag):
        node_i, node_j = ops.eleNodes(ele_tag)[:2]
        idx_i, idx_j = self.node_index[node_i], self.node_index[node_j]
        # --------------------------------------------------------
        self.beam_tags.append(ele_tag)
        self.beam_cells.append([2, idx_i, idx_j])
        # --------------------------------------------------------
        self.beam_centers.append((self.node_coords[idx_i] + self.node_coords[idx_j]) / 2)
        self.beam_lengths.append(np.sqrt(np.sum((self.node_coords[idx_i] - self.node_coords[idx_j]) ** 2)))
        xaxis, yaxis, zaxis = self._get_local_axis(ele_tag)
        self.beam_xaxis.append(xaxis)
        self.beam_yaxis.append(yaxis)
        self.beam_zaxis.append(zaxis)

    def _make_contact_info(self, ele_tag, ele_class_tag):
        ele_nodes = ops.eleNodes(ele_tag)
        key = OPS_ELE_CLASSTAG2TYPE[ele_class_tag]
        self.contact_tags.append(ele_tag)
        cell = []
        if ele_class_tag in [22, 23, 24, 25, 140]:  # zero-length element
            if len(ele_nodes) == 2:
                pass
            elif len(ele_nodes) > 2:
                mid = len(ele_nodes) // 2
                part1 = ele_nodes[:mid]
                part2 = ele_nodes[mid:]
                part2 = part2[::-1]
                for tag1, tag2 in zip(part1, part2):
                    cell.extend([2, self.node_index[tag1], self.node_index[tag2]])
                    self.ELE_CELLS_VTK[key].append([2, self.node_index[tag1], self.node_index[tag2]])
                    self.ELE_CELLS_TYPE_VTK[key].append(LINE_CELL_TYPE_VTK[2])
                    self.ELE_CELLS_TAGS[key].append(ele_tag)
        else:
            ele_nodes = ops.eleNodes(ele_tag)
            cNode = ele_nodes[-2]
            rNodes = ele_nodes[:-2]
            # record the last Lagrange multiplier node that will be not used
            self.unused_node_tags.append(ele_nodes[-1])
            for rntag in rNodes:
                cell.extend([2, self.node_index[cNode], self.node_index[rntag]])
                self.ELE_CELLS_VTK[key].append([2, self.node_index[cNode], self.node_index[rntag]])
                self.ELE_CELLS_TYPE_VTK[key].append(LINE_CELL_TYPE_VTK[2])
                self.ELE_CELLS_TAGS[key].append(ele_tag)
        self.contact_cells.append(cell)

    def _make_all_line_info(self, ele_tag, class_tag):
        idxs = [self.node_index[tag_] for tag_ in ops.eleNodes(ele_tag)[:2]]
        key = OPS_ELE_CLASSTAG2TYPE[class_tag]
        self.ELE_CELLS_VTK[key].append([2, *idxs])
        self.ELE_CELLS_TYPE_VTK[key].append(LINE_CELL_TYPE_VTK[2])
        self.ELE_CELLS_TAGS[key].append(ele_tag)
        self.all_line_tags.append(ele_tag)
        self.all_line_cells.append([2, *idxs])

    def _make_plane_shell_solid_info(self, ele_type: str, ele_tag, class_tag):
        ele_nodes = ops.eleNodes(ele_tag)
        idxs = [self.node_index[tag_] for tag_ in ele_nodes]
        key = OPS_ELE_CLASSTAG2TYPE[class_tag]
        if class_tag in OPS_ELE_TAGS.Wall:
            idxs = [idxs[0], idxs[1], idxs[3], idxs[2]]
        # ---------------------------------------------------------------------------
        self.ELE_CELLS_VTK[key].append([len(idxs), *idxs])
        self.ELE_CELLS_TAGS[key].append(ele_tag)
        self.unstru_tags.append(ele_tag)
        self.unstru_cells.append([len(idxs), *idxs])
        if ele_type.lower() == "plane":
            self.ELE_CELLS_TYPE_VTK[key].append(PLANE_CELL_TYPE_VTK[len(ele_nodes)])
            self.unstru_cells_type.append(PLANE_CELL_TYPE_VTK[len(ele_nodes)])
            self.plane_cells.append([len(idxs), *idxs])
            self.plane_cells_type.append(PLANE_CELL_TYPE_VTK[len(ele_nodes)])
        elif ele_type.lower() == "shell":
            self.ELE_CELLS_TYPE_VTK[key].append(PLANE_CELL_TYPE_VTK[len(ele_nodes)])
            self.unstru_cells_type.append(PLANE_CELL_TYPE_VTK[len(ele_nodes)])
            self.shell_cells.append([len(idxs), *idxs])
            self.shell_cells_type.append(PLANE_CELL_TYPE_VTK[len(ele_nodes)])
        else:  # solid
            self.ELE_CELLS_TYPE_VTK[key].append(SOLID_CELL_TYPE_VTK[len(ele_nodes)])
            self.unstru_cells_type.append(SOLID_CELL_TYPE_VTK[len(ele_nodes)])
            self.brick_cells.append([len(idxs), *idxs])
            self.brick_cells_type.append(SOLID_CELL_TYPE_VTK[len(ele_nodes)])

    def _make_plane_info(self, ele_tag, class_tag):
        self.plane_tags.append(ele_tag)
        self._make_plane_shell_solid_info("plane", ele_tag, class_tag)

    def _make_shell_info(self, ele_tag, class_tag):
        self.shell_tags.append(ele_tag)
        self._make_plane_shell_solid_info("shell", ele_tag, class_tag)

    def _make_solid_info(self, ele_tag, class_tag):
        self.brick_tags.append(ele_tag)
        self._make_plane_shell_solid_info("solid", ele_tag, class_tag)

    def _make_joint_info(self, ele_tag, class_tag):
        self.unstru_tags.append(ele_tag)
        self.joint_tags.append(ele_tag)
        ele_nodes = ops.eleNodes(ele_tag)
        idxs = [self.node_index[tag_] for tag_ in ele_nodes]
        key = OPS_ELE_CLASSTAG2TYPE[class_tag]
        self.ELE_CELLS_TAGS[key].append(ele_tag)
        #  both len(idxs) in (4, 5) and len(idxs) == 7
        self.ELE_CELLS_VTK[key].append([4] + idxs[:4])
        self.ELE_CELLS_TYPE_VTK[key].append(PLANE_CELL_TYPE_VTK[4])
        self.unstru_cells.append([4] + idxs[:4])
        self.unstru_cells_type.append(PLANE_CELL_TYPE_VTK[4])
        if len(idxs) == 7:  # Joint3D, add a quad
            self.ELE_CELLS_VTK[key].append([4, idxs[4], idxs[1], idxs[5], idxs[3]])
            self.ELE_CELLS_TYPE_VTK[key].append(PLANE_CELL_TYPE_VTK[4])
            self.unstru_cells.append([4, idxs[4], idxs[1], idxs[5], idxs[3]])
            self.unstru_cells_type.append(PLANE_CELL_TYPE_VTK[4])

    def _make_ele_centers(self, ele_tag, class_tag):
        # coords
        ele_nodes = ops.eleNodes(ele_tag)
        idxs = [self.node_index[tag_] for tag_ in ele_nodes]
        coords = [self.node_coords[idx] for idx in idxs]
        self.ele_centers.append(np.mean(coords, axis=0))
        # ele_class_tags
        self.ele_class_tags.append(class_tag)
        self.ele_tags.append(ele_tag)

    def _make_ele_info(self):
        for ele_tag in ops.getEleTags():
            class_tag = ops.getEleClassTags(ele_tag)
            if not isinstance(class_tag, int):
                class_tag = class_tag[0]
            num_nodes = len(ops.eleNodes(ele_tag))
            if num_nodes == 2:
                self._make_all_line_info(ele_tag, class_tag)
                if class_tag in OPS_ELE_TAGS.Truss:
                    self._make_truss_info(ele_tag)
                    self._make_ele_centers(ele_tag, class_tag)
                elif class_tag in OPS_ELE_TAGS.Beam:
                    self._make_beam_info(ele_tag)
                    self._make_ele_centers(ele_tag, class_tag)
                elif class_tag in OPS_ELE_TAGS.Link:
                    self._make_link_info(ele_tag)
                    self._make_ele_centers(ele_tag, class_tag)
            else:
                if class_tag in OPS_ELE_TAGS.Plane:
                    self._make_plane_info(ele_tag, class_tag)
                    self._make_ele_centers(ele_tag, class_tag)
                elif class_tag in OPS_ELE_TAGS.Shell:
                    self._make_shell_info(ele_tag, class_tag)
                    self._make_ele_centers(ele_tag, class_tag)
                elif class_tag in OPS_ELE_TAGS.Solid:
                    self._make_solid_info(ele_tag, class_tag)
                    self._make_ele_centers(ele_tag, class_tag)
                elif class_tag in OPS_ELE_TAGS.Joint:
                    self._make_joint_info(ele_tag, class_tag)
                    self._make_ele_centers(ele_tag, class_tag)
            if class_tag in OPS_ELE_TAGS.Contact:
                self._make_contact_info(ele_tag, class_tag)

        # reshape, ensure array alignment, starting with the element with the most nodes
        self.plane_cells = self._reshape_ele_cells(self.plane_cells)
        self.shell_cells = self._reshape_ele_cells(self.shell_cells)
        self.brick_cells = self._reshape_ele_cells(self.brick_cells)
        self.unstru_cells = self._reshape_ele_cells(self.unstru_cells)

    @staticmethod
    def _reshape_ele_cells(cells):
        if len(cells) > 0:
            nums = [len(cell) for cell in cells]
            if len(set(nums)) > 1:
                max_num = np.max(nums)
                cells = [cell + [np.nan] * (max_num - len(cell)) for cell in cells]
            else:
                cells = cells
        else:
            cells = cells
        return cells

    def _make_model_info(self):
        self._make_nodal_info()
        self._make_node_fixed()
        self._make_bounds()
        self._make_nodal_load()
        self._make_ele_load()
        self._make_mp_constraint()
        self._make_ele_info()
        self.MODEL_INFO["NodalCoords"] = np.array(self.node_coords)
        self.MODEL_INFO["EleCenterCoords"] = np.array(self.ele_centers)
        self.MODEL_INFO["NodalDims"] = self.node_ndims
        self.MODEL_INFO["Bounds"] = self.bounds
        self.MODEL_INFO["BoundMaxSize"] = self.max_bound
        self.MODEL_INFO["BoundMinSize"] = self.min_bound
        self.MODEL_INFO["NodeTags"] = self.node_tags
        self.MODEL_INFO["NumNode"] = len(self.node_tags)
        self.MODEL_INFO["FixNodeTags"] = self.fixed_node_tags
        self.MODEL_INFO["FixNodeDofs"] = self.fixed_dofs
        self.MODEL_INFO["FixNodeCoords"] = self.fixed_coords
        self.MODEL_INFO["ConstrainedMidCoords"] = self.mp_centers
        self.MODEL_INFO["ConstrainedDofs"] = self.mp_dofs
        self.MODEL_INFO["ConstrainedCells"] = self.mp_cells
        self.MODEL_INFO["NumEle"] = len(self.ele_tags)
        self.MODEL_INFO["EleTags"] = self.ele_tags
        self.MODEL_INFO["EleClassTags"] = self.ele_class_tags
        self.MODEL_INFO["BeamMidpoints"] = self.beam_centers
        self.MODEL_INFO["BeamLengths"] = self.beam_lengths
        self.MODEL_INFO["BeamXaxis"] = self.beam_xaxis
        self.MODEL_INFO["BeamYaxis"] = self.beam_yaxis
        self.MODEL_INFO["BeamZaxis"] = self.beam_zaxis
        self.MODEL_INFO["LinkMidpoints"] = self.link_centers
        self.MODEL_INFO["LinkLengths"] = self.link_lengths
        self.MODEL_INFO["LinkXaxis"] = self.link_xaxis
        self.MODEL_INFO["LinkYaxis"] = self.link_yaxis
        self.MODEL_INFO["LinkZaxis"] = self.link_zaxis
        self.MODEL_INFO["NodeLoadData"] = self.node_load_data
        self.MODEL_INFO["EleLoadData"] = self.ele_load_data
        # ----------------------------------------------------
        self.MODEL_INFO["TrussEleTags"] = self.truss_tags
        self.MODEL_INFO["TrussCells"] = self.truss_cells
        self.MODEL_INFO["LinkEleTags"] = self.link_tags
        self.MODEL_INFO["LinkCells"] = self.link_cells
        self.MODEL_INFO["BeamEleTags"] = self.beam_tags
        self.MODEL_INFO["BeamCells"] = self.beam_cells
        # self.MODEL_INFO["OtherLineEleTags"] = self.other_line_tags
        self.MODEL_INFO["LineEleTags"] = self.all_line_tags
        self.MODEL_INFO["LineCells"] = self.all_line_cells
        self.MODEL_INFO["PlaneEleTags"] = self.plane_tags
        self.MODEL_INFO["PlaneEleCells"] = self.plane_cells
        self.MODEL_INFO["ShellEleTags"] = self.shell_tags
        self.MODEL_INFO["ShellEleCells"] = self.shell_cells
        self.MODEL_INFO["JointEleTags"] = self.joint_tags
        self.MODEL_INFO["SolidEleTags"] = self.brick_tags
        self.MODEL_INFO["UnstruEleTags"] = self.unstru_tags
        self.MODEL_INFO["UnstruCells"] = self.unstru_cells
        self.MODEL_INFO["UnstruCellTypes"] = self.unstru_cells_type
        # self.MODEL_INFO["BeamSecExtPoints"] = beam_sec_ext_points
        # self.MODEL_INFO["BeamSecIntPoints"] = beam_sec_int_points
        # self.MODEL_INFO["BeamSecPoints"] = beam_sec_points
        # self.MODEL_INFO["ShellThickPoints"] = shell_thick_points
        # ----------------------------------------------------
        self.ELE_CELLS["VTK"] = self.ELE_CELLS_VTK
        self.ELE_CELLS["VTKType"] = self.ELE_CELLS_TYPE_VTK
        self.ELE_CELLS["EleTags"] = self.ELE_CELLS_TAGS
        # self.ELE_CELLS["BeamSecExt"] = beam_sec_ext_cells
        # self.ELE_CELLS["BeamSecInt"] = beam_sec_int_cells
        # self.ELE_CELLS["BeamSec"] = beam_sec_cells
        # self.ELE_CELLS["ShellThick"] = shell_thick_cells

    def get_model_info(self):
        self._make_model_info()
        return self.MODEL_INFO, self.ELE_CELLS
