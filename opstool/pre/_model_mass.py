from collections import defaultdict
from typing import Optional, Union

import numpy as np
import openseespy.opensees as ops


class ModelMass:
    """
    A class used to generate node masses.
    Element mass is distributed equally to each connected node.
    """

    def __init__(self):
        self.node_mass = defaultdict(lambda: 0.0)

    def reset(self):
        self.node_mass = defaultdict(lambda: 0.0)

    def _add_node_mass(self, node_mass: dict):
        for ntag, mass in node_mass.items():
            self.node_mass[ntag] += float(mass)

    def add_node_mass(self, node_tag: int, mass: float):
        """
        Add mass to the existing node mass.

        Parameters
        ----------
        node_tag: int, node tag.
        mass: float, mass value.

        Returns
        -------
        None
        """
        self._add_node_mass({node_tag: mass})

    def add_mass_from_line(
        self,
        ele_tags: Union[list, int],
        rho: Union[list, float],
        area: Union[list, float],
    ):
        """Add mass from two-node line elements.

        Parameters
        ----------
        ele_tags: Union[list, int]
            Element tag or list.
        rho: Union[list, float]
            Mass density or list, if list length shoud same as ele_tags.
        area: Union[list, float]
            Cross-sectional area or list, if list length shoud same as ele_tags.
        """
        ele_tags = np.atleast_1d(ele_tags)
        rhos = np.atleast_1d(rho)
        areas = np.atleast_1d(area)
        if len(rhos) < len(ele_tags):
            rhos = np.zeros_like(ele_tags) + rhos[0]
        if len(areas) < len(ele_tags):
            areas = np.zeros_like(ele_tags) + areas[0]
        for etag, rho, area in zip(ele_tags, rhos, areas):
            node_tags = ops.eleNodes(int(etag))
            if len(node_tags) != 2:
                raise ValueError(f"Element {etag} node number must be 2!")  # noqa: TRY003
            ntag1, ntag2 = node_tags
            coord1 = np.array(ops.nodeCoord(ntag1))
            coord2 = np.array(ops.nodeCoord(ntag2))
            length = np.sqrt(np.sum((coord2 - coord1) ** 2))
            mass = rho * area * length
            node_mass = {ntag1: mass / 2, ntag2: mass / 2}
            self._add_node_mass(node_mass)

    def add_mass_from_surf(self, ele_tags: Union[list, int], rho: Union[list, float], d: Union[list, float]):
        """Add mass from a planar element, including shell.

        Parameters
        ----------
        ele_tags: Union[list, int]
            Element tag or list.
        rho: Union[list, float]
            Mass density or list, if list length shoud same as ele_tags.
        d: Union[list, float]
            Element thickness or list, if list length shoud same as ele_tags.
        """
        ele_tags = np.atleast_1d(ele_tags)
        rhos = np.atleast_1d(rho)
        ds = np.atleast_1d(d)
        if len(rhos) < len(ele_tags):
            rhos = np.zeros_like(ele_tags) + rhos[0]
        if len(ds) < len(ele_tags):
            ds = np.zeros_like(ele_tags) + ds[0]
        for etag, rho, d in zip(ele_tags, rhos, ds):
            node_tags = ops.eleNodes(int(etag))
            node_num = len(node_tags)
            if node_num in [3, 6]:
                node_tags = node_tags[:3]
            elif node_num in [4, 8, 9]:
                node_tags = node_tags[:4]
            else:
                ValueError(f"Ele {etag} is not a valid surf!")
            points = []
            for ntag in node_tags:
                points.append(ops.nodeCoord(ntag))
            area = _PolyArea(points).area
            mass = rho * area * d
            node_mass = {}
            for ntag in node_tags:
                node_mass[ntag] = mass / node_num
            self._add_node_mass(node_mass)

    def add_mass_from_brick(self, ele_tags: Union[list, int], rho: Union[list, float]):
        """Add mass from tetrahedral or hexahedral elements.

        Parameters
        ----------
        ele_tags: Union[list, int]
            Element tag or list.
        rho: Union[list, float]
            Mass density or list, if list length shoud same as ele_tags.
        """
        self.add_mass_from_solid(ele_tags, rho)

    def add_mass_from_solid(self, ele_tags: Union[list, int], rho: Union[list, float]):
        """Add mass from tetrahedral or hexahedral elements.

        Parameters
        ----------
        ele_tags: Union[list, int]
            Element tag or list.
        rho: Union[list, float]
            Mass density or list, if list length shoud same as ele_tags.
        """
        ele_tags = np.atleast_1d(ele_tags)
        rhos = np.atleast_1d(rho)
        if len(rhos) < len(ele_tags):
            rhos = np.zeros_like(ele_tags) + rhos[0]
        for etag, rho in zip(ele_tags, rhos):
            node_tags = ops.eleNodes(int(etag))
            node_num = len(node_tags)
            if node_num in [8, 20, 27]:
                node_tags = node_tags[:8]
                points = []
                for ntag in node_tags:
                    points.append(ops.nodeCoord(ntag))
                vol = _calculate_hexahedron_volume(points)
            elif node_num in [4, 10]:
                node_tags = node_tags[:4]
                points = []
                for ntag in node_tags:
                    points.append(ops.nodeCoord(ntag))
                vol = _calculate_tetrahedron_volume(points)
            else:
                raise ValueError(f"Ele {etag} is not a valid brick!")  # noqa: TRY003
            mass = rho * vol
            node_mass = {}
            for ntag in node_tags:
                node_mass[ntag] = mass / node_num
            self._add_node_mass(node_mass)

    def get_total_mass(self):
        """
        Get the total mass of the model.

        Returns
        -------
        total_mass: float
        """
        total_mass = 0.0
        for mass in self.node_mass.values():
            total_mass += mass
        return total_mass

    @property
    def total_mass(self):
        """
        Returns:
        ---------
        total_mass: float
            The total mass.
        """
        return self.get_total_mass()

    @property
    def nodal_mass(self):
        """
        Returns:
        ---------
        nodal_mass: dict[int, float]
            The nodal mass dict, the key is nodeTag, value is nodal mass.
        """
        return self.get_node_mass()

    def generate_ops_node_mass(self):
        """
        Call the OpenSeesPy node ``mass`` command to generate all node masses.
        The inertia moment of the rotational dof will be ignored.

        .. Note::
            This function will use the ``mass`` command to generate a lumped mass matrix, and does
            not apply it repeatedly with the mass parameters when the element is defined.
            This means that if you use this function, please ignore the mass parameter in the element definition,
            such as the ``'-mass'`` option of some ``beam--column`` elements,
            and the ``density`` parameter in ``nDMaterial``.

        Returns
        -------
        None
        """
        for ntag, mass in self.node_mass.items():
            dim = ops.getNDM(ntag)[0]
            dof = ops.getNDF(ntag)[0]
            if dim == 1:
                ops.mass(ntag, mass)
            elif dim == 2 and dof == 2:
                ops.mass(ntag, mass, mass)
            elif dim == 2 and dof == 3:
                ops.mass(ntag, mass, mass, 0.0)
            elif dim == 3 and dof == 3:
                ops.mass(ntag, mass, mass, mass)
            elif dim == 3 and dof == 6:
                ops.mass(ntag, mass, mass, mass, 0.0, 0.0, 0.0)

    def generate_ops_gravity_load(self, direction: str, factor: float = -9.81, exclude_nodes: Optional[list] = None):
        """
        Call the OpenSeesPy ``load`` command to generate a nodal gravity load.

        Parameters
        ----------
        direction: str,
            The gravity load direction.
        factor: float, default=-9.81
            The factor applied to the mass values, it should be the multiplication of gravitational acceleration
            and directional indicators, e.g., -9.81, where 9.81 is the gravitational acceleration
            and -1 indicates along the negative Z axis.
            Of course, it can be multiplied by an additional factor to account for additional constant loads,
            e.g., 1.05 * (-9.81).
        exclude_nodes: list, default=None
            Excluded node tags, whose masses will not be used to generate gravity loads.

        Returns
        -------
        None

        """
        direction = direction.upper()
        if exclude_nodes is not None:
            node_mass = {ntag: mass for ntag, mass in self.node_mass.items() if ntag not in exclude_nodes}
        else:
            node_mass = self.node_mass
        load_fact_3d6 = {
            "Z": np.array([0.0, 0.0, factor, 0.0, 0.0, 0.0]),
            "Y": np.array([0.0, factor, 0.0, 0.0, 0.0, 0.0]),
            "X": np.array([factor, 0.0, 0.0, 0.0, 0.0, 0.0]),
        }
        load_fact_3d3 = {
            "Z": np.array([0.0, 0.0, factor]),
            "Y": np.array([0.0, factor, 0]),
            "X": np.array([factor, 0.0, 0.0]),
        }
        load_fact_2d3 = {
            "Y": np.array([0.0, factor, 0.0]),
            "X": np.array([factor, 0.0, 0.0]),
        }
        load_fact_2d2 = {
            "Y": np.array([0.0, factor]),
            "X": np.array([factor, 0.0]),
        }
        load_fact_1d = {"X": np.array([factor])}
        load_fact = {
            (3, 6): load_fact_3d6,
            (3, 3): load_fact_3d3,
            (2, 3): load_fact_2d3,
            (2, 2): load_fact_2d2,
            (1, 1): load_fact_1d,
        }
        for ntag, mass in node_mass.items():
            dim = ops.getNDM(ntag)[0]
            dof = ops.getNDF(ntag)[0]
            loadValues = list(mass * load_fact[(dim, dof)][direction])
            ops.load(ntag, *loadValues)

    def get_node_mass(self, node_tags: Optional[list] = None):
        """Get nodal mass.

        Parameters
        ----------
        node_tags: list, optional, default=None
            If None, return all node masses, else return node mass in node_tags.

        Returns
        -------
        node_mass: dict
            A dict obj whose keys are node tags and whose values are masses.
        """
        if node_tags is None:
            return dict(self.node_mass)
        else:
            node_mass = {}
            for ntag in node_tags:
                node_mass[ntag] = self.node_mass[ntag]
            return node_mass


class _PolyArea:
    def __init__(self, points: list):
        if len(points[0]) == 2:
            for i in range(len(points)):
                points[i] += [0.0]
        self.points = points

    @staticmethod
    def det(a):
        # determinant of matrix a
        temp = a[0][0] * a[1][1] * a[2][2] + a[0][1] * a[1][2] * a[2][0] + a[0][2] * a[1][0] * a[2][1]
        temp += -a[0][2] * a[1][1] * a[2][0] - a[0][1] * a[1][0] * a[2][2] - a[0][0] * a[1][2] * a[2][1]
        return temp

    def unit_normal(self, a, b, c):
        # unit normal vector of plane defined by points a, b, and c
        x = self.det([[1, a[1], a[2]], [1, b[1], b[2]], [1, c[1], c[2]]])
        y = self.det([[a[0], 1, a[2]], [b[0], 1, b[2]], [c[0], 1, c[2]]])
        z = self.det([[a[0], a[1], 1], [b[0], b[1], 1], [c[0], c[1], 1]])
        magnitude = (x**2 + y**2 + z**2) ** 0.5
        return x / magnitude, y / magnitude, z / magnitude

    @staticmethod
    def dot(a, b):
        # dot product of vectors a and b
        return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]

    @staticmethod
    def cross(a, b):
        # cross product of vectors a and b
        x = a[1] * b[2] - a[2] * b[1]
        y = a[2] * b[0] - a[0] * b[2]
        z = a[0] * b[1] - a[1] * b[0]
        return x, y, z

    @property
    def area(self):
        if len(self.points) < 3:  # not a plane - no area
            return 0.0
        # area of polygon poly
        total = [0, 0, 0]
        for i in range(len(self.points)):
            vi1 = self.points[i]
            vi2 = self.points[0] if i is len(self.points) - 1 else self.points[i + 1]
            prod = self.cross(vi1, vi2)
            total[0] += prod[0]
            total[1] += prod[1]
            total[2] += prod[2]
        result = self.dot(total, self.unit_normal(self.points[0], self.points[1], self.points[2]))
        return abs(result / 2)


def _calculate_tetrahedron_volume(vertices: list):
    vertices = np.array(vertices)
    if vertices.shape != (4, 3):
        raise ValueError("shape must be (4, 3)!")  # noqa: TRY003

    # Jacobian matrix
    B = np.array([
        [
            vertices[1, 0] - vertices[0, 0],
            vertices[2, 0] - vertices[0, 0],
            vertices[3, 0] - vertices[0, 0],
        ],
        [
            vertices[1, 1] - vertices[0, 1],
            vertices[2, 1] - vertices[0, 1],
            vertices[3, 1] - vertices[0, 1],
        ],
        [
            vertices[1, 2] - vertices[0, 2],
            vertices[2, 2] - vertices[0, 2],
            vertices[3, 2] - vertices[0, 2],
        ],
    ])
    det_B = np.linalg.det(B)
    volume = np.abs(det_B) / 6.0

    return volume


def _calculate_hexahedron_volume(vertices):
    vertices = np.array(vertices)
    if vertices.shape != (8, 3):
        raise ValueError("The shape of the input array must be (8, 3).")  # noqa: TRY003

    # Calculate the Jacobian matrix B
    B = np.array([
        [
            vertices[1, 0] - vertices[0, 0],
            vertices[2, 0] - vertices[0, 0],
            vertices[4, 0] - vertices[0, 0],
        ],
        [
            vertices[1, 1] - vertices[0, 1],
            vertices[2, 1] - vertices[0, 1],
            vertices[4, 1] - vertices[0, 1],
        ],
        [
            vertices[1, 2] - vertices[0, 2],
            vertices[2, 2] - vertices[0, 2],
            vertices[4, 2] - vertices[0, 2],
        ],
    ])

    # Calculate the determinant of the Jacobian matrix
    det_B = np.linalg.det(B)

    # Calculate the volume of the hexahedron
    volume = np.abs(det_B)

    return volume
