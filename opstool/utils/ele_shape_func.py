# https://github.com/febiosoftware/FEBio/blob/3f7516d5866694657fdb1be845397612e3641dc4/FECore/FEElementShape.cpp
# https://github.com/febiosoftware/FEBio/blob/3f7516d5866694657fdb1be845397612e3641dc4/FECore/FEElementTraits.cpp#L400

import numpy as np


def get_shape_func(ele_type: str, n: int, gp: int) -> object:
    """Get shape function by element type, num nodes, and num Gauss points.

    Parameters
    ----------
    ele_type: str
        Element type, e.g., "tri", "quad", "tet", "brick".
    n: int
        Number of nodes.
        For triangles: 3 or 6.
        For quadrilaterals: 4, 8, or 9.
        For tetrahedrons: 4 or 10.
        For bricks: 8, 20, or 27.
    gp: int
        Number of Gauss points.

    Returns
    -------
    shape_func: function object.
        For 2D elements, it takes two arguments (r, s).
        For 3D elements, it takes three arguments (r, s, t).
        r, s, t are normalized coordinates in the element.
        For triangles, r and s are in the range [0, 1].
        For quadrilaterals, r and s are in the range [-1, 1].
        For tetrahedrons, r, s, and t are in the range [0, 1].
        For bricks, r, s, and t are in the range [-1, 1].

    """
    ele_type = ele_type.lower()

    shape_func_map = {
        ("tri", 3, 1): FEtriN3GP1,
        ("tri", 6, 3): FEtriN6GP3,
        ("quad", 4, 4): FEquadN4GP4,
        ("quad", 9, 9): FEquadN9GP9,
        ("quad", 8, 9): FEquadN8GP9,
        ("tet", 4, 1): FEtetN4GP1,
        ("tet", 10, 4): FEtetN10GP4,
        ("brick", 8, 8): FEbrickN8GP8,
        ("brick", 27, 27): FEbrickN27GP27,
        ("brick", 20, 27): FEbrickN20GP27,
    }

    cls = shape_func_map.get((ele_type, n, gp))
    return cls().shape_func if cls else None


def get_gp2node_func(ele_type: str, n: int, gp: int) -> object:
    """Get Gauss points response to nodes projection function by element type, num nodes, and num Gauss points.

    Parameters
    ----------
    ele_type: str
        Element type, e.g., "tri", "quad", "tet", "brick".
    n: int
        Number of nodes.
        For triangles: 3 or 6.
        For quadrilaterals: 4, 8, or 9.
        For tetrahedrons: 4 or 10.
        For bricks: 8, 20, or 27.
    gp: int
        Number of Gauss points.

    Returns
    -------
    weight_func: function object.
        Function to project Gauss point responses to nodes.
        It takes two arguments (method, gp_resp).

        method: str
            Method to project Gauss point responses to nodes.
            * If "extrapolate", use extrapolation method by element shape function.
            * If "average", use weighted averaging method, weight is the Gauss point weight.
            * If "copy", use copy method, copy the nearest Gauss point responses to nodes.

        gp_resp: np.ndarray
            Gauss point responses, shape[n, m], where n is the number of Gauss points,
            m is the number of responses at each Gauss point.
    """
    ele_type = ele_type.lower()

    weight_map = {
        ("tri", 3, 1): FEtriN3GP1(),
        ("tri", 6, 3): FEtriN6GP3(),
        ("quad", 4, 4): FEquadN4GP4(),
        ("quad", 9, 9): FEquadN9GP9(),
        ("quad", 8, 9): FEquadN8GP9(),
        ("tet", 4, 1): FEtetN4GP1(),
        ("tet", 10, 4): FEtetN10GP4(),
        ("brick", 8, 8): FEbrickN8GP8(),
        ("brick", 27, 27): FEbrickN27GP27(),
        ("brick", 20, 27): FEbrickN20GP27(),
    }

    cls = weight_map.get((ele_type, n, gp))
    return cls.project_to_nodes if cls else None


def get_shell_gp2node_func(ele_type: str, n: int, gp: int) -> object:
    """Get Gauss points response to nodes projection function by element type, num nodes, and num Gauss points.

    * ("tri", 3, 1) for 3-node triangle with 1 Gauss point, ASDShellT3.
    * ("tri", 3, 3) for 3-node triangle with 3 Gauss points, ASDShellT3.
    * ("tri", 3, 4) for 3-node triangle with 4 Gauss points, ShellDKGT and ShellNLDKGT.
    * ("quad", 4, 4) for 4-node quadrilateral with 4 Gauss points, ASDShellQ4, ShellDKGQ, ShellNLDKGQ and ShellMITC4.
    * ("quad", 9, 9) for 9-node quadrilateral with 9 Gauss points, ShellMITC9.

    Parameters
    ----------
    ele_type: str
        Element type, e.g., "tri", "quad"
    n: int
        Number of nodes.
        For triangles: 3.
        For quadrilaterals: 4, or 9.
    gp: int
        Number of Gauss points.
        For triangles: 1, 3, or 4.
        For quadrilaterals: 4 or 9.

    Returns
    -------
    weight_func: function object.
        Function to project Gauss point responses to nodes.
        It takes two arguments (method, gp_resp).

        method: str
            Method to project Gauss point responses to nodes.
            * If "extrapolate", use extrapolation method by element shape function.
            * If "average", use weighted averaging method, weight is the Gauss point weight.
            * If "copy", use copy method, copy the nearest Gauss point responses to nodes.

        gp_resp: np.ndarray
            Gauss point responses, shape[n, m], where n is the number of Gauss points,
            m is the number of responses at each Gauss point.
    """
    ele_type = ele_type.lower()

    weight_map = {
        ("tri", 3, 1): FEshellN3GP1(),
        ("tri", 3, 3): FEshellN3GP3(),
        ("tri", 3, 4): FEshellN3GP4(),
        ("quad", 4, 4): FEshellN4GP4(),
        ("quad", 9, 9): FEshellN9GP9(),
    }

    cls = weight_map.get((ele_type, n, gp))
    return cls.project_to_nodes if cls else None


class FEtriN3GP1:
    """Topology:

    ↑ s

    3
    + +
    +  +
    +   +
    +    +
    1-----2  --> r
    """

    def __init__(self):
        """Initialize FEtri class."""
        self.gp_rs = np.array([1 / 3])
        self.gp_ss = np.array([1 / 3])
        self.gp_wts = np.array([0.5])

        self.num_nodes = 3

    def project_to_nodes(self, method: str, gp_resp: np.ndarray) -> np.ndarray:
        weights = np.array([[1.0], [1.0], [1.0]])
        if gp_resp.ndim == 2:
            return np.dot(weights, gp_resp)
        elif gp_resp.ndim == 3:
            return np.einsum("ng,gfr->nfr", weights, gp_resp)
        else:
            raise ValueError("gp_resp must be 2D or 3D array.")  # noqa: TRY003

    @staticmethod
    def shape_func(r: float, s: float) -> np.ndarray:
        """Triangular shape function for 3-node triangle element.

        Parameters
        ----------
        r, s: normalized coordinates in the triangle, 0--1

        Returns
        -------
        N: shape function values at the given r, s coordinates.
        """
        n1 = 1 - r - s
        n2 = r
        n3 = s
        return np.array([n1, n2, n3])


class FEtriN6GP3:
    """Topology:

    ↑ s

    3
    + +
    +  +
    6   5
    +    +
    +     +
    1--4---2  --> r
    """

    def __init__(self):
        a, b = 1 / 6, 2 / 3
        self.gp_rs = np.array([a, b, a])
        self.gp_ss = np.array([a, a, b])
        self.gp_wts = np.array([a, a, a])

        self.node_rs = np.array([0.0, 1.0, 0.0, 0.5, 0.5, 0.0])
        self.node_ss = np.array([0.0, 0.0, 1.0, 0.0, 0.5, 0.5])
        # rs = [-1 / 3, 5 / 3, -1 / 3, 2 / 3, 2 / 3, -1 / 3]
        # ss = [-1 / 3, -1 / 3, 5 / 3, -1 / 3, 2 / 3, 2 / 3]

        self.num_nodes = 6

    def project_to_nodes(self, method: str, gp_resp: np.ndarray) -> np.ndarray:
        """Project Gauss point responses to nodes.

        Parameters
        ----------
        method: str
            Method to project Gauss point responses to nodes.
            If "extrapolate", use extrapolation method.
            If "average", use weighted averaging method, weight is the Gauss point weight.
            If "copy", use copy method, copy Gauss point responses to nodes.
        gp_resp: np.ndarray
            Gauss point responses, shape[n, m], where n is the number of Gauss points,
            m is the number of responses at each Gauss point.

        Returns
        -------
        np.ndarray
            Projected responses at nodes, shape[6, m], where 6 is the number of nodes.
        """
        method = method.lower()
        if method.startswith("extra"):
            weights = np.array([
                [1.66666667, -0.33333333, -0.33333333],  # Node 1
                [-0.33333333, 1.66666667, -0.33333333],  # Node 2
                [-0.33333333, -0.33333333, 1.66666667],  # Node 3
                [0.66666667, 0.66666667, -0.33333333],  # Node 4
                [-0.33333333, 0.66666667, 0.66666667],  # Node 5
                [0.66666667, -0.33333333, 0.66666667],  # Node 6
            ])
        elif method.startswith("ave"):
            weights = np.tile(self.gp_wts, (self.num_nodes, 1)) / np.sum(self.gp_wts)
        elif method.startswith("copy"):
            weights = np.array([
                [1.0, 0.0, 0.0],  # N1 ← G1
                [0.0, 1.0, 0.0],  # N2 ← G2
                [0.0, 0.0, 1.0],  # N3 ← G3
                [0.5, 0.5, 0.0],  # N4 ← G1-G2 Midpoint
                [0.0, 0.5, 0.5],  # N5 ← G2-G3 Midpoint
                [0.5, 0.0, 0.5],  # N6 ← G3-G1 Midpoint
            ])
        if gp_resp.ndim == 2:
            return np.dot(weights, gp_resp)
        elif gp_resp.ndim == 3:
            return np.einsum("ng,gfr->nfr", weights, gp_resp)
        else:
            raise ValueError("gp_resp must be 2D or 3D array.")  # noqa: TRY003

    @staticmethod
    def shape_func(r: float, s: float) -> np.ndarray:
        """Triangular shape function for 6-node triangle element.

        Parameters
        ----------
        r, s: normalized coordinates in the triangle, 0--1

        Returns
        -------
        N: shape function values at the given r, s coordinates.
        """
        t = 1 - r - s
        N = np.array([
            t * (2 * t - 1),  # N1
            r * (2 * r - 1),  # N2
            s * (2 * s - 1),  # N3
            4 * r * t,  # N4
            4 * r * s,  # N5
            4 * s * t,  # N6
        ])
        return N


class FEquadN4GP4:
    """
    |      s ↑
    |
    |      4 -------- 3
    |      |          |
    |      |          |
    |      1 -------- 2       → r
    """

    def __init__(self):
        a = 1 / np.sqrt(3)
        self.gp_rs = np.array([-a, a, a, -a])
        self.gp_ss = np.array([-a, -a, a, a])
        self.gp_wts = np.array([1, 1, 1, 1])

        self.node_rs = np.array([-1, 1, 1, -1])
        self.node_ss = np.array([-1, -1, 1, 1])
        # node_rs = [-np.sqrt(3), np.sqrt(3), np.sqrt(3), -np.sqrt(3)]
        # node_ss = [-np.sqrt(3), -np.sqrt(3), np.sqrt(3), np.sqrt(3)]

        self.num_nodes = 4

    def project_to_nodes(self, method: str, gp_resp: np.ndarray) -> np.ndarray:
        """Project Gauss point responses to nodes.

        Parameters
        ----------
        method: str
            Method to project Gauss point responses to nodes.
            If "extrapolate", use extrapolation method.
            If "average", use weighted averaging method, weight is the Gauss point weight.
            If "copy", use copy method, copy Gauss point responses to nodes.
        gp_resp: np.ndarray
            Gauss point responses, shape[n, m], where n is the number of Gauss points,
            m is the number of responses at each Gauss point.

        Returns
        -------
        np.ndarray
            Projected responses at nodes, shape[4, m], where 4 is the number of nodes.
        """
        method = method.lower()
        if method.startswith("extra"):
            weights = np.array([
                [1.8660254, -0.5, 0.1339746, -0.5],
                [-0.5, 1.8660254, -0.5, 0.1339746],
                [0.1339746, -0.5, 1.8660254, -0.5],
                [-0.5, 0.1339746, -0.5, 1.8660254],
            ])
        elif method.startswith("ave"):
            weights = np.tile(self.gp_wts / np.sum(self.gp_wts), (self.num_nodes, 1))
        elif method.startswith("copy"):
            weights = np.array([
                [1.0, 0.0, 0.0, 0.0],  # N1 ← G1
                [0.0, 1.0, 0.0, 0.0],  # N2 ← G2
                [0.0, 0.0, 1.0, 0.0],  # N3 ← G3
                [0.0, 0.0, 0.0, 1.0],  # N4 ← G4
            ])
        if gp_resp.ndim == 2:
            return np.dot(weights, gp_resp)
        elif gp_resp.ndim == 3:
            return np.einsum("ng,gfr->nfr", weights, gp_resp)
        else:
            raise ValueError("gp_resp must be 2D or 3D array.")  # noqa: TRY003

    @staticmethod
    def shape_func(r: float, s: float) -> np.ndarray:
        """
        Quadrilateral shape function for 4-node quad element.

        Parameters
        ----------
        r, s: normalized coordinates in the quad, -1 <= r, s <= 1

        Returns
        -------
        N: shape function values at the given r, s coordinates.
        """
        N1 = 0.25 * (1 - r) * (1 - s)
        N2 = 0.25 * (1 + r) * (1 - s)
        N3 = 0.25 * (1 + r) * (1 + s)
        N4 = 0.25 * (1 - r) * (1 + s)
        return np.array([N1, N2, N3, N4])


class FEquadN9GP9:
    """9-node quadrilateral element with 9 Gauss points.

    |              s ↑
    |
    |   4 —— 7 —— 3
    |   |         |
    |   |         |
    |   8 —— 9 —— 6
    |   |         |
    |   |         |
    |   1 —— 5 —— 2      → r
    """

    def __init__(self):
        a = np.sqrt(0.6)
        w1 = 25.0 / 81.0
        w2 = 40.0 / 81.0
        w3 = 64.0 / 81.0
        self.gp_rs = np.array([-a, a, a, -a, 0, a, 0, -a, 0])
        self.gp_ss = np.array([-a, -a, a, a, -a, 0, a, 0, 0])
        self.gp_wts = np.array([w1, w1, w1, w1, w2, w2, w2, w2, w3])

        self.node_rs = np.array([-1, 1, 1, -1, 0, 1, 0, -1, 0])
        self.node_ss = np.array([-1, -1, 1, 1, -1, 0, 1, 0, 0])
        # node_rs = [-sqrt53, sqrt53, sqrt53, -sqrt53, 0.0, sqrt53, 0.0, -sqrt53, 0.0]
        # node_ss = [-sqrt53, -sqrt53, sqrt53, sqrt53, -sqrt53, 0.0, sqrt53, 0.0, 0.0]

        self.num_nodes = 9

    def project_to_nodes(self, method: str, gp_resp: np.ndarray) -> np.ndarray:
        """Project Gauss point responses to nodes.

        Parameters
        ----------
        method: str
            Method to project Gauss point responses to nodes.
            If "extrapolate", use extrapolation method.
            If "average", use weighted averaging method, weight is the Gauss point weight.
            If "copy", use copy method, copy Gauss point responses to nodes.
        gp_resp: np.ndarray
            Gauss point responses, shape[n, m], where n is the number of Gauss points,
            m is the number of responses at each Gauss point.

        Returns
        -------
        np.ndarray
            Projected responses at nodes, shape[4, m], where 4 is the number of nodes.
        """
        method = method.lower()
        if method.startswith("extra"):
            weights = np.array([
                [
                    2.18693982,
                    0.27777778,
                    0.0352824,
                    0.27777778,
                    -0.98588704,
                    -0.12522407,
                    -0.12522407,
                    -0.98588704,
                    0.44444444,
                ],
                [
                    0.27777778,
                    2.18693982,
                    0.27777778,
                    0.0352824,
                    -0.98588704,
                    -0.98588704,
                    -0.12522407,
                    -0.12522407,
                    0.44444444,
                ],
                [
                    0.0352824,
                    0.27777778,
                    2.18693982,
                    0.27777778,
                    -0.12522407,
                    -0.98588704,
                    -0.98588704,
                    -0.12522407,
                    0.44444444,
                ],
                [
                    0.27777778,
                    0.0352824,
                    0.27777778,
                    2.18693982,
                    -0.12522407,
                    -0.12522407,
                    -0.98588704,
                    -0.98588704,
                    0.44444444,
                ],
                [-0.0, 0.0, 0.0, -0.0, 1.47883056, -0.0, 0.18783611, 0.0, -0.66666667],
                [-0.0, -0.0, 0.0, 0.0, 0.0, 1.47883056, -0.0, 0.18783611, -0.66666667],
                [-0.0, 0.0, 0.0, -0.0, 0.18783611, -0.0, 1.47883056, 0.0, -0.66666667],
                [-0.0, -0.0, 0.0, 0.0, 0.0, 0.18783611, -0.0, 1.47883056, -0.66666667],
                [0.0, -0.0, 0.0, -0.0, -0.0, 0.0, 0.0, -0.0, 1.0],
            ])
        elif method.startswith("ave"):
            weights = np.tile(self.gp_wts / np.sum(self.gp_wts), (self.num_nodes, 1))
        elif method.startswith("copy"):
            weights = np.eye(self.num_nodes)
        if gp_resp.ndim == 2:
            return np.dot(weights, gp_resp)
        elif gp_resp.ndim == 3:
            return np.einsum("ng,gfr->nfr", weights, gp_resp)
        else:
            raise ValueError("gp_resp must be 2D or 3D array.")  # noqa: TRY003

    @staticmethod
    def shape_func(r: float, s: float) -> np.ndarray:
        """Quadrilateral shape function for 9-node quad element.

        Parameters
        ----------
        r, s: normalized coordinates in the quad, -1 <= r, s <= 1

        Returns
        -------
        N: shape function values at the given r, s coordinates.
        """
        n1 = 0.25 * r * s * (1 - r) * (1 - s)
        n2 = -0.25 * r * s * (1 + r) * (1 - s)
        n3 = 0.25 * r * s * (1 + r) * (1 + s)
        n4 = -0.25 * r * s * (1 - r) * (1 + s)

        n5 = -0.5 * s * (1 - r * r) * (1 - s)
        n6 = 0.5 * r * (1 + r) * (1 - s * s)
        n7 = 0.5 * s * (1 - r * r) * (1 + s)
        n8 = -0.5 * r * (1 - r) * (1 - s * s)

        n9 = (1 - r * r) * (1 - s * s)
        return np.array([n1, n2, n3, n4, n5, n6, n7, n8, n9])


class FEquadN8GP9:
    """8-node quadrilateral element with 9 Gauss points.

    |             s ↑
    |
    |   4 —— 7 —— 3
    |   |         |
    |   |         |
    |   8         6
    |   |         |
    |   |         |
    |   1 —— 5 —— 2      → r
    """

    def __init__(self):
        a = np.sqrt(0.6)
        w1 = 25.0 / 81.0
        w2 = 40.0 / 81.0
        w3 = 64.0 / 81.0
        self.gp_rs = np.array([-a, a, a, -a, 0, a, 0, -a, 0])
        self.gp_ss = np.array([-a, -a, a, a, -a, 0, a, 0, 0])
        self.gp_wts = np.array([w1, w1, w1, w1, w2, w2, w2, w2, w3])

        self.node_rs = np.array([-1, 1, 1, -1, 0, 1, 0, -1])
        self.node_ss = np.array([-1, -1, 1, 1, -1, 0, 1, 0])
        # node_rs = [-sqrt53, sqrt53, sqrt53, -sqrt53, 0.0, sqrt53, 0.0, -sqrt53]
        # node_ss = [-sqrt53, -sqrt53, sqrt53, sqrt53, -sqrt53, 0.0, sqrt53, 0.0]

        self.num_nodes = 8

    def project_to_nodes(self, method: str, gp_resp: np.ndarray) -> np.ndarray:
        """Project Gauss point responses to nodes.

        Parameters
        ----------
        method: str
            Method to project Gauss point responses to nodes.
            If "extrapolate", use extrapolation method.
            If "average", use weighted averaging method, weight is the Gauss point weight.
            If "copy", use copy method, copy Gauss point responses to nodes.
        gp_resp: np.ndarray
            Gauss point responses, shape[n, m], where n is the number of Gauss points,
            m is the number of responses at each Gauss point.

        Returns
        -------
        np.ndarray
            Projected responses at nodes, shape[4, m], where 4 is the number of nodes.
        """
        method = method.lower()
        if method.startswith("extra"):
            weights = np.array([
                [
                    2.18693982,
                    0.27777778,
                    0.0352824,
                    0.27777778,
                    -0.98588704,
                    -0.12522407,
                    -0.12522407,
                    -0.98588704,
                    0.44444444,
                ],
                [
                    0.27777778,
                    2.18693982,
                    0.27777778,
                    0.0352824,
                    -0.98588704,
                    -0.98588704,
                    -0.12522407,
                    -0.12522407,
                    0.44444444,
                ],
                [
                    0.0352824,
                    0.27777778,
                    2.18693982,
                    0.27777778,
                    -0.12522407,
                    -0.98588704,
                    -0.98588704,
                    -0.12522407,
                    0.44444444,
                ],
                [
                    0.27777778,
                    0.0352824,
                    0.27777778,
                    2.18693982,
                    -0.12522407,
                    -0.12522407,
                    -0.98588704,
                    -0.98588704,
                    0.44444444,
                ],
                [-0.0, 0.0, 0.0, -0.0, 1.47883056, -0.0, 0.18783611, 0.0, -0.66666667],
                [-0.0, -0.0, 0.0, 0.0, 0.0, 1.47883056, -0.0, 0.18783611, -0.66666667],
                [-0.0, 0.0, 0.0, -0.0, 0.18783611, -0.0, 1.47883056, 0.0, -0.66666667],
                [-0.0, -0.0, 0.0, 0.0, 0.0, 0.18783611, -0.0, 1.47883056, -0.66666667],
            ])
        elif method.startswith("ave"):
            weights = np.tile(self.gp_wts / np.sum(self.gp_wts), (self.num_nodes, 1))
        elif method.startswith("copy"):
            weights = np.eye(9)[: self.num_nodes]
        if gp_resp.ndim == 2:
            return np.dot(weights, gp_resp)
        elif gp_resp.ndim == 3:
            return np.einsum("ng,gfr->nfr", weights, gp_resp)
        else:
            raise ValueError("gp_resp must be 2D or 3D array.")  # noqa: TRY003

    @staticmethod
    def shape_func(r: float, s: float) -> np.ndarray:
        """Quadrilateral shape function for 9-node quad element.

        Parameters
        ----------
        r, s: normalized coordinates in the quad, -1 <= r, s <= 1

        Returns
        -------
        N: shape function values at the given r, s coordinates.
        """
        n1 = 0.25 * (1 - r) * (1 - s) * (-r - s - 1)
        n2 = 0.25 * (1 + r) * (1 - s) * (r - s - 1)
        n3 = 0.25 * (1 + r) * (1 + s) * (r + s - 1)
        n4 = 0.25 * (1 - r) * (1 + s) * (-r + s - 1)

        n5 = 0.5 * (1 - r * r) * (1 - s)
        n6 = 0.5 * (1 + r) * (1 - s * s)
        n7 = 0.5 * (1 - r * r) * (1 + s)
        n8 = 0.5 * (1 - r) * (1 - s * s)

        return np.array([n1, n2, n3, n4, n5, n6, n7, n8])


class FEtetN4GP1:
    """4-node tetrahedral element with 1 Gauss point.

    Topology:

                            t
                            .
                        ,/
                        /
                    2
                    ,/|`\
                ,/  |  `\
                ,/    '.   `\\
            ,/       |     `\\
            ,/         |       `\
            0-----------'.--------1 --> r
            `\\.         |      ,/
                `\\.      |    ,/9
                `\\.   '. ,/
                    `\\. |/
                        `3
                            `\\.
                            ` s
    """

    def __init__(self):
        a = 0.25
        w = 0.61
        self.gp_rs = np.array([a])
        self.gp_ss = np.array([a])
        self.gp_ts = np.array([a])
        self.gp_wts = np.array([w])

        self.node_rs = np.array([0, 1, 0, 0])
        self.node_ss = np.array([0, 0, 1, 0])
        # node_rs = [-sqrt53, sqrt53, sqrt53, -sqrt53, 0.0, sqrt53, 0.0, -sqrt53]
        # node_ss = [-sqrt53, -sqrt53, sqrt53, sqrt53, -sqrt53, 0.0, sqrt53, 0.0]

        self.num_nodes = 4

    def project_to_nodes(self, method: str, gp_resp: np.ndarray) -> np.ndarray:
        """Project Gauss point responses to nodes.

        Parameters
        ----------
        method: str
            Method to project Gauss point responses to nodes.
            If "extrapolate", use extrapolation method.
            If "average", use weighted averaging method, weight is the Gauss point weight.
            If "copy", use copy method, copy Gauss point responses to nodes.
        gp_resp: np.ndarray
            Gauss point responses, shape[n, m], where n is the number of Gauss points,
            m is the number of responses at each Gauss point.

        Returns
        -------
        np.ndarray
            Projected responses at nodes, shape[4, m], where 4 is the number of nodes.
        """
        weights = np.array([[1.0], [1.0], [1.0], [1.0]])
        if gp_resp.ndim == 2:
            return np.dot(weights, gp_resp)
        elif gp_resp.ndim == 3:
            return np.einsum("ng,gfr->nfr", weights, gp_resp)
        else:
            raise ValueError("gp_resp must be 2D or 3D array.")  # noqa: TRY003

    @staticmethod
    def shape_func(r: float, s: float, t: float) -> np.ndarray:
        """Tetrahedral shape function for 4-node tet element.

        Parameters
        ----------
        r, s, t: normalized coordinates in the tetrahedron, 0 <= r, s, t <= 1

        Returns
        -------
        N: shape function values at the given r, s, t coordinates.
        """
        n1 = 1 - r - s - t
        n2 = r
        n3 = s
        n4 = t
        return np.array([n1, n2, n3, n4])


class FEtetN10GP4:
    """10-node tetrahedral element with 4 Gauss points.

    * Edge1: 1--5--2
    * Edge2: 2--6--3
    * Edge3: 3--7--1
    * Edge4: 1--8--4
    * Edge5: 2--9--4
    * Edge6: 3--10--4
    """

    def __init__(self):
        a = 0.5854101966249685
        b = 0.1381966011250105
        w = 0.25 / 6.0
        self.gp_rs = np.array([a, b, b, b])
        self.gp_ss = np.array([b, a, b, b])
        self.gp_ts = np.array([b, b, a, b])
        self.gp_wts = np.array([w, w, w, w])

        self.num_nodes = 10

    def project_to_nodes(self, method: str, gp_resp: np.ndarray) -> np.ndarray:
        """Project Gauss point responses to nodes.

        Parameters
        ----------
        method: str
            Method to project Gauss point responses to nodes.
            If "extrapolate", use extrapolation method.
            If "average", use weighted averaging method, weight is the Gauss point weight.
            If "copy", use copy method, copy Gauss point responses to nodes.
        gp_resp: np.ndarray
            Gauss point responses, shape[n, m], where n is the number of Gauss points,
            m is the number of responses at each Gauss point.

        Returns
        -------
        np.ndarray
            Projected responses at nodes, shape[4, m], where 4 is the number of nodes.
        """
        method = method.lower()
        if method.startswith("extra"):
            weights = np.array([
                [-0.309017, -0.309017, -0.309017, 1.927051],
                [1.927051, -0.309017, -0.309017, -0.309017],
                [-0.309017, 1.927051, -0.309017, -0.309017],
                [-0.309017, -0.309017, 1.927051, -0.309017],
                [0.809017, 0.809017, -0.309017, -0.309017],
                [0.809017, -0.309017, 0.809017, -0.309017],
                [0.809017, -0.309017, -0.309017, 0.809017],
                [-0.309017, -0.309017, 0.809017, 0.809017],
                [-0.309017, 0.809017, -0.309017, 0.809017],
                [-0.30932, 0.43644, 0.43644, 0.43644],
            ])
        elif method.startswith("ave"):
            weights = np.tile(self.gp_wts / np.sum(self.gp_wts), (self.num_nodes, 1))
        elif method.startswith("copy"):
            weights = np.array([
                [0.0, 0.0, 0.0, 1.0],
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [1.0, 0.0, 0.0, 0.0],
            ])
        if gp_resp.ndim == 2:
            return np.dot(weights, gp_resp)
        elif gp_resp.ndim == 3:
            return np.einsum("ng,gfr->nfr", weights, gp_resp)
        else:
            raise ValueError("gp_resp must be 2D or 3D array.")  # noqa: TRY003

    @staticmethod
    def shape_func(r: float, s: float, t: float) -> np.ndarray:
        """Tetrahedral shape function for 10-node tet element.

        Parameters
        ----------
        r, s, t: normalized coordinates in the 10-nodes tetrahedron, 0 <= r, s, t <= 1

        Returns
        -------
        N: shape function values at the given r, s, t coordinates.
        """
        u = 1 - r - s - t
        N = np.zeros(10)
        N[0] = u * (2 * u - 1)
        N[1] = r * (2 * r - 1)
        N[2] = s * (2 * s - 1)
        N[3] = t * (2 * t - 1)
        N[4] = 4 * r * u
        N[5] = 4 * r * s
        N[6] = 4 * s * u
        N[7] = 4 * t * u
        N[8] = 4 * r * t
        N[9] = 4 * s * t
        return N


class FEbrickN8GP8:
    """8-node brick element with 8 Gauss points.

    Topology:

            t ↑
         |
         8--------7
        /|       /|
        5--------6 |
        | |      | |
        | 4------|-3      → r
        |/       |/
        1--------2
    → s
    """

    def __init__(self):
        a = 1.0 / np.sqrt(3.0)
        w = 1.0
        self.gp_rs = np.array([-a, a, a, -a, -a, a, a, -a])
        self.gp_ss = np.array([-a, -a, a, a, -a, -a, a, a])
        self.gp_ts = np.array([-a, -a, -a, -a, a, a, a, a])
        self.gp_wts = np.array([w, w, w, w, w, w, w, w])

        self.num_nodes = 8

    def project_to_nodes(self, method: str, gp_resp: np.ndarray) -> np.ndarray:
        """Project Gauss point responses to nodes.

        Parameters
        ----------
        method: str
            Method to project Gauss point responses to nodes.
            If "extrapolate", use extrapolation method.
            If "average", use weighted averaging method, weight is the Gauss point weight.
            If "copy", use copy method, copy Gauss point responses to nodes.
        gp_resp: np.ndarray
            Gauss point responses, shape[n, m], where n is the number of Gauss points,
            m is the number of responses at each Gauss point.

        Returns
        -------
        np.ndarray
            Projected responses at nodes, shape[4, m], where 4 is the number of nodes.
        """
        method = method.lower()
        if method.startswith("extra"):
            weights = np.array([
                [2.54903811, -0.6830127, 0.1830127, -0.6830127, -0.6830127, 0.1830127, -0.04903811, 0.1830127],
                [-0.6830127, 2.54903811, -0.6830127, 0.1830127, 0.1830127, -0.6830127, 0.1830127, -0.04903811],
                [0.1830127, -0.6830127, 2.54903811, -0.6830127, -0.04903811, 0.1830127, -0.6830127, 0.1830127],
                [-0.6830127, 0.1830127, -0.6830127, 2.54903811, 0.1830127, -0.04903811, 0.1830127, -0.6830127],
                [-0.6830127, 0.1830127, -0.04903811, 0.1830127, 2.54903811, -0.6830127, 0.1830127, -0.6830127],
                [0.1830127, -0.6830127, 0.1830127, -0.04903811, -0.6830127, 2.54903811, -0.6830127, 0.1830127],
                [-0.04903811, 0.1830127, -0.6830127, 0.1830127, 0.1830127, -0.6830127, 2.54903811, -0.6830127],
                [0.1830127, -0.04903811, 0.1830127, -0.6830127, -0.6830127, 0.1830127, -0.6830127, 2.54903811],
            ])
        elif method.startswith("ave"):
            weights = np.tile(self.gp_wts / np.sum(self.gp_wts), (self.num_nodes, 1))
        elif method.startswith("copy"):
            weights = np.eye(self.num_nodes)
        if gp_resp.ndim == 2:
            return np.dot(weights, gp_resp)
        elif gp_resp.ndim == 3:
            return np.einsum("ng,gfr->nfr", weights, gp_resp)
        else:
            raise ValueError("gp_resp must be 2D or 3D array.")  # noqa: TRY003

    @staticmethod
    def shape_func(r: float, s: float, t: float) -> np.ndarray:
        """8 nodes brick shape function for 8-node brick element.

        Parameters
        ----------
        r, s, t: normalized coordinates in the brick, -1 <= r, s, t <= 1

        Returns
        -------
        N: shape function values at the given r, s, t coordinates.
        """
        coords = np.array([
            [-1, -1, -1],
            [1, -1, -1],
            [1, 1, -1],
            [-1, 1, -1],
            [-1, -1, 1],
            [1, -1, 1],
            [1, 1, 1],
            [-1, 1, 1],
        ])
        N = []
        for xi_i, eta_i, zeta_i in coords:
            val = (1 + xi_i * r) * (1 + eta_i * s) * (1 + zeta_i * t) / 8
            N.append(val)
        return np.array(N)


class FEbrickN20GP27:
    """20-node brick element with 27 Gauss points.

    Topology:

        Top (t = +1):       Mid (t =  0):       Bottom (t = -1):

            7---15---8           20------19           4---11----3
            |        |           |        |           |         |
            16  23  14           |        |           12  26   10
            |        |           |        |           |         |
            5---13---6           17------18           1----9----2
    """

    def __init__(self):
        a = 0.774596669241483
        self.gp_rs = [-a, a, a, -a, -a, a, a, -a, 0, a, 0, -a, 0, a, 0, -a, -a, a, a, -a, a, 0, 0, -a, 0, 0, 0]
        self.gp_ss = [-a, -a, a, a, -a, -a, a, a, -a, 0, a, 0, -a, 0, a, 0, -a, -a, a, a, 0, a, 0, 0, -a, 0, 0]
        self.gp_ts = [-a, -a, -a, -a, a, a, a, a, -a, -a, -a, -a, a, a, a, a, 0, 0, 0, 0, 0, 0, a, 0, 0, -a, 0]
        self.gp_wts = [
            0.1714677640603567,
            0.1714677640603567,
            0.1714677640603567,
            0.1714677640603567,
            0.1714677640603567,
            0.1714677640603567,
            0.1714677640603567,
            0.1714677640603567,
            0.2743484224965707,
            0.2743484224965707,
            0.2743484224965707,
            0.2743484224965707,
            0.2743484224965707,
            0.2743484224965707,
            0.2743484224965707,
            0.2743484224965707,
            0.2743484224965707,
            0.2743484224965707,
            0.2743484224965707,
            0.2743484224965707,
            0.43974468799451316,
            0.43974468799451316,
            0.43974468799451316,
            0.43974468799451316,
            0.43974468799451316,
            0.7023319610971642,
        ]
        self.gp_rs = np.array(self.gp_rs)
        self.gp_ss = np.array(self.gp_ss)
        self.gp_ts = np.array(self.gp_ts)
        self.gp_wts = np.array(self.gp_wts)

        self.num_nodes = 20

    def project_to_nodes(self, method: str, gp_resp: np.ndarray) -> np.ndarray:
        """Project Gauss point responses to nodes.

        Parameters
        ----------
        method: str
            Method to project Gauss point responses to nodes.
            If "extrapolate", use extrapolation method.
            If "average", use weighted averaging method, weight is the Gauss point weight.
            If "copy", use copy method, copy Gauss point responses to nodes.
        gp_resp: np.ndarray
            Gauss point responses, shape[n, m], where n is the number of Gauss points,
            m is the number of responses at each Gauss point.

        Returns
        -------
        np.ndarray
            Projected responses at nodes, shape[4, m], where 4 is the number of nodes.
        """
        method = method.lower()
        if method.startswith("extra"):
            weights = weights_brick20_gp27_extra
        elif method.startswith("ave"):
            weights = np.tile(self.gp_wts / np.sum(self.gp_wts), (self.num_nodes, 1))
        elif method.startswith("copy"):
            weights = np.eye(27)[: self.num_nodes]
        if gp_resp.ndim == 2:
            return np.dot(weights, gp_resp)
        elif gp_resp.ndim == 3:
            return np.einsum("ng,gfr->nfr", weights, gp_resp)
        else:
            raise ValueError("gp_resp must be 2D or 3D array.")  # noqa: TRY003

    @staticmethod
    def shape_func(r: float, s: float, t: float) -> np.ndarray:
        """20 nodes brick shape function for 20-node brick element.

        Parameters
        ----------
        r, s, t: normalized coordinates in the brick, -1 <= r, s, t <= 1

        Returns
        -------
        N: shape function values at the given r, s, t coordinates.
        """
        H = np.zeros(20)
        H[8] = 0.25 * (1 - r * r) * (1 - s) * (1 - t)
        H[9] = 0.25 * (1 - s * s) * (1 + r) * (1 - t)
        H[10] = 0.25 * (1 - r * r) * (1 + s) * (1 - t)
        H[11] = 0.25 * (1 - s * s) * (1 - r) * (1 - t)
        H[12] = 0.25 * (1 - r * r) * (1 - s) * (1 + t)
        H[13] = 0.25 * (1 - s * s) * (1 + r) * (1 + t)
        H[14] = 0.25 * (1 - r * r) * (1 + s) * (1 + t)
        H[15] = 0.25 * (1 - s * s) * (1 - r) * (1 + t)
        H[16] = 0.25 * (1 - t * t) * (1 - r) * (1 - s)
        H[17] = 0.25 * (1 - t * t) * (1 + r) * (1 - s)
        H[18] = 0.25 * (1 - t * t) * (1 + r) * (1 + s)
        H[19] = 0.25 * (1 - t * t) * (1 - r) * (1 + s)
        H[0] = 0.125 * (1 - r) * (1 - s) * (1 - t) - 0.5 * (H[8] + H[11] + H[16])
        H[1] = 0.125 * (1 + r) * (1 - s) * (1 - t) - 0.5 * (H[8] + H[9] + H[17])
        H[2] = 0.125 * (1 + r) * (1 + s) * (1 - t) - 0.5 * (H[9] + H[10] + H[18])
        H[3] = 0.125 * (1 - r) * (1 + s) * (1 - t) - 0.5 * (H[10] + H[11] + H[19])
        H[4] = 0.125 * (1 - r) * (1 - s) * (1 + t) - 0.5 * (H[12] + H[15] + H[16])
        H[5] = 0.125 * (1 + r) * (1 - s) * (1 + t) - 0.5 * (H[12] + H[13] + H[17])
        H[6] = 0.125 * (1 + r) * (1 + s) * (1 + t) - 0.5 * (H[13] + H[14] + H[18])
        H[7] = 0.125 * (1 - r) * (1 + s) * (1 + t) - 0.5 * (H[14] + H[15] + H[19])
        return H


class FEbrickN27GP27:
    """27-node brick element with 27 Gauss points.

    Top (t = +1):       Mid (t =  0):       Bottom (t = -1):

        7---15---8           20--22--19           4---11----3
        |        |           |        |           |         |
        16  23  14           24  27  21           12  26   10
        |        |           |        |           |         |
        5---13---6           17--25--18           1----9----2
    """

    def __init__(self):
        brick20 = FEbrickN20GP27()
        self.gp_rs = brick20.gp_rs
        self.gp_ss = brick20.gp_ss
        self.gp_ts = brick20.gp_ts
        self.gp_wts = brick20.gp_wts

        self.num_nodes = 27

    def project_to_nodes(self, method: str, gp_resp: np.ndarray) -> np.ndarray:
        """Project Gauss point responses to nodes.

        Parameters
        ----------
        method: str
            Method to project Gauss point responses to nodes.
            If "extrapolate", use extrapolation method.
            If "average", use weighted averaging method, weight is the Gauss point weight.
            If "copy", use copy method, copy Gauss point responses to nodes.
        gp_resp: np.ndarray
            Gauss point responses, shape[n, m], where n is the number of Gauss points,
            m is the number of responses at each Gauss point.

        Returns
        -------
        np.ndarray
            Projected responses at nodes, shape[4, m], where 4 is the number of nodes.
        """
        method = method.lower()
        if method.startswith("extra"):
            weights = weights_brick27_gp27_extra
        elif method.startswith("ave"):
            weights = np.tile(self.gp_wts / np.sum(self.gp_wts), (self.num_nodes, 1))
        elif method.startswith("copy"):
            weights = np.eye(self.num_nodes)
        if gp_resp.ndim == 2:
            return np.dot(weights, gp_resp)
        elif gp_resp.ndim == 3:
            return np.einsum("ng,gfr->nfr", weights, gp_resp)
        else:
            raise ValueError("gp_resp must be 2D or 3D array.")  # noqa: TRY003

    @staticmethod
    def shape_func(r: float, s: float, t: float) -> np.ndarray:
        """27 nodes brick shape function for 27-node brick element.

        Parameters
        ----------
        r, s, t: normalized coordinates in the brick, -1 <= r, s, t <= 1

        Returns
        -------
        N: shape function values at the given r, s, t coordinates.
        """
        R = [0.5 * r * (r - 1.0), 0.5 * r * (r + 1.0), 1.0 - r * r]
        S = [0.5 * s * (s - 1.0), 0.5 * s * (s + 1.0), 1.0 - s * s]
        T = [0.5 * t * (t - 1.0), 0.5 * t * (t + 1.0), 1.0 - t * t]

        H = np.zeros(27)
        H[0] = R[0] * S[0] * T[0]
        H[1] = R[1] * S[0] * T[0]
        H[2] = R[1] * S[1] * T[0]
        H[3] = R[0] * S[1] * T[0]
        H[4] = R[0] * S[0] * T[1]
        H[5] = R[1] * S[0] * T[1]
        H[6] = R[1] * S[1] * T[1]
        H[7] = R[0] * S[1] * T[1]
        H[8] = R[2] * S[0] * T[0]
        H[9] = R[1] * S[2] * T[0]
        H[10] = R[2] * S[1] * T[0]
        H[11] = R[0] * S[2] * T[0]
        H[12] = R[2] * S[0] * T[1]
        H[13] = R[1] * S[2] * T[1]
        H[14] = R[2] * S[1] * T[1]
        H[15] = R[0] * S[2] * T[1]
        H[16] = R[0] * S[0] * T[2]
        H[17] = R[1] * S[0] * T[2]
        H[18] = R[1] * S[1] * T[2]
        H[19] = R[0] * S[1] * T[2]
        H[20] = R[2] * S[0] * T[2]
        H[21] = R[1] * S[2] * T[2]
        H[22] = R[2] * S[1] * T[2]
        H[23] = R[0] * S[2] * T[2]
        H[24] = R[2] * S[2] * T[0]
        H[25] = R[2] * S[2] * T[1]
        H[26] = R[2] * S[2] * T[2]
        return H


class FEshellN3GP1(FEtriN3GP1):  # ASDShellT3
    def __init__(self):
        super().__init__()


class FEshellN3GP3(FEtriN3GP1):  # ASDShellT3
    def __init__(self):
        self.gp_rs = np.array([0.5, 0.0, 0.5])
        self.gp_ss = np.array([0.5, 0.5, 0.0])
        self.gp_wts = np.array([1.0 / 6.0, 1.0 / 6.0, 1.0 / 6.0])

        self.node_rs = np.array([0.0, 1.0, 0.0])
        self.node_ss = np.array([0.0, 0.0, 1.0])

        self.num_nodes = 3

    def project_to_nodes(self, method: str, gp_resp: np.ndarray) -> np.ndarray:
        method = method.lower()
        if method.startswith("extra"):
            weights = np.array([
                [0.261204, 0.369398, 0.369398],
                [0.328227, 0.207589, 0.464183],
                [0.328227, 0.464183, 0.207589],
            ])
        elif method.startswith("ave"):
            weights = np.tile(self.gp_wts / np.sum(self.gp_wts), (self.num_nodes, 1))
        elif method.startswith("copy"):
            weights = np.array([
                [0, 0.5, 0.5],  # N1 ← G3, G2
                [0.5, 0, 0.5],  # N2 ← G1, G3
                [0.5, 0.5, 0],  # N3 ← G1, G2
            ])
        if gp_resp.ndim == 2:
            return np.dot(weights, gp_resp)
        elif gp_resp.ndim == 3:
            return np.einsum("ng,gfr->nfr", weights, gp_resp)
        else:
            raise ValueError("gp_resp must be 2D or 3D array.")  # noqa: TRY003


class FEshellN3GP4:  # DKGT and NLDKGT
    def __init__(self):
        self.gp_rs = np.array([1 / 3, 1 / 5, 3 / 5, 1 / 5])
        self.gp_ss = np.array([1 / 3, 3 / 5, 1 / 5, 1 / 5])
        self.gp_wts = np.array([-9.0 / 16.0, 25.0 / 48.0, 25.0 / 48.0, 25.0 / 48.0])

        self.node_rs = np.array([0.0, 1.0, 0.0])
        self.node_ss = np.array([0.0, 0.0, 1.0])

        self.num_nodes = 3

    def project_to_nodes(self, method: str, gp_resp: np.ndarray) -> np.ndarray:
        """Project Gauss point responses to nodes."""
        method = method.lower()
        if method.startswith("extra"):
            weights = np.array([
                [0.240536, 0.179285, 0.179285, 0.400893],  # N1
                [0.231701, 0.172700, 0.386169, 0.209430],  # N2
                [0.231701, 0.386169, 0.172700, 0.209430],  # N3
            ])
        elif method.startswith("ave"):
            weights = np.tile(self.gp_wts / np.sum(self.gp_wts), (self.num_nodes, 1))
        elif method.startswith("copy"):
            weights = np.array([
                [0, 0, 0, 1],  # N1 ← G3
                [0, 0, 1, 0],  # N2 ← G2
                [0, 1, 0, 0],  # N3 ← G1
            ])
        if gp_resp.ndim == 2:
            return np.dot(weights, gp_resp)
        elif gp_resp.ndim == 3:
            return np.einsum("ng,gfr->nfr", weights, gp_resp)
        else:
            raise ValueError("gp_resp must be 2D or 3D array.")  # noqa: TRY003


class FEshellN4GP4(FEquadN4GP4):
    def __init__(self):
        super().__init__()


class FEshellN9GP9(FEquadN9GP9):
    def __init__(self):
        super().__init__()


weights_brick20_gp27_extra = np.array([
    [
        3.23411343,
        0.41078627,
        0.0521767,
        0.41078627,
        0.41078627,
        0.0521767,
        0.00662731,
        0.0521767,
        -1.45795988,
        -0.18518519,
        -0.18518519,
        -1.45795988,
        -0.18518519,
        -0.0235216,
        -0.0235216,
        -0.18518519,
        -1.45795988,
        -0.18518519,
        -0.0235216,
        -0.18518519,
        0.65725803,
        0.08348272,
        0.08348272,
        0.65725803,
        0.65725803,
        0.08348272,
        -0.2962963,
    ],
    [
        0.41078627,
        3.23411343,
        0.41078627,
        0.0521767,
        0.0521767,
        0.41078627,
        0.0521767,
        0.00662731,
        -1.45795988,
        -1.45795988,
        -0.18518519,
        -0.18518519,
        -0.18518519,
        -0.18518519,
        -0.0235216,
        -0.0235216,
        -0.18518519,
        -1.45795988,
        -0.18518519,
        -0.0235216,
        0.65725803,
        0.65725803,
        0.08348272,
        0.08348272,
        0.65725803,
        0.08348272,
        -0.2962963,
    ],
    [
        0.0521767,
        0.41078627,
        3.23411343,
        0.41078627,
        0.00662731,
        0.0521767,
        0.41078627,
        0.0521767,
        -0.18518519,
        -1.45795988,
        -1.45795988,
        -0.18518519,
        -0.0235216,
        -0.18518519,
        -0.18518519,
        -0.0235216,
        -0.0235216,
        -0.18518519,
        -1.45795988,
        -0.18518519,
        0.08348272,
        0.65725803,
        0.65725803,
        0.08348272,
        0.65725803,
        0.08348272,
        -0.2962963,
    ],
    [
        0.41078627,
        0.0521767,
        0.41078627,
        3.23411343,
        0.0521767,
        0.00662731,
        0.0521767,
        0.41078627,
        -0.18518519,
        -0.18518519,
        -1.45795988,
        -1.45795988,
        -0.0235216,
        -0.0235216,
        -0.18518519,
        -0.18518519,
        -0.18518519,
        -0.0235216,
        -0.18518519,
        -1.45795988,
        0.08348272,
        0.08348272,
        0.65725803,
        0.65725803,
        0.65725803,
        0.08348272,
        -0.2962963,
    ],
    [
        0.41078627,
        0.0521767,
        0.00662731,
        0.0521767,
        3.23411343,
        0.41078627,
        0.0521767,
        0.41078627,
        -0.18518519,
        -0.0235216,
        -0.0235216,
        -0.18518519,
        -1.45795988,
        -0.18518519,
        -0.18518519,
        -1.45795988,
        -1.45795988,
        -0.18518519,
        -0.0235216,
        -0.18518519,
        0.65725803,
        0.08348272,
        0.08348272,
        0.65725803,
        0.08348272,
        0.65725803,
        -0.2962963,
    ],
    [
        0.0521767,
        0.41078627,
        0.0521767,
        0.00662731,
        0.41078627,
        3.23411343,
        0.41078627,
        0.0521767,
        -0.18518519,
        -0.18518519,
        -0.0235216,
        -0.0235216,
        -1.45795988,
        -1.45795988,
        -0.18518519,
        -0.18518519,
        -0.18518519,
        -1.45795988,
        -0.18518519,
        -0.0235216,
        0.65725803,
        0.65725803,
        0.08348272,
        0.08348272,
        0.08348272,
        0.65725803,
        -0.2962963,
    ],
    [
        0.00662731,
        0.0521767,
        0.41078627,
        0.0521767,
        0.0521767,
        0.41078627,
        3.23411343,
        0.41078627,
        -0.0235216,
        -0.18518519,
        -0.18518519,
        -0.0235216,
        -0.18518519,
        -1.45795988,
        -1.45795988,
        -0.18518519,
        -0.0235216,
        -0.18518519,
        -1.45795988,
        -0.18518519,
        0.08348272,
        0.65725803,
        0.65725803,
        0.08348272,
        0.08348272,
        0.65725803,
        -0.2962963,
    ],
    [
        0.0521767,
        0.00662731,
        0.0521767,
        0.41078627,
        0.41078627,
        0.0521767,
        0.41078627,
        3.23411343,
        -0.0235216,
        -0.0235216,
        -0.18518519,
        -0.18518519,
        -0.18518519,
        -0.18518519,
        -1.45795988,
        -1.45795988,
        -0.18518519,
        -0.0235216,
        -0.18518519,
        -1.45795988,
        0.08348272,
        0.08348272,
        0.65725803,
        0.65725803,
        0.08348272,
        0.65725803,
        -0.2962963,
    ],
    [
        -0.0,
        0.0,
        0.0,
        -0.0,
        -0.0,
        0.0,
        0.0,
        -0.0,
        2.18693982,
        -0.0,
        0.27777778,
        0.0,
        0.27777778,
        -0.0,
        0.0352824,
        0.0,
        0.0,
        -0.0,
        -0.0,
        0.0,
        -0.98588704,
        0.0,
        -0.12522407,
        -0.0,
        -0.98588704,
        -0.12522407,
        0.44444444,
    ],
    [
        -0.0,
        -0.0,
        0.0,
        0.0,
        -0.0,
        -0.0,
        0.0,
        0.0,
        0.0,
        2.18693982,
        -0.0,
        0.27777778,
        0.0,
        0.27777778,
        -0.0,
        0.0352824,
        0.0,
        0.0,
        -0.0,
        -0.0,
        -0.0,
        -0.98588704,
        0.0,
        -0.12522407,
        -0.98588704,
        -0.12522407,
        0.44444444,
    ],
    [
        -0.0,
        0.0,
        0.0,
        -0.0,
        -0.0,
        0.0,
        0.0,
        -0.0,
        0.27777778,
        -0.0,
        2.18693982,
        0.0,
        0.0352824,
        -0.0,
        0.27777778,
        0.0,
        0.0,
        -0.0,
        -0.0,
        0.0,
        -0.12522407,
        0.0,
        -0.98588704,
        -0.0,
        -0.98588704,
        -0.12522407,
        0.44444444,
    ],
    [
        -0.0,
        -0.0,
        0.0,
        0.0,
        -0.0,
        -0.0,
        0.0,
        0.0,
        0.0,
        0.27777778,
        -0.0,
        2.18693982,
        0.0,
        0.0352824,
        -0.0,
        0.27777778,
        0.0,
        0.0,
        -0.0,
        -0.0,
        -0.0,
        -0.12522407,
        0.0,
        -0.98588704,
        -0.98588704,
        -0.12522407,
        0.44444444,
    ],
    [
        -0.0,
        0.0,
        0.0,
        -0.0,
        -0.0,
        0.0,
        0.0,
        -0.0,
        0.27777778,
        -0.0,
        0.0352824,
        0.0,
        2.18693982,
        -0.0,
        0.27777778,
        0.0,
        0.0,
        -0.0,
        -0.0,
        0.0,
        -0.98588704,
        0.0,
        -0.12522407,
        -0.0,
        -0.12522407,
        -0.98588704,
        0.44444444,
    ],
    [
        -0.0,
        -0.0,
        0.0,
        0.0,
        -0.0,
        -0.0,
        0.0,
        0.0,
        0.0,
        0.27777778,
        -0.0,
        0.0352824,
        0.0,
        2.18693982,
        -0.0,
        0.27777778,
        0.0,
        0.0,
        -0.0,
        -0.0,
        -0.0,
        -0.98588704,
        0.0,
        -0.12522407,
        -0.12522407,
        -0.98588704,
        0.44444444,
    ],
    [
        -0.0,
        0.0,
        0.0,
        -0.0,
        -0.0,
        0.0,
        0.0,
        -0.0,
        0.0352824,
        -0.0,
        0.27777778,
        0.0,
        0.27777778,
        -0.0,
        2.18693982,
        0.0,
        0.0,
        -0.0,
        -0.0,
        0.0,
        -0.12522407,
        0.0,
        -0.98588704,
        -0.0,
        -0.12522407,
        -0.98588704,
        0.44444444,
    ],
    [
        -0.0,
        -0.0,
        0.0,
        0.0,
        -0.0,
        -0.0,
        0.0,
        0.0,
        0.0,
        0.0352824,
        -0.0,
        0.27777778,
        0.0,
        0.27777778,
        -0.0,
        2.18693982,
        0.0,
        0.0,
        -0.0,
        -0.0,
        -0.0,
        -0.12522407,
        0.0,
        -0.98588704,
        -0.12522407,
        -0.98588704,
        0.44444444,
    ],
    [
        -0.0,
        -0.0,
        -0.0,
        -0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        -0.0,
        -0.0,
        -0.0,
        -0.0,
        2.18693982,
        0.27777778,
        0.0352824,
        0.27777778,
        -0.98588704,
        -0.12522407,
        -0.12522407,
        -0.98588704,
        -0.0,
        0.0,
        0.44444444,
    ],
    [
        -0.0,
        -0.0,
        -0.0,
        -0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        -0.0,
        -0.0,
        -0.0,
        -0.0,
        0.27777778,
        2.18693982,
        0.27777778,
        0.0352824,
        -0.98588704,
        -0.98588704,
        -0.12522407,
        -0.12522407,
        -0.0,
        0.0,
        0.44444444,
    ],
    [
        -0.0,
        -0.0,
        -0.0,
        -0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        -0.0,
        -0.0,
        -0.0,
        -0.0,
        0.0352824,
        0.27777778,
        2.18693982,
        0.27777778,
        -0.12522407,
        -0.98588704,
        -0.98588704,
        -0.12522407,
        -0.0,
        0.0,
        0.44444444,
    ],
    [
        -0.0,
        -0.0,
        -0.0,
        -0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        -0.0,
        -0.0,
        -0.0,
        -0.0,
        0.27777778,
        0.0352824,
        0.27777778,
        2.18693982,
        -0.12522407,
        -0.12522407,
        -0.98588704,
        -0.98588704,
        -0.0,
        0.0,
        0.44444444,
    ],
])

weights_brick27_gp27_extra = np.array([
    [
        3.23411343,
        0.41078627,
        0.0521767,
        0.41078627,
        0.41078627,
        0.0521767,
        0.00662731,
        0.0521767,
        -1.45795988,
        -0.18518519,
        -0.18518519,
        -1.45795988,
        -0.18518519,
        -0.0235216,
        -0.0235216,
        -0.18518519,
        -1.45795988,
        -0.18518519,
        -0.0235216,
        -0.18518519,
        0.65725803,
        0.08348272,
        0.08348272,
        0.65725803,
        0.65725803,
        0.08348272,
        -0.2962963,
    ],
    [
        0.41078627,
        3.23411343,
        0.41078627,
        0.0521767,
        0.0521767,
        0.41078627,
        0.0521767,
        0.00662731,
        -1.45795988,
        -1.45795988,
        -0.18518519,
        -0.18518519,
        -0.18518519,
        -0.18518519,
        -0.0235216,
        -0.0235216,
        -0.18518519,
        -1.45795988,
        -0.18518519,
        -0.0235216,
        0.65725803,
        0.65725803,
        0.08348272,
        0.08348272,
        0.65725803,
        0.08348272,
        -0.2962963,
    ],
    [
        0.0521767,
        0.41078627,
        3.23411343,
        0.41078627,
        0.00662731,
        0.0521767,
        0.41078627,
        0.0521767,
        -0.18518519,
        -1.45795988,
        -1.45795988,
        -0.18518519,
        -0.0235216,
        -0.18518519,
        -0.18518519,
        -0.0235216,
        -0.0235216,
        -0.18518519,
        -1.45795988,
        -0.18518519,
        0.08348272,
        0.65725803,
        0.65725803,
        0.08348272,
        0.65725803,
        0.08348272,
        -0.2962963,
    ],
    [
        0.41078627,
        0.0521767,
        0.41078627,
        3.23411343,
        0.0521767,
        0.00662731,
        0.0521767,
        0.41078627,
        -0.18518519,
        -0.18518519,
        -1.45795988,
        -1.45795988,
        -0.0235216,
        -0.0235216,
        -0.18518519,
        -0.18518519,
        -0.18518519,
        -0.0235216,
        -0.18518519,
        -1.45795988,
        0.08348272,
        0.08348272,
        0.65725803,
        0.65725803,
        0.65725803,
        0.08348272,
        -0.2962963,
    ],
    [
        0.41078627,
        0.0521767,
        0.00662731,
        0.0521767,
        3.23411343,
        0.41078627,
        0.0521767,
        0.41078627,
        -0.18518519,
        -0.0235216,
        -0.0235216,
        -0.18518519,
        -1.45795988,
        -0.18518519,
        -0.18518519,
        -1.45795988,
        -1.45795988,
        -0.18518519,
        -0.0235216,
        -0.18518519,
        0.65725803,
        0.08348272,
        0.08348272,
        0.65725803,
        0.08348272,
        0.65725803,
        -0.2962963,
    ],
    [
        0.0521767,
        0.41078627,
        0.0521767,
        0.00662731,
        0.41078627,
        3.23411343,
        0.41078627,
        0.0521767,
        -0.18518519,
        -0.18518519,
        -0.0235216,
        -0.0235216,
        -1.45795988,
        -1.45795988,
        -0.18518519,
        -0.18518519,
        -0.18518519,
        -1.45795988,
        -0.18518519,
        -0.0235216,
        0.65725803,
        0.65725803,
        0.08348272,
        0.08348272,
        0.08348272,
        0.65725803,
        -0.2962963,
    ],
    [
        0.00662731,
        0.0521767,
        0.41078627,
        0.0521767,
        0.0521767,
        0.41078627,
        3.23411343,
        0.41078627,
        -0.0235216,
        -0.18518519,
        -0.18518519,
        -0.0235216,
        -0.18518519,
        -1.45795988,
        -1.45795988,
        -0.18518519,
        -0.0235216,
        -0.18518519,
        -1.45795988,
        -0.18518519,
        0.08348272,
        0.65725803,
        0.65725803,
        0.08348272,
        0.08348272,
        0.65725803,
        -0.2962963,
    ],
    [
        0.0521767,
        0.00662731,
        0.0521767,
        0.41078627,
        0.41078627,
        0.0521767,
        0.41078627,
        3.23411343,
        -0.0235216,
        -0.0235216,
        -0.18518519,
        -0.18518519,
        -0.18518519,
        -0.18518519,
        -1.45795988,
        -1.45795988,
        -0.18518519,
        -0.0235216,
        -0.18518519,
        -1.45795988,
        0.08348272,
        0.08348272,
        0.65725803,
        0.65725803,
        0.08348272,
        0.65725803,
        -0.2962963,
    ],
    [
        -0.0,
        0.0,
        0.0,
        -0.0,
        -0.0,
        0.0,
        0.0,
        -0.0,
        2.18693982,
        -0.0,
        0.27777778,
        0.0,
        0.27777778,
        -0.0,
        0.0352824,
        0.0,
        0.0,
        -0.0,
        -0.0,
        0.0,
        -0.98588704,
        0.0,
        -0.12522407,
        -0.0,
        -0.98588704,
        -0.12522407,
        0.44444444,
    ],
    [
        -0.0,
        -0.0,
        0.0,
        0.0,
        -0.0,
        -0.0,
        0.0,
        0.0,
        0.0,
        2.18693982,
        -0.0,
        0.27777778,
        0.0,
        0.27777778,
        -0.0,
        0.0352824,
        0.0,
        0.0,
        -0.0,
        -0.0,
        -0.0,
        -0.98588704,
        0.0,
        -0.12522407,
        -0.98588704,
        -0.12522407,
        0.44444444,
    ],
    [
        -0.0,
        0.0,
        0.0,
        -0.0,
        -0.0,
        0.0,
        0.0,
        -0.0,
        0.27777778,
        -0.0,
        2.18693982,
        0.0,
        0.0352824,
        -0.0,
        0.27777778,
        0.0,
        0.0,
        -0.0,
        -0.0,
        0.0,
        -0.12522407,
        0.0,
        -0.98588704,
        -0.0,
        -0.98588704,
        -0.12522407,
        0.44444444,
    ],
    [
        -0.0,
        -0.0,
        0.0,
        0.0,
        -0.0,
        -0.0,
        0.0,
        0.0,
        0.0,
        0.27777778,
        -0.0,
        2.18693982,
        0.0,
        0.0352824,
        -0.0,
        0.27777778,
        0.0,
        0.0,
        -0.0,
        -0.0,
        -0.0,
        -0.12522407,
        0.0,
        -0.98588704,
        -0.98588704,
        -0.12522407,
        0.44444444,
    ],
    [
        -0.0,
        0.0,
        0.0,
        -0.0,
        -0.0,
        0.0,
        0.0,
        -0.0,
        0.27777778,
        -0.0,
        0.0352824,
        0.0,
        2.18693982,
        -0.0,
        0.27777778,
        0.0,
        0.0,
        -0.0,
        -0.0,
        0.0,
        -0.98588704,
        0.0,
        -0.12522407,
        -0.0,
        -0.12522407,
        -0.98588704,
        0.44444444,
    ],
    [
        -0.0,
        -0.0,
        0.0,
        0.0,
        -0.0,
        -0.0,
        0.0,
        0.0,
        0.0,
        0.27777778,
        -0.0,
        0.0352824,
        0.0,
        2.18693982,
        -0.0,
        0.27777778,
        0.0,
        0.0,
        -0.0,
        -0.0,
        -0.0,
        -0.98588704,
        0.0,
        -0.12522407,
        -0.12522407,
        -0.98588704,
        0.44444444,
    ],
    [
        -0.0,
        0.0,
        0.0,
        -0.0,
        -0.0,
        0.0,
        0.0,
        -0.0,
        0.0352824,
        -0.0,
        0.27777778,
        0.0,
        0.27777778,
        -0.0,
        2.18693982,
        0.0,
        0.0,
        -0.0,
        -0.0,
        0.0,
        -0.12522407,
        0.0,
        -0.98588704,
        -0.0,
        -0.12522407,
        -0.98588704,
        0.44444444,
    ],
    [
        -0.0,
        -0.0,
        0.0,
        0.0,
        -0.0,
        -0.0,
        0.0,
        0.0,
        0.0,
        0.0352824,
        -0.0,
        0.27777778,
        0.0,
        0.27777778,
        -0.0,
        2.18693982,
        0.0,
        0.0,
        -0.0,
        -0.0,
        -0.0,
        -0.12522407,
        0.0,
        -0.98588704,
        -0.12522407,
        -0.98588704,
        0.44444444,
    ],
    [
        -0.0,
        -0.0,
        -0.0,
        -0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        -0.0,
        -0.0,
        -0.0,
        -0.0,
        2.18693982,
        0.27777778,
        0.0352824,
        0.27777778,
        -0.98588704,
        -0.12522407,
        -0.12522407,
        -0.98588704,
        -0.0,
        0.0,
        0.44444444,
    ],
    [
        -0.0,
        -0.0,
        -0.0,
        -0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        -0.0,
        -0.0,
        -0.0,
        -0.0,
        0.27777778,
        2.18693982,
        0.27777778,
        0.0352824,
        -0.98588704,
        -0.98588704,
        -0.12522407,
        -0.12522407,
        -0.0,
        0.0,
        0.44444444,
    ],
    [
        -0.0,
        -0.0,
        -0.0,
        -0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        -0.0,
        -0.0,
        -0.0,
        -0.0,
        0.0352824,
        0.27777778,
        2.18693982,
        0.27777778,
        -0.12522407,
        -0.98588704,
        -0.98588704,
        -0.12522407,
        -0.0,
        0.0,
        0.44444444,
    ],
    [
        -0.0,
        -0.0,
        -0.0,
        -0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        -0.0,
        -0.0,
        -0.0,
        -0.0,
        0.27777778,
        0.0352824,
        0.27777778,
        2.18693982,
        -0.12522407,
        -0.12522407,
        -0.98588704,
        -0.98588704,
        -0.0,
        0.0,
        0.44444444,
    ],
    [
        0.0,
        0.0,
        -0.0,
        -0.0,
        -0.0,
        -0.0,
        0.0,
        0.0,
        -0.0,
        -0.0,
        0.0,
        -0.0,
        0.0,
        0.0,
        -0.0,
        0.0,
        -0.0,
        -0.0,
        0.0,
        0.0,
        0.0,
        1.47883056,
        -0.0,
        0.18783611,
        0.0,
        -0.0,
        -0.66666667,
    ],
    [
        0.0,
        -0.0,
        -0.0,
        0.0,
        -0.0,
        0.0,
        0.0,
        -0.0,
        -0.0,
        0.0,
        -0.0,
        -0.0,
        0.0,
        -0.0,
        0.0,
        0.0,
        -0.0,
        0.0,
        0.0,
        -0.0,
        0.18783611,
        -0.0,
        1.47883056,
        0.0,
        0.0,
        -0.0,
        -0.66666667,
    ],
    [
        0.0,
        -0.0,
        0.0,
        -0.0,
        0.0,
        -0.0,
        0.0,
        -0.0,
        -0.0,
        0.0,
        0.0,
        -0.0,
        -0.0,
        0.0,
        0.0,
        -0.0,
        -0.0,
        0.0,
        -0.0,
        0.0,
        0.0,
        -0.0,
        -0.0,
        0.0,
        0.18783611,
        1.47883056,
        -0.66666667,
    ],
    [
        0.0,
        0.0,
        -0.0,
        -0.0,
        -0.0,
        -0.0,
        0.0,
        0.0,
        -0.0,
        -0.0,
        0.0,
        -0.0,
        0.0,
        0.0,
        -0.0,
        0.0,
        -0.0,
        -0.0,
        0.0,
        0.0,
        0.0,
        0.18783611,
        -0.0,
        1.47883056,
        0.0,
        -0.0,
        -0.66666667,
    ],
    [
        0.0,
        -0.0,
        -0.0,
        0.0,
        -0.0,
        0.0,
        0.0,
        -0.0,
        -0.0,
        0.0,
        -0.0,
        -0.0,
        0.0,
        -0.0,
        0.0,
        0.0,
        -0.0,
        0.0,
        0.0,
        -0.0,
        1.47883056,
        -0.0,
        0.18783611,
        0.0,
        0.0,
        -0.0,
        -0.66666667,
    ],
    [
        0.0,
        -0.0,
        0.0,
        -0.0,
        0.0,
        -0.0,
        0.0,
        -0.0,
        -0.0,
        0.0,
        0.0,
        -0.0,
        -0.0,
        0.0,
        0.0,
        -0.0,
        -0.0,
        0.0,
        -0.0,
        0.0,
        0.0,
        -0.0,
        -0.0,
        0.0,
        1.47883056,
        0.18783611,
        -0.66666667,
    ],
    [
        -0.0,
        0.0,
        -0.0,
        0.0,
        0.0,
        -0.0,
        0.0,
        -0.0,
        0.0,
        -0.0,
        -0.0,
        0.0,
        -0.0,
        0.0,
        0.0,
        -0.0,
        0.0,
        -0.0,
        0.0,
        -0.0,
        -0.0,
        0.0,
        0.0,
        -0.0,
        -0.0,
        0.0,
        1.0,
    ],
])
