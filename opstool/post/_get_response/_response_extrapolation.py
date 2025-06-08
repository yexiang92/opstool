import numpy as np


def resp_extrap_tri3(resp):
    """3 nodes tri response extrapolation from 1 Gauss points.

    Parameters
    ----------
    resp: response array, shape[n, m],
         n is the row number 3 of gauss point, column is the responses.

    Returns
    -------
    Extrapolation response.
    """
    resp = np.ravel(resp)
    resp_new = np.zeros((3, len(resp)))
    for i in range(3):
        resp_new[i] = resp
    return resp_new


def shape_tri3(r, s):
    n1 = r
    n2 = s
    n3 = 1 - r - s
    return np.array([n1, n2, n3])


def resp_extrap_tri6(resp):
    """6 nodes tri response extrapolation from 3 Gauss points.

    Parameters
    ----------
    resp: response array, shape[n, m],
         n is the row number 3 of gauss point, column is the responses.

    Returns
    -------
    Extrapolation response.
    """
    shape = np.zeros((6, 3))
    rs = [5 / 3, -1 / 3, -1 / 3, 2 / 3, -1 / 3, 2 / 3]
    ss = [-1 / 3, 5 / 3, -1 / 3, 2 / 3, 2 / 3, -1 / 3]
    for i in range(6):
        r, s = rs[i], ss[i]
        shape[i] = shape_tri3(r, s)
    return shape @ resp


def resp_extrap_tet4(resp):
    """4 nodes tet response extrapolation from 1 Gauss points.

    Parameters
    ----------
    resp: response array, shape[n, m],
         n is the row number 3 of gauss point, column is the responses.

    Returns
    -------
    Extrapolation response.
    """
    resp = np.ravel(resp)
    resp_new = np.zeros((4, len(resp)))
    for i in range(4):
        resp_new[i] = resp
    return resp_new


def shape_tet4(r, s, t):
    n1 = r
    n2 = s
    n3 = t
    n4 = 1 - r - s - t
    return np.array([n1, n2, n3, n4])


def resp_extrap_tet10(resp):
    """10 nodes tet response extrapolation from 4 Gauss points.

    Parameters
    ----------
    resp: response array, shape[n, m],
         n is the row number 3 of gauss point, column is the responses.

    Returns
    -------
    Extrapolation response.
    """
    alpha = 0.5854101966249685
    beta = 0.1381966011250105
    # gauss_locs = [[alpha, beta, beta],
    #               [beta, alpha, beta],
    #               [beta, beta, alpha],
    #               [beta, beta, beta]]
    factor = 1 / (alpha - beta)
    shape = np.zeros((10, 4))
    nodes = [
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1],
        [0, 0, 0],
        [0.5, 0.5, 0.0],
        [0.0, 0.5, 0.5],
        [0.5, 0.0, 0.5],
        [0.5, 0.0, 0.0],
        [0.0, 0.5, 0.0],
        [0.0, 0.0, 0.5],
    ]
    nodes = factor * np.array(nodes) - factor * beta
    for i in range(10):
        r, s, t = nodes[i]
        shape[i] = shape_tet4(r, s, t)
    return shape @ resp


def shape_quad4(r, s):
    n1 = 0.25 * (1 - r) * (1 - s)
    n2 = 0.25 * (1 + r) * (1 - s)
    n3 = 0.25 * (1 + r) * (1 + s)
    n4 = 0.25 * (1 - r) * (1 + s)
    return np.array([n1, n2, n3, n4])


def resp_extrap_quad4(resp):
    """4 nodes quad response extrapolation from Gauss points.

    Parameters
    ----------
    resp: response array, shape[n, m],
         n is the row number 4 of gauss point, column is the responses.

    Returns
    -------
    Extrapolation response.
    """
    sqrt3 = 1.7320508075688772
    shape = np.zeros((4, 4))
    rs = [-sqrt3, sqrt3, sqrt3, -sqrt3]
    ss = [-sqrt3, -sqrt3, sqrt3, sqrt3]
    for i in range(4):
        r, s = rs[i], ss[i]
        shape[i] = shape_quad4(r, s)
    return shape @ resp


def shape_quad9(r, s):
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


def resp_extrap_quad9(resp):
    """9 nodes quad response extrapolation from Gauss points.

    Parameters
    ----------
    resp: response array, shape[n, m],
         n is the row number 9 of gauss point, column is the responses.

    Returns
    -------
    Extrapolation response.
    """
    sqrt53 = 1.2909944487358056
    shape = np.zeros((9, 9))
    rs = [-sqrt53, sqrt53, sqrt53, -sqrt53, 0.0, sqrt53, 0.0, -sqrt53, 0.0]
    ss = [-sqrt53, -sqrt53, sqrt53, sqrt53, -sqrt53, 0.0, sqrt53, 0.0, 0.0]
    for i in range(9):
        r, s = rs[i], ss[i]
        shape[i] = shape_quad9(r, s)
    return shape @ resp


def resp_extrap_quad8(resp):
    """8 nodes quad response extrapolation from Gauss points.

    Parameters
    ----------
    resp: response array, shape[n, m],
         n is the row number 9 of gauss point, column is the responses.

    Returns
    -------
    Extrapolation response.
    """
    sqrt53 = 1.2909944487358056
    shape = np.zeros((8, 9))
    rs = [-sqrt53, sqrt53, sqrt53, -sqrt53, 0.0, sqrt53, 0.0, -sqrt53]
    ss = [-sqrt53, -sqrt53, sqrt53, sqrt53, -sqrt53, 0.0, sqrt53, 0.0]
    for i in range(8):
        r, s = rs[i], ss[i]
        shape[i] = shape_quad9(r, s)
    return shape @ resp


def shape_brick8(r, s, t):
    n1 = 0.125 * (1 - r) * (1 - s) * (1 - t)
    n2 = 0.125 * (1 + r) * (1 - s) * (1 - t)
    n3 = 0.125 * (1 + r) * (1 + s) * (1 - t)
    n4 = 0.125 * (1 - r) * (1 + s) * (1 - t)
    n5 = 0.125 * (1 - r) * (1 - s) * (1 + t)
    n6 = 0.125 * (1 + r) * (1 - s) * (1 + t)
    n7 = 0.125 * (1 + r) * (1 + s) * (1 + t)
    n8 = 0.125 * (1 - r) * (1 + s) * (1 + t)
    return np.array([n1, n2, n3, n4, n5, n6, n7, n8])


def resp_extrap_brick8(resp):
    """8 nodes brick response extrapolation from 8 Gauss points.

    Parameters
    ----------
    resp: response array, shape[n, m],
         n is the row number 8 of gauss point, column is the responses.

    Returns
    -------
    Extrapolation response.
    """
    sqrt3 = 1.7320508075688772
    shape = np.zeros((8, 8))
    rs = [-sqrt3, sqrt3, sqrt3, -sqrt3, -sqrt3, sqrt3, sqrt3, -sqrt3]
    ss = [-sqrt3, -sqrt3, sqrt3, sqrt3, -sqrt3, -sqrt3, sqrt3, sqrt3]
    ts = [-sqrt3, -sqrt3, -sqrt3, -sqrt3, sqrt3, sqrt3, sqrt3, sqrt3]
    for i in range(8):
        r, s, t = rs[i], ss[i], ts[i]
        shape[i] = shape_brick8(r, s, t)
    return shape @ resp


def shape_brick27(r, s, t):
    nodes = np.array([
        [-1, -1, -1],  # Nodes 1-4 Lower surface, counterclockwise
        [1, -1, -1],
        [1, 1, -1],
        [-1, 1, -1],
        [-1, -1, 1],  # Nodes 5-8 Upper surface, counterclockwise
        [1, -1, 1],
        [1, 1, 1],
        [-1, 1, 1],
        [0, -1, -1],  # Nodes 9-12 Midsides of edges 1-2, 2-3, 3-4, 4-1
        [1, 0, -1],
        [0, 1, -1],
        [-1, 0, -1],
        [0, -1, 1],  # Nodes 13-16 Midsides of edges 5-6, 6-7, 7-8, 8-5
        [1, 0, 1],
        [0, 1, 1],
        [-1, 0, 1],
        [-1, -1, 0],  # Nodes 17-20 Midsides of edges 1-5, 2-6, 3-7, 4-8
        [1, -1, 0],
        [1, 1, 0],
        [-1, 1, 0],
        [1, 0, 0],  # Nodes 21 - 26 Mid-face nodes on +r, +s, +t, -r, -s, -t
        [0, 1, 0],
        [0, 0, 1],
        [-1, 0, 0],
        [0, -1, 0],
        [0, 0, -1],
        [0, 0, 0],  # Node 27 Centroid node
    ])
    N = np.zeros(27)
    for i in range(27):
        N[i] = 0.125 * (1 + r * nodes[i, 0]) * (1 + s * nodes[i, 1]) * (1 + t * nodes[i, 2])
    return N


def resp_extrap_brick20(resp):
    """Twenty nodes brick response extrapolation from 27 Gauss points.

    Parameters
    ----------
    resp: response array, shape[n, m],
         n is the row number 8 of gauss point, column is the responses.

    Returns
    -------
    Extrapolation response.
    """
    sqrt53 = 1.2909944487358056
    shape = np.zeros((20, 27))
    nodes = sqrt53 * np.array([
        [-1, -1, -1],  # Nodes 1-4 Lower surface, counterclockwise
        [1, -1, -1],
        [1, 1, -1],
        [-1, 1, -1],
        [-1, -1, 1],  # Nodes 5-8 Upper surface, counterclockwise
        [1, -1, 1],
        [1, 1, 1],
        [-1, 1, 1],
        [0, -1, -1],  # Nodes 9-12 Midsides of edges 1-2, 2-3, 3-4, 4-1
        [1, 0, -1],
        [0, 1, -1],
        [-1, 0, -1],
        [0, -1, 1],  # Nodes 13-16 Midsides of edges 5-6, 6-7, 7-8, 8-5
        [1, 0, 1],
        [0, 1, 1],
        [-1, 0, 1],
        [-1, -1, 0],  # Nodes 17-20 Midsides of edges 1-5, 2-6, 3-7, 4-8
        [1, -1, 0],
        [1, 1, 0],
        [-1, 1, 0],
    ])
    for i in range(20):
        r, s, t = nodes[i]
        shape[i] = shape_brick27(r, s, t)
    return shape @ resp
