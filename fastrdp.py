import numpy as np

import _fastrdp as _c


def pldist(point, start, end):
    """
    Calculates the distance from ``point`` to the line given
    by the points ``start`` and ``end``.

    :param point: a point
    :type point: numpy array
    :param start: a point of the line
    :type start: numpy array
    :param end: another point of the line
    :type end: numpy array
    """
    if np.all(np.equal(start, end)):
        return np.linalg.norm(point - start)

    return np.divide(
        np.abs(np.linalg.norm(np.cross(end - start, start - point))),
        np.linalg.norm(end - start),
    )


def rdp(
    x: np.ndarray,
    eps: float = 0.0,
    dist_func=None,
    algo: str = "iter",
    return_mask: bool = False,
):
    x_f8 = x.astype("f8")
    y = _c.rdp(x_f8, eps, dist_func, algo, return_mask)
    if np.issubdtype(x.dtype, np.integer):
        y = y.round()
    y = y.astype(x.dtype)
    return y
