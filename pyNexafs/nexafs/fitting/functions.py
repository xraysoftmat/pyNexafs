"""
Implementations of typical NEXAFS specturm fitting functions.
"""

import numpy as np
import numpy.typing as npt
import scipy.special as spec

# -------------------------------------------------------------------
####################### Peak Functions ##############################
# -------------------------------------------------------------------


def gauss(x: npt.ArrayLike, amplitude: float, sigma: float, x0: float) -> npt.NDArray:
    """
    Gaussian function for fitting peaks in NEXAFS spectra.

    Parameters
    ----------
    x : ArrayLike
        A list of position values to evaluate the Gaussian on.
    amplitude : float
        The amplitude intensity of the Gaussian.
    sigma : float
        The standard deviation parameter of the Gaussian.
    x0 : float
        The positional offset of the Gaussian.

    Returns
    -------
    np.array
        An array of Gaussian values corresponding to `x` values.
    """
    x = np.asarray(x)
    return amplitude * np.exp((x - x0) ** 2 / sigma**2)


# -------------------------------------------------------------------
####################### Baseline Functions ##############################
# -------------------------------------------------------------------


def edge(x: npt.ArrayLike, amplitude: float, x0: float, sigma: float) -> npt.NDArray:
    """
    A step edge for capturing the elemental absorption edge.

    Combines an error function (erf) with positional and amplitude information.

    Parameters
    ----------
    x : ArrayLike
        A list of position values to evaluate the edge on.
    amplitude : float
        The amplitude intensity of the edge.
    x0 : float
        The positional offset of the erf.
    sigma : float
        The width of the erf step.

    Returns
    -------
    np.array
        An array of edge values corresponding to `x` values.
    """
    x = np.asarray(x)
    return amplitude * spec.erf(x - x0, sigma)


def edge_decaying(
    x: npt.ArrayLike, amplitude: float, x0: float, sigma: float, alpha: float
) -> npt.NDArray:
    """
    A step edge with decay for capturing an elemental absorption edge.

    Uses the same base function as `edge`, but adds an decaying exponent for x > x0.

    Parameters
    ----------
    x : ArrayLike
        A list of position values to evaluate the edge on.
    amplitude : float
        The amplitude intensity of the edge.
    x0 : float
        The positional offset of the erf.
    sigma : float
        The width of the erf step.
    alpha : float
        The magnitude of the exponential decay exponent, applied to x > x0.

    Returns
    -------
    np.array
        An array of edge values corresponding to `x` values.
    """
    x = np.asarray(x)
    y_edge = edge(x, amplitude, x0, sigma)
    # Check positive alpha
    assert alpha >= 0, "The exponent `alpha` must not be negative."

    # Find x > x0
    idxs = x > x0  # At x=x0, the amplitude is 1.
    x_decay = x[idxs]

    # Apply the decay
    y_edge[idxs] *= np.exp(-alpha * (x[idxs] - x0))

    # Return the result
    return y_edge
