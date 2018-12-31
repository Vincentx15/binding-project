from numpy import abs, array, concatenate, sqrt, zeros
from scipy.special import cbrt
from scipy.stats.stats import skew

"""
USRCAT as implemented by A. Schreyer, with some tuning to make the comparison with our tool easier
"""


def distance_to_point(coords, point):
    """
    Returns an array containing the distances of each coordinate in the input
    coordinates to the input point.
    """
    return sqrt(((coords - point) ** 2).sum(axis=1))


def usr_moments(coords):
    """
    Calculates the USR moments for a set of input coordinates as well as the four
    USR reference atoms.

    :param coords: numpy.ndarray
    """
    # centroid of the input coordinates
    ctd = coords.mean(axis=0)

    # get the distances to the centroid
    dist_ctd = distance_to_point(coords, ctd)

    # get the closest and furthest coordinate to/from the centroid
    cst, fct = coords[dist_ctd.argmin()], coords[dist_ctd.argmax()]

    # get the distance distributions for the points that are closest/furthest
    # to/from the centroid
    dist_cst = distance_to_point(coords, cst)
    dist_fct = distance_to_point(coords, fct)

    # get the point that is the furthest from the point that is furthest from
    # the centroid
    ftf = coords[dist_fct.argmax()]
    dist_ftf = distance_to_point(coords, ftf)

    # calculate the first three moments for each of the four distance distributions
    moments = concatenate([(ar.mean(), ar.std(), cbrt(skew(ar)))
                           for ar in (dist_ctd, dist_cst, dist_fct, dist_ftf)])

    # return the USR moments as well as the four points for later re-use
    return (ctd, cst, fct, ftf), moments


def usr_moments_with_existing(coords, ref_points):
    """
    Calculates the USR moments for a set of coordinates and an already existing
    set of four USR reference points.
    """
    ctd, cst, fct, ftf = ref_points
    dist_ctd = distance_to_point(coords, ctd)
    dist_cst = distance_to_point(coords, cst)
    dist_fct = distance_to_point(coords, fct)
    dist_ftf = distance_to_point(coords, ftf)

    moments = concatenate([(ar.mean(), ar.std(), cbrt(skew(ar)))
                           for ar in (dist_ctd, dist_cst, dist_fct, dist_ftf)])

    return moments
