import scipy as np
from sklearn.decomposition import PCA
from scipy.special import cbrt
from scipy.stats.stats import skew
from scipy.stats import gmean, hmean
import timeit

"""
Tweak the implementation of the USRPCAT benchmark in the ligand task to be able to use the PDB pockets as inputs
"""


def find_pivots(X, distance=4):
    """
    :param X: array of positions, shape : 'number of points, dimension'
    :param distance: distance in Angstroms of the pivots to the centroids
    :return: positions of the ufsr pivots as array of coordinates
    """
    center = np.mean(X)
    # make PCA
    n_components = X.shape[1]
    pca = PCA(n_components=n_components)
    pca.fit(X)
    pca_vectors = pca.components_

    # find usfr normalized eigenvectors and add them to the centroid to find the USFR starting point
    full_vectors = np.concatenate((pca_vectors, -1 * pca_vectors))
    # print(full_vectors)
    full_points = distance * full_vectors + center
    return full_points


# geometrical mean
def geometrical_mean(x, y):
    x = np.asarray(x)
    y = np.asarray(y)
    x = np.vstack((x, y))
    m = gmean(x, axis=0)
    return m


# geometrical mean
def harmonical_mean(x, y):
    x = np.asarray(x)
    y = np.asarray(y)
    x = np.vstack((x, y))
    m = hmean(x, axis=0)
    return m


# CHANGE HERE THE MEAN TO USE


def usr_moments_with_existing(coords, ref_points, number_of_moments=3, mean=0):
    """

    :param coords:
    :param ref_points:
    :param number_of_moments:
    :param mean: index in [np.mean, geometrical_mean, harmonical_mean]
    :return:
    """
    n_dimension = coords.shape[1]
    center = np.mean(coords)
    # get distance matrix where rows are pivot points and columns are data points
    dist_to_centroid = np.array([[np.linalg.norm(coords[j] - center) for j in range(coords.shape[0])]])
    dist_matrix = np.spatial.distance_matrix(ref_points, coords)

    # aggregate the symetric pivots
    if mean not in [0, 1, 2]:
        mean = 0
    mean_options = [np.mean, geometrical_mean, harmonical_mean]
    mean = mean_options[mean]
    dist_ufsr = np.array([mean(dist_matrix[i], dist_matrix[n_dimension + i])
                          for i in range(n_dimension)])

    # add the distance to center of mass, dist_ufsr is now a matrix with distribution of distances wrt
    # (000),mean(100,-100), mean(010,0-10), mean(001,00-1) as rows
    dist_ufsr = np.concatenate((dist_to_centroid, dist_ufsr))

    # get the features
    means = np.array([[np.mean(dist_ufsr[i]) for i in range(n_dimension + 1)]])
    means = np.transpose(means)
    # moments = np.array(
    #     [[np.stats.moment(dist_ufsr[i], j) for j in range(2, number_of_moments + 1)] for i in range(n_dimension + 1)])

    # FIXME : VARIANCE VS STANDARD DEVIATION include also other moments to use number_of_moments
    moments = np.array([[dist_ufsr[i].std(), cbrt(skew(dist_ufsr[i]))] for i in range(n_dimension + 1)])
    ufsr_feature = np.concatenate((means, moments), axis=1)

    # mean, moment2, moment3..., moment6, mean, moment1... for each pivot
    ufsr_feature = ufsr_feature.ravel()
    return ufsr_feature


def usr_moments(coords, number_of_moments=3, mean=0, distance=4):
    """
    encode a matrix of positions with the distribution of its moments
    uses annex function

    random distributions of 30 nodes, 1000 examples
    find pivots : 0.28 seconds ---
    dist to pivots : 0.15 seconds ---
    moments : 1.26 (6 moments), 0.55 (3), 0.33 (2) seconds ---

    :param coords: input
    :param number_of_moments: number of moments computed, for instance, [mu, sigma] means number_moments = 2
    :param mean: index in [np.mean, geometrical_mean, harmonical_mean]
    """
    ref_points = find_pivots(coords, distance)
    ufsr_feature = usr_moments_with_existing(coords, ref_points,
                                             number_of_moments=number_of_moments,
                                             mean=mean)
    return ref_points, ufsr_feature


def parse_pdb():
    pass
