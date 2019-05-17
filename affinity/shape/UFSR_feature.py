""""
Performs the PCA to find the most relevant axis and compute the features of UFSR
on the points (000),mean(100,-100), mean(010,0-10), mean(001,00-1)

Compute distances
"""


import scipy as np
import os
import csv
from sklearn.decomposition import PCA
import pandas as pd
import time


# returns the centroid of an array
def centroidnp(arr):
    length, dim = arr.shape
    sums = np.array([np.sum(arr[:, i]) for i in range(dim)])
    center = sums / length
    return center


# takes an array of positions and returns the positions of the ufsr pivots
# distance_of_points represents the distance in Angstroms of the pivots to the centroids
def find_pivots(X, distance_of_points=4):
    center = centroidnp(X)
    # make PCA
    n_components = X.shape[1]
    pca = PCA(n_components=n_components)
    pca.fit(X)
    pca_vectors = pca.components_

    # find usfr normalized eigenvectors and add them to the centroid to find the USFR starting point
    full_vectors = np.concatenate((pca_vectors, -1 * pca_vectors))
    # print(full_vectors)
    full_points = distance_of_points * full_vectors + center
    return full_points


# random distributions of 30 nodes, 1000 examples
# find pivots : 0.28 seconds ---
# dist to pivots : 0.15 seconds ---
# moments : 1.26 (6 moments), 0.55 (3), 0.33 (2) seconds ---
# encode a matrix of positions with the distribution of its moments
# number_moments is the number of moments computed, for instance, [mu, sigma] means number_moments = 2
def encode(X, number_of_moments=3):
    n_dimension = X.shape[1]
    center = centroidnp(X)
    full_points = find_pivots(X)

    # get distance matrix where rows are pivot points and columns are data points
    dist_to_centroid = np.array([[np.linalg.norm(X[j] - center) for j in range(X.shape[0])]])
    dist_matrix = np.spatial.distance_matrix(full_points, X)
    # code à la main, 4 fois plus lent
    # dist_matrix = np.array(
    #     [[np.linalg.norm(X[j] - full_point) for j in range(X.shape[0])] for full_point in full_points])

    # aggregate the symetric pivots
    dist_ufsr = np.array([(dist_matrix[i] + dist_matrix[n_dimension + i]) * 0.5 for i in range(n_dimension)])

    # add the distance to center of mass, dist_ufsr is now a matrix with distribution of distances wrt
    # (000),mean(100,-100), mean(010,0-10), mean(001,00-1) as rows
    dist_ufsr = np.concatenate((dist_to_centroid, dist_ufsr))

    # get the features
    means = np.array([[np.mean(dist_ufsr[i]) for i in range(n_dimension + 1)]])
    means = np.transpose(means)
    moments = np.array(
        [[np.stats.moment(dist_ufsr[i], j) for j in range(2, number_of_moments + 1)] for i in range(n_dimension + 1)])
    ufsr_feature = np.concatenate((means, moments), axis=1)

    # mean, moment2, moment3..., moment6, mean, moment1... for each pivot
    ufsr_feature = ufsr_feature.ravel()

    return ufsr_feature


# returns the similarity measure of two arrays
def usfr_similarity(x, y):
    x = np.asarray(x)
    y = np.asarray(y)
    try:
        similarity_measure = 1 / (1 + np.sum(abs(x - y)))
    except:
        return None
    return similarity_measure


# copied from scipy documentation
def distance_matrix(x, y, output_name=0):
    x = np.asarray(x)
    m, k = x.shape
    y = np.asarray(y)
    n, kk = y.shape
    result = np.empty((m, n), dtype=float)  # FIXME: figure out the best dtype
    for i in range(m):
        result[i, :] = [usfr_similarity(x[i], y_j) for y_j in y]
        print(result[i, :])
        if output_name:
            with open(os.path.join('../data/output_csv', output_name + '.csv'), "a", newline='') as fp:
                wr = csv.writer(fp, dialect='excel')
                wr.writerow(result[i, :])
    return result


'''
apply these computations to data
'''


def build_colnames(moments_kept, centroids):
    # build the names of the columns
    lists = [['m' + str(i + 1) + 'c' + str(j + 1) for i in range(moments_kept)] for j in range(centroids)]
    L = []
    for list in lists:
        L += list
    colnames = ['pdb', 'ligand'] + L
    return colnames


def csv_reader(input_name):
    df = pd.read_csv(os.path.join('../data/output_csv', input_name + '.csv'))
    return df


# keep only a few moments from a pandas df
# returns a df with name, ligand and moments compressed
def reduce_number_of_moment(df, moments_after, moments_before=3, centroids=4):
    # keep the name and ligand, then select only the first moments for each feature
    criteria = [True, True] + [(i % moments_before < moments_after) for i in range(moments_before * centroids)]
    reduced_df = df[df.columns[criteria]]
    return reduced_df


def DM_from_csv(input_name, output_name, moments_kept=0):
    df = csv_reader(input_name)
    if moments_kept:
        df = reduce_number_of_moment(df, moments_kept)
    values = df[df.columns[2:]]
    DM = distance_matrix(values, values, output_name)


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    X = np.array([[2, 1], [5, 2], [1, 1], [2, 2], [3, 2], [5, 5], [6, 6.57]])

    X_x = X[:, 0]
    X_y = X[:, 1]
    center = centroidnp(X)
    full_points = find_pivots(X)
    usfr_x = full_points[:, 0]
    usfr_y = full_points[:, 1]

    plt.scatter(X_x, X_y, alpha=0.5)
    plt.scatter(center[0], center[1], alpha=0.5)
    plt.scatter(usfr_x, usfr_y, alpha=0.5)
    plt.show()


'''
plot



x = X[:, 0]
y = X[:, 1]

fig, ax = plt.subplots(figsize=(10, 5), ncols=3, nrows=2)

left = 0.125  # the left side of the subplots of the figure
right = 0.9  # the right side of the subplots of the figure
bottom = 0.1  # the bottom of the subplots of the figure
top = 0.9  # the top of the subplots of the figure
wspace = .5  # the amount of width reserved for blank space between subplots
hspace = 1.1  # the amount of height reserved for white space between subplots

# This function actually adjusts the sub plots using the above paramters
plt.subplots_adjust(
    left=left,
    bottom=bottom,
    right=right,
    top=top,
    wspace=wspace,
    hspace=hspace
)

y_title_margin = 0.8
ax[0][0].set_title("zero", y=y_title_margin)
ax[0][1].set_title("first", y=y_title_margin)
ax[0][2].set_title("second", y=y_title_margin)

sns.distplot(dist_ufsr[0], ax=ax[0][0], kde=False, rug=True)
sns.distplot(dist_ufsr[1], ax=ax[0][1], kde=False, rug=True)
sns.distplot(dist_ufsr[2], ax=ax[0][2], kde=False, rug=True)

plt.show()


usfr_x = full_points[:, 0]
usfr_y = full_points[:, 1]
plt.scatter(x, y, alpha=0.5)
plt.scatter(center[0], center[1], alpha=0.5)
plt.scatter(usfr_x, usfr_y, alpha=0.5)
plt.show()

# print(preprocessing.normalize(pca.components_, norm='l2'))
'''
