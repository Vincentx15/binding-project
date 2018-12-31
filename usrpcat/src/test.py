import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import csv

# fig, ax = plt.subplots(figsize=(10, 5), ncols=3, nrows=1)
#
# left = 0.125  # the left side of the subplots of the figure
# right = 0.9  # the right side of the subplots of the figure
# bottom = 0.1  # the bottom of the subplots of the figure
# top = 0.9  # the top of the subplots of the figure
# wspace = .5  # the amount of width reserved for blank space between subplots
# hspace = 1.1  # the amount of height reserved for white space between subplots
#
# # This function actually adjusts the sub plots using the above paramters
# plt.subplots_adjust(
#     left=left,
#     bottom=bottom,
#     right=right,
#     top=top,
#     wspace=wspace,
#     hspace=hspace
# )


# y_title_margin = 0.8
# plt.ylim(0, 20)
# ax[0].set_title("EF = 0.25", y=y_title_margin)
# ax[1].set_title("EF = 0.5", y=y_title_margin)
# ax[2].set_title("EF = 1", y=y_title_margin)

# csv_path = '../data/output_csv/whole_separated_sorted.csv'
# df = pd.read_csv(csv_path, header=0)
# # df = df.iloc[[method =='manhattan' for method in df['metrique']]]
#
# # ['FF', 'TF', 'TT', 'GM', 'HM']
# # print( list(df['embedding']))
# subset = ['HM']
# df = df.iloc[[embedding in subset for embedding in df['embedding']]]
# df = df.drop(['embedding'],axis=1)
# df = df.rename(columns={"metrique": "metric"})
# print(df)
#
#
# ar = np.array(df[df.columns[:-1]])
# means = [np.mean(list/ar[0]) for list in ar]
# print(means)
#
#
# df = pd.melt(df, id_vars="metric", var_name="EF", value_name="enrichment")
# df = df.sort_values(['EF'])
# ax = sns.factorplot (x='metric', y="enrichment", col='EF', data=df, kind='bar')
# ax.savefig("Third_plot.png")

# a = sns.factorplot(x='embedding', y="EF=0.25", data=df, kind='bar', ax=ax[0])
# b = sns.factorplot(x='embedding', y="EF=0.5", data=df, kind='bar', ax=ax[1])
# c = sns.factorplot(x='embedding', y="EF=1", data=df, kind='bar', ax=ax[2])

# plt.show()


from rdkit import Chem

sdf_path = '../data/structures/aa2ar/actives_final.sdf'
suppl = Chem.SDMolSupplier(sdf_path)
i = 0
coords = {}
for molecule in suppl:
    if i > 0:
        break
    i += 1

    for conformer in molecule.GetConformers():
        # get the coordinates of all atoms

        for atom in molecule.GetAtoms():
            point = conformer.GetAtomPosition(atom.GetIdx())
            coords[atom.GetIdx()] = (point.x, point.y, point.z)
L = []
for coord in coords.values():
    L.append(coord)
X = np.asarray(L)

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

from numpy import abs, array, concatenate, sqrt, zeros
from scipy.special import cbrt
from scipy.stats.stats import skew

def distance_to_point(coords, point):
    """
    Returns an array containing the distances of each coordinate in the input
    coordinates to the input point.
    """
    return sqrt(((coords-point)**2).sum(axis=1))

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
    return (ctd,cst,fct,ftf), moments


X = X[:,:2]
X_x = X[:, 0]
X_y = X[:, 1]

center = centroidnp(X)
full_points = find_pivots(X)
old_points = np.array(usr_moments(X)[0])

usfr_x = full_points[:, 0]
usfr_y = full_points[:, 1]

old_x = old_points[:, 0]
old_y = old_points[:, 1]

plt.scatter(X_x, X_y, alpha=0.5)
plt.scatter(center[0], center[1], alpha=0.5)
plt.scatter(usfr_x, usfr_y, alpha=0.5)
plt.scatter(old_x, old_y, alpha=0.5)

plt.show()
# df = df.sort_values(['EF'])
# ax = sns.factorplot(x='method', y="enrichment", hue='EF', data=df, kind='bar', ax=ax[0])

