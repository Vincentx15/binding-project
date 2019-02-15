import multiprocessing as mlt
import Bio.PDB
import os
import time
import csv
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt


def coords_to_eigencoords(coords, n_components=3):
    """
    PCA base decomposition for this point set, includes centering
    :param array: array of 3D coordinates [n_samples, [x,y,z]]
    :return: same format but in the eigen_base
    """
    pca = PCA(n_components=n_components)
    coords_eigen = pca.fit_transform(coords)
    # Additional module to do stats on the average size of the grid
    # ampli = (np.max(coords_eigen, axis=0) - np.min(coords_eigen, axis=0))
    # with open('../data/tensor/log', 'a') as f:
    #     writer = csv.writer(f, delimiter=',')
    #     writer.writerow(ampli)
    #     # np.savetxt(f, ampli, delimiter=',')
    return coords_eigen


# coords = np.array([[1, 2], [3, 7], [6, 5], [7, 8]], dtype=float)
# t1 = time.time()
# coords_eigen = coords_to_eigencoords(coords, n_components=2)
# print(time.time() - t1)
# coords -= np.mean(coords, axis=0)
# # centering is part of the PCA routine
# plt.scatter(coords[:, 0], coords[:, 1])
# plt.scatter(coords_eigen[:, 0], coords_eigen[:, 1])
# plt.show()


def closest_grid_points(point, grid_size=None):
    """
    TODO :For now it is only working with 1Angstrom Grid pace
    :param point: np.array
    :param grid_size: Should be the angstrom resolution
    :return:
    """
    return point.astype(int)


# FIXME : round 2.7 to 3 please
# print(closest_grid_points(np.array([-2.7, 2, 1])))

mapping = {'C': 0,
           'O': 1,
           'N': 2,
           'S': 3}


def eigen_coords_to_tensor(eigen_coords, list_labels, grid_size):
    """
    Convert coords to tensor
    :param eigen_coords: coords in eigen vector base
    :param list_labels: labels attached to each point that needs to be attached to each point in space
    :param grid_size: tuple as the shape of the 3D grid
    :return:
    """
    valid = 0
    if not len(eigen_coords) == len(list_labels):
        raise ValueError('Not every points has a label...')
    tensor = np.zeros((len(mapping), *grid_size), dtype=np.int8)
    # move data to put center in the middle of grid :
    eigen_coords += np.array(grid_size) / 2
    for i, coords in enumerate(eigen_coords):
        label = list_labels[i]
        # skip if the associated label is not in the mapping function
        try:
            label_encoding = mapping[label]
        except KeyError:
            continue

        # Get the coordinate on the grid
        grid_coords = closest_grid_points(coords, grid_size)

        # One hot encode it, if it lies out of the grid, drop it
        tensor_coords = label_encoding, *grid_coords
        try:
            tensor[tensor_coords] += 1
            valid += 1
        except IndexError:
            continue
    return tensor, valid


# coords = np.array([[1, 2, 3], [3, 7, 3], [6, 5, 5], [7, 8, 80]], dtype=float)
# coords_eigen = coords_to_eigencoords(coords, n_components=3)
# list_labels = ['O', 'C', 'T', 'N']
# tensor, valid = eigen_coords_to_tensor(coords_eigen, list_labels, (10, 10, 10))
# # print(tensor, valid)
# print(tensor.shape)
# print(np.sum(tensor))


def pdb_to_tensor(structure, grid_size, threshold=10):
    """
    Takes a pdb file as input and turn it into a tensor
    :param structure: biopython structure object
    :param grid_size: tuple shape of the tensor
    :param threshold: minimum number of points in the grid
    :return:
    """
    coords, labels = [], []
    for atom in structure.get_atoms():
        if atom.parent.get_id()[0] == ' ':
            coords.append(atom.coord)
            labels.append(atom.element)
    if len(labels) < threshold:
        # print('too few atoms overall')
        return None
    coords_eigen = coords_to_eigencoords(np.array(coords))
    tensor, valid = eigen_coords_to_tensor(coords_eigen, labels, grid_size)
    if valid < threshold:
        # print('too few atoms lying in the grid')
        return None
    return tensor


def embed_path(pdb, in_path, out_path, grid_size, threshold):
    """
    Full embedding process, from the PDB reading to the tensor creation
    :param name: element of the os.listdir
    :param in_path: path to read the structure from
    :param out_path: path to write the resulting tensor to
    :return:
    """

    parser = Bio.PDB.PDBParser(QUIET=True)
    structure = parser.get_structure('toto', os.path.join(in_path, pdb))
    tensor = pdb_to_tensor(structure, grid_size=grid_size, threshold=threshold)
    if tensor is not None:
        np.save(out_path + pdb, tensor)
        return 0
    else:
        return 1


# embed_path('1a0g_PMP_0.pdb', '../data/output_pdb/sample/', None, (10, 10, 10), 10)


def f(args):
    return embed_path(*args)


def embed_dir_parallel(in_path, out_path='../data/tensor/sample/', grid_size=(20, 15, 10), threshold=10):
    args = [(pdb, in_path, out_path, grid_size, threshold) for pdb in os.listdir(in_path)]
    pool = mlt.Pool()
    failed = pool.map(f, args)
    return failed


t1 = time.time()
failed = embed_dir_parallel('../data/output_pdb/whole/', grid_size=(42, 32, 32))
print(time.time() - t1)
# print(len(os.listdir(('../data/output_pdb/sample/no_duplicate/'))))
print(sum(failed))
