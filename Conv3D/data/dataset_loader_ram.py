from torch.utils.data.dataset import Dataset
import torch
import numpy as np
import pickle
import os
import time
import multiprocessing as mlt
from torch.utils.data import Subset, DataLoader

'''
Data creation could preexist and is currently the bottleneck for the gpu, so try a version where we load everything and create everything
'''


def get_data(pocket_path='data/pockets/unique_pockets/', ligand_path='data/ligands/whole_dict_embed_128.p',
             batch_size=64, num_workers=1):
    """
    Get the data Pytorch way
    :param batch_size: int
    :return:
    """
    ligands_dict = pickle.load(open(ligand_path, 'rb'))
    pockets_rotations = produce_list_pockets(pocket_path)
    print('number of points in total', len(pockets_rotations))
    pockets_rotations = pockets_rotations[:1000]
    pocket_embeddings = produce_items(pockets_rotations)

    dataset = Conv3DDatasetRAM(ligands_dict=ligands_dict,
                               pockets_rotations=pockets_rotations,
                               pocket_embeddings=pocket_embeddings)

    n = len(dataset)
    indices = list(range(n))
    np.random.seed(0)
    np.random.shuffle(indices)
    split_train, split_valid = 0.7, 0.85

    train_indices = indices[:int(split_train * n)]
    valid_indices = indices[int(split_train * n):int(split_valid * n)]
    test_indices = indices[int(split_valid * n):]

    train_set = Subset(dataset, train_indices)
    valid_set = Subset(dataset, valid_indices)
    test_set = Subset(dataset, test_indices)

    train_loader = DataLoader(dataset=train_set, shuffle=True, batch_size=batch_size, num_workers=num_workers)
    valid_loader = DataLoader(dataset=valid_set, shuffle=True, batch_size=batch_size, num_workers=num_workers)
    test_loader = DataLoader(dataset=test_set, shuffle=True, batch_size=batch_size, num_workers=num_workers)

    return dataset, train_loader, valid_loader, test_loader


def rotate(tensor, i):
    """
    :param tensor: the tensor to rotate, a pytorch tensor
    :param i: integer between 0 and 7 the rotation is encoded as follow 7 in base 2 return a4 + b2 + c1 with abc
    indicating the presence of a rotation eg : 6 = 4 + 2 = 110 so it represents the flip along the first two axis
    :return: flipped tensor
    """
    if i == 0:
        return tensor
    assert -1 < i < 8
    axes = []
    if i >= 4:
        axes.append(1)
        i -= 4
    if i >= 2:
        axes.append(2)
        i -= 2
    if i > 0:
        axes.append(3)
    tensor = torch.flip(tensor, dims=axes)
    return tensor


# a = np.array([[1, 2], [3, 4], [5, 6]])
# a = np.flip(a, axis=0)
# print(a)
# rotate(np.random.rand(2, 2, 2, 2), 5)

""" 
Has to be outside f the class to be callable by multiprocessing.Pool
"""


def produce_list_pockets(path):
    """
    Produce the list of ordered rotations with their path, to be called by the pool
    :param path:
    :return:
    """
    pockets_rotations = [(path + pdb, rotation) for pdb in os.listdir(path) for rotation in range(8)]
    return pockets_rotations


def f(x):
    path_to_pdb, rotation = x
    pocket_tensor = np.load(path_to_pdb).astype(dtype=np.uint8)
    pocket_tensor = torch.from_numpy(pocket_tensor)
    pocket_tensor = rotate(pocket_tensor, rotation)
    pocket_tensor = pocket_tensor.float()
    return path_to_pdb, rotation, pocket_tensor


def produce_items(pockets_rotations):
    """
    produces a list of tensor in RAM so that it can be accessed directly by get item
    :return:
    """
    pool = mlt.Pool()
    result = pool.map(f, pockets_rotations)
    return result


class Conv3DDatasetRAM(Dataset):

    def __init__(self, ligands_dict, pockets_rotations, pocket_embeddings):
        self.ligands_dict = ligands_dict
        self.pockets_rotations = pockets_rotations
        self.pocket_embeddings = pocket_embeddings

    def __len__(self):
        return len(self.pockets_rotations)

    def __getitem__(self, item):
        """
        :param item:
        :return:
        """
        pocket_tensor = self.pocket_embeddings[item]

        pdb, rotation = self.pockets_rotations[item]
        *_, ligand_id, _ = pdb.split('_')
        ligand_embedding = self.ligands_dict[ligand_id]
        ligand_embedding = torch.from_numpy(ligand_embedding)
        return pocket_tensor, ligand_embedding


if __name__ == '__main__':
    pass
    ds, *_ = get_data()
    print(ds.pockets_rotations[5])
    print(ds[5])
