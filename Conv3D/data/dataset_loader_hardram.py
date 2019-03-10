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


def get_data(pocket_path='data/pockets/unique_pockets_hard/', ligand_path='data/ligands/whole_dict_embed_128.p',
             batch_size=64, num_workers=1):
    """
    Get the data Pytorch way
    :param batch_size: int
    :return:
    """

    ligands_dict = pickle.load(open(ligand_path, 'rb'))
    pockets = os.listdir(pocket_path)
    pockets_path = [pocket_path + pocket for pocket in pockets]
    pocket_embeddings = produce_items(pockets_path)

    dataset = Conv3DDatasetHardRAM(ligands_dict=ligands_dict, pockets=pockets,
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

    return train_loader, valid_loader, test_loader


""" 
Has to be outside f the class to be callable by multiprocessing.Pool
"""


def f(x):
    path_to_pdb = x
    pocket_tensor = np.load(path_to_pdb).astype(dtype=np.uint8)
    pocket_tensor = torch.from_numpy(pocket_tensor)
    pocket_tensor = pocket_tensor.float()
    return pocket_tensor


def produce_items(pockets):
    """
    produces a list of tensor in RAM so that it can be accessed directly by get item
    :return:
    """
    pool = mlt.Pool()
    result = pool.map(f, pockets, chunksize=20)
    return result


class Conv3DDatasetHardRAM(Dataset):

    def __init__(self, ligands_dict, pockets, pocket_embeddings):
        self.pockets = pockets
        self.pocket_embeddings = pocket_embeddings
        self.ligands_dict = ligands_dict

    def __len__(self):
        return len(self.pockets)

    def __getitem__(self, item):
        """
        :param item:
        :return:
        """

        pdb = self.pockets[item]

        pocket_tensor = self.pocket_embeddings[item]

        *_, ligand_id, _ = pdb.split('_')
        ligand_embedding = self.ligands_dict[ligand_id]
        ligand_embedding = torch.from_numpy(ligand_embedding)

        return pocket_tensor, ligand_embedding


if __name__ == '__main__':
    pass
    # Return the Dataset to debug it in run data
    # ds, _ = get_data(pocket_path='pockets/unique_pockets/', ligand_path='ligands/whole_dict_embed_128.p',
    #                   batch_size=4, num_workers=10)
    # print(ds[5][0].type())
