from torch.utils.data.dataset import Dataset
import torch
import numpy as np
import pickle
import os
import time
from torch.utils.data import Subset, DataLoader

'''
Third version of data loading where we have performed all the data augmentation process beforehand and now just load it 
'''


def get_data(pocket_path='data/pockets/unique_pockets_hard/', ligand_path='data/ligands/whole_dict_embed_128.p',
             batch_size=64, num_workers=1):
    """
    Get the data Pytorch way
    :param batch_size: int
    :return:
    """

    dataset = Conv3DDatasetHard(pocket_path=pocket_path, ligand_path=ligand_path)

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

    train_loader = DataLoader(dataset=train_set, shuffle=True, batch_size=batch_size, num_workers=num_workers, pin_memory=True)
    valid_loader = DataLoader(dataset=valid_set, shuffle=True, batch_size=batch_size, num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(dataset=test_set, shuffle=True, batch_size=batch_size, num_workers=num_workers, pin_memory=True)

    return train_loader, valid_loader, test_loader


class Conv3DDatasetHard(Dataset):

    def __init__(self, pocket_path, ligand_path):
        self.path = pocket_path
        self.pockets = os.listdir(pocket_path)
        self.ligands_dict = pickle.load(open(ligand_path, 'rb'))
        self.tot = 0

    def __len__(self):
        return len(self.pockets)

    def __getitem__(self, item):
        """
        :param item:
        :return:
        """
        pdb = self.pockets[item]
        # a = time.perf_counter()

        pocket_tensor = np.load(self.path + pdb).astype(dtype=np.uint8)
        pocket_tensor = torch.from_numpy(pocket_tensor)
        pocket_tensor = pocket_tensor.float()
        # self.cast += time.perf_counter() - a


        _, ligand_id, *_ = pdb.split('_')
        ligand_embedding = self.ligands_dict[ligand_id]
        ligand_embedding = torch.from_numpy(ligand_embedding)

        return pocket_tensor, ligand_embedding

if __name__ == '__main__':
    pass

    pocket_path='pockets/unique_pockets_hard/'
    ligand_path='ligands/whole_dict_embed_128.p'
    dataset = Conv3DDatasetHard(pocket_path=pocket_path, ligand_path=ligand_path)
    for i in range(500):
        a=dataset[i]
        print(a)
    print(dataset.tot)
