import os
import time
import multiprocessing as mlt
import numpy as np
import pickle
import torch

from data.utils import write_error_log

from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
from data.dataset_loader_hard import Conv3DDatasetHard

# from data.dataset_loader import get_data
# from data.dataset_loader_hard import get_data

# from data.dataset_loader_ram import get_data
# from data.dataset_loader_hardram import get_data

torch.multiprocessing.set_sharing_strategy('file_system')

"""
A script to check that all data loading will work
We need to check the productions on the fly as well as the 'hard' version.
The checking is the same for the two because it is not the rotation process that fails. 
"""


# HARD CHECK, can we open all files with the correct shape?

def find_faulty_batch(pocket_path='pockets/unique_pockets/'):
    #     The iteration process is fast through the Pytorch iteration and the error are better than with mtl

    num_workers = 20
    batch_size = 64
    ligand_path = 'ligands/whole_dict_embed_128.p'

    print('Creation : ')
    a = time.perf_counter()
    dataset = Conv3DDatasetHard(pocket_path=pocket_path, ligand_path=ligand_path)
    train_loader = DataLoader(dataset=dataset, batch_size=batch_size, num_workers=num_workers)
    print('Done in : ', time.perf_counter() - a)
    print()

    print('Use : ')
    a = time.perf_counter()

    count = 0
    for batch_idx, (inputs, labels) in enumerate(train_loader):
        if not batch_idx % 50:
            print(batch_idx)
        # count += 1
        # print(count)

    print('Done in : ', time.perf_counter() - a)
    return count


def check_path(pocket, pocket_path, shape):
    pockets_path = os.path.join(pocket_path, pocket)
    try:
        pocket_tensor = np.load(pockets_path).astype(dtype=np.uint8)
        pocket_tensor = torch.from_numpy(pocket_tensor)
        if pocket_tensor.shape != shape:
            write_error_log(pocket, csv_file='wrong_shape.csv')
    except:
        write_error_log(pocket, csv_file='wrong_pdb.csv')


def check_data(pocket_path='data/pockets/unique_pockets_hard/', shape=(4, 42, 32, 32), start_index=0, end_index=-1):
    """
    Load the data present in a path in a hard way in the ram to check if no file was corrupted
    :param pocket_path:
    :return:
    """
    pockets = os.listdir(pocket_path)[start_index:end_index]
    for i, pocket in enumerate(pockets):
        check_path(pocket, pocket_path, shape)
        if not i % 1000:
            print(i)
    return


if __name__ == '__main__':
    pass
    # find_faulty_batch(pocket_path = 'pockets/unique_pockets/')
    # find_faulty_batch(pocket_path = 'pockets/unique_pockets_hard/')

    # check_data('pockets/unique_pockets/')
    check_data('pockets/unique_pockets_hard')


'''
class Conv3DDatasetHardCheck(Dataset):
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


def check_data_load(path):
    pass

'''
