from torch.utils.data.dataset import Dataset
import torch
import numpy as np
import pickle
import os
import multiprocessing as mlt
from torch.utils.data import Subset, DataLoader

'''
Data generation process, we can choose to read a preprocessed version or to augment it on the fly
One can also choose to load everything in RAM to speed up the process
'''


class Loader():

    def __init__(self,
                 pocket_path='data/pockets/unique_pockets_hard/',
                 ligand_path='data/ligands/whole_dict_embed_128.p',
                 batch_size=128,
                 num_workers=20,
                 augment_flips=False,
                 ram=False):
        """
        Wrapper class to call with all arguments and that returns appropriate data_loaders
        :param pocket_path:
        :param ligand_path:
        :param batch_size:
        :param num_workers:
        :param augment_flips: perform numpy flips
        :param ram: store whole thing in RAM
        """
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.dataset = self.create_dataset(pocket_path=pocket_path,
                                           ligand_path=ligand_path,
                                           augment_flips=augment_flips,
                                           ram=ram)

    def create_dataset(self, pocket_path, ligand_path, augment_flips=False, ram=False):
        if ram:
            return Conv3DDatasetRam(pocket_path=pocket_path, ligand_path=ligand_path, augment_flips=augment_flips)
        else:
            return Conv3DDatasetHard(pocket_path=pocket_path, ligand_path=ligand_path, augment_flips=augment_flips)

    def get_data(self):
        n = len(self.dataset)
        indices = list(range(n))
        np.random.seed(0)
        np.random.shuffle(indices)
        split_train, split_valid = 0.7, 0.85

        train_indices = indices[:int(split_train * n)]
        valid_indices = indices[int(split_train * n):int(split_valid * n)]
        test_indices = indices[int(split_valid * n):]

        train_set = Subset(self.dataset, train_indices)
        valid_set = Subset(self.dataset, valid_indices)
        test_set = Subset(self.dataset, test_indices)

#        train_loader = DataLoader(dataset=train_set, batch_size=self.batch_size,
 #                                 num_workers=self.num_workers)
        train_loader = DataLoader(dataset=train_set, shuffle=True, batch_size=self.batch_size,
                                 num_workers=self.num_workers)
        valid_loader = DataLoader(dataset=valid_set, shuffle=True, batch_size=self.batch_size,
                                  num_workers=self.num_workers)
        test_loader = DataLoader(dataset=test_set, shuffle=True, batch_size=self.batch_size,
                                 num_workers=self.num_workers)

        return train_loader, valid_loader, test_loader


def flip(tensor, i):
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


class Conv3DDatasetHard(Dataset):
    """
    Construct the data on the fly
    """

    def __init__(self, pocket_path, ligand_path, augment_flips):
        self.path = pocket_path
        self.ligands_dict = pickle.load(open(ligand_path, 'rb'))
        self.augment_flips = augment_flips
        self.pockets = os.listdir(pocket_path)
        if self.augment_flips:
            self.pockets_rotations = [(pdb, rotation) for pdb in self.pockets for rotation in range(8)]

    def __len__(self):
        if self.augment_flips:
            return len(self.pockets_rotations)
        return len(self.pockets)

    def __getitem__(self, item):
        """
        :param item:
        :return:
        """
        if self.augment_flips:
            pdb, rotation = self.pockets_rotations[item]
            pocket_tensor = np.load(os.path.join(self.path, pdb)).astype(dtype=np.uint8)
            pocket_tensor = torch.from_numpy(pocket_tensor)
            pocket_tensor = flip(pocket_tensor, rotation)
            pocket_tensor = pocket_tensor.float()
            # self.cast += time.perf_counter() - a

        else:
            pdb = self.pockets[item]
            # a = time.perf_counter()
            pocket_tensor = np.load(os.path.join(self.path, pdb)).astype(dtype=np.uint8)
            pocket_tensor = torch.from_numpy(pocket_tensor)
            pocket_tensor = pocket_tensor.float()
            # self.cast += time.perf_counter() - a

        _, ligand_id, *_ = pdb.split('_')
        ligand_embedding = self.ligands_dict[ligand_id]
        ligand_embedding = torch.from_numpy(ligand_embedding)

        return pocket_tensor, ligand_embedding


def read(x):
    path_to_pdb = x
    pocket_tensor = np.load(path_to_pdb).astype(dtype=np.uint8)
    pocket_tensor = torch.from_numpy(pocket_tensor)
    return pocket_tensor


def read_and_flip(x):
    path_to_pdb, rotation = x
    pocket_tensor = np.load(path_to_pdb).astype(dtype=np.uint8)
    pocket_tensor = torch.from_numpy(pocket_tensor)
    pocket_tensor = flip(pocket_tensor, rotation)
    return pocket_tensor


class Conv3DDatasetRam(Dataset):
    """
    Loads the whole data we will iterate on in the RAM and returns a Dataset object to access it
    """

    def __init__(self, pocket_path, ligand_path, augment_flips):
        self.ligands_dict = pickle.load(open(ligand_path, 'rb'))

        self.count = 0
        self.path = pocket_path
        self.augment_flips = augment_flips

        self.pockets = os.listdir(pocket_path)
        self.path_to_pocket = [os.path.join(self.path, pocket) for pocket in self.pockets]
        if self.augment_flips:
            self.pockets_rotations = [(pdb_path, rotation) for pdb_path in self.path_to_pocket for rotation in range(8)]
        self.pocket_embeddings = self.generate_pocket_embeddings(augment_flips=augment_flips)

    def __len__(self):
        if self.augment_flips:
            return len(self.pockets_rotations)
        return len(self.pockets)

    def generate_pocket_embeddings(self, augment_flips):
        """
        Load all data on the fly and store it in class variable
        :param augment_flips:
        :return:
        """
        pool = mlt.Pool()
        if augment_flips:
            result = pool.map(read_and_flip, self.pockets_rotations, chunksize=20)
        else:
            result = pool.map(read, self.path_to_pocket, chunksize=20)
        return result

    def __getitem__(self, item):
        """
        :param item:
        :return:
        """
        # self.count += 1
        # print(f'{self.count} calls')
        # We need to be cautious here to avoid messing with the order of the tokens
        if self.augment_flips:
            pocket_tensor = self.pocket_embeddings[item]
            pocket_tensor = pocket_tensor.float()

            pdb, rotation = self.pockets_rotations[item]
            *_, ligand_id, _ = pdb.split('_')

        else:
            pocket_tensor = self.pocket_embeddings[item]
            pocket_tensor = pocket_tensor.float()

            pdb = self.pockets[item]

            *_, ligand_id, _ = pdb.split('_')

        ligand_embedding = self.ligands_dict[ligand_id]
        ligand_embedding = torch.from_numpy(ligand_embedding)

        return pocket_tensor, ligand_embedding


if __name__ == '__main__':
    import time

    batch_size = 4
    num_workers = 1
    print('Creation : ')
    a = time.perf_counter()
    loader = Loader(pocket_path='pockets/unique_pockets_hard', ligand_path='ligands/whole_dict_embed_128.p',
                    batch_size=batch_size, num_workers=num_workers, augment_flips=False, ram=True)
    print(len(loader.dataset))
    train_loader, _, test_loader = loader.get_data()

    print('Done in : ', time.perf_counter() - a)
    print()


    print('Use : ')
    a = time.perf_counter()

    for batch_idx, (inputs, labels) in enumerate(train_loader):
        if not batch_idx % 20:
            print(batch_idx, time.perf_counter() - a)
            a = time.perf_counter()
    print('Done in : ', time.perf_counter() - a)
