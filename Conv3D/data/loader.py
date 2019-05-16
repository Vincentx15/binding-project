import numpy as np
import pickle
import os
import multiprocessing as mlt
import torch
from torch.utils.data.dataset import Dataset
from torch.utils.data import Subset, DataLoader
from torch.utils.data import Sampler
import threading as th

'''
Data generation process, we can choose to read a preprocessed version or to augment it on the fly
One can also choose to load everything in RAM to speed up the process
One can use siamese of even full siamese loading where the batch is respectively composed of flips groups or even of
batches of objects of shape 8,4,42,32,32 that have to be dealt with differently
'''


class SiameseSampler(Sampler):
    """
    provide samples of 8 consecutive values for ordered inputs
    """

    def __init__(self, batch_size, size):
        """

        :param batch_size: len of the original batch size
        :param size: len of the total inputs
        """
        # super(Sampler, self).__init__()
        assert batch_size % 8 == 0
        self.batch_size_pdb = int(batch_size / 8)
        self.size = int(size / 8)
        self.indices = np.array(list(range(self.size)))
        self.current = 0
        # print('New BS')
        self.epoch = 0
        self.lock = th.Lock()

        print(size)

    def __len__(self):
        return 8 * self.size

    def create_list(self):
        np.random.shuffle(self.indices)
        full = [pdb * 8 + rotation for pdb in self.indices for rotation in range(8)]
        # print(len(full), full)
        return full

    def __iter__(self):
        list = self.create_list()
        return iter(list)


class BatchSampler(Sampler):
    """
    provide samples of 8 consecutive values for ordered inputs
    """

    def __init__(self, batch_size, size):
        """

        :param batch_size: len of the original batch size
        :param size: len of the total inputs
        """
        # super(Sampler, self).__init__()
        assert batch_size % 8 == 0
        self.batch_size = batch_size
        self.batch_size_pdb = int(batch_size / 8)
        self.size = int(size / 8)
        self.indices = np.array(list(range(self.size)))
        self.current = 0
        # print('New BS')
        # self.epoch = 0
        # self.lock = th.Lock()

    def __len__(self):
        # We do the approximation that we have always one batch even if we only have 100 points and the BS is 128
        return self.size // self.batch_size_pdb + 1

    def create_list(self):
        full = [pdb * 8 + rotation for pdb in self.indices for rotation in range(8)]
        full_batches = [full[i:i + self.batch_size] for i in range(0, len(full), self.batch_size)]
        np.random.shuffle(self.indices)
        # print(full_batches)
        return full_batches

    def __iter__(self):
        list = self.create_list()
        return iter(list)


class Loader():
    def __init__(self,
                 pocket_path='data/pockets/unique_pockets_hard/',
                 ligand_path='data/ligands/whole_dict_embed_128.p',
                 batch_size=128,
                 num_workers=20,
                 augment_flips=False,
                 ram=False,
                 siamese=False,
                 debug=False,
                 shuffled=False,
                 full_siamese=False):
        """
        Wrapper class to call with all arguments and that returns appropriate data_loaders
        :param pocket_path:
        :param ligand_path:
        :param batch_size:
        :param num_workers:
        :param augment_flips: perform numpy flips
        :param ram: store whole thing in RAM
        :param siamese: for the batch siamese technique
        :param full_siamese for the true siamese one
        """
        self.siamese = siamese
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.dataset = self.create_dataset(pocket_path=pocket_path,
                                           ligand_path=ligand_path,
                                           augment_flips=augment_flips,
                                           ram=ram,
                                           debug=debug,
                                           shuffled=shuffled,
                                           full_siamese=full_siamese)
        self.augment_flips = augment_flips
        self.full_siamese = full_siamese
        assert (full_siamese != siamese or siamese is False)

    @staticmethod
    def create_dataset(pocket_path, ligand_path, full_siamese=False, augment_flips=False, ram=False, debug=False,
                       shuffled=False):
        if full_siamese:
            return Conv3DDatasetSiamese(pocket_path=pocket_path, ligand_path=ligand_path, debug=debug,
                                        shuffled=shuffled)
        elif ram:
            return Conv3DDatasetRam(pocket_path=pocket_path, ligand_path=ligand_path, augment_flips=augment_flips,
                                    debug=debug, shuffled=shuffled)
        else:
            return Conv3DDatasetHard(pocket_path=pocket_path, ligand_path=ligand_path, augment_flips=augment_flips,
                                     debug=debug, shuffled=shuffled)

    def get_data(self):
        n = len(self.dataset)

        # indices = list(range(n))
        # np.random.shuffle(indices)

        np.random.seed(0)
        split_train, split_valid = 0.7, 0.85
        train_index, valid_index = int(split_train * n), int(split_valid * n)
        # print(train_index, valid_index)
        # print(train_index, valid_index)

        # If we precompute the augmentations, we need to split the different flips in different subsets
        if self.augment_flips or self.full_siamese:
            indices = list(range(n))
        else:
            indices = [item for sublist in BatchSampler(self.batch_size, n) for item in sublist]
            train_index, valid_index = train_index - train_index % 8, valid_index - valid_index % 8

        train_indices = indices[:train_index]
        valid_indices = indices[train_index:valid_index]
        test_indices = indices[valid_index:]

        train_set = Subset(self.dataset, train_indices)
        valid_set = Subset(self.dataset, valid_indices)
        test_set = Subset(self.dataset, test_indices)

        if self.siamese:
            train_loader = DataLoader(dataset=train_set,
                                      batch_sampler=BatchSampler(self.batch_size, len(train_indices)),
                                      num_workers=self.num_workers)
            valid_loader = DataLoader(dataset=valid_set,
                                      batch_sampler=BatchSampler(self.batch_size, len(valid_indices)),
                                      num_workers=self.num_workers)
            test_loader = DataLoader(dataset=test_set,
                                     batch_sampler=BatchSampler(self.batch_size, len(test_indices)),
                                     num_workers=self.num_workers)
            return train_loader, valid_loader, test_loader

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

    def __init__(self, pocket_path, ligand_path, augment_flips,
                 debug, shuffled):
        self.debug = debug
        self.path = pocket_path
        if not shuffled:
            self.ligands_dict = pickle.load(open(ligand_path, 'rb'))
            print('not shuffled data')

        else:
            import random
            ligands_dict = pickle.load(open(ligand_path, 'rb'))
            keys = list(ligands_dict.keys())
            random.shuffle(keys)
            self.ligands_dict = dict(zip(keys, ligands_dict.values()))
            print('shuffled data')

        self.augment_flips = augment_flips
        self.pockets = sorted(os.listdir(pocket_path))
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
        if self.debug:
            return self.pockets[item], 0, 0

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
            # pocket_tensor=0
            # self.cast += time.perf_counter() - a

        _, ligand_id, *_ = pdb.split('_')
        ligand_embedding = self.ligands_dict[ligand_id]
        ligand_embedding = torch.from_numpy(ligand_embedding)

        return pdb, pocket_tensor, ligand_embedding


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

    def __init__(self, pocket_path, ligand_path, augment_flips, debug, shuffled):
        self.debug = debug

        if not shuffled:
            self.ligands_dict = pickle.load(open(ligand_path, 'rb'))
            print('not shuffled data')

        else:
            import random
            ligands_dict = pickle.load(open(ligand_path, 'rb'))
            keys = list(ligands_dict.keys())
            random.shuffle(keys)
            self.ligands_dict = dict(zip(keys, ligands_dict.values()))
            print('shuffled data')

        self.count = 0
        self.path = pocket_path
        self.augment_flips = augment_flips

        self.pockets = sorted(os.listdir(pocket_path))
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

        if self.debug:
            pdb, rotation = self.pockets_rotations[item]
            *_, ligand_id, _ = pdb.split('_')
            return pdb, ligand_id, pdb
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

        return pdb, pocket_tensor, ligand_embedding


class Conv3DDatasetSiamese(Dataset):
    """
    Uses an unaugmented dataset and does the flipping on the fly (probably faster than loading everything anyway)
    """

    def __init__(self, pocket_path, ligand_path, debug, shuffled):
        self.debug = debug
        self.path = pocket_path
        if not shuffled:
            self.ligands_dict = pickle.load(open(ligand_path, 'rb'))
            print('not shuffled data')

        else:
            import random
            ligands_dict = pickle.load(open(ligand_path, 'rb'))
            keys = list(ligands_dict.keys())
            random.shuffle(keys)
            self.ligands_dict = dict(zip(keys, ligands_dict.values()))
            print('shuffled data')
        self.pockets = sorted(os.listdir(pocket_path))

    def __len__(self):
        return len(self.pockets)

    def __getitem__(self, item):
        """
        :param item:
        :return:
        """
        if self.debug:
            return self.pockets[item],0, 0

        pdb = self.pockets[item]

        pocket_tensor = np.load(os.path.join(self.path, pdb)).astype(dtype=np.uint8)
        pocket_tensor = torch.from_numpy(pocket_tensor)
        res = [pocket_tensor]

        for i in range(1, 8):
            pass
            res.append(flip(pocket_tensor, i))

        tensor = torch.stack(res)
        # print(tensor.size())
        tensor = tensor.float()
        # self.cast += time.perf_counter() - a

        _, ligand_id, *_ = pdb.split('_')
        ligand_embedding = self.ligands_dict[ligand_id]
        ligand_embedding = torch.from_numpy(ligand_embedding)

        return pdb, tensor, ligand_embedding


class Evaluation(Dataset):
    def __init__(self, pocket_path='../data/pockets/unique_pockets_hard', debug=False):
        self.path = pocket_path
        self.pockets = sorted(os.listdir(pocket_path))
        self.debug = debug

    def __len__(self):
        return len(self.pockets)

    def __getitem__(self, item):
        """
        :param item:
        :return:
        """
        pdb = self.pockets[item]
        if self.debug:
            return pdb
        # a = time.perf_counter()
        pocket_tensor = np.load(os.path.join(self.path, pdb)).astype(dtype=np.uint8)
        pocket_tensor = torch.from_numpy(pocket_tensor)
        pocket_tensor = pocket_tensor.float()
        return pdb, pocket_tensor, 0


if __name__ == '__main__':
    import time

    # batch_size = 1
    # num_workers = 1
    # print('Creation : ')
    # a = time.perf_counter()
    # train_loader, _, _ = get_data(pocket_path='pockets/unique_pockets',
    #                               ligand_path='ligands/whole_dict_embed_128.p',
    #                               batch_size=batch_size, num_workers=num_workers, shuffled=False)
    #
    # print('Done in : ', time.perf_counter() - a)
    # print()
    #
    # print('Use : ')
    # a = time.perf_counter()
    #
    # for batch_idx, (inputs, labels) in enumerate(train_loader):
    #     if not batch_idx % 20:
    #         print(batch_idx, time.perf_counter() - a)
    #         a = time.perf_counter()
    # print('Done in : ', time.perf_counter() - a)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    batch_size = 16
    num_workers = 4
    print('Creation : ')
    a = time.perf_counter()
    loader = Loader(pocket_path='pockets/unique_pockets_hard', ligand_path='ligands/whole_dict_embed_128.p',
                    batch_size=batch_size, num_workers=num_workers, debug=False, siamese=False)

    # loader = Loader(pocket_path='pockets/unique_pockets_hard', ligand_path='ligands/whole_dict_embed_128.p',
    #                 batch_size=batch_size, num_workers=num_workers, augment_flips=False, ram=False)
    print(len(loader.dataset))
    train_loader, _, test_loader = loader.get_data()

    print('Done in : ', time.perf_counter() - a)
    print()

    print('Use : ')
    a = time.perf_counter()

    loop = time.perf_counter()
    tot=list()
    for batch_idx, (pdb, inputs, labels) in enumerate(train_loader):
        tot.extend(pdb)
        inputs.to(device)
        if not batch_idx % 20:
            print(batch_idx, time.perf_counter() - loop)
            loop = time.perf_counter()
    print('Done in : ', time.perf_counter() - a)
    print(sorted(tot))
