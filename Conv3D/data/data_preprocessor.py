import shutil
import os
import time
import multiprocessing as mlt
import numpy as np
import pickle
import csv
import torch
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
import sys

if __name__ == "__main__":
    sys.path.append('../')
from data.utils import write_error_log

"""
Script to :
- Remove duplicates
- Check the data
- Create an augmented dataset
"""


def remove_duplicates(src_dir='pockets/whole', dst_dir='pockets/unique_pockets'):
    """
    Script to remove duplicates : pockets from the same pdb, that have the same ligand
    :param src_dir:
    :param dst_dir:
    :return:
    """
    os.mkdir(dst_dir)

    seen = set()
    for i, item in enumerate(os.listdir(src_dir)):
        try:
            pdb, ligand = item.split('_')[0:2]
        except:
            continue
        if (pdb, ligand) not in seen:
            src = os.path.join(src_dir, item)
            shutil.copy(src, dst_dir)
            seen.add((pdb, ligand))
        if not i % 1000:
            print(i)


def rotate(tensor, i):
    """
    :param tensor: the tensor to rotate, a numpy tensor
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
    tensor_flipped = np.flip(tensor, axis=axes)
    return tensor_flipped


def augment_data(x):
    """
    :param x: path , pdb form
    :return:
    """
    out_path, path, pdb = x
    try:
        path_to_pdb = os.path.join(path, pdb)
        pdb_id, ligand_id, *_ = pdb.split('_')
        pocket_tensor = np.load(path_to_pdb).astype(dtype=np.uint8)
        # To debug
        # pocket_tensor = np.random.rand(2, 2, 2, 2)

        for rotation in range(8):
            pocket_tensor_rotated = rotate(pocket_tensor, rotation)
            save_path = os.path.join(out_path, pdb_id + '_' + ligand_id + '_' + str(rotation))
            np.save(save_path, pocket_tensor_rotated)
    except:
        write_error_log(pdb, csv_file='failed_hard.csv')


def make_hard_enumeration(path='pockets/unique_pockets/', out_path='pockets/unique_pockets_hard/'):
    try:
        os.mkdir(out_path)
    except FileExistsError:
        raise ValueError('This name is already taken !')

    a = time.perf_counter()
    inputs = [(out_path, path, pdb) for pdb in os.listdir(path)]

    pool = mlt.Pool()
    pool.map(augment_data, inputs, chunksize=20)
    print('Done in : ', time.perf_counter() - a)


def make_hard_enumeration_serial(start_index=0, end_index=-1):
    path = 'pockets/unique_pockets/'
    out_path = 'pockets/unique_pockets_hard/'

    # try:
    #     os.mkdir(out_path)
    # except FileExistsError:
    #     raise ValueError('This name is already taken !')

    inputs = [(out_path, path, pdb) for pdb in os.listdir(path)[start_index:end_index]]
    for i, path in enumerate(inputs):
        try:
            augment_data(path)
        except:
            write_error_log(path)
        if not i % 1000:
            print(i)


torch.multiprocessing.set_sharing_strategy('file_system')


class Conv3DDatasetHardCheck(Dataset):
    """
    Read data and if it fails, correct it
    """

    def __init__(self, pocket_path, ligand_path, augment_flips, shape=(4, 42, 32, 32)):
        self.path = pocket_path
        self.shape = shape
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
            try:
                pocket_tensor = np.load(os.path.join(self.path, pdb)).astype(dtype=np.uint8)
                pocket_tensor = torch.from_numpy(pocket_tensor)
                pocket_tensor = pocket_tensor.float()
                if pocket_tensor.shape != self.shape:
                    write_error_log(pdb, csv_file='wrong_shape.csv')
            except:
                pocket_tensor = torch.zeros(self.shape)
                write_error_log(pdb, 'failed.csv')

        else:
            try:
                pdb = self.pockets[item]
                # a = time.perf_counter()
                pocket_tensor = np.load(os.path.join(self.path, pdb)).astype(dtype=np.uint8)
                pocket_tensor = torch.from_numpy(pocket_tensor)
                # self.cast += time.perf_counter() - a
            except:
                pocket_tensor = torch.zeros(self.shape, dtype=torch.uint8)
                write_error_log(pdb, 'failed.csv')

        _, ligand_id, *_ = pdb.split('_')
        ligand_embedding = self.ligands_dict[ligand_id]
        ligand_embedding = torch.from_numpy(ligand_embedding)

        return pdb, pocket_tensor, ligand_embedding



def check_data_load(pocket_path='pockets/unique_pockets/', ligand_path='ligands/whole_dict_embed_128.p',
                    augment_flips=False):
    """
    Leverage Pytorch fast iteration with nice traceback to check the dataset fast
    :param pocket_path:
    :param ligand_path:
    :param augment_flips:
    :return:
    """
    num_workers = 20
    batch_size = 64

    print('Creation : ')
    a = time.perf_counter()

    dataset = Conv3DDatasetHardCheck(pocket_path=pocket_path, ligand_path=ligand_path, augment_flips=augment_flips)
    train_loader = DataLoader(dataset=dataset, batch_size=batch_size, num_workers=num_workers)

    print('Done in : ', time.perf_counter() - a)
    print()

    print('Use : ')
    a = time.perf_counter()

    for batch_idx, _ in enumerate(train_loader):
        if not batch_idx % 20:
            print(batch_idx, time.perf_counter() - a)
            a = time.perf_counter()
    print('Done in : ', time.perf_counter() - a)


def remove_faulty_from_csv(directory, csv_path):
    """
    Use the previous screening to remove pdb that don't behave
    :param directory: path of the directory to remove
    :param csv: path of the csv to read
    :return:
    """
    with open(csv_path, "r") as csv_file:
        reader = csv.reader(csv_file)
        for pdb in reader:
            os.remove(os.path.join(directory, pdb[0]))


def test_loader(pocket_file='pockets/',
                pocket_data='unique_pockets_hard',
                batch_size=64,
                num_workers=20,
                augment_flips=False,
                ram=False
                ):
    """
    Test to load the data in a certain way for 2 epochs to see if everything works
    :param pocket_file:
    :param pocket_data:
    :param batch_size:
    :param num_workers:
    :param augment_flips:
    :param ram:
    :return:
    """
    import os
    from .loader import Loader

    pocket_path = os.path.join(pocket_file, pocket_data)

    loader = Loader(pocket_path=pocket_path, ligand_path='ligands/whole_dict_embed_128.p',
                    batch_size=batch_size, num_workers=num_workers,
                    augment_flips=augment_flips, ram=ram, siamese=True, debug=True)
    train_loader, valid_loader, test_loader = loader.get_data()

    print('Created data loader')

    a = time.perf_counter()
    for epoch in range(2):
        print(epoch)
        print(len(train_loader))
        for batch_idx, (pdb, inputs, labels) in enumerate(train_loader):
            # print(f'{batch_idx} points ')
            # raise ValueError
            if not batch_idx % 100:
                pass
                print(batch_idx * batch_size, ' on ', len(train_loader) * batch_size, 'train', pdb[:17])
                # print(batch_idx, time.perf_counter() - a)
                # a = time.perf_counter()

        for batch_idx, (pdb, inputs, labels) in enumerate(valid_loader):
            if not batch_idx % 50:
                pass
                print(batch_idx * batch_size, ' on ', len(valid_loader) * batch_size, 'valid', pdb[:17])
                # print(batch_idx, time.perf_counter() - a)
                # a = time.perf_counter()

        for batch_idx, (pdb, inputs, labels) in enumerate(test_loader):
            if not batch_idx % 50:
                pass
                print(batch_idx * batch_size, ' on ', len(test_loader) * batch_size, 'test', pdb[:17])
                # print(batch_idx, time.perf_counter() - a)
                # a = time.perf_counter()

    print('Done in : ', time.perf_counter() - a)


if __name__ == '__main__':
    pass
    # remove_duplicates()
    # check_data_load('pockets/unaligned')
    # make_hard_enumeration(path='pockets/unaligned', out_path='unaligned_hard')
    # check_data_load('pockets/unaligned')
    # remove_faulty_from_csv('pockets/unaligned', 'failed.csv')

    test_loader()
