import shutil
import os
import time
import multiprocessing as mlt
import numpy as np
import csv
from data.utils import write_error_log


def remove_duplicates():
    # Script to remove duplicates : pockets from the same pdb, that have the same ligand

    src_dir = 'pockets/whole'
    dst_dir = 'pockets/unique_pockets'
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


def make_hard_enumeration():
    path = 'pockets/unique_pockets/'
    out_path = 'pockets/unique_pockets_hard/'

    try:
        os.mkdir(out_path)
    except FileExistsError:
        raise ValueError('This name is already taken !')

    a = time.perf_counter()
    inputs = [(out_path, path, pdb) for pdb in os.listdir(path)]

    pool = mlt.Pool()
    pool.map(f, inputs, chunksize=20)
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
            f(path)
        except:
            write_error_log(path)
        if not i % 1000:
            print(i)


def f(x):
    """
    :param x: path , pdb form
    :return:
    """
    out_path, path, pdb = x
    try:
        path_to_pdb = path + pdb
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





if __name__ == '__main__':
    pass
    # x = 'pockets/unique_pockets_hard/', 'pockets/unique_pockets/', '1bbo_ABA_36.pdb.npy'
    # f(x)
    # make_hard_enumeration()
    make_hard_enumeration_serial()
