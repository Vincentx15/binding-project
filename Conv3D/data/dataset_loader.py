from torch.utils.data.dataset import Dataset
import torch
import numpy as np
import pickle
import os
import time
from torch.utils.data import Subset, DataLoader

'''
The data loading creation pipeline is not very weird except from the generating process for our data:
To avoid storing all the rotation we create them on the fly (not too long)
Therefore, we cannot split the dataset in the usual way to avoid putting all the rotational augmentation 
in the same subset
'''


def get_data(pocket_path='data/pockets/unique_pockets/', ligand_path='data/ligands/whole_dict_embed_128.p',
             batch_size=64, num_gpu=1):
    """
    Get the data Pytorch way
    :param batch_size: int
    :return:
    """

    dataset = Conv3DDataset(pocket_path=pocket_path, ligand_path=ligand_path)

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

    train_loader = DataLoader(dataset=train_set, shuffle=True, batch_size=batch_size, num_workers=num_gpu * 4)
    valid_loader = DataLoader(dataset=valid_set, shuffle=True, batch_size=batch_size, num_workers=num_gpu * 4)
    test_loader = DataLoader(dataset=test_set, shuffle=True, batch_size=batch_size, num_workers=num_gpu * 4)

    return train_loader, valid_loader, test_loader


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


class Conv3DDataset(Dataset):

    def __init__(self, pocket_path, ligand_path):
        self.path = pocket_path
        self.pockets = os.listdir(pocket_path)
        self.pockets_rotations = [(pdb, rotation) for pdb in self.pockets for rotation in range(8)]
        self.ligands_dict = pickle.load(open(ligand_path, 'rb'))

    def __len__(self):
        return len(self.pockets_rotations)

    def __getitem__(self, item):
        """
        When timed, we get :
        dataset.time_load : 0.07545638084411621
        dataset.time_convert : 0.0014777183532714844
        dataset.time_rotate : 0.1373603343963623
        dataset.time_load_ligands : 0.0013277530670166016
        So the rotation could be faster but would still have a comparable runtime of having to load the matrix.
        Maybe with a sparse implementation, we could shorten the loading of the npy file and the rotation would
        become the real bottleneck implying we need to do it as a preprocessing step
        :param item:
        :return:
        """
        pdb, rotation = self.pockets_rotations[item]

        pocket_tensor = np.load(self.path + pdb).astype(dtype=np.uint8)
        pocket_tensor = torch.from_numpy(pocket_tensor)
        pocket_tensor = rotate(pocket_tensor, rotation)
        pocket_tensor = pocket_tensor.float()

        _, ligand_id, *_ = pdb.split('_')
        ligand_embedding = self.ligands_dict[ligand_id]
        ligand_embedding = torch.from_numpy(ligand_embedding)

        return pocket_tensor, ligand_embedding
