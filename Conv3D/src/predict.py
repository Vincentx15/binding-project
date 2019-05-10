import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-m", "--model", default='baby',
                    choices=['baby', 'small', 'se3cnn', 'c3d'],
                    help="kind of models to use")
parser.add_argument("-n", "--name", type=str, default='test_load', help="Name of the trained model to use")
args = parser.parse_args()

import numpy as np
import time
import torch
import sys, os
import pickle
from collections import defaultdict
from torch.utils.data.dataset import Dataset
from torch.utils.data import Subset, DataLoader

if __name__ == "__main__":
    sys.path.append('../')

"""
Class for prediction, we want to know our model prediction on the dataset. Siamese ones will average the results

Usecase :  run the prediction on the model of your choice using python src.predict -m=model_choice -mp=model_name
Then read the resulting dictionnary

"""


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
        return pdb, pocket_tensor


def ligand_to_pdb():
    """
    Get the ligand : pdb mapping with the same splitting than the usual and saves it in data/utils
    :return:
    """
    dataset = Evaluation(debug=True)
    n = len(dataset)
    np.random.seed(0)
    split_train, split_valid = 0.7, 0.85
    train_index, valid_index = int(split_train * n), int(split_valid * n)
    train_index, valid_index = train_index - train_index % 8, valid_index - valid_index % 8

    from data.loader import BatchSampler

    indices = [item for sublist in BatchSampler(8, n) for item in sublist]

    test_indices = indices[valid_index:]
    test_set = Subset(dataset, test_indices)
    test_loader = DataLoader(dataset=test_set,
                             batch_sampler=BatchSampler(8, len(test_indices)),
                             num_workers=20)

    ligand_to_pdb = defaultdict(list)
    for i, item in enumerate(test_loader):
        pdb, ligand = item[0].split('_')[0:2]
        ligand_to_pdb[ligand].append(pdb)
        if not i % 1000:
            print(i)

    pickle.dump(ligand_to_pdb, open('../data/post_processing/utils/lig_to_pdb.p', 'wb'))
    print('ligand_to_pdb done')
    return ligand_to_pdb


# ligand_to_pdb()
# print(ligand_to_pdb)

def run_model(data_loader, model, model_weights_path):
    """
    :param data_loader: an iterator on input data
    :param model: An empty model
    :param optimizer: An empty optimizer
    :param model_weights_path: the path of the model to load
    :return: list of predictions
    """
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(model_weights_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    # optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    model.eval()

    predictions = dict()
    with torch.no_grad():
        for batch_idx, (id, inputs) in enumerate(data_loader):
            if not batch_idx%20:
                print(f'processed {batch_idx} batches')
            inputs = inputs.to(device)
            out = model(inputs).cpu()
            for idx, pocket in enumerate(id):
                predictions[pocket] = out[idx]

    return predictions


def make_predictions(model_choice, model_name):
    """
    Make the prediction for a class of model, trained in a file named model_name
    Saves a dict 'path like 1a0g_PMP_0.pdb.npy : predicted 128 embedding' in predictions/model_name
    :param model_choice:
    :param model_name:
    :return:
    """
    batch_size = 8
    num_workers = 20

    torch.multiprocessing.set_sharing_strategy('file_system')
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    dataset = Evaluation()
    n = len(dataset)
    np.random.seed(0)
    split_train, split_valid = 0.7, 0.85
    train_index, valid_index = int(split_train * n), int(split_valid * n)
    train_index, valid_index = train_index - train_index % 8, valid_index - valid_index % 8

    from data.loader import BatchSampler

    indices = [item for sublist in BatchSampler(batch_size, n) for item in sublist]

    train_indices = indices[:train_index]
    valid_indices = indices[train_index:valid_index]
    test_indices = indices[valid_index:]

    train_set = Subset(dataset, train_indices)
    valid_set = Subset(dataset, valid_indices)
    test_set = Subset(dataset, test_indices)

    train_loader = DataLoader(dataset=train_set,
                              batch_sampler=BatchSampler(batch_size, len(train_indices)),
                              num_workers=num_workers)
    valid_loader = DataLoader(dataset=valid_set,
                              batch_sampler=BatchSampler(batch_size, len(valid_indices)),
                              num_workers=num_workers)
    test_loader = DataLoader(dataset=test_set,
                             batch_sampler=BatchSampler(batch_size, len(test_indices)),
                             num_workers=num_workers)

    # I made a mistake in the saving script
    model_path = os.path.join('../trained_models', model_name, model_name + '.pth')

    if model_choice == 'baby':
        from models.BabyC3D import BabyC3D

        # from models.BabyC3D import Crazy

        model = BabyC3D()
        # model = Crazy()
    elif model_choice == 'small':
        from models.SmallC3D import SmallC3D

        model = SmallC3D()
    elif model_choice == 'se3cnn':
        from models.Se3cnn import Se3cnn

        model = Se3cnn()
    elif model_choice == 'c3d':
        from models.C3D import C3D

        model = C3D()
    else:
        # Not possible because of argparse
        raise ValueError('Not a possible model')
    model.to(device)
    model = torch.nn.DataParallel(model)

    # import torch.optim as optim
    # optimizer = optim.Adam(None)
    # print(model, model_path)

    dict_results = run_model(test_loader, model, model_path)
    pickle.dump(dict_results, open(f'../data/post_processing/predictions/{model_name}.p', 'wb'))
    return dict_results


def rearrange_pdb_dict(preds):
    """
    Put the prediction in a double dict format
    :param preds:
    :return:
    """
    rearranged = {}
    for key, value in preds.items():
        # print(key)
        # print(key.split('_')[:2])
        pdb, lig = key.split('_')[:2]
        if pdb not in rearranged:
            rearranged[pdb] = defaultdict(list)
        rearranged[pdb][lig].append(value)
    return rearranged


def reduce_preds(preds):
    """
    Average this double dict for each 8 predictions
    :param preds:
    :return:
    """
    with torch.no_grad():
        reduced = {}
        for pdb, dic_ligs in preds.items():
            for ligand, tensor in dic_ligs.items():
                # Works also if we only have one prediction
                # tensor = [torch.zeros(128)]
                avg = torch.stack(tensor, 1).mean(dim=1)
                # print(avg.size())
                if pdb not in reduced:
                    reduced[pdb] = {ligand: avg}
                else:
                    reduced[pdb][ligand] = avg
    return reduced


def get_distances(model_name='test_load'):
    # Get predictions
    pred_path = os.path.join('../data/post_processing/predictions/', model_name+'.p')
    predictions = pickle.load(open(pred_path, 'rb'))

    # reduce them
    rearranged = rearrange_pdb_dict(predictions)
    reduced = reduce_preds(rearranged)

    # Get true labels and lig_to_pdb mapping
    ligand_to_pdb = pickle.load(open('../data/post_processing/utils/lig_to_pdb.p', 'rb'))
    emb = pickle.load(open('../data/ligands/whole_dict_embed_128.p', 'rb'))

    import scipy.spatial as sp

    lig_dists = {}
    for ligand, pdb_list in ligand_to_pdb.items():
        temp = []
        true = emb[ligand]
        for pdb in pdb_list:
            pred = reduced[pdb][ligand].cpu().numpy()
            dist = sp.distance.euclidean(pred, true)**2
            temp.append((pdb, dist / 128))
        lig_dists[ligand] = temp
    pickle.dump(lig_dists, open(f'../data/post_processing/distances/{model_name}.p', 'wb'))
    return lig_dists


if __name__ == "__main__":
    pass
    model_choice = args.model
    model_name = args.name

    # make ligand_to_pdb dict
    # ligand_to_pdb()

    # Get prediction for the argparse arguments
    # make_predictions(model_choice=model_choice, model_name=model_name)

    # Get the ligands : distance distribution
    lig_dist = get_distances(model_name)
    print(lig_dist)
