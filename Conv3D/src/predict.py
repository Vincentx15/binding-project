import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-p", "--parallel", default=True, help="If we don't want to run thing in parallel",
                    action='store_false')
parser.add_argument("-s", "--siamese", default=False, help="If we want to use siamese loading",
                    action='store_true')
parser.add_argument("-fs", "--full_siamese", default=False, help="If we want to use the full_siamese loading",
                    action='store_true')
parser.add_argument("--shuffled", default=False, help="If we want to use shuffled labels",
                    action='store_true')
parser.add_argument("-d", "--data_loading", default='hard', choices=['fly', 'hard', 'ram', 'hram'],
                    help="choose the way to load data")
parser.add_argument("-po", "--pockets", default='unique_pockets_hard',
                    choices=['unique_pockets', 'unique_pockets_hard', 'unaligned', 'unaligned_hard'],
                    help="choose the data to use for the pocket inputs")
parser.add_argument("-m", "--model", default='baby',
                    choices=['baby', 'small', 'se3cnn', 'c3d', 'small_siamese', 'baby_siamese', 'babyse3cnn'],
                    help="choose the model")
parser.add_argument("-bs", "--batch_size", type=int, default=128, help="choose the batch size")
parser.add_argument("-nw", "--workers", type=int, default=20, help="Number of workers to load data")
parser.add_argument("-wt", "--wall_time", type=int, default=None, help="Max time to run the model")
parser.add_argument("-n", "--name", type=str, default='default_name', help="Name for the logs")

args = parser.parse_known_args()[0]

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

from data.loader import Loader


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
        return pdb, pocket_tensor, 0


def all_ligand_to_pdb(path='unique_pockets'):
    """
    Get the ligand : pdb mapping and saves it in data/utils
    :return:
    """

    ligand_to_pdb = defaultdict(list)
    pathlist = os.listdir('../data/pockets/' + path)
    for i, path in enumerate(pathlist):
        pdb, ligand = path.split('_')[0:2]
        ligand_to_pdb[ligand].append(pdb)
        if not i % 1000:
            print(i)

    pickle.dump(ligand_to_pdb, open(f'../data/post_processing/utils/all_lig_to_pdb.p', 'wb'))
    print('all ligand_to_pdb done')
    return ligand_to_pdb


def ligand_to_pdb(loader):
    """
    Get the ligand : pdb mapping with the same splitting than the usual and saves it in data/utils
    :return:
    """

    ligand_to_pdb = defaultdict(list)
    for i, (pdb, inputs, labels) in enumerate(loader):
        pdb, ligand = pdb[0].split('_')[0:2]
        ligand_to_pdb[ligand].append(pdb)
        if not i % 1000:
            print(i)

    pickle.dump(ligand_to_pdb, open('../data/post_processing/utils/all_lig_to_pdb.p', 'wb'))
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
        for batch_idx, (pdb, inputs, labels) in enumerate(data_loader):
            if not batch_idx % 20:
                print(f'processed {batch_idx} batches')
            inputs = inputs.to(device)
            out = model(inputs).cpu()
            for idx, pocket in enumerate(pdb):
                predictions[pocket] = out[idx]

    return predictions


def make_predictions(model_choice, model_name, loader):
    """
    Make the prediction for a class of model, trained in a file named model_name
    Saves a dict 'path like 1a0g_PMP_0.pdb.npy : predicted 128 embedding' in predictions/model_name
    :param model_choice:
    :param model_name:
    :return:
    """

    torch.multiprocessing.set_sharing_strategy('file_system')
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # I made a mistake in the saving script
    model_path = os.path.join('trained_models', model_name, model_name + '.pth')

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
    elif model_choice == 'small_siamese':
        from models.Siamese import SmallSiamese

        model = SmallSiamese()
    elif model_choice == 'baby_siamese':
        from models.Siamese import BabySiamese

        model = BabySiamese()
    elif model_choice == 'babyse3cnn':
        from models.BabySe3cnn import BabySe3cnn

        model = BabySe3cnn()
    else:
        # Not possible because of argparse
        raise ValueError('Not a possible model')
    model.to(device)
    model = torch.nn.DataParallel(model)

    # import torch.optim as optim
    # optimizer = optim.Adam(None)
    # print(model, model_path)

    dict_results = run_model(loader, model, model_path)
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


def reduce_preds(preds, average = False):
    """
    Clean this double dict for each 8 predictions by stacking them and optionnaly averaging them (bagging)
    :param preds:
    :return:
    """
    with torch.no_grad():
        reduced = {}
        for pdb, dic_ligs in preds.items():
            for ligand, tensor in dic_ligs.items():
                # Works also if we only have one prediction
                # tensor = [torch.zeros(128)]
                pred = torch.stack(tensor, 1)
                if average:
                    pred=pred.mean(dim=1)
                # print(avg.size())
                if pdb not in reduced:
                    reduced[pdb] = {ligand: pred}
                else:
                    reduced[pdb][ligand] = pred
    return reduced


def get_distances(model_name='test_load'):
    # Get predictions
    pred_path = os.path.join('../data/post_processing/predictions/', model_name + '.p')
    predictions = pickle.load(open(pred_path, 'rb'))

    # reduce them
    rearranged = rearrange_pdb_dict(predictions)
    reduced = reduce_preds(rearranged)

    # Get true labels and lig_to_pdb mapping
    ligand_to_pdb = pickle.load(open('../data/post_processing/utils/all_lig_to_pdb.p', 'rb'))
    emb = pickle.load(open('../data/ligands/whole_dict_embed_128.p', 'rb'))

    import scipy.spatial as sp

    lig_dists = {}
    for ligand, pdb_list in ligand_to_pdb.items():
        temp = []
        true = emb[ligand]
        for pdb in pdb_list:
            if pdb not in reduced:
                continue
            pred = reduced[pdb][ligand].cpu().numpy()
            dist = sp.distance.euclidean(pred, true) ** 2
            temp.append((pdb, dist / 128))

        lig_dists[ligand] = temp
    pickle.dump(lig_dists, open(f'../data/post_processing/distances/{model_name}.p', 'wb'))
    return lig_dists


if __name__ == "__main__":
    pass
    model_choice = args.model
    model_name = args.name

    pocket_file = '../data/pockets/'
    pocket_data = args.pockets
    pocket_path = os.path.join(pocket_file, pocket_data)
    print(f'Using {pocket_path} as the pocket inputs')

    if args.data_loading == 'fly':
        augment_flips = True
        ram = False
    elif args.data_loading == 'hard':
        augment_flips = False
        ram = False
    elif args.data_loading == 'ram':
        augment_flips = True
        ram = True
    elif args.data_loading == 'hram':
        augment_flips = False
        ram = True
    else:
        raise ValueError('Not implemented this DataLoader yet')

    # batch_size = 8
    batch_size = args.batch_size
    num_workers = args.workers
    shuffled = args.shuffled
    siamese = args.siamese
    full_siamese = args.full_siamese
    if full_siamese:
        print(f'Using the full siamese pipeline, with batch size of {batch_size}')
        siamese = False
    else:
        print(f'Using batch_size of {batch_size}, {"siamese" if siamese else "serial"} loading')

    loader = Loader(pocket_path=pocket_path, ligand_path='../data/ligands/whole_dict_embed_128.p',
                    batch_size=batch_size, num_workers=num_workers, siamese=siamese, full_siamese=full_siamese,
                    augment_flips=augment_flips, ram=ram, shuffled=shuffled)
    train_loader, _, test_loader = loader.get_data()

    # make ligand_to_pdb dict
    # test_loader.dataset.debug = True
    # ligand_to_pdb(loader=test_loader)
    all_ligand_to_pdb()

    # Get prediction for the argparse arguments
    test_loader.dataset.debug = False
    make_predictions(model_choice=model_choice, model_name=model_name, loader=test_loader)

    # Get the ligands : distance distribution
    # lig_dist = get_distances('small_siamsplit_aligned_flips')
    # lig_dist = get_distances(model_name)
    # print(lig_dist)
