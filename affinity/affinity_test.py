"""
Test model on affinity benchmark data.
"""
import os
from collections import namedtuple
from collections import Counter
from scipy.spatial.distance import euclidean
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import os.path as osp

import numpy as np
from rdkit import Chem


def sdf_parse(sdf_file, fp_dict):
    """
    Extract screen data from each SDF.
    """
    suppl = Chem.SDMolSupplier(sdf_file)
    screen = []
    Ligand = namedtuple('Ligand', 'smiles affinity fp lig_id')
    for mol in suppl:
        try:
            sm = Chem.MolToSmiles(mol)
            data = mol.GetPropsAsDict()
            fp = fp_dict[sm]
            affinity = data['Enzymologic: Ki nM ']
            lig_id = data['HET ID']
            screen.append(Ligand(sm, affinity, fp, lig_id))
        except:
            pass
    return screen

def screen_score(screen, pred, dist='euclidean'):
    #sort screen ligands by similarity to predicted
    if dist == 'euclidean':
        dist = euclidean
    N = len(screen)
    affinity_sort = sorted(screen, key=lambda m:m.affinity)
    distance_sort = sorted(screen, key=lambda m:dist(pred.numpy(), m.fp))
    distance_ranks = [1 - i/N for i in range(N)]
    affinity_ranks = []
    for m in screen:
        ind = 0
        for a in affinity_sort:
            if m.smiles == a.smiles:
                affinity_ranks.append(1 - ind / N)
                break
            ind += 1

    dists = [dist(pred.numpy(), m.fp) for m in screen]
    screen_plot(dists, affinity_ranks)
    return [dist(pred.numpy(), m.fp) for m in screen], affinity_ranks
def screen_plot(distance_ranks, affinity_ranks):
    sns.regplot(distance_ranks,affinity_ranks)
    plt.xlabel("Distance Rank (1 is most similar to prediction)")
    plt.ylabel("Affinity Rank (1 is highest affinity)")
    plt.show()
    pass

def test_pockets(preds, fps, val_dir='data/validation_sets'):
    # for pdbid, lig_name in test_ids:
    dists_tot = []
    affs_tot = []
    for n in os.listdir(val_dir):
        pdbid = n.split("_")[0]
        try:
            f_name = f'{pdbid.upper()}_Validation_Affinities.sdf'
            screen = sdf_parse(osp.join(val_dir, f_name), fps)
            if not screen:
                continue
            lig = screen[0].lig_id
            pred_key = f"{pdbid.lower()}_{lig}"
            pred = preds[pred_key]
            dists, affs = screen_score(screen, pred[0])
            dists_tot.extend(dists)
            affs_tot.extend(affs)
            print("DONE")
        except KeyError:
            pass
    screen_plot(dists_tot, affs_tot)

def flip_group(preds):
    """
        Average the flips and reorganize the dict.
    """
    ids = {"_".join(k.split("_")[:2]) for k in preds.keys()}
    flip_dict = {}
    for i in ids:
        flips = []
        for flip in range(8):
            flips.append(preds[f"{i}_{flip}.npy"])
        flip_dict[i] = flips
    return flip_dict
if __name__ == "__main__":
    validation_ids = pickle.load(open('data/test_ids.pickle', 'rb'))
    fps = pickle.load(open('data/carlos_smiles_128.p', 'rb'))
    print("loading predictions")
    preds = pickle.load(open('data/preds_filter.pickle', 'rb'))
    print("loaded predictions")
    preds = flip_group(preds)
    test_pockets(preds, fps)
    pass
