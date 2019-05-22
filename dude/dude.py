import sys
import os
import pickle
import gzip
from collections import Counter
from collections import defaultdict

import numpy as np
from tqdm import tqdm
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.spatial.distance import euclidean
from sklearn import metrics

from rdkit import Chem

dude_dir = '../affinity/data/all'
dude_codes = os.listdir(dude_dir)

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
        flip_dict[i] = torch.stack(flips, 1).mean(dim=1)
    return flip_dict

fps = pickle.load(open('data/dude_smiles_128_whole.p', 'rb'))
preds = pickle.load(open('data/dude_mol2/small_siamsplit_aligned_flips_dude_all.p', 'rb'))
# preds = pickle.load(open('data/dude/small_siamsplit_aligned_filps_dude_all.p', 'rb'))
preds = flip_group(preds)

#rename preds keys to just have dude code
preds = {k.split('_')[0]:v for k, v in preds.items()}

tot_actives = []
tot_decoys = []
mean_diff = []
locs = []
EFS = []
dude_results = defaultdict(dict)

i = 0
for t in tqdm(dude_codes):
    try:
        pred = preds[t]
    except KeyError:
        continue
    fp_dict = {'actives': [], 'decoys': []}
    for condition in fp_dict:
        try:
            inf = gzip.open(f'{dude_dir}/{t}/{condition}_final.sdf.gz')
        except:
            print("MISSING DUDE")
            continue
        gzsuppl = Chem.ForwardSDMolSupplier(inf)
        lost_smiles = 0
        for mol in gzsuppl:
            if mol is None: continue
            smile = Chem.MolToSmiles(mol)
            try:
                fp = fps[smile]
            except KeyError:
                lost_smiles += 1
                continue
            fp_dict[condition].append(fp)

    #merge all ligands and sort by dist to prediction
    all_dists = []
    dist_dict = defaultdict(list)
    for cond, dude_fps in fp_dict.items():
        for f in dude_fps:
            # dist = euclidean(pred.numpy(), f)
            dist = euclidean(fp_dict['actives'][0], f)
            all_dists.append((cond, dist))
            dist_dict[cond].append(dist)

        dist_dict['actives'] = dist_dict['actives'][1:]
    dude_results[t] = dist_dict

    # all_dists_sort = sorted(all_dists, key=lambda x:x[1])

    # labels = [lig[0] for lig in all_dists_sort]
    # dists = [lig[1] for lig in all_dists_sort]

    # N = len(all_dists_sort)
    # if N == 0:
        # continue
    # num_actives = Counter(labels)['actives']
    # num_decoys = Counter(labels)['decoys']
    # print(f"num actives: {num_actives}")
    # print(f"num decoys: {num_decoys}")
    # norm = num_actives / N
    # thresholds = np.arange(int(N/100), N, int(N/100))
    # efs = []
    # for t in thresholds:
        # ef = Counter(labels[:t])['actives'] / t
        # ef /= norm
        # efs.append(ef)

    # # EFS.append(efs)
    # plt.plot(thresholds, efs)
    # plt.show()
    # sns.distplot(dist_dict['actives'], label='actives')
    # sns.distplot(dist_dict['decoys'], label='decoys')
    # plt.legend()
    # plt.show()

pickle.dump(dude_results, open('dude_results_active_control.pickle', 'wb'))
# pickle.dump(EFS, open('efs.pickle', 'wb'))

# fig = plt.figure()
# ax = fig.add_subplot(2, 1, 1)

# for e in EFS:
    # ax.plot(e)
# ax.set_xscale('log')
# plt.show()
# # sns.distplot(mean_diff)
# # plt.show()

# plt.scatter(locs, mean_diff)
# plt.show()

# sns.distplot(tot_actives, label='actives')
# sns.distplot(tot_decoys, label='decoys')
# plt.legend()
# plt.show()
