import sys
import os
import pickle
import gzip
from collections import Counter

import numpy as np
from tqdm import tqdm
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.spatial.distance import euclidean
from sklearn import metrics

from rdkit import Chem

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

fps = pickle.load(open('data/dude_smiles_128.p', 'rb'))
preds = pickle.load(open('data/dude/small_siamsplit_aligned_filps_dude_all.p', 'rb'))
preds = flip_group(preds)

preds = {k.split('_')[0]:v for k, v in preds.items()}

targets = {f.split('_')[0] for f in os.listdir('data/selected_data')}

tot_actives = []
tot_decoys = []
mean_diff = []
locs = []
EFS = []

i = 0
for t in tqdm(targets):
    try:
        pred = preds[t]
    except KeyError:
        continue
    fp_dict = {'actives': [], 'decoys': []}
    for condition in fp_dict:
        inf = gzip.open(f'data/selected_data/{t}_{condition}.sdf.gz')
        gzsuppl = Chem.ForwardSDMolSupplier(inf)
        for mol in gzsuppl:
            if mol is None: continue
            smile = Chem.MolToSmiles(mol)
            fp = fps[smile]
            fp_dict[condition].append(fp)
    dist_dict = {'actives': [], 'decoys': []}
    for cond, dude_fps in fp_dict.items():
        for f in dude_fps:
            dist_dict[cond].append(euclidean(pred.numpy(), f))
    merged = [('a', d) for d in dist_dict['actives']] +\
             [('d', d) for d in dist_dict['decoys']]

    merged_sort = sorted(merged, key=lambda x:x[1])

    labels = [lig[0] for lig in merged_sort]
    dists = [lig[1] for lig in merged_sort]

    norm = Counter(labels)['a'] / len(merged_sort)
    thresholds = np.arange(100, len(merged_sort), 100)
    efs = []
    for t in thresholds:
        ef = Counter(labels[:t])['a'] / t
        ef /= norm
        efs.append(ef)

    EFS.append(efs)
    # plt.plot(thresholds, efs)
    # plt.show()
    # sns.distplot(dist_dict['actives'], label='actives')
    # sns.distplot(dist_dict['decoys'], label='decoys')
    # plt.legend()
    # plt.show()

for e in EFS:
    plt.plot(e)
plt.show()
# sns.distplot(mean_diff)
# plt.show()

# plt.scatter(locs, mean_diff)
# plt.show()

# sns.distplot(tot_actives, label='actives')
# sns.distplot(tot_decoys, label='decoys')
# plt.legend()
# plt.show()
