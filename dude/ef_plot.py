import pickle
from collections import Counter

import numpy as np
from numpy import trapz
from scipy.integrate import simps
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns


def merge(actives, decoys):
    merged = [(1, d) for d in actives] +\
             [(0, d) for d in decoys]

    merged = sorted(merged, key=lambda x:x[1])
    return [m[0] for m in merged]

def get_EF(actives, decoys, thresholds):
    merged = merge(actives, decoys)
    efs = []
    N = len(merged)
    norm = sum(merged) / N
    return np.array([(sum(merged[:t])/ t) / norm for t in thresholds])

def dude_plot(dudes, split=1/1000, stop=1/50):

    EFS = []
    for dude_id, result in dudes.items():
        actives = result['actives']
        decoys = result['decoys']
        if len(actives) == 0 or len(decoys) == 0:
            continue
        N = len(actives) + len(decoys)

        cuts = [int(N*10**k) for k in np.arange(-2.5, 0, 0.1)]
        print(N, cuts)
        # thresholds = np.arange(0, int(N * stop), int(N * split))
        # thresholds = np.arange(0, int(N * stop), 1)
        thresholds =  []
        efs = get_EF(actives, decoys, cuts)
        EFS.append(efs)

    fig = plt.figure()
    ax = fig.add_subplot(2, 1, 1)
    colors = sns.color_palette("RdBu", len(EFS))


    efs_sorted = sorted(EFS, key=lambda e: trapz(e))
    for i,e in enumerate(efs_sorted):
        ax.plot(e, color=colors[i], alpha=0.6)
    efs_tot = np.mean(np.array(EFS), axis=0)
    plt.plot(efs_tot, linestyle='--', color='black')
    # ax.set_xscale('log')
    # ax.set_yscale('log')
    # ax.set_xticks(np.arange(split, stop, split))
    # ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    plt.show()

if __name__ == "__main__":
    dudes = pickle.load(open('dude_results.pickle', 'rb'))
    dude_plot(dudes)
