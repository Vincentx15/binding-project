import time
import numpy as np
import re
import pickle

'''
Get the X
'''


def median_graph(DM):
    """
    Rkketurn median graph as:
        median(D) = argmin_g1 \sum_{g2} d(g1, g2).

    Graph whose distance to all other graphs is minimal.
    """
    return np.argmin(np.sum(DM, axis=1))


def spanning_selection(DM, m):
    median = median_graph(DM)

    proto_indices = [median]
    tot_ind = list(np.arange(len(DM)))
    d_indices = set(tot_ind) - {median}

    # get point furthest from prototype set.
    while len(proto_indices) < m:
        proto = np.argmax(
            np.ma.array([min([DM[i][p] for p in proto_indices])
                         for i in tot_ind],
                        mask=np.isin(tot_ind, proto_indices)
                        ))
        proto_indices.append(proto)
    return proto_indices


# DM = pickle.load(open('data/delta_DM.pickle', "rb"))
# print(DM.shape)
#
# t1 = time.time()
# selected = spanning_selection(DM, 190)
# print(time.time() - t1)
#
# embeddings = DM[:, selected]
# np.save('processed/pockets',embeddings)

# pockets = np.load('processed/pockets.npy')
# print(pockets)
# print(pockets.shape)


'''
Get the Y
'''


def dict_smiles(path='data/all_smiles.sm'):
    """
    Create a dict that maps those id to smiles
    :return:
    """
    with open(path) as f:
        res = {}
        for line in f:
            try:
                smiles, id, *_ = line.split()
                res[id] = smiles
            except:
                pass
    return res


def label_smiles(pickled_labels_path):
    """
    Go from the path of a pickled list of name containing ids to a list of smiles (the Carlos way of naming)
    :param pickled_labels_path:
    :return:
    """
    # get raw_labels as string like : '1aju_ARG_B.nxpickle'
    raw_labels = pickle.load(open(pickled_labels_path, "rb"))
    # print(raw_labels)
    # print(len(raw_labels))

    # fetch only the pdb id
    label_id = []
    for label in raw_labels:
        pattern = re.compile("_")
        l = pattern.finditer(label)
        indexes = [match.start() for match in l]
        ligand_id = label[indexes[0] + 1:indexes[1]]
        label_id.append(ligand_id)
    # print(label_id)

    res = dict_smiles()

    #  Map the results to smiles
    label_smiles = [res[label] for label in label_id]
    # print(label_smiles)
    return label_smiles

# labels_id = 'data/delta_graphlist.pickle'
# smiles_li = label_smiles(labels_id)
# print(smiles_li)
# print(len(smiles_li))

# now this smiles list can be used as a feed to the AE and we get an embedding list
# labels = np.load('processed/embed-512.npy')
# print(labels)
# print(labels.shape)
