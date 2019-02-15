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
    Create a dict that maps id 'ACE'... to smiles
    :return: the dict
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


# res = dict_smiles()
# print(res['ML5'])
# print(len(res))


'''
The whole id:smiles dict is 28000 long but the data only contains 3365 ligands so we create 
list or sub-dict to avoid useles cddd computations
'''


def list_smiles(path='data/whole_ligands_id.p'):
    """
    Get the whole list of bound protein into a set of ids, then turn it into smiles
    :return:
    """
    whole_list_id = pickle.load(open(path, 'rb'))
    res = dict_smiles()
    whole_list_smiles = []
    failed = []
    for id in whole_list_id:
        try:
            smiles = res[id]
            whole_list_smiles.append(smiles)
        except KeyError:
            failed.append(id)
    return whole_list_smiles, failed


# whole_list_smiles, failed = list_smiles()
# print(whole_list_smiles)
# print(len(whole_list_smiles))
# print(failed)
# pickle.dump(whole_list_smiles, open('data/whole_ligands_smiles.p', 'wb'))


def subdict_smiles(path='data/whole_ligands_id.p'):
    """
    Subset the full dictionnary to only keep the relevant entries
    :return:
    """
    whole_list_id = pickle.load(open(path, 'rb'))
    print(whole_list_id)
    res = dict_smiles()
    subdict = {id: res[id] for id in set(whole_list_id).intersection(res.keys())}

    return subdict


# Compare it to the full one
# whole_dict_smiles = subdict_smiles()
# print(whole_dict_smiles['ACE'], len(whole_dict_smiles))
# full = dict_smiles()
# print(full['ACE'], len(full))
# pickle.dump(whole_dict_smiles, open('data/whole_dict_smiles.p', 'wb'))


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

# Carlos way : read the list of files and extract their id, then turn it into smiles
# labels_id = 'data/delta_graphlist.pickle'
# smiles_li = label_smiles(labels_id)
# print(smiles_li)
# print(len(smiles_li))

# now this smiles list can be used as a feed to the AE and we get an embedding list
# labels = np.load('processed/embed-512.npy')
# print(labels)
# print(labels.shape)
