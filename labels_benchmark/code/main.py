import time
import numpy as np
import re
import pickle
from sklearn import preprocessing
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, GRU
from sklearn.model_selection import KFold
from sklearn.decomposition import PCA
from sklearn.dummy import DummyClassifier
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPRegressor, MLPClassifier
import scipy
from scipy.stats import wilcoxon
from skbio.stats.distance import mantel

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

    # get point furtherst from prototype set.
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


def label_smiles(pickled_labels_path):
    """
    Go from the path of a pickled list of name containing ids to a list of smiles
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

    # Create a dict that maps those id to smiles
    with open('data/all_smiles.sm') as f:
        res = {}
        for line in f:
            try:
                smiles, id, *_ = line.split()
                res[id] = smiles
            except:
                pass
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

'''
Do models
'''


def NN(input_dim=190, output_dim=166, layers_intermediate=[180]):
    """
    build NN with specified parameters
    """
    layers = layers_intermediate + [output_dim]  # List of int
    activations = ['relu', 'linear']  # List of strings
    dropout = 0.3  # Float
    loss = 'mean_squared_error'  # String
    optimizer = 'sgd'  # String

    # Create model
    m = Sequential()
    m.add(Dense(units=layers[0], activation=activations[0], input_dim=input_dim))
    for i in range(1, len(layers)):
        if dropout != 0:
            m.add(Dropout(dropout))
            m.add(Dense(units=layers[i], activation=activations[i]))
    m.compile(loss=loss, optimizer=optimizer)
    return m


def unison_shuffled_copies(a, b, seed=3):
    """
    Shuffle two lists in the same way
    :param a:
    :param b:
    :return:
    """
    a = np.asarray(a)
    b = np.asarray(b)
    assert len(a) == len(b)
    np.random.seed(seed)
    p = np.random.permutation(len(a))
    return a[p], b[p]


# a = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
# b = list(reversed(a))
# c, d = unison_shuffled_copies(a, b)
# print(a, b)
# print(c, d)


def ES(labels, pred, threshold):
    """
    Enrichment score : computes the distance from predicted label to all labels
    :return:
    """
    res = sorted([(np.linalg.norm(pred - label), label) for i, label in enumerate(labels)], key=lambda x: x[0])
    filtered_duplicates = [res[0][1]]
    for i in range(1, len(res)):
        if res[i][0] == res[i - 1][0]:
            pass
        else:
            filtered_duplicates.append(res[i][1])
        if len(filtered_duplicates) >= threshold:
            break

    return filtered_duplicates


# ES(labels,labels[0])


def compute_score(y_pred, true_y, labels, threshold=10):
    """
    compute the enrichment score over y_pred
    :param y_pred: predicted y (list)
    :param true_y: correct labels
    :param labels:
    :param threshold:
    :return:
    """
    hits = 0
    for i, y in enumerate(y_pred):
        closest = ES(labels, y, threshold)
        dist_to_closest = [np.linalg.norm(neighbor - true_y[i]) for neighbor in closest]
        if np.min(dist_to_closest) == 0:
            hits += 1
    return hits


# Prints show it works ok


def compute_it(pockets_scaled, labels, model):
    """
    :param pockets_scaled: X_embeddings
    :param labels: all labels
    :param ligands: unique ligands list
    :return:
    """
    accuracy = []
    kf = KFold(n_splits=5, shuffle=False)
    for train, test in kf.split(pockets_scaled):
        X_train, X_test, y_train, y_test = pockets_scaled[train], pockets_scaled[test], labels[train], labels[test]
        if isinstance(model, Sequential):
            model.fit(X_train, y_train, verbose=0, epochs=10, batch_size=16)
        else:
            model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        score = compute_score(y_pred, y_test, labels)
        accuracy.append(score)
    return accuracy


pockets = np.load('processed/pockets.npy')
pockets_scaled = preprocessing.scale(pockets)


def benchmark(pockets, labels):
    """
    :param pockets:
    :param labels:
    :return:
    """
    x_len, y_len = len(pockets[0]), len(labels[0])

    model = NN(x_len, y_len, layers_intermediate=[int(np.mean([x_len, y_len]))])
    accuracy_nn = compute_it(pockets, labels, model)

    model = DummyClassifier()
    accuracy_dummy = compute_it(pockets, labels, model)

    model = MLPRegressor(max_iter=2000)
    accuracy_mlp = compute_it(pockets, labels, model)
    return sum(accuracy_nn), sum(accuracy_dummy), sum(accuracy_mlp)


def evaluate_robustness(pockets_scaled, labels, max_seed):
    """
    Do the pipeline for several seeds of the random shuffling
    :param pockets_scaled:
    :param labels:
    :param max_seed:
    :return:
    """
    acc = []
    shuffled_acc = []
    pockets = np.copy(pockets_scaled)
    ligands = np.copy(labels)
    for seed in range(max_seed):
        # Do the computation
        pockets, ligands = unison_shuffled_copies(pockets, ligands, seed=10 + seed)
        acc.append(benchmark(pockets, ligands))

        # Do the shuffle comparison
        shuffled = np.copy(ligands)
        np.random.shuffle(shuffled)
        shuffled_acc.append(benchmark(pockets, shuffled))
    return acc, shuffled_acc


'''
Carlos way
'''

labels_carlos = pickle.load(open('data/delta_L.pickle', 'rb'))

# acc, shuffled_acc = evaluate_robustness(pockets_scaled, labels_carlos, max_seed=10)
# print(acc)
# print(shuffled_acc)
# means = np.mean(acc, axis=0)
# print(means)
# shuffled_means = np.mean(shuffled_acc, axis=0)
# print(shuffled_means)

carlos = [(71, 113, 179), (56, 99, 189), (58, 94, 192), (59, 127, 181), (59, 110, 183), (62, 93, 179), (61, 76, 180),
          (57, 104, 175), (64, 89, 180), (61, 127, 171)]
carlos_shuffled = [(56, 129, 60), (58, 103, 69), (56, 111, 57), (55, 119, 62), (39, 96, 58), (53, 88, 56),
                   (52, 130, 66), (49, 82, 60), (57, 110, 61), (50, 87, 59)]
# [ 60.8 103.2 180.9]
# [ 52.5 105.5  60.8]

'''
My way
'''
labels_continuous = np.load('processed/embed-512.npy')

# acc, shuffled_acc = evaluate_robustness(pockets_scaled, labels_continuous, max_seed=10)
#
# print(acc)
# print(shuffled_acc)
# means = np.mean(acc, axis=0)
# print(means)
# shuffled_means = np.mean(shuffled_acc, axis=0)
# print(shuffled_means)

continuous_512 = [(31, 28, 112), (46, 29, 113), (44, 32, 115), (35, 36, 112), (29, 29, 113), (41, 25, 125),
                  (40, 29, 109), (39, 32, 110), (40, 29, 121), (40, 28, 112)]
continuous_512_shuffled = [(43, 30, 35), (33, 28, 45), (36, 29, 47), (27, 29, 46), (25, 31, 49), (21, 34, 49),
                           (27, 30, 44), (34, 26, 42), (35, 22, 32), (39, 31, 24)]
# [ 38.5  29.7 114.2]
# [32.  29.  41.3]

'''
PCA Shrinked
'''

components = 10
print('n_components = ', components)
pca = PCA(n_components=components)
labels_shrinked = pca.fit_transform(labels_continuous)
# L = pca.explained_variance_ratio_
# print(L)
# print(sum(L))


# acc, shuffled_acc = evaluate_robustness(pockets_scaled, labels_shrinked, max_seed=10)
#
# print(acc)
# print(shuffled_acc)
# means = np.mean(acc, axis=0)
# print(means)
# shuffled_means = np.mean(shuffled_acc, axis=0)
# print(shuffled_means)


continuous_10 = [(73, 28, 73), (67, 29, 70), (81, 27, 78), (81, 25, 71), (84, 28, 71), (67, 36, 81), (73, 31, 77),
                 (71, 36, 79), (82, 28, 83), (74, 33, 76)]
continuous_10_shuffled = [(38, 26, 23), (33, 32, 16), (32, 38, 27), (31, 31, 21), (36, 26, 19), (32, 26, 22),
                          (29, 21, 26), (33, 29, 28), (45, 33, 23), (31, 29, 22)]
# [75.3 30.1 75.9]
# [34.  29.1 22.7]


components = 40
print('n_components = ', components)
pca = PCA(n_components=components)
labels_shrinked = pca.fit_transform(labels_continuous)
# L = pca.explained_variance_ratio_
# print(L)
# print(sum(L))


# acc, shuffled_acc = evaluate_robustness(pockets_scaled, labels_shrinked, max_seed=10)
#
# print(acc)
# print(shuffled_acc)
# means = np.mean(acc, axis=0)
# print(means)
# shuffled_means = np.mean(shuffled_acc, axis=0)
# print(shuffled_means)

continuous_40 = [(41, 19, 83), (34, 20, 81), (33, 34, 74), (38, 24, 91), (33, 33, 95), (36, 31, 91), (38, 31, 83),
                 (43, 27, 89), (43, 34, 96), (39, 35, 82)]
continuous_40_shuffled = [(28, 26, 29), (23, 34, 14), (27, 21, 32), (22, 31, 29), (21, 30, 29), (28, 25, 31),
                          (24, 28, 30), (27, 31, 28), (27, 34, 29), (23, 31, 27)]
# [37.8 28.8 86.5]
# [25.  29.1 27.8]


'''
plot
'''

# components = 2
# pca = PCA(n_components=components)
# labels_shrinked = pca.fit_transform(labels_carlos)
# x, y = (zip(*labels_shrinked))
# plt.scatter(x, y)
# plt.show()
#
# pca = PCA(n_components=components)
# labels_shrinked = pca.fit_transform(labels_continuous)
# x, y = (zip(*labels_shrinked))
# plt.scatter(x, y)
# plt.show()

'''
Stats
'''

# un1 = np.unique(labels_carlos, axis=0)
# un2 = np.unique(labels_continuous, axis=0)
# print(len(un1), len(un2))


DM_carlos = scipy.spatial.distance_matrix(labels_carlos, labels_carlos)
DM_continous = scipy.spatial.distance_matrix(labels_continuous, labels_continuous)
DM_shrinked = scipy.spatial.distance_matrix(labels_shrinked, labels_shrinked)
print(DM_shrinked[0], DM_carlos[0])
# sns.heatmap(DM_carlos)
# plt.savefig('heatmap_carlos.png')
# plt.show()
# sns.heatmap(DM_continous)
# plt.show()
# sns.heatmap(DM_shrinked)
# plt.savefig('heatmap_shrinked.png')
# plt.show()


# carlos = (np.ravel(DM_carlos))
# continuous = (np.ravel(DM_continous))
# shrinked = (np.ravel(DM_shrinked))
#
# # np.random.shuffle(carlos)

# correlation = scipy.stats.pearsonr(carlos, continuous)
# print(correlation)
# correlation = scipy.stats.pearsonr(carlos, shrinked)
# print(correlation)
#
# r, p_value, n = mantel(DM_carlos, DM_continous)
# print(r, p_value)
#
# r, p_value, n = mantel(DM_carlos, DM_shrinked)
# print(r, p_value)

# from tensorflow.python.client import device_lib
# print(device_lib.list_local_devices())
#
# from keras import backend as K
# K.tensorflow_backend._get_available_gpus()

'''
Analyse statistique
'''
'''
NN_carlos, Dummy_carlos, MLP_carlos = zip(*carlos)
NN_shuffled_carlos, Dummy_shuffled_carlos, MLP_shuffled_carlos = zip(*carlos_shuffled)

NN_512, Dummy_512, MLP_512 = zip(*continuous_512)
NN_shuffled_512, Dummy_shuffled_512, MLP_shuffled_512 = zip(*continuous_512_shuffled)

NN_40, Dummy_40, MLP_40 = zip(*continuous_40)
NN_shuffled_40, Dummy_shuffled_40, MLP_shuffled_40 = zip(*continuous_40_shuffled)

NN_10, Dummy_10, MLP_10 = zip(*continuous_10)
NN_shuffled_10, Dummy_shuffled_10, MLP_shuffled_10 = zip(*continuous_10_shuffled)


def compare_distributions(a, b, save=False):
    sns.distplot(a, hist=False, kde_kws={"label": "Unshuffled"})
    sns.distplot(b, hist=False, kde_kws={"label": "Shuffled"})
    if save:
        plt.savefig(save)
    plt.show()

    stat, p = wilcoxon(a, b)
    print(p)

'''
# compare_distributions(NN, NN_shuffled, save='NN.pdf')
# compare_distributions(Dummy_carlos, Dummy_shuffled_carlos, save='Dummy_Carlos.pdf')
# compare_distributions(Dummy_10, Dummy_shuffled_10, save='Dummy_10.pdf')

# sns.distplot(Dummy_carlos, hist=False, kde_kws={"label": "carlos"})
# sns.distplot(Dummy_512, hist=False, kde_kws={"label": "512"})
# sns.distplot(Dummy_40, hist=False, kde_kws={"label": "40"})
# sns.distplot(Dummy_10, hist=False, kde_kws={"label": "10"})
# plt.savefig('Dummies.pdf')
# plt.show()


# MLP_enrichments_10 = np.divide(MLP_10, MLP_shuffled_10)
# MLP_enrichments_40 = np.divide(MLP_40, MLP_shuffled_40)
# MLP_enrichments_512 = np.divide(MLP_512, MLP_shuffled_512)
# MLP_enrichments_carlos = np.divide(MLP_carlos, MLP_shuffled_carlos)
#
# sns.distplot(MLP_enrichments_carlos, hist=False, kde_kws={"label": "carlos"})
# sns.distplot(MLP_enrichments_512, hist=False, kde_kws={"label": "512"})
# sns.distplot(MLP_enrichments_40, hist=False, kde_kws={"label": "40"})
# sns.distplot(MLP_enrichments_10, hist=False, kde_kws={"label": "10"})
# plt.savefig('Enrichments.pdf')
# plt.show()
