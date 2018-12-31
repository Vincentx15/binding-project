import pandas as pd
import scipy as np
import scipy.linalg as npl
import os
import time
import csv
import multiprocessing as mlt

"""
Compute the enrichment factors of a structure, given an embedding
Does it in parallel
"""


def usfr_similarity(x, y):
    """
    similarity defined by the original Ballester paper
    """
    x = np.asarray(x)
    y = np.asarray(y)
    try:
        similarity_measure = 1 / (1 + np.sum(abs(x - y)))
    except:
        return None
    return similarity_measure


def cosine(x, y):
    """
    Cosine similarity
    """
    x = np.asarray(x)
    y = np.asarray(y)

    prod, x_norm, y_norm = (x * y).sum(), npl.norm(x), npl.norm(y)
    cosine = prod / x_norm / y_norm
    return cosine


def super_similarity(x, y):
    return 2 * cosine(x, y) + usfr_similarity(x, y)


# Change here the similarity

def distance_matrix(x, y, distance_function=0):
    """
    copied from scipy documentation
    """
    if distance_function not in [0, 1, 2]:
        distance_function = 0
    distance_options = [usfr_similarity, cosine, super_similarity]
    distance_function = distance_options[distance_function]

    x = np.asarray(x)
    m, k = x.shape
    y = np.asarray(y)
    n, kk = y.shape
    result = np.empty((m, n), dtype=float)  # FIXME: figure out the best dtype
    for i in range(m):
        result[i, :] = [distance_function(x[i], y_j) for y_j in y]
    return result


# X = [[1, 2, 3]]
# Y = [[1, 2, 2]]
# print(distance_matrix(X, Y, 32))


# csv reading = 0.15s
# DM computation = 30.5s
# loop = 0.70s
def average_EF(structure, distance_function, feature=True, pca=False, percent=[0.25], name=False):
    """
    Compute the average enrichment factor of a given structure, over the set of actives
    Does it faster because of a vectorisation process (old version below)
    :param structure: the structure to access, in data/
    :param name: the name of the embedding csv to use
    :return: the average EF for this structure as a dict percent : EF
    """
    # not very useful, just put the name of the embedding csv. Otherwise it will get the baseline emebeddings
    if not name:
        name = '_feature={}_pca={}.csv'.format(feature, pca)

    # META Get the Path of the embeddings
    csv_actives_dir = os.path.join('../data/embeddings', structure)
    csv_decoys_dir = os.path.join('../data/embeddings', structure)
    csv_actives_path = os.path.join(csv_actives_dir, 'csv_actives' + name)
    csv_decoys_path = os.path.join(csv_decoys_dir, 'csv_decoys' + name)

    features = (4 * feature + 1) * 12
    actives_values = pd.read_csv(csv_actives_path, usecols=range(2, features + 1), dtype=float)
    actives_values = np.array(actives_values)
    decoys_values = pd.read_csv(csv_decoys_path, usecols=range(2, features + 1), dtype=float)
    decoys_values = np.array(decoys_values)

    # Compute the DM (bottleneck)
    actives_dist = distance_matrix(actives_values, actives_values, distance_function)
    decoys_dist = distance_matrix(actives_values, decoys_values,distance_function)

    # Sort and compute the EF
    enrichments = {}
    for perc in percent:
        enrichments[perc] = []
    for i in range(actives_dist.shape[1]):
        list_actives = np.sort(actives_dist[i])[:-1]
        list_decoys = np.sort(decoys_dist[i])
        total = np.sort(np.append(list_decoys, list_actives))
        a, d, tot = len(list_actives), len(list_decoys), len(total)

        # avoid to recompute this for all percent
        for perc in percent:
            threshold_index = tot - int(perc / 100 * tot) - 1
            threshold = total[threshold_index]
            selected_actives = [x for x in list_actives if x >= threshold]
            selected_decoys = [x for x in list_decoys if x >= threshold]
            sa, sd = len(selected_actives), len(selected_decoys)
            stot = sa + sd
            numerator = sa / stot
            denominator = a / tot
            enrichments[perc].append(numerator / denominator)

    for perc in percent:
        enrichments[perc] = np.mean(enrichments[perc])
    return enrichments


# average_EF_benchmark('aa2ar', 'test')
# similar runtimes for each different embeddings
def average_EF_benchmark(structure, output_csv_name, percent, name_embedding, distance_function):
    """
    :param structure: The structure to benchmark
    writes in the output csv at 'path' the average EF as a new row in the output csv
    """
    enrichments = average_EF(structure, distance_function = distance_function, name=name_embedding, percent=percent)
    try:
        pass
        # row.append(np.mean(average_EF(structure, feature=False, pca=False)))
        # row.append(np.mean(average_EF(structure, feature=True, pca=False)))
        # row.append(np.mean(average_EF(structure, feature=True, pca=True)))
        pass
    except:
        with open('failed_benchmarks.csv', "a", newline='') as fp:
            wr = csv.writer(fp, dialect='excel')
            wr.writerow([structure])
            print('failed')
            return
    for percent, enrichment in enrichments.items():
        row = [structure]
        row.append(enrichment)
        output_path = '../data/output_csv/' + output_csv_name + '_EF={}'.format(percent) + '.csv'
        with open(output_path, "a", newline='') as fp:
            wr = csv.writer(fp, dialect='excel')
            wr.writerow(row)


def f(parallel_tuple):
    """
    encapsulate the arguments for parallel computation
    """
    average_EF_benchmark(*parallel_tuple)


def parallel_benchmark(output_csv_name, percent=[0.25, 0.5, 1], name_embedding='_HM.csv', distance_function = 0):
    """
    computes all the resulting rows for all structures in the output csv in a parallel (structure-wise) way
    :param output_csv_name: The name of the csv to write on
    :param percent: the percent of the data to consider for the computation of the EF as a list
    :param name_embedding : The name of the embedding to use
    :param distance_function : The index of [usfr_similarity, cosine, super_similarity] to use (default 0)
    """
    # prepare the csv
    for perc in percent:
        output_path = '../data/output_csv/' + output_csv_name + '_EF={}'.format(perc) + '.csv'
        # colnames = ['structure', 'no_feature_no_pca', 'feature_no_pca', 'feature_pca']
        colnames = ['structure', output_csv_name]
        with open(output_path, "a", newline='') as fp:
            wr = csv.writer(fp, dialect='excel')
            wr.writerow(colnames)

    # do the parallel computation (not very safe but only 102 writings each taking several minutes)
    list_of_structure = os.listdir('../data/embeddings')
    # list_of_structure = ['csf1r', 'vgfr2']
    list_of_tuples = [(structure, output_csv_name, percent, name_embedding, distance_function)
                      for structure in list_of_structure]
    pool = mlt.Pool()
    pool.map(f, list_of_tuples)


if __name__ == '__main__':
    pass
    # start_time = time.time()
    # parallel_benchmark('test_EF')
    # time_benchmark = (time.time() - start_time)
    # # print(" time_benchmark :", time_benchmark)
    # EF = average_EF_benchmark('aa2ar', 'test_percent', percent = [0.25,0.5,1])
# # mean EF 2.77426686032
# l,
# print('feature=True, pca = False')
# print(times)
# print('mean runtime', np.mean(times))
# print(l)
# print('mean EF', np.mean(l))
# distance_to_all('../data/aa2ar/csv_actives_feature=False_pca=False.csv',
#                 molecule=np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]))

'''
def distance_to_all(df, molecule, active, distance=usfr_similarity):
    """
    distance from a molecule to every other in a given dataframe
    uses a dictionnary format with distances as keys
    :param df: the dataframe to compare to
    :param molecule: the molecule being compared, embedded
    :param active: Boolean, tells actives from decoys
    :param distance: The distance function to use
    """
    distances = []
    for index, row in df.iterrows():
        dist = distance(row, molecule)
        distances.append(dist)
    return distances


def distance_to_all_from_csv(csv, molecule, distance=usfr_similarity):
    """
    old
    """
    df = pd.read_csv(csv)
    return distance_to_all(df, molecule, distance)


def enrichment_factor(molecule, df_actives, df_decoys, rate=0.0025, is_distance=False):
    """
    Compute the enrichment factor for a given molecule : the rate of true positives with this preprocessing step
    over the rate of true positives in the data
    :param molecule: the active molecule being tested
    :param df_actives: the df of all actives embed
    :param df_decoys: the df of all decoys embed
    :param rate: the rate of closest points
    :param is_distance: not used yet, adapts the program for distance instead of similarity
    """
    # get the distance from this molecule to all other actives and decoys for this structure
    actives = distance_to_all(df_actives, molecule, True)
    decoys = distance_to_all(df_decoys, molecule, False)
    list_actives = sorted(actives)[:-1]
    list_decoys = sorted(decoys)
    total = sorted(list_decoys + list_actives)
    a, d, tot = len(list_actives), len(decoys), len(total)
    # for the distances/similarity different treatments
    if is_distance:
        threshold_index = int(rate * tot)
        threshold = total[threshold_index]
        selected_actives = [x for x in list_actives if x <= threshold]
        selected_decoys = [x for x in list_decoys if x <= threshold]
    else:
        threshold_index = tot - int(rate * tot) - 1
        threshold = total[threshold_index]
        selected_actives = [x for x in list_actives if x >= threshold]
        selected_decoys = [x for x in list_decoys if x >= threshold]
    sa, sd = len(selected_actives), len(selected_decoys)
    stot = sa + sd
    numerator = sa / stot
    denominator = a / tot
    print(a,tot)
    # print(numerator)
    # print(denominator)
    return numerator / denominator



# csv reading = 0.03s, loop = 0.59s
def average_EF(structure, feature=False, pca=False):
    """
    Compute the average enrichment factor of a given structure, over the set of actives
    :param structure: the structure to access, in data/
    :return: the average EF for this method
    """
    # extract the values for actives and decoys
    csv_actives_path = os.path.join('../data/structures', structure,
                                    'csv_actives_feature={}_pca={}.csv'.format(feature, pca))
    csv_decoys_path = os.path.join('../data/structures', structure,
                                   'csv_decoys_feature={}_pca={}.csv'.format(feature, pca))

    features = (4 * feature + 1) * 12
    actives_values = pd.read_csv(csv_actives_path, usecols=range(2,features+1), dtype=float)
    decoys_values = pd.read_csv(csv_decoys_path, usecols=range(2,features+1), dtype=float, nrows=40)


    # start_time = time.time()
    # df_actives = pd.read_csv(csv_actives_path)
    # df_decoys = pd.read_csv(csv_decoys_path)
    # actives_values = df_actives[df_actives.columns[2:]]
    # decoys_values = df_decoys[df_decoys.columns[2:]]
    # time_benchmark = (time.time() - start_time)
    # print('time it took for read_csv 2 : {}'.format(time_benchmark))

    enrichments = []
    for index, row in actives_values.iterrows():
        molecule = actives_values.iloc[index]
        enrichments.append(enrichment_factor(molecule, actives_values, decoys_values))
    return enrichments

'''
