import csv
import os
import time
from rdkit import Chem
import multiprocessing as mlt
import moments_computer as mc
import pathlib

"""
Tool to embed the structures of an sdf, the main doc is in the last function
"""


def build_colnames(moments_kept, centroids, features):
    """
    :return: name of the columns corresponding to the query embedding
    """
    # build the names of the columns
    colnames = ['structure', 'actif']
    for i in range(features):
        for j in range(centroids):
            for k in range(moments_kept):
                colnames += ['f' + str(i + 1) + 'c' + str(j + 1) + 'm' + str(k + 1)]
    return colnames


def prepare_csv(csv_dir, csv_path, moments_kept=3, centroids=4, features=5):
    """
    writes names of the column at the correct address
    """
    pathlib.Path(csv_dir).mkdir(parents=True, exist_ok=True)
    with open(csv_path, "a", newline='') as fp:
        wr = csv.writer(fp, dialect='excel')
        wr.writerow(build_colnames(moments_kept, centroids, features))


def write_csv(sdf_path, csv_path, active=False, features=True, pca=True, distance=4, mean=0):
    """
    Computes the embeddings
    """
    try:
        suppl = Chem.SDMolSupplier(sdf_path)
    except:
        row = ['could not open '] + [sdf_path]
        with open('failed.csv', "a", newline='') as fp:
            wr = csv.writer(fp, dialect='excel')
            wr.writerow(row)
            return
    mg = mc.MomentGenerator(features=features, pca=pca, distance=distance, mean=mean)
    i = 0
    for m in suppl:
        try:
            usrpcat_moments = mg.generate_moments(m)
            usrpcat_row = [str(i)] + [str(active)] + list(usrpcat_moments[0])
        except:
            row = [sdf_path] + [str(i)]
            with open('failed.csv', "a", newline='') as fp:
                wr = csv.writer(fp, dialect='excel')
                wr.writerow(row)
                continue
        with open(csv_path, "a", newline='') as fp:
            wr = csv.writer(fp, dialect='excel')
            wr.writerow(usrpcat_row)
        i += 1


def embed_structure(structure, features=True, pca=True, name=False, distance=4, mean=0):
    """
    full pipeline, takes a structure and outputs the csv corresponding to the embedding with the given parameters
    :param structure: name of the structure
    :param features: add the 4 atom types ?
    :param pca: use the pca method ?
    :param name: name of the embedding to be saved
    :param distance : distance of the ref points to the centroid
    :param mean : index in [np.mean, geometrical_mean, harmonical_mean] to be used as aggregation technique
    """
    # META
    # SDF Path
    actives_path = os.path.join('../data/structures', structure, 'actives_final.sdf')
    decoys_path = os.path.join('../data/structures', structure, 'decoys_final.sdf')

    # CSV Paths
    if not name:
        name = '_feature={}_pca={}.csv'.format(features, pca)
        pass
    csv_actives_dir = os.path.join('../data/embeddings', structure)
    csv_decoys_dir = os.path.join('../data/embeddings', structure)
    csv_actives_path = os.path.join(csv_actives_dir, 'csv_actives' + name)
    csv_decoys_path = os.path.join(csv_decoys_dir, 'csv_decoys' + name)

    num_moments_kept = 3
    num_centroids = 4
    num_features = 1 + 4 * features

    # Write Actives csv
    prepare_csv(csv_actives_dir, csv_actives_path,
                moments_kept=num_moments_kept, centroids=num_centroids, features=num_features)
    write_csv(actives_path, csv_actives_path, active=True, features=features, pca=pca, distance=distance, mean=mean)

    # Write Decoys csv
    prepare_csv(csv_decoys_dir, csv_decoys_path,
                moments_kept=num_moments_kept, centroids=num_centroids, features=num_features)
    write_csv(decoys_path, csv_decoys_path, active=False, features=features, pca=pca, distance=distance, mean=mean)


def all_embeddings(parallel_tuple):
    """
    compute all embeddings and write them in the different csv
    :param parallel_tuple: (structure, name, distance, mean, features, pca)
    """
    structure, name, distance, mean, features, pca = parallel_tuple
    print('computing embeddings for {}'.format(structure))
    embed_structure(structure, name=name, distance=distance, mean=mean, features=features, pca=features)
    # embed_structure(structure, features=True, pca=True)
    # embed_structure(structure, features=True, pca=False)
    # embed_structure(structure, features=False, pca=True)
    # embed_structure(structure, features=False, pca=False)


# TUNE YOUR META PARAMETERS HERE

# 520s pour 3 structures
def parallel_embeddings(parallel=True, name='_3.csv', distance=4, mean=2, features=True, pca=True):
    """
    This is the main pipeline : Take all structures available in the DUDE database, in the data/structure/ directory
    and make the required embedding on it. Saves it in data/embeddings/name or there is a default naming
    :param features: add the 4 atom types ?
    :param pca: use the pca method ?
    :param name: name of the embedding to be saved
    :param distance : distance of the ref points to the centroid
    :param mean : index in [np.mean, geometrical_mean, harmonical_mean] to be used as aggregation technique
    """
    list_of_structure = os.listdir('../data/structures')
    # list_of_structure = ['aa2ar']

    if parallel:
        list_of_tuples = [(structure, name, distance, mean, features, pca) for structure in list_of_structure]
        pool = mlt.Pool()
        pool.map(all_embeddings, list_of_tuples)
    else:
        for structure in list_of_structure:
            embed_structure(structure, name=name, distance=distance, mean=mean, features=features, pca=features)


if __name__ == "__main__":
    start_time = time.time()
    parallel_embeddings('aa2ar')
    print(time.time() - start_time)
