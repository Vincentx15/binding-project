import bp_pdb_from_pdb as bp
import graph_from_pdb as gr
import Bio
import os
import multiprocessing as mlt
import pickle
import time


# takes the path of a directory and write extracted bp pdbs in the output path directory
def extract_bp_from_directory(path, output_path):
    io = Bio.PDB.PDBIO()
    for pdb in os.listdir('../data/source_data/' + path):
        bp.pipeline_path('../data/source_data/' + path + pdb, '../data/output_structure/' + output_path, io=io,
                         cutoff=5)


# from '../data/source_data/' to graph in '../data/output_graph/'
def graph_from_directory(path, output_path):
    for pdb in os.listdir('../data/source_data/' + path):
        name_structure = pdb[-8:-4]
        gr.graphs_from_path('../data/source_data/' + path + pdb, '../data/output_graph/' + output_path,
                            name=name_structure, radius=5, cutoff=5)


# from '../data/source_data/'+ path to graph in '../data/output_graph/' + output_path
def graph_with_hbonds_from_directory(path, output_path):
    i, length = 0, len(os.listdir('../data/source_data/' + path))
    for pdb in os.listdir('../data/source_data/' + path):
        print('{} on {} done'.format(i, length))
        i += 1
        print(pdb)
        gr.pipeline_path('../data/source_data/' + path + pdb, '../data/output_graph/' + output_path)


# from a list of pdb ids to graph in '../data/output_graph/' + output_path
def graph_with_hbonds_from_list(list_of_ids, output_path):
    pdbl = Bio.PDB.PDBList()
    for pdb in list_of_ids:
        pdbl.retrieve_pdb_file(pdb, pdir='../data/temp/cif/', file_format="mmCif")
        path_name = '../data/temp/cif/' + pdb + '.cif'
        gr.pipeline_path(path_name, '../data/output_graph/' + output_path, start_radius=2, number_min=12)
        os.remove(path_name)


# can use only one function to be defined outside with one argument (that can have several fields)
def parallel_graph_from_list(list_of_ids, output_path):
    list_of_tuples = [(a, output_path) for a in list_of_ids]
    pool = mlt.Pool()
    failed_nos = pool.map(f, list_of_tuples)
    failed = sum(failed_nos)
    print('there were {} failed hbonds '.format(failed))


# auxiliary function of the parallel computation
def f(input_tuple):
    pdb_id, output_path = input_tuple
    pdbl = Bio.PDB.PDBList()
    pdbl.retrieve_pdb_file(pdb_id, pdir='../data/temp/cif/', file_format="mmCif")
    path_name = '../data/temp/cif/' + pdb_id + '.cif'
    # catch the failures of hbonds and count them
    try:
        gr.pipeline_path(path_name, '../data/output_graph/' + output_path)
    except:
        return 1
    os.remove(path_name)
    return 0


if __name__ == "__main__":
    failed = 0
    start_time = time.time()

    with open("../data/protein/pdb_ids.p", "rb") as output_file:
        list_of_ids = pickle.load(output_file)
    list_of_ids = [i.lower() for i in list_of_ids]
    list_of_ids = list_of_ids[:2]
    print(len(list_of_ids))
    print(len(bp.set_of_ligands))
    parallel_graph_from_list(list_of_ids, 'protein_babies_whole/')

    # graph_with_hbonds_from_directory('test_subset/', 'border/')

    print("--- %s seconds ---" % (time.time() - start_time))


# 13s pour subset de 60 avec radius de 5
# 28s pour subset, 4,4 connected component average on 65 structures of 16,1 nodes
# 188s pour inclure les hbonds

# bp 6.634366750717163
# pdb 1.3155276775360107
# hbond 212.42895412445068
# files 0.011568069458007812
# graph 0.9575879573822021
# pickle 0.07177352905273438
