import shape.bp_extract as bp
import shape.UFSR_feature as ufsr
import multiprocessing as mlt
import Bio
import os
import pickle
import time
import csv


def extract_bp_pdb_from_directory(path, output_path, start_radius=3, max_radius=10, number_min=4):
    """
    Takes the path of a directory and write extracted bp pdbs in the output path directory
    :param path: directory of protein pdbs
    :param output_path:
    :return:
    """
    io = Bio.PDB.PDBIO()
    for pdb in os.listdir('../data/protein/' + path):
        bp.path_to_pdb('../data/protein/' + path + pdb, '../data/output_pdb/' + output_path,
                       start_radius=start_radius, max_radius=max_radius, number_min=number_min, io=io)


def extract_bp_pdb_from_list(list_of_ids, output_name, temp_path='../data/temp/cif/', start_radius=3,
                             max_radius=10, number_min=4):
    """
    Do the bp extraction pipeline for a list of pdb ids. Fetch them in a temporary dir, then apply bp parsing
    :param list_of_ids: list with string corresponding to pdb ids
    :param output_name: should complement a directory path such as 'test/'
    :param temp_path: Temporary path to download the structure
    :param start_radius: 
    :param max_radius:
    :param number_min:
    :return:
    """
    pdbl = Bio.PDB.PDBList(verbose=False)
    count, failed, tot = 0, 0, len(list_of_ids)
    failed_ids = []

    # create output dir
    output_path = os.path.join('../data/output_pdb', output_name)
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    for pdb in list_of_ids:
        if count % 100 == 0:
            print("{}/{} calculés, {} failed".format(count, tot, failed))
        count += 1

        # download the required pdb
        path_name = pdbl.retrieve_pdb_file(pdb, pdir=temp_path, file_format="mmCif")

        # Extract and write the binding pockets
        try:
            bp.path_to_pdb(path_name, output_path, start_radius=start_radius,

                           max_radius=max_radius,
                           number_min=number_min)
        except:
            failed += 1
            failed_ids.append(pdb)
            continue

        # Clear the full download
        os.remove(path_name)
    return failed_ids


# 14s for 20 structures, downloads are the longest (11s)
# Therefore the parallel is not really the longest part
def extract_ufsr_from_directory(path, output_name):
    """
    :param path: path to a pdb directory
    :param output_name: Where to write the embeddings
    :return:
    """
    for pdb in os.listdir('../data/protein/' + path):
        # create the ufsr feature of the required id ligands
        path_name = "../data/protein/sample/" + pdb
        ufsr_dict = bp.path_to_ufsr_dict(path_name)

        with open(os.path.join('../data/output_csv', output_name + '.csv'), "a", newline='') as fp:
            wr = csv.writer(fp, dialect='excel')
            for ligand, ufsr in ufsr_dict.items():
                row = [pdb] + [ligand.get_resname()] + list(ufsr)
                wr.writerow(row)


def extract_ufsr_from_list(list_of_ids, output_name, temp_path='../data/temp/cif/', number_of_moments=3):
    """
    Do the whole pipeline for a list of pdb ids. Fetch them in a temporary dir, then apply bp parsing
    :param list_of_ids: list with string corresponding to pdb ids
    :param output_name: output_name
    :param temp_path:
    :param number_of_moments:
    :return:
    """
    pdbl = Bio.PDB.PDBList(verbose=False)
    count, failed, tot = 0, 0, len(list_of_ids)
    failed_ids = []

    # put columns names
    centroids = 4
    colnames = ufsr.build_colnames(number_of_moments, centroids)
    with open(os.path.join('../data/output_csv', output_name + '.csv'), "a", newline='') as fp:
        wr = csv.writer(fp, dialect='excel')
        wr.writerow(colnames)

    for pdb in list_of_ids:
        if count % 100 == 0:
            print("{}/{} calculés, {} failed".format(count, tot, failed))
        count += 1

        # download the required pdb
        path_name = pdbl.retrieve_pdb_file(pdb, pdir=temp_path, file_format="mmCif")

        # create the ufsr feature of the required id ligands
        # check if the file has been saved
        try:
            ufsr_dict = bp.path_to_ufsr_dict(path_name, number_of_moments=number_of_moments)
        except:
            failed += 1
            failed_ids.append(pdb)
            continue

        os.remove(path_name)

        # write them in a csv
        with open(os.path.join('../data/output_csv', output_name + '.csv'), "a", newline='') as fp:
            wr = csv.writer(fp, dialect='excel')
            for ligand, feature in ufsr_dict.items():
                row = [pdb] + [ligand.get_resname()] + list(feature)
                wr.writerow(row)
    return failed_ids


# 14s for 20 structures, downloads are the longest (11s)
# Therefore the parallel is not really the longest part


if __name__ == "__main__":
    pass

    with open("../data/protein/pdb_ids.p", "rb") as file:
        list_of_ids = pickle.load(file)

    start_time = time.time()
    failed = extract_bp_pdb_from_list(list_of_ids[:3], 'test/', temp_path='../data/temp/cif/', start_radius=12,
                                      max_radius=15, number_min=6)
    print(failed)
    print("--- %s seconds ---" % (time.time() - start_time))

    # start_time = time.time()
    # failed = extract_ufsr_from_list(['1a0b','1l6l'], "usfr_whole")
    # print(failed)
    # print("--- %s seconds ---" % (time.time() - start_time))

    # start_time = time.time()
    # failed = extract_ufsr_from_list(list_of_ids, "test")
    # print(failed)
    # print("--- %s seconds ---" % (time.time() - start_time))
    # 58000s = 16h
    # ['1l6l', '1qy5', '1qzv', '1rid', '2a01', '2b6e', '2cjt', '2h88', '2j1n', '2oo4', '3dzx', '3flv',
    #  '3gqq', '3kuf', '3tlq', '3tm9', '3ttp', '4cj9', '4fek', '4hw6', '4ip8', '4jdl', '4ksn', '4ndo',
    #  '4otn', '4x01', '4yd9', '5by5', '5d6o', '5i8f', '5lu5', '5suq', '5tti', '5uur', '5vko', '5xzt'] failed (36)

    # start_time = time.time()
    # extract_ufsr_from_directory('sample/', "toto2")
    # print("--- %s seconds ---" % (time.time() - start_time))

    # start_time = time.time()
    # extract_bp_pdb_from_directory('sample/', 'sample/')
    # print("--- %s seconds ---" % (time.time() - start_time))

    # with open("../data/protein/pdb_ids.p", "rb") as output_file:
    #     list_of_ids = pickle.load(output_file)
    #     list_of_ids = [i.lower() for i in list_of_ids]

    # # from '../data/source_data/' to graph in '../data/output_graph/'
    # def graph_from_directory(path, output_path):
    #     for pdb in os.listdir('../data/source_data/' + path):
    #         name_structure = pdb[-8:-4]
    #         gr.graphs_from_path('../data/source_data/' + path + pdb, '../data/output_graph/' + output_path,
