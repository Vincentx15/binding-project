import Bio.PDB
import bp_pdb_from_pdb as bp
import get_interaction as gi
import networkx as nx
import os
import time

'''
take pdb as input and makes a graph out of its binding pocket, within a radius and a cutoff
same treatment for whole pdbs (already extracted) as it would have for small ones (calls find_neighbors function)
writes it in a pickle
'''

AA = ["ALA", "CYS", "ASP", "GLU", "PHE", "GLY", "HIS", "ILE", "LYS", "LEU",
      "MET", "ASN", "PRO", "GLN", "ARG", "SER", "THR", "VAL", "TRP", "TYR"]


def graph_from_binding_pocket(structure, bp_dict):
    """
    makes a graph out of a binding pocket dictionnary and a structure
    :param structure:
    :param bp_dict: a bp_dict is defined in extract_neighborhood, it is a dict of chain : residues_id in the bp
    :return:
    """
    g = nx.Graph()
    for chain in bp_dict:
        previous_id = 0
        for residue_id in bp_dict[chain]:
            try:
                residue = structure[0][chain][residue_id]
            except:
                continue
            g.add_node((chain, residue_id), name=residue.get_resname())
            if previous_id and residue_id == previous_id + 1:
                g.add_edge((chain, previous_id), (chain, residue_id), interaction='BB')
            previous_id = residue_id
    return g



def add_hbond(bp_dict, hbond, g):
    """
    add hbond info
    :param bp_dict:
    :param hbond: hbond dataframe as defined in get_interaction
    :param g: graph we want to add hbond info to
    :return: nothing, modifies g
    """
    number_of_bonds = len(hbond['name_1'])
    for line in range(number_of_bonds):
        # filter ligand
        if hbond['name_1'][line] not in AA or hbond['name_2'][line] not in AA:
            continue
        chain1, num1 = hbond['chain_1'][line], hbond['num_1'][line]
        chain2, num2 = hbond['chain_2'][line], hbond['num_2'][line]
        try:
            num1 = int(num1)
            num2 = int(num2)
        except:
            continue
        if (chain1 not in bp_dict or chain2 not in bp_dict or num1 not in bp_dict[chain1] or num2 not in bp_dict[
            chain2]):
            continue
        g.add_edge((chain1, num1), (chain2, num2), interaction='HBOND')



def add_distance_information(structure, graph):
    """
    Binds the closest points of connected components if they are below a threshold
    :param structure: biopython structure
    :param graph: Networkx graph
    :return:
    """
    visited = list()
    for cc in nx.connected_components(graph):
        for other_cc in visited:
            distance_min_between_cc = 10
            neighbor = 0
            for node_1 in cc:
                for node_2 in other_cc:
                    chain_1, residue_1, chain_2, residue_2 = node_1[0], node_1[1], node_2[0], node_2[1]
                    try:
                        alpha_1 = structure[0][chain_1][residue_1]['CA']
                        alpha_2 = structure[0][chain_2][residue_2]['CA']
                    except:
                        continue
                    distance = alpha_1 - alpha_2
                    if distance < distance_min_between_cc:
                        distance_min_between_cc = distance
                        neighbor = (chain_1, residue_1, chain_2, residue_2, distance)
            if neighbor:
                graph.add_edge((neighbor[0], neighbor[1]), (neighbor[2], neighbor[3]), interaction='Distance',
                               distance=neighbor[4])
        visited.append(cc)
    return


def find_ends(list_of_residues):
    """
    Get all the ends of consecutive strands, needs fixing for chain support
    :param list_of_residues:
    :return:
    """
    first = 0
    last = 1
    ends = []
    while first < len(list_of_residues):
        # if last is in a consecutive sequence
        while last < len(list_of_residues) - 1 and list_of_residues[last + 1] - list_of_residues[last] < 2:
            last += 1
        ends.append(('A', list_of_residues[first], list_of_residues[last]))
        first = last + 1
        last = first + 1
    print(ends)


# L = [1,2,3,5,6,7,8,9,10,14,15]
# find_ends(L)


def add_distance_information_border(structure, bp_list, graph):
    """
    Takes the ends of connex components and binds them
    :param structure:
    :param bp_list:
    :param graph:
    :return: dict of interaction in the graph edges
    """
    ends = []
    for chain, list_of_residues in bp_list.items():
        first = 0
        last = 1
        while first < len(list_of_residues):
            # if last is in a consecutive sequence
            while last < len(list_of_residues) - 1 and list_of_residues[last + 1] - list_of_residues[last] < 2:
                last += 1
            ends.append((chain, list_of_residues[first], list_of_residues[last]))
            first = last + 1
            last = first + 1

    # compute bonds between these bounds
    visited = list()
    for end in ends:
        for other_end in visited:
            # get the 4 distances between two ends
            distance_min_between_cc = 15
            chain_1, residue_1, residue_2, chain_2, residue_3, residue_4 = end[0], end[1], end[2], other_end[0], \
                                                                           other_end[1], other_end[2]
            try:
                alpha_1 = structure[0][chain_1][residue_1]['CA']
                alpha_2 = structure[0][chain_1][residue_2]['CA']
                alpha_3 = structure[0][chain_2][residue_3]['CA']
                alpha_4 = structure[0][chain_2][residue_4]['CA']
            except:
                continue
            # get the possible interactions
            distance_13 = alpha_1 - alpha_3
            distance_14 = alpha_1 - alpha_4
            distance_23 = alpha_2 - alpha_3
            distance_24 = alpha_2 - alpha_4
            neighbor_13 = (chain_1, residue_1, chain_2, residue_3, distance_13)
            neighbor_14 = (chain_1, residue_1, chain_2, residue_4, distance_14)
            neighbor_23 = (chain_1, residue_2, chain_2, residue_3, distance_23)
            neighbor_24 = (chain_1, residue_2, chain_2, residue_4, distance_24)
            neighbors = [neighbor_13, neighbor_14, neighbor_23, neighbor_24]

            # filter them into real neighbors
            for neighbor in neighbors:
                if neighbor[4] < distance_min_between_cc:
                    graph.add_edge((neighbor[0], neighbor[1]), (neighbor[2], neighbor[3]), interaction='Distance',
                                   distance=neighbor[4])
        visited.append(end)


def graph_from_binding_pocket_and_hbond(structure, bp_dict, hbond):
    """
    makes a graph out of binding pocket with hbond extra info
    :param structure:
    :param bp_dict:
    :param hbond:
    :return:
    """
    g = graph_from_binding_pocket(structure, bp_dict)
    # add_distance_information(structure, g)
    add_distance_information_border(structure, bp_dict, g)
    add_hbond(bp_dict, hbond, g)
    return g


def pipeline_structure(structure, name, output_path, start_radius=3, max_radius=10, cutoff=5, number_min=12,
                       io=Bio.PDB.PDBIO()):
    """
    Wrapped in pipeline path
    """
    ligands = bp.find_ligands_residue(structure)
    for i, ligand in enumerate(ligands):
        # write a temporary pdb around the bp of this ligand
        binding_pocket = bp.find_binding_pocket(structure, ligand, start_radius, max_radius, cutoff, number_min)
        if not binding_pocket:
            continue
        output_name_pdb = '../data/temp/pdb/' + name + '_' + ligand.get_resname() + '_' + str(i) + '.pdb'
        bp.write_pdb_output(structure, ligand, binding_pocket, output_name_pdb, io)

        # find the hbonds around it
        output_name_hbond = '../data/temp/hbond/' + name + '_' + ligand.get_resname() + '_' + str(i) + '.hbond'
        gi.chimera_hbond(output_name_pdb, output_name_hbond)
        hbonds = gi.hbonds_parse(output_name_hbond)
        os.remove(output_name_pdb)
        os.remove(output_name_hbond)

        # write a pickled graph of it
        graph = graph_from_binding_pocket_and_hbond(structure, binding_pocket, hbonds)
        nx.write_gpickle(graph, output_path + name + '_' + ligand.get_resname() + '_' + str(i) + '.p', protocol=1)


def pipeline_path(input_path, output_path, start_radius=1, max_radius=10, cutoff=5, number_min=12, io=Bio.PDB.PDBIO()):
    """
    The pipeline : get a pdb from a path, extract the ligands list. For each, get the temp neighborhood, turn it into a graph
    :param input_path:
    :param output_path:
    :param start_radius:
    :param max_radius:
    :param cutoff:
    :param number_min:
    :param io:
    :return:
    """
    parser = Bio.PDB.FastMMCIFParser(QUIET=True)
    name = input_path[-8:-4]
    structure = parser.get_structure(name, input_path)
    pipeline_structure(structure, name, output_path, start_radius, max_radius, cutoff, number_min, io)


if __name__ == "__main__":
    pass
    # start_time = time.time()
    # pipeline_path('../data/source_data/test_subset/1nno.cif', '../data/output_graph/test/')
    # print("--- %s seconds ---" % (time.time() - start_time))

    # start_time = time.time()

    # print("--- %s seconds ---" % (time.time() - start_time))

