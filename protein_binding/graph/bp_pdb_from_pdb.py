import Bio.PDB
import numpy as np
import pickle

'''
Define the extracting pipeline of ligands neighborhood.
Read the pdb and parse it with Biopython fast parser
Scan for ligands and for each of these, scan their neighborhood
Select all corresponding residues and output them in a pdb file
'''

AA = ["ALA", "CYS", "ASP", "GLU", "PHE", "GLY", "HIS", "ILE", "LYS", "LEU",
      "MET", "ASN", "PRO", "GLN", "ARG", "SER", "THR", "VAL", "TRP", "TYR"]
with open("../data/ligands/set_of_ligands.pickle", "rb") as input_file:
    set_of_ligands = pickle.load(input_file)


# is the residue an acceptable ligand (no ion, not water?) plus is it in the ligand set
def is_an_acceptable_ligand(residue):
    return residue.get_resname() not in (["HOH"] + AA) and residue.get_resname() in set_of_ligands


# return the identifiers of all ligands in a structure, faster to compute this way because of the fast parsing
def find_ligands_residue(structure):
    ret = []
    for model in structure:
        for chain in model:
            for residue in chain:
                if is_an_acceptable_ligand(residue):
                    ret += [residue]
    return ret


'''
# just the neighbors
def find_neighbors_of_ligand(structure, ligand, radius):
    atom_result = set()
    residue_result = set()
    searcher = Bio.PDB.NeighborSearch(list(structure.get_atoms()))
    for atom in ligand.get_atoms():
        center = np.array(atom.get_coord())
        atom_neighbor = searcher.search(center, radius)
        for neighbor in atom_neighbor:
            atom_result.add(neighbor)
    for atom in atom_result:
        residue_result.add(atom.get_parent())
    return residue_result
    
'''


# find the bp around a ligand (residues that have atoms less than radius away from an atom of the ligand)
# faster method, just compute the sphere for every atom in the ligand with KDTree for the whole structure
# Expand the pocket if the radius was too small
def find_binding_pocket(structure, ligand, start_radius, max_radius, cutoff, number_min):
    searcher = Bio.PDB.NeighborSearch(list(structure.get_atoms()))

    atom_result = set()
    residue_result = set()
    dict_of_chains = {}

    number_of_nodes = 0
    radius = start_radius
    while number_of_nodes < number_min:
        number_of_nodes = 0

        # search atoms within the radius
        for atom in ligand.get_atoms():
            center = np.array(atom.get_coord())
            atom_neighbor = searcher.search(center, radius)
            for neighbor in atom_neighbor:
                atom_result.add(neighbor)

        # create the set of resulting residues
        for atom in atom_result:
            parent_residue = atom.get_parent()
            # check if it is a protein residue
            if parent_residue.get_id()[0] == ' ':
                residue_result.add(parent_residue)

        # build bp dict chain : list of residue neighbors
        for residue in residue_result:
            chain = residue.get_parent().get_id()
            if chain in dict_of_chains:
                dict_of_chains[chain].append(residue.get_id()[1])
            else:
                dict_of_chains[chain] = [residue.get_id()[1]]

        # build bp dict chain : consecutive neighbors
        for chain in dict_of_chains:
            list_of_residues = dict_of_chains[chain]
            filled_list = fill_list(list_of_residues, cutoff)
            number_of_nodes += len(filled_list)
            dict_of_chains[chain] = filled_list

        radius += 1
        # if radius > start_radius + 1:
        #     print('augmented radius to {}'.format(radius -1))
        if radius == max_radius:
            return 0
    return dict_of_chains


# make a list more complete
# extend around singles, bridge gaps
# filter duplicates
def fill_list(list_to_fill, cutoff):
    list_to_fill.sort()
    min_value = max(0, list_to_fill[0] - 3)
    max_value = list_to_fill[len(list_to_fill) - 1]
    output = []
    previous = list_to_fill[0]
    for i in range(len(list_to_fill)):

        current = list_to_fill[i]

        # bridge gaps within cutoff
        if previous + 2 + (cutoff - 4) > current - 2:
            output.extend(range(previous + 3, current - 2))

        # extend around this element
        extended = range(max(0, current - 2, min_value + 1), min(current + 2, max_value) + 1)
        output.extend(extended)
        previous = current
        min_value = output[len(output) - 1]
    return output


if __name__ == "__main__":
    print(fill_list([5, 23, 23, 45], 5))

'''
# take a list and output another with filled gaps between points closer than a cutoff
# and extension of 2 around others
def fill_list(list_to_fill, cutoff):
    list_to_fill.sort()
    max_value = list_to_fill[len(list_to_fill)-1]
    output = []
    # careful not to put overlap but to include bounds of the consecutive
    # add sequence_started boolean to include the lower bound only if it is the start of a sequence
    sequence_started = False
    for i in range(len(list_to_fill)):
        if list_to_fill[i+1] - list_to_fill[i] < cutoff:
            output.extend(range(list_to_fill[i] + sequence_started, list_to_fill[i+1]+1))
            sequence_started = True
        else:
            sequence_started = False
            isolated = list_to_fill[i]
            print(isolated)
            extended = range(max(0, isolated -2), min(isolated+2, max_value)+1)
            output.extend(extended)
    return output



# Get the binding pocket dictionnary chain : list of residues id (int) of the same chain neighboring a ligand
def fill_neighborhood(neighbors, cutoff):
    dict_of_chains = {}
    for neighbor in neighbors:
        chain = neighbor.get_parent().get_id()

        if chain in dict_of_chains:
            dict_of_chains[chain].append(neighbor.get_id()[1])
        else:
            dict_of_chains[chain] = [neighbor.get_id()[1]]

    for chain in dict_of_chains:
        list_of_residues = dict_of_chains[chain]
        dict_of_chains[chain] = fill_list(list_of_residues, cutoff)
    return dict_of_chains


# encapsulation
# return 0 if there are no atoms around
def find_binding_pocket(structure, ligand, start_radius, max_radius, cutoff, number_min):
    close_neighbors = find_good_neighborhood(structure, ligand, start_radius, max_radius, number_min)
    if not close_neighbors:
        return 0
    binding_pocket = fill_neighborhood(close_neighbors, cutoff)

    return binding_pocket


# returns a dictionary of ligand : binding pocket dictionnary
def find_binding_pockets(structure, start_radius, max_radius, cutoff, number_min):
    ligands = find_ligands_residue(structure)
    binding_pockets = {}
    for ligand in ligands:
        binding_pockets[ligand] = find_binding_pocket(structure, ligand, start_radius, max_radius, cutoff, number_min)
    return binding_pockets
'''


# select residues from the list
# Override Select class to select ligand residues and neighbors residues
class LigandSelect(Bio.PDB.Select):
    def __init__(self, ligand, dict_of_neighbors):
        self.ligand = ligand
        self.dict_of_neighbors = dict_of_neighbors

    def accept_residue(self, residue):
        return residue == self.ligand or \
               (residue.get_parent().get_id() in self.dict_of_neighbors and \
                residue.get_id()[1] in self.dict_of_neighbors[residue.get_parent().get_id()])


# write a list of selected residues in a PDB file at output_path
def write_pdb_output(structure, ligand, neighbors, output_name, io):
    io.set_structure(structure)
    io.save(output_name, LigandSelect(ligand, neighbors))


# takes a string of the path of a pdb file structure and write output of pdb of ligands and binding pocket extracted
def pipeline_path(input_path, output_path, start_radius=5, max_radius=5, cutoff=5, number_min=4, io=Bio.PDB.PDBIO()):
    parser = Bio.PDB.FastMMCIFParser(QUIET=True)
    name = input_path[-8:-4]
    structure = parser.get_structure(name, input_path)
    ligands = find_ligands_residue(structure)
    i = 0
    for ligand in ligands:
        binding_pocket = find_binding_pocket(structure, ligand, start_radius, max_radius, cutoff, number_min)
        if not binding_pocket:
            continue
        output_name = output_path + name + '_' + ligand.get_resname() + '_' + str(i) + '.pdb'
        write_pdb_output(structure, ligand, binding_pocket, output_name, io)
        i += 1


'''
    number_of_nodes = 0
    radius = start_radius

        number_of_nodes = 0

        # search atoms within the radius


        # create the set of resulting residues
        for atom in atom_result:
            parent_residue = atom.get_parent()
            # check if it is a protein residue
            if parent_residue.get_id()[0] == ' ':
                residue_result.add(parent_residue)

        # build bp dict chain : list of residue neighbors
        for residue in residue_result:
            chain = residue.get_parent().get_id()
            if chain in dict_of_chains:
                dict_of_chains[chain].append(residue.get_id()[1])
            else:
                dict_of_chains[chain] = [residue.get_id()[1]]

        # build bp dict chain : consecutive neighbors
        for chain in dict_of_chains:
            list_of_residues = dict_of_chains[chain]
            filled_list = fill_list(list_of_residues, cutoff)
            number_of_nodes += len(filled_list)
            dict_of_chains[chain] = filled_list

        radius += 1
        # if radius > start_radius + 1:
        #     print('augmented radius to {}'.format(radius -1))
        if radius == max_radius:
            return 0
    return dict_of_chains
'''
'''bad slower method
compute first a neighborhood and within this neighborhood compute the radius of each atom of the ligand


# returns the maximal distance between atoms of a ligand
def size_of_ligand(ligand):
    size = 0
    for atom in ligand.get_atoms():
        for atom2 in ligand.get_atoms():
            size = max(atom - atom2, size)
    return size


# quick heuristic for discrimination of the futhest regions from the ligand of the protein
def find_coarse_neighbors_of_ligand(structure, ligand, radius):
    random_atom = next(ligand.get_atoms())
    center = np.array(random_atom.get_coord())

    searcher = Bio.PDB.NeighborSearch(list(structure.get_atoms()))
    return searcher.search(center, radius)


# greedy search in the coarse neighbourhood of the ligand
def find_fine_neighbors_of_ligand(structure, ligand, radius):
    radius_substructure = 1.5 * size_of_ligand(ligand)
    sub_structure = find_coarse_neighbors_of_ligand(structure, ligand, radius_substructure)
    searcher = Bio.PDB.NeighborSearch(sub_structure)

    result = set()
    for atom in ligand.get_atoms():
        L = searcher.search(np.array(atom.get_coord()), radius)
        for i in L:
            result.add(i)
    return result

'''
