import Bio.PDB
import scipy as np
import pickle
import UFSR_feature as ufsr
import time

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

'''
Extraction tools
'''


def is_an_acceptable_ligand(residue):
    """
    :param residue: Biopython Residue
    :return: Boolean : Is the residue an acceptable ligand (no ion, not water?) plus is it in the ligand set ?
    """
    return residue.get_resname() not in (["HOH"] + AA) and residue.get_resname() in set_of_ligands


def find_ligands_residue(structure):
    """
    Faster to compute this way because of the fast parsing of the mmcif
    :param structure: Biopython structure
    :return: identifiers of all ligands in a structure
    """
    ret = []
    for model in structure:
        for chain in model:
            for residue in chain:
                if is_an_acceptable_ligand(residue):
                    ret += [residue]
    return ret


def find_spatial_binding_pocket(structure, ligand, start_radius, max_radius, number_min):
    """
    Find the smaller pocket
    Find the neighborhood of each atom and get the intersection
    :param structure: Biopython structure to search
    :param ligand: Biopython residue
    :param start_radius: Radius to start with, should be small to avoid taking everything, big enough to not be empty
    :param max_radius: The radius to stop to
    :param number_min: The cutoff under which one needs to stop picking neighbors
    :return:
    """
    searcher = Bio.PDB.NeighborSearch(list(structure.get_atoms()))
    neighbors = set()
    # Start by picking the neighbors closer to start radius
    for atom in ligand.get_atoms():
        center = np.array(atom.get_coord())
        radius = start_radius
        atom_result = []
        # Get the number_min neighbors of each point by increasing the start_radius while the max_radius wasn't reached
        while len(atom_result) < number_min and radius < max_radius:
            # Start the search with the given center and radius
            atom_result = []
            atom_neighbors = searcher.search(center, radius)
            # Filter out atoms part of water or pocket residue
            for atom_neighbor in atom_neighbors:
                residue = atom_neighbor.get_parent()
                if residue != ligand and residue.get_resname() != 'HOH':
                    atom_result.append(atom_neighbor)
            radius += 1
        for neighbor in atom_result:
            neighbors.add(neighbor)
    return neighbors


def pocket_to_array(neighbors):
    return np.array([atom.get_coord() for atom in neighbors])


'''
To write a PDB of the result
Select residues from the list : Biopython IO way
'''


# Override Select class to select ligand residues and neighbors residues
class LigandSelect(Bio.PDB.Select):
    def __init__(self, ligand, neighbors):
        self.ligand = ligand
        self.neighbors = neighbors

    def accept_atom(self, atom):
        return atom in self.neighbors or atom.get_parent() == self.ligand


def write_pdb_output(structure, ligand, neighbors, output_name, io):
    """
    write a list of selected residues in a PDB file at output_path
    """
    io.set_structure(structure)
    io.save(output_name, LigandSelect(ligand, neighbors))


'''
Pipelines
'''


# takes a string of the path of a pdb file structure and write output of pdb of ligands and binding pocket extracted
def path_to_pdb(input_path, output_path, start_radius=3, max_radius=10, number_min=4, io=Bio.PDB.PDBIO()):
    parser = Bio.PDB.FastMMCIFParser(QUIET=True)
    name = input_path[-8:-4]
    structure = parser.get_structure(name, input_path)
    ligands = find_ligands_residue(structure)
    i = 0
    for ligand in ligands:
        binding_pocket = find_spatial_binding_pocket(structure, ligand, start_radius, max_radius, number_min)
        if not binding_pocket:
            continue
        output_name = output_path + name + '_' + ligand.get_resname() + '_' + str(i) + '.pdb'
        write_pdb_output(structure, ligand, binding_pocket, output_name, io)
        i += 1


# 0.11s avant la for
# 0.17s dedans (pour construire le vecteur usfr, duplicates est instantané)
# 0.27s en tout
# input_path = '../data/protein/sample/1qfu.cif'
def path_to_ufsr_dict(input_path, start_radius=3, max_radius=10, number_min=4, number_of_moments=3):
    """
    Get the ufsr of the different pockets, sorted by ligands, in a neighborhood
    :param input_path: use the preprocessed pdb
    :param start_radius:
    :param max_radius:
    :param number_min:
    :param number_of_moments:
    :return: dict of ligand - array of coordinates
    """
    parser = Bio.PDB.FastMMCIFParser(QUIET=True)
    name = input_path[-8:-4]
    structure = parser.get_structure(name, input_path)
    ligands = find_ligands_residue(structure)
    output = {}
    for ligand in ligands:
        # get the corresponding array and make it a USFR vector
        binding_pocket = find_spatial_binding_pocket(structure, ligand, start_radius, max_radius, number_min)
        if not binding_pocket:
            continue
        arr = pocket_to_array(binding_pocket)
        usfr_vector = ufsr.encode(arr, number_of_moments=number_of_moments)

        # check for duplicate pockets
        duplicate = False
        for key in output.keys():
            if key.get_resname() == ligand.get_resname():
                if np.array_equal(output[key], usfr_vector):
                    duplicate = True
                    break
        if not duplicate:
            output[ligand] = usfr_vector
    return output


if __name__ == "__main__":
    start_time = time.time()
    dic = path_to_ufsr_dict('../data/protein/sample/1qfu.cif')
    print(" time --- %s seconds ---" % (time.time() - start_time))
    pass
