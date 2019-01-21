import os
import pickle


def get_ligand_id(path):
    ligand_id = set()
    for file in os.listdir(path):
        line = file.split('_')
        ligand_name = line[1]
        ligand_id.add(ligand_name)
    return ligand_id


L = get_ligand_id('../data/output_pdb/whole/')
print(L)
pickle.dump(L, open('../data/ligands/whole_ligands_id.p', 'wb'))
