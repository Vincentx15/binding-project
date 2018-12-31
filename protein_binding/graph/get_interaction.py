"""
Script to take list of PDBs and extract hydrogen bonds.
"""

import subprocess
import os
import pandas as pd
import time

AA = ["ALA", "CYS", "ASP", "GLU", "PHE", "GLY", "HIS", "ILE", "LYS", "LEU",
      "MET", "ASN", "PRO", "GLN", "ARG", "SER", "THR", "VAL", "TRP", "TYR"]


def chimera_hbond(pdb_path, dest):
    """
    Creates a file {pdb}.hbonds with hydrogen bond info.
    Returns path to file
    """
    cmd = " ".join(["chimera", "--nogui", "--silent", "--script", "\"../tools/hbond.py\
        %s %s\"" % (pdb_path, dest)])
    subprocess.call(cmd, shell=True)
    return os.path.join(dest, os.path.basename(pdb_path).replace(".pdb", \
                                                                 ".hbonds"))


def ligand_name(bond_dict):
    ligands = set()
    for a, b in zip(bond_dict['name_1'], bond_dict['name_2']):
        if a not in AA:
            ligands.add(a)
        if b not in AA:
            ligands.add(b)

    return ligands


def hbonds_parse(hbond_path):
    """
    Parse hbond file.

    H-bonds (donor, acceptor, hydrogen, D..A dist, D-H..A dist):
    gt
    G 1.A N1     C 18.A N3   G 1.A H1       2.765  1.812
    """
    with open(hbond_path, "r") as hb:
        start = False
        columns = ['name_1', 'num_1', 'chain_1', 'atom_1',
                   'name_2', 'num_2', 'chain_2', 'atom_2']
        bond_dict = {k: [] for k in columns}
        for line in hb:
            if not line.startswith("H-bonds") and start == False:
                continue
            elif line.startswith("H-bonds"):
                start = True
                continue
            elif start:
                # read lines
                info = line.split()
                # print(info)
                bond_dict['name_1'].append(info[0])
                bond_dict['num_1'].append(info[1].split(".")[0])
                bond_dict['chain_1'].append(info[1].split(".")[1])
                bond_dict['atom_1'].append(info[2])
                bond_dict['name_2'].append(info[3])
                bond_dict['num_2'].append(info[4].split(".")[0])
                bond_dict['chain_2'].append(info[4].split(".")[1])
                bond_dict['atom_2'].append(info[5])

            else:
                print("what?")
    return bond_dict


def get_binding_bonds(bond_dict):
    """
    Returns dataframe with atom pairs between RNA and ligand
    """
    bond_df = pd.DataFrame.from_dict(bond_dict)
    bindings = bond_df.loc[(bond_df['name_1'].isin(AA)) != \
                           (bond_df['name_2'].isin(AA))]
    return bindings


# broken
def non_ligand(bond_dict):
    """
    Returns dataframe with atom pairs between RNA and RNA
    """
    bond_df = pd.DataFrame.from_dict(bond_dict)
    non_ligand = bond_df.loc[(bond_df['name_1'].isin(AA)) == \
                           (bond_df['name_2'].isin(AA))]
    return non_ligand


if __name__ == "__main__":
    start_time = time.time()
    chimera_hbond("../data/output_structure/output_subset_noions/1a0f_GTS_0.pdb", "../data/output_graph/test/1.h")
    print("--- %s seconds ---" % (time.time() - start_time))
    h = hbonds_parse("../data/output_graph/test/1.h")
    print(h)
    print("--- %s seconds ---" % (time.time() - start_time))
    pass
