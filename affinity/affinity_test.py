"""
Test model on affinity benchmark data.
"""
import os
from collections import namedtuple
import pickle
import os.path as osp

import numpy as np
from rdkit import Chem


def sdf_parse(sdf_file, fp_dict):
    """
    Extract screen data from each SDF.
    """
    suppl = Chem.SDMolSupplier(sdf_file)
    screen = []
    Ligand = collections.namedtuple('Ligand', 'smiles affinity fp lig_id')
    for mol in suppl:
        try:
            sm = Chem.MolToSmiles(mol)
            data = mol.GetPropsAsDict()
            fp = fp_dict[sm]
            affinity = data['Enzymologic: Ki nM ']
            lig_id = data['HET ID']
            screen.append(Ligand(sm, affinity, fp, lig_id))
        except:
            pass
    return screen

def screen_score(screen, pred):
    #sort screen ligands by similarity to predicted
    plt.scatter([s.affinity for s in screen],
                [dist(pred, s.affinity) for s in screen])
    pass

def test_pockets(test_ids, preds, val_dir='data/validation_sets'):
    # for pdbid, lig_name in test_ids:
    for n in os.listdir(val_dir):
        pdbid = n.split("_")[0]
        try:
            f_name = f'{pdbid.upper()}_Validation_Affinities.sdf'
            screen = sdf_parse(osp.join(val_dir, f_name), '')
            pred = preds[(pdbid, screen[0].lig_id)]
            screen_score(screen, pred)
        except:
            pass

if __name__ == "__main__":
    to_test = {('1xyz', 'ACA'): 5,
               ('2abc', 'XXX'): 4}
    validation_ids = pickle.load(open('data/test_ids.pickle', 'rb'))
    test_pockets(validation_ids, '')
    pass
