"""
Gets basic information on ligand file
Outputs info as csv from pandas DataFrame
"""

import re
import pandas as pd
import pickle


def parse(filepath):
    word = re.compile('\w+')
    info_dict = {}

    with open(filepath, "r") as sdf:
        current_feature = ""
        current_feature_loc = 0
        current_mol = 0
        for i, l in enumerate(sdf):
            if "$$$$" in l:
                current_mol += 1
                pass
            elif l.startswith(">"):
                feat = re.findall(word, l.split()[1])[0]
                current_feature = feat
                current_feature_loc = i
                pass
            elif i == current_feature_loc + 1 and current_feature_loc > 0:
                val = l.strip()
                if not l.strip():
                    val = "Nan"
                info_dict.setdefault(current_feature, []).append(val)
            else:
                continue

    df = pd.DataFrame.from_dict(info_dict)
    return df


def ligand_dict(filepath):
    df = parse(filepath)
    uniques = df.drop_duplicates(subset='ChemCompId')
    dic = {}
    for row in uniques.itertuples():
        dic[row.ChemCompId] = ("ION" in row.Name, row.MolecularWeight)
    return dic


if __name__ == "__main__":
    dic = ligand_dict("../data/source_data/ligands/Ligands_noHydrogens_noMissing_52573_Instances.sdf")
    s = set()
    for key, value in dic.items():
        if not value[0]:
            s.add(key)
    with open("../data/source_data/ligands/set_of_ligands.pickle", "wb") as output_file:
        pickle.dump(s, output_file)
