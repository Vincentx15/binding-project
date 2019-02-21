import shutil
import os

# Script to remove duplicates : pockets from the same pdb, that have the same ligand

src_dir = 'pockets/whole'
dst_dir = 'pockets/unique_pockets'
os.mkdir(dst_dir)

seen = set()
for i, item in enumerate(os.listdir(src_dir)):
    try:
        pdb, ligand = item.split('_')[0:2]
    except:
        continue
    if (pdb, ligand) not in seen:
        src = os.path.join(src_dir, item)
        shutil.copy(src, dst_dir)
        seen.add((pdb, ligand))
    if not i % 1000:
        print(i)
