# Protein Binding
## Set of tools to extract representation of binding pockets

## tools 
hbond parsing using chimera
sdf parsing for ligand information
graph analyser and 3D drawer

## graph 
This is a set of tools to extract data in graph representation, to make the graphs more continuous and represent 
them as pdb or networkx graphs


This is pretty slow and some speed ups were implemented : for instance using KDTree search vs full search vs 
subsetting and searching in a subset
This is overall pretty useless because the graph representation was not really relevant to represent protein 
binding pockets : several strand of the SSE interact in the same pocket and in a continous way.

In an attempt to make graphs more connected, we tried to include hbonds and distance edges in between disconnected 
components to account for the structure, but since the binding rules mostly depend on the structure, this subproject 
was somehow abandonned

The hbond was also a real bottleneck 212/218s because it uses chimera as a hbond finder, which makes the whole pipeline
very slow

bp writes a bp pdb from a pdb
get interaction is used to parse hbonds
graph creates the graph out of these two outputs
main wraps everything up

## shape
This is a set of tools used to extract 3D information from the PDB files. 

This offers two possibilities : 
- bp embedding using pre-downloaded pdb in data/protein/ but they are pretty heavy (50 structures will weigh 32MB so 
the whole thing should weigh about 50GB). This will speed up the process however because the longest part of the 
embedding is to fetch the data (70% runtime). One solution could be to store only the pockets pdb, because they only 
weigh 320kb for 41 samples (100 times lighter) but this would not allow re-embedding pockets.

- bp embedding using a list of pdbs, they are then fetched from the PDB.

