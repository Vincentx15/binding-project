#Shape pipeline

## bp_extract 
deals with taking a PDB and extracting the relevant atoms for the pocket.
 It mostly runs by looking around each atom in the ligands and 
 writing pdbs of this subregion
 
## USFR_feature
Computes USFR features from an array

## UFSRPCAT
Same with USRPCAT
TODO : merge them

## main
Mostly to call these tools on directories full of pdb and controls the calls to the methods

## data_preprocessor
Reads the written extracted PDB and return some extracted tensor that serves as input for 
3D conv_net/tensorfield

