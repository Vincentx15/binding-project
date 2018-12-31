import UFSRCAT as cat
import UFSRPCAT as pcat
from numpy import append, array, zeros, vstack, mean
from rdkit import Chem

"""
Interface class to implement several embeddings
"""

# SMARTS definition of pharmacophore subsets
PHARMACOPHORES = [
    "[#6+0!$(*~[#7,#8,F]),SH0+0v2,s+0,S^3,Cl+0,Br+0,I+0]",  # hydrophobic
    "[a]",  # aromatic
    "[$([O,S;H1;v2]-[!$(*=[O,N,P,S])]),$([O,S;H0;v2]),$([O,S;-]),$([N&v3;H1,H2]-[!$(*=[O,N,P,S])]),$([N;v3;H0]),$([n,o,s;+0]),F]",
    # acceptor
    "[N!H0v3,N!H0+v4,OH+0,SH+0,nH+0]"  # donor
]

# total number of moments // +1 to include the all atom set
NUM_MOMENTS = (len(PHARMACOPHORES) + 1) * 12


class MomentsComputer:
    def __init__(self, pca=False, distance=4, mean=0):
        self.pca = pca
        self.distance = distance
        self.mean = mean

    def usr_moments(self, coords):
        """
        Calculates the USR moments for a set of input coordinates as well as the four
        USR reference atoms.

        :param coords: numpy.ndarray
        return (ref_points, moments)
        """
        if self.pca:
            return pcat.usr_moments(coords, distance=self.distance, mean=self.mean)
        return cat.usr_moments(coords)

    def usr_moments_with_existing(self, coords, ref_points):
        """
        Calculates the USR moments for a set of coordinates and an already existing
        set of four USR reference points.
        """
        if self.pca:
            return pcat.usr_moments_with_existing(coords, ref_points, mean=self.mean)
        return cat.usr_moments_with_existing(coords, ref_points)


class MomentGenerator:
    def __init__(self, pca=True, features=True, hydrogens=False, distance=4, mean=0):
        """
        Can be applied to molecules to generate some embedding
        :param pca: which method to use (true uses pca)
        :param features: include all the features ?
        :param hydrogens: include hydrogens ?
        """
        self.mc = MomentsComputer(pca, distance=distance, mean=mean)
        self.features = features
        self.hydrogens = hydrogens

        if features:
            self.moments = NUM_MOMENTS
            # initialize SMARTS patterns as Mol objects
            self.patterns = [Chem.MolFromSmarts(smarts) for smarts in PHARMACOPHORES]
        else:
            self.moments = 12

    def generate_moments(self, molecule):
        """
        Returns a 2D array of USRCAT moments for all conformers of the input molecule.

        :param molecule: a rdkit Mol object that is expected to have conformers.
        :param hydrogens: if True, then the coordinates of hydrogen atoms will be
                          included in the moment generation.
        """
        # how to suppress hydrogens?
        if not self.hydrogens:
            Chem.RemoveHs(molecule)

        if self.features:
            # create an atom idx subset for each pharmacophore definition
            subsets = []
            for pattern in self.patterns:

                # get a list of atom identifiers that match the pattern (if any)
                matches = molecule.GetSubstructMatches(pattern)

                # append this list of atoms for this pattern to the subsets
                if matches:
                    subsets.extend(zip(*matches))
                else:
                    subsets.append([])

        # initial zeroed array to use with vstack - will be discarded eventually
        all_moments = zeros(self.moments, dtype=float)

        # iterate through conformers and generate USRCAT moments for each
        for conformer in molecule.GetConformers():

            # get the coordinates of all atoms
            coords = {}
            for atom in molecule.GetAtoms():
                point = conformer.GetAtomPosition(atom.GetIdx())
                coords[atom.GetIdx()] = (point.x, point.y, point.z)

            # generate the four reference points and USR moments for all atoms
            ref_points, moments = self.mc.usr_moments(array(list(coords.values())))

            if self.features:
                # generate the USR moments for the feature specific coordinates
                for subset in subsets:

                    # only keep the atomic coordinates of the subset
                    fcoords = array([coords.get(atomidx) for atomidx in subset])

                    # initial zeroed out USRCAT feature moments
                    new_moment = zeros(12)

                    # only attempt to generate moments if there are enough atoms available!
                    if len(fcoords):
                        new_moment = self.mc.usr_moments_with_existing(fcoords, ref_points)

                    # append feature moments to the existing ones
                    moments = append(moments, new_moment)

        # add conformer USRCAT moments to array for this molecule
        all_moments = vstack((all_moments, moments))

        # do not include first row: all zeros!
        return all_moments[1:]


if __name__ == '__main__':
    import time

    start_time = time.time()
    suppl = Chem.SDMolSupplier('../data/aa2ar/decoys_final.sdf')
    UFSR = MomentGenerator()
    USRCAT = MomentGenerator(features=True)
    USRPCAT = MomentGenerator(features=True, pca=True)

    L = []
    i = 0
    for m in suppl:
        print(m)
        if i > 10:
            break
        ufsr_moments = UFSR.generate_moments(m)
        usrcat_moments = USRCAT.generate_moments(m)
        usrpcat_moments = USRPCAT.generate_moments(m)

        ufsr_row = ufsr_moments[0]
        usrcat_row = usrcat_moments[0]
        usrpcat_row = usrpcat_moments[0]

        # print("ufsr_row", ufsr_row)
        print("usrcat_row", usrcat_row)
        i += 1

    print("--- %s seconds ---" % (time.time() - start_time))
