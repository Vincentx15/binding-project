import os
import sys
from chimera import runCommand


# open model
pdb = sys.argv[1]
dest = sys.argv[2]
runCommand("open %s" % pdb)
runCommand("hbond namingStyle simple saveFile %s" % dest)


