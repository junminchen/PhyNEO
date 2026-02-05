from MDAnalysis import *
import subprocess

# we need to cleanup the last snapshot from an output .dcd trajectory before using this to start a new simulation.  The things that need done are:
#  1) fix names of residues that got shortened/cutoff
#  2) remove Drude oscillators
#  3) fix any nanotubes that got shifted
#
#  the easiest way to do 1 and 3, is use starting pdb file as a template

pdb_start = 'start.pdb'
pdb_drudes = 'start_drudes.pdb'
pdb_intermediate = 'temp.pdb'
dcd_file = 'equil_MC.dcd'

# list of residues to take from starting pdb ..
residues = [ 'grpcA' , 'grphA' ]

# list of resnames that we need to rename due to MDAnalysis print truncation ...
# also, some carbons get relabled "CA" instead of "C", not sure why, but fix this
rename = { "BMI ":"BMIM", "acn ":"acnt" , "grp B":"grpcB" , "grp D":"grphD" , "CA":" C" }

# first, pull last frame from dcd with MDAnalysis ...
u=Universe(pdb_drudes,dcd_file)
# go to last frame
u.trajectory[u.trajectory.n_frames-1]
# get all atoms except Drude oscillators...
atoms=u.select_atoms('not name D*')
# write out temporary pdb file with this intermediate information
atoms.write(pdb_intermediate, bonds=None)
# first line is a title line, i'm not sure how to turn this off in MDAnalysis so just delete it ...
subprocess.call('sed -i -e "1d" ' + pdb_intermediate , shell=True)

# now read starting and intermediate pdb, and print atom information from appropriate file
with open(pdb_start) as f1, open(pdb_intermediate) as f2:
    for line1, line2 in zip(f1,f2):
        if any(key in line1 for key in residues):
            print(line1.rstrip())
        else:
            # rename if necessary
            for key , value in rename.items():
                line2 = line2.replace( key , value )
            print(line2.rstrip())
