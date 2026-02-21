import mdtraj as md

# A simple script to recenter a trajetory so that it appears correctly in VMD
traj = md.load("trajectory.dcd", top="solvated_ethane_from_openmm_setup.pdb")
traj.center_coordinates()
traj.save_dcd("trajectory_recentered.dcd")
