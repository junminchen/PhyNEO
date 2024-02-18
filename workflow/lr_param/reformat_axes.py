#!/usr/bin/env python
import numpy as np
# Open the input file for reading
with open('localframe', 'r') as f:
    # Skip the first 3 lines of the file
    for i in range(4):
        next(f)
    
    # Initialize a dictionary to store the axes information
    #N = sum(1 for line in f)  # Count the number of lines in the file
    #f.seek(0)
    #indices = np.zeros((N,3),dtype=int) 
    #types = np.zeros((N,),dtype=int)
    lines = f.readlines()
    indices = np.zeros((len(lines),3),dtype=int)
    types = np.zeros((len(lines),),dtype=int)    
    # Loop through each line of the file
    for line in lines:
        # Split the line into a list of tokens
        tokens = line.split()
        #print(tokens)
        # Extract the atom number, axis type, and z and x coordinates
        atom_num = int(tokens[0])-1
        axis_type = tokens[2]
        z_coord = int(tokens[3])-1
        x_coord = int(tokens[4])-1
        y_coord = int(tokens[5])-1
        
        indices[atom_num][0] = z_coord
        indices[atom_num][1] = x_coord
        indices[atom_num][2] = y_coord
         
        if axis_type == 'Z-then-X':
            types[atom_num] = 0
        
        elif axis_type == 'Bisector':
            types[atom_num] = 1
        
        elif axis_type == 'Z-Bisect':
            types[atom_num] = 2
        
        elif axis_type == '3-Fold':
            types[atom_num] = 3
        
        elif axis_type == 'Z-Only':
            types[atom_num] = 4

        elif axis_type == 'None':
            types[atom_num] = 5

        else:
            types[atom_num] = 6

#print(types)
#print(indices)

# Open the output file for writing
with open('axes', 'w') as f:
    # Write the header line to the file
    # f.write('Axes\n')
    # Loop through each axis in the dictionary
    for i in range(len(types)):
        # Write the z-axis information to the file
        f.write(f"{str(types[i])}    {'  '.join(str(j) for j in indices[i])}\n")

