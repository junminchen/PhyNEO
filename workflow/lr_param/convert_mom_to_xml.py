#!/usr/bin/env python3
"""

Last Updated:
"""

# Standard modules
import numpy as np
import sys
import os
from copy import copy
import jax.numpy as jnp

from jax.config import config
config.update("jax_enable_x64", True)

###########################################################################
####################### Global Variables ##################################
error_message='''
---------------------------------------------------------------------------
Improperly formatted arguments. Proper usage is as follows:

$ convert_mom_to_xml.py <mom_ifile> <axes_ifile>

(<...> indicates required argument, [...] indicates optional argument)
---------------------------------------------------------------------------
    '''

###########################################################################
###########################################################################


###########################################################################
######################## Command Line Arguments ###########################
# try:
#     mom_ifile = sys.argv[1]
#     axes_ifile = sys.argv[2]
# except IndexError:
#     print(error_message)
#     sys.exit()


###########################################################################
###########################################################################

###########################################################################
def rotate_sph_harm(moments, r):
    '''
    Takes a set of spherical tensor moments and rotation matrix R for ordinary
    3-d vectors and returns the transformed set of spherical harmonics. 

    References
    ----------
    subroutine "wigner" from rotations.f90 in Anthony Stone's ORIENT code; this
    function borrows heavily from his code

    Wigner D-matrices:
    https://en.wikipedia.org/wiki/Wigner_D-matrix
    Stone, Theory of Intermolecular Forces, 2nd ed. p. 274

    Real and complex Spherical Harmonics:
    https://en.wikipedia.org/wiki/Spherical_harmonics#Real_form

    '''


    # Set up Wigner transformation matrix D. Note that D is block-diagonal, so
    # we can set up each block (D0, D1, and D2) separately.
    xx=r[0,0]
    xy=r[0,1]
    xz=r[0,2]
    yx=r[1,0]
    yy=r[1,1]
    yz=r[1,2]
    zx=r[2,0]
    zy=r[2,1]
    zz=r[2,2]

    # D0
    D0 = np.ones(1)

    # D1
    D1 = np.array([[ zz, zx, zy ],
                   [ xz, xx, xy ],
                   [ yz, yx, yy ]])

    # D2
    D2 = np.zeros((5,5))
    rt3 = np.sqrt(3)
    D2[0,0] = (3*zz**2-1)/2
    D2[0,1] = rt3*zx*zz
    D2[0,2] = rt3*zy*zz
    D2[0,3] = (rt3*(-2*zy**2-zz**2+1))/2
    D2[0,4] = rt3*zx*zy
    D2[1,0] = rt3*xz*zz
    D2[1,1] = 2*xx*zz-yy
    D2[1,2] = yx+2*xy*zz
    D2[1,3] = -2*xy*zy-xz*zz
    D2[1,4] = xx*zy+zx*xy
    D2[2,0] = rt3*yz*zz
    D2[2,1] = 2*yx*zz+xy
    D2[2,2] = -xx+2*yy*zz
    D2[2,3] = -2*yy*zy-yz*zz
    D2[2,4] = yx*zy+zx*yy
    D2[3,0] = rt3*(-2*yz**2-zz**2+1)/2
    D2[3,1] = -2*yx*yz-zx*zz
    D2[3,2] = -2*yy*yz-zy*zz
    D2[3,3] = (4*yy**2+2*zy**2+2*yz**2+zz**2-3)/2
    D2[3,4] = -2*yx*yy-zx*zy
    D2[4,0] = rt3*xz*yz
    D2[4,1] = xx*yz+yx*xz
    D2[4,2] = xy*yz+yy*xz
    D2[4,3] = -2*xy*yy-xz*yz
    D2[4,4] = xx*yy+yx*xy

    # Create D matrix from D0, D1, and D2
    D = np.zeros((9,9))
    D[0,0] = D0
    D[1:4,1:4] = D1
    D[4:9,4:9] = D2

    # Perform the rotation
    rotated_moments = np.dot(D,moments)

    return rotated_moments
###########################################################################


####################################################################################################    
def get_local_to_global_rotation_matrix(global_xyz,local_xyz):
    '''Compute the rotation matrix that transforms coordinates from the
    local to global coordinate frame.

    Parameters
    ----------
    global_xyz : 3darray
        xyz coordinates of all atoms and all monomer configurations for which multipole
        interactions will later be computed, of size (ndatpts,natoms,3).
    local_xyz : 2darray
        xyz coordinates of each atom in the multipolar coordinate frame,
        of size (natoms,3).

    Returns
    -------
    rotation_matrix : 3x3 array
        Rotation matrix that transforms coordinates from the local (i.e.
        monomer multipole) frame to the global coordinate frame.

    '''


    # Get rotation vector from the global axis to the local axis
    # http://stackoverflow.com/questions/1171849/finding-quaternion-representing-the-rotation-from-one-vector-to-another
    rotation_matrix = np.array([np.identity(3) for x in global_xyz])
    trans_local_xyz = local_xyz[np.newaxis,:]
    trans_global_xyz = global_xyz
    # if xyz is an atom, don't need to rotate local axes
    if len(global_xyz[0]) == 1:
        assert np.allclose(trans_local_xyz,trans_global_xyz,atol=1e-5)
        return rotation_matrix

    # Otherwise, rotate coordinate frame in order to properly align x-axis
    v1 = trans_local_xyz[:,0]
    v2 = trans_global_xyz[:,0] 
    q_w,q_vec = get_rotation_quaternion(v1,v2)

    np.seterr(all="ignore")
    trans_local_xyz = rotate_local_xyz(q_w, q_vec, trans_local_xyz)
    rotation_matrix = rotate_local_xyz(q_w,q_vec, rotation_matrix)
    np.seterr(all="warn")

    # If xyz is diatomic, don't need to rotate second set of axes
    if len(global_xyz[0]) == 2:
        assert np.allclose(trans_local_xyz,trans_global_xyz,atol=1e-5)
        return rotation_matrix

    # Otherwise, rotate coordinate frame to properly align y- and z-axes
    v1 = trans_local_xyz[:,0]
    v2 = trans_global_xyz[:,0]
    for i in range(2,len(global_xyz[0])):
        # Search for vector in molecule that is not parallel to the first
        v1b = trans_local_xyz[:,i] #- trans_local_xyz[:,0]
        v2b = trans_global_xyz[:,i] #- trans_global_xyz[:,0]
        v3 = np.cross(v1,v1b)
        v4 = np.cross(v2,v2b)

        if not np.array_equal(v3,np.zeros_like(v3)):
            break
    else:
        # All vectors in molecule are parallel; hopefully molecules are
        # now aligned
        assert np.allclose(trans_local_xyz,trans_global_xyz,atol=1e-5)
        return rotation_matrix


    # Align an orthogonal vector to v1; once this vector is aligned, the
    # molecules should be perfectly aligned
    q_w,q_vec = get_rotation_quaternion(v3,v4,v_orth=v1)
    np.seterr(all="ignore")
    trans_local_xyz = rotate_local_xyz(q_w, q_vec, trans_local_xyz)
    rotation_matrix = rotate_local_xyz(q_w,q_vec, rotation_matrix)
    np.seterr(all="warn")

    try:
        assert np.allclose(trans_local_xyz,trans_global_xyz,atol=1e-5)
    except AssertionError:
        print('bad rotation!!')
        print( np.max(trans_local_xyz-trans_global_xyz))

        print( trans_local_xyz[0])
        print( '---')
        print( trans_global_xyz[0])
        sys.exit()

    # Check that rotation matrix actually transforms local axis into global
    # axis
    if not np.allclose(np.dot(rotation_matrix,local_xyz)[0],global_xyz):
        print( rotation_matrix)
        print(local_xyz)
        print( np.dot(rotation_matrix,local_xyz))
        print() 
        print( 'Rotation matrix does not actually transform local axis into global reference frame!')
        sys.exit()

    return rotation_matrix
####################################################################################################    


####################################################################################################    
def get_rotation_quaternion(v1,v2,v_orth=np.array([1,0,0]),tol=1e-16):
    '''Given two vectors v1 and v2, compute the rotation quaternion
    necessary to align vector v1 with v2.

    In the event that v1 and v2 are antiparallel to within some tolold
    tol, v_orth serves as a default vector about which the rotation
    from v2 to v1 will occur.

    Parameters
    ----------
    v1 : 2darray
        Vector of size (ndatpts,3) to be rotated.
    v2 : 2darray
        Vector of size (ndatpts,3) that is to remain fixed.
    v_orth : 1darray, optional
        Cartesian vector providing a default rotation vector in the event
        that v1 and v2 are antiparallel.
    tol : float, optional
        Tolerance from zero at which two vectors are still considered
        antiparallel.

    Returns
    -------
    q_w : 1darray
        Rotation amount (in radians) for each of the ndatpts by which v1
        is to be rotated.
    q_vec : 2darray
        Vector for each of the ndatpts about which v1 is to be rotated.

    '''

    v1 /= np.sqrt(np.sum(v1*v1,axis=-1))[:,np.newaxis]
    v2 /= np.sqrt(np.sum(v2*v2,axis=-1))[:,np.newaxis]

    dot = np.sum(v1*v2,axis=-1)

    q_vec = np.where(dot[:,np.newaxis] > -1.0 + tol,
            np.cross(v1,v2), v_orth)

    # For antiparallel vectors, rotate 180 degrees
    q_w = np.sqrt(np.sum(v1*v1,axis=-1)*np.sum(v2*v2,axis=-1)) + dot

    # Normalize quaternion
    q_norm = np.sqrt(q_w**2 + np.sum(q_vec**2,axis=-1))
    q_w /= q_norm
    q_vec /= q_norm[:,np.newaxis]

    return q_w, q_vec
####################################################################################################    


####################################################################################################    
def normalize(matrix, axis=1, ord=2):
    normalised = matrix / jnp.linalg.norm(matrix, axis=axis, keepdims=True, ord=ord)
    return normalised

def read_local_axis_information(atoms,positions,axisfile):
    ZThenX = 0
    Bisector = 1
    ZBisect = 2
    ThreeFold = 3
    Zonly = 4
    NoAxisType = 5
    LastAxisTypeIndex = 6
   
    # axisfile = np.loadtxt(ifile,dtype=int)
    axis_types = axisfile[:,0]
    axis_indices = axisfile[:,1:] 
 
    z_atoms = jnp.array(axis_indices[:, 0])
    x_atoms = jnp.array(axis_indices[:, 1])
    y_atoms = jnp.array(axis_indices[:, 2])
    
    # z_atoms = axis_indices[:, 0]
    # x_atoms = axis_indices[:, 1]
    # y_atoms = axis_indices[:, 2]
    
    Zonly_filter = (axis_types == Zonly)
    not_Zonly_filter = jnp.logical_not(Zonly_filter)
    Bisector_filter = (axis_types == Bisector)
    ZBisect_filter = (axis_types == ZBisect)
    ThreeFold_filter = (axis_types == ThreeFold)

    positions = jnp.array(positions)
    n_sites = positions.shape[0]
    # box_inv = jnp.linalg.inv(box)

    ### Process the x, y, z vectors according to local axis rules
    vec_z = positions[z_atoms] - positions
    # vec_z = pbc_shift(vec_z, box, box_inv)
    vec_z = normalize(vec_z)
    vec_x = jnp.zeros((n_sites, 3))
    vec_y = jnp.zeros((n_sites, 3))
    # Z-Only
    x_of_vec_z = jnp.round(jnp.abs(vec_z[:,0]))
    vec_x_Zonly = jnp.array([1.-x_of_vec_z, x_of_vec_z, jnp.zeros_like(x_of_vec_z)]).T[Zonly_filter]
    vec_x = vec_x.at[Zonly_filter].set(vec_x_Zonly)
    # for those that are not Z-Only, get normalized vecX
    vec_x_not_Zonly = positions[x_atoms[not_Zonly_filter]] - positions[not_Zonly_filter]
    # vec_x_not_Zonly = pbc_shift(vec_x_not_Zonly, box, box_inv)

    vec_x = vec_x.at[not_Zonly_filter].set(normalize(vec_x_not_Zonly, axis=1))
    # Bisector
    if np.sum(Bisector_filter) > 0:
        vec_z_Bisector = vec_z[Bisector_filter] + vec_x[Bisector_filter]
        vec_z = vec_z.at[Bisector_filter].set(normalize(vec_z_Bisector, axis=1))
    # z-bisector
    if np.sum(ZBisect_filter) > 0:
        vec_y_ZBisect = positions[y_atoms[ZBisect_filter]] - positions[ZBisect_filter]
        # vec_y_ZBisect = pbc_shift(vec_y_ZBisect, box, box_inv)
        vec_y_ZBisect = normalize(vec_y_ZBisect, axis=1)
        vec_x_ZBisect = vec_x[ZBisect_filter] + vec_y_ZBisect
        vec_x = vec_x.at[ZBisect_filter].set(normalize(vec_x_ZBisect, axis=1))
    # ThreeFold
    if np.sum(ThreeFold_filter) > 0:
        vec_x_threeFold = vec_x[ThreeFold_filter]
        vec_z_threeFold = vec_z[ThreeFold_filter]
        
        vec_y_threeFold = positions[y_atoms[ThreeFold_filter]] - positions[ThreeFold_filter]
        # vec_y_threeFold = pbc_shift(vec_y_threeFold, box, box_inv)
        vec_y_threeFold = normalize(vec_y_threeFold, axis=1)
        vec_z_threeFold += (vec_x_threeFold + vec_y_threeFold)
        vec_z_threeFold = normalize(vec_z_threeFold)
        
        vec_y = vec_y.at[ThreeFold_filter].set(vec_y_threeFold)
        vec_z = vec_z.at[ThreeFold_filter].set(vec_z_threeFold)
    
    # up to this point, z-axis should already be set up and normalized
    xz_projection = jnp.sum(vec_x*vec_z, axis = 1, keepdims=True)
    vec_x = normalize(vec_x - vec_z * xz_projection, axis=1)
    # up to this point, x-axis should be ready
    vec_y = jnp.cross(vec_z, vec_x)

    return np.array(jnp.stack((vec_x, vec_y, vec_z), axis=1))


####################################################################################################    


####################################################################################################    
def rotate_local_xyz(a,vector=np.array([0,0,1]),local_xyz=np.array([1,2,3]),thresh=1e-14):
    """Compute the new position of a set of points 'local_xyz' in 3-space (given as a
    3-membered list) after a rotation by 2*arccos(a) about the vector [b,c,d].

    This method uses quaternions to accomplish the transformation. For more
    information about the mathematics of quaternions, refer to 
    http://graphics.stanford.edu/courses/cs164-09-spring/Handouts/handout12.pdf

    Parameters
    ----------
    a : 1darray
        Array of size ndatpts indicating the angle by which local_xyz is to be
        rotated, where the rotation angle phi = 2*arccos(a).
    vector : 2darray
        Array of size (ndatpts,3) containing the vector about which local_xyz
        is to be rotated.
    local_xyz : 3darray
        Array of size (ndatpts,natoms,3) in cartesian space to be rotated by phi about
        vector.

    Returns
    -------
    new_local_xyz : 3darray
        Array of size (ndatpts,natoms,3) in cartesian space corresponding
        to the rotated set of points.

    """

    #Compute unit quaternion a+bi+cj+dk
    b = vector[:,0]
    c = vector[:,1]
    d = vector[:,2]
    norm = np.sqrt(np.sin(np.arccos(a))**2/(b**2+c**2+d**2))
    b *= norm
    c *= norm
    d *= norm

    # Compute quaternion rotation matrix:
    [a2,b2,c2,d2] = [a**2,b**2,c**2,d**2]
    [ab,ac,ad,bc,bd,cd] = [a*b,a*c,a*d,b*c,b*d,c*d]

    rotation = np.array([[ a2+b2-c2-d2 ,  2*bc-2*ad  ,  2*bd+2*ac  ],\
                         [  2*bc+2*ad  , a2-b2+c2-d2 ,  2*cd-2*ab  ],\
                         [  2*bd-2*ac  ,  2*cd+2*ab  , a2-b2-c2+d2 ]])
    rotation = np.rollaxis(rotation,-1,0)

    # Compute rotation of local_xyz about the axis
    new_local_xyz = np.sum(rotation[:,np.newaxis,:,:]*local_xyz[...,np.newaxis,:],axis=-1)
    # Return new local_xyz unless rotation vector is ill-defined (occurs for
    # zero rotation), in which case return the original local_xyz
    new_local_xyz = np.where(np.sqrt(np.sum(vector*vector,axis=-1))[:,np.newaxis,np.newaxis] > thresh, 
                        new_local_xyz, local_xyz)
    return new_local_xyz
####################################################################################################    


###########################################################################
########################## Main Code ######################################
def mainprocess(mom_ifile, axes_ifile):
    # Read in data from mom file
    # with open(mom_ifile,'r') as f:
    #     mom_lines = [ line.split() for line in f.readlines() ]
    mom_lines = [ line.split() for line in mom_ifile ]

    # Obtain point charge, dipole, and quadrupole information for each atom
    tag = 'Type'
    atoms = []
    labels = []
    moments = []
    xyz = []
    for line in mom_lines:
        if len(line) > 5 and line[4] == tag:
            atoms.append(line[0])
            labels.append([])
            moments.append([])
            xyz.append([float(i) for i in line[1:4]])
        elif not line or '!' in line[0] or line[0] in ['End','Units']:
            continue
        else:
            labels[-1].append(line[0])
            moments[-1].append(float(line[2]))

    # Ensure labels are ordered normally
    label_order = ['Q00', 'Q10', 'Q11c', 'Q11s', 'Q20', 'Q21c', 'Q21s', 'Q22c', 'Q22s']
    for l in labels:
        assert l == label_order

    # Read in axis information for each atom
    local_axis = read_local_axis_information(atoms,xyz,axes_ifile)
    global_xyz = np.eye(3)

    # Compute quaternion to rotate global axis frame to local one
    rotated_moments = []
    for iatom in range(len(atoms)):
        R = get_local_to_global_rotation_matrix(global_xyz[np.newaxis,...],local_axis[iatom])
        Rinv = np.linalg.inv(R)

        rotated_moments.append(rotate_sph_harm(moments[iatom], Rinv[0]))



    # Rotate each multipole according to the local axis information provided in
    # the .axes file
    au_to_nm = 0.0529177
    template = '<Atom type="{:s}" kz="fill" kx="fill" c0="{:.6g}" d1="{:.6g}" d2="{:.6g}" d3="{:.6g}" q11="{:.6g}" q21="{:.6g}" q22="{:.6g}" q31="{:.6g}" q32="{:.6g}" q33="{:.6g}"  />'

    # Print labels in Cartesian coordinates and OpenMM units
    # print
    tmpf=[]
    for atom, m in zip(atoms,rotated_moments):
        c0 = m[0] # point charge
        d1 = m[2] # Q11c = dx = d1
        d2 = m[3] # Q11s = dy = d2
        d3 = m[1] # Q10  = dz = d3
        # OpenMM prints uses values for the quadrupole moments that are 3x the
        # value given by CamCASP (presumably due to normalization)
        q11 = (1.0/3)*(np.sqrt(3)/2*m[7] - 0.5*m[4])
        q21 = (1.0/3)*(np.sqrt(3)/2*m[8])
        q22 = (1.0/3)*(-np.sqrt(3)/2*m[7] - 0.5*m[4])
        q31 = (1.0/3)*(np.sqrt(3)/2*m[5])
        q32 = (1.0/3)*(np.sqrt(3)/2*m[6])
        q33 = (1.0/3)*(m[4])

        d1 *= au_to_nm
        d2 *= au_to_nm
        d3 *= au_to_nm

        q11 *= au_to_nm**2
        q21 *= au_to_nm**2
        q22 *= au_to_nm**2
        q31 *= au_to_nm**2
        q32 *= au_to_nm**2
        q33 *= au_to_nm**2

        xml = template.format(atom,c0,d1,d2,d3,q11,q21,q22,q31,q32,q33)
        tmpf.append(' '.join(xml.split()))
        # print( ' '.join(xml.split()))
    return tmpf
    # print()

###########################################################################
###########################################################################
def main():
    mom_file='./reform.mom'
    axes_ifile=np.loadtxt('axes',dtype=int)
    with open(mom_file,'r') as f:
        momf = f.readlines()
    tmpf=mainprocess(momf,axes_ifile)
    f=open('tmp_new','w')
    for line in tmpf:
        print(line,file=f)

if __name__ == "__main__":
    main()
