#!/usr/bin/env python

from simtk.openmm.app import *
from simtk.openmm import *
from simtk.unit import *
from sys import stdout
from time import gmtime, strftime
from datetime import datetime
from copy import deepcopy
import os
import sys
import numpy
import argparse
import shutil


#**********************************
# this model contains the following primary parent and child classes:
# 
# Conductor_Virtual(Parent):  This is the general parent class for any conductor in the system.
#      every conductor has duplicate copies of each atom, encompassing a "Virtual" and "Real" layer.
#      the "Virtual" layer has electrostatic interactions only, so that electrostatic boundary conditions
#      can be solved to derive the charges of each conductor during MD steps.
#      the "Real" layer is then used for repulsive/VDWs (non-electrostatic) interactions.
#      this parent class takes care of setting up the necessary exclusions among and between the Real and Virtual layers
#
# Electrode_Virtual(Conductor_Virtual):  This is a child class of the Conductor_Virtual Parent class.  This child class is created
#                                        for the flat electrodes (cathode and anode) that are used to fix the applied voltage to the system
#
# Buckyball_Virtual(Conductor_Virtual):  This is a child class of the Conductor_Virtual Parent class for a Buckyball (conductor) near an Electrode
#
#
#**********************************

# conversion factors/parameters:  Could do this internally with OpenMM units/quantities, which is more transparent?  Maybe we should change...
conversion_nmBohr = 18.8973
conversion_KjmolNm_Au = conversion_nmBohr / 2625.5  # bohr/nm * au/(kJ/mol)
conversion_eV_Kjmol = 96.487


#  Simple class to hold atom info.  It is named specifically for this code (atom_MM), so as not to be confused with intrinsic atom class of OpenMM ...
class atom_MM(object):
    def __init__(self, element, charge, atom_index , x , y , z ):
        self.element = element
        self.charge  = charge
        self.atom_index = atom_index
        self.x = x; self.y=y; self.z=z

        # this is vector normal to conductor surface at this atom.
        # will only be used by certain classes (e.g. Buckyball_Virtual)
        # we don't define getters/setters for this because it is so specialized...
        self.nx = 0.0; self.ny = 0.0; self.nz = 0.0

        # this will be set within specific child class 
        self.area_atom = 0.0



#*********************************
# this is an electrode class, that
# should be created for cathode/anode `virtual' layers
#
# practically, each conductor is divided into `real' and `virtual'
# layers, with the `real' layers for steric/VDWs interactions and the
# `virtual' layer for electrostatics.  The `virtual' layer is then utilized/modfied
#  within the Poisson solver to solve for charges satisifying the specific conductor boundary conditions
#
#  Note this is done slightly different in the 'Buckyball_Virtual' child class, as the 'real' layer will
#  contain uniform charges from the net charge transfer---see the child class for this...
#
#  input:
#        electrode_identifier : this could be either residuename of the electrode or chain index of a group of residues in the .pdb input file
#        electrode_type:  set to either "cathode" or "anode"
#        Voltage       : Voltage drop between electrodes, in Volts
#        MMsys           : QM/MM simulation object for the MM system
#        chain  (True/False) : if true, initialize electrodes by chain name rather than residue name
#        exclude_element     : this is tuple that can be filled with elements to exclude, e.g dummy Hydrogen
#*********************************
class Conductor_Virtual(object):
    def __init__(self, electrode_identifier, electrode_type, Voltage, MMsys, chain_flag, exclude_element):
        if isinstance( electrode_identifier , tuple ):
            self.electrode_identifier    = electrode_identifier[0] # first entry in tuple is primary electrode sheet
        else:
            self.electrode_identifier    = electrode_identifier
        self.electrode_type = electrode_type
        self.Voltage        = Voltage * conversion_eV_Kjmol  # convert input Voltage from Volts to kJ/mol
        self.z_pos          = 0.0  # z position of electrode, will be set later ..
        self.Q_analytic     = 0.0  # will be calculated later, is a function of electrolyte coordinates...

        if not (self.electrode_type == "cathode" or self.electrode_type == "anode" ):
            print(' to create Electrode_Virtual object, must set electrode_type to either "cathode" or "anode" !')
            sys.exit(0)
        
        #**** these are only used for conductor objects attached to electrodes, and not flat Electrodes ... Should we introduce a better data structure ???
        self.Electrode_contact_atom = False # atom object of Electrode contact atom
        self.close_conductor_Electrode = True
        # this is threshold for determining whether nearest conductor is "close enough".
        self.close_conductor_threshold = 1.5 # nanometers, note this is from center of conductor that's why larger value...
        #*****

        # fill this with any extra electrode residues/chains that need exclusions
        self.electrode_extra_exclusions=[]

        # create a list and fill with atom_MM objects for every atom of electrode ...
        self.electrode_atoms=[]
        flag=0 # make sure we match electrodes
        # loop over residues, and find electrode       

        # need positions for creating atom objects ... note electrode positions won't change during the simulation
        # so we can store these initially once and for all ...
        state = MMsys.simmd.context.getState(getEnergy=False,getForces=False,getVelocities=False,getPositions=True)
        positions = state.getPositions()


        if chain_flag == True:
        # ***** initialize electrode by chain
            for chain in MMsys.simmd.topology.chains():
                if chain.index == self.electrode_identifier:
                    flag=1
                    for atom in chain.atoms():
                        element = atom.element
                        if element.symbol not in exclude_element:
                            # get initial charge from force field
                            (q_i, sig, eps) = MMsys.nbondedForce.getParticleParameters(atom.index)
                            # create atom_MM object
                            atom_object = atom_MM( element.symbol , q_i._value , atom.index , positions[atom.index][0]._value , positions[atom.index][1]._value , positions[atom.index][2]._value )
                            # add atom_object to electrode_atoms list
                            self.electrode_atoms.append( atom_object )

            # now collect any additional residue atoms to exclude
            if isinstance( electrode_identifier , tuple ) and ( len(electrode_identifier) > 1 ) :
                iterelectrode = iter(electrode_identifier)
                next(iterelectrode)
                for identifier in iterelectrode:
                    electrode_chain_atoms=[]
                    for chain in MMsys.simmd.topology.chains():
                        if chain.index == identifier:
                            for atom in chain.atoms():
                                electrode_chain_atoms.append( atom.index )
                    self.electrode_extra_exclusions.append( electrode_chain_atoms )

        else:
        # **** initialize electrode by residue
            for res in MMsys.simmd.topology.residues():
                if res.name == self.electrode_identifier:
                    flag=1
                    for atom in res._atoms:
                        element = atom.element
                        if element.symbol not in exclude_element:
                            # get initial charge from force field
                            (q_i, sig, eps) = MMsys.nbondedForce.getParticleParameters(atom.index)
                            # create atom_MM object
                            atom_object = atom_MM( element.symbol , q_i._value , atom.index, positions[atom.index][0]._value , positions[atom.index][1]._value , positions[atom.index][2]._value )
                            # add atom_object to electrode_atoms list
                            self.electrode_atoms.append( atom_object )


        # now make sure we've matched electrode residue name
        if flag == 0:
            print(' Couldnt find electrode residue...please check input electrode_identifier when constructing Electrode_Virtual object ! ')
            sys.exit(0)

        # total number of atoms on electrode...
        self.Natoms = len(self.electrode_atoms)


    #*******************************************
    # getter for returning the total charge on the electrode
    def get_total_charge( self ):
        sumQ = 0.0
        for atom in self.electrode_atoms:
            sumQ += atom.charge

        return sumQ


    #*********************************************
    # This finds the closest contact neighboring conductor/atom to the conductor
    # Initially, assume the closest contact is either the flat Cathode/Anode.
    # if this Electrode is too far, then loop over conductors in the conductor list
    #*********************************************
    def find_contact_neighbor_conductor( self, positions , r_center , MMsys ):

       # Find Cathode/Anode contact atom.  This is used to Enforce Constant Potential between electrode and this conductor.
       # Assume that MMsys.Cathode and MMsys.Anode objects have been created (which are of Electrode_Virtual type...)
       if self.electrode_type == "cathode":
           Electrode_contact = MMsys.Cathode
       else:
           Electrode_contact = MMsys.Anode

       # if this is a simulation without a Cathode/Anode, return False ...
       if Electrode_contact is None :
           return False

       min_dist = 10.0 # something large...
       # find contact atom based on closest distance to r_center.  The reason we calculate distance based on r_center is that for a nanotube/flat sheet, there is no uniquely defined close-contact atom pair... 
       # Assume we don't need PBC !!
       for atom in Electrode_contact.electrode_atoms:
           dr_atom = numpy.sqrt( ( r_center[0] - positions[atom.atom_index][0]._value )**2 + ( r_center[1] - positions[atom.atom_index][1]._value )**2 + ( r_center[2] - positions[atom.atom_index][2]._value )**2 )
           if dr_atom < min_dist:
               self.Electrode_contact_atom = atom
               min_dist = dr_atom

       # We are likely done here... only evaluate further code if conductor isn't in contact with the primary Electrode(s)...
       if  min_dist < self.close_conductor_threshold : 
           self.dr_center_contact = min_dist
           return False  # indicates that dr_vector isn't returned

       else:
       # if this loop evaluates, then conductor is in contact with another conductor off the electrode--search the list ...
           self.close_conductor_Electrode = False  # primary Electrode is not close contact ...
           print( "Searching Conductors for close-contact distance pair ... ")
           for Conductor in MMsys.Conductor_list :             
               # here we can't search based on r_center, because neither conductor is flat... need to do double loop over atom pairs (slow...)
               for atom1 in self.electrode_atoms:
                   for atom2 in Conductor.electrode_atoms:
                       dr_atom = numpy.sqrt( ( positions[atom1.atom_index][0]._value - positions[atom2.atom_index][0]._value )**2 + ( positions[atom1.atom_index][1]._value - positions[atom2.atom_index][1]._value )**2 +( positions[atom1.atom_index][2]._value - positions[atom2.atom_index][2]._value )**2 )
                       if dr_atom < min_dist:
                           self.Electrode_contact_atom = atom2
                           min_dist = dr_atom


               # see if this is the close conductor ...
               if  min_dist < self.close_conductor_threshold :
                   # compute distance from r_center to this closest atom
                   # if this is a Nanotube, we will project out component along axis, so return
                   # displacement vector and self.dr_center_contact will be recomputed within the child class after the projection ...
                   dr_vector = [0] * 3
                   for i in range(3):
                       dr_vector[i] = positions[self.Electrode_contact_atom.atom_index][i]._value - r_center[i]                  
                   self.dr_center_contact = numpy.sqrt( dr_vector[0]**2 + dr_vector[1]**2 + dr_vector[2]**2 )
                   return dr_vector

       # if we haven't exited yet, then we have failed to find close conductor ...
       print( "Failed to find close Conductor for threshold " , self.close_conductor_threshold )
       sys.exit()




#*********************************
# this is a child class of Conductor_Virtual parent class, that
# should be created for flat cathode/anode to set Voltage drop
#
# practically, the cathode and anode are divided into `real' and `virtual'
# layers, with the `real' layers for steric/VDWs interactions and the
# `virtual' layer for electrostatics.  The `virtual' layer is then utilized/modfied
#  within the Poisson solver to fix the voltage drop
#
#  input:
#        electrode_identifier : this could be either residuename of the electrode or chain index of a group of residues in the .pdb input file
#        electrode_type:  set to either "cathode" or "anode"
#        Voltage       : Voltage drop between electrodes, in Volts
#        MMsys           : QM/MM simulation object for the MM system
#        chain  (True/False) : if true, initialize electrodes by chain name rather than residue name
#        exclude_element     : this is tuple that can be filled with elements to exclude, e.g dummy Hydrogen
#*********************************
class Electrode_Virtual(Conductor_Virtual):

    # if electrode_identifier is empty tuple, this signals that we don't have electrodes in our
    # system, so don't create instance of class ...
    def __new__(cls, electrode_identifier, electrode_type, Voltage, MMsys, chain_flag, exclude_element):
        if electrode_identifier :
            return super().__new__(cls)

    def __init__(self, electrode_identifier, electrode_type, Voltage, MMsys, chain_flag, exclude_element):

        # constructor for Parent...
        super().__init__(electrode_identifier, electrode_type, Voltage, MMsys, chain_flag, exclude_element)

        # Area of electrode/area per atom:  Assume electrode spans x/y plane, and electrodes are separated along z ...
        boxVecs = MMsys.simmd.topology.getPeriodicBoxVectors()
        crossBox = numpy.cross(boxVecs[0], boxVecs[1])
        self.sheet_area = numpy.dot(crossBox, crossBox)**0.5 / nanometer**2 # divide by nanometer to get rid of units, this is area in nm^2 ...
        self.area_atom = self.sheet_area / self.Natoms # area per atom in nm^2

        self.Q_electrolyte = 0.0 # Electrolyte charge will be computed in compute_Electrode_charge_analytic

        # FIX:  Currently we assume conductors are placed only on cathode/left electrode of cell.
        # the below code needs to be generalized if conductors are placed on both cathode/anode,
        # because for one of these electrodes the normal vector will be along negative Z-axis ...
        for atom in self.electrode_atoms:
            atom.nx = 0.0 ; atom.ny = 0.0 ; atom.nz = 1.0
            # set area of atom
            atom.area_atom = self.area_atom
               



    #*************************
    # this method initializes the charge on an electrode,
    # given a Voltage drop, and assuming the electrolyte is vacuum
    #
    # Lgap            : vacuum gap between electrodes in nanometers
    # Lcell           : distance (nm) between electrodes in the electrochemical cell.
    # MMsys           : QM/MM simulation object for the MM system
    #*************************
    def initialize_Charge( self, Lgap, Lcell, MMsys):
        # positive sign for cathode, negative for anode
        sign=1.0
        if self.electrode_type == 'anode':
            sign=-1.0

        flag_small=False
        # add a small value to charge if Voltage is 0, so that charges aren't numerically zero and we can compute a field
        if abs(self.Voltage) < 0.01:
            print( "adding small value to initial charges in initialize_Charge routine for small Voltage input..." )
            flag_small=True

        # loop over atoms in electrode
        for atom in self.electrode_atoms:
            # note Voltage should be in kJ/mol, conversion is from kJ/mol*nm to atomic units
            q_i = sign / ( 4.0 * numpy.pi ) * self.area_atom * (self.Voltage / Lgap + self.Voltage / Lcell) * conversion_KjmolNm_Au
            if flag_small:
               # add small value to keep from being numerically zero ...
               q_i = q_i + sign * MMsys.small_threshold

            # now set charge in data structure and OpenMM force object
            atom.charge = q_i
            MMsys.nbondedForce.setParticleParameters(atom.atom_index, q_i, 1.0 , 0.0)

        # update parameters in OpenMM context
        MMsys.nbondedForce.updateParametersInContext(MMsys.simmd.context)



    #**************************
    # this method computes the total charge on the electrode
    # based on analytic formula from Green's reciprocity theorem.
    # naturally, this charge dependes on the electrolyte coordinates
    # and is computed on a per-time-step basis
    #
    #  input:
    #       MMsys :  MMsystem object
    #       positions : full position list returned directly from OpenMM
    #               includes all atoms, contains quantities ( value , unit)  in nanometers
    #**************************
    def compute_Electrode_charge_analytic( self, MMsys , positions, Conductor_list, z_opposite ):
        # positive sign for cathode, negative for anode
        sign=1.0
        if self.electrode_type == 'anode':
            sign=-1.0

        #********** Geometrical contribution:  Note use the Sheet Area rather than area_atom since we want total charge...
        self.Q_analytic = sign / ( 4.0 * numpy.pi ) * self.sheet_area * (self.Voltage / MMsys.Lgap + self.Voltage / MMsys.Lcell) * conversion_KjmolNm_Au

        self.Q_electrolyte = 0.0 # Recompute electrolyte charge, in case of redox process

        #********** Image charge contribution:  sum over electrolyte atoms and Drude oscillators ...
        for index in MMsys.electrolyte_atom_indices:
            (q_i, sig, eps) = MMsys.nbondedForce.getParticleParameters(index)
            z_atom = positions[index][2]._value # in nm
            z_distance = abs(z_atom - z_opposite)
            # add image charge contribution
            self.Q_analytic += (z_distance / MMsys.Lcell) * (- q_i._value)
            self.Q_electrolyte += q_i._value

        #*********  Conductors are effectively in electrolyte as far as flat electrodes are concerned, sum over these atoms ...
        if Conductor_list:
            for Conductor in Conductor_list:
                for atom in Conductor.electrode_atoms:
                    index = atom.atom_index
                    (q_i, sig, eps) = MMsys.nbondedForce.getParticleParameters(index)
                    z_atom = positions[index][2]._value # in nm
                    z_distance = abs(z_atom - z_opposite)
                    # add image charge contribution
                    self.Q_analytic += (z_distance / MMsys.Lcell) * (- q_i._value)

        

    #****************************
    # this routine scales all electrode atom charges such
    # that analytic normalization condition is satisfied.
    # this should be called after electrode charges are computed numerically,
    # and analytic total charge is evaluated...
    #***************************
    def Scale_charges_analytic( self, MMsys , print_flag = False ):
        # total charge on electrode as computed numerically
        Q_numeric = self.get_total_charge()

        # don't scale charges if total charge on Electrode is zero ...
        if abs(Q_numeric) < 0.01 :
            return

        if print_flag :
            print( "Q_numeric , Q_analytic charges on " , self.electrode_type , Q_numeric , self.Q_analytic )

        # scale factor, make sure not to divide by zero on rare occasions ...
        scale_factor = -1
        if abs(Q_numeric) > MMsys.small_threshold:
            scale_factor = self.Q_analytic / Q_numeric

        # now scale all electrode charges
        if scale_factor > 0.0:
            # loop over atoms in electrode
            for atom in self.electrode_atoms:               
                atom.charge = atom.charge * scale_factor
                MMsys.nbondedForce.setParticleParameters(atom.atom_index, atom.charge, 1.0 , 0.0)
 


    #**************** setter for electrode position ...
    def set_z_pos(self, z):
        self.z_pos = z





#*************************
# this is child class of Parent class Conductor_Virtual, used for buckyballs (conductors) that
# are attached/near the flat electrodes
#
# a difference with the flat electrodes is here we DON'T WANT exclusions between charges in 'virtual layer'.  This is because
# the image charges of the virtual layer contribute to the normal component of the field at the Buckyball surface
# however, we do still want exclusions in 'real' layer, and between 'real'/'virtual' layers...
#*************************
class Buckyball_Virtual(Conductor_Virtual):
    def __init__(self, electrode_identifier, electrode_type, Voltage, MMsys, chain_flag, exclude_element):

       # constructor for Parent...
       super().__init__(electrode_identifier, electrode_type, Voltage, MMsys, chain_flag, exclude_element)

       # must match by chain for this class, and must input chain indices for both 'virtual' and 'real' atoms...
       if chain_flag == False:
           print( 'must match by chain index for Buckyball_Virtual class!' )
           sys.exit()
       if not ( isinstance( electrode_identifier , tuple ) and ( len(electrode_identifier) > 1 ) ) :
           print( 'must input chain index for both virtual and real electrode atoms for BuckyBall class' )
           sys.exit()


       # In this child class, we need an additional electrode_atoms_real list for the 'real' atoms ...
       self.electrode_atoms_real=[]

       state = MMsys.simmd.context.getState(getEnergy=False,getForces=False,getVelocities=False,getPositions=True)
       positions = state.getPositions()

       # assume virtual first, then real, skip to real ...
       identifier = electrode_identifier[1]
       for chain in MMsys.simmd.topology.chains():
           if chain.index == identifier:
               for atom in chain.atoms():
                   element = atom.element
                   if element.symbol not in exclude_element:
                       # get initial charge from force field
                       (q_i, sig, eps) = MMsys.nbondedForce.getParticleParameters(atom.index)
                       # create atom_MM object
                       atom_object = atom_MM( element.symbol , q_i._value , atom.index , positions[atom.index][0]._value , positions[atom.index][1]._value , positions[atom.index][2]._value )
                       # add atom_object to electrode_atoms list
                       self.electrode_atoms_real.append( atom_object )


       # Find center of buckyball
       self.r_center = [ 0.0 , 0.0 , 0.0 ] # in nm
       for atom in self.electrode_atoms:
           self.r_center[0] += positions[atom.atom_index][0]._value 
           self.r_center[1] += positions[atom.atom_index][1]._value
           self.r_center[2] += positions[atom.atom_index][2]._value

       self.r_center[0] = self.r_center[0] / self.Natoms
       self.r_center[1] = self.r_center[1] / self.Natoms
       self.r_center[2] = self.r_center[2] / self.Natoms      
       #print( 'center' , self.r_center )

       # compute area per atom, get radius from first atom in Buckyball
       self.radius=0.0
       for atom in self.electrode_atoms:
           rx = positions[atom.atom_index][0]._value - self.r_center[0]
           ry = positions[atom.atom_index][1]._value - self.r_center[1]
           rz = positions[atom.atom_index][2]._value - self.r_center[2]
           self.radius = sqrt( rx**2 + ry**2 + rz**2 )
           break
       self.area_atom = 4.0 * numpy.pi * self.radius**2 / self.Natoms


       # calculate surface normal vector at each atom.  This will be used to project normal component of Electric field...
       for atom in self.electrode_atoms:
           nx = positions[atom.atom_index][0]._value - self.r_center[0]
           ny = positions[atom.atom_index][1]._value - self.r_center[1]
           nz = positions[atom.atom_index][2]._value - self.r_center[2]           
           norm = sqrt( nx**2 + ny**2 + nz**2)
           atom.nx = nx / norm ; atom.ny = ny / norm ; atom.nz = nz / norm
           # set area of atom
           atom.area_atom = self.area_atom

       # find close neighbor conductor/atom distance ...
       self.find_contact_neighbor_conductor( positions , self.r_center , MMsys )
      





#*************************
# this is child class of Parent class Conductor_Virtual, used for NanoTubes (conductors) that
# are attached/near the flat electrodes

# a difference with the flat electrodes is here we DON'T WANT exclusions between charges in 'virtual layer'.  This is because
# the image charges of the virtual layer contribute to the normal component of the field at the Nanotube surface
# however, we do still want exclusions in 'real' layer, and between 'real'/'virtual' layers...
#*************************
class Nanotube_Virtual(Conductor_Virtual):
    def __init__(self, electrode_identifier, electrode_type, Voltage, MMsys, chain_flag, exclude_element, axis ):

       # constructor for Parent...
       super().__init__(electrode_identifier, electrode_type, Voltage, MMsys, chain_flag, exclude_element)

       # must match by chain for this class, and must input chain indices for both 'virtual' and 'real' atoms...
       if chain_flag == False:
           print( 'must match by chain index for Nanotube_Virtual class!' )
           sys.exit()
       if not ( isinstance( electrode_identifier , tuple ) and ( len(electrode_identifier) > 1 ) ) :
           print( 'must input chain index for both virtual and real electrode atoms for Nanotube class' )
           sys.exit()

       # for now, nanotube axis must be input.  Eventually, write code to automate...
       self.axis = axis

       # In this child class, we need an additional electrode_atoms_real list for the 'real' atoms ...
       self.electrode_atoms_real=[]

       state = MMsys.simmd.context.getState(getEnergy=False,getForces=False,getVelocities=False,getPositions=True)
       positions = state.getPositions()

       # assume virtual first, then real, skip to real ...
       identifier = electrode_identifier[1]
       for chain in MMsys.simmd.topology.chains():
           if chain.index == identifier:
               for atom in chain.atoms():
                   element = atom.element
                   if element.symbol not in exclude_element:
                       # get initial charge from force field
                       (q_i, sig, eps) = MMsys.nbondedForce.getParticleParameters(atom.index)
                       # create atom_MM object
                       atom_object = atom_MM( element.symbol , q_i._value , atom.index , positions[atom.index][0]._value , positions[atom.index][1]._value , positions[atom.index][2]._value )
                       # add atom_object to electrode_atoms list
                       self.electrode_atoms_real.append( atom_object )


       # Find center of nanotube
       self.r_center = [ 0.0 , 0.0 , 0.0 ] # in nm
       for atom in self.electrode_atoms:
           self.r_center[0] += positions[atom.atom_index][0]._value 
           self.r_center[1] += positions[atom.atom_index][1]._value
           self.r_center[2] += positions[atom.atom_index][2]._value

       self.r_center[0] = self.r_center[0] / self.Natoms
       self.r_center[1] = self.r_center[1] / self.Natoms
       self.r_center[2] = self.r_center[2] / self.Natoms      
       #print( 'center' , self.r_center )

       # assume that length of nanotube is same as length of 'a' box vector
       # print a Warning that this is our assumption
       print( 'WARNING:  Assuming Nanotube length is equal to length of "a" box vector.  Need to modify code if this is not the case!')
       boxVecs = MMsys.simmd.topology.getPeriodicBoxVectors()
       self.length = boxVecs[0][0] / nanometer


       # compute radial vector at each atom, get radius.  Make sure radius is approximately the same for all atoms
       radius_threshold=0.001
       self.radius= -1.0
       for atom in self.electrode_atoms:
           dr = [0] * 3
           for i in range(3):
               dr[i] = positions[atom.atom_index][i]._value - self.r_center[i]
           # project out radial component
           radial_vector =  self.project_orthogonal_to_axis( numpy.asarray(dr) )
           radius = sqrt( radial_vector[0]**2 + radial_vector[1]**2 + radial_vector[2]**2 )
           # check that radius matches stored value for nanotube
           if self.radius < 0:
               self.radius = radius
           else:
               if abs( self.radius - radius ) > radius_threshold :
                   print( atom.atom_index , radius , self.radius )
                   print( 'different radius for atoms in nanotube, something is wrong!')
                   sys.exit()
           # store radial vector for atom
           atom.nx = radial_vector[0] / radius ; atom.ny = radial_vector[1] / radius ; atom.nz = radial_vector[2] / radius ;

       # compute area per atom
       self.area_atom = 2.0 * numpy.pi * self.radius * self.length / self.Natoms

       # set area of atom in each atom object
       for atom in self.electrode_atoms:
           atom.area_atom = self.area_atom

       # find close neighbor conductor/atom distance ...
       dr_vector = self.find_contact_neighbor_conductor( positions , self.r_center , MMsys )
       
       # if not returned as 'False', then need to do projection...
       if dr_vector :
           # 'find_contact_neighbor_conductor' sets self.dr_center_contact to distance between r_center and closest atom of conductor.  We need to project out component along nanotube axis to get radial component of this distance...
           radial_vector =  self.project_orthogonal_to_axis( numpy.asarray(dr_vector) )
           self.dr_center_contact = numpy.sqrt( radial_vector[0]**2 + radial_vector[1]**2 + radial_vector[2]**2 )

       #print( "Conductor " , self.close_conductor_Electrode  , self.Electrode_contact_atom.atom_index , self.dr_center_contact )



    # this method takes as input a vector, and projects out the component that is orthogonal to the nanotube axis
    def project_orthogonal_to_axis( self, vec_in ) :
        axis_local = numpy.asarray( self.axis )  
        vec_out = vec_in - axis_local * numpy.dot( vec_in , axis_local )
        return vec_out


