from simtk.openmm.app import *
from simtk.openmm import *
from simtk.unit import *
from sys import stdout
#******* Fixed voltage routines
from Fixed_Voltage_routines import *
#******* electrode exclusions routines
from electrode_exclusions import *
#******** contains parent class
from shared.MM_class_base import *
#******** exclusions for force field 
from shared.MM_exclusions_base import *
#*****************

import random
import numpy
import subprocess
import sys



#*************************** README  **************************************
# This is a child MM system class for Fixed-Voltage simulations
# that inherits from parent MM_base class in MM_class_base.py
#
#**************************************************************************
class MM_FixedVoltage(MM_base):
    # required input: 1) list of pdb files, 2) list of residue xml files, 3) list of force field xml files.
    def __init__( self , pdb_list , residue_xml_list , ff_xml_list , **kwargs  ):
        self.qmmm_ewald = False
        self.FV_solver = True
        self.analytic_charge_scaling = True # Turn on/off analytic charge scaling, default is on

        # constructor for Parent...
        super().__init__( pdb_list , residue_xml_list , ff_xml_list , **kwargs )


        # inputs from **kwargs
        if 'out_dir' in kwargs :
            self.out_dir = kwargs['out_dir']
        if 'platform' in kwargs :
            self.platformname = kwargs['platform']
        if 'set_periodic_residue' in kwargs :
            self.periodic_residue = eval(kwargs['set_periodic_residue'])
        if 'collect_charge_data' in kwargs :
            self.collect_charge_data = eval(kwargs['collect_charge_data'])
        if 'return_system_init_pdb' in kwargs :
            self.return_system_init_pdb = eval(kwargs['return_system_init_pdb'])
        if 'return_system_final_pdb' in kwargs :
            self.return_system_final_pdb = eval(kwargs['return_system_final_pdb'])
        if 'write_frequency' in kwargs :
            self.write_frequency = int(kwargs['write_frequency'])
        if 'simulation_length' in kwargs :
            self.simulation_length = float(kwargs['simulation_length'])
            self.loop_range = int(self.simulation_length * 1000000 / self.write_frequency)
        if 'analytic_charge_scaling' in kwargs:
            self.analytic_charge_scaling=kwargs['analytic_charge_scaling'] 
        if 'FV_solver' in kwargs:
            self.FV_solver=kwargs['FV_solver']

    #***********************************************
    # this initializes the Electrode objects for Constant-Voltage simulation
    # if input chain = True, we initialize electrodes by chain rather than residue name
    def initialize_electrodes( self, Voltage, cathode_identifier , anode_identifier , chain=False, exclude_element=(), **kwargs ):
        # first create electrode objects
        self.Cathode = Electrode_Virtual( cathode_identifier , "cathode" , Voltage , self , chain , exclude_element )
        self.Anode   = Electrode_Virtual( anode_identifier   , "anode"   , Voltage , self , chain , exclude_element )

        # add any Conductors on electrodes, currently, these could be "Buckyballs" or "Nanotubes" ...
        # FIX! Assume all Conductors are on Cathode, see below code.  Need to generalize this!
        self.Conductor_list = []
        if 'BuckyBalls' in kwargs :
            list_temp = kwargs['BuckyBalls'] # this is a list of identifiers (residue or chain) of BuckyBalls
            for identifier in list_temp:
                Buckyball = Buckyball_Virtual( identifier , "cathode" , Voltage , self, chain , exclude_element ) 
                self.Conductor_list.append( Buckyball )

        # for now, need to input cylindrical axis of nanotube.  Eventually, we should write code to
        # determine this automatically....
        if 'NanoTubes' in kwargs :
            list_temp = kwargs['NanoTubes'] # this is a list of identifiers (residue or chain) of NanoTubes
            list_temp2 = kwargs['nanotube_axis'] # corresponding list of nanotube axis
            for identifier , nanotube_axis in zip(list_temp, list_temp2):
                # make sure we have nanotube axis
                if nanotube_axis:
                    Nanotube = Nanotube_Virtual( identifier , "cathode" , Voltage , self, chain , exclude_element, nanotube_axis )
                    self.Conductor_list.append( Nanotube )
                else:
                    print('must input nanotube_axis for all nanotubes!')
                    sys.exit()


        if self.Cathode is not None and self.Anode is not None :
            state = self.simmd.context.getState(getEnergy=False,getForces=False,getVelocities=False,getPositions=True) 
            positions = state.getPositions()
            boxVecs = self.simmd.topology.getPeriodicBoxVectors()
            # set electrochemical cell parameters...
            self.set_electrochemical_cell_parameters( positions, boxVecs )

            # now initialize charge on the electrodes based on applied Voltage ...
            if self.FV_solver == True:
                self.Cathode.initialize_Charge( self.Lgap, self.Lcell, self )
                self.Anode.initialize_Charge( self.Lgap, self.Lcell, self )
        else :
            print( "No Electrodes in the simulation !" )



    #******************************************
    # this resets the geometry parameters of the electrochemical cell,
    # specifically 'Lcell' and 'Lgap' which are used in the Poisson solver
    # we make this a standalone method rather than putting it in the 'initialize_electrode'
    # method because we need to call externally if we are doing MC moves on electrodes ...
    #******************************************
    def set_electrochemical_cell_parameters( self , positions , boxVecs ):
       
        # need to figure out Lcell, Lgap .  Get Lcell from first atoms in each electrode, assume electrodes are separated along z-axis
        
        # use z coord of 1st atom in each electrode to compute Lcell
        atom_index_cathode = self.Cathode.electrode_atoms[0].atom_index
        atom_index_anode   = self.Anode.electrode_atoms[0].atom_index

        # set these z coordinates, need these for analytic charge evaluation ...
        z_cath = positions[atom_index_cathode][2] / nanometer
        self.Cathode.set_z_pos(z_cath)
        z_anod = positions[atom_index_anode][2] / nanometer
        self.Anode.set_z_pos(z_anod)
        self.Lcell = abs(z_cath - z_anod)

        # now vacuum gap, = full z length of box minus Lcell
        self.Lgap = boxVecs[2][2] / nanometer - self.Lcell  # in nanometers ...



    #***********************************************
    # this initializes a list of all electrolyte atoms to use for analytic correction of Poisson solver...
    #  
    # rather than passing/hard-coding in a list of all electrolyte residue names, lets do something simple that should be pretty robust
    # if a residue has > Natom_cutoff number of atoms, its an electrode residue
    # if a residue has < Natom_cutoff number of atoms, its an electrolyte residue
    #  a reasonable choice of Natom_cutoff=100, which i don't think will ever lead to a bug...
    def initialize_electrolyte( self , Natom_cutoff=100):
        # make a set of electrolyte residue names, so that we don't have to keep counting atom numbers...
        electrolyte_names=set()
        # initialize list of electrolyte residue objects /atom indices
        self.electrolyte_residues=[]
        self.electrolyte_atom_indices=[]
        for res in self.simmd.topology.residues():
            if res.name in electrolyte_names:
                # add to electrolyte list
                self.electrolyte_residues.append(res)
                for atom in res._atoms:
                    self.electrolyte_atom_indices.append(atom.index)
            else:
                # this is a new residue name, see if its an electrolyte residue
                natoms = 0
                for a in res._atoms:
                    natoms+=1    
                if natoms < Natom_cutoff:
                    # this is an electrolyte residue
                    self.electrolyte_residues.append(res)
                    electrolyte_names.add( res.name )
                    # add to electrolyte list
                    for atom in res._atoms:
                        self.electrolyte_atom_indices.append(atom.index)


    #************************************************
    # This is just a wrapper that calls different versions
    # of the Poisson solver depending on whether Cathode/Anode are present ...
    #************************************************
    def Poisson_solver_fixed_voltage(self, Niterations=3, compute_intermediate_forces = True , net_charge=0.0, print_flag = False ):

        if self.FV_solver == False:
            return
        # if QM/MM , make sure we turn off vext_grid calculation to save time with forces... turn back on after converged
        if self.qmmm_ewald :
            platform=self.simmd.context.getPlatform()
            platform.setPropertyValue( self.simmd.context , 'ReferenceVextGrid' , "false" )

        # if Electrodes are present, call most general version of Poisson solver ...
        if self.Cathode is not None and self.Anode is not None :
            self.Poisson_solver_fixed_voltage_Electrodes( Niterations , compute_intermediate_forces = compute_intermediate_forces , print_flag = print_flag )
        else:
            # no Electrodes, call version for Conductors only...
            self.Poisson_solver_fixed_voltage_Conductors_only( Niterations , net_charge )

        # if QM/MM , turn vext back on ...
        if self.qmmm_ewald :
            platform.setPropertyValue( self.simmd.context , 'ReferenceVextGrid' , "true" )



    #************************************************
    # This is the Fixed-Voltage Poisson Solver to optimize charges
    # on the electrode subject to applied voltage ...
    #************************************************
    def Poisson_solver_fixed_voltage_Electrodes(self, Niterations=3 , compute_intermediate_forces = True , print_flag = False ):
      
        #********* Analytic evaluation of total charge on electrodes based on electrolyte coordinates
        state = self.simmd.context.getState(getEnergy=False,getForces=False,getVelocities=False,getPositions=True)
        positions = state.getPositions()
        # compute charge for both anode/cathode
        self.Cathode.compute_Electrode_charge_analytic( self , positions , self.Conductor_list, z_opposite = self.Anode.z_pos ) # this is correct, input z_pos(Anode) for cathode charge, and vice-versa...
        self.Anode.compute_Electrode_charge_analytic( self , positions , self.Conductor_list, z_opposite = self.Cathode.z_pos )

        #print(" initial charge on Cathode/Anode " , self.Cathode.get_total_charge() , self.Anode.get_total_charge() )

        # make sure these are equal and opposite, but only if no additional conductors ...
        if abs(self.Cathode.Q_analytic + self.Anode.Q_analytic + self.Cathode.Q_electrolyte) > self.small_threshold and not self.Conductor_list :
           print( "Analytic charges on Cathode and Anode don't match...something is wrong!" )
           sys.exit(0)

        #*********  Self-consistently solve for electrode charges that obey Fixed-Voltage boundary condition ...
        for i_iter in range(Niterations):

            # need Efield on all electrode atoms, get this from forces on virtual electrode sheets ...
            state = self.simmd.context.getState(getEnergy=True,getForces=True,getVelocities=False,getPositions=True)
            forces = state.getForces()

            # print to see convergence, do we want this?
            #for j in range(self.system.getNumForces()):
            #    f = self.system.getForce(j)
            #    print(type(f), str(self.simmd.context.getState(getEnergy=True, groups=2**j).getPotentialEnergy()))

            # first solve charges on cathode ...
            # note that Cathode.Voltage and Anode.Voltage are Total Voltage drop, i realize this is slightly misleading...
            for atom in self.Cathode.electrode_atoms:
                index = atom.atom_index
                q_i_old = atom.charge
                # Ez , don't divide by zero!
                Ez_external = ( forces[index][2]._value / q_i_old ) if abs(q_i_old) > (0.9*self.small_threshold) else 0.
                # new charge that satisfies fixed Voltage boundary condition...
                # when we switch to atomic units on the right, sigma/2*epsilon0 becomes 4*pi*sigma/2 , since 4*pi*epsilon0=1 in a.u.
                q_i = 2.0 / ( 4.0 * numpy.pi ) * self.Cathode.area_atom * (self.Cathode.Voltage / self.Lgap + Ez_external) * conversion_KjmolNm_Au

                atom.charge = self.update_charge_iteration( q_i_old , q_i )
                # don't allow charges to get below small_threshold, otherwise can't compute Efield next iteration, and will get stuck at zero forever ...
                if abs(atom.charge) < self.small_threshold:
                    atom.charge = self.small_threshold  # Cathode, make positive
                self.nbondedForce.setParticleParameters(index, atom.charge, 1.0 , 0.0)            

            # now charges on anode ...
            for atom in self.Anode.electrode_atoms:
                index = atom.atom_index
                q_i_old = atom.charge
                # Ez , don't divide by zero!
                Ez_external = ( forces[index][2]._value / q_i_old ) if abs(q_i_old) > (0.9*self.small_threshold) else 0.
                # new charge that satisfies fixed Voltage boundary condition...
                # when we switch to atomic units on the right, sigma/2*epsilon0 becomes 4*pi*sigma/2 , since 4*pi*epsilon0=1 in a.u.
                q_i = -2.0 / ( 4.0 * numpy.pi ) * self.Anode.area_atom * (self.Anode.Voltage / self.Lgap + Ez_external) * conversion_KjmolNm_Au

                atom.charge = self.update_charge_iteration( q_i_old , q_i )
                # don't allow charges to get below small_threshold, otherwise can't compute Efield next iteration, and will get stuck at zero forever ...
                if abs(atom.charge) < self.small_threshold:
                    atom.charge = -1.0 * self.small_threshold  # Anode, make positive
                self.nbondedForce.setParticleParameters(index, atom.charge, 1.0 , 0.0)

            # now charges on any conductors that are near electrodes...
            if self.Conductor_list:
                for Conductor in self.Conductor_list:
                    self.Numerical_charge_Conductor( Conductor , forces , fix_Voltage = True , compute_intermediate_forces = compute_intermediate_forces )
                    #print( i_iter , 'Conductor charge is' , Conductor.get_total_charge() )

                self.nbondedForce.updateParametersInContext(self.simmd.context)
                # because conductors within cell are "part of electrolyte" as far as analytic charge formula is concerned, need to recomput analytic charges here...
                self.Cathode.compute_Electrode_charge_analytic( self , positions , self.Conductor_list, z_opposite = self.Anode.z_pos ) # this is correct, input z_pos(Anode) for cathode charge, and vice-versa...
                self.Anode.compute_Electrode_charge_analytic( self , positions , self.Conductor_list, z_opposite = self.Cathode.z_pos )

            # Now scale charges to exact Analytic normalization....
            if self.analytic_charge_scaling == True:
                self.Scale_charges_analytic_general()

            # update charges in context ...
            self.nbondedForce.updateParametersInContext(self.simmd.context)

            if print_flag == True :
                self.print_charges_on_Conductors( i_iter , self.Cathode , self.Anode , self.Conductor_list )

        # this call is just for printing converged charges ...
        #self.Scale_charges_analytic_general( print_flag = True )


    #***********************************************
    # prints total charges on all conductors:  Should be called after a Poisson solver iteration
    #***********************************************
    def print_charges_on_Conductors( self , i_iter , Cathode , Anode , Conductor_list ): 
        print( 'Total Charges on conductors after Poisson Solver iteratation ' , i_iter )
        print( 'Cathode total charge ' , Cathode.get_total_charge() )
        print( 'Anode total charge ' , Anode.get_total_charge() )
        print( 'total charge on additional conductors ' )
        for Conductor in Conductor_list:
            print( 'charge ' , Conductor.get_total_charge() )



    #************************************************
    # This is the Poisson Solver to optimize image charges
    # on Conductors only, without applied Voltage to Cathode/Anode ...
    #
    # Because the potential isn't specified, the net charge on the conductor must
    # be specified to ensure a unique solution, since any addition of uniform charge
    # over surface doesn't alter the conducting boundary condition
    #************************************************
    def Poisson_solver_fixed_voltage_Conductors_only(self, Niterations=3, net_charge = 0.0 ):

       #*********  Self-consistently solve for image charges that obey conducting boundary condition ...
        for i_iter in range(Niterations):
            # need Efield on all electrode atoms, get this from forces on virtual conductors ...
            state = self.simmd.context.getState(getEnergy=True,getForces=True,getVelocities=False,getPositions=True)
            forces = state.getForces()
            if self.Conductor_list:
                for Conductor in self.Conductor_list:
                    self.Numerical_charge_Conductor( Conductor , forces , fix_Voltage = False )

                    # fix total charge on conductor
                    print( 'Conductor charge was' , Conductor.get_total_charge() , 'before fixing to ', net_charge )
                    Q_numeric = Conductor.get_total_charge()
                    # per atom charge
                    dq_atom = -( Q_numeric - net_charge ) / Conductor.Natoms

                    # make sure to not numerically zero charges so that field can't be computed ...
                    if abs( dq_atom ) > 2.0 * self.small_threshold :
                        # now ADD this excess charge to Conductor
                        for atom in Conductor.electrode_atoms:
                            index = atom.atom_index
                            (q_i_quantity, sig, eps) = self.nbondedForce.getParticleParameters(index)
                            q_i = q_i_quantity._value +  dq_atom
                            atom.charge = q_i
                            self.nbondedForce.setParticleParameters(index, q_i, sig , eps)

                    self.nbondedForce.updateParametersInContext(self.simmd.context)




    #***************************************
    # this solves conducting boundary condition for Conductor on electrode, given electric field on each atom and applied Voltage
    #
    # so far this works for either Buckyballs (spheres) / Nanotubes (cylindars)
    #
    # 'forces' contain all the forces in the system, used for calculating electric field on virtual atoms
    # these should be updated with current charge distribution on conductors
    #
    # 'Conductor' is a conductor object
    #***************************************
    def Numerical_charge_Conductor( self, Conductor, forces , fix_Voltage = True , compute_intermediate_forces = True ):
       
        #****************************************************************************
        # Step 1:  Image charges on Conductor.  Project Efield to surface normal vector
        #          solve for the image charge on the Conductor such that the normal field
        #          component is zero inside Conductor
        #******************************************************************************

        # Images charges are set on 'Virtual' atoms of Conductor ...
        for atom in Conductor.electrode_atoms:
            index = atom.atom_index
            (q_i_old_quantity, sig, eps) = self.nbondedForce.getParticleParameters(index)
            q_i_old = q_i_old_quantity._value # quantity = value * units ...

            E_external=[]
            # normal component of Field...
            if abs(q_i_old) > (0.9*self.small_threshold): 
                E_external.append( forces[index][0]._value / q_i_old ) # Ex
                E_external.append( forces[index][1]._value / q_i_old ) # Ey
                E_external.append( forces[index][2]._value / q_i_old ) # Ez

                # project out normal
                En_external = numpy.dot( numpy.array( E_external ) , numpy.array( [ atom.nx , atom.ny , atom.nz ] ) )
                # now solve for surface charge, requiring Enormal be zero inside conductor...
                q_i = 2.0 / ( 4.0 * numpy.pi ) * Conductor.area_atom * En_external * conversion_KjmolNm_Au

                atom.charge = self.update_charge_iteration( q_i_old , q_i )

            # don't allow charges to stay below small_threshold, otherwise can't compute Efield next iteration, and will get stuck at zero forever ...
            else: 
                atom.charge = self.small_threshold  # Cathode, make positive

            self.nbondedForce.setParticleParameters(index, atom.charge, sig , eps)

        # if we're not fixing the Voltage of the conductor(s) using electrodes, then we are done here ...
        if not fix_Voltage:
            return

        # if we are computing intermediate forces ...
        # this is expensive, but will accelerate convergence
        if compute_intermediate_forces:
            self.nbondedForce.updateParametersInContext(self.simmd.context)
            state = self.simmd.context.getState(getEnergy=True,getForces=True,getVelocities=False,getPositions=True)
            forces = state.getForces()



        #****************************************************************************
        # Step 2:  Charge transfer to Conductor.  Distribute uniformly on atoms.
        #          this is determined from electric field to the right (for cathode) of closest electrode atom being zero
        #          so that the Conductor is at the same Potential as the electrode...
        #******************************************************************************

        # index of close contact atom ...
        conductor_atom = Conductor.Electrode_contact_atom
        conductor_atom_index = conductor_atom.atom_index 
        (q_i_quantity, sig, eps) = self.nbondedForce.getParticleParameters(conductor_atom_index)        
        q_i = q_i_quantity._value # quantity = value * units ...

        # get field normal to surface, most likely this will be in Z-direction, but use general code....
        E_external=[]
        # normal component of Field...
        if abs(q_i) > (0.9*self.small_threshold):
            E_external.append( forces[conductor_atom_index][0]._value / q_i ) # Ex
            E_external.append( forces[conductor_atom_index][1]._value / q_i ) # Ey
            E_external.append( forces[conductor_atom_index][2]._value / q_i ) # Ez

            # project out normal
            En_external = numpy.dot( numpy.array( E_external ) , numpy.array( [ conductor_atom.nx , conductor_atom.ny , conductor_atom.nz ] ) )
        else:
            En_external = 0.0


        # the boundary condition depends on whether the contact is with the Electrode with applied Voltage, or another conductor...
        if Conductor.close_conductor_Electrode :
            # Electrostatics must satisfy on L/R of electrode atom:
            # Left:  -dV/L = - sigma/2eps + Eext + dE_conductor
            # Right:    0  =   sigma/2eps + Eext + dE_conductor
            # therefore, sigma/eps = dV/L
            # and dE_conductor = -( Eext + dV/2L )
            dE_conductor = - ( En_external + self.Cathode.Voltage / self.Lgap / 2.0 ) * conversion_KjmolNm_Au
        else :
            # this is another conductor, no explicit delta_V / L ...
            # there can be no surface charge at this element, because E=0 inside and outside the surface for both boundary conditions
            dE_conductor = - En_external * conversion_KjmolNm_Au


        # Charge depends on geometry of conductor, in general, Q = E * A / 4 *pi where A is area of volume in Gauss' Law integration ...
        if type(Conductor).__name__ == "Buckyball_Virtual" :
            # if buckyball is postive z displacement from cathode, then the field points in negative z for positive charge...
            sign=-1.0
            dQ_conductor =  sign * dE_conductor * Conductor.dr_center_contact**2 

        elif type(Conductor).__name__ == "Nanotube_Virtual" :
            sign=-1.0
            dQ_conductor =  sign * dE_conductor * Conductor.dr_center_contact * Conductor.length / 2.0

        else :
            print( "can't recognize Conductor type in Numerical_charge_Conductor method!" )
            sys.exit()


        #print ( 'dQ_conductor' , dQ_conductor )

        # per atom charge
        dq_atom = dQ_conductor / Conductor.Natoms

        # now ADD this excess charge to Conductor
        for atom in Conductor.electrode_atoms:
            index = atom.atom_index
            (q_i_quantity, sig, eps) = self.nbondedForce.getParticleParameters(index)
            q_i_old = q_i_quantity._value
            q_i = q_i_quantity._value +  dq_atom

            atom.charge = self.update_charge_iteration( q_i_old , q_i )
            self.nbondedForce.setParticleParameters(index, atom.charge, sig , eps)



    #**************************************
    # trivial scaling algorithm for updating charges during iterative solver
    #**************************************
    def update_charge_iteration( self , q_old , q_new , lambda_scale = 0.4 ):
        q_update = (1.0 - lambda_scale) * q_old + lambda_scale * q_new
        return q_update



    #***************************************
    # this scales charges to analytic normalization, 
    # which depends on the geometry of the electrodes.
    #
    # for two flat electrodes, the analytic normalization is
    # done independently and this method merely serves as a wrapper
    #
    # however if we have additional conductors (Buckyballs, nanotubes) at
    # the electrodes, then becomes more complicated ...
    #******************************************
    def Scale_charges_analytic_general(self , print_flag = False ):

        #NOTE:  NEED to Generalize.  Currently, assume Conductors are on Cathode if they are present

        if self.Conductor_list:
           # assume anode is scaled normally...
           self.Anode.Scale_charges_analytic( self , print_flag )
           # get analytic correction from anode
           Q_analytic = -1.0 * self.Anode.Q_analytic
            
           # add up total charge on Cathode and all conductors
           Q_numeric_total = self.Cathode.get_total_charge() 

           # loop over Conductors ...
           for Conductor in self.Conductor_list:
               # add charges from 'virtual' atoms
               Q_numeric_total += Conductor.get_total_charge()
           
           if print_flag :
               print( "Q_numeric , Q_analytic charges on Cathode and extra conductors" , Q_numeric_total , Q_analytic )

           # scale factor, make sure not to divide by zero on rare occasions ...
           scale_factor = -1
           if abs(Q_numeric_total) > self.small_threshold:
               scale_factor = Q_analytic / Q_numeric_total

           # now scale all charges on Cathode and conductors
           if scale_factor > 0.0:
               # loop over atoms in Cathode
               for atom in self.Cathode.electrode_atoms:
                   atom.charge = atom.charge * scale_factor
                   self.nbondedForce.setParticleParameters(atom.atom_index, atom.charge, 1.0 , 0.0)
               # loop over Conductors
               for Conductor in self.Conductor_list:
                   for atom in Conductor.electrode_atoms:
                       atom.charge = atom.charge * scale_factor
                       self.nbondedForce.setParticleParameters(atom.atom_index, atom.charge, 1.0 , 0.0)

        else:
            # no extra conductors, scale each electrode to individual Analytic normalization....
            self.Cathode.Scale_charges_analytic( self , print_flag )
            self.Anode.Scale_charges_analytic( self , print_flag )




    #***************************************
    # this generates exclusions for intra-electrode interactions,
    # 
    # if flag_SAPT_FF_exclusions=True, then will also set exclusions for SAPT-FF force field...
    #***************************************
    def generate_exclusions(self, water_name = 'HOH', flag_hybrid_water_model = False ,  flag_SAPT_FF_exclusions = True ):

        # if Cathode/Anode are present, setup exclusions ...
        if self.Cathode is not None and self.Anode is not None :

            # first electrodes, make temporary list of electrode atom indices to pass to exclusions subroutine
            cathode_list=[]
            for atom in self.Cathode.electrode_atoms:
                cathode_list.append( atom.atom_index )
            anode_list=[]
            for atom in self.Anode.electrode_atoms:
                anode_list.append( atom.atom_index )

            # first electrostatic exclusions between all atoms in principle electrode sheet
            exclusion_Electrode_NonbondedForce(self.simmd , self.system, cathode_list, cathode_list, self.customNonbondedForce , self.nbondedForce )
            exclusion_Electrode_NonbondedForce(self.simmd , self.system, anode_list, anode_list, self.customNonbondedForce , self.nbondedForce)

            # now see if we need to add exclusions between any other chains in Cathode
            if len( self.Cathode.electrode_extra_exclusions ) > 0 :
                for chain1 in range(len( self.Cathode.electrode_extra_exclusions )):
                    # first exclude between principle electrode sheet and other chains
                    exclusion_Electrode_NonbondedForce(self.simmd , self.system, cathode_list, self.Cathode.electrode_extra_exclusions[chain1], self.customNonbondedForce , self.nbondedForce )
                    # now between extra chains
                    for chain2 in range(chain1 , len( self.Cathode.electrode_extra_exclusions )):
                        exclusion_Electrode_NonbondedForce(self.simmd , self.system, self.Cathode.electrode_extra_exclusions[chain1], self.Cathode.electrode_extra_exclusions[chain2], self.customNonbondedForce , self.nbondedForce )
            # now see if we need to add exclusions between any other chains in Anode
            if len( self.Anode.electrode_extra_exclusions ) > 0 :
                for chain1 in range(len( self.Anode.electrode_extra_exclusions )):
                    # first exclude between principle electrode sheet and other chains
                    exclusion_Electrode_NonbondedForce(self.simmd , self.system, anode_list, self.Anode.electrode_extra_exclusions[chain1], self.customNonbondedForce , self.nbondedForce )
                    # now between extra chains
                    for chain2 in range(chain1 , len( self.Anode.electrode_extra_exclusions )):
                        exclusion_Electrode_NonbondedForce(self.simmd , self.system, self.Anode.electrode_extra_exclusions[chain1], self.Anode.electrode_extra_exclusions[chain2], self.customNonbondedForce , self.nbondedForce )
        
        
        
        # now exclusions for Conductors on Electrodes ... DON'T exclude virtual/virtual, exclude real/real and virtual/real
        for Conductor in self.Conductor_list:
            # temporary lists
            Conductor_real_list=[]; Conductor_virtual_list=[]
            for atom in Conductor.electrode_atoms:
                Conductor_virtual_list.append( atom.atom_index )
            for atom in Conductor.electrode_atoms_real:
                Conductor_real_list.append( atom.atom_index )

            exclusion_Electrode_NonbondedForce(self.simmd , self.system, Conductor_real_list, Conductor_real_list, self.customNonbondedForce , self.nbondedForce ) 
            exclusion_Electrode_NonbondedForce(self.simmd , self.system, Conductor_real_list, Conductor_virtual_list, self.customNonbondedForce , self.nbondedForce )


        # if special exclusion for SAPT-FF force field ...
        if flag_SAPT_FF_exclusions:
            generate_SAPT_FF_exclusions( self )

        # if using a hybrid water model, need to create interaction groups for customnonbonded force....
        if flag_hybrid_water_model:
            generate_exclusions_water(self.simmd, self.customNonbondedForce, water_name )

        # having both is redundant, as SAPT-FF already creates interaction groups for water/other
        if flag_SAPT_FF_exclusions and flag_hybrid_water_model:
            print( "redundant settiong of flag_SAPT_FF_exclusions and flag_hybrid_water_model")
            sys.exit()


        # now reinitialize to make sure changes are stored in context
        state = self.simmd.context.getState(getEnergy=False,getForces=False,getVelocities=False,getPositions=True)
        positions = state.getPositions()
        self.simmd.context.reinitialize()
        self.simmd.context.setPositions(positions)



    

    #*******************************************
    #   this method performs MC/MD steps for moving electrode sheets to
    #   equilibrate the density of the electrolyte
    #
    #         !!! assumes object self.MC exists !!!!
    #
    #   before calling this method, make sure to initialize MC parameters by
    #   construcing self.MC object of class MC_parameters(object):
    #*******************************************
    def MC_Barostat_step( self ):

        # inner functions ...
        def metropolis(pecomp):
            if pecomp < 0.0 * self.MC.RT :
                return True
            elif (random.uniform(0.0,1.0) < numpy.exp(-pecomp/self.MC.RT)):
                return True

        def intra_molecular_vectors( residue_object , pos_ref, positions_array ):
            intra_vec = []
            # loop over atoms in residue
            for atom in residue_object._atoms:
                pos_res_i = positions_array[atom.index]
                vec_i = pos_res_i - pos_ref
                intra_vec.append(numpy.asarray(vec_i))
            return numpy.asarray(intra_vec)


        self.MC.ntrials += 1

        # ************** normal MD steps ****************
        self.simmd.step(self.MC.barofreq)

        # get final positions
        state = self.simmd.context.getState(getEnergy=True, getPositions=True)
        positions = state.getPositions().value_in_unit(nanometer)

        # energy before move
        oldE = state.getPotentialEnergy()
        # store positions
        oldpos = numpy.asarray(positions)
        newpos = numpy.asarray(positions)     

        # now generate trial move
        # randomly choose move distance ... (-1 , 1) Angstrom for now...
        deltalen = self.MC.shiftscale*(random.uniform(0, 1) * 2 - 1)

        # need a reference point for scaling relative positions.  Choose the stationary electrode for this.
        reference_atom_index = -1

        # Currently, can only move Anode, because we might have other conductors on Cathode (Buckyballs, nanotubes)
        # could easily generalize this to move other conductors as well, but haven't yet...
        if self.MC.electrode_move == "Anode" :
            reference_atom_index = self.Cathode.electrode_atoms[0].atom_index # since we are moving Anode, choose stationary cathode as reference...
            # move Anode
            for atom in self.Anode.electrode_atoms:
                newpos[atom.atom_index,2] += deltalen
            # now see if we need to move any other chains in Anode
            if len( self.Anode.electrode_extra_exclusions ) > 0 :
                for extra_anode_sheet in self.Anode.electrode_extra_exclusions :
                    for index in extra_anode_sheet:
                        newpos[index,2] += deltalen
        else:
            print( "Currently, can only move Anode in MC_Barostat_step...need to generalize for other Conductors ...")
            sys.exit()

        Lcell_old = self.Lcell
        Lcell_new = Lcell_old + deltalen


        N_electrolyte_mol=0
        # now loop over electrolyte molecules and move their COM
        for res in self.electrolyte_residues:
            N_electrolyte_mol += 1
            # use first atom in residue as reference...
            for atom in res._atoms:
                pos_ref = newpos[atom.index]
                break
            # get relative coordinates          
            intra_vec = intra_molecular_vectors( res , pos_ref, newpos )

            # convert to coordinate system relative to stationary electrode ...
            pos_ref[2] = pos_ref[2] - newpos[reference_atom_index,2]
            # now scale this reference position by ratio of change in electrochemical cell volume
            pos_ref[2] = pos_ref[2] * Lcell_new / Lcell_old
            # now back to global coordinate system ...
            pos_ref[2] = pos_ref[2] + newpos[reference_atom_index,2]
            # now update positions of all atoms in this molecule
            index_in_molecule = 0
            for atom in res._atoms:
                newpos[atom.index] = pos_ref + intra_vec[index_in_molecule]
                index_in_molecule+=1


        #  Energy of trial move 
        self.simmd.context.setPositions(newpos)
        statenew = self.simmd.context.getState(getEnergy=True,getPositions=True)
        newE = statenew.getPotentialEnergy()
      
        w = newE-oldE + self.MC.pressure*(deltalen * nanometer) - N_electrolyte_mol * self.MC.RT * numpy.log(Lcell_new/Lcell_old)
        if metropolis(w):
            self.MC.naccept += 1
            # move is accepted, update electrochemical cell parameters...
            boxVecs = self.simmd.topology.getPeriodicBoxVectors()
            positions = statenew.getPositions()
            self.set_electrochemical_cell_parameters( positions, boxVecs )
        else:
            # move is rejected, revert to old positions ...
            self.simmd.context.setPositions(oldpos)

        if self.MC.ntrials > 50 :
            #print(" After 50 more MC steps ...")
            #print("dE, exp(-dE/RT) ", w, numpy.exp(-w/ self.MC.RT))
            #print("Accept ratio for last 50 MC moves", self.MC.naccept / self.MC.ntrials)
            if (self.MC.naccept < 0.25*self.MC.ntrials) :
                self.MC.shiftscale /= 1.1
            elif self.MC.naccept > 0.75*self.MC.ntrials :
                self.MC.shiftscale *= 1.1
            # reset ...
            self.MC.ntrials = 0
            self.MC.naccept = 0



    # this method sets an umbrella potential constraining the centroid of input molecule "mol1".
    # this can be done multiple ways, as is controlled by the specific **kwarg passed to the method.
    #      **kwargs:  mol2, atomtype, r0centrold :  constrain to distance from atom "atom" on mol2
    #      **kwargs:  z_global : constrain to absolute z position
    def setumbrella(self, mol1, k , **kwargs ):

        #create mol1 group for centroid
        g1=[]
        for res in self.simmd.topology.residues():
            if res.name == mol1:
                for i in range(len(res._atoms)):
                    g1.append(res._atoms[i].index)
                break

        # option 1: input mol2, atomtype, r0centrold :  constrain to distance from atom "atom" on mol2
        if ('mol2' in kwargs) and ('atomtype' in kwargs) and ('r0centroid' in kwargs ) :
            mol2 = kwargs['mol2'] ; atomtype = kwargs['atomtype'] ; r0centroid = kwargs['r0centroid']
            g2=[]
            for res in self.simmd.topology.residues():
                if res.name == mol2:
                    for i in range(len(res._atoms)):
                        if res._atoms[i].name == atomtype:
                            g2.append(res._atoms[i].index)
                    break

            self.Centroidforce = CustomCentroidBondForce(2,"0.5*k*(distance(g1,g2)-r0centroid)^2")
            self.system.addForce(self.Centroidforce)
            self.Centroidforce.addPerBondParameter("k")
            self.Centroidforce.addPerBondParameter("r0centroid")
            self.Centroidforce.addGroup(g1)
            self.Centroidforce.addGroup(g2)
            bondgroups =[0,1]
            bondparam = [k,r0centroid]
            self.Centroidforce.addBond(bondgroups,bondparam)
            self.Centroidforce.setUsesPeriodicBoundaryConditions(True)
            self.Centroidforce.addGlobalParameter('r0centroid',r0centroid)
            #self.Centroidforce.addEnergyParameterDerivative('r0centroid')

            for i in range(self.system.getNumForces()):
                f = self.system.getForce(i)
                f.setForceGroup(i)

        # option 2: input 'z_global' :  constrain to absolute z distance
        elif 'z_global' in kwargs :
            z_global = kwargs['z_global']
            self.ZForce = CustomExternalForce("0.5 * k * periodicdistance(x,y,z,x,y,z0)^2")
            self.system.addForce(self.ZForce)
            # add particles to force
            for index in g1:
                self.ZForce.addParticle(index)
            self.ZForce.addGlobalParameter('z0', z_global)
            self.ZForce.addGlobalParameter('k', k )

        else:
            print("couldn't recognize **kwargs input in setumbrella method...")
            sys.exit()


        # reinitialize context and set positions
        self.simmd.context.reinitialize()
        self.simmd.context.setPositions(self.modeller.positions)



    #************************************************
    # this method writes electrode charges to output file
    #
    #  FIX:  Not sure the best way to determine order???
    #   we might need to write cathode, conductor , anode charges,
    #   or cathode, anode , conductor charges in either order??
    #   how to automate this??
    #************************************************
    def write_electrode_charges( self, chargeFile ):

        if self.Cathode is not None and self.Anode is not None :
            # first cathode then anode charges
            for atom in self.Cathode.electrode_atoms:
                chargeFile.write("{:f} ".format(atom.charge))          

            # loop over additional Conductors (buckyballs/nanotubes) if we have them
            for Conductor in self.Conductor_list:
                for atom in Conductor.electrode_atoms:
                    chargeFile.write("{:f} ".format(atom.charge))

            for atom in self.Anode.electrode_atoms:
                chargeFile.write("{:f} ".format(atom.charge))

        else:
            # only have Conductors ....
            for Conductor in self.Conductor_list:
                for atom in Conductor.electrode_atoms:
                    chargeFile.write("{:f} ".format(atom.charge))


        # write newline to charge file after charge write
        chargeFile.write("\n")
        chargeFile.flush() # flush buffer


    #************************************************
    # this method reads electrode charges from output file for simulation restart
    # and sets the charges in the electrode data structures and context
    #
    #  FIX:  same issue as write_electrode_charges with ordering ...
    #************************************************
    def read_and_set_electrode_charges( self, chargeFile , charge_name ):
        # this file is already open for append, close, read, then re-open for append
        chargeFile.close()

	# reopen for read, and read charges from last line
        with open(charge_name) as f:
            for line in f:
                pass
            last_line = line

        charges = line.split()

        charge_index=0
        # now assign charges in same order that they were written to file
        if self.Cathode is not None and self.Anode is not None :
            # first cathode then anode charges
            for atom in self.Cathode.electrode_atoms:
                atom.charge = float(charges[charge_index])
                (q_i_old, sig, eps) = self.nbondedForce.getParticleParameters(atom.atom_index)
                self.nbondedForce.setParticleParameters(atom.atom_index, atom.charge, sig , eps)
                charge_index += 1          

            # loop over additional Conductors (buckyballs/nanotubes) if we have them
            for Conductor in self.Conductor_list:
                for atom in Conductor.electrode_atoms:
                    atom.charge = float(charges[charge_index])
                    (q_i_old, sig, eps) = self.nbondedForce.getParticleParameters(atom.atom_index)
                    self.nbondedForce.setParticleParameters(atom.atom_index, atom.charge, sig , eps)
                    charge_index += 1

            for atom in self.Anode.electrode_atoms:
                atom.charge = float(charges[charge_index])
                (q_i_old, sig, eps) = self.nbondedForce.getParticleParameters(atom.atom_index)
                self.nbondedForce.setParticleParameters(atom.atom_index, atom.charge, sig , eps)
                charge_index += 1

        else:
            # only have Conductors ....
            for Conductor in self.Conductor_list:
                for atom in Conductor.electrode_atoms:
                    atom.charge = float(charges[charge_index])
                    (q_i_old, sig, eps) = self.nbondedForce.getParticleParameters(atom.atom_index)
                    self.nbondedForce.setParticleParameters(atom.atom_index, atom.charge, sig , eps)
                    charge_index += 1

        # put charges into context
        self.nbondedForce.updateParametersInContext(self.simmd.context)

        # close and reopen for append
        chargeFile.close()
        chargeFile = open(charge_name, "a")
        
        return chargeFile


#**************************************************
# this is a stand-alone method to copy positions/charges from
# one simulation context to another.  Currently we use this
# if we are running a simulation to compute vext grid, and
# we want to switch back and forth between GPU/CPU contexts ...
#**************************************************
def copy_positions_charges_between_contexts( simmd_copy_from , simmd_copy_to , nbondedForce_copy_from , nbondedForce_copy_to ):

    # we might as well compare energies here to make sure copy over is correct.
    # this will of course slow things down, but we are taking a big performance hit
    # by transfering GPU/CPU and computing vext ...

    # first copy charges so that we can reinitialize before setting positions ...
    for atom in simmd_copy_from.topology.atoms():
        # get charges from force field, these have been updated with the Poisson solver ...
        (q_i, sig, eps) = nbondedForce_copy_from.getParticleParameters(atom.index)
        # copy to context, might as well copy sig, eps also ...
        nbondedForce_copy_to.setParticleParameters(atom.index, q_i, sig , eps)

    # reinitialize context with the new charges ...
    simmd_copy_to.context.reinitialize()

    # now copy positions
    state_copy_from = simmd_copy_from.context.getState(getEnergy=True,getForces=False,getVelocities=False,getPositions=True)
    energy_copy_from = state_copy_from.getPotentialEnergy()
    positions = state_copy_from.getPositions()
    simmd_copy_to.context.setPositions(positions)


    # turn off vext grid to test energy, assume simmd_copy_to is CPU ...
    platform_copy_to = simmd_copy_to.context.getPlatform()
    property_value_store = platform_copy_to.getPropertyValue( simmd_copy_to.context , 'ReferenceVextGrid')
    platform_copy_to.setPropertyValue( simmd_copy_to.context , 'ReferenceVextGrid' , "false" )

    # energy comparison
    state_copy_to = simmd_copy_to.context.getState(getEnergy=True,getForces=False,getVelocities=False,getPositions=True)
    energy_copy_to = state_copy_to.getPotentialEnergy()

    print( 'Energy comparison from switching contexts ...' , energy_copy_from , energy_copy_to )

    # now turn vext grid back on if it was previously ...
    platform_copy_to.setPropertyValue( simmd_copy_to.context , 'ReferenceVextGrid' , property_value_store )



# this is standalone helper method, outside of class
# input is an OpenMM state object
def get_box_vectors_Bohr( state, nm_to_bohr ):
    # this is returned as a list of Openmm Vec3 filled with quantities (value, unit)
    # convert to unitless list to pass to Psi4
    box_temp = state.getPeriodicBoxVectors()
    box=[]
    for i in range(3):
        box.append( [ box_temp[i][0]._value * nm_to_bohr , box_temp[i][1]._value * nm_to_bohr, box_temp[i][2]._value * nm_to_bohr ] )

    return box


# print a cube file with vext
def print_cube_file( cubefile, vext , box_bohr , pme_grid_size_a , pme_grid_size_b , pme_grid_size_c ):
    # first write header comment lines , need at least 1 atom for VMD so add junk atom ...
    cubefile.write("Potential Cube file\n")
    cubefile.write("-------------------\n")
    cubefile.write("{0:5d}{1:12.6f}{2:12.6f}{3:12.6f}\n".format( 1 , 0.0 , 0.0 , 0.0 ) )
    cubefile.write("{0:5d}{1:12.6f}{2:12.6f}{3:12.6f}\n".format( pme_grid_size_a , box_bohr[0][0] / pme_grid_size_a , box_bohr[0][1]/pme_grid_size_a , box_bohr[0][2]/pme_grid_size_a ) )
    cubefile.write("{0:5d}{1:12.6f}{2:12.6f}{3:12.6f}\n".format( pme_grid_size_b , box_bohr[1][0] / pme_grid_size_b , box_bohr[1][1]/pme_grid_size_b , box_bohr[1][2]/pme_grid_size_b ) )
    cubefile.write("{0:5d}{1:12.6f}{2:12.6f}{3:12.6f}\n".format( pme_grid_size_c , box_bohr[2][0] / pme_grid_size_c , box_bohr[2][1]/pme_grid_size_c , box_bohr[2][2]/pme_grid_size_c ) )
    # add junk atom
    cubefile.write("{0:5d}{1:12.6f}{2:12.6f}{3:12.6f}{4:12.6f}\n".format( 6 , 0.0 , 0.0 , 0.0 , 0.0 ) )

    # now print vext cube data
    for i in range(pme_grid_size_a):
        for j in range(pme_grid_size_b):
            for k in range(pme_grid_size_c):
                index = i * pme_grid_size_b * pme_grid_size_c + j * pme_grid_size_c + k
                cubefile.write( "{0:12.6f}".format( vext[index] ) )
                if ( k % 6 == 5 ):
                    cubefile.write("\n")
            cubefile.write("\n")

    cubefile.flush() # flush buffer




#****************************************************
# this is a small class used for storing Monte Carlo parameters ...
#****************************************************
class MC_parameters(object):
    def __init__( self , temperature , celldim , electrode_move="Anode" , pressure = 1.0*bar , barofreq = 25 , shiftscale = 0.2 ):
        self.RT = BOLTZMANN_CONSTANT_kB * temperature * AVOGADRO_CONSTANT_NA     
        self.pressure = pressure*celldim[0] * celldim[1] * AVOGADRO_CONSTANT_NA # convert pressure to force ...
        self.electrode_move = electrode_move
        self.barofreq = barofreq
        self.shiftscale = shiftscale
        self.ntrials = 0
        self.naccept = 0
