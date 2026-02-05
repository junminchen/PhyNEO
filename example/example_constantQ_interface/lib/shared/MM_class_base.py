from simtk.openmm.app import *
from simtk.openmm import *
from simtk.unit import *
from sys import stdout
#******** exclusions for force field 
from .MM_exclusions_base import *
from .rigid import *

#*************************************************
# This is the base MM parent class that is meant for general use when invoking OpenMM
#
# Any specialized/simulation-specific run-control should be implemented in a child class
# of this parent class.  Because this parent class may be used for different types of
# simulations, it may be utilized in several different github repositories which may
# be best managed using Git subtrees
#
# We set reasonable default run-parameters in this base class, and allow
# modification to these arguments with **kwargs input.
# 
#**************************************************
class MM_base(object):
    # required input: 1) list of pdb files, 2) list of residue xml files, 3) list of force field xml files.
    def __init__(self, pdb_list, residue_xml_list, ff_xml_list, **kwargs):
        #*************************************
        #  DEFAULT RUN PARAMETERS: input in **kwargs may overide defaults
        #**************************************
        self.temperature = 300*kelvin
        self.temperature_drude = 1*kelvin
        self.friction = 1/picosecond
        self.friction_drude = 1/picosecond
        self.timestep = 0.001*picoseconds
        self.small_threshold = 1e-6  # threshold for charge magnitude
        self.cutoff = 1.4*nanometer
        self.nonbonded_method = "PME"
        self.npt_barostat = False
        self.rigid_body = None

        # reading inputs from **kwargs
        if 'temperature' in kwargs :
            self.temperature = int(kwargs['temperature'])*kelvin
        if 'temperature_drude' in kwargs :
            self.temperature_drude = int(kwargs['temperature_drude'])*kelvin
        if 'friction' in kwargs :
            self.friction = int(kwargs['friction'])/picosecond
        if 'friction_drude' in kwargs :
            self.friction_drude = int(kwargs['friction_drude'])/picosecond
        if 'timestep' in kwargs :
            self.timestep = float(kwargs['timestep'])*picoseconds
        if 'small_threshold' in kwargs :
            self.small_threshold = float(kwargs['small_threshold'])
        if 'cutoff' in kwargs :
            self.cutoff = float(kwargs['cutoff'])*nanometer
        if 'nonbonded_method' in kwargs :
            self.nonbonded_method = kwargs['nonbonded_method']
        if 'npt_barostat' in kwargs :
            self.npt_barostat = bool(kwargs['npt_barostat'])
            self.pressure = float(kwargs['pressure'])*atmosphere
        if 'rigid_body' in kwargs :
            self.rigid_body = kwargs['rigid_body']


        # load bond definitions before creating pdb object (which calls createStandardBonds() internally upon __init__).  Note that loadBondDefinitions is a static method
        # of Topology, so even though PDBFile creates its own topology object, these bond definitions will be applied...
        for residue_file in residue_xml_list:
            Topology().loadBondDefinitions(residue_file)

        # now create pdb object, use first pdb file input
        self.pdb = PDBFile( pdb_list[0] )

        # create modeller
        self.modeller = Modeller(self.pdb.topology, self.pdb.positions)
        # create force field
        self.forcefield = ForceField(*ff_xml_list)
        # add extra particles
        self.modeller.addExtraParticles(self.forcefield)
        
        # create openMM system object
        self.system = self.forcefield.createSystem(self.modeller.topology, nonbondedCutoff=self.cutoff, constraints=HBonds, rigidWater=True)
        # get force types and set method
        self.nbondedForce = [f for f in [self.system.getForce(i) for i in range(self.system.getNumForces())] if type(f) == NonbondedForce][0]
        self.customNonbondedForce = [f for f in [self.system.getForce(i) for i in range(self.system.getNumForces())] if type(f) == CustomNonbondedForce][0]
        
        # check if we have a DrudeForce for polarizable simulation
        drudeF = [f for f in [self.system.getForce(i) for i in range(self.system.getNumForces())] if type(f) == DrudeForce]
        if drudeF:
            self.polarization = True
            self.drudeForce = drudeF[0]
            # will only have this for certain polarizable molecules
            self.custombond = [f for f in [self.system.getForce(i) for i in range(self.system.getNumForces())] if type(f) == CustomBondForce][0]
        else:
            self.polarization = False
        
        # set long-range interaction method
        if self.nonbonded_method == 'NoCutoff':
            print( "setting NonbondedForce method to NoCutoff" )
            self.nbondedForce.setNonbondedMethod(NonbondedForce.NoCutoff)
        elif self.nonbonded_method == 'PME':
            print( "setting NonbondedForce method to PME" )
            self.nbondedForce.setNonbondedMethod(NonbondedForce.PME)
        else:
            print ('No such method for nbondedForce (long range interaction method not set correctly in MM_base)')
            sys.exit()

        if self.customNonbondedForce :
            print( "setting CustomNonbondedForce method to CutoffPeriodic" )
            self.customNonbondedForce.setNonbondedMethod(min(self.nbondedForce.getNonbondedMethod(),NonbondedForce.CutoffPeriodic))

        if self.npt_barostat:
            barofreq = 100
            barostat = MonteCarloBarostat(self.pressure, self.temperature, barofreq)
            self.system.addForce(barostat)
            print ('Simulation set to run using NPT ensemble with external pressure of %s atm.' % self.NPT_barostat_pressure)
        
        if self.polarization :
            #************** Polarizable simulation, use Drude integrator with standard settings
            self.integrator = DrudeLangevinIntegrator(self.temperature, self.friction, self.temperature_drude, self.friction_drude, self.timestep)
            # this should prevent polarization catastrophe during equilibration, but shouldn't affect results afterwards ( 0.2 Angstrom displacement is very large for equil. Drudes)
            self.integrator.setMaxDrudeDistance(0.02)
        else :
            #************** Non-polarizable simulation
            self.integrator = LangevinIntegrator(self.temperature, self.friction, self.timestep)

        # Create rigid bodies (self.rigid_body should be a list of atom types/classes)
        if self.rigid_body is not None:
            # In order to make rigid body selection by atom type, need to find mapping from atom_type -> atom_name -> atom_index
            # Atom_type -> atom_name mapping is found in the in internal forcefield templates
            rigid_body_atom_names = []
            for template in self.forcefield._templates:
                for atom in self.forcefield._templates[template].atoms:
                    if atom.type in self.rigid_body:
                        rigid_body_atom_names.append(atom.name)

            # Atom_name -> atom_index mapping is found in the modeller topology
            bodies = []
            for res in self.modeller.topology.residues():
                body = []
                for atom in res._atoms:
                    if atom.name in rigid_body_atom_names:
                        body.append(atom.index)
                if body != []:
                    bodies.append(body)
            createRigidBodies(self.system, self.modeller.positions, bodies)

    #*********************************************
    # set output frequency for coordinate dcd file
    #*********************************************
    def set_trajectory_output( self, filename , write_frequency , append_trajectory=False, checkpointfile = None , write_checkpoint_frequency = 10000 ):
        self.simmd.reporters = []
        self.simmd.reporters.append(DCDReporter(filename, write_frequency, append=append_trajectory))
        # add checkpointing reporter if input
        if checkpointfile :
            self.simmd.reporters.append(CheckpointReporter(checkpointfile, write_checkpoint_frequency))

    #*********************************************
    # this sets the force groups to be used with PBC
    # call this if a molecule/residue is broken up over PBC,
    # e.g. graphene electrode ...
    #*********************************************
    def set_periodic_residue(self, flag):
        for i in range(self.system.getNumForces()):
            f = self.system.getForce(i)
            f.setForceGroup(i)
            # if using PBC
            if flag:
                # Here we are adding periodic boundaries to intra-molecular interactions.  Note that DrudeForce does not have this attribute, and
                # so if we want to use thole screening for graphite sheets we might have to implement periodic boundaries for this force type
                if type(f) == HarmonicBondForce or type(f) == HarmonicAngleForce or type(f) == PeriodicTorsionForce or type(f) == RBTorsionForce:
                    f.setUsesPeriodicBoundaryConditions(True)
                    f.usesPeriodicBoundaryConditions()

    #*********************************************
    # set the platform/OpenMM kernel and initialize simulation object
    #*********************************************
    #*********** Currently can only use 'Reference' for QM/MM ...
    def set_platform( self, platformname ):
        if platformname == 'Reference':
            self.platform = Platform.getPlatformByName('Reference')
            self.simmd = Simulation(self.modeller.topology, self.system, self.integrator, self.platform)
        elif platformname == 'CPU':
            self.platform = Platform.getPlatformByName('CPU')
            self.simmd = Simulation(self.modeller.topology, self.system, self.integrator, self.platform)
        elif platformname == 'OpenCL':
            self.platform = Platform.getPlatformByName('OpenCL')
            # we found weird bug with 'mixed' precision on OpenCL related to updating parameters in context for gold/water simulation...
            #self.properties = {'OpenCLPrecision': 'mixed'}
            self.simmd = Simulation(self.modeller.topology, self.system, self.integrator, self.platform)
        elif platformname == 'CUDA':
            self.platform = Platform.getPlatformByName('CUDA')
            self.properties = {'Precision': 'mixed'}
            self.simmd = Simulation(self.modeller.topology, self.system, self.integrator, self.platform, self.properties)
        else:
            print(' Could not recognize platform selection ... ')
            sys.exit(0)
        self.simmd.context.setPositions(self.modeller.positions)

    #***************************************
    # this generates force field exclusions that we commonly utilize for water simulations
    #
    # if flag_SAPT_FF_exclusions=True, then will also set exclusions for SAPT-FF force field...
    #***************************************
    def generate_exclusions(self, water_name = 'HOH', flag_hybrid_water_model = False ,  flag_SAPT_FF_exclusions = True ):

        # if special exclusion for SAPT-FF force field ...
        if flag_SAPT_FF_exclusions:
            generate_SAPT_FF_exclusions( self )

        # if using a hybrid water model, need to create interaction groups for customnonbonded force....
        if flag_hybrid_water_model:
            generate_exclusions_water(self.simmd, self.customNonbondedForce, water_name )

        # having both is redundant, as SAPT-FF already creates interaction groups for water/other
        if flag_SAPT_FF_exclusions and flag_hybrid_water_model:
            print( "redundant setting of flag_SAPT_FF_exclusions and flag_hybrid_water_model")
            sys.exit()


        # now reinitialize to make sure changes are stored in context
        state = self.simmd.context.getState(getEnergy=False,getForces=False,getVelocities=False,getPositions=True)
        positions = state.getPositions()
        self.simmd.context.reinitialize()
        self.simmd.context.setPositions(positions)
