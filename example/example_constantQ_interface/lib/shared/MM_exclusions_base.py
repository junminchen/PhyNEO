#********** OpenMM Drivers
from simtk.openmm.app import *
from simtk.openmm import *
from simtk.unit import *
from sys import *

#*************************** README  **************************************
#  The methods in this module setup additional exclusions
#  that are commonly used in our simulations, in particular with the SAPT force field.
#  This module should thus be considered "part of" the force field,
#**************************************************************************



#*********************************
# water models generally use specific water-water interaction parameters
# and separate interaction parameters for water-other
# here assume water-water interactions are in NonbondedForce, and
# water-other are in CustomNonbonded force, and create interaction
# groups for CustomNonbonded force...
#**********************************
def generate_exclusions_water(sim,customNonbondedForce,watername):

    # create Interaction Groups for hybrid water model
    water=set()
    notwater=set()
    print('Creating Interaction Groups for CustomNonBonded.  These interactions will be computed between water-other, not water-water.')
    for res in sim.topology.residues():
        if res.name == watername:
            for i in range(len(res._atoms)):
                water.update([res._atoms[i].index])
        else:
            for i in range(len(res._atoms)):
                notwater.update([res._atoms[i].index])

    customNonbondedForce.addInteractionGroup(water, notwater)
    customNonbondedForce.addInteractionGroup(notwater, notwater)


#********************************
# this sets exclusions for the SAPT-FF force field
#********************************
def generate_SAPT_FF_exclusions(MMsys):

    #************************** List of molecule types that have special exclusions ******
    # now see what molecule types are present.  These name are hardcoded in, i don't see any way around this since special exclusions
    # have to be molecule specific
    watername = 'HOH'
    TFSIname  = 'Tf2N'
    #***********************************************

    # ************* Add Water exclusions
    for res in MMsys.simmd.topology.residues():
        if res.name == watername:
            generate_exclusions_water(MMsys.simmd, MMsys.customNonbondedForce, watername)
            break

    # ************* Add TFSI exclusions, note might refer to this as Tf2N
    for res in MMsys.simmd.topology.residues():
        if res.name == TFSIname:
            generate_exclusions_TFSI(MMsys.simmd, MMsys.drudeForce, MMsys.nbondedForce, MMsys.customNonbondedForce, TFSIname)
            break


#*******************************
# this is for the TFSI anion when modeled with SAPT-FF,
# all intra-molecular non-bonded interactions must be exclused
#*******************************
def generate_exclusions_TFSI(sim, drudeForce , nbondedForce, customNonbondedForce, TFSIname):
    """
    This creates exclusions for TFSI nonbonded interactions, and update
    Screened Drude interactions.  1-5 non-Coulomb interaction are accounted for
    using CustomBondForce
    """
    print('Creating Exclusions for TFSI')

    # map from global particle index to drudeforce object index
    particleMap = {}
    for i in range(drudeForce.getNumParticles()):
        particleMap[drudeForce.getParticleParameters(i)[0]] = i

    # can't add duplicate ScreenedPairs, so store what we already have
    flagexceptions = {}
    for i in range(nbondedForce.getNumExceptions()):
        (particle1, particle2, charge, sigma, epsilon) = nbondedForce.getExceptionParameters(i)
        string1=str(particle1)+"_"+str(particle2)
        string2=str(particle2)+"_"+str(particle1)
        flagexceptions[string1]=1
        flagexceptions[string2]=1

    # can't add duplicate customNonbonded exclusions, so store what we already have
    flagexclusions = {}
    for i in range(customNonbondedForce.getNumExclusions()):
        (particle1, particle2) = customNonbondedForce.getExclusionParticles(i)
        string1=str(particle1)+"_"+str(particle2)
        string2=str(particle2)+"_"+str(particle1)
        flagexclusions[string1]=1
        flagexclusions[string2]=1

    # add exclusions for all atom pairs on TFSI residues, and when a drude pair is
    # excluded add a corresponding screened thole interaction in its place
    for res in sim.topology.residues():
        if res.name == TFSIname:
            for i in range(len(res._atoms)-1):
                for j in range(i+1,len(res._atoms)):
                    (indi,indj) = (res._atoms[i].index, res._atoms[j].index)
                    # here it doesn't matter if we already have this, since we pass the "True" flag
                    nbondedForce.addException(indi,indj,0,1,0,True)
                    # make sure we don't already exlude this customnonbond
                    string1=str(indi)+"_"+str(indj)
                    string2=str(indj)+"_"+str(indi)
                    if string1 in flagexclusions and string2 in flagexclusions:
                        continue
                    else:
                        customNonbondedForce.addExclusion(indi,indj)
                    # add thole if we're excluding two drudes
                    if indi in particleMap and indj in particleMap:
                        # make sure we don't already have this screened pair
                        if string1 in flagexceptions or string2 in flagexceptions:
                            continue
                        else:
                            drudei = particleMap[indi]
                            drudej = particleMap[indj]
                            drudeForce.addScreenedPair(drudei, drudej, 2.0)
