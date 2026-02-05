
"""
    Set up exclusions for electrodes 
    exclusions for electrolyte should be defined in
    MM_exclusions_base.py
"""


#**********************************
# Adds exclusions for intra-electrode interactions of atoms
# Call this method with lists of electrode atoms 'electrode1' and 'electrode2'
# and all long range interactions between atoms on this electrode(s) will
# be excluded.  These lists could be the same or different, and we take two lists to allow versatility
#
# Note that different code is executed depending on whether we have a CustomNonbondedForce ...
#
#**********************************
def exclusion_Electrode_NonbondedForce(sim, system, electrode1, electrode2, customNonbondedForce , nbondedForce ):

  if customNonbondedForce:

    # first figure out which exclusions we already have (1-4 atoms and closer).  The code doesn't
    # allow the same exclusion to be added twice
    flagexclusions = {}

    for i in range(customNonbondedForce.getNumExclusions()):
        (particle1, particle2) = customNonbondedForce.getExclusionParticles(i)
        string1=str(particle1)+"_"+str(particle2)
        string2=str(particle2)+"_"+str(particle1)
        flagexclusions[string1]=1
        flagexclusions[string2]=1

    # now add exclusions for every atom pair in electrode if we don't already have them
    if electrode1 == electrode2:
        for i in range(len(electrode1)):
            indexi = electrode1[i]
            for j in range(i+1,len(electrode2)):
                indexj = electrode2[j]
                string1=str(indexi)+"_"+str(indexj)
                string2=str(indexj)+"_"+str(indexi)
                if string1 in flagexclusions and string2 in flagexclusions:
                    continue
                else:
                    customNonbondedForce.addExclusion(indexi,indexj)
                    nbondedForce.addException(indexi,indexj,0,1,0,True)
    else:
        # different chains/residues
        for i in range(len(electrode1)):
            indexi = electrode1[i]
            for j in range(len(electrode2)):
                indexj = electrode2[j]
                string1=str(indexi)+"_"+str(indexj)
                string2=str(indexj)+"_"+str(indexi)
                if string1 in flagexclusions and string2 in flagexclusions:
                    continue
                else:
                    customNonbondedForce.addExclusion(indexi,indexj)
                    nbondedForce.addException(indexi,indexj,0,1,0,True)

  # this code executes if no CustomNonbondedForce ...
  else :
    # now add exclusions for every atom pair in electrode if we don't already have them
    if electrode1 == electrode2:
        for i in range(len(electrode1)):
            indexi = electrode1[i]
            for j in range(i+1,len(electrode2)):
                indexj = electrode2[j]
                nbondedForce.addException(indexi,indexj,0,1,0,True)
    else:
        # different chains/residues
        for i in range(len(electrode1)):
            indexi = electrode1[i]
            for j in range(len(electrode2)):
                indexj = electrode2[j]
                nbondedForce.addException(indexi,indexj,0,1,0,True)


