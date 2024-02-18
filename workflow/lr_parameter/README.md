# README
In this example, We show a calculation with a OCCOCCO molecule. The major input is the clt file: occocco.clt                  
There are several tags in the clt file that worths noticing: 

Asymptotic correction is to be applied to the exchange-correlation potential, I.P. and HOMO provide alternative ways to specify data for it. 
* The I.P. and HOMO: these two are used to correct the asymptotic behavior of the density functional: 
  * I.P. specifies the ionization potential, normally in hartree, but it can be given in eV if the eV unit is specified. Usually We compute it using PBE0/AVTZ: $IP=E_{cation}-E_{mol}$. 
  * HOMO specifies the energy of the highest occupied molecular orbital at PBE0/AVTZ level., given in hartree. 
* The basis set specification for ISA-pol is a bit complicated, the current setting seems to be fine, you can use directly.
* The *SCFcode* tag specifies which code you use to compute the response matrix: camcasp itself does not compute response matrix, it uses other code to do TDDFT. We usually use NWChem, but you can also use others, like Psi4(see camcasp manual).

In the package, We provide the script (scamcasp) which is used to run camcasp, most of it are just slurm settings, the real command is simply:

```bash
python $CCPROOT/bin/$CCP_EXEC $NAME -d $FOLDER --memory ${MEM/mb/} --cores $NCPUS $OPTS > logfile
```

Once you run it, you will see the folder named "occocco", and an "OUT" folder in it. Entering that folder, you may see the "occocco_ISA-GRID.mom" file, in which all multipoles are given for each atom. But up to now, the polarization matrix is still nonlocal, so you do not have atomic polarizabilities and dispersion coefficients. To get those properties, you need to run the "localize.sh" script to distribute and refine the atomic polarizabilities. 
Then you would see "occocco_ref_wt3_L2iso_Cniso.pot" file, which gives you the dispersion coefficients, and the "occocco_ref_wt3_L2iso_000.pol" file, which gives you static isotropic dipole and quadrupole polarizabilities.

Now, we've get multipoles, atomic polarizabilities and dispersion coefficients by CAMCASP, the next step is take these properties into XML format force field, which can be read by DMFF directly.

Before that, we need to do the multipole local frame definition and atom typing, such as "occocco.axes" and "atypes.dat" in the "create_xml" folder. You can do you own definition by your chemical intuition.

Then, you can use these python scripts in "create_xml" folder to do the work.

```bash
# get multipoles
python reformat_mom.py *_ISA-GRID.mom > reform.mom
python mom2xml_direct.py reform.mom *.axes > 1_mom_openmm.xml
./convert_mom_to_xml.py reform.mom *.axes > tmp
python symmetrize_param.py tmp > 2_multipole
# get atomic polarizabilities
python gen_pol.py *_ref_wt3_L2iso_000.out > 3_pol
# get dispersion coefficients
python gen_dispersion_Cn.py *_ref_wt3_L2iso_Cniso.pot > 4_disp
```
You can manualy integrate your own XML force field XML in element "Residues" with bond information and "AtomTypes" by you own definition to "occocco.xml".