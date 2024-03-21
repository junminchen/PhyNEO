# PhyNEO

[![DOI: 10.1021/acs.jctc.3c01045](https://img.shields.io/badge/DOI-10.1021%2Facs.jctc.3c01045-blue)](https://pubs.acs.org/doi/10.1021/acs.jctc.3c01045)
 
## About PhyNEO

**PhyNEO** (**Phy**sics-driven and **N**eural-Network **E**nhanced **O**rganic and polymer Force Field) is a force field development workflow, based on [DMFF](https://github.com/deepmodeling/DMFF). PhyNEO features a hybrid approach that combines both the physics-driven and the data-driven methods and is able to generate a bulk potential with chemical accuracy using only quantum chemistry data of very small
clusters. Careful separations of long-/short-range interactions and nonbonding/bonding interactions are the key to the success of PhyNEO. By such a strategy, PhyNEO mitigate the limitations of pure data-driven methods in long-range interactions, thus largely increasing the data efficiency and the scalability of machine learning models.

### License and credits

The project PhyNEO is licensed under [GNU LGPL v3.0](LICENSE). If you use this code in any future publications, please cite this using `CHEN, Junmin; YU, Kuang. PhyNEO: A Neural-Network-Enhanced Physics-Driven Force Field Development Workflow for Bulk Organic Molecule and Polymer Simulations. Journal of Chemical Theory and Computation, 2023, 20.1: 253-265. DOI: 10.1021/acs.jctc.3c01045`

## User Guide

+ `md_example`: demos in papers interfaced i-Pi.

### Long-range Parameters

Dependency: 
`CAMCASP`: https://www-stone.ch.cam.ac.uk/programs.html
`Tinker (for the Local frame generator POLEDIT or we can do it by ourself)`: https://dasher.wustl.edu/tinker/

Firstly, we need to use CAMCASP to do TD-DFT and ISA-pol calculation. See it in 'conf.DMC'.
Then, we can use our scripts to generate XML force field file with the help of Tinker tool.
+ `worflow/lr_param/poledit`: Tinker files for constructing local frame definition to 'localframe'. 
```bash
./poledit DMC.xyz
```
Notes: JUST PRESS 'ENTER' TO LAST QUESTIONS!!! All we need is coping the local frame definition to 'localframe' file.
```txt
Local Frame Definition for Multipole Sites :

     Atom     Name      Axis Type     Z Axis  X Axis  Y Axis

       1      C         Z-then-X         2       3       0
       2      O         Z-then-X         3       1       0
       3      C         Bisector         2       5       0
       4      O         Z-Only           3       0       0
       5      O         Z-then-X         3       6       0
       6      C         Z-then-X         5       3       0
       7      H         Z-then-X         1       2       0
       8      H         Z-then-X         1       2       0
       9      H         Z-then-X         1       2       0
      10      H         Z-then-X         6       5       0
      11      H         Z-then-X         6       5       0
      12      H         Z-then-X         6       5       0
```
+ `worflow/lr_param/1_gen_atype.py`: generate atom type definition to 'atype_data.pickle'.
```bash
python 1_gen_atype.py
```
+ `worflow/lr_param/2_ff_gen.py`: generate PhyNEO forcefield by one step.
```bash
python 2_ff_gen.py > dmff_forcefield.xml
```
The force field is like: 
```xml
<?xml version="1.0" ?>
<forcefield>
  <AtomTypes>
    <Type name="11" class="CT" element="C" mass="12.0107"/>
    <Type name="12" class="OS" element="O" mass="15.999"/>
    <Type name="13" class="CT" element="C" mass="12.0107"/>
    <Type name="14" class="OS" element="O" mass="15.999"/>
    <Type name="15" class="HC" element="H" mass="1.00784"/>
  </AtomTypes>
  <Residues>
    <Residue name="DMC">
      <Atom name="C00" type="11"/>
      <Atom name="O01" type="12"/>
      <Atom name="C02" type="13"/>
      <Atom name="O03" type="14"/>
      <Atom name="O04" type="12"/>
      <Atom name="C05" type="11"/>
      <Atom name="H06" type="15"/>
      <Atom name="H07" type="15"/>
      <Atom name="H08" type="15"/>
      <Atom name="H09" type="15"/>
      <Atom name="H10" type="15"/>
      <Atom name="H11" type="15"/>
      <Bond from="0" to="1"/>
      <Bond from="0" to="6"/>
      <Bond from="0" to="7"/>
      <Bond from="0" to="8"/>
      <Bond from="1" to="2"/>
      <Bond from="2" to="3"/>
      <Bond from="2" to="4"/>
      <Bond from="4" to="5"/>
      <Bond from="5" to="9"/>
      <Bond from="5" to="10"/>
      <Bond from="5" to="11"/>
    </Residue>
  </Residues>
  <ADMPPmeForce lmax="2" mScale12="0.00" mScale13="0.00" mScale14="0.00" mScale15="0.00" mScale16="0.00" pScale12="0.00" pScale13="0.00" pScale14="0.00" pScale15="0.00" pScale16="0.00" dScale12="1.00" dScale13="1.00" dScale14="1.00" dScale15="1.00" dScale16="1.00">
    <Atom type="11" kz="12" kx="13" c0="-0.03898950" dX="0.00033182" dY="0.00000048" dZ="0.00865776" qXX="-0.00005796" qXY="-0.00000009" qYY="-0.00012532" qXZ="0.00000085" qYZ="-0.00000004" qZZ="0.00018328"/>
    <Atom type="12" kz="13" kx="11" c0="-0.37016750" dX="0.00809055" dY="0.00000042" dZ="-0.00135041" qXX="0.00005658" qXY="-0.00000002" qYY="-0.00018218" qXZ="-0.00017535" qYZ="0.00000001" qZZ="0.00012560"/>
    <Atom type="13" kz="-12" kx="12" c0="0.93485900" dX="0.00001033" dY="-0.00000127" dZ="0.00176034" qXX="0.00005266" qXY="-0.00000005" qYY="-0.00002300" qXZ="0.00000001" qYZ="0.00000001" qZZ="-0.00002966"/>
    <Atom type="14" kz="13" kx="0" c0="-0.64127600" dX="-0.00000413" dY="0.00000942" dZ="-0.00464310" qXX="-0.00003035" qXY="0.00000013" qYY="-0.00006016" qXZ="0.00000000" qYZ="0.00000008" qZZ="0.00009051"/>
    <Atom type="15" kz="11" kx="12" c0="0.08762733" dX="0.00041250" dY="-0.00000049" dZ="-0.00226565" qXX="0.00001204" qXY="-0.00000000" qYY="-0.00003004" qXZ="0.00000172" qYZ="0.00000001" qZZ="0.00001800"/>
    <Polarize type="11" polarizabilityXX="1.0526e-03" polarizabilityYY="1.0526e-03" polarizabilityZZ="1.0526e-03" thole="0.33"/>
    <Polarize type="12" polarizabilityXX="8.4683e-04" polarizabilityYY="8.4683e-04" polarizabilityZZ="8.4683e-04" thole="0.33"/>
    <Polarize type="13" polarizabilityXX="4.4148e-04" polarizabilityYY="4.4148e-04" polarizabilityZZ="4.4148e-04" thole="0.33"/>
    <Polarize type="14" polarizabilityXX="1.2051e-03" polarizabilityYY="1.2051e-03" polarizabilityZZ="1.2051e-03" thole="0.33"/>
    <Polarize type="15" polarizabilityXX="2.8361e-04" polarizabilityYY="2.8361e-04" polarizabilityZZ="2.8361e-04" thole="0.33"/>
  </ADMPPmeForce>
  <ADMPDispPmeForce mScale12="0.00" mScale13="0.00" mScale14="0.00" mScale15="0.00" mScale16="0.00">
    <Atom type="11" C6="1.497383e-03" C8="1.101717e-04" C10="3.837074e-06"/>
    <Atom type="12" C6="9.685357e-04" C8="6.265103e-05" C10="1.955545e-06"/>
    <Atom type="13" C6="4.278543e-04" C8="5.019178e-05" C10="2.751691e-06"/>
    <Atom type="14" C6="1.626594e-03" C8="7.121221e-05" C10="1.487238e-06"/>
    <Atom type="15" C6="8.767545e-05" C8="2.992354e-06" C10="5.200268e-08"/>
  </ADMPDispPmeForce>
  <SlaterExForce mScale12="0.00" mScale13="0.00" mScale14="0.00" mScale15="0.00" mScale16="0.00">
    <Atom type="11" A="1.0" B="34.37579209"/>
    <Atom type="12" A="1.0" B="37.81250801"/>
    <Atom type="13" A="1.0" B="34.37579209"/>
    <Atom type="14" A="1.0" B="37.81250801"/>
    <Atom type="15" A="1.0" B="37.78372041"/>
  </SlaterExForce>
  <SlaterSrEsForce mScale12="0.00" mScale13="0.00" mScale14="0.00" mScale15="0.00" mScale16="0.00">
    <Atom type="11" A="1.0" B="34.37579209" Q="-0.03898950"/>
    <Atom type="12" A="1.0" B="37.81250801" Q="-0.37016750"/>
    <Atom type="13" A="1.0" B="34.37579209" Q="0.93485900"/>
    <Atom type="14" A="1.0" B="37.81250801" Q="-0.64127600"/>
    <Atom type="15" A="1.0" B="37.78372041" Q="0.08762733"/>
  </SlaterSrEsForce>
  <SlaterSrPolForce mScale12="0.00" mScale13="0.00" mScale14="0.00" mScale15="0.00" mScale16="0.00">
    <Atom type="11" A="1.0" B="34.37579209"/>
    <Atom type="12" A="1.0" B="37.81250801"/>
    <Atom type="13" A="1.0" B="34.37579209"/>
    <Atom type="14" A="1.0" B="37.81250801"/>
    <Atom type="15" A="1.0" B="37.78372041"/>
  </SlaterSrPolForce>
  <SlaterSrDispForce mScale12="0.00" mScale13="0.00" mScale14="0.00" mScale15="0.00" mScale16="0.00">
    <Atom type="11" A="1.0" B="34.37579209"/>
    <Atom type="12" A="1.0" B="37.81250801"/>
    <Atom type="13" A="1.0" B="34.37579209"/>
    <Atom type="14" A="1.0" B="37.81250801"/>
    <Atom type="15" A="1.0" B="37.78372041"/>
  </SlaterSrDispForce>
  <SlaterDhfForce mScale12="0.00" mScale13="0.00" mScale14="0.00" mScale15="0.00" mScale16="0.00">
    <Atom type="11" A="1.0" B="34.37579209"/>
    <Atom type="12" A="1.0" B="37.81250801"/>
    <Atom type="13" A="1.0" B="34.37579209"/>
    <Atom type="14" A="1.0" B="37.81250801"/>
    <Atom type="15" A="1.0" B="37.78372041"/>
  </SlaterDhfForce>
  <QqTtDampingForce mScale12="0.00" mScale13="0.00" mScale14="0.00" mScale15="0.00" mScale16="0.00">
    <Atom type="11" B="34.37579209" Q="-0.03898950"/>
    <Atom type="12" B="37.81250801" Q="-0.37016750"/>
    <Atom type="13" B="34.37579209" Q="0.93485900"/>
    <Atom type="14" B="37.81250801" Q="-0.64127600"/>
    <Atom type="15" B="37.78372041" Q="0.08762733"/>
  </QqTtDampingForce>
  <SlaterDampingForce mScale12="0.00" mScale13="0.00" mScale14="0.00" mScale15="0.00" mScale16="0.00">
    <Atom type="11" B="34.37579209" C6="1.497383e-03" C8="1.101717e-04" C10="3.837074e-06"/>
    <Atom type="12" B="37.81250801" C6="9.685357e-04" C8="6.265103e-05" C10="1.955545e-06"/>
    <Atom type="13" B="34.37579209" C6="4.278543e-04" C8="5.019178e-05" C10="2.751691e-06"/>
    <Atom type="14" B="37.81250801" C6="1.626594e-03" C8="7.121221e-05" C10="1.487238e-06"/>
    <Atom type="15" B="37.78372041" C6="8.767545e-05" C8="2.992354e-06" C10="5.200268e-08"/>
  </SlaterDampingForce>
</forcefield>
```

### Short-range Parameters

Notes: In this part, it's a **different and independent** case from Long-range Parameters.

Dependency: 
`Molpro (or we can use Psi4)`: https://www.molpro.net

#### Data Preparation 

+ `worflow/sr_param/abinitio/run_md.py`: generate the trajectory of dimer by classical force field MD simulation
+ `worflow/sr_param/abinitio/geoms_op.py`: generate dimer scan Molpro input file to do SAPT calculation
+ `worflow/sr_param/abinitio/sapt/pack_data.py`: pack dimer scan data to 'data.pickle'

#### Training 

+ `worflow/sr_param/remove_lr.py`: separate the long-range interaction energy to fitting short-range parameters 
+ `worflow/sr_param/fit_basepair.py`: fit the short-range parameters by dimer scan data, and save it in 'params.pickle', then plot it in 'test_deomp3.png'.
![image](https://github.com/Jeremydream/PhyNEO/blob/main/workflow/sr_param/test_decomp3.png)

### sub-Graph Neural Networks for Bonding interaction

Dependency: 
`DMFF`: https://github.com/deepmodeling/DMFF

#### Data Preparation 

As our paper mentioned, we generate 20000 monomer points at 300K and 600K, set training-testing ratio at 9:1. 

+ `worflow/GNN_bonding/abinitio_intra*/run_md.py`: generate the trajectory of monomer by classical force field MD simulation
+ `worflow/GNN_bonding/abinitio_intra*/gen_inputs.py`: generate monomer Molpro input file to do SAPT calculation
+ `worflow/GNN_bonding/abinitio_intra*/serialize.py`: serialize monomer data to 'set*.pickle'

#### Training 

+ `worflow/GNN_bonding/creat_dataset.py`: integrate all the data to 'dataset_train.pickle', 'dataset_test.pickle' and 'dataset_test_pe16.pickle'
+ `worflow/GNN_bonding/remove_nonbonding.py`: separate the nonbonding interaction energy to fitting bonding parameters 
```bash
python remove_nonbonding.py dataset_train.pickle pe8.pdb
python remove_nonbonding.py dataset_test.pickle pe8.pdb
python remove_nonbonding.py dataset_test_pe16.pickle pe16.pdb
```
+ `worflow/GNN_bonding/train.py`: train the bonding parameters by pe8 monomer data, and save it in 'params_sgnn.pickle', then we can use 'plot_data.py' plot it in 'test_data.png'.
+ `worflow/GNN_bonding/test_pe16.py`: test the bonding parameters by pe16 monomer data, and save it in 'test_data_pe16.xvg'. Then we can use 'plot_data.py' plot it in 'test_data_pe16.png', which shows the **transferability** in polymer. 
![image](https://github.com/Jeremydream/PhyNEO/blob/main/workflow/GNN_bonding/test_data.png)![image](https://github.com/Jeremydream/PhyNEO/blob/main/workflow/GNN_bonding/test_data_pe16.png)

### Molecular Dynamics Simulation by PhyNEO: PE Polymer

Dependency: 
`DMFF`: https://github.com/deepmodeling/DMFF
`i-pi`: https://github.com/i-pi/i-pi
`packmol`: https://m3g.github.io/packmol/

#### Preparation and run simulation

+ `worflow/md_pe/bulk.inp.py`: we use **packmol** to generate the initial box before run simulation
+ `worflow/md_pe/client_dmff.py`: the client calulator with PhyNEO and DMFF to produce energy, force and virial. 

## Support and Contribution

Please visit our repository on [GitHub](https://github.com/Jeremydream/PhyNEO) for the library source code. Any issues or bugs may be reported at our issue tracker. All contributions to PhyNEO are welcomed via pull requests!
