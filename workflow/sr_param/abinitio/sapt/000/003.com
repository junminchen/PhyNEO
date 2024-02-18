! shift= -2.237500
memory,1.4,G
gdirect; gthresh,energy=1.d-8,orbital=1.d-8,grid=1.d-8
spherical
angstrom
symmetry,nosym
orient,noorient
geometry={
1,C,,15.84099960,12.23099995,16.81900024
2,H,,16.68899918,12.78299999,17.18700027
3,H,,15.91800022,12.06299973,15.70600033
4,H,,15.78199959,11.22799969,17.34300041
5,C,,14.52799988,12.96899986,17.05100060
6,H,,14.62899971,13.89900017,16.40200043
7,H,,14.44600010,13.28800011,18.12800026
8,C,,13.21899986,12.19699955,16.56200027
9,H,,13.17199993,11.23400021,17.00699997
10,H,,13.44900036,11.92399979,15.48900032
11,C,,11.89900017,12.92899990,16.73800087
12,H,,12.07900047,13.79500008,16.05900002
13,H,,11.82299995,13.35000038,17.83499908
14,C,,10.63300037,12.19099998,16.37400055
15,H,,10.38399982,11.46100044,17.12400055
16,H,,10.80399990,11.61900043,15.46000004
17,C,,9.50399971,13.27499962,16.26899910
18,H,,9.33199978,13.83699989,17.20899963
19,H,,8.59099960,12.80099964,15.85799980
20,H,,9.71500015,13.93400002,15.47200012
21,C,,16.03562927,18.21379471,15.05923748
22,H,,16.88062859,17.86179543,14.44823742
23,H,,16.43962860,18.76479530,15.90523720
24,H,,15.44762802,18.77979469,14.41123772
25,C,,15.20462799,17.12779617,15.67023754
26,H,,15.69362926,16.49379539,16.37123680
27,H,,14.87462807,16.44179535,14.87223721
28,C,,13.97762871,17.74379539,16.34823799
29,H,,13.41762829,18.39979553,15.65723705
30,H,,14.37962818,18.38179588,17.13623810
31,C,,12.93062878,16.63579559,16.77223778
32,H,,13.17962837,16.12579536,17.66223717
33,H,,12.84862804,15.98379517,15.94223690
34,C,,11.47662830,17.09779549,17.01923752
35,H,,11.14162827,17.69879532,16.12523651
36,H,,11.51562881,17.87379456,17.82923698
37,C,,10.46362877,16.02779579,17.30523682
38,H,,10.51162815,15.36579514,16.39723778
39,H,,9.48562813,16.52379608,17.43123817
40,H,,10.77762794,15.37079620,18.15423775
41,He,,12.97651458,14.88924484,16.49106666
}
basis={
set,orbital
default=avtz         !for orbitals
s,He,even,nprim=5,ratio=2.5,center=0.5
p,He,even,nprim=5,ratio=2.5,center=0.5
d,He,even,nprim=3,ratio=2.5,center=0.3
f,He,even,nprim=2,ratio=2.5,center=0.3
set,jkfit
default=avtz/jkfit   !for JK integrals
s,He,even,nprim=5,ratio=2.5,center=0.5
p,He,even,nprim=5,ratio=2.5,center=0.5
d,He,even,nprim=3,ratio=2.5,center=0.3
f,He,even,nprim=2,ratio=2.5,center=0.3
set,mp2fit 
default=avtz/mp2fit  !for E2disp/E2exch-disp
s,He,even,nprim=5,ratio=2.5,center=0.5
p,He,even,nprim=5,ratio=2.5,center=0.5
d,He,even,nprim=3,ratio=2.5,center=0.3
f,He,even,nprim=2,ratio=2.5,center=0.3
set,dflhf
default=avtz/jkfit   !for LHF
s,He,even,nprim=5,ratio=2.5,center=0.5
p,He,even,nprim=5,ratio=2.5,center=0.5
d,He,even,nprim=3,ratio=2.5,center=0.3
f,He,even,nprim=2,ratio=2.5,center=0.3
}

!=========delta(HF) contribution for higher order interaction terms====
ca=2101.2; cb=2102.2 !sapt files

!dimer
dummy,41
{df-hf,basis=jkfit,locorb=0}
edm=energy

!monomer A
dummy,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41

{df-hf,basis=jkfit,locorb=0; save,$ca}
ema=energy
{sapt;monomerA}

!monomer B
dummy,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,41

{df-hf,basis=jkfit,locorb=0; save,$cb}
emb=energy
{sapt;monomerB}

!interaction contributions
{sapt,SAPT_LEVEL=2;intermol,ca=$ca,cb=$cb,icpks=1,fitlevel=3
dfit,basis_coul=jkfit,basis_exch=jkfit,cfit_scf=3}

!calculate high-order terms by subtracting 1st+2nd order energies
eint_hf=(edm-ema-emb)*1000 mH
delta_hf=eint_hf-e1pol-e1ex-e2ind-e2exind

!=========DFT-SAPT at second order intermol. perturbation theory====
ca=2103.2; cb=2104.2 !sapt files;

!shifts for asymptotic correction to xc potential
eps_homo_PBE0_B=-0.318417
eps_homo_PBE0_A=-0.318417
ip_B=0.396796
ip_A=0.396796
shift_B=ip_B+eps_homo_pbe0_B !shift for bulk xc potential (B)
shift_A=ip_A+eps_homo_pbe0_A !shift for bulk xc potential (A)

!monomer A, perform LPBE0AC calculation
dummy,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41

{df-ks,pbex,pw91c,lhf,df_basis=jkfit; dftfac,0.75,1.0,0.25; asymp,shift_A; save,$ca}
{sapt;monomerA}

!monomer B, perform LPBE0AC calculation
dummy,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,41

{df-ks,pbex,pw91c,lhf,df_basis=jkfit; dftfac,0.75,1.0,0.25; start,atdens; asymp,shift_B; save,$cb}
{sapt;monomerB}

!interaction contributions
{sapt,SAPT_LEVEL=3;intermol,ca=$ca,cb=$cb,icpks=0,fitlevel=3,nlexfac=0.0
dfit,basis_coul=jkfit,basis_exch=jkfit,basis_mp2=mp2fit,cfit_scf=3}

!add high-order approximation to obtain the total interaction energy
eint_dftsapt=e12tot+delta_hf

