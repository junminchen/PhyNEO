! shift = -0.837500
! shift= -0.837500
memory,1.4,G
gdirect; gthresh,energy=1.d-8,orbital=1.d-8,grid=1.d-8
spherical
angstrom
symmetry,nosym
orient,noorient
geometry={
1,C,,14.87800026,17.89299965,12.85200024
2,H,,15.63599968,17.51000023,12.08199978
3,H,,14.07499981,18.39500046,12.37199974
4,O,,15.53299999,18.69099998,13.90299988
5,C,,16.32399940,19.74099922,13.32900047
6,H,,16.80599976,20.18799973,14.15999985
7,H,,15.77799988,20.54800034,12.77000046
8,H,,17.18499947,19.43099976,12.74300003
9,C,,14.21700001,16.66699982,13.53499985
10,H,,14.98299980,15.85700035,13.43599987
11,H,,13.31900024,16.40500069,13.02499962
12,O,,14.10900021,16.99699974,14.97999954
13,H,,14.77700043,17.71800041,15.08699989
14,C,,12.57422638,14.73502445,16.44172478
15,H,,12.16922665,15.68702412,16.59472466
16,H,,12.07722664,14.25302410,15.63872528
17,O,,12.43722630,13.94702435,17.62472534
18,C,,11.14522648,13.56602478,17.99472427
19,H,,11.03422642,12.92002392,18.83672523
20,H,,10.71622658,13.03602409,17.13172531
21,H,,10.48022652,14.36602402,18.19172478
22,C,,14.07922649,14.79502392,16.17572594
23,H,,14.46222591,15.59902477,16.79272461
24,H,,14.30622673,15.06002426,15.12472534
25,O,,14.68522644,13.58102417,16.61472511
26,H,,14.11122608,13.20702457,17.27972412
27,He,,14.00481741,16.05553205,15.36253182
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
dummy,27
{df-hf,basis=jkfit,locorb=0}
edm=energy

!monomer A
dummy,14,15,16,17,18,19,20,21,22,23,24,25,26,27

{df-hf,basis=jkfit,locorb=0; save,$ca}
ema=energy
{sapt;monomerA}

!monomer B
dummy,1,2,3,4,5,6,7,8,9,10,11,12,13,27

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
eps_homo_PBE0_B=-0.286797
eps_homo_PBE0_A=-0.286797
ip_B=0.369864
ip_A=0.369864
shift_B=ip_B+eps_homo_pbe0_B !shift for bulk xc potential (B)
shift_A=ip_A+eps_homo_pbe0_A !shift for bulk xc potential (A)

!monomer A, perform LPBE0AC calculation
dummy,14,15,16,17,18,19,20,21,22,23,24,25,26,27

{df-ks,pbex,pw91c,lhf,df_basis=jkfit; dftfac,0.75,1.0,0.25; asymp,shift_A; save,$ca}
{sapt;monomerA}

!monomer B, perform LPBE0AC calculation
dummy,1,2,3,4,5,6,7,8,9,10,11,12,13,27

{df-ks,pbex,pw91c,lhf,df_basis=jkfit; dftfac,0.75,1.0,0.25; start,atdens; asymp,shift_B; save,$cb}
{sapt;monomerB}

!interaction contributions
{sapt,SAPT_LEVEL=3;intermol,ca=$ca,cb=$cb,icpks=0,fitlevel=3,nlexfac=0.0
dfit,basis_coul=jkfit,basis_exch=jkfit,basis_mp2=mp2fit,cfit_scf=3}

!add high-order approximation to obtain the total interaction energy
eint_dftsapt=e12tot+delta_hf

