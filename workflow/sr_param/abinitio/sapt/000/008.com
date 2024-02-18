! shift = 1.150000
! shift= 1.150000
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
14,C,,11.83327198,13.31583309,17.61945152
15,H,,11.42827225,14.26783276,17.77245140
16,H,,11.33627224,12.83383274,16.81645203
17,O,,11.69627190,12.52783298,18.80245209
18,C,,10.40427208,12.14683342,19.17245102
19,H,,10.29327202,11.50083256,20.01445198
20,H,,9.97527218,11.61683273,18.30945206
21,H,,9.73927212,12.94683266,19.36945152
22,C,,13.33827209,13.37583256,17.35345268
23,H,,13.72127151,14.17983341,17.97045135
24,H,,13.56527233,13.64083290,16.30245209
25,O,,13.94427204,12.16183281,17.79245186
26,H,,13.37027168,11.78783321,18.45745087
27,He,,13.63434021,15.34593637,15.95139519
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

