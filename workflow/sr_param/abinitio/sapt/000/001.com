! shift = -1.279167
! shift= -1.279167
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
14,C,,12.73888302,15.05039978,16.18000793
15,H,,12.33388329,16.00239944,16.33300781
16,H,,12.24188328,14.56839943,15.37700844
17,O,,12.60188293,14.26239967,17.36300850
18,C,,11.30988312,13.88140011,17.73300743
19,H,,11.19888306,13.23539925,18.57500839
20,H,,10.88088322,13.35139942,16.87000847
21,H,,10.64488316,14.68139935,17.93000793
22,C,,14.24388313,15.11039925,15.91400909
23,H,,14.62688255,15.91440010,16.53100777
24,H,,14.47088337,15.37539959,14.86300850
25,O,,14.84988308,13.89639950,16.35300827
26,H,,14.27588272,13.52239990,17.01800728
27,He,,14.08714573,16.21321972,15.23167339
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

