! shift = 1.987500
memory,800,m
gdirect; gthresh,energy=1.d-8,orbital=1.d-8,grid=1.d-8
spherical
angstrom
symmetry,nosym
orient,noorient
geometry={
1,C,,13.43799973,18.83499908,15.08699989
2,H,,13.12600040,17.79800034,15.25000000
3,H,,14.44099998,18.95499992,15.45499992
4,O,,12.53600025,19.84499931,15.76000023
5,C,,12.27499962,19.61199951,17.10199928
6,H,,11.74300003,20.40600014,17.49399948
7,H,,13.18000031,19.41200066,17.71599960
8,H,,11.53299999,18.84700012,17.17300034
9,C,,13.47999954,19.13100052,13.58500004
10,H,,12.61499977,18.81999969,13.08800030
11,H,,14.30900002,18.68700027,13.06099987
12,O,,13.60599995,20.54800034,13.47000027
13,H,,13.22999954,20.86599922,14.27000046
14,C,,14.83028126,9.53403473,17.23929787
15,H,,13.93928146,8.93003464,17.45429802
16,H,,15.10328197,9.94003487,18.25829887
17,O,,14.43828106,10.51803493,16.29229927
18,C,,13.29328156,11.35303497,16.55629730
19,H,,13.46928120,12.29503536,16.05829811
20,H,,13.22828197,11.42803478,17.67529869
21,H,,12.36728191,10.87903500,16.29929924
22,C,,15.95028114,8.66203499,16.64129829
23,H,,15.47028160,7.86003494,15.97329903
24,H,,16.54528236,8.27103519,17.46429825
25,O,,16.65928268,9.58003521,15.90929890
26,H,,16.01028252,10.26503468,15.86229897
27,He,,14.04800953,14.78289522,15.76354548
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

!dimer
dummy,27
df-hf, df_basis=jkfit
df-mp2, df_basis=mp2fit
edm=energy

!monomer A
dummy,14,15,16,17,18,19,20,21,22,23,24,25,26,27

df-hf, df_basis=jkfit
df-mp2, df_basis=mp2fit
ema=energy
{sapt;monomerA}

!monomer B
dummy,1,2,3,4,5,6,7,8,9,10,11,12,13,27

df-hf, df_basis=jkfit
df-mp2, df_basis=mp2fit
emb=energy

eint=edm-ema-emb
