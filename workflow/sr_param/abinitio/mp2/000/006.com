! shift = -0.666667
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
14,C,,14.30641651,12.10469055,16.83693504
15,H,,13.41541672,11.50069046,17.05193520
16,H,,14.57941723,12.51069069,17.85593605
17,O,,13.91441631,13.08869076,15.88993549
18,C,,12.76941681,13.92369080,16.15393448
19,H,,12.94541645,14.86569118,15.65593529
20,H,,12.70441723,13.99869061,17.27293587
21,H,,11.84341717,13.44969082,15.89693546
22,C,,15.42641640,11.23269081,16.23893547
23,H,,14.94641685,10.43069077,15.57093525
24,H,,16.02141762,10.84169102,17.06193542
25,O,,16.13541794,12.15069103,15.50693512
26,H,,15.48641682,12.83569050,15.45993519
27,He,,13.78607715,16.06822313,15.56236385
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
