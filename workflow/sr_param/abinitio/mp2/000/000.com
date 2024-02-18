! shift = -2.300000
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
14,C,,13.98403835,13.68663216,16.58932686
15,H,,13.09303856,13.08263206,16.80432701
16,H,,14.25703907,14.09263229,17.60832787
17,O,,13.59203815,14.67063236,15.64232731
18,C,,12.44703865,15.50563240,15.90632629
19,H,,12.62303829,16.44763184,15.40832710
20,H,,12.38203907,15.58063221,17.02532768
21,H,,11.52103901,15.03163242,15.64932728
22,C,,15.10403824,12.81463242,15.99132729
23,H,,14.62403870,12.01263237,15.32332706
24,H,,15.69903851,12.42363262,16.81432724
25,O,,15.81303883,13.73263264,15.25932693
26,H,,15.16403866,14.41763210,15.21232700
27,He,,13.62488796,16.85919393,15.43855976
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
