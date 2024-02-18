! shift = -1.075000
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
14,C,,14.22582245,12.50017548,16.77503204
15,H,,13.33482265,11.89617538,16.99003220
16,H,,14.49882317,12.90617561,17.79403305
17,O,,13.83382225,13.48417568,15.82803345
18,C,,12.68882275,14.31917572,16.09203148
19,H,,12.86482239,15.26117611,15.59403324
20,H,,12.62382317,14.39417553,17.21103287
21,H,,11.76282310,13.84517574,15.83503342
22,C,,15.34582233,11.62817574,16.17703247
23,H,,14.86582279,10.82617569,15.50903320
24,H,,15.94082260,11.23717594,17.00003242
25,O,,16.05482292,12.54617596,15.44503307
26,H,,15.40582275,13.23117542,15.39803314
27,He,,13.74578001,16.26596559,15.53141258
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
