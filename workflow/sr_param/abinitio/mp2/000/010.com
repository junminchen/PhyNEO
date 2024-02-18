! shift = 1.375000
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
14,C,,14.70938969,10.12726307,17.14644623
15,H,,13.81838989,9.52326298,17.36144638
16,H,,14.98239040,10.53326321,18.16544724
17,O,,14.31738949,11.11126328,16.19944572
18,C,,13.17238998,11.94626331,16.46344566
19,H,,13.34838963,12.88826370,15.96544552
20,H,,13.10739040,12.02126312,17.58244705
21,H,,12.24639034,11.47226334,16.20644569
22,C,,15.82938957,9.25526333,16.54844666
23,H,,15.34939003,8.45326328,15.88044548
24,H,,16.42438889,8.86426353,17.37144661
25,O,,16.53838921,10.17326355,15.81644535
26,H,,15.88938999,10.85826302,15.76944542
27,He,,13.98756352,15.07950939,15.71711922
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
