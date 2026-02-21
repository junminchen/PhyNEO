import numpy as np
bohr = 0.52917720859
dCon = 0.1*bohr
qCon = 0.01*bohr*bohr/3.0
oCon = 0.001*bohr*bohr*bohr/15.0
sf = 3*[dCon]+6*[qCon]+10*[oCon]

SWMO = list(map(float, "0.00000 0.00000 -0.50626 0.11497 0.00000 0.11497 0.00000 0.00000 -0.22994 0.00000 0.00000 0.00000 0.00000 0.05222 0.00000 0.05222 0.00000 0.00000 -0.10443".split()))

def prettyprint(atomlabel, vals):
    vals = np.multiply(sf, vals)
    X = vals[0]
    Y = vals[1]
    Z = vals[2]
    XX = vals[3]
    XY = vals[4]
    YY = vals[5]
    XZ = vals[6]
    YZ = vals[7]
    ZZ = vals[8]
    opoles = vals[9:]
    XXX = opoles[0]
    XXY = opoles[1]
    XYY = opoles[2]
    YYY = opoles[3]
    XXZ = opoles[4]
    XYZ = opoles[5]
    YYZ = opoles[6]
    XZZ = opoles[7]
    YZZ = opoles[8]
    ZZZ = opoles[9]
    print('dX="%.10g" dY="%.10g" dZ="%.10g"' % (X, Y, Z))
    print('qXX="%.10g" qXY="%.10g" qXZ="%.10g" qYY="%.10g" qYZ="%.10g" qZZ="%.10g"' % (XX, XY, XZ, YY, YZ, ZZ))
    print('oXXX="%.10g" oXXY="%.10g" oXYY="%.10g" oYYY="%.10g" oXXZ="%.10g" oXYZ="%.10g" oYYZ="%.10g" oXZZ="%.10f" oYZZ="%.10g" oZZZ="%.10g"' % (XXX, XXY, XYY, YYY, XXZ, XYZ, YYZ, XZZ, YZZ, ZZZ))

prettyprint('o', SWMO)
