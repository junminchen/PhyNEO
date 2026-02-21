#define WARPS_PER_GROUP (THREAD_BLOCK_SIZE/TILE_SIZE)

typedef struct {
    real3 pos, force, torque, inducedDipole, sphericalDipole;
    real q;
    float thole, damp;
#ifdef INCLUDE_QUADRUPOLES
    real sphericalQuadrupole[5];
#endif
#ifdef INCLUDE_OCTOPOLES
    real sphericalOctopole[7];
#endif
} AtomData;

inline __device__ void loadAtomData(AtomData& data, int atom, const real4* __restrict__ posq, const real* __restrict__ sphericalDipole,
            const real* __restrict__ sphericalQuadrupole, const real* __restrict__ sphericalOctopole, const real* __restrict__ inducedDipole,
            const float2* __restrict__ dampingAndThole) {
    real4 atomPosq = posq[atom];
    data.pos = make_real3(atomPosq.x, atomPosq.y, atomPosq.z);
    data.q = atomPosq.w;
    data.sphericalDipole.x = sphericalDipole[atom*3];
    data.sphericalDipole.y = sphericalDipole[atom*3+1];
    data.sphericalDipole.z = sphericalDipole[atom*3+2];
#ifdef INCLUDE_QUADRUPOLES
    data.sphericalQuadrupole[0] = sphericalQuadrupole[atom*5];
    data.sphericalQuadrupole[1] = sphericalQuadrupole[atom*5+1];
    data.sphericalQuadrupole[2] = sphericalQuadrupole[atom*5+2];
    data.sphericalQuadrupole[3] = sphericalQuadrupole[atom*5+3];
    data.sphericalQuadrupole[4] = sphericalQuadrupole[atom*5+4];
#endif
#ifdef INCLUDE_OCTOPOLES
    data.sphericalOctopole[0] = sphericalOctopole[atom*7];
    data.sphericalOctopole[1] = sphericalOctopole[atom*7+1];
    data.sphericalOctopole[2] = sphericalOctopole[atom*7+2];
    data.sphericalOctopole[3] = sphericalOctopole[atom*7+3];
    data.sphericalOctopole[4] = sphericalOctopole[atom*7+4];
    data.sphericalOctopole[5] = sphericalOctopole[atom*7+5];
    data.sphericalOctopole[6] = sphericalOctopole[atom*7+6];
#endif
    data.inducedDipole.x = inducedDipole[atom*3];
    data.inducedDipole.y = inducedDipole[atom*3+1];
    data.inducedDipole.z = inducedDipole[atom*3+2];
    float2 temp = dampingAndThole[atom];
    data.damp = temp.x;
    data.thole = temp.y;
}

__device__ float computeMScaleFactor(uint2 covalent, int index) {
    int mask = 1<<index;
    bool x = (covalent.x & mask);
    bool y = (covalent.y & mask);
	return (x ? (y ? 0.0f : (float) SCALEFACTOR14): 1.0f);
}


__device__ void computeOneInteraction(AtomData& atom1, AtomData& atom2, bool hasExclusions, float pScale, float mScale, float forceFactor,
                                      mixed& energy, real4 periodicBoxSize, real4 invPeriodicBoxSize, real4 periodicBoxVecX, real4 periodicBoxVecY, real4 periodicBoxVecZ) {
    // Compute the displacement.
    real3 delta;
    delta.x = atom2.pos.x - atom1.pos.x;
    delta.y = atom2.pos.y - atom1.pos.y;
    delta.z = atom2.pos.z - atom1.pos.z;
    APPLY_PERIODIC_TO_DELTA(delta)
    real r2 = delta.x*delta.x + delta.y*delta.y + delta.z*delta.z;
    if (r2 > CUTOFF_SQUARED)
        return;

    real rInv = RSQRT(r2);
    real r = r2*rInv;

    // Rotate the various dipoles and quadrupoles.

    real qiRotationMatrix[3][3];
    buildQIRotationMatrix(delta, rInv, qiRotationMatrix);

    real3 qiUindI = 0.5f*make_real3(qiRotationMatrix[0][1]*atom1.inducedDipole.x + qiRotationMatrix[0][2]*atom1.inducedDipole.y + qiRotationMatrix[0][0]*atom1.inducedDipole.z,
                                    qiRotationMatrix[1][1]*atom1.inducedDipole.x + qiRotationMatrix[1][2]*atom1.inducedDipole.y + qiRotationMatrix[1][0]*atom1.inducedDipole.z,
                                    qiRotationMatrix[2][1]*atom1.inducedDipole.x + qiRotationMatrix[2][2]*atom1.inducedDipole.y + qiRotationMatrix[2][0]*atom1.inducedDipole.z);
    real3 qiUindJ = 0.5f*make_real3(qiRotationMatrix[0][1]*atom2.inducedDipole.x + qiRotationMatrix[0][2]*atom2.inducedDipole.y + qiRotationMatrix[0][0]*atom2.inducedDipole.z,
                                    qiRotationMatrix[1][1]*atom2.inducedDipole.x + qiRotationMatrix[1][2]*atom2.inducedDipole.y + qiRotationMatrix[1][0]*atom2.inducedDipole.z,
                                    qiRotationMatrix[2][1]*atom2.inducedDipole.x + qiRotationMatrix[2][2]*atom2.inducedDipole.y + qiRotationMatrix[2][0]*atom2.inducedDipole.z);

    real3 rotatedDipole1 = rotateDipole(atom1.sphericalDipole, qiRotationMatrix);
    real3 rotatedDipole2 = rotateDipole(atom2.sphericalDipole, qiRotationMatrix);
    real rotatedQuadrupole1[] = {0, 0, 0, 0, 0};
    real rotatedQuadrupole2[] = {0, 0, 0, 0, 0};
#ifdef INCLUDE_QUADRUPOLES
    rotateQuadupoles(qiRotationMatrix, atom1.sphericalQuadrupole, atom2.sphericalQuadrupole, rotatedQuadrupole1, rotatedQuadrupole2);
#endif    
    real rotatedOctopole1[] = {0, 0, 0, 0, 0, 0, 0};
    real rotatedOctopole2[] = {0, 0, 0, 0, 0, 0, 0};
#ifdef INCLUDE_OCTOPOLES
    rotateOctopoles(qiRotationMatrix, atom1.sphericalOctopole, atom2.sphericalOctopole, rotatedOctopole1, rotatedOctopole2);
#endif    
    // The field derivatives at I due to permanent and induced moments on J, and vice-versa.
    // Also, their derivatives w.r.t. R, which are needed for force calculations
    real Vij[16], Vji[16], VjiR[16], VijR[16];
    // The field derivatives at I due to only permanent moments on J, and vice-versa.
    real Vijd[3], Vjid[3];
    real rInvVec[9], alphaRVec[10], bVec[6];

    // The rInvVec array is defined such that the ith element is R^-i, with the
    // dieleectric constant folded in, to avoid conversions later.
    rInvVec[1] = rInv;
    for (int i = 2; i < 9; ++i)
        rInvVec[i] = rInvVec[i-1] * rInv;

    // The alpharVec array is defined such that the ith element is (alpha R)^i,
    // where kappa (alpha in OpenMM parlance) is the Ewald attenuation parameter.
    real ralpha = EWALD_ALPHA*r;
    real exp2a = EXP(-(ralpha*ralpha));
#ifdef USE_DOUBLE_PRECISION
    const real erfAlphaR = erf(ralpha);
#else
    // This approximation for erfc is from Abramowitz and Stegun (1964) p. 299.  They cite the following as
    // the original source: C. Hastings, Jr., Approximations for Digital Computers (1955).  It has a maximum
    // error of 1.5e-7.

    const real t = RECIP(1.0f+0.3275911f*ralpha);
    const real erfAlphaR = 1-(0.254829592f+(-0.284496736f+(1.421413741f+(-1.453152027f+1.061405429f*t)*t)*t)*t)*t*exp2a;
#endif
    alphaRVec[1] = ralpha;
    for (int i = 2; i < 10; ++i)
        alphaRVec[i] = alphaRVec[i-1]*ralpha;
    real X = 2*exp2a/SQRT_PI;
    int doubleFactorial = 1, facCount = 1;
    real tmp = alphaRVec[1];
    bVec[1] = -erfAlphaR;
    for (int i = 2; i < 6; ++i) {
        bVec[i] = bVec[i-1] + tmp * X / (real)(doubleFactorial);
        facCount = facCount + 2;
        doubleFactorial = doubleFactorial * facCount;
        tmp *= 2*alphaRVec[2];
    }

    real dmp = fabs(atom1.damp*atom2.damp);
    real a = pScale == 0 ? atom1.thole + atom2.thole : DEFAULT_THOLE_WIDTH;
    real u = dmp > (real)1.0E-5 ? r/dmp : (real)1E6;
    real au = a*u;
    real expau = au < (real)50 ? exp(-au) : (real)0;
    au =  dmp > (real)1.0E-5 ? au : 0;
    real au2 = au*au;
    real au3 = au2*au;
    real au4 = au3*au;
    real au5 = au4*au;
    real au6 = au5*au;
    // Thole damping factors for energies
    real thole_c   = 1 - expau*(1 + au + au2/2);
    real thole_d0  = 1 - expau*(1 + au + au2/2 + au3/4);
    real thole_d1  = 1 - expau*(1 + au + au2/2);
    real thole_q0  = 1 - expau*(1 + au + au2/2 + au3/6 + au4/18);
    real thole_q1  = 1 - expau*(1 + au + au2/2 + au3/6);
    real thole_o0  = 1 - expau*(1 + au + au2/2 + au3/6 + au4/24 + au5/120);
    real thole_o1  = 1 - expau*(1 + au + au2/2 + au3/6 + au4/30);
    // Thole damping factors for derivatives
    real dthole_c  = 1 - expau*(1 + au + au2/2 + au3/4);
    real dthole_d0 = 1 - expau*(1 + au + au2/2 + au3/6 + au4/12);
    real dthole_d1 = 1 - expau*(1 + au + au2/2 + au3/6);
    real dthole_q0 = 1 - expau*(1 + au + au2/2 + au3/6 + au4/24 + au5/72);
    real dthole_q1 = 1 - expau*(1 + au + au2/2 + au3/6 + au4/24);
    real dthole_o0 = 1 - expau*(1 + au + au2/2 + au3/6 + au4/24 + au5/120 + au6/600);
    real dthole_o1 = 1 - expau*(1 + au + au2/2 + au3/6 + au4/25 + au5/150);

    // Now we compute the (attenuated) Coulomb operator and its derivatives, contracted with
    // permanent moments and induced dipoles.  Note that the coefficient of the permanent force
    // terms is half of the expected value; this is because we compute the interaction of I with
    // the sum of induced and permanent moments on J, as well as the interaction of J with I's
    // permanent and induced moments; doing so double counts the permanent-permanent interaction.
    real ePermCoef, dPermCoef, eUIndCoef, dUIndCoef;

    // C-C terms (m=0)
    ePermCoef = rInvVec[1]*(mScale + bVec[2] - alphaRVec[1]*X);
    dPermCoef = -0.5f*(mScale + bVec[2])*rInvVec[2];
    Vij[0]  = ePermCoef*atom2.q;
    Vji[0]  = ePermCoef*atom1.q;
    VijR[0] = dPermCoef*atom2.q;
    VjiR[0] = dPermCoef*atom1.q;

    // C-D and C-Uind terms (m=0)
    ePermCoef = rInvVec[2]*(mScale + bVec[2]);
    eUIndCoef = 2*rInvVec[2]*(pScale*thole_c + bVec[2]);
    dPermCoef = -rInvVec[3]*(mScale + bVec[2] + alphaRVec[3]*X);
    dUIndCoef = -4*rInvVec[3]*(pScale*dthole_c + bVec[2] + alphaRVec[3]*X);
    Vij[0]  += -(ePermCoef*rotatedDipole2.x + eUIndCoef*qiUindJ.x);
    Vji[1]   = -(ePermCoef*atom1.q);
    VijR[0] += -(dPermCoef*rotatedDipole2.x + dUIndCoef*qiUindJ.x);
    VjiR[1]  = -(dPermCoef*atom1.q);
    Vjid[0]  = -(eUIndCoef*atom1.q);
    // D-C and Uind-C terms (m=0)
    Vij[1]   = ePermCoef*atom2.q;
    Vji[0]  += ePermCoef*rotatedDipole1.x + eUIndCoef*qiUindI.x;
    VijR[1]  = dPermCoef*atom2.q;
    VjiR[0] += dPermCoef*rotatedDipole1.x + dUIndCoef*qiUindI.x;
    Vijd[0]  = eUIndCoef*atom2.q;

    // D-D and D-Uind terms (m=0)
    const real twoThirds = (real) 2/3;
    ePermCoef = -twoThirds*rInvVec[3]*(3*(mScale + bVec[3]) + alphaRVec[3]*X);
    eUIndCoef = -2*twoThirds*rInvVec[3]*(3*(pScale*thole_d0 + bVec[3]) + alphaRVec[3]*X);
    dPermCoef = rInvVec[4]*(3*(mScale + bVec[3]) + 2*alphaRVec[5]*X);
    dUIndCoef = 2*rInvVec[4]*(6*(pScale*dthole_d0 + bVec[3]) + 4*alphaRVec[5]*X);
    Vij[1]  += ePermCoef*rotatedDipole2.x + eUIndCoef*qiUindJ.x;
    Vji[1]  += ePermCoef*rotatedDipole1.x + eUIndCoef*qiUindI.x;
    VijR[1] += dPermCoef*rotatedDipole2.x + dUIndCoef*qiUindJ.x;
    VjiR[1] += dPermCoef*rotatedDipole1.x + dUIndCoef*qiUindI.x;
    Vijd[0] += eUIndCoef*rotatedDipole2.x;
    Vjid[0] += eUIndCoef*rotatedDipole1.x;
    // D-D and D-Uind terms (m=1)
    ePermCoef = rInvVec[3]*(mScale + bVec[3] - twoThirds*alphaRVec[3]*X);
    eUIndCoef = 2*rInvVec[3]*(pScale*thole_d1 + bVec[3] - twoThirds*alphaRVec[3]*X);
    dPermCoef = -1.5f*rInvVec[4]*(mScale + bVec[3]);
    dUIndCoef = -6*rInvVec[4]*(pScale*dthole_d1 + bVec[3]);
    Vij[2]  = ePermCoef*rotatedDipole2.y + eUIndCoef*qiUindJ.y;
    Vji[2]  = ePermCoef*rotatedDipole1.y + eUIndCoef*qiUindI.y;
    VijR[2] = dPermCoef*rotatedDipole2.y + dUIndCoef*qiUindJ.y;
    VjiR[2] = dPermCoef*rotatedDipole1.y + dUIndCoef*qiUindI.y;
    Vij[3]  = ePermCoef*rotatedDipole2.z + eUIndCoef*qiUindJ.z;
    Vji[3]  = ePermCoef*rotatedDipole1.z + eUIndCoef*qiUindI.z;
    VijR[3] = dPermCoef*rotatedDipole2.z + dUIndCoef*qiUindJ.z;
    VjiR[3] = dPermCoef*rotatedDipole1.z + dUIndCoef*qiUindI.z;
    Vijd[1] = eUIndCoef*rotatedDipole2.y;
    Vjid[1] = eUIndCoef*rotatedDipole1.y;
    Vijd[2] = eUIndCoef*rotatedDipole2.z;
    Vjid[2] = eUIndCoef*rotatedDipole1.z;

    // C-Q terms (m=0)
    ePermCoef = (mScale + bVec[3])*rInvVec[3];
    dPermCoef = -((real) 1/3)*rInvVec[4]*(4.5f*(mScale + bVec[3]) + 2*alphaRVec[5]*X);
    Vij[0]  += ePermCoef*rotatedQuadrupole2[0];
    Vji[4]   = ePermCoef*atom1.q;
    VijR[0] += dPermCoef*rotatedQuadrupole2[0];
    VjiR[4]  = dPermCoef*atom1.q;
    // Q-C terms (m=0)
    Vij[4]   = ePermCoef*atom2.q;
    Vji[0]  += ePermCoef*rotatedQuadrupole1[0];
    VijR[4]  = dPermCoef*atom2.q;
    VjiR[0] += dPermCoef*rotatedQuadrupole1[0];

    // D-Q and Uind-Q terms (m=0)
    const real fourThirds = (real) 4/3;
    ePermCoef = rInvVec[4]*(3*(mScale + bVec[3]) + fourThirds*alphaRVec[5]*X);
    eUIndCoef = 2*rInvVec[4]*(3*(pScale*thole_q0 + bVec[3]) + fourThirds*alphaRVec[5]*X);
    dPermCoef = -fourThirds*rInvVec[5]*(4.5f*(mScale + bVec[3]) + (1 + alphaRVec[2])*alphaRVec[5]*X);
    dUIndCoef = -2*fourThirds*rInvVec[5]*(9*(pScale*dthole_q0 + bVec[3]) + 2*(1 + alphaRVec[2])*alphaRVec[5]*X);
    Vij[1]  += ePermCoef*rotatedQuadrupole2[0];
    Vji[4]  += ePermCoef*rotatedDipole1.x + eUIndCoef*qiUindI.x;
    VijR[1] += dPermCoef*rotatedQuadrupole2[0];
    VjiR[4] += dPermCoef*rotatedDipole1.x + dUIndCoef*qiUindI.x;
    Vijd[0] += eUIndCoef*rotatedQuadrupole2[0];
    // Q-D and Q-Uind terms (m=0)
    Vij[4]  += -(ePermCoef*rotatedDipole2.x + eUIndCoef*qiUindJ.x);
    Vji[1]  += -(ePermCoef*rotatedQuadrupole1[0]);
    VijR[4] += -(dPermCoef*rotatedDipole2.x + dUIndCoef*qiUindJ.x);
    VjiR[1] += -(dPermCoef*rotatedQuadrupole1[0]);
    Vjid[0] += -(eUIndCoef*rotatedQuadrupole1[0]);

    // D-Q and Uind-Q terms (m=1)
    const real sqrtThree = SQRT((real) 3);
    ePermCoef = -sqrtThree*rInvVec[4]*(mScale + bVec[3]);
    eUIndCoef = -2*sqrtThree*rInvVec[4]*(pScale*thole_q1 + bVec[3]);
    const real fourSqrtOneThird = 4/sqrt((real) 3);
    dPermCoef = fourSqrtOneThird*rInvVec[5]*(1.5f*(mScale + bVec[3]) + 0.5f*alphaRVec[5]*X);
    dUIndCoef = 2*fourSqrtOneThird*rInvVec[5]*(3*(pScale*dthole_q1 + bVec[3]) + alphaRVec[5]*X);
    Vij[2]  += ePermCoef*rotatedQuadrupole2[1];
    Vji[5]   = ePermCoef*rotatedDipole1.y + eUIndCoef*qiUindI.y;
    VijR[2] += dPermCoef*rotatedQuadrupole2[1];
    VjiR[5]  = dPermCoef*rotatedDipole1.y + dUIndCoef*qiUindI.y;
    Vij[3]  += ePermCoef*rotatedQuadrupole2[2];
    Vji[6]   = ePermCoef*rotatedDipole1.z + eUIndCoef*qiUindI.z;
    VijR[3] += dPermCoef*rotatedQuadrupole2[2];
    VjiR[6]  = dPermCoef*rotatedDipole1.z + dUIndCoef*qiUindI.z;
    Vijd[1] += eUIndCoef*rotatedQuadrupole2[1];
    Vijd[2] += eUIndCoef*rotatedQuadrupole2[2];
    // D-Q and Uind-Q terms (m=1)
    Vij[5]   = -(ePermCoef*rotatedDipole2.y + eUIndCoef*qiUindJ.y);
    Vji[2]  += -(ePermCoef*rotatedQuadrupole1[1]);
    VijR[5]  = -(dPermCoef*rotatedDipole2.y + dUIndCoef*qiUindJ.y);
    VjiR[2] += -(dPermCoef*rotatedQuadrupole1[1]);
    Vij[6]   = -(ePermCoef*rotatedDipole2.z + eUIndCoef*qiUindJ.z);
    Vji[3]  += -(ePermCoef*rotatedQuadrupole1[2]);
    VijR[6]  = -(dPermCoef*rotatedDipole2.z + dUIndCoef*qiUindJ.z);
    VjiR[3] += -(dPermCoef*rotatedQuadrupole1[2]);
    Vjid[1] += -(eUIndCoef*rotatedQuadrupole1[1]);
    Vjid[2] += -(eUIndCoef*rotatedQuadrupole1[2]);

    // Q-Q terms (m=0)
    ePermCoef = rInvVec[5]*(6*(mScale + bVec[4]) + ((real) 4/45)*(-3 + 10*alphaRVec[2])*alphaRVec[5]*X);
    dPermCoef = -rInvVec[6]*(135*(mScale + bVec[4]) + 4*(1 + 2*alphaRVec[2])*alphaRVec[7]*X)/9;
    Vij[4]  += ePermCoef*rotatedQuadrupole2[0];
    Vji[4]  += ePermCoef*rotatedQuadrupole1[0];
    VijR[4] += dPermCoef*rotatedQuadrupole2[0];
    VjiR[4] += dPermCoef*rotatedQuadrupole1[0];
    // Q-Q terms (m=1)
    const real fourOverFifteen = (real) 4/15;
    ePermCoef = -fourOverFifteen*rInvVec[5]*(15*(mScale + bVec[4]) + alphaRVec[5]*X);
    dPermCoef = rInvVec[6]*(10*(mScale + bVec[4]) + fourThirds*alphaRVec[7]*X);
    Vij[5]  += ePermCoef*rotatedQuadrupole2[1];
    Vji[5]  += ePermCoef*rotatedQuadrupole1[1];
    VijR[5] += dPermCoef*rotatedQuadrupole2[1];
    VjiR[5] += dPermCoef*rotatedQuadrupole1[1];
    Vij[6]  += ePermCoef*rotatedQuadrupole2[2];
    Vji[6]  += ePermCoef*rotatedQuadrupole1[2];
    VijR[6] += dPermCoef*rotatedQuadrupole2[2];
    VjiR[6] += dPermCoef*rotatedQuadrupole1[2];
    // Q-Q terms (m=2)
    ePermCoef = rInvVec[5]*(mScale + bVec[4] - fourOverFifteen*alphaRVec[5]*X);
    dPermCoef = -2.5f*(mScale + bVec[4])*rInvVec[6];
    Vij[7]  = ePermCoef*rotatedQuadrupole2[3];
    Vji[7]  = ePermCoef*rotatedQuadrupole1[3];
    VijR[7] = dPermCoef*rotatedQuadrupole2[3];
    VjiR[7] = dPermCoef*rotatedQuadrupole1[3];
    Vij[8]  = ePermCoef*rotatedQuadrupole2[4];
    Vji[8]  = ePermCoef*rotatedQuadrupole1[4];
    VijR[8] = dPermCoef*rotatedQuadrupole2[4];
    VjiR[8] = dPermCoef*rotatedQuadrupole1[4];

    // C-O (m=0)
    ePermCoef = rInvVec[4]*(-mScale - bVec[3] - fourOverFifteen*alphaRVec[5]*X);
    dPermCoef = 0.5f*fourOverFifteen*rInvVec[5]*(15*(mScale+bVec[3])+2*(2*alphaRVec[5]+alphaRVec[7])*X);
    Vij[0]  += ePermCoef*rotatedOctopole2[0];
    Vji[9]   = ePermCoef*atom1.q;
    VijR[0] += dPermCoef*rotatedOctopole2[0];
    VjiR[9]  = dPermCoef*atom1.q;
    // O-C (m=0)
    Vij[9]   = -ePermCoef*atom2.q;
    Vji[0]  -=  ePermCoef*rotatedOctopole1[0];
    VijR[9]  = -dPermCoef*atom2.q;
    VjiR[0] -=  dPermCoef*rotatedOctopole1[0];

    // D-O and Uind-O (m=0)
    const real twoOverFifteen = (real) 2/15;
    ePermCoef = -4*rInvVec[5]*(mScale+bVec[4]+twoOverFifteen*alphaRVec[7]*X);
    eUIndCoef = -8*rInvVec[5]*(pScale*thole_o0+bVec[4]+twoOverFifteen*alphaRVec[7]*X);
    dPermCoef = 0.5f*fourOverFifteen*rInvVec[6]*(75*(mScale+bVec[4])+4*(1+alphaRVec[2])*alphaRVec[7]*X);
    dUIndCoef = 2*fourOverFifteen*rInvVec[6]*(75*(pScale*dthole_o0+bVec[4])+4*(1+alphaRVec[2])*alphaRVec[7]*X);
    Vij[1]  += ePermCoef*rotatedOctopole2[0];
    Vji[9]  += ePermCoef*rotatedDipole1.x + eUIndCoef*qiUindI.x;
    VijR[1] += dPermCoef*rotatedOctopole2[0];
    VjiR[9] += dPermCoef*rotatedDipole1.x + dUIndCoef*qiUindI.x;
    // O-D and O-Uind (m=0)
    Vij[9]  += ePermCoef*rotatedDipole2.x + eUIndCoef*qiUindJ.x;
    Vji[1]  += ePermCoef*rotatedOctopole1[0];
    VijR[9] += dPermCoef*rotatedDipole2.x + dUIndCoef*qiUindJ.x;
    VjiR[1] += dPermCoef*rotatedOctopole1[0];
    Vijd[0] += eUIndCoef*rotatedOctopole2[0];
    Vjid[0] += eUIndCoef*rotatedOctopole1[0];
    // D-O and O-Uind (m=1)
    const real sqrtSix = SQRT((real) 6);
    const real sqrtTwoThirds = sqrt((real) 2/3);
    ePermCoef = sqrtSix*(mScale+bVec[4])*rInvVec[5];
    eUIndCoef = 2*sqrtSix*(pScale*thole_o1+bVec[4])*rInvVec[5];
    dPermCoef = -0.1f*sqrtTwoThirds*rInvVec[6]*(75*(mScale+bVec[4])+8*alphaRVec[7]*X);
    dUIndCoef = -0.4f*sqrtTwoThirds*rInvVec[6]*(75*(pScale*dthole_o1+bVec[4])+8*alphaRVec[7]*X);
    Vij[2]   += ePermCoef*rotatedOctopole2[1];
    Vji[10]   = ePermCoef*rotatedDipole1.y + eUIndCoef*qiUindI.y;
    VijR[2]  += dPermCoef*rotatedOctopole2[1];
    VjiR[10]  = dPermCoef*rotatedDipole1.y + dUIndCoef*qiUindI.y;
    Vij[3]   += ePermCoef*rotatedOctopole2[2];
    Vji[11]   = ePermCoef*rotatedDipole1.z + eUIndCoef*qiUindI.z;
    VijR[3]  += dPermCoef*rotatedOctopole2[2];
    VjiR[11]  = dPermCoef*rotatedDipole1.z + dUIndCoef*qiUindI.z;
    Vijd[1] += eUIndCoef*rotatedOctopole2[1];
    Vijd[2] += eUIndCoef*rotatedOctopole2[2];
    // O-D and O-Uind (m=1)
    Vij[10]   = ePermCoef*rotatedDipole2.y + eUIndCoef*qiUindJ.y;
    Vji[2]   += ePermCoef*rotatedOctopole1[1];
    VijR[10]  = dPermCoef*rotatedDipole2.y + dUIndCoef*qiUindJ.y;
    VjiR[2]  += dPermCoef*rotatedOctopole1[1];
    Vij[11]   = ePermCoef*rotatedDipole2.z + eUIndCoef*qiUindJ.z;
    Vji[3]   += ePermCoef*rotatedOctopole1[2];
    VijR[11]  = dPermCoef*rotatedDipole2.z + dUIndCoef*qiUindJ.z;
    VjiR[3]  += dPermCoef*rotatedOctopole1[2];
    Vjid[1] += eUIndCoef*rotatedOctopole1[1];
    Vjid[2] += eUIndCoef*rotatedOctopole1[2];

    // Q-O (m=0)
    const real fourOverFortyFive = (real) 4/45;
    ePermCoef = rInvVec[6]*(-10*(mScale+bVec[4]) - 2*fourOverFortyFive*(3+2*alphaRVec[2])*alphaRVec[7]*X);
    dPermCoef = 0.5f*fourOverFortyFive*rInvVec[7]*(675*(mScale+bVec[4])+2*(27+4*alphaRVec[4])*alphaRVec[7]*X);
    Vij[4]  += ePermCoef*rotatedOctopole2[0];
    Vji[9]  += ePermCoef*rotatedQuadrupole1[0];
    VijR[4] += dPermCoef*rotatedOctopole2[0];
    VjiR[9] += dPermCoef*rotatedQuadrupole1[0];
    // O-Q (m=0)
    Vij[9]  -= ePermCoef*rotatedQuadrupole2[0];
    Vji[4]  -= ePermCoef*rotatedOctopole1[0];
    VijR[9] -= dPermCoef*rotatedQuadrupole2[0];
    VjiR[4] -= dPermCoef*rotatedOctopole1[0];
    // Q-O (m=1)
    const real sqrtTwo = SQRT((real) 2);
    const real sqrtEightOverFifteen = SQRT((real) 8)/15;
    const real eightOverSeventyFive = (real) 8/75;
    ePermCoef = 5*sqrtTwo*rInvVec[6]*(mScale+bVec[4] + eightOverSeventyFive*alphaRVec[7]*X);
    dPermCoef = -0.5f*sqrtEightOverFifteen*rInvVec[7]*(225*(mScale+bVec[4])+8*(2+alphaRVec[2])*alphaRVec[7]*X);
    Vij[5]   += ePermCoef*rotatedOctopole2[1];
    Vji[10]  += ePermCoef*rotatedQuadrupole1[1];
    VijR[5]  += dPermCoef*rotatedOctopole2[1];
    VjiR[10] += dPermCoef*rotatedQuadrupole1[1];
    Vij[6]   += ePermCoef*rotatedOctopole2[2];
    Vji[11]  += ePermCoef*rotatedQuadrupole1[2];
    VijR[6]  += dPermCoef*rotatedOctopole2[2];
    VjiR[11] += dPermCoef*rotatedQuadrupole1[2];
    // O-Q (m=1)
    Vij[10]  -= ePermCoef*rotatedQuadrupole2[1];
    Vji[5]   -= ePermCoef*rotatedOctopole1[1];
    VijR[10] -= dPermCoef*rotatedQuadrupole2[1];
    VjiR[5]  -= dPermCoef*rotatedOctopole1[1];
    Vij[11]  -= ePermCoef*rotatedQuadrupole2[2];
    Vji[6]   -= ePermCoef*rotatedOctopole1[2];
    VijR[11] -= dPermCoef*rotatedQuadrupole2[2];
    VjiR[6]  -= dPermCoef*rotatedOctopole1[2];
    // Q-O (m=2)
    const real sqrtFive = SQRT((real) 5);
    const real twoSqrtFiveOverFifteen = 2*SQRT((real) 5)/15;
    ePermCoef = -sqrtFive*(mScale+bVec[4])*rInvVec[6];
    dPermCoef = 0.5f*twoSqrtFiveOverFifteen*rInvVec[7]*(45*(mScale+bVec[4])+4*alphaRVec[7]*X);
    Vij[7]  += ePermCoef*rotatedOctopole2[3];
    Vji[12]  = ePermCoef*rotatedQuadrupole1[3];
    VijR[7] += dPermCoef*rotatedOctopole2[3];
    VjiR[12] = dPermCoef*rotatedQuadrupole1[3];
    Vij[8]  += ePermCoef*rotatedOctopole2[4];
    Vji[13]  = ePermCoef*rotatedQuadrupole1[4];
    VijR[8] += dPermCoef*rotatedOctopole2[4];
    VjiR[13] = dPermCoef*rotatedQuadrupole1[4];
    // O-Q (m=2)
    Vij[12]  = -ePermCoef*rotatedQuadrupole2[3];
    Vji[7]  -=  ePermCoef*rotatedOctopole1[3];
    VijR[12] = -dPermCoef*rotatedQuadrupole2[3];
    VjiR[7] -=  dPermCoef*rotatedOctopole1[3];
    Vij[13]  = -ePermCoef*rotatedQuadrupole2[4];
    Vji[8]  -=  ePermCoef*rotatedOctopole1[4];
    VijR[13] = -dPermCoef*rotatedQuadrupole2[4];
    VjiR[8] -=  dPermCoef*rotatedOctopole1[4];

    // O-O (m=0)
    const real eightOverOneFiveSevenFive = (real) 8/1575;
    const real fourOverTwoTwoFive = (real) 4/225;
    ePermCoef = rInvVec[7]*(-20*(mScale+bVec[5]) - eightOverOneFiveSevenFive*(15+28*alphaRVec[2]+28*alphaRVec[4])*alphaRVec[7]*X);
    dPermCoef = 0.5f*fourOverTwoTwoFive*rInvVec[8]*(7875*(mScale+bVec[5])+4*(41 - 4*alphaRVec[2]+4*alphaRVec[4])*alphaRVec[9]*X);
    Vij[9]  += ePermCoef*rotatedOctopole2[0];
    Vji[9]  += ePermCoef*rotatedOctopole1[0];
    VijR[9] += dPermCoef*rotatedOctopole2[0];
    VjiR[9] += dPermCoef*rotatedOctopole1[0];
    // O-O (m=1)
    const real twoOverOneFiveZero = (real) 2/150;
    const real eightOverFiveTwoFive = (real) 8/525;
    ePermCoef = rInvVec[7]*(15*(mScale+bVec[5]) + eightOverFiveTwoFive*(-5 + 28*alphaRVec[2])* alphaRVec[7]*X);
    dPermCoef = -0.5f*twoOverOneFiveZero*rInvVec[8]*(7875*(mScale+bVec[5]) + 32*(3 + 2*alphaRVec[2])*alphaRVec[9]*X);
    Vij[10]  += ePermCoef*rotatedOctopole2[1];
    Vji[10]  += ePermCoef*rotatedOctopole1[1];
    VijR[10] += dPermCoef*rotatedOctopole2[1];
    VjiR[10] += dPermCoef*rotatedOctopole1[1];
    Vij[11]  += ePermCoef*rotatedOctopole2[2];
    Vji[11]  += ePermCoef*rotatedOctopole1[2];
    VijR[11] += dPermCoef*rotatedOctopole2[2];
    VjiR[11] += dPermCoef*rotatedOctopole1[2];
    // O-O (m=2)
    const real eightOverOneHundredFive = (real) 8/105;
    const real sixteenOverFifteen = (real) 16/15;
    ePermCoef = rInvVec[7]*(-6*(mScale+bVec[5]) - eightOverOneHundredFive*alphaRVec[7]*X);
    dPermCoef = 0.5f*rInvVec[8]*(42*(mScale+bVec[5]) + sixteenOverFifteen*alphaRVec[9]*X);
    Vij[12]  += ePermCoef*rotatedOctopole2[3];
    Vji[12]  += ePermCoef*rotatedOctopole1[3];
    VijR[12] += dPermCoef*rotatedOctopole2[3];
    VjiR[12] += dPermCoef*rotatedOctopole1[3];
    Vij[13]  += ePermCoef*rotatedOctopole2[4];
    Vji[13]  += ePermCoef*rotatedOctopole1[4];
    VijR[13] += dPermCoef*rotatedOctopole2[4];
    VjiR[13] += dPermCoef*rotatedOctopole1[4];
    // O-O (m=3)
    ePermCoef = rInvVec[7]*((mScale+bVec[5]) - eightOverOneHundredFive*alphaRVec[7]*X);
    dPermCoef = -0.5f*7*(mScale+bVec[5])*rInvVec[8];
    Vij[14]  = ePermCoef*rotatedOctopole2[5];
    Vji[14]  = ePermCoef*rotatedOctopole1[5];
    VijR[14] = dPermCoef*rotatedOctopole2[5];
    VjiR[14] = dPermCoef*rotatedOctopole1[5];
    Vij[15]  = ePermCoef*rotatedOctopole2[6];
    Vji[15]  = ePermCoef*rotatedOctopole1[6];
    VijR[15] = dPermCoef*rotatedOctopole2[6];
    VjiR[15] = dPermCoef*rotatedOctopole1[6];

    // Evaluate the energies, forces and torques due to permanent+induced moments
    // interacting with just the permanent moments.
    energy += forceFactor*0.5f*(
        atom1.q*Vij[0]
              + rotatedDipole1.x*Vij[1] + rotatedDipole1.y*Vij[2] + rotatedDipole1.z*Vij[3]
              + rotatedQuadrupole1[0]*Vij[4] + rotatedQuadrupole1[1]*Vij[5] + rotatedQuadrupole1[2]*Vij[6] + rotatedQuadrupole1[3]*Vij[7] + rotatedQuadrupole1[4]*Vij[8]
              + rotatedOctopole1[0]*Vij[9] + rotatedOctopole1[1]*Vij[10] + rotatedOctopole1[2]*Vij[11] + rotatedOctopole1[3]*Vij[12]
              + rotatedOctopole1[4]*Vij[13] + rotatedOctopole1[5]*Vij[14] + rotatedOctopole1[6]*Vij[15]
      + atom2.q*Vji[0]
              + rotatedDipole2.x*Vji[1] + rotatedDipole2.y*Vji[2] + rotatedDipole2.z*Vji[3]
              + rotatedQuadrupole2[0]*Vji[4] + rotatedQuadrupole2[1]*Vji[5] + rotatedQuadrupole2[2]*Vji[6] + rotatedQuadrupole2[3]*Vji[7] + rotatedQuadrupole2[4]*Vji[8]
              + rotatedOctopole2[0]*Vji[9] + rotatedOctopole2[1]*Vji[10] + rotatedOctopole2[2]*Vji[11] + rotatedOctopole2[3]*Vji[12]
              + rotatedOctopole2[4]*Vji[13] + rotatedOctopole2[5]*Vji[14] + rotatedOctopole2[6]*Vji[15]
    );
    real fIZ = atom1.q*VijR[0]
             + rotatedDipole1.x*VijR[1] + rotatedDipole1.y*VijR[2] + rotatedDipole1.z*VijR[3]
             + rotatedQuadrupole1[0]*VijR[4] + rotatedQuadrupole1[1]*VijR[5] + rotatedQuadrupole1[2]*VijR[6] + rotatedQuadrupole1[3]*VijR[7] + rotatedQuadrupole1[4]*VijR[8]
             + rotatedOctopole1[0]*VijR[9] + rotatedOctopole1[1]*VijR[10] + rotatedOctopole1[2]*VijR[11] + rotatedOctopole1[3]*VijR[12]
             + rotatedOctopole1[4]*VijR[13] + rotatedOctopole1[5]*VijR[14] + rotatedOctopole1[6]*VijR[15]
             ;
    real fJZ = atom2.q*VjiR[0]
             + rotatedDipole2.x*VjiR[1] + rotatedDipole2.y*VjiR[2] + rotatedDipole2.z*VjiR[3]
             + rotatedQuadrupole2[0]*VjiR[4] + rotatedQuadrupole2[1]*VjiR[5] + rotatedQuadrupole2[2]*VjiR[6] + rotatedQuadrupole2[3]*VjiR[7] + rotatedQuadrupole2[4]*VjiR[8]
             + rotatedOctopole2[0]*VjiR[9] + rotatedOctopole2[1]*VjiR[10] + rotatedOctopole2[2]*VjiR[11] + rotatedOctopole2[3]*VjiR[12]
             + rotatedOctopole2[4]*VjiR[13] + rotatedOctopole2[5]*VjiR[14] + rotatedOctopole2[6]*VjiR[15]
             ;

    const real sqrtThreeHalves = SQRT((real)1.5f);
    const real sqrtFiveHalves = SQRT((real)2.5f);
    real EIX = 0 
             + rotatedDipole1.z*Vij[1]
             - rotatedDipole1.x*Vij[3]
             + sqrtThree*rotatedQuadrupole1[2]*Vij[4]
             + rotatedQuadrupole1[4]*Vij[5]
             - (sqrtThree*rotatedQuadrupole1[0]+rotatedQuadrupole1[3])*Vij[6]
             + rotatedQuadrupole1[2]*Vij[7]
             - rotatedQuadrupole1[1]*Vij[8]
             + sqrtSix*rotatedOctopole1[2]*Vij[9]
             + sqrtFiveHalves*rotatedOctopole1[4]*Vij[10]
             - (sqrtSix*rotatedOctopole1[0]+sqrtFiveHalves*rotatedOctopole1[3])*Vij[11]
             + (sqrtFiveHalves*rotatedOctopole1[2]+sqrtThreeHalves*rotatedOctopole1[6])*Vij[12]
             - (sqrtFiveHalves*rotatedOctopole1[1]+sqrtThreeHalves*rotatedOctopole1[5])*Vij[13]
             + sqrtThreeHalves*rotatedOctopole1[4]*Vij[14]
             - sqrtThreeHalves*rotatedOctopole1[3]*Vij[15]
             ;
    real EIY = 0
             - rotatedDipole1.y*Vij[1]
             + rotatedDipole1.x*Vij[2]
             - sqrtThree*rotatedQuadrupole1[1]*Vij[4]
             + (sqrtThree*rotatedQuadrupole1[0]-rotatedQuadrupole1[3])*Vij[5]
             - rotatedQuadrupole1[4]*Vij[6] 
             + rotatedQuadrupole1[1]*Vij[7] 
             + rotatedQuadrupole1[2]*Vij[8]
             - sqrtSix*rotatedOctopole1[1]*Vij[9]
             + (sqrtSix*rotatedOctopole1[0]-sqrtFiveHalves*rotatedOctopole1[3])*Vij[10]
             - sqrtFiveHalves*rotatedOctopole1[4]*Vij[11]
             + (sqrtFiveHalves*rotatedOctopole1[1]-sqrtThreeHalves*rotatedOctopole1[5])*Vij[12]
             + (sqrtFiveHalves*rotatedOctopole1[2]-sqrtThreeHalves*rotatedOctopole1[6])*Vij[13]
             + sqrtThreeHalves*rotatedOctopole1[3]*Vij[14]
             + sqrtThreeHalves*rotatedOctopole1[4]*Vij[15]
             ;
    real EIZ = 0
             - rotatedDipole1.z*Vij[2]
             + rotatedDipole1.y*Vij[3] 
             - rotatedQuadrupole1[2]*Vij[5] 
             + rotatedQuadrupole1[1]*Vij[6] 
             - 2*rotatedQuadrupole1[4]*Vij[7] 
             + 2*rotatedQuadrupole1[3]*Vij[8]
             - rotatedOctopole1[2]*Vij[10]
             + rotatedOctopole1[1]*Vij[11]
             - 2*rotatedOctopole1[4]*Vij[12]
             + 2*rotatedOctopole1[3]*Vij[13]
             - 3*rotatedOctopole1[6]*Vij[14]
             + 3*rotatedOctopole1[5]*Vij[15]
             ;
    real EJX = 0
             + rotatedDipole2.z*Vji[1]
             - rotatedDipole2.x*Vji[3]
             + sqrtThree*rotatedQuadrupole2[2]*Vji[4]
             + rotatedQuadrupole2[4]*Vji[5]
             - (sqrtThree*rotatedQuadrupole2[0]+rotatedQuadrupole2[3])*Vji[6]
             + rotatedQuadrupole2[2]*Vji[7]
             - rotatedQuadrupole2[1]*Vji[8]
             + sqrtSix*rotatedOctopole2[2]*Vji[9]
             + sqrtFiveHalves*rotatedOctopole2[4]*Vji[10]
             - (sqrtSix*rotatedOctopole2[0]+sqrtFiveHalves*rotatedOctopole2[3])*Vji[11]
             + (sqrtFiveHalves*rotatedOctopole2[2]+sqrtThreeHalves*rotatedOctopole2[6])*Vji[12]
             - (sqrtFiveHalves*rotatedOctopole2[1]+sqrtThreeHalves*rotatedOctopole2[5])*Vji[13]
             + sqrtThreeHalves*rotatedOctopole2[4]*Vji[14]
             - sqrtThreeHalves*rotatedOctopole2[3]*Vji[15]
             ;
    real EJY = 0
             - rotatedDipole2.y*Vji[1]
             + rotatedDipole2.x*Vji[2]
             - sqrtThree*rotatedQuadrupole2[1]*Vji[4]
             + (sqrtThree*rotatedQuadrupole2[0]-rotatedQuadrupole2[3])*Vji[5]
             - rotatedQuadrupole2[4]*Vji[6]
             + rotatedQuadrupole2[1]*Vji[7]
             + rotatedQuadrupole2[2]*Vji[8]
             - sqrtSix*rotatedOctopole2[1]*Vji[9]
             + (sqrtSix*rotatedOctopole2[0]-sqrtFiveHalves*rotatedOctopole2[3])*Vji[10]
             - sqrtFiveHalves*rotatedOctopole2[4]*Vji[11]
             + (sqrtFiveHalves*rotatedOctopole2[1]-sqrtThreeHalves*rotatedOctopole2[5])*Vji[12]
             + (sqrtFiveHalves*rotatedOctopole2[2]-sqrtThreeHalves*rotatedOctopole2[6])*Vji[13]
             + sqrtThreeHalves*rotatedOctopole2[3]*Vji[14]
             + sqrtThreeHalves*rotatedOctopole2[4]*Vji[15]
             ;
    real EJZ = 0
             - rotatedDipole2.z*Vji[2]
             + rotatedDipole2.y*Vji[3]
             - rotatedQuadrupole2[2]*Vji[5]
             + rotatedQuadrupole2[1]*Vji[6]
             - 2*rotatedQuadrupole2[4]*Vji[7]
             + 2*rotatedQuadrupole2[3]*Vji[8]
             - rotatedOctopole2[2]*Vji[10]
             + rotatedOctopole2[1]*Vji[11]
             - 2*rotatedOctopole2[4]*Vji[12]
             + 2*rotatedOctopole2[3]*Vji[13]
             - 3*rotatedOctopole2[6]*Vji[14]
             + 3*rotatedOctopole2[5]*Vji[15]
             ;
    // Define the torque intermediates for the induced dipoles. These are simply the induced dipole torque
    // intermediates dotted with the field due to permanent moments only, at each center. We inline the
    // induced dipole torque intermediates here, for simplicity. N.B. There are no torques on the dipoles
    // themselves, so we accumulate the torque intermediates into separate variables to allow them to be
    // used only in the force calculation.
    //
    // The torque about the x axis (needed to obtain the y force on the induced dipoles, below)
    //    qiUindIx[0] = qiQUindI[2];    qiUindIx[1] = 0;    qiUindIx[2] = -qiQUindI[0]
    real iEIX = qiUindI.z*Vijd[0] - qiUindI.x*Vijd[2];
    real iEJX = qiUindJ.z*Vjid[0] - qiUindJ.x*Vjid[2];
    // The torque about the y axis (needed to obtain the x force on the induced dipoles, below)
    //    qiUindIy[0] = -qiQUindI[1];   qiUindIy[1] = qiQUindI[0];    qiUindIy[2] = 0
    real iEIY = qiUindI.x*Vijd[1] - qiUindI.y*Vijd[0];
    real iEJY = qiUindJ.x*Vjid[1] - qiUindJ.y*Vjid[0];
    // The torque about the z axis (needed to obtain the x force on the induced dipoles, below)
    //    qiUindIz[0] = 0;  qiUindIz[1] = -qiQUindI[2];    qiUindIz[2] = qiQUindI[1]
    real iEIZ = qiUindI.y*Vijd[2] - qiUindI.z*Vijd[1];
    real iEJZ = qiUindJ.y*Vjid[2] - qiUindJ.z*Vjid[1];
#ifdef MUTUAL_POLARIZATION
    // Uind-Uind terms (m=0)
    real eCoef = -2*fourThirds*rInvVec[3]*(3*(thole_d0 + bVec[3]) + alphaRVec[3]*X);
    real dCoef = 2*rInvVec[4]*(6*(dthole_d0 + bVec[3]) + 4*alphaRVec[5]*X);
    iEIX += eCoef*qiUindI.z*qiUindJ.x;
    iEJX += eCoef*qiUindJ.z*qiUindI.x;
    iEIY -= eCoef*qiUindI.y*qiUindJ.x;
    iEJY -= eCoef*qiUindJ.y*qiUindI.x;
    fIZ  += dCoef*qiUindI.x*qiUindJ.x;
    fJZ  += dCoef*qiUindJ.x*qiUindI.x;
    // Uind-Uind terms (m=1)
    eCoef = 4*rInvVec[3]*(thole_d1 + bVec[3] - twoThirds*alphaRVec[3]*X);
    dCoef = -6*rInvVec[4]*(dthole_d1 + bVec[3]);
    iEIX -= eCoef*qiUindI.x*qiUindJ.z;
    iEJX -= eCoef*qiUindJ.x*qiUindI.z;
    iEIY += eCoef*qiUindI.x*qiUindJ.y;
    iEJY += eCoef*qiUindJ.x*qiUindI.y;
    iEIZ += eCoef*qiUindI.y*qiUindJ.z;
    iEJZ += eCoef*qiUindJ.y*qiUindI.z;
    fIZ  += dCoef*(qiUindI.y*qiUindJ.y + qiUindI.z*qiUindJ.z);
    fJZ  += dCoef*(qiUindJ.y*qiUindI.y + qiUindJ.z*qiUindI.z);
#endif

    // The quasi-internal frame forces and torques.  Note that the induced torque intermediates are
    // used in the force expression, but not in the torques; the induced dipoles are isotropic.
    // The quasi-internal frame forces and torques.  Note that the induced torque intermediates are
    // used in the force expression, but not in the torques; the induced dipoles are isotropic.
    real qiForce[3] = {rInv*(EIY+EJY+iEIY+iEJY), -rInv*(EIX+EJX+iEIX+iEJX), -(fJZ+fIZ)};
    real qiTorqueI[3] = {-EIX, -EIY, -EIZ};
    real qiTorqueJ[3] = {-EJX, -EJY, -EJZ};

    if(atom1.damp > 0){
        qiTorqueI[0] += -iEIX;
        qiTorqueI[1] += -iEIY;
        qiTorqueI[2] += -iEIZ;
    }
    if(atom2.damp > 0){
        qiTorqueJ[0] += -iEJX;
        qiTorqueJ[1] += -iEJY;
        qiTorqueJ[2] += -iEJZ;
    }


    real3 force = make_real3(qiRotationMatrix[1][1]*qiForce[0] + qiRotationMatrix[2][1]*qiForce[1] + qiRotationMatrix[0][1]*qiForce[2],
                             qiRotationMatrix[1][2]*qiForce[0] + qiRotationMatrix[2][2]*qiForce[1] + qiRotationMatrix[0][2]*qiForce[2],
                             qiRotationMatrix[1][0]*qiForce[0] + qiRotationMatrix[2][0]*qiForce[1] + qiRotationMatrix[0][0]*qiForce[2]);
    atom1.force += force;
    atom1.torque += make_real3(qiRotationMatrix[1][1]*qiTorqueI[0] + qiRotationMatrix[2][1]*qiTorqueI[1] + qiRotationMatrix[0][1]*qiTorqueI[2],
                               qiRotationMatrix[1][2]*qiTorqueI[0] + qiRotationMatrix[2][2]*qiTorqueI[1] + qiRotationMatrix[0][2]*qiTorqueI[2],
                               qiRotationMatrix[1][0]*qiTorqueI[0] + qiRotationMatrix[2][0]*qiTorqueI[1] + qiRotationMatrix[0][0]*qiTorqueI[2]);
    if (forceFactor == 1) {
        atom2.force -= force;
        atom2.torque += make_real3(qiRotationMatrix[1][1]*qiTorqueJ[0] + qiRotationMatrix[2][1]*qiTorqueJ[1] + qiRotationMatrix[0][1]*qiTorqueJ[2],
                                   qiRotationMatrix[1][2]*qiTorqueJ[0] + qiRotationMatrix[2][2]*qiTorqueJ[1] + qiRotationMatrix[0][2]*qiTorqueJ[2],
                                   qiRotationMatrix[1][0]*qiTorqueJ[0] + qiRotationMatrix[2][0]*qiTorqueJ[1] + qiRotationMatrix[0][0]*qiTorqueJ[2]);
    }
}

/**
 * Compute the self energy and self torque.
 */
__device__ void computeSelfEnergyAndTorque(AtomData& atom1, mixed& energy, const real3& cphi) {
    real cii = atom1.q*atom1.q;
    real3 dipole = make_real3(atom1.sphericalDipole.y, atom1.sphericalDipole.z, atom1.sphericalDipole.x);
    real dii = dot(dipole, dipole+atom1.inducedDipole);
#ifdef INCLUDE_QUADRUPOLES
    real qii = (atom1.sphericalQuadrupole[0]*atom1.sphericalQuadrupole[0] +
                atom1.sphericalQuadrupole[1]*atom1.sphericalQuadrupole[1] +
                atom1.sphericalQuadrupole[2]*atom1.sphericalQuadrupole[2] +
                atom1.sphericalQuadrupole[3]*atom1.sphericalQuadrupole[3] +
                atom1.sphericalQuadrupole[4]*atom1.sphericalQuadrupole[4]);
#else
    real qii = 0;
#endif
#ifdef INCLUDE_OCTOPOLES
    real oii = (atom1.sphericalOctopole[0]*atom1.sphericalOctopole[0] +
                atom1.sphericalOctopole[1]*atom1.sphericalOctopole[1] +
                atom1.sphericalOctopole[2]*atom1.sphericalOctopole[2] +
                atom1.sphericalOctopole[3]*atom1.sphericalOctopole[3] +
                atom1.sphericalOctopole[4]*atom1.sphericalOctopole[4] +
                atom1.sphericalOctopole[5]*atom1.sphericalOctopole[5] +
                atom1.sphericalOctopole[6]*atom1.sphericalOctopole[6]);
#else
    real oii = 0;
#endif
    real prefac = -EWALD_ALPHA/SQRT_PI;
    real a2 = EWALD_ALPHA*EWALD_ALPHA;
    real a4 = a2*a2;
    real a6 = a4*a2;
    energy += prefac*(cii + ((real)2/3)*a2*dii + ((real) 4/15)*a4*qii + ((real) 8/105)*a6*oii);

    // self-torque for PME
    real3 ui = atom1.inducedDipole;
    bool isIsotropic = atom1.damp < 0;
    prefac = isIsotropic ? 4*a2*EWALD_ALPHA/((real) 3*SQRT_PI) : 0;
    atom1.torque += prefac*cross(dipole, ui);
    // The interaction of induced dipoles with rec-space permanent field
    prefac = isIsotropic ? 0 : 1;
    atom1.torque += prefac*cross(cphi, ui);
}

/**
 * Compute electrostatic interactions.
 */
extern "C" __global__ void computeElectrostatics(
        unsigned long long* __restrict__ forceBuffers, unsigned long long* __restrict__ torqueBuffers, mixed* __restrict__ energyBuffer,
        const real4* __restrict__ posq, const uint2* __restrict__ covalentFlags,
        const int2* __restrict__ exclusionTiles, unsigned int startTileIndex, unsigned int numTileIndices,
#ifdef USE_CUTOFF
        const int* __restrict__ tiles, const unsigned int* __restrict__ interactionCount, real4 periodicBoxSize, real4 invPeriodicBoxSize,
        real4 periodicBoxVecX, real4 periodicBoxVecY, real4 periodicBoxVecZ, unsigned int maxTiles, const real4* __restrict__ blockCenter,
        const unsigned int* __restrict__ interactingAtoms,
#endif
        const real* __restrict__ sphericalDipole, const real* __restrict__ sphericalQuadrupole, const real* __restrict__ sphericalOctopole,
        const real* __restrict__ inducedDipole, const float2* __restrict__ dampingAndThole, const real* __restrict__ cphi) {
    const unsigned int totalWarps = (blockDim.x*gridDim.x)/TILE_SIZE;
    const unsigned int warp = (blockIdx.x*blockDim.x+threadIdx.x)/TILE_SIZE;
    const unsigned int tgx = threadIdx.x & (TILE_SIZE-1);
    const unsigned int tbx = threadIdx.x - tgx;
    mixed energy = 0;
    __shared__ AtomData localData[THREAD_BLOCK_SIZE];

    // First loop: process tiles that contain exclusions.
    
    const unsigned int firstExclusionTile = FIRST_EXCLUSION_TILE+warp*(LAST_EXCLUSION_TILE-FIRST_EXCLUSION_TILE)/totalWarps;
    const unsigned int lastExclusionTile = FIRST_EXCLUSION_TILE+(warp+1)*(LAST_EXCLUSION_TILE-FIRST_EXCLUSION_TILE)/totalWarps;
    for (int pos = firstExclusionTile; pos < lastExclusionTile; pos++) {
        const int2 tileIndices = exclusionTiles[pos];
        const unsigned int x = tileIndices.x;
        const unsigned int y = tileIndices.y;
        AtomData data;
        unsigned int atom1 = x*TILE_SIZE + tgx;
        loadAtomData(data, atom1, posq, sphericalDipole, sphericalQuadrupole, sphericalOctopole, inducedDipole, dampingAndThole);
        data.force = make_real3(0);
        data.torque = make_real3(0);
        uint2 covalent = covalentFlags[pos*TILE_SIZE+tgx];
        if (x == y) {
            // This tile is on the diagonal.

            localData[threadIdx.x].pos = data.pos;
            localData[threadIdx.x].q = data.q;
            localData[threadIdx.x].sphericalDipole = data.sphericalDipole;
#ifdef INCLUDE_QUADRUPOLES
            localData[threadIdx.x].sphericalQuadrupole[0] = data.sphericalQuadrupole[0];
            localData[threadIdx.x].sphericalQuadrupole[1] = data.sphericalQuadrupole[1];
            localData[threadIdx.x].sphericalQuadrupole[2] = data.sphericalQuadrupole[2];
            localData[threadIdx.x].sphericalQuadrupole[3] = data.sphericalQuadrupole[3];
            localData[threadIdx.x].sphericalQuadrupole[4] = data.sphericalQuadrupole[4];
#endif
#ifdef INCLUDE_OCTOPOLES
            localData[threadIdx.x].sphericalOctopole[0] = data.sphericalOctopole[0];
            localData[threadIdx.x].sphericalOctopole[1] = data.sphericalOctopole[1];
            localData[threadIdx.x].sphericalOctopole[2] = data.sphericalOctopole[2];
            localData[threadIdx.x].sphericalOctopole[3] = data.sphericalOctopole[3];
            localData[threadIdx.x].sphericalOctopole[4] = data.sphericalOctopole[4];
            localData[threadIdx.x].sphericalOctopole[5] = data.sphericalOctopole[5];
            localData[threadIdx.x].sphericalOctopole[6] = data.sphericalOctopole[6];
#endif
            localData[threadIdx.x].inducedDipole = data.inducedDipole;
            localData[threadIdx.x].thole = data.thole;
            localData[threadIdx.x].damp = data.damp;

            // Compute forces.

            for (unsigned int j = 0; j < TILE_SIZE; j++) {
                int atom2 = y*TILE_SIZE+j;
                if (atom1 != atom2 && atom1 < NUM_ATOMS && atom2 < NUM_ATOMS) {
                    float m = computeMScaleFactor(covalent, j);
                    float p = m;
                    computeOneInteraction(data, localData[tbx+j], true, p, m, 0.5f, energy, periodicBoxSize, invPeriodicBoxSize, periodicBoxVecX, periodicBoxVecY, periodicBoxVecZ);
                }
            }
            if (atom1 < NUM_ATOMS)
                computeSelfEnergyAndTorque(data, energy, make_real3(cphi[20*atom1+1], cphi[20*atom1+2], cphi[20*atom1+3]));
            data.force *= -ENERGY_SCALE_FACTOR;
            data.torque *= ENERGY_SCALE_FACTOR;
            atomicAdd(&forceBuffers[atom1], static_cast<unsigned long long>((long long) (data.force.x*0x100000000)));
            atomicAdd(&forceBuffers[atom1+PADDED_NUM_ATOMS], static_cast<unsigned long long>((long long) (data.force.y*0x100000000)));
            atomicAdd(&forceBuffers[atom1+2*PADDED_NUM_ATOMS], static_cast<unsigned long long>((long long) (data.force.z*0x100000000)));
            atomicAdd(&torqueBuffers[atom1], static_cast<unsigned long long>((long long) (data.torque.x*0x100000000)));
            atomicAdd(&torqueBuffers[atom1+PADDED_NUM_ATOMS], static_cast<unsigned long long>((long long) (data.torque.y*0x100000000)));
            atomicAdd(&torqueBuffers[atom1+2*PADDED_NUM_ATOMS], static_cast<unsigned long long>((long long) (data.torque.z*0x100000000)));
        }
        else {
            // This is an off-diagonal tile.

            unsigned int j = y*TILE_SIZE + tgx;
            loadAtomData(localData[threadIdx.x], j, posq, sphericalDipole, sphericalQuadrupole, sphericalOctopole, inducedDipole, dampingAndThole);
            localData[threadIdx.x].force = make_real3(0);
            localData[threadIdx.x].torque = make_real3(0);
            unsigned int tj = tgx;
            for (j = 0; j < TILE_SIZE; j++) {
                int atom2 = y*TILE_SIZE+tj;
                if (atom1 < NUM_ATOMS && atom2 < NUM_ATOMS) {
                    float m = computeMScaleFactor(covalent, tj);
                    float p = m;
                    computeOneInteraction(data, localData[tbx+tj], true, p, m, 1, energy, periodicBoxSize, invPeriodicBoxSize, periodicBoxVecX, periodicBoxVecY, periodicBoxVecZ);
                }
                tj = (tj + 1) & (TILE_SIZE - 1);
            }
            data.force *= -ENERGY_SCALE_FACTOR;
            data.torque *= ENERGY_SCALE_FACTOR;
            localData[threadIdx.x].force *= -ENERGY_SCALE_FACTOR;
            localData[threadIdx.x].torque *= ENERGY_SCALE_FACTOR;
            unsigned int offset = x*TILE_SIZE + tgx;
            atomicAdd(&forceBuffers[offset], static_cast<unsigned long long>((long long) (data.force.x*0x100000000)));
            atomicAdd(&forceBuffers[offset+PADDED_NUM_ATOMS], static_cast<unsigned long long>((long long) (data.force.y*0x100000000)));
            atomicAdd(&forceBuffers[offset+2*PADDED_NUM_ATOMS], static_cast<unsigned long long>((long long) (data.force.z*0x100000000)));
            atomicAdd(&torqueBuffers[offset], static_cast<unsigned long long>((long long) (data.torque.x*0x100000000)));
            atomicAdd(&torqueBuffers[offset+PADDED_NUM_ATOMS], static_cast<unsigned long long>((long long) (data.torque.y*0x100000000)));
            atomicAdd(&torqueBuffers[offset+2*PADDED_NUM_ATOMS], static_cast<unsigned long long>((long long) (data.torque.z*0x100000000)));
            offset = y*TILE_SIZE + tgx;
            atomicAdd(&forceBuffers[offset], static_cast<unsigned long long>((long long) (localData[threadIdx.x].force.x*0x100000000)));
            atomicAdd(&forceBuffers[offset+PADDED_NUM_ATOMS], static_cast<unsigned long long>((long long) (localData[threadIdx.x].force.y*0x100000000)));
            atomicAdd(&forceBuffers[offset+2*PADDED_NUM_ATOMS], static_cast<unsigned long long>((long long) (localData[threadIdx.x].force.z*0x100000000)));
            atomicAdd(&torqueBuffers[offset], static_cast<unsigned long long>((long long) (localData[threadIdx.x].torque.x*0x100000000)));
            atomicAdd(&torqueBuffers[offset+PADDED_NUM_ATOMS], static_cast<unsigned long long>((long long) (localData[threadIdx.x].torque.y*0x100000000)));
            atomicAdd(&torqueBuffers[offset+2*PADDED_NUM_ATOMS], static_cast<unsigned long long>((long long) (localData[threadIdx.x].torque.z*0x100000000)));
        }
    }

    // Second loop: tiles without exclusions, either from the neighbor list (with cutoff) or just enumerating all
    // of them (no cutoff).

#ifdef USE_CUTOFF
    const unsigned int numTiles = interactionCount[0];
    if (numTiles > maxTiles)
        return; // There wasn't enough memory for the neighbor list.
    int pos = (int) (numTiles > maxTiles ? startTileIndex+warp*(long long)numTileIndices/totalWarps : warp*(long long)numTiles/totalWarps);
    int end = (int) (numTiles > maxTiles ? startTileIndex+(warp+1)*(long long)numTileIndices/totalWarps : (warp+1)*(long long)numTiles/totalWarps);
#else
    const unsigned int numTiles = numTileIndices;
    int pos = (int) (startTileIndex+warp*(long long)numTiles/totalWarps);
    int end = (int) (startTileIndex+(warp+1)*(long long)numTiles/totalWarps);
#endif
    int skipBase = 0;
    int currentSkipIndex = tbx;
    __shared__ int atomIndices[THREAD_BLOCK_SIZE];
    __shared__ volatile int skipTiles[THREAD_BLOCK_SIZE];
    skipTiles[threadIdx.x] = -1;

    while (pos < end) {
        bool includeTile = true;

        // Extract the coordinates of this tile.

        int x, y;
#ifdef USE_CUTOFF
        x = tiles[pos];
#else
        y = (int) floor(NUM_BLOCKS+0.5f-SQRT((NUM_BLOCKS+0.5f)*(NUM_BLOCKS+0.5f)-2*pos));
        x = (pos-y*NUM_BLOCKS+y*(y+1)/2);
        if (x < y || x >= NUM_BLOCKS) { // Occasionally happens due to roundoff error.
            y += (x < y ? -1 : 1);
            x = (pos-y*NUM_BLOCKS+y*(y+1)/2);
        }

        // Skip over tiles that have exclusions, since they were already processed.

        while (skipTiles[tbx+TILE_SIZE-1] < pos) {
            if (skipBase+tgx < NUM_TILES_WITH_EXCLUSIONS) {
                int2 tile = exclusionTiles[skipBase+tgx];
                skipTiles[threadIdx.x] = tile.x + tile.y*NUM_BLOCKS - tile.y*(tile.y+1)/2;
            }
            else
                skipTiles[threadIdx.x] = end;
            skipBase += TILE_SIZE;            
            currentSkipIndex = tbx;
        }
        while (skipTiles[currentSkipIndex] < pos)
            currentSkipIndex++;
        includeTile = (skipTiles[currentSkipIndex] != pos);
#endif
        if (includeTile) {
            unsigned int atom1 = x*TILE_SIZE + tgx;

            // Load atom data for this tile.

            AtomData data;
            loadAtomData(data, atom1, posq, sphericalDipole, sphericalQuadrupole, sphericalOctopole, inducedDipole, dampingAndThole);
            data.force = make_real3(0);
            data.torque = make_real3(0);
#ifdef USE_CUTOFF
            unsigned int j = interactingAtoms[pos*TILE_SIZE+tgx];
#else
            unsigned int j = y*TILE_SIZE + tgx;
#endif
            atomIndices[threadIdx.x] = j;
            loadAtomData(localData[threadIdx.x], j, posq, sphericalDipole, sphericalQuadrupole, sphericalOctopole, inducedDipole, dampingAndThole);
            localData[threadIdx.x].force = make_real3(0);
            localData[threadIdx.x].torque = make_real3(0);

            // Compute forces.

            unsigned int tj = tgx;
            for (j = 0; j < TILE_SIZE; j++) {
                int atom2 = atomIndices[tbx+tj];
                if (atom1 < NUM_ATOMS && atom2 < NUM_ATOMS) {
                    computeOneInteraction(data, localData[tbx+tj], false, 1, 1, 1, energy, periodicBoxSize, invPeriodicBoxSize, periodicBoxVecX, periodicBoxVecY, periodicBoxVecZ);
                }
                tj = (tj + 1) & (TILE_SIZE - 1);
            }
            data.force *= -ENERGY_SCALE_FACTOR;
            data.torque *= ENERGY_SCALE_FACTOR;
            localData[threadIdx.x].force *= -ENERGY_SCALE_FACTOR;
            localData[threadIdx.x].torque *= ENERGY_SCALE_FACTOR;

            // Write results.

            unsigned int offset = x*TILE_SIZE + tgx;
            atomicAdd(&forceBuffers[offset], static_cast<unsigned long long>((long long) (data.force.x*0x100000000)));
            atomicAdd(&forceBuffers[offset+PADDED_NUM_ATOMS], static_cast<unsigned long long>((long long) (data.force.y*0x100000000)));
            atomicAdd(&forceBuffers[offset+2*PADDED_NUM_ATOMS], static_cast<unsigned long long>((long long) (data.force.z*0x100000000)));
            atomicAdd(&torqueBuffers[offset], static_cast<unsigned long long>((long long) (data.torque.x*0x100000000)));
            atomicAdd(&torqueBuffers[offset+PADDED_NUM_ATOMS], static_cast<unsigned long long>((long long) (data.torque.y*0x100000000)));
            atomicAdd(&torqueBuffers[offset+2*PADDED_NUM_ATOMS], static_cast<unsigned long long>((long long) (data.torque.z*0x100000000)));
#ifdef USE_CUTOFF
            offset = atomIndices[threadIdx.x];
#else
            offset = y*TILE_SIZE + tgx;
#endif
            atomicAdd(&forceBuffers[offset], static_cast<unsigned long long>((long long) (localData[threadIdx.x].force.x*0x100000000)));
            atomicAdd(&forceBuffers[offset+PADDED_NUM_ATOMS], static_cast<unsigned long long>((long long) (localData[threadIdx.x].force.y*0x100000000)));
            atomicAdd(&forceBuffers[offset+2*PADDED_NUM_ATOMS], static_cast<unsigned long long>((long long) (localData[threadIdx.x].force.z*0x100000000)));
            atomicAdd(&torqueBuffers[offset], static_cast<unsigned long long>((long long) (localData[threadIdx.x].torque.x*0x100000000)));
            atomicAdd(&torqueBuffers[offset+PADDED_NUM_ATOMS], static_cast<unsigned long long>((long long) (localData[threadIdx.x].torque.y*0x100000000)));
            atomicAdd(&torqueBuffers[offset+2*PADDED_NUM_ATOMS], static_cast<unsigned long long>((long long) (localData[threadIdx.x].torque.z*0x100000000)));
        }
        pos++;
    }
    energyBuffer[blockIdx.x*blockDim.x+threadIdx.x] += energy*ENERGY_SCALE_FACTOR;
}
