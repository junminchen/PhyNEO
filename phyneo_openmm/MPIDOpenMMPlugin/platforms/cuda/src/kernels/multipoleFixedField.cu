#define WARPS_PER_GROUP (THREAD_BLOCK_SIZE/TILE_SIZE)

typedef struct {
    real4 posq;
    real3 field, dipole;
#ifdef INCLUDE_QUADRUPOLES
    real quadrupoleXX, quadrupoleXY, quadrupoleXZ;
    real quadrupoleYY, quadrupoleYZ, quadrupoleZZ;
#endif
#ifdef INCLUDE_OCTOPOLES
    real octopoleXXX, octopoleXXY, octopoleXXZ, octopoleXYY, octopoleXYZ;
    real octopoleXZZ, octopoleYYY, octopoleYYZ, octopoleYZZ, octopoleZZZ;
#endif
    float thole, damp;
} AtomData;

inline __device__ void loadAtomData(AtomData& data, int atom, const real4* __restrict__ posq, const real* __restrict__ labFrameDipole,
                                   const real* __restrict__ labFrameQuadrupole, const real* __restrict__ labFrameOctopole,
                                   const float2* __restrict__ dampingAndThole) {
    data.posq = posq[atom];
    data.dipole.x = labFrameDipole[atom*3];
    data.dipole.y = labFrameDipole[atom*3+1];
    data.dipole.z = labFrameDipole[atom*3+2];
#ifdef INCLUDE_QUADRUPOLES
    data.quadrupoleXX = labFrameQuadrupole[atom*5];
    data.quadrupoleXY = labFrameQuadrupole[atom*5+1];
    data.quadrupoleXZ = labFrameQuadrupole[atom*5+2];
    data.quadrupoleYY = labFrameQuadrupole[atom*5+3];
    data.quadrupoleYZ = labFrameQuadrupole[atom*5+4];
    data.quadrupoleZZ = -(data.quadrupoleXX+data.quadrupoleYY);
#endif
#ifdef INCLUDE_OCTOPOLES
    data.octopoleXXX = labFrameOctopole[atom*7+0];
    data.octopoleXXY = labFrameOctopole[atom*7+1];
    data.octopoleXXZ = labFrameOctopole[atom*7+2];
    data.octopoleXYY = labFrameOctopole[atom*7+3];
    data.octopoleXYZ = labFrameOctopole[atom*7+4];
    data.octopoleYYY = labFrameOctopole[atom*7+5];
    data.octopoleYYZ = labFrameOctopole[atom*7+6];
    data.octopoleXZZ = -data.octopoleXXX-data.octopoleXYY;
    data.octopoleYZZ = -data.octopoleXXY-data.octopoleYYY;
    data.octopoleZZZ = -data.octopoleXXZ-data.octopoleYYZ;
#endif
    float2 temp = dampingAndThole[atom];
    data.damp = temp.x;
    data.thole = temp.y;
}

#ifdef USE_EWALD
__device__ void computeOneInteraction(AtomData& atom1, AtomData& atom2, real3 deltaR, float pScale, real3* fields) {
    real r2 = dot(deltaR, deltaR);
    if (r2 <= CUTOFF_SQUARED) {
        // calculate the error function damping terms

        real r = SQRT(r2);
        real ralpha = EWALD_ALPHA*r;
        real exp2a = EXP(-(ralpha*ralpha));
#ifdef USE_DOUBLE_PRECISION
        const real erfcAlphaR = erfc(ralpha);
#else
        // This approximation for erfc is from Abramowitz and Stegun (1964) p. 299.  They cite the following as
        // the original source: C. Hastings, Jr., Approximations for Digital Computers (1955).  It has a maximum
        // error of 1.5e-7.

        const real t = RECIP(1.0f+0.3275911f*ralpha);
        const real erfcAlphaR = (0.254829592f+(-0.284496736f+(1.421413741f+(-1.453152027f+1.061405429f*t)*t)*t)*t)*t*exp2a;
#endif
        real bn0 = erfcAlphaR/r;
        real alsq2 = 2*EWALD_ALPHA*EWALD_ALPHA;
        real alsq2n = RECIP(SQRT_PI*EWALD_ALPHA);
        alsq2n *= alsq2;
        real bn1 = (bn0+alsq2n*exp2a)/r2;
        alsq2n *= alsq2;
        real bn2 = (3*bn1+alsq2n*exp2a)/r2;

        // compute the error function scaled and unscaled terms

        real damp = fabs(atom1.damp*atom2.damp);
        real pgamma = (pScale == 0 ? atom1.thole + atom2.thole : DEFAULT_THOLE_WIDTH);
        real dfac = (damp == 0 ? 9999 : pgamma * r / damp); // TODO the inverses should be computed at parse time
        real expdamp = (dfac < 50 ? EXP(-dfac) : 0);

        real scale3 = 1 - expdamp*(1 + dfac + 0.5f*dfac*dfac);
        real scale5 = 1 - expdamp*(1 + dfac + 0.5f*dfac*dfac + dfac*dfac*dfac/6);

        real psc3 = pScale*scale3;
        real psc5 = pScale*scale5;

        real r3 = r*r2;
        real r5 = r3*r2;

        real prr3 = (1-psc3)/r3;
        real prr5 = 3*(1-psc5)/r5;

        real dir = dot(atom1.dipole, deltaR);
        real dkr = dot(atom2.dipole, deltaR);

#ifdef INCLUDE_QUADRUPOLES
        alsq2n *= alsq2;
        real bn3 = (5*bn2+alsq2n*exp2a)/r2;
        real scale7 = 1 - expdamp*(1 + dfac + 0.5f*dfac*dfac + dfac*dfac*dfac/6 + dfac*dfac*dfac*dfac/30);
        real psc7 = pScale*scale7;
        real r7 = r5*r2;
        real prr7 = 15*(1-psc7)/r7;

        real3 qi;
        qi.x = atom1.quadrupoleXX*deltaR.x + atom1.quadrupoleXY*deltaR.y + atom1.quadrupoleXZ*deltaR.z;
        qi.y = atom1.quadrupoleXY*deltaR.x + atom1.quadrupoleYY*deltaR.y + atom1.quadrupoleYZ*deltaR.z;
        qi.z = atom1.quadrupoleXZ*deltaR.x + atom1.quadrupoleYZ*deltaR.y + atom1.quadrupoleZZ*deltaR.z;
        real qir = dot(qi, deltaR);

        real3 qk;
        qk.x = atom2.quadrupoleXX*deltaR.x + atom2.quadrupoleXY*deltaR.y + atom2.quadrupoleXZ*deltaR.z;
        qk.y = atom2.quadrupoleXY*deltaR.x + atom2.quadrupoleYY*deltaR.y + atom2.quadrupoleYZ*deltaR.z;
        qk.z = atom2.quadrupoleXZ*deltaR.x + atom2.quadrupoleYZ*deltaR.y + atom2.quadrupoleZZ*deltaR.z;
        real qkr = dot(qk, deltaR);

        real3 fim = -deltaR*(bn1*atom2.posq.w-bn2*dkr+bn3*qkr) - bn1*atom2.dipole + 2*bn2*qk;
        real3 fkm = deltaR*(bn1*atom1.posq.w+bn2*dir+bn3*qir) - bn1*atom1.dipole - 2*bn2*qi;
        real3 fip = -deltaR*(prr3*atom2.posq.w-prr5*dkr+prr7*qkr) - prr3*atom2.dipole + 2*prr5*qk;
        real3 fkp = deltaR*(prr3*atom1.posq.w+prr5*dir+prr7*qir) - prr3*atom1.dipole - 2*prr5*qi;
#ifdef INCLUDE_OCTOPOLES
        alsq2n *= alsq2;
        real bn4 = (7*bn3+alsq2n*exp2a)/r2;
        real scale9 = 1 - expdamp*(1 + dfac + 0.5f*dfac*dfac + dfac*dfac*dfac/6 + 4*dfac*dfac*dfac*dfac/105 + dfac*dfac*dfac*dfac*dfac/210);
        real psc9 = pScale*scale9;
        real r9 = r7*r2;
        real prr9 = 105*(1-psc9)/r9;

        real3 oxx = make_real3(atom2.octopoleXXX, atom2.octopoleXXY, atom2.octopoleXXZ);
        real3 oxy = make_real3(atom2.octopoleXXY, atom2.octopoleXYY, atom2.octopoleXYZ);
        real3 oxz = make_real3(atom2.octopoleXXZ, atom2.octopoleXYZ, atom2.octopoleXZZ);
        real3 oyy = make_real3(atom2.octopoleXYY, atom2.octopoleYYY, atom2.octopoleYYZ);
        real3 oyz = make_real3(atom2.octopoleXYZ, atom2.octopoleYYZ, atom2.octopoleYZZ);
        real3 ozz = make_real3(atom2.octopoleXZZ, atom2.octopoleYZZ, atom2.octopoleZZZ);
        real3 ox  = make_real3(dot(oxx,deltaR), dot(oxy,deltaR), dot(oxz,deltaR));
        real3 oy  = make_real3(dot(oxy,deltaR), dot(oyy,deltaR), dot(oyz,deltaR));
        real3 oz  = make_real3(dot(oxz,deltaR), dot(oyz,deltaR), dot(ozz,deltaR));
        real3 o   = make_real3(dot(ox,deltaR), dot(oy,deltaR), dot(oz,deltaR));
        fim += -o*(3*bn3) + deltaR*bn4*dot(o,deltaR);
        fip += -o*(3*prr7) + deltaR*prr9*dot(o,deltaR);

        oxx = make_real3(atom1.octopoleXXX, atom1.octopoleXXY, atom1.octopoleXXZ);
        oxy = make_real3(atom1.octopoleXXY, atom1.octopoleXYY, atom1.octopoleXYZ);
        oxz = make_real3(atom1.octopoleXXZ, atom1.octopoleXYZ, atom1.octopoleXZZ);
        oyy = make_real3(atom1.octopoleXYY, atom1.octopoleYYY, atom1.octopoleYYZ);
        oyz = make_real3(atom1.octopoleXYZ, atom1.octopoleYYZ, atom1.octopoleYZZ);
        ozz = make_real3(atom1.octopoleXZZ, atom1.octopoleYZZ, atom1.octopoleZZZ);
        ox  = make_real3(dot(oxx,deltaR), dot(oxy,deltaR), dot(oxz,deltaR));
        oy  = make_real3(dot(oxy,deltaR), dot(oyy,deltaR), dot(oyz,deltaR));
        oz  = make_real3(dot(oxz,deltaR), dot(oyz,deltaR), dot(ozz,deltaR));
        o   = make_real3(dot(ox,deltaR), dot(oy,deltaR), dot(oz,deltaR));
        fkm += -o*(3*bn3) + deltaR*bn4*dot(o,deltaR);
        fkp += -o*(3*prr7) + deltaR*prr9*dot(o,deltaR);
#endif // End INCLUDE_OCTOPOLES
#else
        // Charge-only routine
        real3 fim = -deltaR*(bn1*atom2.posq.w-bn2*dkr) - bn1*atom2.dipole;
        real3 fkm = deltaR*(bn1*atom1.posq.w+bn2*dir) - bn1*atom1.dipole;
        real3 fip = -deltaR*(prr3*atom2.posq.w-prr5*dkr) - prr3*atom2.dipole;
        real3 fkp = deltaR*(prr3*atom1.posq.w+prr5*dir) - prr3*atom1.dipole;
#endif
        // increment the field at each site due to this interaction
        fields[0] = fim-fip;
        fields[1] = fkm-fkp;
    }
    else {
        fields[0] = make_real3(0);
        fields[1] = make_real3(0);
    }
}
#else
__device__ void computeOneInteraction(AtomData& atom1, AtomData& atom2, real3 deltaR, float pScale, real3* fields) {
    real rI = RSQRT(dot(deltaR, deltaR));
    real r = RECIP(rI);
    real r2I = rI*rI;

    real rr3 = rI*r2I;
    real rr5 = 3*rr3*r2I;
    real rr7 = 5*rr5*r2I;
    real rr9 = 7*rr7*r2I;
 
    // get scaling factors, if needed
    
    float damp = fabs(atom1.damp*atom2.damp);
    real dampExp;
    if (damp != 0) {

        // get scaling factors
      
        real ratio = r/damp;
        float pGamma  = pScale == 0.0f ? atom1.thole + atom2.thole : DEFAULT_THOLE_WIDTH;
        damp = ratio*pGamma;
        dampExp = EXP(-damp);
    }
    else
        dampExp = 0;
    rr3 *= 1 - dampExp*(1 + damp + damp*damp/2);
    rr5 *= 1 - dampExp*(1 + damp + damp*damp/2 + damp*damp*damp/6);
    rr7 *= 1 - dampExp*(1 + damp + damp*damp/2 + damp*damp*damp/6 + damp*damp*damp*damp/30);
    rr9 *= 1 - dampExp*(1 + damp + damp*damp/2 + damp*damp*damp/6 + 4*damp*damp*damp*damp/105 + damp*damp*damp*damp*damp/210);
    real rr5_2 = 2*rr5;
    real rr7_3 = 3*rr7;
 
    real dir = dot(atom1.dipole, deltaR);
    real dkr = dot(atom2.dipole, deltaR);

#ifdef INCLUDE_QUADRUPOLES
    real3 qi;
    qi.x = atom1.quadrupoleXX*deltaR.x + atom1.quadrupoleXY*deltaR.y + atom1.quadrupoleXZ*deltaR.z;
    qi.y = atom1.quadrupoleXY*deltaR.x + atom1.quadrupoleYY*deltaR.y + atom1.quadrupoleYZ*deltaR.z;
    qi.z = atom1.quadrupoleXZ*deltaR.x + atom1.quadrupoleYZ*deltaR.y + atom1.quadrupoleZZ*deltaR.z;
    real qir = dot(qi, deltaR);

    real3 qk;
    qk.x = atom2.quadrupoleXX*deltaR.x + atom2.quadrupoleXY*deltaR.y + atom2.quadrupoleXZ*deltaR.z;
    qk.y = atom2.quadrupoleXY*deltaR.x + atom2.quadrupoleYY*deltaR.y + atom2.quadrupoleYZ*deltaR.z;
    qk.z = atom2.quadrupoleXZ*deltaR.x + atom2.quadrupoleYZ*deltaR.y + atom2.quadrupoleZZ*deltaR.z;
    real qkr = dot(qk, deltaR);

    real factor = -rr3*atom2.posq.w + rr5*dkr - rr7*qkr;
    real3 field1 = deltaR*factor - rr3*atom2.dipole + rr5_2*qk;
    factor = rr3*atom1.posq.w + rr5*dir + rr7*qir;
    real3 field2 = deltaR*factor - rr3*atom1.dipole - rr5_2*qi;
#ifdef INCLUDE_OCTOPOLES
    real3 oxx = make_real3(atom2.octopoleXXX, atom2.octopoleXXY, atom2.octopoleXXZ);
    real3 oxy = make_real3(atom2.octopoleXXY, atom2.octopoleXYY, atom2.octopoleXYZ);
    real3 oxz = make_real3(atom2.octopoleXXZ, atom2.octopoleXYZ, atom2.octopoleXZZ);
    real3 oyy = make_real3(atom2.octopoleXYY, atom2.octopoleYYY, atom2.octopoleYYZ);
    real3 oyz = make_real3(atom2.octopoleXYZ, atom2.octopoleYYZ, atom2.octopoleYZZ);
    real3 ozz = make_real3(atom2.octopoleXZZ, atom2.octopoleYZZ, atom2.octopoleZZZ);
    real3 ox  = make_real3(dot(oxx,deltaR), dot(oxy,deltaR), dot(oxz,deltaR));
    real3 oy  = make_real3(dot(oxy,deltaR), dot(oyy,deltaR), dot(oyz,deltaR));
    real3 oz  = make_real3(dot(oxz,deltaR), dot(oyz,deltaR), dot(ozz,deltaR));
    real3 o   = make_real3(dot(ox,deltaR), dot(oy,deltaR), dot(oz,deltaR));
    field1 += deltaR*rr9*dot(o,deltaR) - o*rr7_3;
    oxx = make_real3(atom1.octopoleXXX, atom1.octopoleXXY, atom1.octopoleXXZ);
    oxy = make_real3(atom1.octopoleXXY, atom1.octopoleXYY, atom1.octopoleXYZ);
    oxz = make_real3(atom1.octopoleXXZ, atom1.octopoleXYZ, atom1.octopoleXZZ);
    oyy = make_real3(atom1.octopoleXYY, atom1.octopoleYYY, atom1.octopoleYYZ);
    oyz = make_real3(atom1.octopoleXYZ, atom1.octopoleYYZ, atom1.octopoleYZZ);
    ozz = make_real3(atom1.octopoleXZZ, atom1.octopoleYZZ, atom1.octopoleZZZ);
    ox  = make_real3(dot(oxx,deltaR), dot(oxy,deltaR), dot(oxz,deltaR));
    oy  = make_real3(dot(oxy,deltaR), dot(oyy,deltaR), dot(oyz,deltaR));
    oz  = make_real3(dot(oxz,deltaR), dot(oyz,deltaR), dot(ozz,deltaR));
    o   = make_real3(dot(ox,deltaR), dot(oy,deltaR), dot(oz,deltaR));
    field2 += deltaR*rr9*dot(o,deltaR) - o*rr7_3;
#endif
#else
    real factor = -rr3*atom2.posq.w + rr5*dkr;
    real3 field1 = deltaR*factor - rr3*atom2.dipole;
    factor = rr3*atom1.posq.w + rr5*dir;
    real3 field2 = deltaR*factor - rr3*atom1.dipole;
#endif
    fields[0] = pScale*field1;
    fields[1] = pScale*field2;
}
#endif


__device__ float computePScaleFactor(uint2 covalent, int index) {
    int mask = 1<<index;
    bool x = (covalent.x & mask);
    bool y = (covalent.y & mask);
	return (x ? (y ? 0.0f : (float) SCALEFACTOR14): 1.0f);
}

/**
 * Compute nonbonded interactions.
 */
extern "C" __global__ void computeFixedField(
        unsigned long long* __restrict__ fieldBuffers, const real4* __restrict__ posq,
        const uint2* __restrict__ covalentFlags, const int2* __restrict__ exclusionTiles,
        unsigned int startTileIndex, unsigned int numTileIndices,
#ifdef USE_CUTOFF
        const int* __restrict__ tiles, const unsigned int* __restrict__ interactionCount, real4 periodicBoxSize, real4 invPeriodicBoxSize,
        real4 periodicBoxVecX, real4 periodicBoxVecY, real4 periodicBoxVecZ, unsigned int maxTiles, const real4* __restrict__ blockCenter,
        const unsigned int* __restrict__ interactingAtoms,
#endif
        const real* __restrict__ labFrameDipole, const real* __restrict__ labFrameQuadrupole, const real* __restrict__ labFrameOctopole,
        const float2* __restrict__ dampingAndThole) {
    const unsigned int totalWarps = (blockDim.x*gridDim.x)/TILE_SIZE;
    const unsigned int warp = (blockIdx.x*blockDim.x+threadIdx.x)/TILE_SIZE;
    const unsigned int tgx = threadIdx.x & (TILE_SIZE-1);
    const unsigned int tbx = threadIdx.x - tgx;
    __shared__ AtomData localData[THREAD_BLOCK_SIZE];

    // First loop: process tiles that contain exclusions.

    const unsigned int firstExclusionTile = FIRST_EXCLUSION_TILE+warp*(LAST_EXCLUSION_TILE-FIRST_EXCLUSION_TILE)/totalWarps;
    const unsigned int lastExclusionTile = FIRST_EXCLUSION_TILE+(warp+1)*(LAST_EXCLUSION_TILE-FIRST_EXCLUSION_TILE)/totalWarps;
    for (int pos = firstExclusionTile; pos < lastExclusionTile; pos++) {
        const int2 tileIndices = exclusionTiles[pos];
        const unsigned int x = tileIndices.x;
        const unsigned int y = tileIndices.y;
        AtomData data;
        data.field = make_real3(0);
        unsigned int atom1 = x*TILE_SIZE + tgx;
        loadAtomData(data, atom1, posq, labFrameDipole, labFrameQuadrupole, labFrameOctopole, dampingAndThole);
        uint2 covalent = covalentFlags[pos*TILE_SIZE+tgx];
        if (x == y) {
            // This tile is on the diagonal.

            const unsigned int localAtomIndex = threadIdx.x;
            localData[localAtomIndex].posq = data.posq;
            localData[localAtomIndex].dipole = data.dipole;
#ifdef INCLUDE_QUADRUPOLES
            localData[localAtomIndex].quadrupoleXX = data.quadrupoleXX;
            localData[localAtomIndex].quadrupoleXY = data.quadrupoleXY;
            localData[localAtomIndex].quadrupoleXZ = data.quadrupoleXZ;
            localData[localAtomIndex].quadrupoleYY = data.quadrupoleYY;
            localData[localAtomIndex].quadrupoleYZ = data.quadrupoleYZ;
            localData[localAtomIndex].quadrupoleZZ = data.quadrupoleZZ;
#endif
#ifdef INCLUDE_OCTOPOLES
            localData[localAtomIndex].octopoleXXX = data.octopoleXXX;
            localData[localAtomIndex].octopoleXXY = data.octopoleXXY;
            localData[localAtomIndex].octopoleXXZ = data.octopoleXXZ;
            localData[localAtomIndex].octopoleXYY = data.octopoleXYY;
            localData[localAtomIndex].octopoleXYZ = data.octopoleXYZ;
            localData[localAtomIndex].octopoleXZZ = data.octopoleXZZ;
            localData[localAtomIndex].octopoleYYY = data.octopoleYYY;
            localData[localAtomIndex].octopoleYYZ = data.octopoleYYZ;
            localData[localAtomIndex].octopoleYZZ = data.octopoleYZZ;
            localData[localAtomIndex].octopoleZZZ = data.octopoleZZZ;
#endif
            localData[localAtomIndex].thole = data.thole;
            localData[localAtomIndex].damp = data.damp;
            for (unsigned int j = 0; j < TILE_SIZE; j++) {
                real3 delta = trimTo3(localData[tbx+j].posq-data.posq);
#ifdef USE_PERIODIC
                APPLY_PERIODIC_TO_DELTA(delta)
#endif
                int atom2 = y*TILE_SIZE+j;
                if (atom1 != atom2 && atom1 < NUM_ATOMS && atom2 < NUM_ATOMS) {
                    real3 fields[2];
                    float p = computePScaleFactor(covalent, j);
                    computeOneInteraction(data, localData[tbx+j], delta, p, fields);
                    data.field += fields[0];
                }
            }
        }
        else {
            // This is an off-diagonal tile.

            const unsigned int localAtomIndex = threadIdx.x;
            unsigned int j = y*TILE_SIZE + tgx;
            loadAtomData(localData[localAtomIndex], j, posq, labFrameDipole, labFrameQuadrupole, labFrameOctopole, dampingAndThole);
            localData[localAtomIndex].field = make_real3(0);
            unsigned int tj = tgx;
            for (j = 0; j < TILE_SIZE; j++) {
                real3 delta = trimTo3(localData[tbx+tj].posq-data.posq);
#ifdef USE_PERIODIC
                APPLY_PERIODIC_TO_DELTA(delta)
#endif
                int atom2 = y*TILE_SIZE+tj;
                if (atom1 < NUM_ATOMS && atom2 < NUM_ATOMS) {
                    real3 fields[2];
                    float p = computePScaleFactor(covalent, tj);
                    computeOneInteraction(data, localData[tbx+tj], delta, p, fields);
                    data.field += fields[0];
                    localData[tbx+tj].field += fields[1];
                }
                tj = (tj + 1) & (TILE_SIZE - 1);
            }
        }
        
        // Write results.
        
        unsigned int offset = x*TILE_SIZE + tgx;
        atomicAdd(&fieldBuffers[offset], static_cast<unsigned long long>((long long) (data.field.x*0x100000000)));
        atomicAdd(&fieldBuffers[offset+PADDED_NUM_ATOMS], static_cast<unsigned long long>((long long) (data.field.y*0x100000000)));
        atomicAdd(&fieldBuffers[offset+2*PADDED_NUM_ATOMS], static_cast<unsigned long long>((long long) (data.field.z*0x100000000)));
        if (x != y) {
            offset = y*TILE_SIZE + tgx;
            atomicAdd(&fieldBuffers[offset], static_cast<unsigned long long>((long long) (localData[threadIdx.x].field.x*0x100000000)));
            atomicAdd(&fieldBuffers[offset+PADDED_NUM_ATOMS], static_cast<unsigned long long>((long long) (localData[threadIdx.x].field.y*0x100000000)));
            atomicAdd(&fieldBuffers[offset+2*PADDED_NUM_ATOMS], static_cast<unsigned long long>((long long) (localData[threadIdx.x].field.z*0x100000000)));
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
        
        int x;
#ifdef USE_CUTOFF
        x = tiles[pos];
#else
        int y = (int) floor(NUM_BLOCKS+0.5f-SQRT((NUM_BLOCKS+0.5f)*(NUM_BLOCKS+0.5f)-2*pos));
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
            data.field = make_real3(0);
            loadAtomData(data, atom1, posq, labFrameDipole, labFrameQuadrupole, labFrameOctopole, dampingAndThole);
#ifdef USE_CUTOFF
            unsigned int j = interactingAtoms[pos*TILE_SIZE+tgx];
#else
            unsigned int j = y*TILE_SIZE + tgx;
#endif
            atomIndices[threadIdx.x] = j;
            const unsigned int localAtomIndex = threadIdx.x;
            loadAtomData(localData[localAtomIndex], j, posq, labFrameDipole, labFrameQuadrupole, labFrameOctopole, dampingAndThole);
            localData[localAtomIndex].field = make_real3(0);

            // Compute the full set of interactions in this tile.

            unsigned int tj = tgx;
            for (j = 0; j < TILE_SIZE; j++) {
                real3 delta = trimTo3(localData[tbx+tj].posq-data.posq);
#ifdef USE_PERIODIC
                APPLY_PERIODIC_TO_DELTA(delta)
#endif
                int atom2 = atomIndices[tbx+tj];
                if (atom1 < NUM_ATOMS && atom2 < NUM_ATOMS) {
                    real3 fields[2];
                    computeOneInteraction(data, localData[tbx+tj], delta, 1, fields);
                    data.field += fields[0];
                    localData[tbx+tj].field += fields[1];
                }
                tj = (tj + 1) & (TILE_SIZE - 1);
            }

            // Write results.

            unsigned int offset = x*TILE_SIZE + tgx;
            atomicAdd(&fieldBuffers[offset], static_cast<unsigned long long>((long long) (data.field.x*0x100000000)));
            atomicAdd(&fieldBuffers[offset+PADDED_NUM_ATOMS], static_cast<unsigned long long>((long long) (data.field.y*0x100000000)));
            atomicAdd(&fieldBuffers[offset+2*PADDED_NUM_ATOMS], static_cast<unsigned long long>((long long) (data.field.z*0x100000000)));
#ifdef USE_CUTOFF
            offset = atomIndices[threadIdx.x];
#else
            offset = y*TILE_SIZE + tgx;
#endif
            atomicAdd(&fieldBuffers[offset], static_cast<unsigned long long>((long long) (localData[threadIdx.x].field.x*0x100000000)));
            atomicAdd(&fieldBuffers[offset+PADDED_NUM_ATOMS], static_cast<unsigned long long>((long long) (localData[threadIdx.x].field.y*0x100000000)));
            atomicAdd(&fieldBuffers[offset+2*PADDED_NUM_ATOMS], static_cast<unsigned long long>((long long) (localData[threadIdx.x].field.z*0x100000000)));
        }
        pos++;
    }
}
