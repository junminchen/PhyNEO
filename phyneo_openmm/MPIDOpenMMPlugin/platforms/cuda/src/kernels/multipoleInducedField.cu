#define WARPS_PER_GROUP (THREAD_BLOCK_SIZE/TILE_SIZE)

typedef struct {
    real3 pos;
    real3 field, inducedDipole;
#ifdef EXTRAPOLATED_POLARIZATION
    real fieldGradient[6];
#endif
    float thole, damp;
} AtomData;

inline __device__ void loadAtomData(AtomData& data, int atom, const real4* __restrict__ posq, const real* __restrict__ inducedDipole,
        const float2* __restrict__ dampingAndThole) {
    real4 atomPosq = posq[atom];
    data.pos = make_real3(atomPosq.x, atomPosq.y, atomPosq.z);
    data.inducedDipole.x = inducedDipole[atom*3];
    data.inducedDipole.y = inducedDipole[atom*3+1];
    data.inducedDipole.z = inducedDipole[atom*3+2];
    float2 temp = dampingAndThole[atom];
    data.damp = fabs(temp.x);
    data.thole = temp.y;
}

__device__ float computePScaleFactor(uint2 covalent, int index) {
    int mask = 1<<index;
    bool x = (covalent.x & mask);
    bool y = (covalent.y & mask);
	return (x ? (y ? 0.0f : (float) SCALEFACTOR14): 1.0f);
}

inline __device__ void zeroAtomData(AtomData& data) {
    data.field = make_real3(0);
#ifdef EXTRAPOLATED_POLARIZATION
    for (int i = 0; i < 6; i++) {
        data.fieldGradient[i] = 0;
    }
#endif
}

#ifdef EXTRAPOLATED_POLARIZATION
    #define SAVE_ATOM_DATA(index, data) saveAtomData(index, data, field, fieldGradient);
#else
    #define SAVE_ATOM_DATA(index, data) saveAtomData(index, data, field);
#endif

inline __device__ void saveAtomData(int index, AtomData& data, unsigned long long* __restrict__ field
#ifdef EXTRAPOLATED_POLARIZATION
        , unsigned long long* __restrict__ fieldGradient
#endif
        ) {
    atomicAdd(&field[index], static_cast<unsigned long long>((long long) (data.field.x*0x100000000)));
    atomicAdd(&field[index+PADDED_NUM_ATOMS], static_cast<unsigned long long>((long long) (data.field.y*0x100000000)));
    atomicAdd(&field[index+2*PADDED_NUM_ATOMS], static_cast<unsigned long long>((long long) (data.field.z*0x100000000)));
#ifdef EXTRAPOLATED_POLARIZATION
    for (int i = 0; i < 6; i++) {
        atomicAdd(&fieldGradient[6*index+i], static_cast<unsigned long long>((long long) (data.fieldGradient[i]*0x100000000)));
    }
#endif
}

#ifdef USE_EWALD
__device__ void computeOneInteraction(AtomData& atom1, AtomData& atom2, real3 deltaR, float pScale, bool isSelfInteraction) {
    if (isSelfInteraction)
        return;
    real scale1, scale2, scale3;
    real r2 = dot(deltaR, deltaR);
    if (r2 < CUTOFF_SQUARED) {
        real rI = RSQRT(r2);
        real r = RECIP(rI);
        real rI2 = rI*rI;

        // calculate the error function damping terms

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
        real bn0 = erfcAlphaR*rI;
        real alsq2 = 2*EWALD_ALPHA*EWALD_ALPHA;
        real alsq2n = RECIP(SQRT_PI*EWALD_ALPHA);
        alsq2n *= alsq2;
        real bn1 = (bn0+alsq2n*exp2a)*rI2;

        alsq2n *= alsq2;
        real bn2 = (3*bn1+alsq2n*exp2a)*rI2;

        alsq2n *= alsq2;
        real bn3 = (5*bn2+alsq2n*exp2a)*rI2;

        // compute the error function scaled and unscaled terms

        real damp = fabs(atom1.damp*atom2.damp);
        real ratio = damp == 0 ? 0 : (r/damp);
        float pgamma = pScale == 0 ? atom1.thole + atom2.thole : DEFAULT_THOLE_WIDTH;
        damp = pgamma*ratio;
        real expdamp = ratio == 0 ? 0 : EXP(-damp);
        real dsc3 = 1 - expdamp*(1 + damp + damp*damp/2);
        real dsc5 = 1 - expdamp*(1 + damp + damp*damp/2 + damp*damp*damp/6);
        real dsc7 = 1 - expdamp*(1 + damp + damp*damp/2 + damp*damp*damp/6 + damp*damp*damp*damp/30);
        real r3 = (r*r2);
        real r5 = (r3*r2);
        real r7 = (r5*r2);
        real rr3 = (1-dsc3)/r3;
        real rr5 = 3*(1-dsc5)/r5;
        real rr7 = 15*(1-dsc7)/r7;

        scale1 = rr3 - bn1;
        scale2 = bn2 - rr5;
        scale3 = bn3 - rr7;
    }
    else {
        scale1 = 0;
        scale2 = 0;
        scale3 = 0;
    }
    real dDotDelta = scale2*dot(deltaR, atom2.inducedDipole);
    atom1.field += scale1*atom2.inducedDipole + dDotDelta*deltaR;
    dDotDelta = scale2*dot(deltaR, atom1.inducedDipole);
    atom2.field += scale1*atom1.inducedDipole + dDotDelta*deltaR;
#ifdef EXTRAPOLATED_POLARIZATION
    // Compute and store the field gradients for later use.
    
    real3 dipole = atom1.inducedDipole;
    real muDotR = dipole.x*deltaR.x + dipole.y*deltaR.y + dipole.z*deltaR.z;
    atom2.fieldGradient[0] -= muDotR*deltaR.x*deltaR.x*scale3 - (2*dipole.x*deltaR.x + muDotR)*scale2;
    atom2.fieldGradient[1] -= muDotR*deltaR.y*deltaR.y*scale3 - (2*dipole.y*deltaR.y + muDotR)*scale2;
    atom2.fieldGradient[2] -= muDotR*deltaR.z*deltaR.z*scale3 - (2*dipole.z*deltaR.z + muDotR)*scale2;
    atom2.fieldGradient[3] -= muDotR*deltaR.x*deltaR.y*scale3 - (dipole.x*deltaR.y + dipole.y*deltaR.x)*scale2;
    atom2.fieldGradient[4] -= muDotR*deltaR.x*deltaR.z*scale3 - (dipole.x*deltaR.z + dipole.z*deltaR.x)*scale2;
    atom2.fieldGradient[5] -= muDotR*deltaR.y*deltaR.z*scale3 - (dipole.y*deltaR.z + dipole.z*deltaR.y)*scale2;

    dipole = atom2.inducedDipole;
    muDotR = dipole.x*deltaR.x + dipole.y*deltaR.y + dipole.z*deltaR.z;
    atom1.fieldGradient[0] += muDotR*deltaR.x*deltaR.x*scale3 - (2*dipole.x*deltaR.x + muDotR)*scale2;
    atom1.fieldGradient[1] += muDotR*deltaR.y*deltaR.y*scale3 - (2*dipole.y*deltaR.y + muDotR)*scale2;
    atom1.fieldGradient[2] += muDotR*deltaR.z*deltaR.z*scale3 - (2*dipole.z*deltaR.z + muDotR)*scale2;
    atom1.fieldGradient[3] += muDotR*deltaR.x*deltaR.y*scale3 - (dipole.x*deltaR.y + dipole.y*deltaR.x)*scale2;
    atom1.fieldGradient[4] += muDotR*deltaR.x*deltaR.z*scale3 - (dipole.x*deltaR.z + dipole.z*deltaR.x)*scale2;
    atom1.fieldGradient[5] += muDotR*deltaR.y*deltaR.z*scale3 - (dipole.y*deltaR.z + dipole.z*deltaR.y)*scale2;
#endif
}

#else

__device__ void computeOneInteraction(AtomData& atom1, AtomData& atom2, real3 deltaR, float pScale, bool isSelfInteraction) {
    if (isSelfInteraction)
        return;
    real rI = RSQRT(dot(deltaR, deltaR));
    real r = RECIP(rI);
    real r2I = rI*rI;
    real rr3 = -rI*r2I;
    real rr5 = -3*rr3*r2I;
    real rr7 = 5*rr5*r2I;
    real dampProd = fabs(atom1.damp*atom2.damp);
    real ratio = (dampProd != 0 ? r/dampProd : 1);
    float pGamma  = pScale == 0.0 ? atom1.thole + atom2.thole : DEFAULT_THOLE_WIDTH;
    real damp = ratio*pGamma;
    real dampExp = (dampProd != 0 ? EXP(-damp) : 0); 
    rr3 *= 1 - dampExp*(1 + damp + damp*damp/2);
    rr5 *= 1 - dampExp*(1 + damp + damp*damp/2 + damp*damp*damp/6);
    rr7 *= 1 - dampExp*(1 + damp + damp*damp/2 + damp*damp*damp/6 + damp*damp*damp*damp/30);
    real dDotDelta = rr5*dot(deltaR, atom2.inducedDipole);
    atom1.field += rr3*atom2.inducedDipole + dDotDelta*deltaR;
    dDotDelta = rr5*dot(deltaR, atom1.inducedDipole);
    atom2.field += rr3*atom1.inducedDipole + dDotDelta*deltaR;
#ifdef EXTRAPOLATED_POLARIZATION
    // Compute and store the field gradients for later use.
    real3 dipole = atom1.inducedDipole;
    real muDotR = dipole.x*deltaR.x + dipole.y*deltaR.y + dipole.z*deltaR.z;
    atom2.fieldGradient[0] -= muDotR*deltaR.x*deltaR.x*rr7 - (2*dipole.x*deltaR.x + muDotR)*rr5;
    atom2.fieldGradient[1] -= muDotR*deltaR.y*deltaR.y*rr7 - (2*dipole.y*deltaR.y + muDotR)*rr5;
    atom2.fieldGradient[2] -= muDotR*deltaR.z*deltaR.z*rr7 - (2*dipole.z*deltaR.z + muDotR)*rr5;
    atom2.fieldGradient[3] -= muDotR*deltaR.x*deltaR.y*rr7 - (dipole.x*deltaR.y + dipole.y*deltaR.x)*rr5;
    atom2.fieldGradient[4] -= muDotR*deltaR.x*deltaR.z*rr7 - (dipole.x*deltaR.z + dipole.z*deltaR.x)*rr5;
    atom2.fieldGradient[5] -= muDotR*deltaR.y*deltaR.z*rr7 - (dipole.y*deltaR.z + dipole.z*deltaR.y)*rr5;

    dipole = atom2.inducedDipole;
    muDotR = dipole.x*deltaR.x + dipole.y*deltaR.y + dipole.z*deltaR.z;
    atom1.fieldGradient[0] += muDotR*deltaR.x*deltaR.x*rr7 - (2*dipole.x*deltaR.x + muDotR)*rr5;
    atom1.fieldGradient[1] += muDotR*deltaR.y*deltaR.y*rr7 - (2*dipole.y*deltaR.y + muDotR)*rr5;
    atom1.fieldGradient[2] += muDotR*deltaR.z*deltaR.z*rr7 - (2*dipole.z*deltaR.z + muDotR)*rr5;
    atom1.fieldGradient[3] += muDotR*deltaR.x*deltaR.y*rr7 - (dipole.x*deltaR.y + dipole.y*deltaR.x)*rr5;
    atom1.fieldGradient[4] += muDotR*deltaR.x*deltaR.z*rr7 - (dipole.x*deltaR.z + dipole.z*deltaR.x)*rr5;
    atom1.fieldGradient[5] += muDotR*deltaR.y*deltaR.z*rr7 - (dipole.y*deltaR.z + dipole.z*deltaR.y)*rr5;
#endif
}
#endif

/**
 * Compute the mutual induced field.
 */
extern "C" __global__ void computeInducedField(
        unsigned long long* __restrict__ field, const real4* __restrict__ posq, const uint2* __restrict__ covalentFlags, const int2* __restrict__ exclusionTiles, 
        const real* __restrict__ inducedDipole, unsigned int startTileIndex, unsigned int numTileIndices,
#ifdef EXTRAPOLATED_POLARIZATION
        unsigned long long* __restrict__ fieldGradient,
#endif
#ifdef USE_CUTOFF
        const int* __restrict__ tiles, const unsigned int* __restrict__ interactionCount, real4 periodicBoxSize, real4 invPeriodicBoxSize,
        real4 periodicBoxVecX, real4 periodicBoxVecY, real4 periodicBoxVecZ, unsigned int maxTiles, const real4* __restrict__ blockCenter, const unsigned int* __restrict__ interactingAtoms,
#endif
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
        zeroAtomData(data);
        unsigned int atom1 = x*TILE_SIZE + tgx;
        loadAtomData(data, atom1, posq, inducedDipole, dampingAndThole);
        uint2 covalent = covalentFlags[pos*TILE_SIZE+tgx];
        if (x == y) {
            // This tile is on the diagonal.

            localData[threadIdx.x].pos = data.pos;
            localData[threadIdx.x].inducedDipole = data.inducedDipole;
            localData[threadIdx.x].thole = data.thole;
            localData[threadIdx.x].damp = data.damp;
            for (unsigned int j = 0; j < TILE_SIZE; j++) {
                real3 delta = localData[tbx+j].pos-data.pos;
#ifdef USE_PERIODIC
                APPLY_PERIODIC_TO_DELTA(delta)
#endif
                int atom2 = y*TILE_SIZE+j;
                if (atom1 < NUM_ATOMS && atom2 < NUM_ATOMS){
                    float p = computePScaleFactor(covalent, j);
                    computeOneInteraction(data, localData[tbx+j], delta, p, atom1 == atom2);
                }
            }
        }
        else {
            // This is an off-diagonal tile.

            loadAtomData(localData[threadIdx.x], y*TILE_SIZE+tgx, posq, inducedDipole, dampingAndThole);
            zeroAtomData(localData[threadIdx.x]);
            unsigned int tj = tgx;
            for (unsigned int j = 0; j < TILE_SIZE; j++) {
                real3 delta = localData[tbx+tj].pos-data.pos;
#ifdef USE_PERIODIC
                APPLY_PERIODIC_TO_DELTA(delta)
#endif
                int atom2 = y*TILE_SIZE+j;
                if (atom1 < NUM_ATOMS && atom2 < NUM_ATOMS){
                    float p = computePScaleFactor(covalent, tj);
                    computeOneInteraction(data, localData[tbx+tj], delta, p, false);
                }
                tj = (tj + 1) & (TILE_SIZE - 1);
            }
        }

        // Write results.

        unsigned int offset = x*TILE_SIZE + tgx;
        SAVE_ATOM_DATA(offset, data)
        if (x != y) {
            offset = y*TILE_SIZE + tgx;
            SAVE_ATOM_DATA(offset, localData[threadIdx.x])
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
            zeroAtomData(data);
            loadAtomData(data, atom1, posq, inducedDipole, dampingAndThole);
#ifdef USE_CUTOFF
            unsigned int j = interactingAtoms[pos*TILE_SIZE+tgx];
#else
            unsigned int j = y*TILE_SIZE + tgx;
#endif
            atomIndices[threadIdx.x] = j;
            loadAtomData(localData[threadIdx.x], j, posq, inducedDipole, dampingAndThole);
            zeroAtomData(localData[threadIdx.x]);

            // Compute the full set of interactions in this tile.

            unsigned int tj = tgx;
            for (j = 0; j < TILE_SIZE; j++) {
                real3 delta = localData[tbx+tj].pos-data.pos;
#ifdef USE_PERIODIC
                APPLY_PERIODIC_TO_DELTA(delta)
#endif
                int atom2 = atomIndices[tbx+tj];
                if (atom1 < NUM_ATOMS && atom2 < NUM_ATOMS)
                    computeOneInteraction(data, localData[tbx+tj], delta, 1, false);
                tj = (tj + 1) & (TILE_SIZE - 1);
            }

            // Write results.

            unsigned int offset = x*TILE_SIZE + tgx;
            SAVE_ATOM_DATA(offset, data)
#ifdef USE_CUTOFF
            offset = atomIndices[threadIdx.x];
#else
            offset = y*TILE_SIZE + tgx;
#endif
            SAVE_ATOM_DATA(offset, localData[threadIdx.x])
        }
        pos++;
    }
}


extern "C" __global__ void recordInducedDipolesForDIIS(const long long* __restrict__ fixedField, const long long* __restrict__ inducedField,
        const real* __restrict__ inducedDipole, const real* __restrict__ labFramePolarizabilities, float* __restrict__ errors,
        real* __restrict__ prevDipoles, real* __restrict__ prevErrors, int iteration, bool recordPrevErrors, real* __restrict__ matrix) {
    extern __shared__ real buffer[];
    const real fieldScale = 1/(real) 0x100000000;
    real sumErrors = 0;
    for (int atom = blockIdx.x*blockDim.x + threadIdx.x; atom < NUM_ATOMS; atom += blockDim.x*gridDim.x) {
        //
        real3 fld = make_real3(fixedField[atom+0*PADDED_NUM_ATOMS] + inducedField[atom+0*PADDED_NUM_ATOMS],
                               fixedField[atom+1*PADDED_NUM_ATOMS] + inducedField[atom+1*PADDED_NUM_ATOMS],
                               fixedField[atom+2*PADDED_NUM_ATOMS] + inducedField[atom+2*PADDED_NUM_ATOMS]);
        int offset = 6*atom;
        for (int component = 0; component < 3; component++) {
            int dipoleIndex = 3*atom+component;
            if (iteration >= MAX_PREV_DIIS_DIPOLES) {
                // We have filled up the buffer for previous dipoles, so shift them all over by one.
                for (int i = 1; i < MAX_PREV_DIIS_DIPOLES; i++) {
                    int index1 = dipoleIndex+(i-1)*NUM_ATOMS*3;
                    int index2 = dipoleIndex+i*NUM_ATOMS*3;
                    prevDipoles[index1] = prevDipoles[index2];
                    if (recordPrevErrors)
                        prevErrors[index1] = prevErrors[index2];
                }
            }
            // Compute the new dipole, and record it along with the error.
            real3 alpha;
            if(component==0)
                alpha = make_real3(labFramePolarizabilities[offset+0], labFramePolarizabilities[offset+1], labFramePolarizabilities[offset+2]);
            else if(component==1)
                alpha = make_real3(labFramePolarizabilities[offset+1], labFramePolarizabilities[offset+3], labFramePolarizabilities[offset+4]);
            else if(component==2)
                alpha = make_real3(labFramePolarizabilities[offset+2], labFramePolarizabilities[offset+4], labFramePolarizabilities[offset+5]);

            real oldDipole = inducedDipole[dipoleIndex];
            real newDipole = fieldScale*dot(alpha, fld);
            int storePrevIndex = dipoleIndex+min(iteration, MAX_PREV_DIIS_DIPOLES-1)*NUM_ATOMS*3;
            prevDipoles[storePrevIndex] = newDipole;
            if (recordPrevErrors)
                prevErrors[storePrevIndex] = newDipole-oldDipole;
            sumErrors += (newDipole-oldDipole)*(newDipole-oldDipole);
        }
    }

    // Sum the errors over threads and store the total for this block.

    buffer[threadIdx.x] = sumErrors;
    __syncthreads();
    for (int offset = 1; offset < blockDim.x; offset *= 2) {
        if (threadIdx.x+offset < blockDim.x && (threadIdx.x&(2*offset-1)) == 0) {
            buffer[threadIdx.x] += buffer[threadIdx.x+offset];
        }
        __syncthreads();
    }
    if (threadIdx.x == 0)
        errors[blockIdx.x] = buffer[0];

    if (iteration >= MAX_PREV_DIIS_DIPOLES && recordPrevErrors && blockIdx.x == 0) {
        // Shift over the existing matrix elements.
        for (int i = 0; i < MAX_PREV_DIIS_DIPOLES-1; i++) {
            if (threadIdx.x < MAX_PREV_DIIS_DIPOLES-1)
                matrix[threadIdx.x+i*MAX_PREV_DIIS_DIPOLES] = matrix[(threadIdx.x+1)+(i+1)*MAX_PREV_DIIS_DIPOLES];
            __syncthreads();
        }
    }
}

extern "C" __global__ void computeDIISMatrix(real* __restrict__ prevErrors, int iteration, real* __restrict__ matrix) {
    extern __shared__ real sumBuffer[];
    int j = min(iteration, MAX_PREV_DIIS_DIPOLES-1);
    for (int i = blockIdx.x; i <= j; i += gridDim.x) {
        // All the threads in this thread block work together to compute a single matrix element.

        real sum = 0;
        for (int index = threadIdx.x; index < NUM_ATOMS*3; index += blockDim.x)
            sum += prevErrors[index+i*NUM_ATOMS*3]*prevErrors[index+j*NUM_ATOMS*3];
        sumBuffer[threadIdx.x] = sum;
        __syncthreads();
        for (int offset = 1; offset < blockDim.x; offset *= 2) { 
            if (threadIdx.x+offset < blockDim.x && (threadIdx.x&(2*offset-1)) == 0)
                sumBuffer[threadIdx.x] += sumBuffer[threadIdx.x+offset];
            __syncthreads();
        }
        if (threadIdx.x == 0) {
            matrix[i+MAX_PREV_DIIS_DIPOLES*j] = sumBuffer[0];
            if (i != j)
                matrix[j+MAX_PREV_DIIS_DIPOLES*i] = sumBuffer[0];
        }
    }
}

extern "C" __global__ void solveDIISMatrix(int iteration, const real* __restrict__ matrix, float* __restrict__ coefficients) {
    __shared__ real b[MAX_PREV_DIIS_DIPOLES+1][MAX_PREV_DIIS_DIPOLES+1];
    __shared__ real piv[MAX_PREV_DIIS_DIPOLES+1];
    __shared__ real x[MAX_PREV_DIIS_DIPOLES+1];

    // On the first iteration we don't need to do any calculation.
    
    if (iteration == 0) {
        if (threadIdx.x == 0)
            coefficients[0] = 1;
        return;
    }
    
    // Load the matrix.
    
    int numPrev = min(iteration+1, MAX_PREV_DIIS_DIPOLES);
    int rank = numPrev+1;
    for (int index = threadIdx.x; index < numPrev*numPrev; index += blockDim.x) {
        int i = index/numPrev;
        int j = index-i*numPrev;
        b[i+1][j+1] = matrix[i*MAX_PREV_DIIS_DIPOLES+j];
    }
    for (int i = threadIdx.x; i < rank; i += blockDim.x) {
        b[i][0] = -1;
        piv[i] = i;
    }
    __syncthreads();
    
    // Compute the mean absolute value of the values we just loaded.  We use that for preconditioning it,
    // which is essential for doing the computation in single precision.
    
    if (threadIdx.x == 0) {
        real mean = 0;
        for (int i = 0; i < numPrev; i++)
            for (int j = 0; j < numPrev; j++)
                mean += fabs(b[i+1][j+1]);
        mean /= numPrev*numPrev;
        b[0][0] = 0;
        for (int i = 1; i < rank; i++)
            b[0][i] = -mean;

        // Compute the LU decomposition of the matrix.  This code is adapted from JAMA.
    
        int pivsign = 1;
        for (int j = 0; j < rank; j++) {
            // Apply previous transformations.

            for (int i = 0; i < rank; i++) {
                // Most of the time is spent in the following dot product.

                int kmax = min(i, j);
                real s = 0;
                for (int k = 0; k < kmax; k++)
                    s += b[i][k] * b[k][j];
                b[i][j] -= s;
            }

            // Find pivot and exchange if necessary.

            int p = j;
            for (int i = j+1; i < rank; i++)
                if (abs(b[i][j]) > abs(b[p][j]))
                    p = i;
            if (p != j) {
                int k = 0;
                for (k = 0; k < rank; k++) {
                    real t = b[p][k];
                    b[p][k] = b[j][k];
                    b[j][k] = t;
                }
                k = piv[p];
                piv[p] = piv[j];
                piv[j] = k;
                pivsign = -pivsign;
            }

            // Compute multipliers.

            if ((j < rank) && (b[j][j] != 0))
                for (int i = j+1; i < rank; i++)
                    b[i][j] /= b[j][j];
        }
        for (int i = 0; i < rank; i++)
            if (b[i][i] == 0) {
                // The matrix is singular.
                
                for (int j = 0; j < rank-1; j++)
                    coefficients[j] = 0;
                coefficients[rank-1] = 1;
                return;
            }

        // Solve b*Y = X(piv)
        
        for (int i = 0; i < rank; i++) 
            x[i] = (piv[i] == 0 ? -1 : 0);
        for (int k = 0; k < rank; k++)
            for (int i = k+1; i < rank; i++)
                x[i] -= x[k] * b[i][k];

        // Solve U*X = Y;
        
        for (int k = rank-1; k >= 0; k--) {
            x[k] /= b[k][k];
            for (int i = 0; i < k; i++)
                x[i] -= x[k] * b[i][k];
        }
        
        // Record the coefficients.
        
        real lastCoeff = 1;
        for (int i = 0; i < rank-1; i++) {
            real c = x[i+1]*mean;
            coefficients[i] = c;
            lastCoeff -= c;
        }
        coefficients[rank-1] = lastCoeff;
    }
}

extern "C" __global__ void updateInducedFieldByDIIS(real* __restrict__ inducedDipole, const real* __restrict__ prevDipoles, 
                                                    const float* __restrict__ coefficients, int numPrev) {
    for (int index = blockIdx.x*blockDim.x + threadIdx.x; index < 3*NUM_ATOMS; index += blockDim.x*gridDim.x) {
        real sum = 0;
        for (int i = 0; i < numPrev; i++) {
            sum += coefficients[i]*prevDipoles[i*3*NUM_ATOMS+index];
        }
        inducedDipole[index] = sum;
    }
}

extern "C" __global__ void initExtrapolatedDipoles(real* __restrict__ inducedDipole, real* __restrict__ extrapolatedDipole,
        long long* __restrict__ inducedDipoleFieldGradient) {
    for (int index = blockIdx.x*blockDim.x+threadIdx.x; index < 3*NUM_ATOMS; index += blockDim.x*gridDim.x) {
        extrapolatedDipole[index] = inducedDipole[index];
    }
    for (int index = blockIdx.x*blockDim.x+threadIdx.x; index < 6*NUM_ATOMS; index += blockDim.x*gridDim.x) {
        inducedDipoleFieldGradient[index] = 0;
    }
}

extern "C" __global__ void iterateExtrapolatedDipoles(int order, real* __restrict__ inducedDipole, real* __restrict__ extrapolatedDipole,
        long long* __restrict__ inducedDipoleFieldGradient, long long* __restrict__ inducedDipoleField,
        real* __restrict__ extrapolatedDipoleField, real* __restrict__ extrapolatedDipoleFieldGradient, const real* __restrict__ labFramePolarizabilities) {
    const real fieldScale = 1/(real) 0x100000000;
    for (int atom = blockIdx.x*blockDim.x + threadIdx.x; atom < NUM_ATOMS; atom += blockDim.x*gridDim.x) {
        int offset = 6*atom;
        real3 fld = make_real3(inducedDipoleField[atom+0*PADDED_NUM_ATOMS],
                               inducedDipoleField[atom+1*PADDED_NUM_ATOMS],
                               inducedDipoleField[atom+2*PADDED_NUM_ATOMS]);
        for (int component = 0; component < 3; component++) {
            // Compute the new dipole, and record it along with the error.
            real3 alpha;
            if(component==0)
                alpha = make_real3(labFramePolarizabilities[offset+0], labFramePolarizabilities[offset+1], labFramePolarizabilities[offset+2]);
            else if(component==1)
                alpha = make_real3(labFramePolarizabilities[offset+1], labFramePolarizabilities[offset+3], labFramePolarizabilities[offset+4]);
            else if(component==2)
                alpha = make_real3(labFramePolarizabilities[offset+2], labFramePolarizabilities[offset+4], labFramePolarizabilities[offset+5]);
            real value = dot(alpha,fld)*fieldScale;
            inducedDipole[3*atom+component] = value;
            extrapolatedDipole[order*3*NUM_ATOMS+3*atom+component] = value;
        }

    }
    for (int index = blockIdx.x*blockDim.x+threadIdx.x; index < 3*NUM_ATOMS; index += blockDim.x*gridDim.x) {
        int index2 = (order-1)*3*NUM_ATOMS+index;
        int atom = index/3;
        int component = index%3;
        extrapolatedDipoleField[index2] = fieldScale*inducedDipoleField[atom+component*PADDED_NUM_ATOMS];
    }
    for (int index = blockIdx.x*blockDim.x+threadIdx.x; index < 6*NUM_ATOMS; index += blockDim.x*gridDim.x) {
        int index2 = (order-1)*6*NUM_ATOMS+index;
        extrapolatedDipoleFieldGradient[index2] = fieldScale*inducedDipoleFieldGradient[index];
    }
}

extern "C" __global__ void computeExtrapolatedDipoles(real* __restrict__ inducedDipole, real* __restrict__ extrapolatedDipole) {
    real coeff[] = {EXTRAPOLATION_COEFFICIENTS_SUM};
    for (int index = blockIdx.x*blockDim.x+threadIdx.x; index < 3*NUM_ATOMS; index += blockDim.x*gridDim.x) {
        real sum = 0;
        for (int order = 0; order < MAX_EXTRAPOLATION_ORDER; order++) {
            sum += extrapolatedDipole[order*3*NUM_ATOMS+index]*coeff[order];
        }
        inducedDipole[index] = sum;
    }
}

extern "C" __global__ void addExtrapolatedFieldGradientToForce(long long* __restrict__ forceBuffers,
        long long* __restrict__ torqueBuffers,
        const float2* __restrict__ dampingAndThole,
        real* __restrict__ extrapolatedDipole,
        real* __restrict__ extrapolatedDipoleField, real* __restrict__ extrapolatedDipoleFieldGradient ) {
    real coeff[] = {EXTRAPOLATION_COEFFICIENTS_SUM};
    for (int atom = blockIdx.x*blockDim.x+threadIdx.x; atom < NUM_ATOMS; atom += blockDim.x*gridDim.x) {
        real fx = 0, fy = 0, fz = 0;
        real tx = 0, ty = 0, tz = 0;
        bool isAnisotropic = dampingAndThole[atom].x < 0 ? false : true;
        for (int l = 0; l < MAX_EXTRAPOLATION_ORDER-1; l++) {
            int index1 = 3*(l*NUM_ATOMS+atom);
            real dipole[] = {extrapolatedDipole[index1], extrapolatedDipole[index1+1], extrapolatedDipole[index1+2]};
            for (int m = 0; m < MAX_EXTRAPOLATION_ORDER-1-l; m++) {
                int index2 = 6*(m*NUM_ATOMS+atom);
                int index3 = 3*(m*NUM_ATOMS+atom);
                real scale = coeff[l+m+1]*ENERGY_SCALE_FACTOR;
                real fieldGradient[] = {extrapolatedDipoleFieldGradient[index2], extrapolatedDipoleFieldGradient[index2+1], extrapolatedDipoleFieldGradient[index2+2],
                                   extrapolatedDipoleFieldGradient[index2+3], extrapolatedDipoleFieldGradient[index2+4], extrapolatedDipoleFieldGradient[index2+5]};
                real dipoleField[] = {extrapolatedDipoleField[index3], extrapolatedDipoleField[index3+1], extrapolatedDipoleField[index3+2]};
                fx += scale*(dipole[0]*fieldGradient[0] + dipole[1]*fieldGradient[3] + dipole[2]*fieldGradient[4]);
                fy += scale*(dipole[0]*fieldGradient[3] + dipole[1]*fieldGradient[1] + dipole[2]*fieldGradient[5]);
                fz += scale*(dipole[0]*fieldGradient[4] + dipole[1]*fieldGradient[5] + dipole[2]*fieldGradient[2]);
                tx += scale*(dipole[1]*dipoleField[2] - dipole[2]*dipoleField[1]);
                ty += scale*(dipole[2]*dipoleField[0] - dipole[0]*dipoleField[2]);
                tz += scale*(dipole[0]*dipoleField[1] - dipole[1]*dipoleField[0]);
            }
        }
        forceBuffers[atom] += (long long) (fx*0x100000000);
        forceBuffers[atom+PADDED_NUM_ATOMS] += (long long) (fy*0x100000000);
        forceBuffers[atom+PADDED_NUM_ATOMS*2] += (long long) (fz*0x100000000);
        torqueBuffers[atom] += (long long) (isAnisotropic?tx*0x100000000:0);
        torqueBuffers[atom+PADDED_NUM_ATOMS] += (long long) (isAnisotropic?ty*0x100000000:0);
        torqueBuffers[atom+PADDED_NUM_ATOMS*2] += (long long) (isAnisotropic?tz*0x100000000:0);
    }
}
