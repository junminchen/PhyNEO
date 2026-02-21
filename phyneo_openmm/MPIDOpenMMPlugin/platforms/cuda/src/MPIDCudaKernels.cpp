/* -------------------------------------------------------------------------- *
 *                               OpenMMMPID                                 *
 * -------------------------------------------------------------------------- *
 * This is part of the OpenMM molecular simulation toolkit originating from   *
 * Simbios, the NIH National Center for Physics-Based Simulation of           *
 * Biological Structures at Stanford, funded under the NIH Roadmap for        *
 * Medical Research, grant U54 GM072970. See https://simtk.org.               *
 *                                                                            *
 * Portions copyright (c) 2008-2016 Stanford University and the Authors.      *
 * Authors: Peter Eastman, Mark Friedrichs                                    *
 * Contributors:                                                              *
 *                                                                            *
 * This program is free software: you can redistribute it and/or modify       *
 * it under the terms of the GNU Lesser General Public License as published   *
 * by the Free Software Foundation, either version 3 of the License, or       *
 * (at your option) any later version.                                        *
 *                                                                            *
 * This program is distributed in the hope that it will be useful,            *
 * but WITHOUT ANY WARRANTY; without even the implied warranty of             *
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the              *
 * GNU Lesser General Public License for more details.                        *
 *                                                                            *
 * You should have received a copy of the GNU Lesser General Public License   *
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.      *
 * -------------------------------------------------------------------------- */

#ifdef WIN32
  #define _USE_MATH_DEFINES // Needed to get M_PI
#endif
#include "MPIDCudaKernels.h"
#include "CudaMPIDKernelSources.h"
#include "openmm/internal/ContextImpl.h"
#include "openmm/internal/MPIDForceImpl.h"
#include "openmm/internal/NonbondedForceImpl.h"
#include "CudaBondedUtilities.h"
#include "CudaFFT3D.h"
#include "CudaForceInfo.h"
#include "CudaNonbondedUtilities.h"
#include "jama_lu.h"

#include <algorithm>
#include <cmath>
#ifdef _MSC_VER
#include <windows.h>
#endif

using namespace OpenMM;
using namespace std;

#define CHECK_RESULT(result, prefix) \
    if (result != CUDA_SUCCESS) { \
        std::stringstream m; \
        m<<prefix<<": "<<cu.getErrorString(result)<<" ("<<result<<")"<<" at "<<__FILE__<<":"<<__LINE__; \
        throw OpenMMException(m.str());\
    }

/* -------------------------------------------------------------------------- *
 *                             MPIDMultipole                                *
 * -------------------------------------------------------------------------- */

class CudaCalcMPIDForceKernel::ForceInfo : public CudaForceInfo {
public:
    ForceInfo(const MPIDForce& force) : force(force) {
    }
    bool areParticlesIdentical(int particle1, int particle2) {
        double charge1, charge2, thole1, thole2;
        int axis1, axis2, multipole11, multipole12, multipole21, multipole22, multipole31, multipole32;
        vector<double> dipole1, dipole2, quadrupole1, quadrupole2, octopole1, octopole2, polarity1, polarity2;
        force.getMultipoleParameters(particle1, charge1, dipole1, quadrupole1, octopole1, axis1, multipole11, multipole21, multipole31, thole1, polarity1);
        force.getMultipoleParameters(particle2, charge2, dipole2, quadrupole2, octopole2, axis2, multipole12, multipole22, multipole32, thole2, polarity2);
        if (charge1 != charge2 || thole1 != thole2 || polarity1 != polarity2 || axis1 != axis2) {
            return false;
        }
        for (int i = 0; i < (int) dipole1.size(); ++i) {
            if (dipole1[i] != dipole2[i]) {
                return false;
            }
        }
        for (int i = 0; i < (int) quadrupole1.size(); ++i) {
            if (quadrupole1[i] != quadrupole2[i]) {
                return false;
            }
        }
        for (int i = 0; i < (int) octopole1.size(); ++i) {
            if (octopole1[i] != octopole2[i]) {
                return false;
            }
        }
        return true;
    }
    int getNumParticleGroups() {
        return 7*force.getNumMultipoles();
    }
    void getParticlesInGroup(int index, vector<int>& particles) {
        int particle = index/7;
        int type = index-7*particle;
        force.getCovalentMap(particle, MPIDForce::CovalentType(type), particles);
    }
    bool areGroupsIdentical(int group1, int group2) {
        return ((group1%7) == (group2%7));
    }
private:
    const MPIDForce& force;
};

CudaCalcMPIDForceKernel::CudaCalcMPIDForceKernel(std::string name, const Platform& platform, CudaContext& cu, const System& system) : 
        CalcMPIDForceKernel(name, platform), cu(cu), system(system), hasInitializedScaleFactors(false), hasInitializedFFT(false), multipolesAreValid(false), hasCreatedEvent(false),
        multipoleParticles(NULL), molecularDipoles(NULL), molecularQuadrupoles(NULL), molecularOctopoles(NULL), labFramePolarizabilities(NULL),  labFrameDipoles(NULL), labFrameQuadrupoles(NULL), labFrameOctopoles(NULL), sphericalDipoles(NULL), sphericalQuadrupoles(NULL), sphericalOctopoles(NULL),
        fracDipoles(NULL), fracQuadrupoles(NULL), fracOctopoles(NULL), field(NULL), inducedField(NULL), torque(NULL), dampingAndThole(NULL), inducedDipole(NULL),
        diisCoefficients(NULL), inducedDipoleErrors(NULL), prevDipoles(NULL),
        prevErrors(NULL), diisMatrix(NULL), polarizability(NULL), extrapolatedDipole(NULL),
        inducedDipoleFieldGradient(NULL), extrapolatedDipoleField(NULL), extrapolatedDipoleFieldGradient(NULL),
        covalentFlags(NULL),
        pmeGrid(NULL), pmeBsplineModuliX(NULL), pmeBsplineModuliY(NULL), pmeBsplineModuliZ(NULL), pmeIgrid(NULL), pmePhi(NULL),
        pmePhid(NULL), pmePhip(NULL), pmePhidp(NULL), pmeCphi(NULL), lastPositions(NULL), sort(NULL) {
}

CudaCalcMPIDForceKernel::~CudaCalcMPIDForceKernel() {
    cu.setAsCurrent();
    if (multipoleParticles != NULL)
        delete multipoleParticles;
    if (molecularDipoles != NULL)
        delete molecularDipoles;
    if (molecularQuadrupoles != NULL)
        delete molecularQuadrupoles;
    if (molecularOctopoles != NULL)
        delete molecularOctopoles;
    if (labFramePolarizabilities != NULL)
        delete labFramePolarizabilities;
    if (labFrameDipoles != NULL)
        delete labFrameDipoles;
    if (labFrameQuadrupoles != NULL)
        delete labFrameQuadrupoles;
    if (labFrameOctopoles != NULL)
        delete labFrameOctopoles;
    if (sphericalDipoles != NULL)
        delete sphericalDipoles;
    if (sphericalQuadrupoles != NULL)
        delete sphericalQuadrupoles;
    if (sphericalOctopoles != NULL)
        delete sphericalOctopoles;
    if (fracDipoles != NULL)
        delete fracDipoles;
    if (fracQuadrupoles != NULL)
        delete fracQuadrupoles;
    if (fracOctopoles != NULL)
        delete fracOctopoles;
    if (field != NULL)
        delete field;
    if (inducedField != NULL)
        delete inducedField;
    if (torque != NULL)
        delete torque;
    if (dampingAndThole != NULL)
        delete dampingAndThole;
    if (inducedDipole != NULL)
        delete inducedDipole;
    if (inducedDipoleErrors != NULL)
        delete inducedDipoleErrors;
    if (prevDipoles != NULL)
        delete prevDipoles;
    if (prevErrors != NULL)
        delete prevErrors;
    if (diisMatrix != NULL)
        delete diisMatrix;
    if (diisCoefficients != NULL)
        delete diisCoefficients;
    if (extrapolatedDipole != NULL)
        delete extrapolatedDipole;
    if (inducedDipoleFieldGradient != NULL)
        delete inducedDipoleFieldGradient;
    if (extrapolatedDipoleField != NULL)
        delete extrapolatedDipoleField;
    if (extrapolatedDipoleFieldGradient != NULL)
        delete extrapolatedDipoleFieldGradient;
    if (polarizability != NULL)
        delete polarizability;
    if (covalentFlags != NULL)
        delete covalentFlags;
    if (pmeGrid != NULL)
        delete pmeGrid;
    if (pmeBsplineModuliX != NULL)
        delete pmeBsplineModuliX;
    if (pmeBsplineModuliY != NULL)
        delete pmeBsplineModuliY;
    if (pmeBsplineModuliZ != NULL)
        delete pmeBsplineModuliZ;
    if (pmeIgrid != NULL)
        delete pmeIgrid;
    if (pmePhi != NULL)
        delete pmePhi;
    if (pmePhid != NULL)
        delete pmePhid;
    if (pmePhip != NULL)
        delete pmePhip;
    if (pmePhidp != NULL)
        delete pmePhidp;
    if (pmeCphi != NULL)
        delete pmeCphi;
    if (lastPositions != NULL)
        delete lastPositions;
    if (sort != NULL)
        delete sort;
    if (hasInitializedFFT)
        cufftDestroy(fft);
    if (hasCreatedEvent)
        cuEventDestroy(syncEvent);
}

void CudaCalcMPIDForceKernel::initialize(const System& system, const MPIDForce& force) {
    cu.setAsCurrent();

    // Initialize multipole parameters.

    numMultipoles = force.getNumMultipoles();
    CudaArray& posq = cu.getPosq();
    vector<double4> temp(posq.getSize());
    float4* posqf = (float4*) &temp[0];
    double4* posqd = (double4*) &temp[0];
    vector<float2> dampingAndTholeVec;
    vector<float> polarizabilityVec;
    vector<float> molecularDipolesVec;
    vector<float> molecularQuadrupolesVec;
    vector<float> molecularOctopolesVec;
    vector<int4> multipoleParticlesVec;
    for (int i = 0; i < numMultipoles; i++) {
        double charge, thole, damping;
        int axisType, atomX, atomY, atomZ;
        vector<double> dipole, quadrupole, octopole, polarity;
        force.getMultipoleParameters(i, charge, dipole, quadrupole, octopole, axisType, atomZ, atomX, atomY, thole, polarity);
        if (cu.getUseDoublePrecision())
            posqd[i] = make_double4(0, 0, 0, charge);
        else
            posqf[i] = make_float4(0, 0, 0, (float) charge);
        damping = pow((polarity[0]+polarity[1]+polarity[2])/3.0, 1.0/6.0);
        // If the multipole is isotropic, we flag it with a negative damping factor (which should always be positive) so we
        // can easily identify in the pairwise loops when we should be adding torque terms for induced dipoles.
        if(polarity[0]==polarity[1] && polarity[0]==polarity[2])
            damping *= -1;
        dampingAndTholeVec.push_back(make_float2((float) damping, (float) thole));
        for (int j = 0; j < 3; j++)
            polarizabilityVec.push_back((float) polarity[j]);
        multipoleParticlesVec.push_back(make_int4(atomX, atomY, atomZ, axisType));
        for (int j = 0; j < 3; j++)
            molecularDipolesVec.push_back((float) dipole[j]);
        molecularQuadrupolesVec.push_back((float) quadrupole[0]); // XX
        molecularQuadrupolesVec.push_back((float) quadrupole[1]); // XY
        molecularQuadrupolesVec.push_back((float) quadrupole[3]); // XZ
        molecularQuadrupolesVec.push_back((float) quadrupole[2]); // YY
        molecularQuadrupolesVec.push_back((float) quadrupole[4]); // YZ
        molecularOctopolesVec.push_back((float) octopole[0]); // XXX
        molecularOctopolesVec.push_back((float) octopole[1]); // XXY
        molecularOctopolesVec.push_back((float) octopole[4]); // XXZ
        molecularOctopolesVec.push_back((float) octopole[2]); // XYY
        molecularOctopolesVec.push_back((float) octopole[5]); // XYZ
        molecularOctopolesVec.push_back((float) octopole[3]); // YYY
        molecularOctopolesVec.push_back((float) octopole[6]); // YYZ
    }
    hasQuadrupoles = false;
    for (auto q : molecularQuadrupolesVec)
        if (q != 0.0)
            hasQuadrupoles = true;
    hasOctopoles = false;
    for (auto q : molecularOctopolesVec)
        if (q != 0.0){
            hasQuadrupoles = true;
            hasOctopoles = true;
        }
    int paddedNumAtoms = cu.getPaddedNumAtoms();
    for (int i = numMultipoles; i < paddedNumAtoms; i++) {
        dampingAndTholeVec.push_back(make_float2(0, 0));
        for (int j = 0; j < 3; j++)
            polarizabilityVec.push_back(0);
        multipoleParticlesVec.push_back(make_int4(0, 0, 0, 0));
        for (int j = 0; j < 3; j++)
            molecularDipolesVec.push_back(0);
        for (int j = 0; j < 5; j++)
            molecularQuadrupolesVec.push_back(0);
        for (int j = 0; j < 7; j++)
            molecularOctopolesVec.push_back(0);
    }
    dampingAndThole = CudaArray::create<float2>(cu, paddedNumAtoms, "dampingAndThole");
    polarizability = CudaArray::create<float>(cu, 3*paddedNumAtoms, "polarizability");
    multipoleParticles = CudaArray::create<int4>(cu, paddedNumAtoms, "multipoleParticles");
    molecularDipoles = CudaArray::create<float>(cu, 3*paddedNumAtoms, "molecularDipoles");
    molecularQuadrupoles = CudaArray::create<float>(cu, 5*paddedNumAtoms, "molecularQuadrupoles");
    molecularOctopoles = CudaArray::create<float>(cu, 7*paddedNumAtoms, "molecularOctopoles");
    lastPositions = new CudaArray(cu, cu.getPosq().getSize(), cu.getPosq().getElementSize(), "lastPositions");
    dampingAndThole->upload(dampingAndTholeVec);
    polarizability->upload(polarizabilityVec);
    multipoleParticles->upload(multipoleParticlesVec);
    molecularDipoles->upload(molecularDipolesVec);
    molecularQuadrupoles->upload(molecularQuadrupolesVec);
    molecularOctopoles->upload(molecularOctopolesVec);
    posq.upload(&temp[0]);

    // Create workspace arrays.

    polarizationType = force.getPolarizationType();
    int elementSize = (cu.getUseDoublePrecision() ? sizeof(double) : sizeof(float));
    labFramePolarizabilities = new CudaArray(cu, 6*paddedNumAtoms, elementSize, "labFramePolarizabilities");
    labFrameDipoles = new CudaArray(cu, 3*paddedNumAtoms, elementSize, "labFrameDipoles");
    labFrameQuadrupoles = new CudaArray(cu, 5*paddedNumAtoms, elementSize, "labFrameQuadrupoles");
    labFrameOctopoles = new CudaArray(cu, 7*paddedNumAtoms, elementSize, "labFrameOctopoles");
    sphericalDipoles = new CudaArray(cu, 3*paddedNumAtoms, elementSize, "sphericalDipoles");
    sphericalQuadrupoles = new CudaArray(cu, 5*paddedNumAtoms, elementSize, "sphericalQuadrupoles");
    sphericalOctopoles = new CudaArray(cu, 7*paddedNumAtoms, elementSize, "sphericalOctopoles");
    fracDipoles = new CudaArray(cu, 3*paddedNumAtoms, elementSize, "fracDipoles");
    fracQuadrupoles = new CudaArray(cu, 6*paddedNumAtoms, elementSize, "fracQuadrupoles");
    fracOctopoles = new CudaArray(cu, 10*paddedNumAtoms, elementSize, "fracOctopoles");
    field = new CudaArray(cu, 3*paddedNumAtoms, sizeof(long long), "field");
    torque = new CudaArray(cu, 3*paddedNumAtoms, sizeof(long long), "torque");
    inducedDipole = new CudaArray(cu, 3*paddedNumAtoms, elementSize, "inducedDipole");
    if (polarizationType == MPIDForce::Mutual) {
        inducedDipoleErrors = new CudaArray(cu, cu.getNumThreadBlocks(), sizeof(float2), "inducedDipoleErrors");
        prevDipoles = new CudaArray(cu, 3*numMultipoles*MaxPrevDIISDipoles, elementSize, "prevDipoles");
        prevErrors = new CudaArray(cu, 3*numMultipoles*MaxPrevDIISDipoles, elementSize, "prevErrors");
        diisMatrix = new CudaArray(cu, MaxPrevDIISDipoles*MaxPrevDIISDipoles, elementSize, "diisMatrix");
        diisCoefficients = new CudaArray(cu, MaxPrevDIISDipoles+1, sizeof(float), "diisMatrix");
        CHECK_RESULT(cuEventCreate(&syncEvent, CU_EVENT_DISABLE_TIMING), "Error creating event for MPIDForce");
        hasCreatedEvent = true;
    }
    else if (polarizationType == MPIDForce::Extrapolated) {
        int numOrders = force.getExtrapolationCoefficients().size();
        extrapolatedDipole = new CudaArray(cu, 3*numMultipoles*numOrders, elementSize, "extrapolatedDipole");
        inducedDipoleFieldGradient = new CudaArray(cu, 6*paddedNumAtoms, sizeof(long long), "inducedDipoleFieldGradient");
        extrapolatedDipoleField = new CudaArray(cu, 3*numMultipoles*(numOrders-1), elementSize, "extrapolatedDipoleField");
        extrapolatedDipoleFieldGradient = new CudaArray(cu, 6*numMultipoles*(numOrders-1), elementSize, "extrapolatedDipoleFieldGradient");
    }
    cu.addAutoclearBuffer(*field);
    cu.addAutoclearBuffer(*torque);

    // Record which atoms should be flagged as exclusions based on covalent groups, and determine
    // the values for the covalent group flags.

    vector<vector<int> > exclusions(numMultipoles);
    for (int i = 0; i < numMultipoles; i++) {
        vector<int> atoms;
        set<int> allAtoms;
        allAtoms.insert(i);
        force.getCovalentMap(i, MPIDForce::Covalent12, atoms);
        allAtoms.insert(atoms.begin(), atoms.end());
        force.getCovalentMap(i, MPIDForce::Covalent13, atoms);
        allAtoms.insert(atoms.begin(), atoms.end());
        for (int atom : allAtoms)
            covalentFlagValues.push_back(make_int3(i, atom, 0));
        force.getCovalentMap(i, MPIDForce::Covalent14, atoms);
        allAtoms.insert(atoms.begin(), atoms.end());
        for (int atom : atoms)
            covalentFlagValues.push_back(make_int3(i, atom, 1));
        force.getCovalentMap(i, MPIDForce::Covalent15, atoms);
        for (int atom : atoms)
            covalentFlagValues.push_back(make_int3(i, atom, 2));
        allAtoms.insert(atoms.begin(), atoms.end());
        force.getCovalentMap(i, MPIDForce::PolarizationCovalent11, atoms);
        allAtoms.insert(atoms.begin(), atoms.end());
        exclusions[i].insert(exclusions[i].end(), allAtoms.begin(), allAtoms.end());

        // Workaround for bug in TINKER: if an atom is listed in both the PolarizationCovalent11
        // and PolarizationCovalent12 maps, the latter takes precedence.

        vector<int> atoms12;
        force.getCovalentMap(i, MPIDForce::PolarizationCovalent12, atoms12);
        for (int atom : atoms)
            if (find(atoms12.begin(), atoms12.end(), atom) == atoms12.end())
                polarizationFlagValues.push_back(make_int2(i, atom));
    }
    set<pair<int, int> > tilesWithExclusions;
    for (int atom1 = 0; atom1 < (int) exclusions.size(); ++atom1) {
        int x = atom1/CudaContext::TileSize;
        for (int atom2 : exclusions[atom1]) {
            int y = atom2/CudaContext::TileSize;
            tilesWithExclusions.insert(make_pair(max(x, y), min(x, y)));
        }
    }

    // Record other options.

    if (polarizationType == MPIDForce::Mutual) {
        maxInducedIterations = force.getMutualInducedMaxIterations();
        inducedEpsilon = force.getMutualInducedTargetEpsilon();
    }
    else
        maxInducedIterations = 0;
    if (polarizationType != MPIDForce::Direct) {
        inducedField = new CudaArray(cu, 3*paddedNumAtoms, sizeof(long long), "inducedField");
    }
    usePME = (force.getNonbondedMethod() == MPIDForce::PME);

    // Create the kernels.

    bool useShuffle = (cu.getComputeCapability() >= 3.0 && !cu.getUseDoublePrecision());
    double fixedThreadMemory = 26*elementSize+2*sizeof(float)+3*sizeof(int)/(double) cu.TileSize;
    double inducedThreadMemory = 11*elementSize+sizeof(float);
    if (polarizationType == MPIDForce::Extrapolated)
        inducedThreadMemory += 9*elementSize;
    double electrostaticsThreadMemory = 0;
    if (!useShuffle)
        fixedThreadMemory += 3*elementSize;
    map<string, string> defines;
    defines["NUM_ATOMS"] = cu.intToString(numMultipoles);
    defines["PADDED_NUM_ATOMS"] = cu.intToString(cu.getPaddedNumAtoms());
    defines["NUM_BLOCKS"] = cu.intToString(cu.getNumAtomBlocks());
    defines["ENERGY_SCALE_FACTOR"] = cu.doubleToString(138.9354558456);
    defines["DEFAULT_THOLE_WIDTH"] = cu.doubleToString(force.getDefaultTholeWidth());
    if (polarizationType == MPIDForce::Direct)
        defines["DIRECT_POLARIZATION"] = "";
    else if (polarizationType == MPIDForce::Mutual)
        defines["MUTUAL_POLARIZATION"] = "";
    else if (polarizationType == MPIDForce::Extrapolated)
        defines["EXTRAPOLATED_POLARIZATION"] = "";
    if (useShuffle)
        defines["USE_SHUFFLE"] = "";
    if (hasQuadrupoles)
        defines["INCLUDE_QUADRUPOLES"] = "";
    if (hasOctopoles)
        defines["INCLUDE_OCTOPOLES"] = "";
    defines["TILE_SIZE"] = cu.intToString(CudaContext::TileSize);
    int numExclusionTiles = tilesWithExclusions.size();
    defines["NUM_TILES_WITH_EXCLUSIONS"] = cu.intToString(numExclusionTiles);
    int numContexts = cu.getPlatformData().contexts.size();
    int startExclusionIndex = cu.getContextIndex()*numExclusionTiles/numContexts;
    int endExclusionIndex = (cu.getContextIndex()+1)*numExclusionTiles/numContexts;
    defines["FIRST_EXCLUSION_TILE"] = cu.intToString(startExclusionIndex);
    defines["LAST_EXCLUSION_TILE"] = cu.intToString(endExclusionIndex);
    maxExtrapolationOrder = force.getExtrapolationCoefficients().size();
    defines["MAX_EXTRAPOLATION_ORDER"] = cu.intToString(maxExtrapolationOrder);
    stringstream coefficients;
    for (int i = 0; i < maxExtrapolationOrder; i++) {
        if (i > 0)
            coefficients << ",";
        double sum = 0;
        for (int j = i; j < maxExtrapolationOrder; j++)
            sum += force.getExtrapolationCoefficients()[j];
        coefficients << cu.doubleToString(sum);
    }
    defines["EXTRAPOLATION_COEFFICIENTS_SUM"] = coefficients.str();
    if (usePME) {
        int nx, ny, nz;
        force.getPMEParameters(alpha, nx, ny, nz);
        if (nx == 0 || alpha == 0.0) {
            NonbondedForce nb;
            nb.setEwaldErrorTolerance(force.getEwaldErrorTolerance());
            nb.setCutoffDistance(force.getCutoffDistance());
            NonbondedForceImpl::calcPMEParameters(system, nb, alpha, gridSizeX, gridSizeY, gridSizeZ, false);
            gridSizeX = CudaFFT3D::findLegalDimension(gridSizeX);
            gridSizeY = CudaFFT3D::findLegalDimension(gridSizeY);
            gridSizeZ = CudaFFT3D::findLegalDimension(gridSizeZ);
        } else {
            gridSizeX = CudaFFT3D::findLegalDimension(nx);
            gridSizeY = CudaFFT3D::findLegalDimension(ny);
            gridSizeZ = CudaFFT3D::findLegalDimension(nz);
        }
        defines["EWALD_ALPHA"] = cu.doubleToString(alpha);
        defines["SQRT_PI"] = cu.doubleToString(sqrt(M_PI));
        defines["USE_EWALD"] = "";
        defines["USE_CUTOFF"] = "";
        defines["USE_PERIODIC"] = "";
        defines["CUTOFF_SQUARED"] = cu.doubleToString(force.getCutoffDistance()*force.getCutoffDistance());
    }
    defines["SCALEFACTOR14"] = cu.doubleToString(force.get14ScaleFactor());
    int maxThreads = cu.getNonbondedUtilities().getForceThreadBlockSize();
    fixedFieldThreads = min(maxThreads, cu.computeThreadBlockSize(fixedThreadMemory));
    inducedFieldThreads = min(maxThreads, cu.computeThreadBlockSize(inducedThreadMemory));
    CUmodule module = cu.createModule(CudaMPIDKernelSources::vectorOps+CudaMPIDKernelSources::multipoles, defines);
    computeMomentsKernel = cu.getKernel(module, "computeLabFrameMoments");
    recordInducedDipolesKernel = cu.getKernel(module, "recordInducedDipoles");
    mapTorqueKernel = cu.getKernel(module, "mapTorqueToForce");
    computePotentialKernel = cu.getKernel(module, "computePotentialAtPoints");
    defines["THREAD_BLOCK_SIZE"] = cu.intToString(fixedFieldThreads);
    module = cu.createModule(CudaMPIDKernelSources::vectorOps+CudaMPIDKernelSources::multipoleFixedField, defines);
    computeFixedFieldKernel = cu.getKernel(module, "computeFixedField");
    if (polarizationType != MPIDForce::Direct) {
        defines["THREAD_BLOCK_SIZE"] = cu.intToString(inducedFieldThreads);
        defines["MAX_PREV_DIIS_DIPOLES"] = cu.intToString(MaxPrevDIISDipoles);
        module = cu.createModule(CudaMPIDKernelSources::vectorOps+CudaMPIDKernelSources::multipoleInducedField, defines);
        computeInducedFieldKernel = cu.getKernel(module, "computeInducedField");
        updateInducedFieldKernel = cu.getKernel(module, "updateInducedFieldByDIIS");
        recordDIISDipolesKernel = cu.getKernel(module, "recordInducedDipolesForDIIS");
        buildMatrixKernel = cu.getKernel(module, "computeDIISMatrix");
        solveMatrixKernel = cu.getKernel(module, "solveDIISMatrix");
        initExtrapolatedKernel = cu.getKernel(module, "initExtrapolatedDipoles");
        iterateExtrapolatedKernel = cu.getKernel(module, "iterateExtrapolatedDipoles");
        computeExtrapolatedKernel = cu.getKernel(module, "computeExtrapolatedDipoles");
        addExtrapolatedGradientKernel = cu.getKernel(module, "addExtrapolatedFieldGradientToForce");
    }
    stringstream electrostaticsSource;
    electrostaticsSource << CudaMPIDKernelSources::vectorOps;
    electrostaticsSource << CudaMPIDKernelSources::sphericalMultipoles;
    if (usePME)
        electrostaticsSource << CudaMPIDKernelSources::pmeMultipoleElectrostatics;
    else
        electrostaticsSource << CudaMPIDKernelSources::multipoleElectrostatics;
    electrostaticsThreadMemory = 30*elementSize+3*sizeof(float)+3*sizeof(int)/(double) cu.TileSize;
    electrostaticsThreads = min(maxThreads, cu.computeThreadBlockSize(electrostaticsThreadMemory));
    defines["THREAD_BLOCK_SIZE"] = cu.intToString(electrostaticsThreads);
    module = cu.createModule(electrostaticsSource.str(), defines);
    electrostaticsKernel = cu.getKernel(module, "computeElectrostatics");

    // Set up PME.

    if (usePME) {
        // Create the PME kernels.

        map<string, string> pmeDefines;
        pmeDefines["EWALD_ALPHA"] = cu.doubleToString(alpha);
        pmeDefines["PME_ORDER"] = cu.intToString(PmeOrder);
        pmeDefines["NUM_ATOMS"] = cu.intToString(numMultipoles);
        pmeDefines["PADDED_NUM_ATOMS"] = cu.intToString(cu.getPaddedNumAtoms());
        pmeDefines["EPSILON_FACTOR"] = cu.doubleToString(138.9354558456);
        pmeDefines["GRID_SIZE_X"] = cu.intToString(gridSizeX);
        pmeDefines["GRID_SIZE_Y"] = cu.intToString(gridSizeY);
        pmeDefines["GRID_SIZE_Z"] = cu.intToString(gridSizeZ);
        pmeDefines["M_PI"] = cu.doubleToString(M_PI);
        pmeDefines["SQRT_PI"] = cu.doubleToString(sqrt(M_PI));
        if (polarizationType == MPIDForce::Direct)
            pmeDefines["DIRECT_POLARIZATION"] = "";
        else if (polarizationType == MPIDForce::Mutual)
            pmeDefines["MUTUAL_POLARIZATION"] = "";
        else if (polarizationType == MPIDForce::Extrapolated)
            pmeDefines["EXTRAPOLATED_POLARIZATION"] = "";
        CUmodule module = cu.createModule(CudaMPIDKernelSources::vectorOps+CudaMPIDKernelSources::multipolePme, pmeDefines);
        pmeTransformMultipolesKernel = cu.getKernel(module, "transformMultipolesToFractionalCoordinates");
        pmeTransformPotentialKernel = cu.getKernel(module, "transformPotentialToCartesianCoordinates");
        pmeSpreadFixedMultipolesKernel = cu.getKernel(module, "gridSpreadFixedMultipoles");
        pmeSpreadInducedDipolesKernel = cu.getKernel(module, "gridSpreadInducedDipoles");
        pmeFinishSpreadChargeKernel = cu.getKernel(module, "finishSpreadCharge");
        pmeConvolutionKernel = cu.getKernel(module, "reciprocalConvolution");
        pmeFixedPotentialKernel = cu.getKernel(module, "computeFixedPotentialFromGrid");
        pmeInducedPotentialKernel = cu.getKernel(module, "computeInducedPotentialFromGrid");
        pmeFixedForceKernel = cu.getKernel(module, "computeFixedMultipoleForceAndEnergy");
        pmeInducedForceKernel = cu.getKernel(module, "computeInducedDipoleForceAndEnergy");
        pmeRecordInducedFieldDipolesKernel = cu.getKernel(module, "recordInducedFieldDipoles");
        cuFuncSetCacheConfig(pmeSpreadFixedMultipolesKernel, CU_FUNC_CACHE_PREFER_L1);
        cuFuncSetCacheConfig(pmeSpreadInducedDipolesKernel, CU_FUNC_CACHE_PREFER_L1);
        cuFuncSetCacheConfig(pmeFixedPotentialKernel, CU_FUNC_CACHE_PREFER_L1);
        cuFuncSetCacheConfig(pmeInducedPotentialKernel, CU_FUNC_CACHE_PREFER_L1);

        // Create required data structures.

        int elementSize = (cu.getUseDoublePrecision() ? sizeof(double) : sizeof(float));
        pmeGrid = new CudaArray(cu, gridSizeX*gridSizeY*gridSizeZ, 2*elementSize, "pmeGrid");
        cu.addAutoclearBuffer(*pmeGrid);
        pmeBsplineModuliX = new CudaArray(cu, gridSizeX, elementSize, "pmeBsplineModuliX");
        pmeBsplineModuliY = new CudaArray(cu, gridSizeY, elementSize, "pmeBsplineModuliY");
        pmeBsplineModuliZ = new CudaArray(cu, gridSizeZ, elementSize, "pmeBsplineModuliZ");
        pmeIgrid = CudaArray::create<int4>(cu, numMultipoles, "pmeIgrid");
        pmePhi = new CudaArray(cu, 35*numMultipoles, elementSize, "pmePhi");
        pmePhid = new CudaArray(cu, 10*numMultipoles, elementSize, "pmePhid");
        pmePhip = new CudaArray(cu, 10*numMultipoles, elementSize, "pmePhip");
        pmePhidp = new CudaArray(cu, 35*numMultipoles, elementSize, "pmePhidp");
        pmeCphi = new CudaArray(cu, 20*numMultipoles, elementSize, "pmeCphi");
        pmeAtomRange = CudaArray::create<int>(cu, gridSizeX*gridSizeY*gridSizeZ+1, "pmeAtomRange");
        sort = new CudaSort(cu, new SortTrait(), cu.getNumAtoms());
        cufftResult result = cufftPlan3d(&fft, gridSizeX, gridSizeY, gridSizeZ, cu.getUseDoublePrecision() ? CUFFT_Z2Z : CUFFT_C2C);
        if (result != CUFFT_SUCCESS)
            throw OpenMMException("Error initializing FFT: "+cu.intToString(result));
        hasInitializedFFT = true;

        // Initialize the b-spline moduli.

        double data[PmeOrder];
        double x = 0.0;
        data[0] = 1.0 - x;
        data[1] = x;
        for (int i = 2; i < PmeOrder; i++) {
            double denom = 1.0/i;
            data[i] = x*data[i-1]*denom;
            for (int j = 1; j < i; j++)
                data[i-j] = ((x+j)*data[i-j-1] + ((i-j+1)-x)*data[i-j])*denom;
            data[0] = (1.0-x)*data[0]*denom;
        }
        int maxSize = max(max(gridSizeX, gridSizeY), gridSizeZ);
        vector<double> bsplines_data(maxSize+1, 0.0);
        for (int i = 2; i <= PmeOrder+1; i++)
            bsplines_data[i] = data[i-2];
        for (int dim = 0; dim < 3; dim++) {
            int ndata = (dim == 0 ? gridSizeX : dim == 1 ? gridSizeY : gridSizeZ);
            vector<double> moduli(ndata);

            // get the modulus of the discrete Fourier transform

            double factor = 2.0*M_PI/ndata;
            for (int i = 0; i < ndata; i++) {
                double sc = 0.0;
                double ss = 0.0;
                for (int j = 1; j <= ndata; j++) {
                    double arg = factor*i*(j-1);
                    sc += bsplines_data[j]*cos(arg);
                    ss += bsplines_data[j]*sin(arg);
                }
                moduli[i] = sc*sc+ss*ss;
            }

            // Fix for exponential Euler spline interpolation failure.

            double eps = 1.0e-7;
            if (moduli[0] < eps)
                moduli[0] = 0.9*moduli[1];
            for (int i = 1; i < ndata-1; i++)
                if (moduli[i] < eps)
                    moduli[i] = 0.9*(moduli[i-1]+moduli[i+1]);
            if (moduli[ndata-1] < eps)
                moduli[ndata-1] = 0.9*moduli[ndata-2];

            // Compute and apply the optimal zeta coefficient.

            int jcut = 50;
            for (int i = 1; i <= ndata; i++) {
                int k = i - 1;
                if (i > ndata/2)
                    k = k - ndata;
                double zeta;
                if (k == 0)
                    zeta = 1.0;
                else {
                    double sum1 = 1.0;
                    double sum2 = 1.0;
                    factor = M_PI*k/ndata;
                    for (int j = 1; j <= jcut; j++) {
                        double arg = factor/(factor+M_PI*j);
                        sum1 += pow(arg, PmeOrder);
                        sum2 += pow(arg, 2*PmeOrder);
                    }
                    for (int j = 1; j <= jcut; j++) {
                        double arg = factor/(factor-M_PI*j);
                        sum1 += pow(arg, PmeOrder);
                        sum2 += pow(arg, 2*PmeOrder);
                    }
                    zeta = sum2/sum1;
                }
                moduli[i-1] = moduli[i-1]*zeta*zeta;
            }
            if (cu.getUseDoublePrecision()) {
                if (dim == 0)
                    pmeBsplineModuliX->upload(moduli);
                else if (dim == 1)
                    pmeBsplineModuliY->upload(moduli);
                else
                    pmeBsplineModuliZ->upload(moduli);
            }
            else {
                vector<float> modulif(ndata);
                for (int i = 0; i < ndata; i++)
                    modulif[i] = (float) moduli[i];
                if (dim == 0)
                    pmeBsplineModuliX->upload(modulif);
                else if (dim == 1)
                    pmeBsplineModuliY->upload(modulif);
                else
                    pmeBsplineModuliZ->upload(modulif);
            }
        }
    }

    // Add an interaction to the default nonbonded kernel.  This doesn't actually do any calculations.  It's
    // just so that CudaNonbondedUtilities will build the exclusion flags and maintain the neighbor list.

    cu.getNonbondedUtilities().addInteraction(usePME, usePME, true, force.getCutoffDistance(), exclusions, "", force.getForceGroup());
    cu.getNonbondedUtilities().setUsePadding(false);
    cu.addForce(new ForceInfo(force));
}

void CudaCalcMPIDForceKernel::initializeScaleFactors() {
    hasInitializedScaleFactors = true;
    CudaNonbondedUtilities& nb = cu.getNonbondedUtilities();

    // Figure out the covalent flag values to use for each atom pair.

    vector<int2> exclusionTiles;
    nb.getExclusionTiles().download(exclusionTiles);
    map<pair<int, int>, int> exclusionTileMap;
    for (int i = 0; i < (int) exclusionTiles.size(); i++) {
        int2 tile = exclusionTiles[i];
        exclusionTileMap[make_pair(tile.x, tile.y)] = i;
    }
    covalentFlags = CudaArray::create<uint2>(cu, nb.getExclusions().getSize(), "covalentFlags");
    vector<uint2> covalentFlagsVec(nb.getExclusions().getSize(), make_uint2(0, 0));
    for (int3 values : covalentFlagValues) {
        int atom1 = values.x;
        int atom2 = values.y;
        int value = values.z;
        int x = atom1/CudaContext::TileSize;
        int offset1 = atom1-x*CudaContext::TileSize;
        int y = atom2/CudaContext::TileSize;
        int offset2 = atom2-y*CudaContext::TileSize;
        int f1 = (value == 0 || value == 1 ? 1 : 0);
        int f2 = (value == 0 || value == 2 ? 1 : 0);
        if (x == y) {
            int index = exclusionTileMap[make_pair(x, y)]*CudaContext::TileSize;
            covalentFlagsVec[index+offset1].x |= f1<<offset2;
            covalentFlagsVec[index+offset1].y |= f2<<offset2;
            covalentFlagsVec[index+offset2].x |= f1<<offset1;
            covalentFlagsVec[index+offset2].y |= f2<<offset1;
        }
        else if (x > y) {
            int index = exclusionTileMap[make_pair(x, y)]*CudaContext::TileSize;
            covalentFlagsVec[index+offset1].x |= f1<<offset2;
            covalentFlagsVec[index+offset1].y |= f2<<offset2;
        }
        else {
            int index = exclusionTileMap[make_pair(y, x)]*CudaContext::TileSize;
            covalentFlagsVec[index+offset2].x |= f1<<offset1;
            covalentFlagsVec[index+offset2].y |= f2<<offset1;
        }
    }
    covalentFlags->upload(covalentFlagsVec);

}

double CudaCalcMPIDForceKernel::execute(ContextImpl& context, bool includeForces, bool includeEnergy) {
    if (!hasInitializedScaleFactors) {
        initializeScaleFactors();
    }
    CudaNonbondedUtilities& nb = cu.getNonbondedUtilities();

    // Compute the lab frame moments.

    void* computeMomentsArgs[] = {&cu.getPosq().getDevicePointer(), &multipoleParticles->getDevicePointer(),
        &molecularDipoles->getDevicePointer(), &molecularQuadrupoles->getDevicePointer(), &molecularOctopoles->getDevicePointer(),
        &labFrameDipoles->getDevicePointer(), &labFrameQuadrupoles->getDevicePointer(), &labFrameOctopoles->getDevicePointer(),
        &sphericalDipoles->getDevicePointer(), &sphericalQuadrupoles->getDevicePointer(), &sphericalOctopoles->getDevicePointer(),
        &polarizability->getDevicePointer(), &labFramePolarizabilities->getDevicePointer()};
    cu.executeKernel(computeMomentsKernel, computeMomentsArgs, cu.getNumAtoms());
    int startTileIndex = nb.getStartTileIndex();
    int numTileIndices = nb.getNumTiles();
    int numForceThreadBlocks = nb.getNumForceThreadBlocks();
    int elementSize = (cu.getUseDoublePrecision() ? sizeof(double) : sizeof(float));
    if (pmeGrid == NULL) {
        // Compute induced dipoles.
        void* computeFixedFieldArgs[] = {&field->getDevicePointer(), &cu.getPosq().getDevicePointer(),
            &covalentFlags->getDevicePointer(), &nb.getExclusionTiles().getDevicePointer(), &startTileIndex, &numTileIndices,
            &labFrameDipoles->getDevicePointer(), &labFrameQuadrupoles->getDevicePointer(), &labFrameOctopoles->getDevicePointer(),
            &dampingAndThole->getDevicePointer()};
        cu.executeKernel(computeFixedFieldKernel, computeFixedFieldArgs, numForceThreadBlocks*fixedFieldThreads, fixedFieldThreads);
        void* recordInducedDipolesArgs[] = {&field->getDevicePointer(),
            &inducedDipole->getDevicePointer(), &labFramePolarizabilities->getDevicePointer()};
        cu.executeKernel(recordInducedDipolesKernel, recordInducedDipolesArgs, cu.getNumAtoms());

        // Iterate until the dipoles converge.

        if (polarizationType == MPIDForce::Extrapolated)
            computeExtrapolatedDipoles(NULL);
        for (int i = 0; i < maxInducedIterations; i++) {
            computeInducedField(NULL);
            bool converged = iterateDipolesByDIIS(i);
            if (converged)
                break;
        }
        // Compute electrostatic force.

       void* electrostaticsArgs[] = {&cu.getForce().getDevicePointer(), &torque->getDevicePointer(), &cu.getEnergyBuffer().getDevicePointer(),
           &cu.getPosq().getDevicePointer(), &covalentFlags->getDevicePointer(),
           &nb.getExclusionTiles().getDevicePointer(), &startTileIndex, &numTileIndices,
           &sphericalDipoles->getDevicePointer(), &sphericalQuadrupoles->getDevicePointer(), &sphericalOctopoles->getDevicePointer(),
           &inducedDipole->getDevicePointer(), &dampingAndThole->getDevicePointer()};
       cu.executeKernel(electrostaticsKernel, electrostaticsArgs, numForceThreadBlocks*electrostaticsThreads, electrostaticsThreads);
    }
    else {
        // Compute reciprocal box vectors.
        Vec3 boxVectors[3];
        cu.getPeriodicBoxVectors(boxVectors[0], boxVectors[1], boxVectors[2]);
        double determinant = boxVectors[0][0]*boxVectors[1][1]*boxVectors[2][2];
        double scale = 1.0/determinant;
        double3 recipBoxVectors[3];
        recipBoxVectors[0] = make_double3(boxVectors[1][1]*boxVectors[2][2]*scale, 0, 0);
        recipBoxVectors[1] = make_double3(-boxVectors[1][0]*boxVectors[2][2]*scale, boxVectors[0][0]*boxVectors[2][2]*scale, 0);
        recipBoxVectors[2] = make_double3((boxVectors[1][0]*boxVectors[2][1]-boxVectors[1][1]*boxVectors[2][0])*scale, -boxVectors[0][0]*boxVectors[2][1]*scale, boxVectors[0][0]*boxVectors[1][1]*scale);
        float3 recipBoxVectorsFloat[3];
        void* recipBoxVectorPointer[3];
        if (cu.getUseDoublePrecision()) {
            recipBoxVectorPointer[0] = &recipBoxVectors[0];
            recipBoxVectorPointer[1] = &recipBoxVectors[1];
            recipBoxVectorPointer[2] = &recipBoxVectors[2];
        }
        else {
            recipBoxVectorsFloat[0] = make_float3((float) recipBoxVectors[0].x, 0, 0);
            recipBoxVectorsFloat[1] = make_float3((float) recipBoxVectors[1].x, (float) recipBoxVectors[1].y, 0);
            recipBoxVectorsFloat[2] = make_float3((float) recipBoxVectors[2].x, (float) recipBoxVectors[2].y, (float) recipBoxVectors[2].z);
            recipBoxVectorPointer[0] = &recipBoxVectorsFloat[0];
            recipBoxVectorPointer[1] = &recipBoxVectorsFloat[1];
            recipBoxVectorPointer[2] = &recipBoxVectorsFloat[2];
        }

        // Reciprocal space calculation.

        unsigned int maxTiles = nb.getInteractingTiles().getSize();
        void* pmeTransformMultipolesArgs[] = {&labFrameDipoles->getDevicePointer(), &labFrameQuadrupoles->getDevicePointer(), &labFrameOctopoles->getDevicePointer(),
            &fracDipoles->getDevicePointer(), &fracQuadrupoles->getDevicePointer(), &fracOctopoles->getDevicePointer(), recipBoxVectorPointer[0], recipBoxVectorPointer[1], recipBoxVectorPointer[2]};
        cu.executeKernel(pmeTransformMultipolesKernel, pmeTransformMultipolesArgs, cu.getNumAtoms());
        void* pmeSpreadFixedMultipolesArgs[] = {&cu.getPosq().getDevicePointer(), &fracDipoles->getDevicePointer(), &fracQuadrupoles->getDevicePointer(), &fracOctopoles->getDevicePointer(),
            &pmeGrid->getDevicePointer(), cu.getPeriodicBoxVecXPointer(), cu.getPeriodicBoxVecYPointer(), cu.getPeriodicBoxVecZPointer(),
            recipBoxVectorPointer[0], recipBoxVectorPointer[1], recipBoxVectorPointer[2]};
        cu.executeKernel(pmeSpreadFixedMultipolesKernel, pmeSpreadFixedMultipolesArgs, cu.getNumAtoms());
        void* finishSpreadArgs[] = {&pmeGrid->getDevicePointer()};
        if (cu.getUseDoublePrecision())
            cu.executeKernel(pmeFinishSpreadChargeKernel, finishSpreadArgs, pmeGrid->getSize());
        if (cu.getUseDoublePrecision())
            cufftExecZ2Z(fft, (double2*) pmeGrid->getDevicePointer(), (double2*) pmeGrid->getDevicePointer(), CUFFT_FORWARD);
        else
            cufftExecC2C(fft, (float2*) pmeGrid->getDevicePointer(), (float2*) pmeGrid->getDevicePointer(), CUFFT_FORWARD);
        void* pmeConvolutionArgs[] = {&pmeGrid->getDevicePointer(), &pmeBsplineModuliX->getDevicePointer(), &pmeBsplineModuliY->getDevicePointer(),
            &pmeBsplineModuliZ->getDevicePointer(), cu.getPeriodicBoxSizePointer(), recipBoxVectorPointer[0], recipBoxVectorPointer[1], recipBoxVectorPointer[2]};
        cu.executeKernel(pmeConvolutionKernel, pmeConvolutionArgs, gridSizeX*gridSizeY*gridSizeZ, 256);
        if (cu.getUseDoublePrecision())
            cufftExecZ2Z(fft, (double2*) pmeGrid->getDevicePointer(), (double2*) pmeGrid->getDevicePointer(), CUFFT_INVERSE);
        else
            cufftExecC2C(fft, (float2*) pmeGrid->getDevicePointer(), (float2*) pmeGrid->getDevicePointer(), CUFFT_INVERSE);
        void* pmeFixedPotentialArgs[] = {&pmeGrid->getDevicePointer(), &pmePhi->getDevicePointer(), &field->getDevicePointer(),
            &cu.getPosq().getDevicePointer(), &labFrameDipoles->getDevicePointer(),
            cu.getPeriodicBoxVecXPointer(), cu.getPeriodicBoxVecYPointer(), cu.getPeriodicBoxVecZPointer(),
            recipBoxVectorPointer[0], recipBoxVectorPointer[1], recipBoxVectorPointer[2]};
        cu.executeKernel(pmeFixedPotentialKernel, pmeFixedPotentialArgs, cu.getNumAtoms());
        void* pmeTransformFixedPotentialArgs[] = {&pmePhi->getDevicePointer(), &pmeCphi->getDevicePointer(), recipBoxVectorPointer[0], recipBoxVectorPointer[1], recipBoxVectorPointer[2]};
        cu.executeKernel(pmeTransformPotentialKernel, pmeTransformFixedPotentialArgs, cu.getNumAtoms());
        void* pmeFixedForceArgs[] = {&cu.getPosq().getDevicePointer(), &cu.getForce().getDevicePointer(), &torque->getDevicePointer(),
            &cu.getEnergyBuffer().getDevicePointer(), &labFrameDipoles->getDevicePointer(), &labFrameQuadrupoles->getDevicePointer(), &labFrameOctopoles->getDevicePointer(),
            &fracDipoles->getDevicePointer(), &fracQuadrupoles->getDevicePointer(), &fracOctopoles->getDevicePointer(), &pmePhi->getDevicePointer(), &pmeCphi->getDevicePointer(),
            recipBoxVectorPointer[0], recipBoxVectorPointer[1], recipBoxVectorPointer[2]};
        cu.executeKernel(pmeFixedForceKernel, pmeFixedForceArgs, cu.getNumAtoms());

        // Direct space calculation.

        void* computeFixedFieldArgs[] = {&field->getDevicePointer(), &cu.getPosq().getDevicePointer(),
            &covalentFlags->getDevicePointer(), &nb.getExclusionTiles().getDevicePointer(), &startTileIndex, &numTileIndices,
            &nb.getInteractingTiles().getDevicePointer(), &nb.getInteractionCount().getDevicePointer(), cu.getPeriodicBoxSizePointer(),
            cu.getInvPeriodicBoxSizePointer(), cu.getPeriodicBoxVecXPointer(), cu.getPeriodicBoxVecYPointer(), cu.getPeriodicBoxVecZPointer(),
            &maxTiles, &nb.getBlockCenters().getDevicePointer(), &nb.getInteractingAtoms().getDevicePointer(),
            &labFrameDipoles->getDevicePointer(), &labFrameQuadrupoles->getDevicePointer(), &labFrameOctopoles->getDevicePointer(), &dampingAndThole->getDevicePointer()};
        cu.executeKernel(computeFixedFieldKernel, computeFixedFieldArgs, numForceThreadBlocks*fixedFieldThreads, fixedFieldThreads);
        void* recordInducedDipolesArgs[] = {&field->getDevicePointer(), &inducedDipole->getDevicePointer(), &labFramePolarizabilities->getDevicePointer()};
        cu.executeKernel(recordInducedDipolesKernel, recordInducedDipolesArgs, cu.getNumAtoms());

        // Reciprocal space calculation for the induced dipoles.

        cu.clearBuffer(*pmeGrid);
        void* pmeSpreadInducedDipolesArgs[] = {&cu.getPosq().getDevicePointer(), &inducedDipole->getDevicePointer(),
            &pmeGrid->getDevicePointer(), cu.getPeriodicBoxVecXPointer(), cu.getPeriodicBoxVecYPointer(), cu.getPeriodicBoxVecZPointer(),
            recipBoxVectorPointer[0], recipBoxVectorPointer[1], recipBoxVectorPointer[2]};
        cu.executeKernel(pmeSpreadInducedDipolesKernel, pmeSpreadInducedDipolesArgs, cu.getNumAtoms());
        if (cu.getUseDoublePrecision())
            cu.executeKernel(pmeFinishSpreadChargeKernel, finishSpreadArgs, pmeGrid->getSize());
        if (cu.getUseDoublePrecision())
            cufftExecZ2Z(fft, (double2*) pmeGrid->getDevicePointer(), (double2*) pmeGrid->getDevicePointer(), CUFFT_FORWARD);
        else
            cufftExecC2C(fft, (float2*) pmeGrid->getDevicePointer(), (float2*) pmeGrid->getDevicePointer(), CUFFT_FORWARD);
        cu.executeKernel(pmeConvolutionKernel, pmeConvolutionArgs, gridSizeX*gridSizeY*gridSizeZ, 256);
        if (cu.getUseDoublePrecision())
            cufftExecZ2Z(fft, (double2*) pmeGrid->getDevicePointer(), (double2*) pmeGrid->getDevicePointer(), CUFFT_INVERSE);
        else
            cufftExecC2C(fft, (float2*) pmeGrid->getDevicePointer(), (float2*) pmeGrid->getDevicePointer(), CUFFT_INVERSE);
        void* pmeInducedPotentialArgs[] = {&pmeGrid->getDevicePointer(), &pmePhidp->getDevicePointer(),
           &cu.getPosq().getDevicePointer(), cu.getPeriodicBoxVecXPointer(), cu.getPeriodicBoxVecYPointer(),
            cu.getPeriodicBoxVecZPointer(), recipBoxVectorPointer[0], recipBoxVectorPointer[1], recipBoxVectorPointer[2]};
        cu.executeKernel(pmeInducedPotentialKernel, pmeInducedPotentialArgs, cu.getNumAtoms());

        // Iterate until the dipoles converge.

        if (polarizationType == MPIDForce::Extrapolated)
            computeExtrapolatedDipoles(recipBoxVectorPointer);
        for (int i = 0; i < maxInducedIterations; i++) {
            computeInducedField(recipBoxVectorPointer);
            bool converged = iterateDipolesByDIIS(i);
            if (converged)
                break;
        }
        // Compute electrostatic force.
        void* electrostaticsArgs[] = {&cu.getForce().getDevicePointer(), &torque->getDevicePointer(), &cu.getEnergyBuffer().getDevicePointer(),
            &cu.getPosq().getDevicePointer(), &covalentFlags->getDevicePointer(),
            &nb.getExclusionTiles().getDevicePointer(), &startTileIndex, &numTileIndices,
            &nb.getInteractingTiles().getDevicePointer(), &nb.getInteractionCount().getDevicePointer(),
            cu.getPeriodicBoxSizePointer(), cu.getInvPeriodicBoxSizePointer(), cu.getPeriodicBoxVecXPointer(), cu.getPeriodicBoxVecYPointer(), cu.getPeriodicBoxVecZPointer(),
            &maxTiles, &nb.getBlockCenters().getDevicePointer(), &nb.getInteractingAtoms().getDevicePointer(),
            &sphericalDipoles->getDevicePointer(), &sphericalQuadrupoles->getDevicePointer(), &sphericalOctopoles->getDevicePointer(),
            &inducedDipole->getDevicePointer(), &dampingAndThole->getDevicePointer(), &pmeCphi->getDevicePointer()};
        cu.executeKernel(electrostaticsKernel, electrostaticsArgs, numForceThreadBlocks*electrostaticsThreads, electrostaticsThreads);
        void* pmeTransformInducedPotentialArgs[] = {&pmePhidp->getDevicePointer(), &pmeCphi->getDevicePointer(), recipBoxVectorPointer[0], recipBoxVectorPointer[1], recipBoxVectorPointer[2]};
        cu.executeKernel(pmeTransformPotentialKernel, pmeTransformInducedPotentialArgs, cu.getNumAtoms());
        void* pmeInducedForceArgs[] = {&cu.getPosq().getDevicePointer(), &cu.getForce().getDevicePointer(), &torque->getDevicePointer(),
            &cu.getEnergyBuffer().getDevicePointer(), &labFrameDipoles->getDevicePointer(), &labFrameQuadrupoles->getDevicePointer(), &labFrameOctopoles->getDevicePointer(),
            &fracDipoles->getDevicePointer(), &fracQuadrupoles->getDevicePointer(), &fracOctopoles->getDevicePointer(),
            &inducedDipole->getDevicePointer(), &pmePhi->getDevicePointer(), &pmePhidp->getDevicePointer(), &pmeCphi->getDevicePointer(),
            recipBoxVectorPointer[0], recipBoxVectorPointer[1], recipBoxVectorPointer[2], &dampingAndThole->getDevicePointer()};
        cu.executeKernel(pmeInducedForceKernel, pmeInducedForceArgs, cu.getNumAtoms());
    }

    // If using extrapolated polarization, add in force contributions from µ(m) T µ(n).
    if (polarizationType == MPIDForce::Extrapolated) {
        void* extrapolatedArgs[] = {&cu.getForce().getDevicePointer(), &torque->getDevicePointer(),
        &dampingAndThole->getDevicePointer(), &extrapolatedDipole->getDevicePointer(),
            &extrapolatedDipoleField->getDevicePointer(), &extrapolatedDipoleFieldGradient->getDevicePointer()};
        cu.executeKernel(addExtrapolatedGradientKernel, extrapolatedArgs, numMultipoles);
    }

    // Map torques to force.

    void* mapTorqueArgs[] = {&cu.getForce().getDevicePointer(), &torque->getDevicePointer(),
        &cu.getPosq().getDevicePointer(), &multipoleParticles->getDevicePointer()};
    cu.executeKernel(mapTorqueKernel, mapTorqueArgs, cu.getNumAtoms());

    // Record the current atom positions so we can tell later if they have changed.

    cu.getPosq().copyTo(*lastPositions);
    multipolesAreValid = true;
    return 0.0;
}

void CudaCalcMPIDForceKernel::computeInducedField(void** recipBoxVectorPointer) {
    CudaNonbondedUtilities& nb = cu.getNonbondedUtilities();
    int startTileIndex = nb.getStartTileIndex();
    int numTileIndices = nb.getNumTiles();
    int numForceThreadBlocks = nb.getNumForceThreadBlocks();
    unsigned int maxTiles = 0;
    vector<void*> computeInducedFieldArgs;
    computeInducedFieldArgs.push_back(&inducedField->getDevicePointer());
    computeInducedFieldArgs.push_back(&cu.getPosq().getDevicePointer());
    computeInducedFieldArgs.push_back(&covalentFlags->getDevicePointer());
    computeInducedFieldArgs.push_back(&nb.getExclusionTiles().getDevicePointer());
    computeInducedFieldArgs.push_back(&inducedDipole->getDevicePointer());
    computeInducedFieldArgs.push_back(&startTileIndex);
    computeInducedFieldArgs.push_back(&numTileIndices);
    if (polarizationType == MPIDForce::Extrapolated) {
        computeInducedFieldArgs.push_back(&inducedDipoleFieldGradient->getDevicePointer());
    }
    if (pmeGrid != NULL) {
        computeInducedFieldArgs.push_back(&nb.getInteractingTiles().getDevicePointer());
        computeInducedFieldArgs.push_back(&nb.getInteractionCount().getDevicePointer());
        computeInducedFieldArgs.push_back(cu.getPeriodicBoxSizePointer());
        computeInducedFieldArgs.push_back(cu.getInvPeriodicBoxSizePointer());
        computeInducedFieldArgs.push_back(cu.getPeriodicBoxVecXPointer());
        computeInducedFieldArgs.push_back(cu.getPeriodicBoxVecYPointer());
        computeInducedFieldArgs.push_back(cu.getPeriodicBoxVecZPointer());
        computeInducedFieldArgs.push_back(&maxTiles);
        computeInducedFieldArgs.push_back(&nb.getBlockCenters().getDevicePointer());
        computeInducedFieldArgs.push_back(&nb.getInteractingAtoms().getDevicePointer());
    }
    computeInducedFieldArgs.push_back(&dampingAndThole->getDevicePointer());
    cu.clearBuffer(*inducedField);
    if (polarizationType == MPIDForce::Extrapolated) {
        cu.clearBuffer(*inducedDipoleFieldGradient);
    }
    if (pmeGrid == NULL)
        cu.executeKernel(computeInducedFieldKernel, &computeInducedFieldArgs[0], numForceThreadBlocks*inducedFieldThreads, inducedFieldThreads);
    else {
        maxTiles = nb.getInteractingTiles().getSize();
        cu.executeKernel(computeInducedFieldKernel, &computeInducedFieldArgs[0], numForceThreadBlocks*inducedFieldThreads, inducedFieldThreads);
        cu.clearBuffer(*pmeGrid);
        void* pmeSpreadInducedDipolesArgs[] = {&cu.getPosq().getDevicePointer(), &inducedDipole->getDevicePointer(),
            &pmeGrid->getDevicePointer(), cu.getPeriodicBoxVecXPointer(), cu.getPeriodicBoxVecYPointer(), cu.getPeriodicBoxVecZPointer(),
            recipBoxVectorPointer[0], recipBoxVectorPointer[1], recipBoxVectorPointer[2]};
        cu.executeKernel(pmeSpreadInducedDipolesKernel, pmeSpreadInducedDipolesArgs, cu.getNumAtoms());
        if (cu.getUseDoublePrecision()) {
            void* finishSpreadArgs[] = {&pmeGrid->getDevicePointer()};
            cu.executeKernel(pmeFinishSpreadChargeKernel, finishSpreadArgs, pmeGrid->getSize());
        }
        if (cu.getUseDoublePrecision())
            cufftExecZ2Z(fft, (double2*) pmeGrid->getDevicePointer(), (double2*) pmeGrid->getDevicePointer(), CUFFT_FORWARD);
        else
            cufftExecC2C(fft, (float2*) pmeGrid->getDevicePointer(), (float2*) pmeGrid->getDevicePointer(), CUFFT_FORWARD);
        void* pmeConvolutionArgs[] = {&pmeGrid->getDevicePointer(), &pmeBsplineModuliX->getDevicePointer(), &pmeBsplineModuliY->getDevicePointer(),
            &pmeBsplineModuliZ->getDevicePointer(), cu.getPeriodicBoxSizePointer(), recipBoxVectorPointer[0], recipBoxVectorPointer[1], recipBoxVectorPointer[2]};
        cu.executeKernel(pmeConvolutionKernel, pmeConvolutionArgs, gridSizeX*gridSizeY*gridSizeZ, 256);
        if (cu.getUseDoublePrecision())
            cufftExecZ2Z(fft, (double2*) pmeGrid->getDevicePointer(), (double2*) pmeGrid->getDevicePointer(), CUFFT_INVERSE);
        else
            cufftExecC2C(fft, (float2*) pmeGrid->getDevicePointer(), (float2*) pmeGrid->getDevicePointer(), CUFFT_INVERSE);
        void* pmeInducedPotentialArgs[] = {&pmeGrid->getDevicePointer(),
            &pmePhidp->getDevicePointer(), &cu.getPosq().getDevicePointer(), cu.getPeriodicBoxVecXPointer(), cu.getPeriodicBoxVecYPointer(),
            cu.getPeriodicBoxVecZPointer(), recipBoxVectorPointer[0], recipBoxVectorPointer[1], recipBoxVectorPointer[2]};
        cu.executeKernel(pmeInducedPotentialKernel, pmeInducedPotentialArgs, cu.getNumAtoms());
        if (polarizationType == MPIDForce::Extrapolated) {
            void* pmeRecordInducedFieldDipolesArgs[] = {&pmePhidp->getDevicePointer(),
                &inducedField->getDevicePointer(),  &inducedDipole->getDevicePointer(),
                &inducedDipoleFieldGradient->getDevicePointer(),
                recipBoxVectorPointer[0], recipBoxVectorPointer[1], recipBoxVectorPointer[2]};
            cu.executeKernel(pmeRecordInducedFieldDipolesKernel, pmeRecordInducedFieldDipolesArgs, cu.getNumAtoms());
        }
        else {
            void* pmeRecordInducedFieldDipolesArgs[] = {&pmePhidp->getDevicePointer(),
                &inducedField->getDevicePointer(), &inducedDipole->getDevicePointer(),
                recipBoxVectorPointer[0], recipBoxVectorPointer[1], recipBoxVectorPointer[2]};
            cu.executeKernel(pmeRecordInducedFieldDipolesKernel, pmeRecordInducedFieldDipolesArgs, cu.getNumAtoms());
        }
    }
}

bool CudaCalcMPIDForceKernel::iterateDipolesByDIIS(int iteration) {
    bool trueValue = true, falseValue = false;
    int elementSize = (cu.getUseDoublePrecision() ? sizeof(double) : sizeof(float));

    // Record the dipoles and errors into the lists of previous dipoles.

    void* recordDIISDipolesArgs[] = {&field->getDevicePointer(), &inducedField->getDevicePointer(),
        &inducedDipole->getDevicePointer(), &labFramePolarizabilities->getDevicePointer(), &inducedDipoleErrors->getDevicePointer(), &prevDipoles->getDevicePointer(),
        &prevErrors->getDevicePointer(), &iteration, &trueValue, &diisMatrix->getDevicePointer()};
    cu.executeKernel(recordDIISDipolesKernel, recordDIISDipolesArgs, cu.getNumThreadBlocks()*cu.ThreadBlockSize, cu.ThreadBlockSize, cu.ThreadBlockSize*elementSize);
    float* errors = (float*) cu.getPinnedBuffer();
    inducedDipoleErrors->download(errors, false);
    cuEventRecord(syncEvent, cu.getCurrentStream());

    // Build the DIIS matrix.

    int numPrev = (iteration+1 < MaxPrevDIISDipoles ? iteration+1 : MaxPrevDIISDipoles);
    void* buildMatrixArgs[] = {&prevErrors->getDevicePointer(), &iteration, &diisMatrix->getDevicePointer()};
    int threadBlocks = min(numPrev, cu.getNumThreadBlocks());
    int blockSize = 512;
    cu.executeKernel(buildMatrixKernel, buildMatrixArgs, threadBlocks*blockSize, blockSize, blockSize*elementSize);
    // Solve the matrix.

    void* solveMatrixArgs[] = {&iteration, &diisMatrix->getDevicePointer(), &diisCoefficients->getDevicePointer()};
    cu.executeKernel(solveMatrixKernel, solveMatrixArgs, 32, 32);

    // Determine whether the iteration has converged.

    cuEventSynchronize(syncEvent);
    double total1 = 0.0;
    for (int j = 0; j < inducedDipoleErrors->getSize(); j++) {
        total1 += errors[j];
    }

    if (48.033324*sqrt(total1/cu.getNumAtoms()) < inducedEpsilon)
        return true;

    // Compute the dipoles.

    void* updateInducedFieldArgs[] = {&inducedDipole->getDevicePointer(), &prevDipoles->getDevicePointer(),
                    &diisCoefficients->getDevicePointer(), &numPrev};
    cu.executeKernel(updateInducedFieldKernel, updateInducedFieldArgs, 3*cu.getNumAtoms(), 256);
    return false;
}

void CudaCalcMPIDForceKernel::computeExtrapolatedDipoles(void** recipBoxVectorPointer) {
    // Start by storing the direct dipoles as PT0

    void* initArgs[] = {&inducedDipole->getDevicePointer(), &extrapolatedDipole->getDevicePointer(), &inducedDipoleFieldGradient->getDevicePointer()};
    cu.executeKernel(initExtrapolatedKernel, initArgs, extrapolatedDipole->getSize());

    // Recursively apply alpha.Tau to the µ_(n) components to generate µ_(n+1), and store the result

    for (int order = 1; order < maxExtrapolationOrder; ++order) {
        computeInducedField(recipBoxVectorPointer);
        void* iterateArgs[] = {&order, &inducedDipole->getDevicePointer(), &extrapolatedDipole->getDevicePointer(),
            &inducedDipoleFieldGradient->getDevicePointer(),
            &inducedField->getDevicePointer(), &extrapolatedDipoleField->getDevicePointer(), &extrapolatedDipoleFieldGradient->getDevicePointer(),
            &labFramePolarizabilities->getDevicePointer()};
        cu.executeKernel(iterateExtrapolatedKernel, iterateArgs, extrapolatedDipole->getSize());
    }
    
    // Take a linear combination of the µ_(n) components to form the total dipole

    void* computeArgs[] = {&inducedDipole->getDevicePointer(), &extrapolatedDipole->getDevicePointer()};
    cu.executeKernel(computeExtrapolatedKernel, computeArgs, extrapolatedDipole->getSize());
    computeInducedField(recipBoxVectorPointer);
}

void CudaCalcMPIDForceKernel::ensureMultipolesValid(ContextImpl& context) {
    if (multipolesAreValid) {
        int numParticles = cu.getNumAtoms();
        if (cu.getUseDoublePrecision()) {
            vector<double4> pos1, pos2;
            cu.getPosq().download(pos1);
            lastPositions->download(pos2);
            for (int i = 0; i < numParticles; i++)
                if (pos1[i].x != pos2[i].x || pos1[i].y != pos2[i].y || pos1[i].z != pos2[i].z) {
                    multipolesAreValid = false;
                    break;
                }
        }
        else {
            vector<float4> pos1, pos2;
            cu.getPosq().download(pos1);
            lastPositions->download(pos2);
            for (int i = 0; i < numParticles; i++)
                if (pos1[i].x != pos2[i].x || pos1[i].y != pos2[i].y || pos1[i].z != pos2[i].z) {
                    multipolesAreValid = false;
                    break;
                }
        }
    }
    if (!multipolesAreValid)
        context.calcForcesAndEnergy(false, false, -1);
}


void CudaCalcMPIDForceKernel::getLabFramePermanentDipoles(ContextImpl& context, vector<Vec3>& dipoles) {
    ensureMultipolesValid(context);
    int numParticles = cu.getNumAtoms();
    dipoles.resize(numParticles);
    const vector<int>& order = cu.getAtomIndex();
    if (cu.getUseDoublePrecision()) {
        vector<double> labDipoleVec;
        labFrameDipoles->download(labDipoleVec);
        for (int i = 0; i < numParticles; i++)
            dipoles[order[i]] = Vec3(labDipoleVec[3*i], labDipoleVec[3*i+1], labDipoleVec[3*i+2]);
    }
    else {
        vector<float> labDipoleVec;
        labFrameDipoles->download(labDipoleVec);
        for (int i = 0; i < numParticles; i++)
            dipoles[order[i]] = Vec3(labDipoleVec[3*i], labDipoleVec[3*i+1], labDipoleVec[3*i+2]);
    }
}


void CudaCalcMPIDForceKernel::getInducedDipoles(ContextImpl& context, vector<Vec3>& dipoles) {
    ensureMultipolesValid(context);
    int numParticles = cu.getNumAtoms();
    dipoles.resize(numParticles);
    const vector<int>& order = cu.getAtomIndex();
    if (cu.getUseDoublePrecision()) {
        vector<double> d;
        inducedDipole->download(d);
        for (int i = 0; i < numParticles; i++)
            dipoles[order[i]] = Vec3(d[3*i], d[3*i+1], d[3*i+2]);
    }
    else {
        vector<float> d;
        inducedDipole->download(d);
        for (int i = 0; i < numParticles; i++)
            dipoles[order[i]] = Vec3(d[3*i], d[3*i+1], d[3*i+2]);
    }
}


void CudaCalcMPIDForceKernel::getTotalDipoles(ContextImpl& context, vector<Vec3>& dipoles) {
    ensureMultipolesValid(context);
    int numParticles = cu.getNumAtoms();
    dipoles.resize(numParticles);
    const vector<int>& order = cu.getAtomIndex();
    if (cu.getUseDoublePrecision()) {
        vector<double4> posqVec;
        vector<double> labDipoleVec;
        vector<double> inducedDipoleVec;
        double totalDipoleVecX;
        double totalDipoleVecY;
        double totalDipoleVecZ;
        inducedDipole->download(inducedDipoleVec);
        labFrameDipoles->download(labDipoleVec);
        cu.getPosq().download(posqVec);
        for (int i = 0; i < numParticles; i++) {
            totalDipoleVecX = labDipoleVec[3*i] + inducedDipoleVec[3*i];
            totalDipoleVecY = labDipoleVec[3*i+1] + inducedDipoleVec[3*i+1];
            totalDipoleVecZ = labDipoleVec[3*i+2] + inducedDipoleVec[3*i+2];
            dipoles[order[i]] = Vec3(totalDipoleVecX, totalDipoleVecY, totalDipoleVecZ);
        }
    }
    else {
        vector<float4> posqVec;
        vector<float> labDipoleVec;
        vector<float> inducedDipoleVec;
        float totalDipoleVecX;
        float totalDipoleVecY;
        float totalDipoleVecZ;
        inducedDipole->download(inducedDipoleVec);
        labFrameDipoles->download(labDipoleVec);
        cu.getPosq().download(posqVec);
        for (int i = 0; i < numParticles; i++) {
            totalDipoleVecX = labDipoleVec[3*i] + inducedDipoleVec[3*i];
            totalDipoleVecY = labDipoleVec[3*i+1] + inducedDipoleVec[3*i+1];
            totalDipoleVecZ = labDipoleVec[3*i+2] + inducedDipoleVec[3*i+2];
            dipoles[order[i]] = Vec3(totalDipoleVecX, totalDipoleVecY, totalDipoleVecZ);
        }
    }
}

void CudaCalcMPIDForceKernel::getElectrostaticPotential(ContextImpl& context, const vector<Vec3>& inputGrid, vector<double>& outputElectrostaticPotential) {
    ensureMultipolesValid(context);
    int numPoints = inputGrid.size();
    int elementSize = (cu.getUseDoublePrecision() ? sizeof(double) : sizeof(float));
    CudaArray points(cu, numPoints, 4*elementSize, "points");
    CudaArray potential(cu, numPoints, elementSize, "potential");
    
    // Copy the grid points to the GPU.
    
    if (cu.getUseDoublePrecision()) {
        vector<double4> p(numPoints);
        for (int i = 0; i < numPoints; i++)
            p[i] = make_double4(inputGrid[i][0], inputGrid[i][1], inputGrid[i][2], 0);
        points.upload(p);
    }
    else {
        vector<float4> p(numPoints);
        for (int i = 0; i < numPoints; i++)
            p[i] = make_float4((float) inputGrid[i][0], (float) inputGrid[i][1], (float) inputGrid[i][2], 0);
        points.upload(p);
    }
    
    // Compute the potential.
    
    void* computePotentialArgs[] = {&cu.getPosq().getDevicePointer(), &labFrameDipoles->getDevicePointer(),
        &labFrameQuadrupoles->getDevicePointer(), &inducedDipole->getDevicePointer(), &points.getDevicePointer(),
        &potential.getDevicePointer(), &numPoints, cu.getPeriodicBoxSizePointer(), cu.getInvPeriodicBoxSizePointer(),
        cu.getPeriodicBoxVecXPointer(), cu.getPeriodicBoxVecYPointer(), cu.getPeriodicBoxVecZPointer()};
    int blockSize = 128;
    cu.executeKernel(computePotentialKernel, computePotentialArgs, numPoints, blockSize, blockSize*15*elementSize);
    outputElectrostaticPotential.resize(numPoints);
    if (cu.getUseDoublePrecision())
        potential.download(outputElectrostaticPotential);
    else {
        vector<float> p(numPoints);
        potential.download(p);
        for (int i = 0; i < numPoints; i++)
            outputElectrostaticPotential[i] = p[i];
    }
}

template <class T, class T4, class M4>
void CudaCalcMPIDForceKernel::computeSystemMultipoleMoments(ContextImpl& context, vector<double>& outputMultipoleMoments) {
    // Compute the local coordinates relative to the center of mass.
    int numAtoms = cu.getNumAtoms();
    vector<T4> posq;
    vector<M4> velm;
    cu.getPosq().download(posq);
    cu.getVelm().download(velm);
    double totalMass = 0.0;
    Vec3 centerOfMass(0, 0, 0);
    for (int i = 0; i < numAtoms; i++) {
        double mass = (velm[i].w > 0 ? 1.0/velm[i].w : 0.0);
        totalMass += mass;
        centerOfMass[0] += mass*posq[i].x;
        centerOfMass[1] += mass*posq[i].y;
        centerOfMass[2] += mass*posq[i].z;
    }
    if (totalMass > 0.0) {
        centerOfMass[0] /= totalMass;
        centerOfMass[1] /= totalMass;
        centerOfMass[2] /= totalMass;
    }
    vector<double4> posqLocal(numAtoms);
    for (int i = 0; i < numAtoms; i++) {
        posqLocal[i].x = posq[i].x - centerOfMass[0];
        posqLocal[i].y = posq[i].y - centerOfMass[1];
        posqLocal[i].z = posq[i].z - centerOfMass[2];
        posqLocal[i].w = posq[i].w;
    }

    // Compute the multipole moments.
    
    double totalCharge = 0.0;
    double xdpl = 0.0;
    double ydpl = 0.0;
    double zdpl = 0.0;
    double xxqdp = 0.0;
    double xyqdp = 0.0;
    double xzqdp = 0.0;
    double yxqdp = 0.0;
    double yyqdp = 0.0;
    double yzqdp = 0.0;
    double zxqdp = 0.0;
    double zyqdp = 0.0;
    double zzqdp = 0.0;
    vector<T> labDipoleVec, inducedDipoleVec, quadrupoleVec;
    labFrameDipoles->download(labDipoleVec);
    inducedDipole->download(inducedDipoleVec);
    labFrameQuadrupoles->download(quadrupoleVec);
    for (int i = 0; i < numAtoms; i++) {
        totalCharge += posqLocal[i].w;
        double netDipoleX = (labDipoleVec[3*i]    + inducedDipoleVec[3*i]);
        double netDipoleY = (labDipoleVec[3*i+1]  + inducedDipoleVec[3*i+1]);
        double netDipoleZ = (labDipoleVec[3*i+2]  + inducedDipoleVec[3*i+2]);
        xdpl += posqLocal[i].x*posqLocal[i].w + netDipoleX;
        ydpl += posqLocal[i].y*posqLocal[i].w + netDipoleY;
        zdpl += posqLocal[i].z*posqLocal[i].w + netDipoleZ;
        xxqdp += posqLocal[i].x*posqLocal[i].x*posqLocal[i].w + 2*posqLocal[i].x*netDipoleX;
        xyqdp += posqLocal[i].x*posqLocal[i].y*posqLocal[i].w + posqLocal[i].x*netDipoleY + posqLocal[i].y*netDipoleX;
        xzqdp += posqLocal[i].x*posqLocal[i].z*posqLocal[i].w + posqLocal[i].x*netDipoleZ + posqLocal[i].z*netDipoleX;
        yxqdp += posqLocal[i].y*posqLocal[i].x*posqLocal[i].w + posqLocal[i].y*netDipoleX + posqLocal[i].x*netDipoleY;
        yyqdp += posqLocal[i].y*posqLocal[i].y*posqLocal[i].w + 2*posqLocal[i].y*netDipoleY;
        yzqdp += posqLocal[i].y*posqLocal[i].z*posqLocal[i].w + posqLocal[i].y*netDipoleZ + posqLocal[i].z*netDipoleY;
        zxqdp += posqLocal[i].z*posqLocal[i].x*posqLocal[i].w + posqLocal[i].z*netDipoleX + posqLocal[i].x*netDipoleZ;
        zyqdp += posqLocal[i].z*posqLocal[i].y*posqLocal[i].w + posqLocal[i].z*netDipoleY + posqLocal[i].y*netDipoleZ;
        zzqdp += posqLocal[i].z*posqLocal[i].z*posqLocal[i].w + 2*posqLocal[i].z*netDipoleZ;
    }

    // Convert the quadrupole from traced to traceless form.
 
    double qave = (xxqdp + yyqdp + zzqdp)/3;
    xxqdp = 1.5*(xxqdp-qave);
    xyqdp = 1.5*xyqdp;
    xzqdp = 1.5*xzqdp;
    yxqdp = 1.5*yxqdp;
    yyqdp = 1.5*(yyqdp-qave);
    yzqdp = 1.5*yzqdp;
    zxqdp = 1.5*zxqdp;
    zyqdp = 1.5*zyqdp;
    zzqdp = 1.5*(zzqdp-qave);

    // Add the traceless atomic quadrupoles to the total quadrupole moment.

    for (int i = 0; i < numAtoms; i++) {
        xxqdp = xxqdp + 3*quadrupoleVec[5*i];
        xyqdp = xyqdp + 3*quadrupoleVec[5*i+1];
        xzqdp = xzqdp + 3*quadrupoleVec[5*i+2];
        yxqdp = yxqdp + 3*quadrupoleVec[5*i+1];
        yyqdp = yyqdp + 3*quadrupoleVec[5*i+3];
        yzqdp = yzqdp + 3*quadrupoleVec[5*i+4];
        zxqdp = zxqdp + 3*quadrupoleVec[5*i+2];
        zyqdp = zyqdp + 3*quadrupoleVec[5*i+4];
        zzqdp = zzqdp + -3*(quadrupoleVec[5*i]+quadrupoleVec[5*i+3]);
    }
 
    double debye = 4.80321;
    outputMultipoleMoments.resize(13);
    outputMultipoleMoments[0] = totalCharge;
    outputMultipoleMoments[1] = 10.0*xdpl*debye;
    outputMultipoleMoments[2] = 10.0*ydpl*debye;
    outputMultipoleMoments[3] = 10.0*zdpl*debye;
    outputMultipoleMoments[4] = 100.0*xxqdp*debye;
    outputMultipoleMoments[5] = 100.0*xyqdp*debye;
    outputMultipoleMoments[6] = 100.0*xzqdp*debye;
    outputMultipoleMoments[7] = 100.0*yxqdp*debye;
    outputMultipoleMoments[8] = 100.0*yyqdp*debye;
    outputMultipoleMoments[9] = 100.0*yzqdp*debye;
    outputMultipoleMoments[10] = 100.0*zxqdp*debye;
    outputMultipoleMoments[11] = 100.0*zyqdp*debye;
    outputMultipoleMoments[12] = 100.0*zzqdp*debye;
}


void CudaCalcMPIDForceKernel::getSystemMultipoleMoments(ContextImpl& context, vector<double>& outputMultipoleMoments) {
    ensureMultipolesValid(context);
    if (cu.getUseDoublePrecision())
        computeSystemMultipoleMoments<double, double4, double4>(context, outputMultipoleMoments);
    else if (cu.getUseMixedPrecision())
        computeSystemMultipoleMoments<float, float4, double4>(context, outputMultipoleMoments);
    else
        computeSystemMultipoleMoments<float, float4, float4>(context, outputMultipoleMoments);
}

void CudaCalcMPIDForceKernel::copyParametersToContext(ContextImpl& context, const MPIDForce& force) {
    // Make sure the new parameters are acceptable.
    
    cu.setAsCurrent();
    if (force.getNumMultipoles() != cu.getNumAtoms())
        throw OpenMMException("updateParametersInContext: The number of multipoles has changed");
    
    // Record the per-multipole parameters.
    
    cu.getPosq().download(cu.getPinnedBuffer());
    float4* posqf = (float4*) cu.getPinnedBuffer();
    double4* posqd = (double4*) cu.getPinnedBuffer();
    vector<float2> dampingAndTholeVec;
    vector<float> polarizabilityVec;
    vector<float> molecularDipolesVec;
    vector<float> molecularQuadrupolesVec;
    vector<float> molecularOctopolesVec;
    vector<int4> multipoleParticlesVec;
   
    for (int i = 0; i < force.getNumMultipoles(); i++) {
        double charge, thole, damping;
        int axisType, atomX, atomY, atomZ;
        vector<double> dipole, quadrupole, octopole, polarity;
        force.getMultipoleParameters(i, charge, dipole, quadrupole, octopole, axisType, atomZ, atomX, atomY, thole, polarity);
        if (cu.getUseDoublePrecision())
            posqd[i].w = charge;
        else
            posqf[i].w = (float) charge;
        damping = pow((polarity[0]+polarity[1]+polarity[2])/3.0, 1.0/6.0);
        // If the multipole is isotropic, we flag it with a negative damping factor (which should always be positive) so we
        // can easily identify in the pairwise loops when we should be adding torque terms for induced dipoles.
        if(polarity[0]==polarity[1] && polarity[0]==polarity[2])
            damping *= -1;
        dampingAndTholeVec.push_back(make_float2((float) damping, (float) thole));
        polarizabilityVec.push_back((float) polarity[0]);
        polarizabilityVec.push_back((float) polarity[1]);
        polarizabilityVec.push_back((float) polarity[2]);
        multipoleParticlesVec.push_back(make_int4(atomX, atomY, atomZ, axisType));
        for (int j = 0; j < 3; j++)
            molecularDipolesVec.push_back((float) dipole[j]);
        molecularQuadrupolesVec.push_back((float) quadrupole[0]); // XX
        molecularQuadrupolesVec.push_back((float) quadrupole[1]); // XY
        molecularQuadrupolesVec.push_back((float) quadrupole[3]); // XZ
        molecularQuadrupolesVec.push_back((float) quadrupole[2]); // YY
        molecularQuadrupolesVec.push_back((float) quadrupole[4]); // YZ
        molecularOctopolesVec.push_back((float) octopole[0]); // XXX
        molecularOctopolesVec.push_back((float) octopole[1]); // XXY
        molecularOctopolesVec.push_back((float) octopole[4]); // XXZ
        molecularOctopolesVec.push_back((float) octopole[2]); // XYY
        molecularOctopolesVec.push_back((float) octopole[5]); // XYZ
        molecularOctopolesVec.push_back((float) octopole[3]); // YYY
        molecularOctopolesVec.push_back((float) octopole[6]); // YYZ
    }
    if (!hasQuadrupoles) {
        for (auto q : molecularQuadrupolesVec)
            if (q != 0.0)
                throw OpenMMException("updateParametersInContext: Cannot set a non-zero quadrupole moment, because quadrupoles were excluded from the kernel");
    }
    for (int i = force.getNumMultipoles(); i < cu.getPaddedNumAtoms(); i++) {
        dampingAndTholeVec.push_back(make_float2(0, 0));
        polarizabilityVec.push_back(0);
        polarizabilityVec.push_back(0);
        polarizabilityVec.push_back(0);
        multipoleParticlesVec.push_back(make_int4(0, 0, 0, 0));
        for (int j = 0; j < 3; j++)
            molecularDipolesVec.push_back(0);
        for (int j = 0; j < 5; j++)
            molecularQuadrupolesVec.push_back(0);
        for (int j = 0; j < 7; j++)
            molecularOctopolesVec.push_back(0);
    }
    dampingAndThole->upload(dampingAndTholeVec);
    polarizability->upload(polarizabilityVec);
    multipoleParticles->upload(multipoleParticlesVec);
    molecularDipoles->upload(molecularDipolesVec);
    molecularQuadrupoles->upload(molecularQuadrupolesVec);
    molecularOctopoles->upload(molecularOctopolesVec);
    cu.getPosq().upload(cu.getPinnedBuffer());
    cu.invalidateMolecules();
    multipolesAreValid = false;
}

void CudaCalcMPIDForceKernel::getPMEParameters(double& alpha, int& nx, int& ny, int& nz) const {
    if (!usePME)
        throw OpenMMException("getPMEParametersInContext: This Context is not using PME");
    alpha = this->alpha;
    nx = gridSizeX;
    ny = gridSizeY;
    nz = gridSizeZ;
}

