#ifndef MPID_OPENMM_CUDAKERNELS_H_
#define MPID_OPENMM_CUDAKERNELS_H_

/* -------------------------------------------------------------------------- *
 *                              OpenMMMPID                                  *
 * -------------------------------------------------------------------------- *
 * This is part of the OpenMM molecular simulation toolkit originating from   *
 * Simbios, the NIH National Center for Physics-Based Simulation of           *
 * Biological Structures at Stanford, funded under the NIH Roadmap for        *
 * Medical Research, grant U54 GM072970. See https://simtk.org.               *
 *                                                                            *
 * Portions copyright (c) 2008-2015 Stanford University and the Authors.      *
 * Authors: Mark Friedrichs, Peter Eastman                                    *
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

#include "openmm/mpidKernels.h"
#include "openmm/kernels.h"
#include "openmm/System.h"
#include "CudaArray.h"
#include "CudaContext.h"
#include "CudaSort.h"
#include <cufft.h>

namespace OpenMM {

/**
 * This kernel is invoked by MPIDForce to calculate the forces acting on the system and the energy of the system.
 */
class CudaCalcMPIDForceKernel : public CalcMPIDForceKernel {
public:
    CudaCalcMPIDForceKernel(std::string name, const Platform& platform, CudaContext& cu, const System& system);
    ~CudaCalcMPIDForceKernel();
    /**
     * Initialize the kernel.
     * 
     * @param system     the System this kernel will be applied to
     * @param force      the MPIDForce this kernel will be used for
     */
    void initialize(const System& system, const MPIDForce& force);
    /**
     * Execute the kernel to calculate the forces and/or energy.
     *
     * @param context        the context in which to execute this kernel
     * @param includeForces  true if forces should be calculated
     * @param includeEnergy  true if the energy should be calculated
     * @return the potential energy due to the force
     */
    double execute(ContextImpl& context, bool includeForces, bool includeEnergy);
     /**
     * Get the LabFrame dipole moments of all particles.
     * 
     * @param context    the Context for which to get the induced dipoles
     * @param dipoles    the induced dipole moment of particle i is stored into the i'th element
     */
    void getLabFramePermanentDipoles(ContextImpl& context, std::vector<Vec3>& dipoles);
    /**
     * Get the induced dipole moments of all particles.
     * 
     * @param context    the Context for which to get the induced dipoles
     * @param dipoles    the induced dipole moment of particle i is stored into the i'th element
     */
    void getInducedDipoles(ContextImpl& context, std::vector<Vec3>& dipoles);
    /**
     * Get the total dipole moments of all particles.
     * 
     * @param context    the Context for which to get the induced dipoles
     * @param dipoles    the induced dipole moment of particle i is stored into the i'th element
     */
    void getTotalDipoles(ContextImpl& context, std::vector<Vec3>& dipoles);
    /**
     * Execute the kernel to calculate the electrostatic potential
     *
     * @param context        the context in which to execute this kernel
     * @param inputGrid      input grid coordinates
     * @param outputElectrostaticPotential output potential 
     */
    void getElectrostaticPotential(ContextImpl& context, const std::vector< Vec3 >& inputGrid,
                                   std::vector< double >& outputElectrostaticPotential);

   /** 
     * Get the system multipole moments
     *
     * @param context      context
     * @param outputMultipoleMoments (charge,
     *                                dipole_x, dipole_y, dipole_z,
     *                                quadrupole_xx, quadrupole_xy, quadrupole_xz,
     *                                quadrupole_yx, quadrupole_yy, quadrupole_yz,
     *                                quadrupole_zx, quadrupole_zy, quadrupole_zz)
     */
    void getSystemMultipoleMoments(ContextImpl& context, std::vector<double>& outputMultipoleMoments);
    /**
     * Copy changed parameters over to a context.
     *
     * @param context    the context to copy parameters to
     * @param force      the MPIDForce to copy the parameters from
     */
    void copyParametersToContext(ContextImpl& context, const MPIDForce& force);
    /**
     * Get the parameters being used for PME.
     * 
     * @param alpha   the separation parameter
     * @param nx      the number of grid points along the X axis
     * @param ny      the number of grid points along the Y axis
     * @param nz      the number of grid points along the Z axis
     */
    void getPMEParameters(double& alpha, int& nx, int& ny, int& nz) const;
private:
    class ForceInfo;
    class SortTrait : public CudaSort::SortTrait {
        int getDataSize() const {return 8;}
        int getKeySize() const {return 4;}
        const char* getDataType() const {return "int2";}
        const char* getKeyType() const {return "int";}
        const char* getMinKey() const {return "(-2147483647 - 1)";}
        const char* getMaxKey() const {return "2147483647";}
        const char* getMaxValue() const {return "make_int2(2147483647, 2147483647)";}
        const char* getSortKey() const {return "value.y";}
    };
    void initializeScaleFactors();
    void computeInducedField(void** recipBoxVectorPointer);
    bool iterateDipolesByDIIS(int iteration);
    void computeExtrapolatedDipoles(void** recipBoxVectorPointer);
    void ensureMultipolesValid(ContextImpl& context);
    template <class T, class T4, class M4> void computeSystemMultipoleMoments(ContextImpl& context, std::vector<double>& outputMultipoleMoments);
    int numMultipoles, maxInducedIterations, maxExtrapolationOrder;
    int fixedFieldThreads, inducedFieldThreads, electrostaticsThreads;
    int gridSizeX, gridSizeY, gridSizeZ;
    double alpha, inducedEpsilon;
    bool usePME, hasQuadrupoles, hasOctopoles, hasInitializedScaleFactors, hasInitializedFFT, multipolesAreValid, hasCreatedEvent;
    MPIDForce::PolarizationType polarizationType;
    CudaContext& cu;
    const System& system;
    std::vector<int3> covalentFlagValues;
    std::vector<int2> polarizationFlagValues;
    CudaArray* multipoleParticles;
    CudaArray* molecularDipoles;
    CudaArray* molecularQuadrupoles;
    CudaArray* molecularOctopoles;
    CudaArray* labFrameDipoles;
    CudaArray* labFramePolarizabilities;
    CudaArray* labFrameQuadrupoles;
    CudaArray* labFrameOctopoles;
    CudaArray* sphericalDipoles;
    CudaArray* sphericalQuadrupoles;
    CudaArray* sphericalOctopoles;
    CudaArray* fracDipoles;
    CudaArray* fracQuadrupoles;
    CudaArray* fracOctopoles;
    CudaArray* field;
    CudaArray* inducedField;
    CudaArray* torque;
    CudaArray* dampingAndThole;
    CudaArray* inducedDipole;
    CudaArray* inducedDipoleErrors;
    CudaArray* prevDipoles;
    CudaArray* prevErrors;
    CudaArray* diisMatrix;
    CudaArray* diisCoefficients;
    CudaArray* extrapolatedDipole;
    CudaArray* inducedDipoleField;
    CudaArray* inducedDipoleFieldGradient;
    CudaArray* extrapolatedDipoleField;
    CudaArray* extrapolatedDipoleFieldGradient;
    CudaArray* polarizability;
    CudaArray* covalentFlags;
    CudaArray* pmeGrid;
    CudaArray* pmeBsplineModuliX;
    CudaArray* pmeBsplineModuliY;
    CudaArray* pmeBsplineModuliZ;
    CudaArray* pmeIgrid;
    CudaArray* pmePhi;
    CudaArray* pmePhid;
    CudaArray* pmePhip;
    CudaArray* pmePhidp;
    CudaArray* pmeCphi;
    CudaArray* pmeAtomRange;
    CudaArray* lastPositions;
    CudaSort* sort;
    cufftHandle fft;
    CUfunction computeMomentsKernel, recordInducedDipolesKernel, computeFixedFieldKernel, computeInducedFieldKernel, updateInducedFieldKernel, electrostaticsKernel, mapTorqueKernel;
    CUfunction pmeSpreadFixedMultipolesKernel, pmeSpreadInducedDipolesKernel, pmeFinishSpreadChargeKernel, pmeConvolutionKernel;
    CUfunction pmeFixedPotentialKernel, pmeInducedPotentialKernel, pmeFixedForceKernel, pmeInducedForceKernel, pmeRecordInducedFieldDipolesKernel, computePotentialKernel;
    CUfunction recordDIISDipolesKernel, buildMatrixKernel, solveMatrixKernel;
    CUfunction initExtrapolatedKernel, iterateExtrapolatedKernel, computeExtrapolatedKernel, addExtrapolatedGradientKernel;
    CUfunction pmeTransformMultipolesKernel, pmeTransformPotentialKernel;
    CUevent syncEvent;
    static const int PmeOrder = 6;
    static const int MaxPrevDIISDipoles = 20;
};

} // namespace OpenMM

#endif /*MPID_OPENMM_CUDAKERNELS_H*/
