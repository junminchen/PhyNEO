/* -------------------------------------------------------------------------- *
 *                               OpenMMMPID                                 *
 * -------------------------------------------------------------------------- *
 * This is part of the OpenMM molecular simulation toolkit originating from   *
 * Simbios, the NIH National Center for Physics-Based Simulation of           *
 * Biological Structures at Stanford, funded under the NIH Roadmap for        *
 * Medical Research, grant U54 GM072970. See https://simtk.org.               *
 *                                                                            *
 * Portions copyright (c) 2008 Stanford University and the Authors.           *
 * Authors:                                                                   *
 * Contributors:                                                              *
 *                                                                            *
 * Permission is hereby granted, free of charge, to any person obtaining a    *
 * copy of this software and associated documentation files (the "Software"), *
 * to deal in the Software without restriction, including without limitation  *
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,   *
 * and/or sell copies of the Software, and to permit persons to whom the      *
 * Software is furnished to do so, subject to the following conditions:       *
 *                                                                            *
 * The above copyright notice and this permission notice shall be included in *
 * all copies or substantial portions of the Software.                        *
 *                                                                            *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR *
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,   *
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL    *
 * THE AUTHORS, CONTRIBUTORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,    *
 * DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR      *
 * OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE  *
 * USE OR OTHER DEALINGS IN THE SOFTWARE.                                     *
 * -------------------------------------------------------------------------- */

#include "openmm/internal/ContextImpl.h"
#include "openmm/internal/MPIDForceImpl.h"
#include "openmm/mpidKernels.h"
#include <stdio.h>
#include <math.h>

using namespace OpenMM;

using std::vector;

bool MPIDForceImpl::initializedCovalentDegrees = false;
int MPIDForceImpl::CovalentDegrees[]           = { 1,2,3,4,0,1,2,3};

MPIDForceImpl::MPIDForceImpl(const MPIDForce& owner) : owner(owner) {
}

MPIDForceImpl::~MPIDForceImpl() {
}

void MPIDForceImpl::initialize(ContextImpl& context) {

    const System& system = context.getSystem();
    int numParticles = system.getNumParticles();

    if (owner.getNumMultipoles() != numParticles)
        throw OpenMMException("MPIDForce must have exactly as many particles as the System it belongs to.");

    // check cutoff < 0.5*boxSize

    if (owner.getNonbondedMethod() == MPIDForce::PME) {
        Vec3 boxVectors[3];
        system.getDefaultPeriodicBoxVectors(boxVectors[0], boxVectors[1], boxVectors[2]);
        double cutoff = owner.getCutoffDistance();
        if (cutoff > 0.5*boxVectors[0][0] || cutoff > 0.5*boxVectors[1][1] || cutoff > 0.5*boxVectors[2][2])
            throw OpenMMException("MPIDForce: The cutoff distance cannot be greater than half the periodic box size.");
    }

    double quadrupoleValidationTolerance = 1.0e-05;
    double octopoleValidationTolerance = 1.0e-05;
    for (int ii = 0; ii < numParticles; ii++) {

        int axisType, multipoleAtomZ, multipoleAtomX, multipoleAtomY;
        double charge, thole, dampingFactor, polarity ;
        std::vector<double> molecularDipole;
        std::vector<double> molecularQuadrupole;
        std::vector<double> molecularOctopole;
        std::vector<double> alphas;

        owner.getMultipoleParameters(ii, charge, molecularDipole, molecularQuadrupole, molecularOctopole, axisType,
                                     multipoleAtomZ, multipoleAtomX, multipoleAtomY,
                                     thole, alphas);

       // check quadrupole is traceless and symmetric

       double trace = fabs(molecularQuadrupole[0] + molecularQuadrupole[2] + molecularQuadrupole[5]);
       if (trace > quadrupoleValidationTolerance) {
             std::stringstream buffer;
             buffer << "MPIDForce: quadrupole for particle=" << ii;
             buffer << " has nonzero trace: " << trace << "; MPID plugin assumes traceless quadrupole.";
             throw OpenMMException(buffer.str());
       }

       trace = fabs(molecularOctopole[0] + molecularOctopole[2] + molecularOctopole[7]);
       if (trace > octopoleValidationTolerance) {
             std::stringstream buffer;
             buffer << "MPIDForce: (XXX,XYY,XZZ) octopole for particle=" << ii;
             buffer << " has nonzero trace: " << trace << "; MPID plugin assumes traceless octopoles.";
             throw OpenMMException(buffer.str());
       }

       trace = fabs(molecularOctopole[1] + molecularOctopole[3] + molecularOctopole[8]);
       if (trace > octopoleValidationTolerance) {
             std::stringstream buffer;
             buffer << "MPIDForce: (YXX,YYY,YZZ) octopole for particle=" << ii;
             buffer << " has nonzero trace: " << trace << "; MPID plugin assumes traceless octopoles.";
             throw OpenMMException(buffer.str());
       }

       trace = fabs(molecularOctopole[4] + molecularOctopole[6] + molecularOctopole[9]);
       if (trace > octopoleValidationTolerance) {
             std::stringstream buffer;
             buffer << "MPIDForce: (ZXX,ZYY,ZZZ) octopole for particle=" << ii;
             buffer << " has nonzero trace: " << trace << "; MPID plugin assumes traceless octopoles.";
             throw OpenMMException(buffer.str());
       }


       // only 'Z-then-X', 'Bisector', Z-Bisect, ThreeFold  currently handled

        if (axisType != MPIDForce::ZThenX     && axisType != MPIDForce::Bisector &&
            axisType != MPIDForce::ZBisect    && axisType != MPIDForce::ThreeFold &&
            axisType != MPIDForce::ZOnly      && axisType != MPIDForce::NoAxisType) {
             std::stringstream buffer;
             buffer << "MPIDForce: axis type=" << axisType;
             buffer << " not currently handled - only axisTypes[ ";
             buffer << MPIDForce::ZThenX   << ", " << MPIDForce::Bisector  << ", ";
             buffer << MPIDForce::ZBisect  << ", " << MPIDForce::ThreeFold << ", ";
             buffer << MPIDForce::NoAxisType;
             buffer << "] (ZThenX, Bisector, Z-Bisect, ThreeFold, NoAxisType) currently handled .";
             throw OpenMMException(buffer.str());
        }
        if (axisType != MPIDForce::NoAxisType && (multipoleAtomZ < 0 || multipoleAtomZ >= numParticles)) {
            std::stringstream buffer;
            buffer << "MPIDForce: invalid z axis particle: " << multipoleAtomZ;
            throw OpenMMException(buffer.str());
        }
        if (axisType != MPIDForce::NoAxisType && axisType != MPIDForce::ZOnly &&
                (multipoleAtomX < 0 || multipoleAtomX >= numParticles)) {
            std::stringstream buffer;
            buffer << "MPIDForce: invalid x axis particle: " << multipoleAtomX;
            throw OpenMMException(buffer.str());
        }
        if ((axisType == MPIDForce::ZBisect || axisType == MPIDForce::ThreeFold) &&
                (multipoleAtomY < 0 || multipoleAtomY >= numParticles)) {
            std::stringstream buffer;
            buffer << "MPIDForce: invalid y axis particle: " << multipoleAtomY;
            throw OpenMMException(buffer.str());
        }
    }
    kernel = context.getPlatform().createKernel(CalcMPIDForceKernel::Name(), context);
    kernel.getAs<CalcMPIDForceKernel>().initialize(context.getSystem(), owner);
}

double MPIDForceImpl::calcForcesAndEnergy(ContextImpl& context, bool includeForces, bool includeEnergy, int groups) {
    if ((groups&(1<<owner.getForceGroup())) != 0)
        return kernel.getAs<CalcMPIDForceKernel>().execute(context, includeForces, includeEnergy);
    return 0.0;
}

std::vector<std::string> MPIDForceImpl::getKernelNames() {
    std::vector<std::string> names;
    names.push_back(CalcMPIDForceKernel::Name());
    return names;
}

const int* MPIDForceImpl::getCovalentDegrees() {
    if (!initializedCovalentDegrees) {
        initializedCovalentDegrees                                      = true;
        CovalentDegrees[MPIDForce::Covalent12]               = 1;
        CovalentDegrees[MPIDForce::Covalent13]               = 2;
        CovalentDegrees[MPIDForce::Covalent14]               = 3;
        CovalentDegrees[MPIDForce::Covalent15]               = 4;
        CovalentDegrees[MPIDForce::PolarizationCovalent11]   = 0;
        CovalentDegrees[MPIDForce::PolarizationCovalent12]   = 1;
        CovalentDegrees[MPIDForce::PolarizationCovalent13]   = 2;
        CovalentDegrees[MPIDForce::PolarizationCovalent14]   = 3;
    }
    return CovalentDegrees;
}

void MPIDForceImpl::getCovalentRange(const MPIDForce& force, int atomIndex, const std::vector<MPIDForce::CovalentType>& lists,
                                                int* minCovalentIndex, int* maxCovalentIndex) {

    *minCovalentIndex =  999999999;
    *maxCovalentIndex = -999999999;
    for (unsigned int kk = 0; kk < lists.size(); kk++) {
        MPIDForce::CovalentType jj = lists[kk];
        std::vector<int> covalentList;
        force.getCovalentMap(atomIndex, jj, covalentList);
        for (unsigned int ii = 0; ii < covalentList.size(); ii++) {
            if (*minCovalentIndex > covalentList[ii]) {
               *minCovalentIndex = covalentList[ii];
            }
            if (*maxCovalentIndex < covalentList[ii]) {
               *maxCovalentIndex = covalentList[ii];
            }
        }
    }
    return;
}

void MPIDForceImpl::getCovalentDegree(const MPIDForce& force, std::vector<int>& covalentDegree) {
    covalentDegree.resize(MPIDForce::CovalentEnd);
    const int* CovalentDegrees = MPIDForceImpl::getCovalentDegrees();
    for (unsigned int kk = 0; kk < MPIDForce::CovalentEnd; kk++) {
        covalentDegree[kk] = CovalentDegrees[kk];
    }
    return;
}

void MPIDForceImpl::getLabFramePermanentDipoles(ContextImpl& context, vector<Vec3>& dipoles) {
    kernel.getAs<CalcMPIDForceKernel>().getLabFramePermanentDipoles(context, dipoles);
}

void MPIDForceImpl::getInducedDipoles(ContextImpl& context, vector<Vec3>& dipoles) {
    kernel.getAs<CalcMPIDForceKernel>().getInducedDipoles(context, dipoles);
}

void MPIDForceImpl::getTotalDipoles(ContextImpl& context, vector<Vec3>& dipoles) {
    kernel.getAs<CalcMPIDForceKernel>().getTotalDipoles(context, dipoles);
}

void MPIDForceImpl::getElectrostaticPotential(ContextImpl& context, const std::vector< Vec3 >& inputGrid,
                                                         std::vector< double >& outputElectrostaticPotential) {
    kernel.getAs<CalcMPIDForceKernel>().getElectrostaticPotential(context, inputGrid, outputElectrostaticPotential);
}

void MPIDForceImpl::getSystemMultipoleMoments(ContextImpl& context, std::vector< double >& outputMultipoleMoments) {
    kernel.getAs<CalcMPIDForceKernel>().getSystemMultipoleMoments(context, outputMultipoleMoments);
}

void MPIDForceImpl::updateParametersInContext(ContextImpl& context) {
    kernel.getAs<CalcMPIDForceKernel>().copyParametersToContext(context, owner);
    context.systemChanged();
}

void MPIDForceImpl::getPMEParameters(double& alpha, int& nx, int& ny, int& nz) const {
    kernel.getAs<CalcMPIDForceKernel>().getPMEParameters(alpha, nx, ny, nz);
}
