/* -------------------------------------------------------------------------- *
 *                                OpenMMMPID                                *
 * -------------------------------------------------------------------------- *
 * This is part of the OpenMM molecular simulation toolkit originating from   *
 * Simbios, the NIH National Center for Physics-Based Simulation of           *
 * Biological Structures at Stanford, funded under the NIH Roadmap for        *
 * Medical Research, grant U54 GM072970. See https://simtk.org.               *
 *                                                                            *
 * Portions copyright (c) 2008-2016 Stanford University and the Authors.      *
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

#include "openmm/Force.h"
#include "openmm/OpenMMException.h"
#include "openmm/MPIDForce.h"
#include "openmm/internal/MPIDForceImpl.h"
#include <stdio.h>
#include <iostream>

using namespace OpenMM;
using std::string;
using std::vector;

MPIDForce::MPIDForce() : nonbondedMethod(NoCutoff), polarizationType(Extrapolated), pmeBSplineOrder(6), cutoffDistance(1.0), ewaldErrorTol(5e-4), mutualInducedMaxIterations(60),
                                               mutualInducedTargetEpsilon(1.0e-5), scalingDistanceCutoff(100.0), electricConstant(138.9354558456), defaultThole(5.0),
                                               alpha(0.0), nx(0), ny(0), nz(0), scaleFactor14(1.0) {
    extrapolationCoefficients.push_back(-0.154);
    extrapolationCoefficients.push_back(0.017);
    extrapolationCoefficients.push_back(0.658);
    extrapolationCoefficients.push_back(0.474);
}

MPIDForce::NonbondedMethod MPIDForce::getNonbondedMethod() const {
    return nonbondedMethod;
}

void MPIDForce::setNonbondedMethod(MPIDForce::NonbondedMethod method) {
    if (method < 0 || method > 1)
        throw OpenMMException("MPIDForce: Illegal value for nonbonded method");
    nonbondedMethod = method;
}

MPIDForce::PolarizationType MPIDForce::getPolarizationType() const {
    return polarizationType;
}

void MPIDForce::setPolarizationType(MPIDForce::PolarizationType type) {
    polarizationType = type;
}

void MPIDForce::setExtrapolationCoefficients(const std::vector<double> &coefficients) {
    extrapolationCoefficients = coefficients;
}

const std::vector<double> & MPIDForce::getExtrapolationCoefficients() const {
    return extrapolationCoefficients;
}

double MPIDForce::getCutoffDistance() const {
    return cutoffDistance;
}

void MPIDForce::setCutoffDistance(double distance) {
    cutoffDistance = distance;
}

void MPIDForce::getPMEParameters(double& alpha, int& nx, int& ny, int& nz) const {
    alpha = this->alpha;
    nx = this->nx;
    ny = this->ny;
    nz = this->nz;
}

void MPIDForce::setPMEParameters(double alpha, int nx, int ny, int nz) {
    this->alpha = alpha;
    this->nx = nx;
    this->ny = ny;
    this->nz = nz;
}

double MPIDForce::getAEwald() const { 
    return alpha; 
} 
 
void MPIDForce::setAEwald(double inputAewald) { 
    alpha = inputAewald; 
} 
 
int MPIDForce::getPmeBSplineOrder() const { 
    return pmeBSplineOrder; 
} 
 
void MPIDForce::getPmeGridDimensions(std::vector<int>& gridDimension) const { 
    if (gridDimension.size() < 3)
        gridDimension.resize(3);
    gridDimension[0] = nx;
    gridDimension[1] = ny;
    gridDimension[2] = nz;
} 
 
void MPIDForce::setPmeGridDimensions(const std::vector<int>& gridDimension) {
    nx = gridDimension[0];
    ny = gridDimension[1];
    nz = gridDimension[2];
}

void MPIDForce::getPMEParametersInContext(const Context& context, double& alpha, int& nx, int& ny, int& nz) const {
    dynamic_cast<const MPIDForceImpl&>(getImplInContext(context)).getPMEParameters(alpha, nx, ny, nz);
}

int MPIDForce::getMutualInducedMaxIterations() const {
    return mutualInducedMaxIterations;
}

void MPIDForce::setMutualInducedMaxIterations(int inputMutualInducedMaxIterations) {
    mutualInducedMaxIterations = inputMutualInducedMaxIterations;
}

double MPIDForce::getMutualInducedTargetEpsilon() const {
    return mutualInducedTargetEpsilon;
}

void MPIDForce::setMutualInducedTargetEpsilon(double inputMutualInducedTargetEpsilon) {
    mutualInducedTargetEpsilon = inputMutualInducedTargetEpsilon;
}

double MPIDForce::getEwaldErrorTolerance() const {
    return ewaldErrorTol;
}

void MPIDForce::setEwaldErrorTolerance(double tol) {
    ewaldErrorTol = tol;
}

double MPIDForce::get14ScaleFactor() const {
    return scaleFactor14;
}

void MPIDForce::set14ScaleFactor(double fac) {
    scaleFactor14 = fac;
}

int MPIDForce::addMultipole(double charge, const std::vector<double>& molecularDipole, const std::vector<double>& molecularQuadrupole,
                                       const std::vector<double>& molecularOctopole, int axisType, int multipoleAtomZ, int multipoleAtomX,
                                       int multipoleAtomY, double thole, const std::vector<double>& alphas) {
    multipoles.push_back(MultipoleInfo(charge, molecularDipole, molecularQuadrupole,  molecularOctopole, axisType, multipoleAtomZ,  multipoleAtomX, multipoleAtomY, thole, alphas));
    return multipoles.size()-1;
}

void MPIDForce::getMultipoleParameters(int index, double& charge, std::vector<double>& molecularDipole, std::vector<double>& molecularQuadrupole, std::vector<double> &molecularOctopole,
                                                  int& axisType, int& multipoleAtomZ, int& multipoleAtomX, int& multipoleAtomY, double& thole, std::vector<double>& alphas) const {
    charge                      = multipoles[index].charge;

    molecularDipole.resize(3);
    molecularQuadrupole.resize(6);
    molecularOctopole.resize(10);
    for(int i = 0; i < 3; ++i) molecularDipole[i] = multipoles[index].molecularDipole[i];
    for(int i = 0; i < 6; ++i) molecularQuadrupole[i] = multipoles[index].molecularQuadrupole[i];
    for(int i = 0; i < 10; ++i) molecularOctopole[i] = multipoles[index].molecularOctopole[i];

    axisType                    = multipoles[index].axisType;
    multipoleAtomZ              = multipoles[index].multipoleAtomZ;
    multipoleAtomX              = multipoles[index].multipoleAtomX;
    multipoleAtomY              = multipoles[index].multipoleAtomY;

    thole                       = multipoles[index].thole;
    alphas.resize(3);
    for(int i = 0; i < 3; ++i) alphas[i] = multipoles[index].polarity[i];
}

void MPIDForce::setMultipoleParameters(int index, double charge, const std::vector<double>& molecularDipole, const std::vector<double>& molecularQuadrupole, const std::vector<double>& molecularOctopole,
                                                  int axisType, int multipoleAtomZ, int multipoleAtomX, int multipoleAtomY, double thole, const std::vector<double>& alphas) {

    multipoles[index].charge                      = charge;

    for(int i = 0; i < 3; ++i) multipoles[index].molecularDipole[i] = molecularDipole[i];
    for(int i = 0; i < 6; ++i) multipoles[index].molecularQuadrupole[i] = molecularQuadrupole[i];
    for(int i = 0; i < 10; ++i) multipoles[index].molecularOctopole[i] = molecularOctopole[i];

    double dampingFactor = pow((alphas[0]+alphas[1]+alphas[2])/3.0, 1.0/6.0);
    multipoles[index].axisType                    = axisType;
    multipoles[index].multipoleAtomZ              = multipoleAtomZ;
    multipoles[index].multipoleAtomX              = multipoleAtomX;
    multipoles[index].multipoleAtomY              = multipoleAtomY;
    multipoles[index].thole                       = thole;
    multipoles[index].dampingFactor               = dampingFactor;
    multipoles[index].polarity                    = alphas;
    for(int i = 0; i < 3; ++i) multipoles[index].polarity[i] = alphas[i];

}

void MPIDForce::setCovalentMap(int index, CovalentType typeId, const std::vector<int>& covalentAtoms) {

    std::vector<int>& covalentList = multipoles[index].covalentInfo[typeId];
    covalentList.resize(covalentAtoms.size());
    for (unsigned int ii = 0; ii < covalentAtoms.size(); ii++) {
       covalentList[ii] = covalentAtoms[ii];
    }
}

void MPIDForce::getCovalentMap(int index, CovalentType typeId, std::vector<int>& covalentAtoms) const {

    // load covalent atom index entries for atomId==index and covalentId==typeId into covalentAtoms

    std::vector<int> covalentList = multipoles[index].covalentInfo[typeId];
    covalentAtoms.resize(covalentList.size());
    for (unsigned int ii = 0; ii < covalentList.size(); ii++) {
       covalentAtoms[ii] = covalentList[ii];
    }
}

void MPIDForce::getCovalentMaps(int index, std::vector< std::vector<int> >& covalentLists) const {

    covalentLists.resize(CovalentEnd);
    for (unsigned int jj = 0; jj < CovalentEnd; jj++) {
        std::vector<int> covalentList = multipoles[index].covalentInfo[jj];
        std::vector<int> covalentAtoms;
        covalentAtoms.resize(covalentList.size());
        for (unsigned int ii = 0; ii < covalentList.size(); ii++) {
           covalentAtoms[ii] = covalentList[ii];
        }
        covalentLists[jj] = covalentAtoms;
    }
}

void MPIDForce::setDefaultTholeWidth(double val) {
    defaultThole = val;
}

double MPIDForce::getDefaultTholeWidth() const {
    return defaultThole;
}

void MPIDForce::getInducedDipoles(Context& context, vector<Vec3>& dipoles) {
    dynamic_cast<MPIDForceImpl&>(getImplInContext(context)).getInducedDipoles(getContextImpl(context), dipoles);
}

void MPIDForce::getLabFramePermanentDipoles(Context& context, vector<Vec3>& dipoles) {
    dynamic_cast<MPIDForceImpl&>(getImplInContext(context)).getLabFramePermanentDipoles(getContextImpl(context), dipoles);
}

void MPIDForce::getTotalDipoles(Context& context, vector<Vec3>& dipoles) {
    dynamic_cast<MPIDForceImpl&>(getImplInContext(context)).getTotalDipoles(getContextImpl(context), dipoles);
}

void MPIDForce::getElectrostaticPotential(const std::vector< Vec3 >& inputGrid, Context& context, std::vector< double >& outputElectrostaticPotential) {
    dynamic_cast<MPIDForceImpl&>(getImplInContext(context)).getElectrostaticPotential(getContextImpl(context), inputGrid, outputElectrostaticPotential);
}

void MPIDForce::getSystemMultipoleMoments(Context& context, std::vector< double >& outputMultipoleMoments) {
    dynamic_cast<MPIDForceImpl&>(getImplInContext(context)).getSystemMultipoleMoments(getContextImpl(context), outputMultipoleMoments);
}

ForceImpl* MPIDForce::createImpl()  const {
    return new MPIDForceImpl(*this);
}

void MPIDForce::updateParametersInContext(Context& context) {
    dynamic_cast<MPIDForceImpl&>(getImplInContext(context)).updateParametersInContext(getContextImpl(context));
}
