
/* Portions copyright (c) 2006-2015 Stanford University and Simbios.
 * Contributors: Pande Group
 *
 * Permission is hereby granted, free of charge, to any person obtaining
 * a copy of this software and associated documentation files (the
 * "Software"), to deal in the Software without restriction, including
 * without limitation the rights to use, copy, modify, merge, publish,
 * distribute, sublicense, and/or sell copies of the Software, and to
 * permit persons to whom the Software is furnished to do so, subject
 * to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included
 * in all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
 * OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 * MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
 * IN NO EVENT SHALL THE AUTHORS, CONTRIBUTORS OR COPYRIGHT HOLDERS BE
 * LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
 * OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION
 * WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 */

#include "MPIDReferenceForce.h"
#include "jama_svd.h"
#include <algorithm>

// In case we're using some primitive version of Visual Studio this will
// make sure that erf() and erfc() are defined.
#include "openmm/internal/MSVC_erfc.h"

using std::vector;
using namespace OpenMM;

MPIDReferenceForce::MPIDReferenceForce() :
                                                   _nonbondedMethod(NoCutoff),
                                                   _numParticles(0),
                                                   _electric(138.935455846),
                                                   _dielectric(1.0),
                                                   _mutualInducedDipoleConverged(0),
                                                   _defaultTholeWidth(0.3),
                                                   _mutualInducedDipoleIterations(0),
                                                   _maximumMutualInducedDipoleIterations(100),
                                                   _mutualInducedDipoleEpsilon(1.0e+50),
                                                   _mutualInducedDipoleTargetEpsilon(1.0e-04),
                                                   _polarSOR(0.55),
                                                   _debye(48.033324)
{
    initialize();
}

MPIDReferenceForce::MPIDReferenceForce(NonbondedMethod nonbondedMethod) :
                                                   _nonbondedMethod(nonbondedMethod),
                                                   _numParticles(0),
                                                   _electric(138.935455846),
                                                   _dielectric(1.0),
                                                   _mutualInducedDipoleConverged(0),
                                                   _defaultTholeWidth(0.3),
                                                   _mutualInducedDipoleIterations(0),
                                                   _maximumMutualInducedDipoleIterations(100),
                                                   _mutualInducedDipoleEpsilon(1.0e+50),
                                                   _mutualInducedDipoleTargetEpsilon(1.0e-04),
                                                   _polarSOR(0.55),
                                                   _debye(48.033324)
{
    initialize();
}

void MPIDReferenceForce::initialize()
{
    unsigned int index    = 0;
    _mScale[index++]      = 0.0;
    _mScale[index++]      = 0.0;
    _mScale[index++]      = 0.0;
    _mScale[index++]      = 1.0;
    _mScale[index++]      = 1.0;

    index    = 0;
    _pScale[index++]      = 0.0;
    _pScale[index++]      = 0.0;
    _pScale[index++]      = 0.0;
    _pScale[index++]      = 1.0;
    _pScale[index++]      = 1.0;
}

void MPIDReferenceForce::set14ScaleFactor(double factor)
{
    _pScale[3] = _mScale[3] = factor;
}

double MPIDReferenceForce::get14ScaleFactor() const
{
    assert(_pScale[3] == _mScale[3]);
    return _pScale[3];
}

MPIDReferenceForce::NonbondedMethod MPIDReferenceForce::getNonbondedMethod() const
{
    return _nonbondedMethod;
}

void MPIDReferenceForce::setNonbondedMethod(MPIDReferenceForce::NonbondedMethod nonbondedMethod)
{
    _nonbondedMethod = nonbondedMethod;
}

MPIDReferenceForce::PolarizationType MPIDReferenceForce::getPolarizationType() const
{
    return _polarizationType;
}

void MPIDReferenceForce::setPolarizationType(MPIDReferenceForce::PolarizationType polarizationType)
{
    _polarizationType = polarizationType;
}

double MPIDReferenceForce::getDefaultTholeWidth() const
{
    return _defaultTholeWidth;
}

void MPIDReferenceForce::setDefaultTholeWidth(double width)
{
    _defaultTholeWidth = width;
}

int MPIDReferenceForce::getMutualInducedDipoleConverged() const
{
    return _mutualInducedDipoleConverged;
}

void MPIDReferenceForce::setMutualInducedDipoleConverged(int mutualInducedDipoleConverged)
{
    _mutualInducedDipoleConverged = mutualInducedDipoleConverged;
}

int MPIDReferenceForce::getMutualInducedDipoleIterations() const
{
    return _mutualInducedDipoleIterations;
}

void MPIDReferenceForce::setMutualInducedDipoleIterations(int mutualInducedDipoleIterations)
{
    _mutualInducedDipoleIterations = mutualInducedDipoleIterations;
}

double MPIDReferenceForce::getMutualInducedDipoleEpsilon() const
{
    return _mutualInducedDipoleEpsilon;
}

void MPIDReferenceForce::setMutualInducedDipoleEpsilon(double mutualInducedDipoleEpsilon)
{
    _mutualInducedDipoleEpsilon = mutualInducedDipoleEpsilon;
}

int MPIDReferenceForce::getMaximumMutualInducedDipoleIterations() const
{
    return _maximumMutualInducedDipoleIterations;
}

void MPIDReferenceForce::setMaximumMutualInducedDipoleIterations(int maximumMutualInducedDipoleIterations)
{
    _maximumMutualInducedDipoleIterations = maximumMutualInducedDipoleIterations;
}

double MPIDReferenceForce::getMutualInducedDipoleTargetEpsilon() const
{
    return _mutualInducedDipoleTargetEpsilon;
}

void MPIDReferenceForce::setExtrapolationCoefficients(const std::vector<double> &coefficients)
{
    _maxPTOrder = coefficients.size(); // This accounts for the zero-based counting; actual highest order is 1 less
    _extrapolationCoefficients = coefficients;
    _extPartCoefficients.resize(_maxPTOrder);
    for (int i = 0; i < _maxPTOrder; ++i) {
        _extPartCoefficients[i] = 0.0;
        for (int j = i; j < _maxPTOrder; ++j)
            _extPartCoefficients[i] += _extrapolationCoefficients[j];
    }
}

void MPIDReferenceForce::setMutualInducedDipoleTargetEpsilon(double mutualInducedDipoleTargetEpsilon)
{
    _mutualInducedDipoleTargetEpsilon = mutualInducedDipoleTargetEpsilon;
}

void MPIDReferenceForce::setupScaleMaps(const vector< vector< vector<int> > >& multipoleParticleCovalentInfo)
{

    /* Setup for scaling maps:
     *
     *     _scaleMaps[particleIndex][ScaleType] = map, where map[covalentIndex] = scaleFactor
     *     _maxScaleIndex[particleIndex]        = max covalent index for particleIndex
     *
     *     multipoleParticleCovalentInfo[ii][jj], jj =0,1,2,3 contains covalent indices (c12, c13, c14, c15)
     *     multipoleParticleCovalentInfo[ii][jj], jj =4,5,6,7 contains covalent indices (p11, p12, p13, p14)
     *
     *     only including covalent particles w/ index >= ii
     */

    _scaleMaps.resize(multipoleParticleCovalentInfo.size());
    _maxScaleIndex.resize(multipoleParticleCovalentInfo.size());

    for (unsigned int ii = 0; ii < multipoleParticleCovalentInfo.size(); ii++) {

        _scaleMaps[ii].resize(LAST_SCALE_TYPE_INDEX);
        _maxScaleIndex[ii] = 0;
        const vector< vector<int> >& covalentInfo = multipoleParticleCovalentInfo[ii];

        // pScale & mScale

        for (unsigned jj = 0; jj < MPIDForce::PolarizationCovalent11; jj++) {
            for (int covalentIndex : covalentInfo[jj]) {
                if (covalentIndex < ii)
                    continue;
                _scaleMaps[ii][M_SCALE][covalentIndex] = _mScale[jj+1];
                _scaleMaps[ii][P_SCALE][covalentIndex] = _pScale[jj+1];
                _maxScaleIndex[ii]                     = _maxScaleIndex[ii] < covalentIndex ? covalentIndex : _maxScaleIndex[ii];
            }
        }
    }
}

double MPIDReferenceForce::getMultipoleScaleFactor(unsigned int particleI, unsigned int particleJ, ScaleType scaleType) const
{

    MapIntRealOpenMM  scaleMap   = _scaleMaps[particleI][scaleType];
    MapIntRealOpenMMCI isPresent = scaleMap.find(particleJ);
    if (isPresent != scaleMap.end()) {
        return isPresent->second;
    } else {
        return 1.0;
    }
}

void MPIDReferenceForce::getDScaleAndPScale(unsigned int particleI, unsigned int particleJ, double& dScale, double& pScale) const
{
    pScale = getMultipoleScaleFactor(particleI, particleJ, P_SCALE);
    dScale = pScale;
}

void MPIDReferenceForce::getMultipoleScaleFactors(unsigned int particleI, unsigned int particleJ, vector<double>& scaleFactors) const
{
    scaleFactors[P_SCALE] = getMultipoleScaleFactor(particleI, particleJ, P_SCALE);
    scaleFactors[M_SCALE] = getMultipoleScaleFactor(particleI, particleJ, M_SCALE);
}

double MPIDReferenceForce::normalizeVec3(Vec3& vectorToNormalize) const
{
    double norm = sqrt(vectorToNormalize.dot(vectorToNormalize));
    if (norm > 0.0) {
        vectorToNormalize *= (1.0/norm);
    }
    return norm;
}

void MPIDReferenceForce::initializeRealOpenMMVector(vector<double>& vectorToInitialize) const
{
    double zero = 0.0;
    vectorToInitialize.resize(_numParticles);
    std::fill(vectorToInitialize.begin(), vectorToInitialize.end(), zero);
}

void MPIDReferenceForce::initializeVec3Vector(vector<Vec3>& vectorToInitialize) const
{
    vectorToInitialize.resize(_numParticles);
    Vec3 zeroVec(0.0, 0.0, 0.0);
    std::fill(vectorToInitialize.begin(), vectorToInitialize.end(), zeroVec);
}

void MPIDReferenceForce::copyVec3Vector(const vector<OpenMM::Vec3>& inputVector, vector<OpenMM::Vec3>& outputVector) const
{
    outputVector.resize(inputVector.size());
    for (unsigned int ii = 0; ii < inputVector.size(); ii++) {
        outputVector[ii] = inputVector[ii];
    }
}

void MPIDReferenceForce::loadParticleData(const vector<Vec3>& particlePositions,
                                                     const vector<double>& charges,
                                                     const vector<double>& dipoles,
                                                     const vector<double>& quadrupoles,
                                                     const vector<double>& octopoles,
                                                     const vector<double>& tholes,
                                                     const vector<double>& dampingFactors,
                                                     const vector<std::vector<double> >& polarity,
                                                     vector<MultipoleParticleData>& particleData) const
{

    particleData.resize(_numParticles);

    for (unsigned int ii = 0; ii < _numParticles; ii++) {

        particleData[ii].particleIndex        = ii;
        particleData[ii].position             = particlePositions[ii];
        particleData[ii].charge               = charges[ii];

        double dX = particleData[ii].dipole[0] = dipoles[3*ii+0];
        double dY = particleData[ii].dipole[1] = dipoles[3*ii+1];
        double dZ = particleData[ii].dipole[2] = dipoles[3*ii+2];

        double qXX = particleData[ii].quadrupole[QXX] = quadrupoles[6*ii+0];
        double qXY = particleData[ii].quadrupole[QXY] = quadrupoles[6*ii+1];
        double qXZ = particleData[ii].quadrupole[QXZ] = quadrupoles[6*ii+3];
        double qYY = particleData[ii].quadrupole[QYY] = quadrupoles[6*ii+2];
        double qYZ = particleData[ii].quadrupole[QYZ] = quadrupoles[6*ii+4];
        double qZZ = particleData[ii].quadrupole[QZZ] = quadrupoles[6*ii+5];

        double oXXX = particleData[ii].octopole[QXXX] = octopoles[10*ii+0];
        double oXXY = particleData[ii].octopole[QXXY] = octopoles[10*ii+1];
        double oXXZ = particleData[ii].octopole[QXXZ] = octopoles[10*ii+4];
        double oXYY = particleData[ii].octopole[QXYY] = octopoles[10*ii+2];
        double oXYZ = particleData[ii].octopole[QXYZ] = octopoles[10*ii+5];
        double oXZZ = particleData[ii].octopole[QXZZ] = octopoles[10*ii+7];
        double oYYY = particleData[ii].octopole[QYYY] = octopoles[10*ii+3];
        double oYYZ = particleData[ii].octopole[QYYZ] = octopoles[10*ii+6];
        double oYZZ = particleData[ii].octopole[QYZZ] = octopoles[10*ii+8];
        double oZZZ = particleData[ii].octopole[QZZZ] = octopoles[10*ii+9];

        // Form spherical harmonic dipoles from Cartesian moments.
        particleData[ii].sphericalDipole[0]  = dZ; // z -> Q_10
        particleData[ii].sphericalDipole[1]  = dX; // x -> Q_11c
        particleData[ii].sphericalDipole[2]  = dY; // y -> Q_11s

        // Form spherical harmonic quadrupoles from Cartesian moments.
        particleData[ii].sphericalQuadrupole[0] = qZZ*3; // zz -> Q_20
        particleData[ii].sphericalQuadrupole[1] = sqrtFourThirds * qXZ * 3; // xz -> Q_21c
        particleData[ii].sphericalQuadrupole[2] = sqrtFourThirds * qYZ * 3; // yz -> Q_21s
        particleData[ii].sphericalQuadrupole[3] = sqrtOneThird * (qXX - qYY) * 3; // xx-yy -> Q_22c
        particleData[ii].sphericalQuadrupole[4] = sqrtFourThirds * qXY * 3; // xy -> Q_22s

        // Form spherical harmonic octopoles from Cartesian moments.
        particleData[ii].sphericalOctopole[0] = oZZZ*15; // zzz -> Q_30
        particleData[ii].sphericalOctopole[1] = sqrtThreeHalves*oXZZ*15; // xzz -> Q_31c
        particleData[ii].sphericalOctopole[2] = sqrtThreeHalves*oYZZ*15; // yzz -> Q_31s
        particleData[ii].sphericalOctopole[3] = sqrtThreeFifths*(oXXZ-oYYZ)*15; // xxz-yyz -> Q_32c
        particleData[ii].sphericalOctopole[4] = 2*sqrtThreeFifths*oXYZ*15; // xyz -> Q_32s
        particleData[ii].sphericalOctopole[5] = sqrtOneTenth*(oXXX-3*oXYY)*15; // xxx-xyy -> Q_33c
        particleData[ii].sphericalOctopole[6] = sqrtOneTenth*(3*oXXY-oYYY)*15; // xxy-yyy -> Q_33s

        particleData[ii].thole                = tholes[ii];
        particleData[ii].dampingFactor        = dampingFactors[ii];
        particleData[ii].polarity             = polarity[ii];
        particleData[ii].isAnisotropic        = polarity[ii][0] != polarity[ii][1] || polarity[ii][0] != polarity[ii][2];

    }
}

void MPIDReferenceForce::zeroFixedMultipoleFields()
{
    initializeVec3Vector(_fixedMultipoleField);
}

void MPIDReferenceForce::checkChiralCenterAtParticle(MultipoleParticleData& particleI, int axisType,
                                                                MultipoleParticleData& particleZ, MultipoleParticleData& particleX,
                                                                MultipoleParticleData& particleY) const
{

    if (axisType != MPIDForce::ZThenX || particleY.particleIndex == -1) {
        return;
    }

    Vec3 deltaAD   = particleI.position - particleY.position;
    Vec3 deltaBD   = particleZ.position - particleY.position;
    Vec3 deltaCD   = particleX.position - particleY.position;

    Vec3 deltaC    = deltaBD.cross(deltaCD);
    double volume = deltaC.dot(deltaAD);

    if (volume < 0.0) {
        particleI.dipole[1] *= -1.0; // pole(3,i)
        particleI.quadrupole[QXY] *= -1.0; // pole(6,i) && pole(8,i)
        particleI.quadrupole[QYZ] *= -1.0; // pole(10,i) && pole(12,i)
        particleI.sphericalDipole[2]     *= -1.0;   // y
        particleI.sphericalQuadrupole[2] *= -1.0;   // yz
        particleI.sphericalQuadrupole[4] *= -1.0;   // xy
    }
}

void MPIDReferenceForce::checkChiral(vector<MultipoleParticleData>& particleData,
                                                const vector<int>& multipoleAtomXs,
                                                const vector<int>& multipoleAtomYs,
                                                const vector<int>& multipoleAtomZs,
                                                const vector<int>& axisTypes) const
{
    for (unsigned int ii = 0; ii < _numParticles; ii++) {
        if (multipoleAtomYs[ii] > -1) {
            checkChiralCenterAtParticle(particleData[ii], axisTypes[ii],
                                        particleData[multipoleAtomZs[ii]],
                                        particleData[multipoleAtomXs[ii]],
                                        particleData[multipoleAtomYs[ii]]);
        }
    }
}

void MPIDReferenceForce::applyRotationMatrixToParticle(      MultipoleParticleData& particleI,
                                                                  const MultipoleParticleData* particleZ,
                                                                  const MultipoleParticleData* particleX,
                                                                        MultipoleParticleData* particleY,
                                                                        int axisType) const
{

    // handle case where rotation matrix is identity (e.g. single ion)

    // get coordinates of this atom and the z & x axis atoms
    // compute the vector between the atoms and 1/sqrt(d2), d2 is distance between
    // this atom and the axis atom


    Vec3 vectorX, vectorY;
    Vec3 vectorZ = particleZ->position - particleI.position;
    normalizeVec3(vectorZ);

    // branch based on axis type

    if (axisType == MPIDForce::ZOnly) {

        // z-only

        if (fabs(vectorZ[0]) < 0.866)
            vectorX = Vec3(1.0, 0.0, 0.0);
        else
            vectorX = Vec3(0.0, 1.0, 0.0);
    }
    else {
        vectorX = particleX->position - particleI.position;
        if (axisType == MPIDForce::Bisector) {

            // bisector

            // dx = dx1 + dx2 (in TINKER code)

            normalizeVec3(vectorX);
            vectorZ += vectorX;
            normalizeVec3(vectorZ);
        }
        else if (axisType == MPIDForce::ZBisect) {

            // z-bisect

            // dx = dx1 + dx2 (in TINKER code)

            normalizeVec3(vectorX);

            vectorY  = particleY->position - particleI.position;
            normalizeVec3(vectorY);

            vectorX += vectorY;
            normalizeVec3(vectorX);
        }
        else if (axisType == MPIDForce::ThreeFold) {

            // 3-fold

            // dx = dx1 + dx2 + dx3 (in TINKER code)

            normalizeVec3(vectorX);

            vectorY   = particleY->position - particleI.position;
            normalizeVec3(vectorY);

            vectorZ  += vectorX +  vectorY;
            normalizeVec3(vectorZ);
        }
    }

    double dot = vectorZ.dot(vectorX);
    vectorX -= vectorZ*dot;

    normalizeVec3(vectorX);
    vectorY = vectorZ.cross(vectorX);

    Vec3 rotationMatrix[3];
    rotationMatrix[0] = vectorX;
    rotationMatrix[1] = vectorY;
    rotationMatrix[2] = vectorZ;

    Vec3 labDipole;
    for (int ii = 0; ii < 3; ii++) {
        labDipole[ii] = particleI.dipole[0]*rotationMatrix[0][ii];
        for (int jj = 1; jj < 3; jj++) {
            labDipole[ii] += particleI.dipole[jj]*rotationMatrix[jj][ii];
        }
    }
    particleI.dipole = labDipole;
    double mPole[3][3];
    double rPole[3][3] = { { 0.0, 0.0, 0.0 },
                           { 0.0, 0.0, 0.0 },
                           { 0.0, 0.0, 0.0 } };

    mPole[0][0] = particleI.quadrupole[QXX];
    mPole[0][1] = particleI.quadrupole[QXY];
    mPole[0][2] = particleI.quadrupole[QXZ];

    mPole[1][0] = particleI.quadrupole[QXY];
    mPole[1][1] = particleI.quadrupole[QYY];
    mPole[1][2] = particleI.quadrupole[QYZ];

    mPole[2][0] = particleI.quadrupole[QXZ];
    mPole[2][1] = particleI.quadrupole[QYZ];
    mPole[2][2] = particleI.quadrupole[QZZ];

    for (int ii = 0; ii < 3; ii++) {
       for (int jj = ii; jj < 3; jj++) {
          for (int kk = 0; kk < 3; kk++) {
             for (int mm = 0; mm < 3; mm++) {
                 rPole[ii][jj] += rotationMatrix[kk][ii]*rotationMatrix[mm][jj]*mPole[kk][mm];
             }
          }
       }
    }

    particleI.quadrupole[QXX] = rPole[0][0];
    particleI.quadrupole[QXY] = rPole[0][1];
    particleI.quadrupole[QXZ] = rPole[0][2];

    particleI.quadrupole[QYY] = rPole[1][1];
    particleI.quadrupole[QYZ] = rPole[1][2];
    particleI.quadrupole[QZZ] = rPole[2][2];

    // Rotate the polarizabilities
    double bAlpha[3][3] = { { 0.0, 0.0, 0.0 },
                            { 0.0, 0.0, 0.0 },
                            { 0.0, 0.0, 0.0 } };
    double lAlpha[3][3] = { { 0.0, 0.0, 0.0 },
                            { 0.0, 0.0, 0.0 },
                            { 0.0, 0.0, 0.0 } };
    bAlpha[0][0] = particleI.polarity[0];
    bAlpha[1][1] = particleI.polarity[1];
    bAlpha[2][2] = particleI.polarity[2];

    for (int ii = 0; ii < 3; ii++) {
       for (int jj = ii; jj < 3; jj++) {
          for (int kk = 0; kk < 3; kk++) {
             for (int mm = 0; mm < 3; mm++) {
                 lAlpha[ii][jj] += rotationMatrix[kk][ii]*rotationMatrix[mm][jj]*bAlpha[kk][mm];
             }
          }
       }
    }
    particleI.labPolarization[QXX] = lAlpha[0][0];
    particleI.labPolarization[QXY] = lAlpha[0][1];
    particleI.labPolarization[QXZ] = lAlpha[0][2];
    particleI.labPolarization[QYY] = lAlpha[1][1];
    particleI.labPolarization[QYZ] = lAlpha[1][2];
    particleI.labPolarization[QZZ] = lAlpha[2][2];


    double laboPole[3][3][3] = {{{ 0.0, 0.0, 0.0 },
                                 { 0.0, 0.0, 0.0 },
                                 { 0.0, 0.0, 0.0 }},
                                {{ 0.0, 0.0, 0.0 },
                                 { 0.0, 0.0, 0.0 },
                                 { 0.0, 0.0, 0.0 }},
                                {{ 0.0, 0.0, 0.0 },
                                 { 0.0, 0.0, 0.0 },
                                 { 0.0, 0.0, 0.0 }}};
    double bodyoPole[3][3][3] = { { { particleI.octopole[QXXX], particleI.octopole[QXXY], particleI.octopole[QXXZ] },
                                    { particleI.octopole[QXXY], particleI.octopole[QXYY], particleI.octopole[QXYZ] },
                                    { particleI.octopole[QXXZ], particleI.octopole[QXYZ], particleI.octopole[QXZZ] } },
                                  { { particleI.octopole[QXXY], particleI.octopole[QXYY], particleI.octopole[QXYZ] },
                                    { particleI.octopole[QXYY], particleI.octopole[QYYY], particleI.octopole[QYYZ] },
                                    { particleI.octopole[QXYZ], particleI.octopole[QYYZ], particleI.octopole[QYZZ] } },
                                  { { particleI.octopole[QXXZ], particleI.octopole[QXYZ], particleI.octopole[QXZZ] },
                                    { particleI.octopole[QXYZ], particleI.octopole[QYYZ], particleI.octopole[QYZZ] },
                                    { particleI.octopole[QXZZ], particleI.octopole[QYZZ], particleI.octopole[QZZZ] } } };
    for (int ii = 0; ii < 3; ii++) {
       for (int jj = ii; jj < 3; jj++) {
          for (int kk = 0; kk < 3; kk++) {
             for (int ll = 0; ll < 3; ll++) {
                for (int mm = 0; mm < 3; mm++) {
                   for (int oo = 0; oo < 3; oo++) {
                      laboPole[ii][jj][kk] += rotationMatrix[ll][ii]*rotationMatrix[mm][jj]*rotationMatrix[oo][kk]*bodyoPole[ll][mm][oo];
                   }
                }
             }
          }
       }
    }
    particleI.octopole[QXXX] = laboPole[0][0][0];
    particleI.octopole[QXXY] = laboPole[0][0][1];
    particleI.octopole[QXXZ] = laboPole[0][0][2];
    particleI.octopole[QXYY] = laboPole[0][1][1];
    particleI.octopole[QXYZ] = laboPole[0][1][2];
    particleI.octopole[QXZZ] = laboPole[0][2][2];
    particleI.octopole[QYYY] = laboPole[1][1][1];
    particleI.octopole[QYYZ] = laboPole[1][1][2];
    particleI.octopole[QYZZ] = laboPole[1][2][2];
    particleI.octopole[QZZZ] = laboPole[2][2][2];

    double dipoleRotationMatrix[3][3];

    // Reorder the Cartesian {x,y,z} dipole rotation matrix, to account
    // for spherical harmonic ordering {z,x,y}.
    dipoleRotationMatrix[0][0] = vectorZ[2];
    dipoleRotationMatrix[0][1] = vectorX[2];
    dipoleRotationMatrix[0][2] = vectorY[2];
    dipoleRotationMatrix[1][0] = vectorZ[0];
    dipoleRotationMatrix[1][1] = vectorX[0];
    dipoleRotationMatrix[1][2] = vectorY[0];
    dipoleRotationMatrix[2][0] = vectorZ[1];
    dipoleRotationMatrix[2][1] = vectorX[1];
    dipoleRotationMatrix[2][2] = vectorY[1];

    double quadrupoleRotationMatrix[5][5];
    buildSphericalQuadrupoleRotationMatrix(dipoleRotationMatrix, quadrupoleRotationMatrix);
    double octopoleRotationMatrix[7][7];
    buildSphericalOctopoleRotationMatrix(dipoleRotationMatrix, quadrupoleRotationMatrix, octopoleRotationMatrix);

    // Rotate the dipoles
    double rotatedDipole[3];
    for (int ii = 0; ii < 3; ii++) {
        double val = 0.0;
        for (int jj = 0; jj < 3; jj++) {
            val += dipoleRotationMatrix[ii][jj] * particleI.sphericalDipole[jj];
        }
        rotatedDipole[ii] = val;
    }
    for (int ii = 0; ii < 3; ii++)
        particleI.sphericalDipole[ii] = rotatedDipole[ii];

    // Rotate the quadrupoles
    double rotatedQuadrupole[5];
    for (int ii = 0; ii < 5; ii++) {
        double val = 0.0;
        for (int jj = 0; jj < 5; jj++) {
            val += quadrupoleRotationMatrix[ii][jj] * particleI.sphericalQuadrupole[jj];
        }
        rotatedQuadrupole[ii] = val;
    }
    for (int ii = 0; ii < 5; ii++)
        particleI.sphericalQuadrupole[ii] = rotatedQuadrupole[ii];

    // Rotate the octopoles
    double rotatedOctopole[7];
    for (int ii = 0; ii < 7; ii++) {
        double val = 0.0;
        for (int jj = 0; jj < 7; jj++) {
            val += octopoleRotationMatrix[ii][jj] * particleI.sphericalOctopole[jj];
        }
        rotatedOctopole[ii] = val;
    }
    for (int ii = 0; ii < 7; ii++)
        particleI.sphericalOctopole[ii] = rotatedOctopole[ii];
}

void MPIDReferenceForce::formQIRotationMatrix(const Vec3& iPosition,
                                                         const Vec3& jPosition,
                                                         const Vec3 &deltaR,
                                                         double r,
                                                         double (&rotationMatrix)[3][3]) const
{
    Vec3 vectorZ = (deltaR)/r;
    Vec3 vectorX(vectorZ);
    if ((iPosition[1] != jPosition[1]) || (iPosition[2] != jPosition[2])) {
        vectorX[0] += 1.0;
    }else{
        vectorX[1] += 1.0;
    }
    Vec3 vectorY;

    double dot = vectorZ.dot(vectorX);
    vectorX -= vectorZ*dot;
    normalizeVec3(vectorX);
    vectorY = vectorZ.cross(vectorX);

    // Reorder the Cartesian {x,y,z} dipole rotation matrix, to account
    // for spherical harmonic ordering {z,x,y}.
    rotationMatrix[0][0] = vectorZ[2];
    rotationMatrix[0][1] = vectorZ[0];
    rotationMatrix[0][2] = vectorZ[1];
    rotationMatrix[1][0] = vectorX[2];
    rotationMatrix[1][1] = vectorX[0];
    rotationMatrix[1][2] = vectorX[1];
    rotationMatrix[2][0] = vectorY[2];
    rotationMatrix[2][1] = vectorY[0];
    rotationMatrix[2][2] = vectorY[1];
}




void MPIDReferenceForce::buildSphericalQuadrupoleRotationMatrix(const double (&D1)[3][3], double (&D2)[5][5]) const
{
    D2[0][0] = 0.5*(3.0*D1[0][0]*D1[0][0] - 1.0);
    D2[1][0] = sqrtThree*D1[0][0]*D1[1][0];
    D2[2][0] = sqrtThree*D1[0][0]*D1[2][0];
    D2[3][0] = 0.5*sqrtThree*(D1[1][0]*D1[1][0] - D1[2][0]*D1[2][0]);
    D2[4][0] = sqrtThree*D1[1][0]*D1[2][0];
    D2[0][1] = sqrtThree*D1[0][0]*D1[0][1];
    D2[1][1] = D1[1][0]*D1[0][1] + D1[0][0]*D1[1][1];
    D2[2][1] = D1[2][0]*D1[0][1] + D1[0][0]*D1[2][1];
    D2[3][1] = D1[1][0]*D1[1][1] - D1[2][0]*D1[2][1];
    D2[4][1] = D1[2][0]*D1[1][1] + D1[1][0]*D1[2][1];
    D2[0][2] = sqrtThree*D1[0][0]*D1[0][2];
    D2[1][2] = D1[1][0]*D1[0][2] + D1[0][0]*D1[1][2];
    D2[2][2] = D1[2][0]*D1[0][2] + D1[0][0]*D1[2][2];
    D2[3][2] = D1[1][0]*D1[1][2] - D1[2][0]*D1[2][2];
    D2[4][2] = D1[2][0]*D1[1][2] + D1[1][0]*D1[2][2];
    D2[0][3] = 0.5*sqrtThree*(D1[0][1]*D1[0][1] - D1[0][2]*D1[0][2]);
    D2[1][3] = D1[0][1]*D1[1][1] - D1[0][2]*D1[1][2];
    D2[2][3] = D1[0][1]*D1[2][1] - D1[0][2]*D1[2][2];
    D2[3][3] = 0.5*(D1[1][1]*D1[1][1] - D1[2][1]*D1[2][1] - D1[1][2]*D1[1][2] + D1[2][2]*D1[2][2]);
    D2[4][3] = D1[1][1]*D1[2][1] - D1[1][2]*D1[2][2];
    D2[0][4] = sqrtThree*D1[0][1]*D1[0][2];
    D2[1][4] = D1[1][1]*D1[0][2] + D1[0][1]*D1[1][2];
    D2[2][4] = D1[2][1]*D1[0][2] + D1[0][1]*D1[2][2];
    D2[3][4] = D1[1][1]*D1[1][2] - D1[2][1]*D1[2][2];
    D2[4][4] = D1[2][1]*D1[1][2] + D1[1][1]*D1[2][2];
}

void MPIDReferenceForce::buildSphericalOctopoleRotationMatrix(const double (&D1)[3][3], const double (&D2)[5][5], double (&D3)[7][7]) const
{
    D3[0][0] = D1[0][0]*D2[0][0] - 0.5773502691896258*(D1[1][0]*D2[1][0] + D1[2][0]*D2[2][0]);
    D3[0][1] = 0.25*(4.242640687119285*D1[0][0]*D2[0][1] - 2.449489742783178*(D1[1][0]*D2[1][1] + D1[2][0]*D2[2][1]));
    D3[0][2] = 0.25*(4.242640687119285*D1[0][0]*D2[0][2] - 2.449489742783178*(D1[1][0]*D2[1][2] + D1[2][0]*D2[2][2]));
    D3[0][3] = 0.4472135954999579*(3.*D1[0][0]*D2[0][3] - 1.732050807568877*(D1[1][0]*D2[1][3] + D1[2][0]*D2[2][3]));
    D3[0][4] = 0.4472135954999579*(3.*D1[0][0]*D2[0][4] - 1.732050807568877*(D1[1][0]*D2[1][4] + D1[2][0]*D2[2][4]));
    D3[0][5] = 0.3162277660168379*(1.732050807568877*(D1[0][1]*D2[0][3] - D1[0][2]*D2[0][4]) - D1[1][1]*D2[1][3] + D1[1][2]*D2[1][4] - D1[2][1]*D2[2][3] + D1[2][2]*D2[2][4]);
    D3[0][6] = 0.5477225575051661*(D1[0][2]*D2[0][3] + D1[0][1]*D2[0][4]) - 0.3162277660168379*(D1[1][2]*D2[1][3] + D1[1][1]*D2[1][4] + D1[2][2]*D2[2][3] + D1[2][1]*D2[2][4]);
    D3[1][0] = 0.2357022603955158*(3.464101615137755*D1[1][0]*D2[0][0] + 4.*D1[0][0]*D2[1][0] - D1[1][0]*D2[3][0] - D1[2][0]*D2[4][0]);
    D3[1][1] = 0.25*(3.464101615137755*D1[1][0]*D2[0][1] + 4.*D1[0][0]*D2[1][1] - D1[1][0]*D2[3][1] - D1[2][0]*D2[4][1]);
    D3[1][2] = 0.25*(3.464101615137755*D1[1][0]*D2[0][2] + 4.*D1[0][0]*D2[1][2] - D1[1][0]*D2[3][2] - D1[2][0]*D2[4][2]);
    D3[1][3] = 0.3162277660168379*(3.464101615137755*D1[1][0]*D2[0][3] + 4.*D1[0][0]*D2[1][3] - D1[1][0]*D2[3][3] - D1[2][0]*D2[4][3]);
    D3[1][4] = 0.3162277660168379*(3.464101615137755*D1[1][0]*D2[0][4] + 4.*D1[0][0]*D2[1][4] - D1[1][0]*D2[3][4] - D1[2][0]*D2[4][4]);
    D3[1][5] = 0.07453559924999299*(-6.*D1[1][2]*D2[0][4] + D1[1][1]*(6.*D2[0][3] - 1.732050807568877*D2[3][3]) + 1.732050807568877*(4.*D1[0][1]*D2[1][3] - 4.*D1[0][2]*D2[1][4] + D1[1][2]*D2[3][4] - D1[2][1]*D2[4][3] + D1[2][2]*D2[4][4]));
    D3[1][6] = 0.07453559924999299*(6.*(D1[1][2]*D2[0][3] + D1[1][1]*D2[0][4]) + 6.928203230275509*(D1[0][2]*D2[1][3] + D1[0][1]*D2[1][4]) - 1.732050807568877*(D1[1][2]*D2[3][3] + D1[1][1]*D2[3][4] + D1[2][2]*D2[4][3] + D1[2][1]*D2[4][4]));
    D3[2][0] = 0.2357022603955158*(4.*D1[0][0]*D2[2][0] + D1[2][0]*(3.464101615137755*D2[0][0] + D2[3][0]) - D1[1][0]*D2[4][0]);
    D3[2][1] = 0.25*(4.*D1[0][0]*D2[2][1] + D1[2][0]*(3.464101615137755*D2[0][1] + D2[3][1]) - D1[1][0]*D2[4][1]);
    D3[2][2] = 0.25*(4.*D1[0][0]*D2[2][2] + D1[2][0]*(3.464101615137755*D2[0][2] + D2[3][2]) - D1[1][0]*D2[4][2]);
    D3[2][3] = 0.3162277660168379*(4.*D1[0][0]*D2[2][3] + D1[2][0]*(3.464101615137755*D2[0][3] + D2[3][3]) - D1[1][0]*D2[4][3]);
    D3[2][4] = 0.3162277660168379*(4.*D1[0][0]*D2[2][4] + D1[2][0]*(3.464101615137755*D2[0][4] + D2[3][4]) - D1[1][0]*D2[4][4]);
    D3[2][5] = 0.07453559924999299*(-6.*D1[2][2]*D2[0][4] + D1[2][1]*(6.*D2[0][3] + 1.732050807568877*D2[3][3]) + 1.732050807568877*(4.*D1[0][1]*D2[2][3] - 4.*D1[0][2]*D2[2][4] - D1[2][2]*D2[3][4] - D1[1][1]*D2[4][3] + D1[1][2]*D2[4][4]));
    D3[2][6] = 0.07453559924999299*(6.*D1[2][1]*D2[0][4] + D1[2][2]*(6.*D2[0][3] + 1.732050807568877*D2[3][3]) + 1.732050807568877*(4.*D1[0][2]*D2[2][3] + 4.*D1[0][1]*D2[2][4] + D1[2][1]*D2[3][4] - D1[1][2]*D2[4][3] - D1[1][1]*D2[4][4]));
    D3[3][0] = 0.7453559924999299*(D1[1][0]*D2[1][0] - D1[2][0]*D2[2][0] + D1[0][0]*D2[3][0]);
    D3[3][1] = 0.7905694150420948*(D1[1][0]*D2[1][1] - D1[2][0]*D2[2][1] + D1[0][0]*D2[3][1]);
    D3[3][2] = 0.7905694150420948*(D1[1][0]*D2[1][2] - D1[2][0]*D2[2][2] + D1[0][0]*D2[3][2]);
    D3[3][3] = D1[1][0]*D2[1][3] - D1[2][0]*D2[2][3] + D1[0][0]*D2[3][3];
    D3[3][4] = D1[1][0]*D2[1][4] - D1[2][0]*D2[2][4] + D1[0][0]*D2[3][4];
    D3[3][5] = 0.408248290463863*(D1[1][1]*D2[1][3] - D1[1][2]*D2[1][4] - D1[2][1]*D2[2][3] + D1[2][2]*D2[2][4] + D1[0][1]*D2[3][3] - D1[0][2]*D2[3][4]);
    D3[3][6] = 0.408248290463863*(D1[1][2]*D2[1][3] + D1[1][1]*D2[1][4] - D1[2][2]*D2[2][3] - D1[2][1]*D2[2][4] + D1[0][2]*D2[3][3] + D1[0][1]*D2[3][4]);
    D3[4][0] = 0.7453559924999299*(D1[2][0]*D2[1][0] + D1[1][0]*D2[2][0] + D1[0][0]*D2[4][0]);
    D3[4][1] = 0.7905694150420948*(D1[2][0]*D2[1][1] + D1[1][0]*D2[2][1] + D1[0][0]*D2[4][1]);
    D3[4][2] = 0.7905694150420948*(D1[2][0]*D2[1][2] + D1[1][0]*D2[2][2] + D1[0][0]*D2[4][2]);
    D3[4][3] = D1[2][0]*D2[1][3] + D1[1][0]*D2[2][3] + D1[0][0]*D2[4][3];
    D3[4][4] = D1[2][0]*D2[1][4] + D1[1][0]*D2[2][4] + D1[0][0]*D2[4][4];
    D3[4][5] = 0.408248290463863*(D1[2][1]*D2[1][3] - D1[2][2]*D2[1][4] + D1[1][1]*D2[2][3] - D1[1][2]*D2[2][4] + D1[0][1]*D2[4][3] - D1[0][2]*D2[4][4]);
    D3[4][6] = 0.408248290463863*(D1[2][2]*D2[1][3] + D1[2][1]*D2[1][4] + D1[1][2]*D2[2][3] + D1[1][1]*D2[2][4] + D1[0][2]*D2[4][3] + D1[0][1]*D2[4][4]);
    D3[5][0] = 0.9128709291752769*(D1[1][0]*D2[3][0] - D1[2][0]*D2[4][0]);
    D3[5][1] = 0.9682458365518542*(D1[1][0]*D2[3][1] - D1[2][0]*D2[4][1]);
    D3[5][2] = 0.9682458365518542*(D1[1][0]*D2[3][2] - D1[2][0]*D2[4][2]);
    D3[5][3] = 1.224744871391589*(D1[1][0]*D2[3][3] - D1[2][0]*D2[4][3]);
    D3[5][4] = 1.224744871391589*(D1[1][0]*D2[3][4] - D1[2][0]*D2[4][4]);
    D3[5][5] = 0.5*(D1[1][1]*D2[3][3] - D1[1][2]*D2[3][4] - D1[2][1]*D2[4][3] + D1[2][2]*D2[4][4]);
    D3[5][6] = 0.5*(D1[1][2]*D2[3][3] + D1[1][1]*D2[3][4] - D1[2][2]*D2[4][3] - D1[2][1]*D2[4][4]);
    D3[6][0] = 0.9128709291752769*(D1[2][0]*D2[3][0] + D1[1][0]*D2[4][0]);
    D3[6][1] = 0.9682458365518542*(D1[2][0]*D2[3][1] + D1[1][0]*D2[4][1]);
    D3[6][2] = 0.9682458365518542*(D1[2][0]*D2[3][2] + D1[1][0]*D2[4][2]);
    D3[6][3] = 1.224744871391589*(D1[2][0]*D2[3][3] + D1[1][0]*D2[4][3]);
    D3[6][4] = 1.224744871391589*(D1[2][0]*D2[3][4] + D1[1][0]*D2[4][4]);
    D3[6][5] = 0.5*(D1[2][1]*D2[3][3] - D1[2][2]*D2[3][4] + D1[1][1]*D2[4][3] - D1[1][2]*D2[4][4]);
    D3[6][6] = 0.5*(D1[2][2]*D2[3][3] + D1[2][1]*D2[3][4] + D1[1][2]*D2[4][3] + D1[1][1]*D2[4][4]);
}


void MPIDReferenceForce::buildPartialSphericalQuadrupoleRotationMatrix(const double (&D1)[3][3], double (&D2)[3][5]) const
{
    D2[0][0] = 0.5*(3.0*D1[0][0]*D1[0][0] - 1.0);
    D2[0][1] = sqrtThree*D1[0][0]*D1[0][1];
    D2[0][2] = sqrtThree*D1[0][0]*D1[0][2];
    D2[0][3] = 0.5*sqrtThree*(D1[0][1]*D1[0][1] - D1[0][2]*D1[0][2]);
    D2[0][4] = sqrtThree*D1[0][1]*D1[0][2];
    D2[1][0] = sqrtThree*D1[0][0]*D1[1][0];
    D2[1][1] = D1[1][0]*D1[0][1] + D1[0][0]*D1[1][1];
    D2[1][2] = D1[1][0]*D1[0][2] + D1[0][0]*D1[1][2];
    D2[1][3] = D1[0][1]*D1[1][1] - D1[0][2]*D1[1][2];
    D2[1][4] = D1[1][1]*D1[0][2] + D1[0][1]*D1[1][2];
    D2[2][0] = sqrtThree*D1[0][0]*D1[2][0];
    D2[2][1] = D1[2][0]*D1[0][1] + D1[0][0]*D1[2][1];
    D2[2][2] = D1[2][0]*D1[0][2] + D1[0][0]*D1[2][2];
    D2[2][3] = D1[0][1]*D1[2][1] - D1[0][2]*D1[2][2];
    D2[2][4] = D1[2][1]*D1[0][2] + D1[0][1]*D1[2][2];
}

void MPIDReferenceForce::applyRotationMatrix(vector<MultipoleParticleData>& particleData,
                                                        const vector<int>& multipoleAtomXs,
                                                        const vector<int>& multipoleAtomYs,
                                                        const vector<int>& multipoleAtomZs,
                                                        const vector<int>& axisTypes) const
{

    for (unsigned int ii = 0; ii < _numParticles; ii++) {
        if (multipoleAtomZs[ii] >= 0) {
            applyRotationMatrixToParticle(particleData[ii], &particleData[multipoleAtomZs[ii]],
                                          multipoleAtomXs[ii] > -1 ? &particleData[multipoleAtomXs[ii]] : NULL,
                                          multipoleAtomYs[ii] > -1 ? &particleData[multipoleAtomYs[ii]] : NULL, axisTypes[ii]);
        }
    }
}

void MPIDReferenceForce::getAndScaleInverseRs(double dampI, double dampJ, double pscale,
                                                         double tholeI, double tholeJ,
                                                         double r, vector<double>& rrI) const
{

    double rI             =  1.0/r;
    double r2I            =  rI*rI;

    rrI[0]                = rI*r2I;
    double constantFactor = 3.0;
    for (unsigned int ii  = 1; ii < rrI.size(); ii++) {
       rrI[ii]         = constantFactor*rrI[ii-1]*r2I;
       constantFactor += 2.0;
    }

    double damp = dampI*dampJ;
    if (damp != 0.0) {
        double ratio   = (r/damp);

        double pgamma  = pscale == 0.0 ? tholeI + tholeJ : _defaultTholeWidth;
               damp    = pgamma*ratio;
        if (damp < 50.0) {
            double expdamp = exp(-damp);
            rrI[0] *= 1.0 - expdamp*(1.0 + damp + 0.5*damp*damp);
            rrI[1] *= 1.0 - expdamp*(1.0 + damp + 0.5*damp*damp + damp*damp*damp/6.0);
            if(rrI.size()>2)
                rrI[2] *= 1.0 - expdamp*(1.0 + damp + 0.5*damp*damp + damp*damp*damp/6.0 + damp*damp*damp*damp/30.0);
            if(rrI.size()>3)
                rrI[3] *= 1.0 - expdamp*(1.0 + damp + 0.5*damp*damp + damp*damp*damp/6.0 + 4.0*damp*damp*damp*damp/105.0 + damp*damp*damp*damp*damp/210.0);
        }
    }
}

void MPIDReferenceForce::calculateFixedMultipoleFieldPairIxn(const MultipoleParticleData& particleI,
                                                                        const MultipoleParticleData& particleJ,
                                                                        double dScale, double pScale)
{

    if (particleI.particleIndex == particleJ.particleIndex)
        return;

    Vec3 deltaR = particleJ.position - particleI.position;
    double r = sqrt(deltaR.dot(deltaR));

    vector<double> rrI(4);

    // get scaling factors, if needed

    getAndScaleInverseRs(particleI.dampingFactor, particleJ.dampingFactor, pScale, particleI.thole, particleJ.thole, r, rrI);
    double rr3    = rrI[0];
    double rr5    = rrI[1];
    double rr7    = rrI[2];
    double rr9    = rrI[3];
    double rr5_2  = 2.0*rr5;
    double rr7_3  = 3.0*rr7;

    // field at particle I due multipoles at particle J

    Vec3 qDotDelta;
    qDotDelta[0]                            = deltaR[0]*particleJ.quadrupole[QXX] + deltaR[1]*particleJ.quadrupole[QXY] + deltaR[2]*particleJ.quadrupole[QXZ];
    qDotDelta[1]                            = deltaR[0]*particleJ.quadrupole[QXY] + deltaR[1]*particleJ.quadrupole[QYY] + deltaR[2]*particleJ.quadrupole[QYZ];
    qDotDelta[2]                            = deltaR[0]*particleJ.quadrupole[QXZ] + deltaR[1]*particleJ.quadrupole[QYZ] + deltaR[2]*particleJ.quadrupole[QZZ];
    Vec3 oxxJ        = Vec3(particleJ.octopole[QXXX], particleJ.octopole[QXXY], particleJ.octopole[QXXZ]);
    Vec3 oxyJ        = Vec3(particleJ.octopole[QXXY], particleJ.octopole[QXYY], particleJ.octopole[QXYZ]);
    Vec3 oxzJ        = Vec3(particleJ.octopole[QXXZ], particleJ.octopole[QXYZ], particleJ.octopole[QXZZ]);
    Vec3 oyyJ        = Vec3(particleJ.octopole[QXYY], particleJ.octopole[QYYY], particleJ.octopole[QYYZ]);
    Vec3 oyzJ        = Vec3(particleJ.octopole[QXYZ], particleJ.octopole[QYYZ], particleJ.octopole[QYZZ]);
    Vec3 ozzJ        = Vec3(particleJ.octopole[QXZZ], particleJ.octopole[QYZZ], particleJ.octopole[QZZZ]);
    Vec3 oxJ         = Vec3(oxxJ.dot(deltaR), oxyJ.dot(deltaR), oxzJ.dot(deltaR));
    Vec3 oyJ         = Vec3(oxyJ.dot(deltaR), oyyJ.dot(deltaR), oyzJ.dot(deltaR));
    Vec3 ozJ         = Vec3(oxzJ.dot(deltaR), oyzJ.dot(deltaR), ozzJ.dot(deltaR));
    Vec3 oJ          = Vec3(oxJ.dot(deltaR), oyJ.dot(deltaR), ozJ.dot(deltaR));

    double dipoleDelta                      = particleJ.dipole.dot(deltaR);
    double qdpoleDelta                      = qDotDelta.dot(deltaR);
    double factor                           = rr3*particleJ.charge - rr5*dipoleDelta + rr7*qdpoleDelta - rr9*oJ.dot(deltaR);

    Vec3 field                              = deltaR*factor + particleJ.dipole*rr3 - qDotDelta*rr5_2 + oJ*rr7_3;
    unsigned int particleIndex              = particleI.particleIndex;
    _fixedMultipoleField[particleIndex]    -= field*dScale;

    // field at particle J due multipoles at particle I

    qDotDelta[0]                              = deltaR[0]*particleI.quadrupole[QXX] + deltaR[1]*particleI.quadrupole[QXY] + deltaR[2]*particleI.quadrupole[QXZ];
    qDotDelta[1]                              = deltaR[0]*particleI.quadrupole[QXY] + deltaR[1]*particleI.quadrupole[QYY] + deltaR[2]*particleI.quadrupole[QYZ];
    qDotDelta[2]                              = deltaR[0]*particleI.quadrupole[QXZ] + deltaR[1]*particleI.quadrupole[QYZ] + deltaR[2]*particleI.quadrupole[QZZ];
    Vec3 oxxI        = Vec3(particleI.octopole[QXXX], particleI.octopole[QXXY], particleI.octopole[QXXZ]);
    Vec3 oxyI        = Vec3(particleI.octopole[QXXY], particleI.octopole[QXYY], particleI.octopole[QXYZ]);
    Vec3 oxzI        = Vec3(particleI.octopole[QXXZ], particleI.octopole[QXYZ], particleI.octopole[QXZZ]);
    Vec3 oyyI        = Vec3(particleI.octopole[QXYY], particleI.octopole[QYYY], particleI.octopole[QYYZ]);
    Vec3 oyzI        = Vec3(particleI.octopole[QXYZ], particleI.octopole[QYYZ], particleI.octopole[QYZZ]);
    Vec3 ozzI        = Vec3(particleI.octopole[QXZZ], particleI.octopole[QYZZ], particleI.octopole[QZZZ]);
    Vec3 oxI         = Vec3(oxxI.dot(deltaR), oxyI.dot(deltaR), oxzI.dot(deltaR));
    Vec3 oyI         = Vec3(oxyI.dot(deltaR), oyyI.dot(deltaR), oyzI.dot(deltaR));
    Vec3 ozI         = Vec3(oxzI.dot(deltaR), oyzI.dot(deltaR), ozzI.dot(deltaR));
    Vec3 oI          = Vec3(oxI.dot(deltaR), oyI.dot(deltaR), ozI.dot(deltaR));

    dipoleDelta                               = particleI.dipole.dot(deltaR);
    qdpoleDelta                               = qDotDelta.dot(deltaR);
    factor                                    = rr3*particleI.charge + rr5*dipoleDelta + rr7*qdpoleDelta + rr9*oI.dot(deltaR);

    field                                     = deltaR*factor - particleI.dipole*rr3 - qDotDelta*rr5_2 - oI*rr7_3;
    particleIndex                             = particleJ.particleIndex;

    _fixedMultipoleField[particleIndex]      += field*dScale;
}

void MPIDReferenceForce::calculateFixedMultipoleField(const vector<MultipoleParticleData>& particleData)
{

    // calculate fixed multipole fields

    // loop includes diagonal term ii == jj for GK ixn; other calculateFixedMultipoleFieldPairIxn() methods
    // skip calculations for this case

    for (unsigned int ii = 0; ii < _numParticles; ii++) {
        for (unsigned int jj = ii; jj < _numParticles; jj++) {

            // if site jj is less than max covalent scaling index then get/apply scaling constants
            // otherwise add unmodified field and fieldPolar to particle fields

            double dScale, pScale;
            if (jj <= _maxScaleIndex[ii]) {
                getDScaleAndPScale(ii, jj, dScale, pScale);
            } else {
                dScale = pScale = 1.0;
            }
            calculateFixedMultipoleFieldPairIxn(particleData[ii], particleData[jj], dScale, pScale);
        }
    }
}

void MPIDReferenceForce::initializeInducedDipoles(vector<UpdateInducedDipoleFieldStruct>& updateInducedDipoleFields)
{

    // initialize inducedDipoles

    _inducedDipole.resize(_numParticles);

    for (unsigned int ii = 0; ii < _numParticles; ii++) {
        _inducedDipole[ii]       = _fixedMultipoleField[ii];
    }
}

void MPIDReferenceForce::calculateInducedDipolePairIxn(unsigned int particleI,
                                                                  unsigned int particleJ,
                                                                  double rr3,
                                                                  double rr5,
                                                                  const Vec3& deltaR,
                                                                  const vector<Vec3>& inducedDipole,
                                                                  vector<Vec3>& field) const
{

    double dDotDelta            = rr5*(inducedDipole[particleJ].dot(deltaR));
    field[particleI]           += inducedDipole[particleJ]*rr3 + deltaR*dDotDelta;
    dDotDelta                   = rr5*(inducedDipole[particleI].dot(deltaR));
    field[particleJ]           += inducedDipole[particleI]*rr3 + deltaR*dDotDelta;
}

void MPIDReferenceForce::calculateInducedDipolePairIxns(const MultipoleParticleData& particleI,
                                                                   const MultipoleParticleData& particleJ,
                                                                   vector<UpdateInducedDipoleFieldStruct>& updateInducedDipoleFields)
{

   if (particleI.particleIndex == particleJ.particleIndex)
       return;

    Vec3 deltaR   = particleJ.position - particleI.position;
    double r      = sqrt(deltaR.dot(deltaR));
    vector<double> rrI(2);
    // If we're using the extrapolation algorithm, we need to compute the field gradient, so ask for one more rrI value.
    if (getPolarizationType() == MPIDReferenceForce::Extrapolated)
        rrI.push_back(0.0);

    double pscale = getMultipoleScaleFactor(particleI.particleIndex, particleJ.particleIndex, P_SCALE);
    getAndScaleInverseRs(particleI.dampingFactor, particleJ.dampingFactor, pscale,
                          particleI.thole, particleJ.thole, r, rrI);

    double rr3       = -rrI[0];
    double rr5       =  rrI[1];

    for (auto& field : updateInducedDipoleFields) {
        calculateInducedDipolePairIxn(particleI.particleIndex, particleJ.particleIndex, rr3, rr5, deltaR,
                                       *field.inducedDipoles, field.inducedDipoleField);
        if (getPolarizationType() == MPIDReferenceForce::Extrapolated) {
            // Compute and store the field gradient for later use.
            double dx = deltaR[0];
            double dy = deltaR[1];
            double dz = deltaR[2];

            OpenMM::Vec3 &dipolesI = (*field.inducedDipoles)[particleI.particleIndex];
            double xDipole = dipolesI[0];
            double yDipole = dipolesI[1];
            double zDipole = dipolesI[2];
            double muDotR = xDipole*dx + yDipole*dy + zDipole*dz;
            double Exx = muDotR*dx*dx*rrI[2] - (2.0*xDipole*dx + muDotR)*rrI[1];
            double Eyy = muDotR*dy*dy*rrI[2] - (2.0*yDipole*dy + muDotR)*rrI[1];
            double Ezz = muDotR*dz*dz*rrI[2] - (2.0*zDipole*dz + muDotR)*rrI[1];
            double Exy = muDotR*dx*dy*rrI[2] - (xDipole*dy + yDipole*dx)*rrI[1];
            double Exz = muDotR*dx*dz*rrI[2] - (xDipole*dz + zDipole*dx)*rrI[1];
            double Eyz = muDotR*dy*dz*rrI[2] - (yDipole*dz + zDipole*dy)*rrI[1];

            field.inducedDipoleFieldGradient[particleJ.particleIndex][0] -= Exx;
            field.inducedDipoleFieldGradient[particleJ.particleIndex][1] -= Eyy;
            field.inducedDipoleFieldGradient[particleJ.particleIndex][2] -= Ezz;
            field.inducedDipoleFieldGradient[particleJ.particleIndex][3] -= Exy;
            field.inducedDipoleFieldGradient[particleJ.particleIndex][4] -= Exz;
            field.inducedDipoleFieldGradient[particleJ.particleIndex][5] -= Eyz;

            OpenMM::Vec3 &dipolesJ = (*field.inducedDipoles)[particleJ.particleIndex];
            xDipole = dipolesJ[0];
            yDipole = dipolesJ[1];
            zDipole = dipolesJ[2];
            muDotR = xDipole*dx + yDipole*dy + zDipole*dz;
            Exx = muDotR*dx*dx*rrI[2] - (2.0*xDipole*dx + muDotR)*rrI[1];
            Eyy = muDotR*dy*dy*rrI[2] - (2.0*yDipole*dy + muDotR)*rrI[1];
            Ezz = muDotR*dz*dz*rrI[2] - (2.0*zDipole*dz + muDotR)*rrI[1];
            Exy = muDotR*dx*dy*rrI[2] - (xDipole*dy + yDipole*dx)*rrI[1];
            Exz = muDotR*dx*dz*rrI[2] - (xDipole*dz + zDipole*dx)*rrI[1];
            Eyz = muDotR*dy*dz*rrI[2] - (yDipole*dz + zDipole*dy)*rrI[1];

            field.inducedDipoleFieldGradient[particleI.particleIndex][0] += Exx;
            field.inducedDipoleFieldGradient[particleI.particleIndex][1] += Eyy;
            field.inducedDipoleFieldGradient[particleI.particleIndex][2] += Ezz;
            field.inducedDipoleFieldGradient[particleI.particleIndex][3] += Exy;
            field.inducedDipoleFieldGradient[particleI.particleIndex][4] += Exz;
            field.inducedDipoleFieldGradient[particleI.particleIndex][5] += Eyz;
        }
    }
}

void MPIDReferenceForce::calculateInducedDipoleFields(const vector<MultipoleParticleData>& particleData, vector<UpdateInducedDipoleFieldStruct>& updateInducedDipoleFields) {
    // Initialize the fields to zero.

    Vec3 zeroVec(0.0, 0.0, 0.0);
    for (auto& field : updateInducedDipoleFields)
        std::fill(field.inducedDipoleField.begin(), field.inducedDipoleField.end(), zeroVec);

    // Add fields from all induced dipoles.

    for (unsigned int ii = 0; ii < particleData.size(); ii++)
        for (unsigned int jj = ii; jj < particleData.size(); jj++)
            calculateInducedDipolePairIxns(particleData[ii], particleData[jj], updateInducedDipoleFields);
}

double MPIDReferenceForce::updateInducedDipoleFields(const vector<MultipoleParticleData>& particleData,
                                                                vector<UpdateInducedDipoleFieldStruct>& updateInducedDipoleFields)
{
    // Calculate the fields coming from induced dipoles.

    calculateInducedDipoleFields(particleData, updateInducedDipoleFields);

    // Update the induced dipoles and calculate the convergence factor, maxEpsilon

    double maxEpsilon = 0.0;
    for (auto& field : updateInducedDipoleFields) {
        double epsilon = updateInducedDipole(particleData,
                                             *field.fixedMultipoleField,
                                             field.inducedDipoleField,
                                             *field.inducedDipoles);

        maxEpsilon = epsilon > maxEpsilon ? epsilon : maxEpsilon;
    }

    return maxEpsilon;
}

double MPIDReferenceForce::updateInducedDipole(const vector<MultipoleParticleData>& particleData,
                                                          const vector<Vec3>& fixedMultipoleField,
                                                          const vector<Vec3>& inducedDipoleField,
                                                          vector<Vec3>& inducedDipole)
{

    double epsilon = 0.0;
    for (unsigned int ii = 0; ii < particleData.size(); ii++) {
        Vec3 vx = Vec3(particleData[ii].labPolarization[QXX], particleData[ii].labPolarization[QXY], particleData[ii].labPolarization[QXZ]);
        Vec3 vy = Vec3(particleData[ii].labPolarization[QXY], particleData[ii].labPolarization[QYY], particleData[ii].labPolarization[QYZ]);
        Vec3 vz = Vec3(particleData[ii].labPolarization[QXZ], particleData[ii].labPolarization[QYZ], particleData[ii].labPolarization[QZZ]);
        Vec3 oldValue               = inducedDipole[ii];
        Vec3 newValue               = fixedMultipoleField[ii]
                                    + Vec3(vx.dot(inducedDipoleField[ii]), vy.dot(inducedDipoleField[ii]), vz.dot(inducedDipoleField[ii]));
        Vec3 delta                  = newValue - oldValue;
        inducedDipole[ii]           = oldValue + delta*_polarSOR;
        epsilon                    += delta.dot(delta);
    }
    return epsilon;
}

void MPIDReferenceForce::convergeInduceDipolesBySOR(const vector<MultipoleParticleData>& particleData,
                                                               vector<UpdateInducedDipoleFieldStruct>& updateInducedDipoleField)
{

    bool done = false;
    setMutualInducedDipoleConverged(false);
    int iteration = 0;
    double currentEpsilon = 1.0e+50;

    // loop until (1) induced dipoles are converged or
    //            (2) iterations == max iterations or
    //            (3) convergence factor (spsilon) increases

    while (!done) {

        double epsilon = updateInducedDipoleFields(particleData, updateInducedDipoleField);
               epsilon = _polarSOR*_debye*sqrt(epsilon/_numParticles);

        if (epsilon < getMutualInducedDipoleTargetEpsilon()) {
            setMutualInducedDipoleConverged(true);
            done = true;
        } else if (currentEpsilon < epsilon || iteration >= getMaximumMutualInducedDipoleIterations()) {
            done = true;
        }

        currentEpsilon = epsilon;
        iteration++;
    }
    setMutualInducedDipoleEpsilon(currentEpsilon);
    setMutualInducedDipoleIterations(iteration);
}


void MPIDReferenceForce::convergeInduceDipolesByExtrapolation(const vector<MultipoleParticleData>& particleData, vector<UpdateInducedDipoleFieldStruct>& updateInducedDipoleField) {
    // Start by storing the direct dipoles as PT0

    int numFields = updateInducedDipoleField.size();
    for (int i = 0; i < numFields; i++) {
        UpdateInducedDipoleFieldStruct& field = updateInducedDipoleField[i];
        field.extrapolatedDipoles->resize(_maxPTOrder);
        (*field.extrapolatedDipoles)[0].resize(_numParticles);
        for (int atom = 0; atom < _numParticles; ++atom)
            (*field.extrapolatedDipoles)[0][atom] = (*field.inducedDipoles)[atom];
        field.inducedDipoleFieldGradient.resize(_numParticles);
    }

    // Recursively apply alpha.Tau to the µ_(n) components to generate µ_(n+1), and store the result

    vector<double> zeros(6, 0.0);
    for (int order = 1; order < _maxPTOrder; ++order) {
        for (int i = 0; i < numFields; i++)
            std::fill(updateInducedDipoleField[i].inducedDipoleFieldGradient.begin(), updateInducedDipoleField[i].inducedDipoleFieldGradient.end(), zeros);
        calculateInducedDipoleFields(particleData, updateInducedDipoleField);
        for (int i = 0; i < numFields; i++) {
            UpdateInducedDipoleFieldStruct& field = updateInducedDipoleField[i];
            (*field.extrapolatedDipoles)[order].resize(_numParticles);
            for (int atom = 0; atom < _numParticles; ++atom) {
                Vec3 vx = Vec3(particleData[atom].labPolarization[QXX], particleData[atom].labPolarization[QXY], particleData[atom].labPolarization[QXZ]);
                Vec3 vy = Vec3(particleData[atom].labPolarization[QXY], particleData[atom].labPolarization[QYY], particleData[atom].labPolarization[QYZ]);
                Vec3 vz = Vec3(particleData[atom].labPolarization[QXZ], particleData[atom].labPolarization[QYZ], particleData[atom].labPolarization[QZZ]);
                Vec3 inddip(vx.dot(field.inducedDipoleField[atom]), vy.dot(field.inducedDipoleField[atom]), vz.dot(field.inducedDipoleField[atom]));
                (*field.inducedDipoles)[atom] = inddip;
                (*field.extrapolatedDipoles)[order][atom] = (*field.inducedDipoles)[atom];
            }
            vector<double> dipfield(3*_numParticles, 0.0);
            for (int atom = 0; atom < _numParticles; ++atom)
                for (int component = 0; component < 3; ++component)
                    dipfield[3*atom + component] = field.inducedDipoleField[atom][component];
            field.extrapolatedDipoleField->push_back(dipfield);
            vector<double> fieldGrad(6*_numParticles, 0.0);
            for (int atom = 0; atom < _numParticles; ++atom)
                for (int component = 0; component < 6; ++component)
                    fieldGrad[6*atom + component] = field.inducedDipoleFieldGradient[atom][component];
            field.extrapolatedDipoleFieldGradient->push_back(fieldGrad);
        }
    }

    // Take a linear combination of the µ_(n) components to form the total dipole
    
    for (int i = 0; i < numFields; i++) {
        UpdateInducedDipoleFieldStruct& field = updateInducedDipoleField[i];
        *field.inducedDipoles = vector<Vec3>(_numParticles, Vec3());
        for (int order = 0; order < _maxPTOrder; ++order)
            for (int atom = 0; atom < _numParticles; ++atom)
                (*field.inducedDipoles)[atom] += (*field.extrapolatedDipoles)[order][atom] * _extPartCoefficients[order];
    }
    calculateInducedDipoleFields(particleData, updateInducedDipoleField);
    setMutualInducedDipoleConverged(true);
}

void MPIDReferenceForce::convergeInduceDipolesByDIIS(const vector<MultipoleParticleData>& particleData, vector<UpdateInducedDipoleFieldStruct>& updateInducedDipoleField) {
    int numFields = updateInducedDipoleField.size();
    vector<vector<vector<Vec3> > > prevDipoles(numFields);
    vector<vector<Vec3> > prevErrors;
    setMutualInducedDipoleConverged(false);
    int maxPrevious = 20;
    for (int iteration = 0; ; iteration++) {
        // Compute the field from the induced dipoles.

        calculateInducedDipoleFields(particleData, updateInducedDipoleField);

        // Record the current dipoles and the errors in them.

        double maxEpsilon = 0;
        prevErrors.push_back(vector<Vec3>());
        prevErrors.back().resize(_numParticles);
        for (int k = 0; k < numFields; k++) {
            UpdateInducedDipoleFieldStruct& field = updateInducedDipoleField[k];
            prevDipoles[k].push_back(vector<Vec3>());
            prevDipoles[k].back().resize(_numParticles);
            double epsilon = 0;
            for (int i = 0; i < _numParticles; i++) {
                prevDipoles[k].back()[i] = (*field.inducedDipoles)[i];
                Vec3 vx = Vec3(particleData[i].labPolarization[QXX], particleData[i].labPolarization[QXY], particleData[i].labPolarization[QXZ]);
                Vec3 vy = Vec3(particleData[i].labPolarization[QXY], particleData[i].labPolarization[QYY], particleData[i].labPolarization[QYZ]);
                Vec3 vz = Vec3(particleData[i].labPolarization[QXZ], particleData[i].labPolarization[QYZ], particleData[i].labPolarization[QZZ]);
                Vec3 newDipole = (*field.fixedMultipoleField)[i] +
                                Vec3(vx.dot(field.inducedDipoleField[i]), vy.dot(field.inducedDipoleField[i]), vz.dot(field.inducedDipoleField[i]));
                Vec3 error = newDipole-(*field.inducedDipoles)[i];
                prevDipoles[k].back()[i] = newDipole;
                if (k == 0)
                    prevErrors.back()[i] = error;
                epsilon += error.dot(error);
            }
            if (epsilon > maxEpsilon)
                maxEpsilon = epsilon;
        }
        maxEpsilon = _debye*sqrt(maxEpsilon/_numParticles);

        // Decide whether to stop or continue iterating.

        if (maxEpsilon < getMutualInducedDipoleTargetEpsilon())
            setMutualInducedDipoleConverged(true);
        if (maxEpsilon < getMutualInducedDipoleTargetEpsilon() || iteration == getMaximumMutualInducedDipoleIterations()) {
            setMutualInducedDipoleEpsilon(maxEpsilon);
            setMutualInducedDipoleIterations(iteration);
            return;
        }

        // Select the new dipoles.

        if (prevErrors.size() > maxPrevious) {
            prevErrors.erase(prevErrors.begin());
            for (int k = 0; k < numFields; k++)
                prevDipoles[k].erase(prevDipoles[k].begin());
        }
        int numPrevious = prevErrors.size();
        vector<double> coefficients(numPrevious);
        computeDIISCoefficients(prevErrors, coefficients);
        for (int k = 0; k < numFields; k++) {
            UpdateInducedDipoleFieldStruct& field = updateInducedDipoleField[k];
            for (int i = 0; i < _numParticles; i++) {
                Vec3 dipole(0.0, 0.0, 0.0);
                for (int j = 0; j < numPrevious; j++)
                    dipole += prevDipoles[k][j][i]*coefficients[j];
                (*field.inducedDipoles)[i] = dipole;
            }
        }
    }

}

void MPIDReferenceForce::computeDIISCoefficients(const vector<vector<Vec3> >& prevErrors, vector<double>& coefficients) const {
    int steps = coefficients.size();
    if (steps == 1) {
        coefficients[0] = 1;
        return;
    }

    // Create the DIIS matrix.

    int rank = steps+1;
    Array2D<double> b(rank, rank);
    b[0][0] = 0;
    for (int i = 0; i < steps; i++)
        b[i+1][0] = b[0][i+1] = -1;
    for (int i = 0; i < steps; i++)
        for (int j = i; j < steps; j++) {
            double sum = 0;
            for (int k = 0; k < _numParticles; k++)
                sum += prevErrors[i][k].dot(prevErrors[j][k]);
            b[i+1][j+1] = b[j+1][i+1] = sum;
        }

    // Solve using SVD.  Since the right hand side is (-1, 0, 0, 0, ...), this is simpler than the general case.

    JAMA::SVD<double> svd(b);
    Array2D<double> u, v;
    svd.getU(u);
    svd.getV(v);
    Array1D<double> s;
    svd.getSingularValues(s);
    int effectiveRank = svd.rank();
    for (int i = 1; i < rank; i++) {
        double d = 0;
        for (int j = 0; j < effectiveRank; j++)
            d -= u[0][j]*v[i][j]/s[j];
        coefficients[i-1] = d;
    }
}

void MPIDReferenceForce::calculateInducedDipoles(const vector<MultipoleParticleData>& particleData)
{

    // calculate fixed electric fields

    zeroFixedMultipoleFields();
    calculateFixedMultipoleField(particleData);

    // initialize inducedDipoles
    // if polarization type is 'Direct', then return after initializing; otherwise
    // converge induced dipoles.

    for (unsigned int ii = 0; ii < _numParticles; ii++) {
        Vec3 vx = Vec3(particleData[ii].labPolarization[QXX], particleData[ii].labPolarization[QXY], particleData[ii].labPolarization[QXZ]);
        Vec3 vy = Vec3(particleData[ii].labPolarization[QXY], particleData[ii].labPolarization[QYY], particleData[ii].labPolarization[QYZ]);
        Vec3 vz = Vec3(particleData[ii].labPolarization[QXZ], particleData[ii].labPolarization[QYZ], particleData[ii].labPolarization[QZZ]);
        _fixedMultipoleField[ii]  =  Vec3(vx.dot(_fixedMultipoleField[ii]), vy.dot(_fixedMultipoleField[ii]), vz.dot(_fixedMultipoleField[ii]));
    }

    _inducedDipole.resize(_numParticles);
    vector<UpdateInducedDipoleFieldStruct> updateInducedDipoleField;
    updateInducedDipoleField.push_back(UpdateInducedDipoleFieldStruct(_fixedMultipoleField, _inducedDipole, _ptDipoleD, _ptDipoleFieldD, _ptDipoleFieldGradientD));

    initializeInducedDipoles(updateInducedDipoleField);

    if (getPolarizationType() == MPIDReferenceForce::Direct) {
        setMutualInducedDipoleConverged(true);
        return;
    }

    // UpdateInducedDipoleFieldStruct contains induced dipole, fixed multipole fields and fields
    // due to other induced dipoles at each site
    if (getPolarizationType() == MPIDReferenceForce::Mutual)
        convergeInduceDipolesByDIIS(particleData, updateInducedDipoleField);
    else if (getPolarizationType() == MPIDReferenceForce::Extrapolated)
        convergeInduceDipolesByExtrapolation(particleData, updateInducedDipoleField);
}

double MPIDReferenceForce::calculateElectrostaticPairIxn(const MultipoleParticleData& particleI,
                                                                        const MultipoleParticleData& particleJ,
                                                                        const vector<double>& scalingFactors,
                                                                        vector<Vec3>& forces,
                                                                        vector<Vec3>& torque) const
{
    unsigned int iIndex = particleI.particleIndex;
    unsigned int jIndex = particleJ.particleIndex;

    Vec3 deltaR = particleJ.position - particleI.position;
    double r2 = deltaR.dot(deltaR);
    double r = sqrt(r2);

    // Start by constructing rotation matrices to put dipoles and
    // quadrupoles into the QI frame, from the lab frame.
    double qiRotationMatrix1[3][3];
    formQIRotationMatrix(particleI.position, particleJ.position, deltaR, r, qiRotationMatrix1);
    double qiRotationMatrix2[5][5];
    buildSphericalQuadrupoleRotationMatrix(qiRotationMatrix1, qiRotationMatrix2);
    double qiRotationMatrix3[7][7];
    buildSphericalOctopoleRotationMatrix(qiRotationMatrix1, qiRotationMatrix2, qiRotationMatrix3);
    // The force rotation matrix rotates the QI forces into the lab
    // frame, and makes sure the result is in {x,y,z} ordering. Its
    // transpose is used to rotate the induced dipoles to the QI frame.
    double forceRotationMatrix[3][3];
    forceRotationMatrix[0][0] = qiRotationMatrix1[1][1];
    forceRotationMatrix[0][1] = qiRotationMatrix1[2][1];
    forceRotationMatrix[0][2] = qiRotationMatrix1[0][1];
    forceRotationMatrix[1][0] = qiRotationMatrix1[1][2];
    forceRotationMatrix[1][1] = qiRotationMatrix1[2][2];
    forceRotationMatrix[1][2] = qiRotationMatrix1[0][2];
    forceRotationMatrix[2][0] = qiRotationMatrix1[1][0];
    forceRotationMatrix[2][1] = qiRotationMatrix1[2][0];
    forceRotationMatrix[2][2] = qiRotationMatrix1[0][0];
    // For efficiency, we go ahead and cache that transposed version
    // now, because we need to do 4 rotations in total (I,J, and p,d).
    // We also fold in the factor of 0.5 needed to average the p and d
    // components.
    double inducedDipoleRotationMatrix[3][3];
    inducedDipoleRotationMatrix[0][0] = 0.5*qiRotationMatrix1[0][1];
    inducedDipoleRotationMatrix[0][1] = 0.5*qiRotationMatrix1[0][2];
    inducedDipoleRotationMatrix[0][2] = 0.5*qiRotationMatrix1[0][0];
    inducedDipoleRotationMatrix[1][0] = 0.5*qiRotationMatrix1[1][1];
    inducedDipoleRotationMatrix[1][1] = 0.5*qiRotationMatrix1[1][2];
    inducedDipoleRotationMatrix[1][2] = 0.5*qiRotationMatrix1[1][0];
    inducedDipoleRotationMatrix[2][0] = 0.5*qiRotationMatrix1[2][1];
    inducedDipoleRotationMatrix[2][1] = 0.5*qiRotationMatrix1[2][2];
    inducedDipoleRotationMatrix[2][2] = 0.5*qiRotationMatrix1[2][0];

    // Rotate the induced dipoles to the QI frame.
    double qiUindI[3], qiUindJ[3];
    for (int ii = 0; ii < 3; ii++) {
        double valID = 0.0;
        double valJD = 0.0;
        for (int jj = 0; jj < 3; jj++) {
            valID += inducedDipoleRotationMatrix[ii][jj] * _inducedDipole[iIndex][jj];
            valJD += inducedDipoleRotationMatrix[ii][jj] * _inducedDipole[jIndex][jj];
        }
        qiUindI[ii] = valID;
        qiUindJ[ii] = valJD;
    }

    // The Qtilde intermediates (QI frame multipoles) for atoms I and J
    double qiQI[16], qiQJ[16];
    // Rotate the permanent multipoles to the QI frame.
    qiQI[0] = particleI.charge;
    qiQJ[0] = particleJ.charge;
    for (int ii = 0; ii < 3; ii++) {
        double valI = 0.0;
        double valJ = 0.0;
        for (int jj = 0; jj < 3; jj++) {
            valI += qiRotationMatrix1[ii][jj] * particleI.sphericalDipole[jj];
            valJ += qiRotationMatrix1[ii][jj] * particleJ.sphericalDipole[jj];
        }
        qiQI[ii+1] = valI;
        qiQJ[ii+1] = valJ;
    }

    for (int ii = 0; ii < 5; ii++) {
        double valI = 0.0;
        double valJ = 0.0;
        for (int jj = 0; jj < 5; jj++) {
            valI += qiRotationMatrix2[ii][jj] * particleI.sphericalQuadrupole[jj];
            valJ += qiRotationMatrix2[ii][jj] * particleJ.sphericalQuadrupole[jj];
        }
        qiQI[ii+4] = valI;
        qiQJ[ii+4] = valJ;
    }

    for (int ii = 0; ii < 7; ii++) {
        double valI = 0.0;
        double valJ = 0.0;
        for (int jj = 0; jj < 7; jj++) {
            valI += qiRotationMatrix3[ii][jj] * particleI.sphericalOctopole[jj];
            valJ += qiRotationMatrix3[ii][jj] * particleJ.sphericalOctopole[jj];
        }
        qiQI[ii+9] = valI;
        qiQJ[ii+9] = valJ;
    }

    // The Qtilde{x,y,z} torque intermediates for atoms I and J, which are used to obtain the torques on the permanent moments.
    // 0    1    2    3    4    5    6    7    8    9   10   11   12   13   14   15
    // q    10  11c  11s   20  21c  21s  22c  22s  30   31c  31s  32c  32s  33c  33s
    double qiQIX[16] = {0.0, qiQI[3], 0.0, -qiQI[1], sqrtThree*qiQI[6], qiQI[8], -sqrtThree*qiQI[4] - qiQI[7], qiQI[6], -qiQI[5],
                        sqrtSix*qiQI[11], sqrtFiveHalves*qiQI[13], -sqrtSix*qiQI[9]-sqrtFiveHalves*qiQI[12],sqrtFiveHalves*qiQI[11]+sqrtThreeHalves*qiQI[15],
                        -sqrtFiveHalves*qiQI[10]-sqrtThreeHalves*qiQI[14], sqrtThreeHalves*qiQI[13], -sqrtThreeHalves*qiQI[12]};
    double qiQIY[16] = {0.0, -qiQI[2], qiQI[1], 0.0, -sqrtThree*qiQI[5], sqrtThree*qiQI[4] - qiQI[7], -qiQI[8], qiQI[5], qiQI[6],
                        -sqrtSix*qiQI[10], sqrtSix*qiQI[9]-sqrtFiveHalves*qiQI[12], -sqrtFiveHalves*qiQI[13], sqrtFiveHalves*qiQI[10]-sqrtThreeHalves*qiQI[14],
                        sqrtFiveHalves*qiQI[11]-sqrtThreeHalves*qiQI[15], sqrtThreeHalves*qiQI[12], sqrtThreeHalves*qiQI[13]};
    double qiQIZ[16] = {0.0, 0.0, -qiQI[3], qiQI[2], 0.0, -qiQI[6], qiQI[5], -2.0*qiQI[8], 2.0*qiQI[7],
                        0.0, -qiQI[11], qiQI[10], -2.0*qiQI[13], 2.0*qiQI[12], -3.0*qiQI[15], 3.0*qiQI[14]};
    double qiQJX[16] = {0.0, qiQJ[3], 0.0, -qiQJ[1], sqrtThree*qiQJ[6], qiQJ[8], -sqrtThree*qiQJ[4] - qiQJ[7], qiQJ[6], -qiQJ[5],
                        sqrtSix*qiQJ[11], sqrtFiveHalves*qiQJ[13], -sqrtSix*qiQJ[9]-sqrtFiveHalves*qiQJ[12],sqrtFiveHalves*qiQJ[11]+sqrtThreeHalves*qiQJ[15],
                        -sqrtFiveHalves*qiQJ[10]-sqrtThreeHalves*qiQJ[14], sqrtThreeHalves*qiQJ[13], -sqrtThreeHalves*qiQJ[12]};
    double qiQJY[16] = {0.0, -qiQJ[2], qiQJ[1], 0.0, -sqrtThree*qiQJ[5], sqrtThree*qiQJ[4] - qiQJ[7], -qiQJ[8], qiQJ[5], qiQJ[6],
                        -sqrtSix*qiQJ[10], sqrtSix*qiQJ[9]-sqrtFiveHalves*qiQJ[12], -sqrtFiveHalves*qiQJ[13], sqrtFiveHalves*qiQJ[10]-sqrtThreeHalves*qiQJ[14],
                        sqrtFiveHalves*qiQJ[11]-sqrtThreeHalves*qiQJ[15], sqrtThreeHalves*qiQJ[12], sqrtThreeHalves*qiQJ[13]};
    double qiQJZ[16] = {0.0, 0.0, -qiQJ[3], qiQJ[2], 0.0, -qiQJ[6], qiQJ[5], -2.0*qiQJ[8], 2.0*qiQJ[7],
                        0.0, -qiQJ[11], qiQJ[10], -2.0*qiQJ[13], 2.0*qiQJ[12], -3.0*qiQJ[15], 3.0*qiQJ[14]};


    // The field derivatives at I due to permanent and induced moments on J, and vice-versa.
    // Also, their derivatives w.r.t. R, which are needed for force calculations
    double Vij[16], Vji[16], VjiR[16], VijR[16];
    // The field derivatives at I due to only permanent moments on J, and vice-versa.
    double Vijd[3], Vjid[3];
    double rInvVec[9];

    double prefac = (_electric/_dielectric);
    double rInv = 1.0 / r;

    // The rInvVec array is defined such that the ith element is R^-i, with the
    // dieleectric constant folded in, to avoid conversions later.
    rInvVec[1] = prefac * rInv;
    for (int i = 2; i < 9; ++i)
        rInvVec[i] = rInvVec[i-1] * rInv;

    double mScale = scalingFactors[M_SCALE];
    double pScale = scalingFactors[P_SCALE];
    double dScale = pScale;
    double uScale = 1.0;

    double dmp = particleI.dampingFactor*particleJ.dampingFactor;
    double a = pScale == 0.0 ? particleI.thole + particleJ.thole : _defaultTholeWidth;
    double u = std::abs(dmp) > 1.0E-5 ? r/dmp : 1E10;
    double au = a*u;
    double expau = au < 50.0 ? exp(-au) : 0.0;
    double au2 = au*au;
    double au3 = au2*au;
    double au4 = au3*au;
    double au5 = au4*au;
    double au6 = au5*au;
    // Thole damping factors for energies
    double thole_c   = 1.0 - expau*(1.0 + au + 0.5*au2);
    double thole_d0  = 1.0 - expau*(1.0 + au + 0.5*au2 + au3/4.0);
    double thole_d1  = 1.0 - expau*(1.0 + au + 0.5*au2);
    double thole_q0  = 1.0 - expau*(1.0 + au + 0.5*au2 + au3/6.0 + au4/18.0);
    double thole_q1  = 1.0 - expau*(1.0 + au + 0.5*au2 + au3/6.0);
    double thole_o0  = 1.0 - expau*(1.0 + au + 0.5*au2 + au3/6.0 + au4/24.0 + au5/120.0);
    double thole_o1  = 1.0 - expau*(1.0 + au + 0.5*au2 + au3/6.0 + au4/30.0);
    // Thole damping factors for derivatives
    double dthole_c  = 1.0 - expau*(1.0 + au + 0.5*au2 + au3/4.0);
    double dthole_d0 = 1.0 - expau*(1.0 + au + 0.5*au2 + au3/6.0 + au4/12.0);
    double dthole_d1 = 1.0 - expau*(1.0 + au + 0.5*au2 + au3/6.0);
    double dthole_q0 = 1.0 - expau*(1.0 + au + 0.5*au2 + au3/6.0 + au4/24.0 + au5/72.0);
    double dthole_q1 = 1.0 - expau*(1.0 + au + 0.5*au2 + au3/6.0 + au4/24.0);
    double dthole_o0 = 1.0 - expau*(1.0 + au + 0.5*au2 + au3/6.0 + au4/24.0 + au5/120.0 + au6/600.0);
    double dthole_o1 = 1.0 - expau*(1.0 + au + 0.5*au2 + au3/6.0 + au4/25.0 + au5/150.0);
    // Now we compute the (attenuated) Coulomb operator and its derivatives, contracted with
    // permanent moments and induced dipoles.  Note that the coefficient of the permanent force
    // terms is half of the expected value; this is because we compute the interaction of I with
    // the sum of induced and permanent moments on J, as well as the interaction of J with I's
    // permanent and induced moments; doing so double counts the permanent-permanent interaction.
    double ePermCoef, dPermCoef, eUindCoef, dUindCoef;

    // C-C terms (m=0)
    ePermCoef = rInvVec[1]*mScale;
    dPermCoef = -0.5*mScale*rInvVec[2];
    Vij[0]  = ePermCoef*qiQJ[0];
    Vji[0]  = ePermCoef*qiQI[0];
    VijR[0] = dPermCoef*qiQJ[0];
    VjiR[0] = dPermCoef*qiQI[0];

    // C-D and C-Uind terms (m=0)
    ePermCoef = rInvVec[2]*mScale;
    eUindCoef = 2.0*rInvVec[2]*pScale*thole_c;
    dPermCoef = -rInvVec[3]*mScale;
    dUindCoef = -4.0*rInvVec[3]*dScale*dthole_c;
    Vij[0]  += -(ePermCoef*qiQJ[1] + eUindCoef*qiUindJ[0]);
    Vji[1]   = -(ePermCoef*qiQI[0]);
    VijR[0] += -(dPermCoef*qiQJ[1] + dUindCoef*qiUindJ[0]);
    VjiR[1]  = -(dPermCoef*qiQI[0]);
    Vjid[0]  = -(eUindCoef*qiQI[0]);
    // D-C and Uind-C terms (m=0)
    Vij[1]   = ePermCoef*qiQJ[0];
    Vji[0]  += ePermCoef*qiQI[1] + eUindCoef*qiUindI[0];
    VijR[1]  = dPermCoef*qiQJ[0];
    VjiR[0] += dPermCoef*qiQI[1] + dUindCoef*qiUindI[0];
    Vijd[0]  = eUindCoef*qiQJ[0];

    // D-D and D-Uind terms (m=0)
    ePermCoef = -2.0*rInvVec[3]*mScale;
    eUindCoef = -4.0*rInvVec[3]*dScale*thole_d0;
    dPermCoef = rInvVec[4]*3.0*mScale;
    dUindCoef = 12.0*rInvVec[4]*dScale*dthole_d0;
    Vij[1]  += ePermCoef*qiQJ[1] + eUindCoef*qiUindJ[0];
    Vji[1]  += ePermCoef*qiQI[1] + eUindCoef*qiUindI[0];
    VijR[1] += dPermCoef*qiQJ[1] + dUindCoef*qiUindJ[0];
    VjiR[1] += dPermCoef*qiQI[1] + dUindCoef*qiUindI[0];
    Vijd[0] += eUindCoef*qiQJ[1];
    Vjid[0] += eUindCoef*qiQI[1];
    // D-D and D-Uind terms (m=1)
    ePermCoef = rInvVec[3]*mScale;
    eUindCoef = 2.0*rInvVec[3]*dScale*thole_d1;
    dPermCoef = -1.5*rInvVec[4]*mScale;
    dUindCoef = -6.0*rInvVec[4]*dScale*dthole_d1;
    Vij[2]  = ePermCoef*qiQJ[2] + eUindCoef*qiUindJ[1];
    Vji[2]  = ePermCoef*qiQI[2] + eUindCoef*qiUindI[1];
    VijR[2] = dPermCoef*qiQJ[2] + dUindCoef*qiUindJ[1];
    VjiR[2] = dPermCoef*qiQI[2] + dUindCoef*qiUindI[1];
    Vij[3]  = ePermCoef*qiQJ[3] + eUindCoef*qiUindJ[2];
    Vji[3]  = ePermCoef*qiQI[3] + eUindCoef*qiUindI[2];
    VijR[3] = dPermCoef*qiQJ[3] + dUindCoef*qiUindJ[2];
    VjiR[3] = dPermCoef*qiQI[3] + dUindCoef*qiUindI[2];
    Vijd[1] = eUindCoef*qiQJ[2];
    Vjid[1] = eUindCoef*qiQI[2];
    Vijd[2] = eUindCoef*qiQJ[3];
    Vjid[2] = eUindCoef*qiQI[3];

    // C-Q terms (m=0)
    ePermCoef = mScale*rInvVec[3];
    dPermCoef = -1.5*rInvVec[4]*mScale;
    Vij[0]  += ePermCoef*qiQJ[4];
    Vji[4]   = ePermCoef*qiQI[0];
    VijR[0] += dPermCoef*qiQJ[4];
    VjiR[4]  = dPermCoef*qiQI[0];
    // Q-C terms (m=0)
    Vij[4]   = ePermCoef*qiQJ[0];
    Vji[0]  += ePermCoef*qiQI[4];
    VijR[4]  = dPermCoef*qiQJ[0];
    VjiR[0] += dPermCoef*qiQI[4];

    // D-Q and Uind-Q terms (m=0)
    ePermCoef = 3.0*rInvVec[4]*mScale;
    eUindCoef = 6.0*rInvVec[4]*dScale*thole_q0;
    dPermCoef = -6.0*rInvVec[5]*mScale;
    dUindCoef = -24.0*rInvVec[5]*dScale*dthole_q0;
    Vij[1]  += ePermCoef*qiQJ[4];
    Vji[4]  += ePermCoef*qiQI[1] + eUindCoef*qiUindI[0];
    VijR[1] += dPermCoef*qiQJ[4];
    VjiR[4] += dPermCoef*qiQI[1] + dUindCoef*qiUindI[0];
    Vijd[0] += eUindCoef*qiQJ[4];
    // Q-D and Q-Uind terms (m=0)
    Vij[4]  += -(ePermCoef*qiQJ[1] + eUindCoef*qiUindJ[0]);
    Vji[1]  += -(ePermCoef*qiQI[4]);
    VijR[4] += -(dPermCoef*qiQJ[1] + dUindCoef*qiUindJ[0]);
    VjiR[1] += -(dPermCoef*qiQI[4]);
    Vjid[0] += -(eUindCoef*qiQI[4]);

    // D-Q and Uind-Q terms (m=1)
    ePermCoef = -sqrtThree*rInvVec[4]*mScale;
    eUindCoef = -2.0*sqrtThree*rInvVec[4]*pScale*thole_q1;
    dPermCoef = 2.0*sqrtThree*rInvVec[5]*mScale;
    dUindCoef = 8.0*sqrtThree*rInvVec[5]*dScale*dthole_q1;
    Vij[2]  += ePermCoef*qiQJ[5];
    Vji[5]   = ePermCoef*qiQI[2] + eUindCoef*qiUindI[1];
    VijR[2] += dPermCoef*qiQJ[5];
    VjiR[5]  = dPermCoef*qiQI[2] + dUindCoef*qiUindI[1];
    Vij[3]  += ePermCoef*qiQJ[6];
    Vji[6]   = ePermCoef*qiQI[3] + eUindCoef*qiUindI[2];
    VijR[3] += dPermCoef*qiQJ[6];
    VjiR[6]  = dPermCoef*qiQI[3] + dUindCoef*qiUindI[2];
    Vijd[1] += eUindCoef*qiQJ[5];
    Vijd[2] += eUindCoef*qiQJ[6];
    // D-Q and Uind-Q terms (m=1)
    Vij[5]   = -(ePermCoef*qiQJ[2] + eUindCoef*qiUindJ[1]);
    Vji[2]  += -(ePermCoef*qiQI[5]);
    VijR[5]  = -(dPermCoef*qiQJ[2] + dUindCoef*qiUindJ[1]);
    VjiR[2] += -(dPermCoef*qiQI[5]);
    Vij[6]   = -(ePermCoef*qiQJ[3] + eUindCoef*qiUindJ[2]);
    Vji[3]  += -(ePermCoef*qiQI[6]);
    VijR[6]  = -(dPermCoef*qiQJ[3] + dUindCoef*qiUindJ[2]);
    VjiR[3] += -(dPermCoef*qiQI[6]);
    Vjid[1] += -(eUindCoef*qiQI[5]);
    Vjid[2] += -(eUindCoef*qiQI[6]);

    // Q-Q terms (m=0)
    ePermCoef = 6.0*rInvVec[5]*mScale;
    dPermCoef = -15.0*rInvVec[6]*mScale;
    Vij[4]  += ePermCoef*qiQJ[4];
    Vji[4]  += ePermCoef*qiQI[4];
    VijR[4] += dPermCoef*qiQJ[4];
    VjiR[4] += dPermCoef*qiQI[4];
    // Q-Q terms (m=1)
    ePermCoef = -4.0*rInvVec[5]*mScale;
    dPermCoef = 10.0*rInvVec[6]*mScale;
    Vij[5]  += ePermCoef*qiQJ[5];
    Vji[5]  += ePermCoef*qiQI[5];
    VijR[5] += dPermCoef*qiQJ[5];
    VjiR[5] += dPermCoef*qiQI[5];
    Vij[6]  += ePermCoef*qiQJ[6];
    Vji[6]  += ePermCoef*qiQI[6];
    VijR[6] += dPermCoef*qiQJ[6];
    VjiR[6] += dPermCoef*qiQI[6];
    // Q-Q terms (m=2)
    ePermCoef = rInvVec[5]*mScale;
    dPermCoef = -2.5*rInvVec[6]*mScale;
    Vij[7]  = ePermCoef*qiQJ[7];
    Vji[7]  = ePermCoef*qiQI[7];
    VijR[7] = dPermCoef*qiQJ[7];
    VjiR[7] = dPermCoef*qiQI[7];
    Vij[8]  = ePermCoef*qiQJ[8];
    Vji[8]  = ePermCoef*qiQI[8];
    VijR[8] = dPermCoef*qiQJ[8];
    VjiR[8] = dPermCoef*qiQI[8];

    // C-O (m=0)
    ePermCoef = -rInvVec[4]*mScale;
    dPermCoef = 2.0*rInvVec[5]*mScale;
    Vij[0]  += ePermCoef*qiQJ[9];
    Vji[9]   = ePermCoef*qiQI[0];
    VijR[0] += dPermCoef*qiQJ[9];
    VjiR[9]  = dPermCoef*qiQI[0];
    // O-C (m=0)
    Vij[9]   = -ePermCoef*qiQJ[0];
    Vji[0]  -=  ePermCoef*qiQI[9];
    VijR[9]  = -dPermCoef*qiQJ[0];
    VjiR[0] -=  dPermCoef*qiQI[9];

    // D-O and Uind-O (m=0)
    ePermCoef = -4.0*rInvVec[5]*mScale;
    eUindCoef = -8.0*rInvVec[5]*dScale*thole_o0;
    dPermCoef = 10.0*rInvVec[6]*mScale;
    dUindCoef = 40.0*rInvVec[6]*dScale*dthole_o0;
    Vij[1]  += ePermCoef*qiQJ[9];
    Vji[9]  += ePermCoef*qiQI[1] + eUindCoef*qiUindI[0];
    VijR[1] += dPermCoef*qiQJ[9];
    VjiR[9] += dPermCoef*qiQI[1] + dUindCoef*qiUindI[0];
    // O-D and O-Uind (m=0)
    Vij[9]  += ePermCoef*qiQJ[1] + eUindCoef*qiUindJ[0];
    Vji[1]  += ePermCoef*qiQI[9];
    VijR[9] += dPermCoef*qiQJ[1] + dUindCoef*qiUindJ[0];
    VjiR[1] += dPermCoef*qiQI[9];
    Vijd[0] += eUindCoef*qiQJ[9];
    Vjid[0] += eUindCoef*qiQI[9];
    // D-O and O-Uind (m=1)
    ePermCoef = sqrtSix*rInvVec[5]*mScale;
    eUindCoef = 2.0*sqrtSix*rInvVec[5]*dScale*thole_o1;
    dPermCoef = -2.5*sqrtSix*rInvVec[6]*mScale;
    dUindCoef = -10.0*sqrtSix*rInvVec[6]*dScale*dthole_o1;
    Vij[2]   += ePermCoef*qiQJ[10];
    Vji[10]   = ePermCoef*qiQI[2] + eUindCoef*qiUindI[1];
    VijR[2]  += dPermCoef*qiQJ[10];
    VjiR[10]  = dPermCoef*qiQI[2] + dUindCoef*qiUindI[1];
    Vij[3]   += ePermCoef*qiQJ[11];
    Vji[11]   = ePermCoef*qiQI[3] + eUindCoef*qiUindI[2];
    VijR[3]  += dPermCoef*qiQJ[11];
    VjiR[11]  = dPermCoef*qiQI[3] + dUindCoef*qiUindI[2];
    Vijd[1] += eUindCoef*qiQJ[10];
    Vijd[2] += eUindCoef*qiQJ[11];
    // O-D and O-Uind (m=1)
    Vij[10]   = ePermCoef*qiQJ[2] + eUindCoef*qiUindJ[1];
    Vji[2]   += ePermCoef*qiQI[10];
    VijR[10]  = dPermCoef*qiQJ[2] + dUindCoef*qiUindJ[1];
    VjiR[2]  += dPermCoef*qiQI[10];
    Vij[11]   = ePermCoef*qiQJ[3] + eUindCoef*qiUindJ[2];
    Vji[3]   += ePermCoef*qiQI[11];
    VijR[11]  = dPermCoef*qiQJ[3] + dUindCoef*qiUindJ[2];
    VjiR[3]  += dPermCoef*qiQI[11];
    Vjid[1] += eUindCoef*qiQI[10];
    Vjid[2] += eUindCoef*qiQI[11];

    // Q-O (m=0)
    ePermCoef = -10.0*rInvVec[6]*mScale;
    dPermCoef = 30.0*rInvVec[7]*mScale;
    Vij[4]  += ePermCoef*qiQJ[9];
    Vji[9]  += ePermCoef*qiQI[4];
    VijR[4] += dPermCoef*qiQJ[9];
    VjiR[9] += dPermCoef*qiQI[4];
    // O-Q (m=0)
    Vij[9]  -= ePermCoef*qiQJ[4];
    Vji[4]  -= ePermCoef*qiQI[9];
    VijR[9] -= dPermCoef*qiQJ[4];
    VjiR[4] -= dPermCoef*qiQI[9];
    // Q-O (m=1)
    ePermCoef = 5.0*sqrtTwo*rInvVec[6]*mScale;
    dPermCoef = -15.0*sqrtTwo*rInvVec[7]*mScale;
    Vij[5]   += ePermCoef*qiQJ[10];
    Vji[10]  += ePermCoef*qiQI[5];
    VijR[5]  += dPermCoef*qiQJ[10];
    VjiR[10] += dPermCoef*qiQI[5];
    Vij[6]   += ePermCoef*qiQJ[11];
    Vji[11]  += ePermCoef*qiQI[6];
    VijR[6]  += dPermCoef*qiQJ[11];
    VjiR[11] += dPermCoef*qiQI[6];
    // O-Q (m=1)
    Vij[10]  -= ePermCoef*qiQJ[5];
    Vji[5]   -= ePermCoef*qiQI[10];
    VijR[10] -= dPermCoef*qiQJ[5];
    VjiR[5]  -= dPermCoef*qiQI[10];
    Vij[11]  -= ePermCoef*qiQJ[6];
    Vji[6]   -= ePermCoef*qiQI[11];
    VijR[11] -= dPermCoef*qiQJ[6];
    VjiR[6]  -= dPermCoef*qiQI[11];
    // Q-O (m=2)
    ePermCoef = -sqrtFive*rInvVec[6]*mScale;
    dPermCoef = 3.0*sqrtFive*rInvVec[7]*mScale;
    Vij[7]  += ePermCoef*qiQJ[12];
    Vji[12]  = ePermCoef*qiQI[7];
    VijR[7] += dPermCoef*qiQJ[12];
    VjiR[12] = dPermCoef*qiQI[7];
    Vij[8]  += ePermCoef*qiQJ[13];
    Vji[13]  = ePermCoef*qiQI[8];
    VijR[8] += dPermCoef*qiQJ[13];
    VjiR[13] = dPermCoef*qiQI[8];
    // O-Q (m=2)
    Vij[12]  = -ePermCoef*qiQJ[7];
    Vji[7]  -=  ePermCoef*qiQI[12];
    VijR[12] = -dPermCoef*qiQJ[7];
    VjiR[7] -=  dPermCoef*qiQI[12];
    Vij[13]  = -ePermCoef*qiQJ[8];
    Vji[8]  -=  ePermCoef*qiQI[13];
    VijR[13] = -dPermCoef*qiQJ[8];
    VjiR[8] -=  dPermCoef*qiQI[13];

    // O-O (m=0)
    ePermCoef = -20.*rInvVec[7]*mScale;
    dPermCoef = 70.0*rInvVec[8]*mScale;
    Vij[9]  += ePermCoef*qiQJ[9];
    Vji[9]  += ePermCoef*qiQI[9];
    VijR[9] += dPermCoef*qiQJ[9];
    VjiR[9] += dPermCoef*qiQI[9];
    // O-O (m=1)
    ePermCoef = 15.0*rInvVec[7]*mScale;
    dPermCoef = -52.5*rInvVec[8]*mScale;
    Vij[10]  += ePermCoef*qiQJ[10];
    Vji[10]  += ePermCoef*qiQI[10];
    VijR[10] += dPermCoef*qiQJ[10];
    VjiR[10] += dPermCoef*qiQI[10];
    Vij[11]  += ePermCoef*qiQJ[11];
    Vji[11]  += ePermCoef*qiQI[11];
    VijR[11] += dPermCoef*qiQJ[11];
    VjiR[11] += dPermCoef*qiQI[11];
    // O-O (m=2)
    ePermCoef = -6.0*rInvVec[7]*mScale;
    dPermCoef = 21.0*rInvVec[8]*mScale;
    Vij[12]  += ePermCoef*qiQJ[12];
    Vji[12]  += ePermCoef*qiQI[12];
    VijR[12] += dPermCoef*qiQJ[12];
    VjiR[12] += dPermCoef*qiQI[12];
    Vij[13]  += ePermCoef*qiQJ[13];
    Vji[13]  += ePermCoef*qiQI[13];
    VijR[13] += dPermCoef*qiQJ[13];
    VjiR[13] += dPermCoef*qiQI[13];
    // O-O (m=3)
    ePermCoef = rInvVec[7]*mScale;
    dPermCoef = -3.5*rInvVec[8]*mScale;
    Vij[14]  = ePermCoef*qiQJ[14];
    Vji[14]  = ePermCoef*qiQI[14];
    VijR[14] = dPermCoef*qiQJ[14];
    VjiR[14] = dPermCoef*qiQI[14];
    Vij[15]  = ePermCoef*qiQJ[15];
    Vji[15]  = ePermCoef*qiQI[15];
    VijR[15] = dPermCoef*qiQJ[15];
    VjiR[15] = dPermCoef*qiQI[15];

    // Evaluate the energies, forces and torques due to permanent+induced moments
    // interacting with just the permanent moments.
    double energy = 0.5*(qiQI[0]*Vij[0] + qiQJ[0]*Vji[0]);
    double fIZ = qiQI[0]*VijR[0];
    double fJZ = qiQJ[0]*VjiR[0];
    double EIX = 0.0, EIY = 0.0, EIZ = 0.0, EJX = 0.0, EJY = 0.0, EJZ = 0.0;
    for (int i = 1; i < 16; ++i) {
        energy += 0.5*(qiQI[i]*Vij[i] + qiQJ[i]*Vji[i]);
        fIZ += qiQI[i]*VijR[i];
        fJZ += qiQJ[i]*VjiR[i];
        EIX += qiQIX[i]*Vij[i];
        EIY += qiQIY[i]*Vij[i];
        EIZ += qiQIZ[i]*Vij[i];
        EJX += qiQJX[i]*Vji[i];
        EJY += qiQJY[i]*Vji[i];
        EJZ += qiQJZ[i]*Vji[i];
    }
    // Define the torque intermediates for the induced dipoles. These are simply the induced dipole torque
    // intermediates dotted with the field due to permanent moments only, at each center. We inline the
    // induced dipole torque intermediates here, for simplicity. N.B. There are no torques on the dipoles
    // themselves, so we accumulate the torque intermediates into separate variables to allow them to be
    // used only in the force calculation.
    //
    // The torque about the x axis (needed to obtain the y force on the induced dipoles, below)
    //    qiUindIx[0] = qiQUindI[2];    qiUindIx[1] = 0;    qiUindIx[2] = -qiQUindI[0]
    double iEIX = qiUindI[2]*Vijd[0] - qiUindI[0]*Vijd[2];
    double iEJX = qiUindJ[2]*Vjid[0] - qiUindJ[0]*Vjid[2];
    // The torque about the y axis (needed to obtain the x force on the induced dipoles, below)
    //    qiUindIy[0] = -qiQUindI[1];   qiUindIy[1] = qiQUindI[0];    qiUindIy[2] = 0
    double iEIY = qiUindI[0]*Vijd[1] - qiUindI[1]*Vijd[0];
    double iEJY = qiUindJ[0]*Vjid[1] - qiUindJ[1]*Vjid[0];
    // The torque about the z axis (needed to obtain the x force on the induced dipoles, below)
    //    qiUindIz[0] = 0;  qiUindIz[1] = -qiQUindI[2];    qiUindIz[2] = qiQUindI[1]
    double iEIZ = qiUindI[1]*Vijd[2] - qiUindI[2]*Vijd[1];
    double iEJZ = qiUindJ[1]*Vjid[2] - qiUindJ[2]*Vjid[1];

    // Add in the induced-induced terms, if needed.
    if(getPolarizationType() == MPIDReferenceForce::Mutual) {
        // Uind-Uind terms (m=0)
        double eCoef = -8.0*rInvVec[3]*uScale*thole_d0;
        double dCoef = 12.0*rInvVec[4]*uScale*dthole_d0;
        iEIX += eCoef*qiUindI[2]*qiUindJ[0];
        iEJX += eCoef*qiUindJ[2]*qiUindI[0];
        iEIY -= eCoef*qiUindI[1]*qiUindJ[0];
        iEJY -= eCoef*qiUindJ[1]*qiUindI[0];
        fIZ  += dCoef*qiUindI[0]*qiUindJ[0];
        fJZ  += dCoef*qiUindJ[0]*qiUindI[0];
        // Uind-Uind terms (m=1)
        eCoef = 4.0*rInvVec[3]*uScale*thole_d1;
        dCoef = -6.0*rInvVec[4]*uScale*dthole_d1;
        iEIX -= eCoef*qiUindI[0]*qiUindJ[2];
        iEJX -= eCoef*qiUindJ[0]*qiUindI[2];
        iEIY += eCoef*qiUindI[0]*qiUindJ[1];
        iEJY += eCoef*qiUindJ[0]*qiUindI[1];
        iEIZ += eCoef*qiUindI[1]*qiUindJ[2];
        iEJZ += eCoef*qiUindJ[1]*qiUindI[2];
        fIZ  += dCoef*(qiUindI[1]*qiUindJ[1] + qiUindI[2]*qiUindJ[2]);
        fJZ  += dCoef*(qiUindJ[1]*qiUindI[1] + qiUindJ[2]*qiUindI[2]);
    }

    // The quasi-internal frame forces and torques.  Note that the induced torque intermediates are
    // used in the force expression, but not in the torques; the induced dipoles are isotropic.
    double qiForce[3] = {rInv*(EIY+EJY+iEIY+iEJY), -rInv*(EIX+EJX+iEIX+iEJX), -(fJZ+fIZ)};
    double qiTorqueI[3] = {-EIX, -EIY, -EIZ};
    double qiTorqueJ[3] = {-EJX, -EJY, -EJZ};
    if(particleI.isAnisotropic){
        qiTorqueI[0] += -iEIX;
        qiTorqueI[1] += -iEIY;
        qiTorqueI[2] += -iEIZ;
    }
    if(particleJ.isAnisotropic){
        qiTorqueJ[0] += -iEJX;
        qiTorqueJ[1] += -iEJY;
        qiTorqueJ[2] += -iEJZ;
    }

    // Rotate the forces and torques back to the lab frame
    Vec3 tmpf, tmpi, tmpj;
    for (int ii = 0; ii < 3; ii++) {
        double forceVal = 0.0;
        double torqueIVal = 0.0;
        double torqueJVal = 0.0;
        for (int jj = 0; jj < 3; jj++) {
            forceVal   += forceRotationMatrix[ii][jj] * qiForce[jj];
            torqueIVal += forceRotationMatrix[ii][jj] * qiTorqueI[jj];
            torqueJVal += forceRotationMatrix[ii][jj] * qiTorqueJ[jj];
        }
        tmpi[ii] = torqueIVal;
        tmpj[ii] = torqueJVal;
        tmpf[ii] = forceVal;
        torque[iIndex][ii] += torqueIVal;
        torque[jIndex][ii] += torqueJVal;
        forces[iIndex][ii]  -= forceVal;
        forces[jIndex][ii]  += forceVal;
    }
    return energy;
}

void MPIDReferenceForce::mapTorqueToForceForParticle(const MultipoleParticleData& particleI,
                                                                const MultipoleParticleData& particleU,
                                                                const MultipoleParticleData& particleV,
                                                                      MultipoleParticleData* particleW,
                                                                      int axisType, const Vec3& torque,
                                                                      vector<Vec3>& forces) const
{

    static const int U                  = 0;
    static const int V                  = 1;
    static const int W                  = 2;
    static const int R                  = 3;
    static const int S                  = 4;
    static const int UV                 = 5;
    static const int UW                 = 6;
    static const int VW                 = 7;
    static const int UR                 = 8;
    static const int US                 = 9;
    static const int VS                 = 10;
    static const int WS                 = 11;
    static const int LastVectorIndex    = 12;

    static const int X                  = 0;
    static const int Y                  = 1;
    static const int Z                  = 2;
    static const int I                  = 3;

    double norms[LastVectorIndex];
    double angles[LastVectorIndex][2];

    // ---------------------------------------------------------------------------------------

    // get coordinates of this atom and the z & x axis atoms
    // compute the vector between the atoms and 1/sqrt(d2), d2 is distance between
    // this atom and the axis atom

    if (axisType == MPIDForce::NoAxisType) {
        return;
    }

    Vec3 vectorU = particleU.position - particleI.position;
    norms[U] = normalizeVec3(vectorU);

    Vec3 vectorV = particleV.position - particleI.position;
    norms[V] = normalizeVec3(vectorV);

    Vec3 vectorW;
    if (particleW && (axisType == MPIDForce::ZBisect || axisType == MPIDForce::ThreeFold)) {
         vectorW = particleW->position - particleI.position;
    } else {
         vectorW = vectorU.cross(vectorV);
    }
    norms[W]  = normalizeVec3(vectorW);

    Vec3 vectorUV, vectorUW, vectorVW;
    vectorUV = vectorV.cross(vectorU);
    vectorUW = vectorW.cross(vectorU);
    vectorVW = vectorW.cross(vectorV);

    norms[UV]                     = normalizeVec3(vectorUV);
    norms[UW]                     = normalizeVec3(vectorUW);
    norms[VW]                     = normalizeVec3(vectorVW);

    // angles[][0] is cosine of angle
    // angles[][1] is sine   of angle

    angles[UV][0]                 = vectorU.dot(vectorV);
    angles[UV][1]                 = sqrt(1.0 - angles[UV][0]*angles[UV][0]);

    angles[UW][0]                 = vectorU.dot(vectorW);
    angles[UW][1]                 = sqrt(1.0 - angles[UW][0]*angles[UW][0]);

    angles[VW][0]                 = vectorV.dot(vectorW);
    angles[VW][1]                 = sqrt(1.0 - angles[VW][0]*angles[VW][0]);

    Vec3 dphi;
    dphi[U]                       = vectorU.dot(torque);
    dphi[V]                       = vectorV.dot(torque);
    dphi[W]                       = vectorW.dot(torque);
    dphi                         *= -1.0;

    // branch based on axis type

    if (axisType == MPIDForce::ZThenX || axisType == MPIDForce::Bisector) {

        double factor1;
        double factor2;
        double factor3;
        double factor4;
        double half = 0.5;

        factor1                 =  dphi[V]/(norms[U]*angles[UV][1]);
        factor2                 =  dphi[W]/(norms[U]);
        factor3                 = -dphi[U]/(norms[V]*angles[UV][1]);

        if (axisType == MPIDForce::Bisector) {
            factor2    *= half;
            factor4     = half*dphi[W]/(norms[V]);
        } else {
            factor4     = 0.0;
        }

        for (int ii = 0; ii < 3; ii++) {
            double forceU                                        =  vectorUV[ii]*factor1 + factor2*vectorUW[ii];
            forces[particleU.particleIndex][ii]                 -=  forceU;

            double forceV                                        =  vectorUV[ii]*factor3 + factor4*vectorVW[ii];
            forces[particleV.particleIndex][ii]                 -=  forceV;

            forces[particleI.particleIndex][ii]                 +=  (forceU + forceV);
        }

    } else if (axisType == MPIDForce::ZBisect) {

        Vec3 vectorR          = vectorV + vectorW;
        Vec3 vectorS          = vectorU.cross(vectorR);

        norms[R]              = normalizeVec3(vectorR);
        norms[S]              = normalizeVec3(vectorS);

        Vec3 vectorUR         =  vectorR.cross(vectorU);
        Vec3 vectorUS         =  vectorS.cross(vectorU);
        Vec3 vectorVS         =  vectorS.cross(vectorV);
        Vec3 vectorWS         =  vectorS.cross(vectorW);

        norms[UR]             = normalizeVec3(vectorUR);
        norms[US]             = normalizeVec3(vectorUS);
        norms[VS]             = normalizeVec3(vectorVS);
        norms[WS]             = normalizeVec3(vectorWS);

        angles[UR][0]         = vectorU.dot(vectorR);
        angles[UR][1]         = sqrt(1.0 - angles[UR][0]*angles[UR][0]);

        angles[US][0]         = vectorU.dot(vectorS);
        angles[US][1]         = sqrt(1.0 - angles[US][0]*angles[US][0]);

        angles[VS][0]         = vectorV.dot(vectorS);
        angles[VS][1]         = sqrt(1.0 - angles[VS][0]*angles[VS][0]);

        angles[WS][0]         = vectorW.dot(vectorS);
        angles[WS][1]         = sqrt(1.0 - angles[WS][0]*angles[WS][0]);

        Vec3 t1               = vectorV - vectorS*angles[VS][0];
        Vec3 t2               = vectorW - vectorS*angles[WS][0];

        double notUsed        = normalizeVec3(t1);
              notUsed         = normalizeVec3(t2);

        double ut1cos         = vectorU.dot(t1);
        double ut1sin         = sqrt(1.0 - ut1cos*ut1cos);

        double ut2cos         = vectorU.dot(t2);
        double ut2sin         = sqrt(1.0 - ut2cos*ut2cos);

        double dphiR          = vectorR.dot(torque)*(-1.0);
        double dphiS          = vectorS.dot(torque)*(-1.0);

        double factor1        = dphiR/(norms[U]*angles[UR][1]);
        double factor2        = dphiS/(norms[U]);
        double factor3        = dphi[U]/(norms[V]*(ut1sin+ut2sin));
        double factor4        = dphi[U]/(norms[W]*(ut1sin+ut2sin));

        Vec3 forceU               =  vectorUR*factor1 + vectorUS*factor2;
        forces[particleU.particleIndex]        -= forceU;

        Vec3 forceV               = (vectorS*angles[VS][1] - t1*angles[VS][0])*factor3;
        forces[particleV.particleIndex]        -= forceV;

        Vec3 forceW               = (vectorS*angles[WS][1] - t2*angles[WS][0])*factor4;
        forces[particleW->particleIndex]       -= forceW;

        forces[particleI.particleIndex]        += (forceU + forceV + forceW);

    } else if (axisType == MPIDForce::ThreeFold) {

        // 3-fold

        for (int ii = 0; ii < 3; ii++) {

            double du =  vectorUW[ii]*dphi[W]/(norms[U]*angles[UW][1]) +
                         vectorUV[ii]*dphi[V]/(norms[U]*angles[UV][1]) -
                         vectorUW[ii]*dphi[U]/(norms[U]*angles[UW][1]) -
                         vectorUV[ii]*dphi[U]/(norms[U]*angles[UV][1]);

            double dv =  vectorVW[ii]*dphi[W]/(norms[V]*angles[VW][1]) -
                         vectorUV[ii]*dphi[U]/(norms[V]*angles[UV][1]) -
                         vectorVW[ii]*dphi[V]/(norms[V]*angles[VW][1]) +
                         vectorUV[ii]*dphi[V]/(norms[V]*angles[UV][1]);

            double dw = -vectorUW[ii]*dphi[U]/(norms[W]*angles[UW][1]) -
                         vectorVW[ii]*dphi[V]/(norms[W]*angles[VW][1]) +
                         vectorUW[ii]*dphi[W]/(norms[W]*angles[UW][1]) +
                         vectorVW[ii]*dphi[W]/(norms[W]*angles[VW][1]);

            du /= 3.0;
            dv /= 3.0;
            dw /= 3.0;

            forces[particleU.particleIndex][ii] -= du;
            forces[particleV.particleIndex][ii] -= dv;
            if (particleW)
                forces[particleW->particleIndex][ii] -= dw;
            forces[particleI.particleIndex][ii] += (du + dv + dw);
        }

    } else if (axisType == MPIDForce::ZOnly) {

        // z-only

        for (int ii = 0; ii < 3; ii++) {
            double du                            = vectorUV[ii]*dphi[V]/(norms[U]*angles[UV][1]) + vectorUW[ii]*dphi[W]/norms[U];
            forces[particleU.particleIndex][ii] -= du;
            forces[particleI.particleIndex][ii] += du;
        }
    }
}

void MPIDReferenceForce::mapTorqueToForce(vector<MultipoleParticleData>& particleData,
                                                     const vector<int>& multipoleAtomXs,
                                                     const vector<int>& multipoleAtomYs,
                                                     const vector<int>& multipoleAtomZs,
                                                     const vector<int>& axisTypes,
                                                     vector<Vec3>& torques,
                                                     vector<Vec3>& forces) const
{

    // map torques to forces

    for (unsigned int ii = 0; ii < particleData.size(); ii++) {
        if (axisTypes[ii] != MPIDForce::NoAxisType) {
             mapTorqueToForceForParticle(particleData[ii],
                                         particleData[multipoleAtomZs[ii]], particleData[multipoleAtomXs[ii]],
                                         multipoleAtomYs[ii] > -1 ? &particleData[multipoleAtomYs[ii]] : NULL,
                                         axisTypes[ii], torques[ii], forces);
        }
    }
}

double MPIDReferenceForce::calculateElectrostatic(const vector<MultipoleParticleData>& particleData,
                                                             vector<Vec3>& torques,
                                                             vector<Vec3>& forces)
{
    double energy = 0.0;
    vector<double> scaleFactors(LAST_SCALE_TYPE_INDEX);
    for (auto& s : scaleFactors)
        s = 1.0;

    // main loop over particle pairs

    for (unsigned int ii = 0; ii < particleData.size(); ii++) {
        for (unsigned int jj = ii+1; jj < particleData.size(); jj++) {

            if (jj <= _maxScaleIndex[ii]) {
                getMultipoleScaleFactors(ii, jj, scaleFactors);
            }

            energy += calculateElectrostaticPairIxn(particleData[ii], particleData[jj], scaleFactors, forces, torques);

            if (jj <= _maxScaleIndex[ii]) {
                for (unsigned int kk = 0; kk < LAST_SCALE_TYPE_INDEX; kk++) {
                    scaleFactors[kk] = 1.0;
                }
            }
        }
    }
    if (getPolarizationType() == MPIDReferenceForce::Extrapolated) {
        double prefac = (_electric/_dielectric);
        for (int i = 0; i < _numParticles; i++) {
            // Compute the µ(m) T µ(n) force contributions here
            for (int l = 0; l < _maxPTOrder-1; ++l) {
                for (int m = 0; m < _maxPTOrder-1-l; ++m) {
                    double p = _extPartCoefficients[l+m+1];
                    if(std::fabs(p) < 1e-6) continue;
                    forces[i][0] += p*prefac*(_ptDipoleD[l][i][0]*_ptDipoleFieldGradientD[m][6*i+0]
                                            + _ptDipoleD[l][i][1]*_ptDipoleFieldGradientD[m][6*i+3]
                                            + _ptDipoleD[l][i][2]*_ptDipoleFieldGradientD[m][6*i+4]);
                    forces[i][1] += p*prefac*(_ptDipoleD[l][i][0]*_ptDipoleFieldGradientD[m][6*i+3]
                                            + _ptDipoleD[l][i][1]*_ptDipoleFieldGradientD[m][6*i+1]
                                            + _ptDipoleD[l][i][2]*_ptDipoleFieldGradientD[m][6*i+5]);
                    forces[i][2] += p*prefac*(_ptDipoleD[l][i][0]*_ptDipoleFieldGradientD[m][6*i+4]
                                            + _ptDipoleD[l][i][1]*_ptDipoleFieldGradientD[m][6*i+5]
                                            + _ptDipoleD[l][i][2]*_ptDipoleFieldGradientD[m][6*i+2]);
                    if(particleData[i].isAnisotropic){
                        torques[i][0] += p*prefac*(_ptDipoleD[l][i][1]*_ptDipoleFieldD[m][3*i+2]
                                                 - _ptDipoleD[l][i][2]*_ptDipoleFieldD[m][3*i+1]);
                        torques[i][1] += p*prefac*(_ptDipoleD[l][i][2]*_ptDipoleFieldD[m][3*i+0]
                                                 - _ptDipoleD[l][i][0]*_ptDipoleFieldD[m][3*i+2]);
                        torques[i][2] += p*prefac*(_ptDipoleD[l][i][0]*_ptDipoleFieldD[m][3*i+1]
                                                 - _ptDipoleD[l][i][1]*_ptDipoleFieldD[m][3*i+0]);
                    }
                }
            }
        }
    }

    return energy;
}

void MPIDReferenceForce::setup(const vector<Vec3>& particlePositions,
                                          const vector<double>& charges,
                                          const vector<double>& dipoles,
                                          const vector<double>& quadrupoles,
                                          const vector<double>& octopoles,
                                          const vector<double>& tholes,
                                          const vector<double>& dampingFactors,
                                          const vector<std::vector<double> >& polarity,
                                          const vector<int>& axisTypes,
                                          const vector<int>& multipoleAtomZs,
                                          const vector<int>& multipoleAtomXs,
                                          const vector<int>& multipoleAtomYs,
                                          const vector< vector< vector<int> > >& multipoleAtomCovalentInfo,
                                          vector<MultipoleParticleData>& particleData)
{


    // load particle parameters into vector of MultipoleParticleData
    // check for inverted chiral centers
    // apply rotation matrix to get lab frame dipole and quadrupoles
    // setup scaling factors
    // get induced dipoles
    // check if induced dipoles converged

    _numParticles = particlePositions.size();
    loadParticleData(particlePositions, charges, dipoles, quadrupoles, octopoles,
                      tholes, dampingFactors, polarity, particleData);

    checkChiral(particleData, multipoleAtomXs, multipoleAtomYs, multipoleAtomZs, axisTypes);

    applyRotationMatrix(particleData, multipoleAtomXs, multipoleAtomYs, multipoleAtomZs, axisTypes);

    setupScaleMaps(multipoleAtomCovalentInfo);

    calculateInducedDipoles(particleData);

    if (!getMutualInducedDipoleConverged()) {
        std::stringstream message;
        message << "Induced dipoles did not converge: ";
        message << " iterations="      << getMutualInducedDipoleIterations();
        message << " eps="             << getMutualInducedDipoleEpsilon();
        throw OpenMMException(message.str());
    }
}

double MPIDReferenceForce::calculateForceAndEnergy(const vector<Vec3>& particlePositions,
                                                             const vector<double>& charges,
                                                             const vector<double>& dipoles,
                                                             const vector<double>& quadrupoles,
                                                             const vector<double>& octopoles,
                                                             const vector<double>& tholes,
                                                             const vector<double>& dampingFactors,
                                                             const vector<std::vector<double> >& polarity,
                                                             const vector<int>& axisTypes,
                                                             const vector<int>& multipoleAtomZs,
                                                             const vector<int>& multipoleAtomXs,
                                                             const vector<int>& multipoleAtomYs,
                                                             const vector< vector< vector<int> > >& multipoleAtomCovalentInfo,
                                                             vector<Vec3>& forces)
{

    // setup, including calculating induced dipoles
    // calculate electrostatic ixns including torques
    // map torques to forces

    vector<MultipoleParticleData> particleData;
    setup(particlePositions, charges, dipoles, quadrupoles, octopoles, tholes,
           dampingFactors, polarity, axisTypes, multipoleAtomZs, multipoleAtomXs, multipoleAtomYs,
           multipoleAtomCovalentInfo, particleData);

    vector<Vec3> torques;
    initializeVec3Vector(torques);
    double energy = calculateElectrostatic(particleData, torques, forces);

    mapTorqueToForce(particleData, multipoleAtomXs, multipoleAtomYs, multipoleAtomZs, axisTypes, torques, forces);

    return energy;
}

void MPIDReferenceForce::calculateInducedDipoles(const vector<Vec3>& particlePositions,
                                                            const vector<double>& charges,
                                                            const vector<double>& dipoles,
                                                            const vector<double>& quadrupoles,
                                                            const vector<double>& octopoles,
                                                            const vector<double>& tholes,
                                                            const vector<double>& dampingFactors,
                                                            const std::vector<std::vector<double> > &polarity,
                                                            const vector<int>& axisTypes,
                                                            const vector<int>& multipoleAtomZs,
                                                            const vector<int>& multipoleAtomXs,
                                                            const vector<int>& multipoleAtomYs,
                                                            const vector< vector< vector<int> > >& multipoleAtomCovalentInfo,
                                                            vector<Vec3>& outputInducedDipoles) {
    // setup, including calculating induced dipoles

    vector<MultipoleParticleData> particleData;
    setup(particlePositions, charges, dipoles, quadrupoles, octopoles, tholes,
           dampingFactors, polarity, axisTypes, multipoleAtomZs, multipoleAtomXs, multipoleAtomYs,
           multipoleAtomCovalentInfo, particleData);
    outputInducedDipoles = _inducedDipole;
}




void MPIDReferenceForce::calculateLabFramePermanentDipoles(const vector<Vec3>& particlePositions,
                                                                      const vector<double>& charges,
                                                                      const vector<double>& dipoles,
                                                                      const vector<double>& quadrupoles,
                                                                      const vector<double>& octopoles,
                                                                      const vector<double>& tholes,
                                                                      const vector<double>& dampingFactors,
                                                                      const vector<std::vector<double> >& polarity,
                                                                      const vector<int>& axisTypes,
                                                                      const vector<int>& multipoleAtomZs,
                                                                      const vector<int>& multipoleAtomXs,
                                                                      const vector<int>& multipoleAtomYs,
                                                                      const vector< vector< vector<int> > >& multipoleAtomCovalentInfo,
                                                                      vector<Vec3>& outputRotatedPermanentDipoles) {
    // setup, including calculating permanent dipoles

    vector<MultipoleParticleData> particleData;
    setup(particlePositions, charges, dipoles, quadrupoles, octopoles, tholes,
           dampingFactors, polarity, axisTypes, multipoleAtomZs, multipoleAtomXs, multipoleAtomYs,
           multipoleAtomCovalentInfo, particleData);
    outputRotatedPermanentDipoles.resize(_numParticles);
    for (int i = 0; i < _numParticles; i++)
        outputRotatedPermanentDipoles[i] = particleData[i].dipole;
}

void MPIDReferenceForce::calculateTotalDipoles(const vector<Vec3>& particlePositions,
                                                          const vector<double>& charges,
                                                          const vector<double>& dipoles,
                                                          const vector<double>& quadrupoles,
                                                          const vector<double>& octopoles,
                                                          const vector<double>& tholes,
                                                          const vector<double>& dampingFactors,
                                                          const vector<std::vector<double> >& polarity,
                                                          const vector<int>& axisTypes,
                                                          const vector<int>& multipoleAtomZs,
                                                          const vector<int>& multipoleAtomXs,
                                                          const vector<int>& multipoleAtomYs,
                                                          const vector< vector< vector<int> > >& multipoleAtomCovalentInfo,
                                                          vector<Vec3>& outputTotalDipoles) {
    // setup, including calculating permanent dipoles

    vector<MultipoleParticleData> particleData;
    setup(particlePositions, charges, dipoles, quadrupoles, octopoles, tholes,
           dampingFactors, polarity, axisTypes, multipoleAtomZs, multipoleAtomXs, multipoleAtomYs,
           multipoleAtomCovalentInfo, particleData);
    outputTotalDipoles.resize(_numParticles);
    for (int i = 0; i < _numParticles; i++)
        for (int j = 0; j < 3; j++)
            outputTotalDipoles[i][j] = particleData[i].dipole[j] + _inducedDipole[i][j];
}

void MPIDReferenceForce::calculateMPIDSystemMultipoleMoments(const vector<double>& masses,
                                                                          const vector<Vec3>& particlePositions,
                                                                          const vector<double>& charges,
                                                                          const vector<double>& dipoles,
                                                                          const vector<double>& quadrupoles,
                                                                          const vector<double>& octopoles,
                                                                          const vector<double>& tholes,
                                                                          const vector<double>& dampingFactors,
                                                                          const std::vector<std::vector<double> > &polarity,
                                                                          const vector<int>& axisTypes,
                                                                          const vector<int>& multipoleAtomZs,
                                                                          const vector<int>& multipoleAtomXs,
                                                                          const vector<int>& multipoleAtomYs,
                                                                          const vector< vector< vector<int> > >& multipoleAtomCovalentInfo,
                                                                          vector<double>& outputMultipoleMoments)
{

    // setup, including calculating induced dipoles
    // remove center of mass
    // calculate system moments

    vector<MultipoleParticleData> particleData;
    setup(particlePositions, charges, dipoles, quadrupoles, octopoles, tholes,
           dampingFactors, polarity, axisTypes, multipoleAtomZs, multipoleAtomXs, multipoleAtomYs,
           multipoleAtomCovalentInfo, particleData);

    double totalMass = 0.0;
    Vec3 centerOfMass = Vec3(0.0, 0.0, 0.0);
    for (unsigned int ii  = 0; ii < _numParticles; ii++) {
        double mass   = masses[ii];
        totalMass    += mass;
        centerOfMass += particleData[ii].position*mass;
    }
    vector<Vec3> localPositions(_numParticles);
    if (totalMass > 0.0) {
        centerOfMass  *= 1.0/totalMass;
    }
    for (unsigned int ii  = 0; ii < _numParticles; ii++) {
        localPositions[ii] = particleData[ii].position - centerOfMass;
    }

    double netchg  = 0.0;

    Vec3 dpl       = Vec3(0.0, 0.0, 0.0);

    double xxqdp   = 0.0;
    double xyqdp   = 0.0;
    double xzqdp   = 0.0;

    double yyqdp   = 0.0;
    double yzqdp   = 0.0;

    double zzqdp   = 0.0;

    for (unsigned int ii  = 0; ii < _numParticles; ii++) {

        double charge         = particleData[ii].charge;
        Vec3 position         = localPositions[ii];
        netchg               += charge;

        Vec3 netDipole        = (particleData[ii].dipole  + _inducedDipole[ii]);

        dpl                  += position*charge + netDipole;

        xxqdp                += position[0]*position[0]*charge + 2.0*position[0]*netDipole[0];
        xyqdp                += position[0]*position[1]*charge + position[0]*netDipole[1] + position[1]*netDipole[0];
        xzqdp                += position[0]*position[2]*charge + position[0]*netDipole[2] + position[2]*netDipole[0];

        yyqdp                += position[1]*position[1]*charge + 2.0*position[1]*netDipole[1];
        yzqdp                += position[1]*position[2]*charge + position[1]*netDipole[2] + position[2]*netDipole[1];

        zzqdp                += position[2]*position[2]*charge + 2.0*position[2]*netDipole[2];

    }

    // convert the quadrupole from traced to traceless form

    outputMultipoleMoments.resize(13);
    double qave                  = (xxqdp + yyqdp + zzqdp)/3.0;
    outputMultipoleMoments[4]    = 0.5*(xxqdp-qave);
    outputMultipoleMoments[5]    = 0.5*xyqdp;
    outputMultipoleMoments[6]    = 0.5*xzqdp;
    outputMultipoleMoments[8]    = 0.5*(yyqdp-qave);
    outputMultipoleMoments[9]    = 0.5*yzqdp;
    outputMultipoleMoments[12]   = 0.5*(zzqdp-qave);

    // add the traceless atomic quadrupoles to total quadrupole

    for (unsigned int ii = 0; ii < _numParticles; ii++) {
        outputMultipoleMoments[4]  += particleData[ii].quadrupole[QXX];
        outputMultipoleMoments[5]  += particleData[ii].quadrupole[QXY];
        outputMultipoleMoments[6]  += particleData[ii].quadrupole[QXZ];
        outputMultipoleMoments[8]  += particleData[ii].quadrupole[QYY];
        outputMultipoleMoments[9]  += particleData[ii].quadrupole[QYZ];
        outputMultipoleMoments[12] += particleData[ii].quadrupole[QZZ];
    }
    outputMultipoleMoments[7]  = outputMultipoleMoments[5];
    outputMultipoleMoments[10] = outputMultipoleMoments[6];
    outputMultipoleMoments[11] = outputMultipoleMoments[9];

    double debye = 4.80321;

    outputMultipoleMoments[0] = netchg;

    dpl                       *= 10.0*debye;
    outputMultipoleMoments[1]  = dpl[0];
    outputMultipoleMoments[2]  = dpl[1];
    outputMultipoleMoments[3]  = dpl[2];

    debye *= 3.0;
    for (unsigned int ii = 4; ii < 13; ii++) {
        outputMultipoleMoments[ii] *= 100.0*debye;
    }
}

double MPIDReferenceForce::calculateElectrostaticPotentialForParticleGridPoint(const MultipoleParticleData& particleI, const Vec3& gridPoint) const
{

    Vec3 deltaR = particleI.position - gridPoint;

    getPeriodicDelta(deltaR);

    double r2            = deltaR.dot(deltaR);
    double r             = sqrt(r2);

    double rr1           = 1.0/r;
    double rr2           = rr1*rr1;
    double rr3           = rr1*rr2;
    double potential     = particleI.charge*rr1;

    double scd           = particleI.dipole.dot(deltaR);
    double scu           = _inducedDipole[particleI.particleIndex].dot(deltaR);
    potential           -= (scd + scu)*rr3;

    double rr5           = 3.0*rr3*rr2;
    double scq           = deltaR[0]*(particleI.quadrupole[QXX]*deltaR[0] + particleI.quadrupole[QXY]*deltaR[1] + particleI.quadrupole[QXZ]*deltaR[2]);
          scq           += deltaR[1]*(particleI.quadrupole[QXY]*deltaR[0] + particleI.quadrupole[QYY]*deltaR[1] + particleI.quadrupole[QYZ]*deltaR[2]);
          scq           += deltaR[2]*(particleI.quadrupole[QXZ]*deltaR[0] + particleI.quadrupole[QYZ]*deltaR[1] + particleI.quadrupole[QZZ]*deltaR[2]);
    potential           += scq*rr5;
    return potential;
}

void MPIDReferenceForce::calculateElectrostaticPotential(const vector<Vec3>& particlePositions,
                                                                    const vector<double>& charges,
                                                                    const vector<double>& dipoles,
                                                                    const vector<double>& quadrupoles,
                                                                    const vector<double>& octopoles,
                                                                    const vector<double>& tholes,
                                                                    const vector<double>& dampingFactors,
                                                                    const std::vector<std::vector<double> > &polarity,
                                                                    const vector<int>& axisTypes,
                                                                    const vector<int>& multipoleAtomZs,
                                                                    const vector<int>& multipoleAtomXs,
                                                                    const vector<int>& multipoleAtomYs,
                                                                    const vector< vector< vector<int> > >& multipoleAtomCovalentInfo,
                                                                    const vector<Vec3>& grid,
                                                                    vector<double>& potential)
{

    // setup, including calculating induced dipoles
    // initialize potential
    // calculate contribution of each particle to potential at grid point
    // apply prefactor

    vector<MultipoleParticleData> particleData;
    setup(particlePositions, charges, dipoles, quadrupoles, octopoles, tholes,
           dampingFactors, polarity, axisTypes, multipoleAtomZs, multipoleAtomXs, multipoleAtomYs,
           multipoleAtomCovalentInfo, particleData);

    potential.resize(grid.size());
    for (auto& p : potential)
        p = 0.0;

    for (unsigned int ii = 0; ii < _numParticles; ii++) {
        for (unsigned int jj = 0; jj < grid.size(); jj++) {
            potential[jj] += calculateElectrostaticPotentialForParticleGridPoint(particleData[ii], grid[jj]);
        }
    }

    double term = _electric/_dielectric;
    for (auto& p : potential)
        p *= term;
}

MPIDReferenceForce::UpdateInducedDipoleFieldStruct::UpdateInducedDipoleFieldStruct(vector<OpenMM::Vec3>& inputFixed_E_Field, vector<OpenMM::Vec3>& inputInducedDipoles, vector<vector<Vec3> >& extrapolatedDipoles,  vector<vector<double> >& extrapolatedDipoleField,  vector<vector<double> >& extrapolatedDipoleFieldGradient) :
        fixedMultipoleField(&inputFixed_E_Field), inducedDipoles(&inputInducedDipoles), extrapolatedDipoles(&extrapolatedDipoles), extrapolatedDipoleField(&extrapolatedDipoleField),  extrapolatedDipoleFieldGradient(&extrapolatedDipoleFieldGradient) {
    inducedDipoleField.resize(fixedMultipoleField->size());
}


const int MPIDReferencePmeForce::MPID_PME_ORDER = 6;

const double MPIDReferencePmeForce::SQRT_PI = 1.77245385091;

MPIDReferencePmeForce::MPIDReferencePmeForce() :
               MPIDReferenceForce(PME),
               _cutoffDistance(1.0), _cutoffDistanceSquared(1.0),
               _pmeGridSize(0), _totalGridSize(0), _alphaEwald(0.0)
{

    _fftplan = NULL;
    _pmeGrid = NULL;
    _pmeGridDimensions = IntVec(-1, -1, -1);
}

MPIDReferencePmeForce::~MPIDReferencePmeForce()
{
    if (_fftplan) {
        fftpack_destroy(_fftplan);
    }
    if (_pmeGrid) {
        delete [] _pmeGrid;
    }
};

double MPIDReferencePmeForce::getCutoffDistance() const
{
     return _cutoffDistance;
};

void MPIDReferencePmeForce::setCutoffDistance(double cutoffDistance)
{
     _cutoffDistance        = cutoffDistance;
     _cutoffDistanceSquared = cutoffDistance*cutoffDistance;
};

double MPIDReferencePmeForce::getAlphaEwald() const
{
     return _alphaEwald;
};

void MPIDReferencePmeForce::setAlphaEwald(double alphaEwald)
{
     _alphaEwald = alphaEwald;
};

void MPIDReferencePmeForce::getPmeGridDimensions(vector<int>& pmeGridDimensions) const
{

    pmeGridDimensions.resize(3);

    pmeGridDimensions[0] = _pmeGridDimensions[0];
    pmeGridDimensions[1] = _pmeGridDimensions[1];
    pmeGridDimensions[2] = _pmeGridDimensions[2];
};

void MPIDReferencePmeForce::setPmeGridDimensions(vector<int>& pmeGridDimensions)
{

    if ((pmeGridDimensions[0] == _pmeGridDimensions[0]) &&
        (pmeGridDimensions[1] == _pmeGridDimensions[1]) &&
        (pmeGridDimensions[2] == _pmeGridDimensions[2]))
        return;

    if (_fftplan) {
        fftpack_destroy(_fftplan);
    }
    fftpack_init_3d(&_fftplan,pmeGridDimensions[0], pmeGridDimensions[1], pmeGridDimensions[2]);

    _pmeGridDimensions[0] = pmeGridDimensions[0];
    _pmeGridDimensions[1] = pmeGridDimensions[1];
    _pmeGridDimensions[2] = pmeGridDimensions[2];

    initializeBSplineModuli();
};

void MPIDReferencePmeForce::setPeriodicBoxSize(OpenMM::Vec3* vectors)
{

    if (vectors[0][0] == 0.0 || vectors[1][1] == 0.0 || vectors[2][2] == 0.0) {
        std::stringstream message;
        message << "Box size of zero is invalid.";
        throw OpenMMException(message.str());
    }

    _periodicBoxVectors[0] = vectors[0];
    _periodicBoxVectors[1] = vectors[1];
    _periodicBoxVectors[2] = vectors[2];
    double determinant = vectors[0][0]*vectors[1][1]*vectors[2][2];
    assert(determinant > 0);
    double scale = 1.0/determinant;
    _recipBoxVectors[0] = Vec3(vectors[1][1]*vectors[2][2], 0, 0)*scale;
    _recipBoxVectors[1] = Vec3(-vectors[1][0]*vectors[2][2], vectors[0][0]*vectors[2][2], 0)*scale;
    _recipBoxVectors[2] = Vec3(vectors[1][0]*vectors[2][1]-vectors[1][1]*vectors[2][0], -vectors[0][0]*vectors[2][1], vectors[0][0]*vectors[1][1])*scale;
};

int compareInt2(const int2& v1, const int2& v2)
{
    return v1[1] < v2[1];
}

void MPIDReferencePmeForce::resizePmeArrays()
{

    _totalGridSize = _pmeGridDimensions[0]*_pmeGridDimensions[1]*_pmeGridDimensions[2];
    if (_pmeGridSize < _totalGridSize) {
        if (_pmeGrid) {
            delete _pmeGrid;
        }
        _pmeGrid      = new t_complex[_totalGridSize];
        _pmeGridSize  = _totalGridSize;
    }

    for (unsigned int ii = 0; ii < 3; ii++) {
       _pmeBsplineModuli[ii].resize(_pmeGridDimensions[ii]);
       _thetai[ii].resize(MPID_PME_ORDER*_numParticles);
    }

    _iGrid.resize(_numParticles);
    _phi.resize(35*_numParticles);
    _phidp.resize(35*_numParticles);
}

void MPIDReferencePmeForce::initializePmeGrid()
{
    if (_pmeGrid == NULL)
        return;

    for (int jj = 0; jj < _totalGridSize; jj++)
        _pmeGrid[jj].re = _pmeGrid[jj].im = 0.0;
}

void MPIDReferencePmeForce::getPeriodicDelta(Vec3& deltaR) const
{
    deltaR -= _periodicBoxVectors[2]*floor(deltaR[2]*_recipBoxVectors[2][2]+0.5);
    deltaR -= _periodicBoxVectors[1]*floor(deltaR[1]*_recipBoxVectors[1][1]+0.5);
    deltaR -= _periodicBoxVectors[0]*floor(deltaR[0]*_recipBoxVectors[0][0]+0.5);
}

void MPIDReferencePmeForce::getDampedInverseDistances(const MultipoleParticleData& particleI,
                                                                 const MultipoleParticleData& particleJ,
                                                                 double dscale, double pscale, double r,
                                                                 std::vector<double>& dampedDInverseDistances,
                                                                 std::vector<double>& dampedPInverseDistances) const
{

    std::vector<double> scaleFactor(4, 1.0);
    dampedDInverseDistances.reserve(4);
    dampedPInverseDistances.reserve(4);
    double damp = particleI.dampingFactor*particleJ.dampingFactor;
    if (damp != 0.0) {

        double ratio   = (r/damp);

        double pgamma  = pscale == 0.0 ? particleI.thole + particleJ.thole : _defaultTholeWidth;
               damp    = pgamma*ratio;
        if (damp < 50.0) {
            double expdamp = exp(-damp);
            scaleFactor[0] = 1.0 - expdamp*(1.0 + damp + 0.5*damp*damp);
            scaleFactor[1] = 1.0 - expdamp*(1.0 + damp + 0.5*damp*damp + damp*damp*damp/6.0);
            scaleFactor[2] = 1.0 - expdamp*(1.0 + damp + 0.5*damp*damp + damp*damp*damp/6.0 + damp*damp*damp*damp/30.0);
            scaleFactor[3] = 1.0 - expdamp*(1.0 + damp + 0.5*damp*damp + damp*damp*damp/6.0 + 4.0*damp*damp*damp*damp/105.0 + damp*damp*damp*damp*damp/210.0);
        }
    }

    double r2              = r*r;
    double r3              = r*r2;
    double r5              = r3*r2;
    double r7              = r5*r2;
    double r9              = r7*r2;

    dampedDInverseDistances[0] =       (1.0-dscale*scaleFactor[0])/r3;
    dampedDInverseDistances[1] =   3.0*(1.0-dscale*scaleFactor[1])/r5;
    dampedDInverseDistances[2] =  15.0*(1.0-dscale*scaleFactor[2])/r7;
    dampedDInverseDistances[3] = 105.0*(1.0-dscale*scaleFactor[3])/r9;
    dampedPInverseDistances[0] =       (1.0-pscale*scaleFactor[0])/r3;
    dampedPInverseDistances[1] =   3.0*(1.0-pscale*scaleFactor[1])/r5;
    dampedPInverseDistances[2] =  15.0*(1.0-pscale*scaleFactor[2])/r7;
    dampedPInverseDistances[3] = 105.0*(1.0-pscale*scaleFactor[3])/r9;
}

void MPIDReferencePmeForce::initializeBSplineModuli()
{

    // Initialize the b-spline moduli.

    int maxSize = -1;
    for (unsigned int ii = 0; ii < 3; ii++) {
       _pmeBsplineModuli[ii].resize(_pmeGridDimensions[ii]);
        maxSize = maxSize  > _pmeGridDimensions[ii] ? maxSize : _pmeGridDimensions[ii];
    }

    double array[MPID_PME_ORDER];
    double x = 0.0;
    array[0]     = 1.0 - x;
    array[1]     = x;
    for (int k = 2; k < MPID_PME_ORDER; k++) {
        double denom = 1.0/k;
        array[k] = x*array[k-1]*denom;
        for (int i = 1; i < k; i++) {
            array[k-i] = ((x+i)*array[k-i-1] + ((k-i+1)-x)*array[k-i])*denom;
        }
        array[0] = (1.0-x)*array[0]*denom;
    }

    vector<double> bsarray(maxSize+1, 0.0);
    for (int i = 2; i <= MPID_PME_ORDER+1; i++) {
        bsarray[i] = array[i-2];
    }
    for (int dim = 0; dim < 3; dim++) {

        int size = _pmeGridDimensions[dim];

        // get the modulus of the discrete Fourier transform

        double factor = 2.0*M_PI/size;
        for (int i = 0; i < size; i++) {
            double sum1 = 0.0;
            double sum2 = 0.0;
            for (int j = 1; j <= size; j++) {
                double arg = factor*i*(j-1);
                sum1 += bsarray[j]*cos(arg);
                sum2 += bsarray[j]*sin(arg);
            }
            _pmeBsplineModuli[dim][i] = (sum1*sum1 + sum2*sum2);
        }

        // fix for exponential Euler spline interpolation failure

        double eps = 1.0e-7;
        if (_pmeBsplineModuli[dim][0] < eps) {
            _pmeBsplineModuli[dim][0] = 0.5*_pmeBsplineModuli[dim][1];
        }
        for (int i = 1; i < size-1; i++) {
            if (_pmeBsplineModuli[dim][i] < eps) {
                _pmeBsplineModuli[dim][i] = 0.5*(_pmeBsplineModuli[dim][i-1]+_pmeBsplineModuli[dim][i+1]);
            }
        }
        if (_pmeBsplineModuli[dim][size-1] < eps) {
            _pmeBsplineModuli[dim][size-1] = 0.5*_pmeBsplineModuli[dim][size-2];
        }

        // compute and apply the optimal zeta coefficient

        int jcut = 50;
        for (int i = 1; i <= size; i++) {
            int k = i - 1;
            if (i > size/2)
                k = k - size;
            double zeta;
            if (k == 0)
                zeta = 1.0;
            else {
                double sum1 = 1.0;
                double sum2 = 1.0;
                factor = M_PI*k/size;
                for (int j = 1; j <= jcut; j++) {
                    double arg = factor/(factor+M_PI*j);
                    sum1 = sum1 + pow(arg,   MPID_PME_ORDER);
                    sum2 = sum2 + pow(arg, 2*MPID_PME_ORDER);
                }
                for (int j = 1; j <= jcut; j++) {
                    double arg  = factor/(factor-M_PI*j);
                    sum1 += pow(arg,   MPID_PME_ORDER);
                    sum2 += pow(arg, 2*MPID_PME_ORDER);
                }
                zeta = sum2/sum1;
            }
            _pmeBsplineModuli[dim][i-1] = _pmeBsplineModuli[dim][i-1]*(zeta*zeta);
        }
    }
}

void MPIDReferencePmeForce::calculateFixedMultipoleFieldPairIxn(const MultipoleParticleData& particleI,
                                                                           const MultipoleParticleData& particleJ,
                                                                           double dscale, double pscale)
{

    unsigned int iIndex    = particleI.particleIndex;
    unsigned int jIndex    = particleJ.particleIndex;

    // compute the real space portion of the Ewald summation

    if (particleI.particleIndex == particleJ.particleIndex)
        return;

    Vec3 deltaR = particleJ.position - particleI.position;
    getPeriodicDelta(deltaR);
    double r2 = deltaR.dot(deltaR);

    if (r2 > _cutoffDistanceSquared)
        return;

    double r           = sqrt(r2);

    // calculate the error function damping terms

    double ralpha      = _alphaEwald*r;

    double bn0         = erfc(ralpha)/r;
    double alsq2       = 2.0*_alphaEwald*_alphaEwald;
    double alsq2n      = 1.0/(SQRT_PI*_alphaEwald);
    double exp2a       = exp(-(ralpha*ralpha));
    alsq2n            *= alsq2;
    double bn1         = (bn0+alsq2n*exp2a)/r2;

    alsq2n            *= alsq2;
    double bn2         = (3.0*bn1+alsq2n*exp2a)/r2;

    alsq2n            *= alsq2;
    double bn3         = (5.0*bn2+alsq2n*exp2a)/r2;

    alsq2n            *= alsq2;
    double bn4         = (7.0*bn3+alsq2n*exp2a)/r2;

    // compute the error function scaled and unscaled terms

    std::vector<double> dampedDInverseDistances;
    std::vector<double> dampedPInverseDistances;
    getDampedInverseDistances(particleI, particleJ, dscale, pscale, r, dampedDInverseDistances, dampedPInverseDistances);

    double drr3        = dampedDInverseDistances[0];
    double drr5        = dampedDInverseDistances[1];
    double drr7        = dampedDInverseDistances[2];
    double drr9        = dampedDInverseDistances[3];

    double dir         = particleI.dipole.dot(deltaR);

    Vec3 qxI           = Vec3(particleI.quadrupole[QXX], particleI.quadrupole[QXY], particleI.quadrupole[QXZ]);
    Vec3 qyI           = Vec3(particleI.quadrupole[QXY], particleI.quadrupole[QYY], particleI.quadrupole[QYZ]);
    Vec3 qzI           = Vec3(particleI.quadrupole[QXZ], particleI.quadrupole[QYZ], particleI.quadrupole[QZZ]);

    Vec3 qi            = Vec3(qxI.dot(deltaR), qyI.dot(deltaR), qzI.dot(deltaR));
    double qir         = qi.dot(deltaR);

    double djr         = particleJ.dipole.dot(deltaR);

    Vec3 qxJ           = Vec3(particleJ.quadrupole[QXX], particleJ.quadrupole[QXY], particleJ.quadrupole[QXZ]);
    Vec3 qyJ           = Vec3(particleJ.quadrupole[QXY], particleJ.quadrupole[QYY], particleJ.quadrupole[QYZ]);
    Vec3 qzJ           = Vec3(particleJ.quadrupole[QXZ], particleJ.quadrupole[QYZ], particleJ.quadrupole[QZZ]);

    Vec3 qj            = Vec3(qxJ.dot(deltaR), qyJ.dot(deltaR), qzJ.dot(deltaR));
    double qjr         = qj.dot(deltaR);

    Vec3 fim           = qj*(2.0*bn2)  - particleJ.dipole*bn1  - deltaR*(bn1*particleJ.charge - bn2*djr+bn3*qjr);
    Vec3 fjm           = qi*(-2.0*bn2)  - particleI.dipole*bn1  + deltaR*(bn1*particleI.charge + bn2*dir+bn3*qir);


    Vec3 fid           = qj*(2.0*drr5) - particleJ.dipole*drr3 - deltaR*(drr3*particleJ.charge - drr5*djr+drr7*qjr);
    Vec3 fjd           = qi*(-2.0*drr5) - particleI.dipole*drr3 + deltaR*(drr3*particleI.charge + drr5*dir+drr7*qir);

    // Octopole terms
    Vec3 oxxI        = Vec3(particleI.octopole[QXXX], particleI.octopole[QXXY], particleI.octopole[QXXZ]);
    Vec3 oxyI        = Vec3(particleI.octopole[QXXY], particleI.octopole[QXYY], particleI.octopole[QXYZ]);
    Vec3 oxzI        = Vec3(particleI.octopole[QXXZ], particleI.octopole[QXYZ], particleI.octopole[QXZZ]);
    Vec3 oyyI        = Vec3(particleI.octopole[QXYY], particleI.octopole[QYYY], particleI.octopole[QYYZ]);
    Vec3 oyzI        = Vec3(particleI.octopole[QXYZ], particleI.octopole[QYYZ], particleI.octopole[QYZZ]);
    Vec3 ozzI        = Vec3(particleI.octopole[QXZZ], particleI.octopole[QYZZ], particleI.octopole[QZZZ]);
    Vec3 oxI         = Vec3(oxxI.dot(deltaR), oxyI.dot(deltaR), oxzI.dot(deltaR));
    Vec3 oyI         = Vec3(oxyI.dot(deltaR), oyyI.dot(deltaR), oyzI.dot(deltaR));
    Vec3 ozI         = Vec3(oxzI.dot(deltaR), oyzI.dot(deltaR), ozzI.dot(deltaR));
    Vec3 oI          = Vec3(oxI.dot(deltaR), oyI.dot(deltaR), ozI.dot(deltaR));
    Vec3 oxxJ        = Vec3(particleJ.octopole[QXXX], particleJ.octopole[QXXY], particleJ.octopole[QXXZ]);
    Vec3 oxyJ        = Vec3(particleJ.octopole[QXXY], particleJ.octopole[QXYY], particleJ.octopole[QXYZ]);
    Vec3 oxzJ        = Vec3(particleJ.octopole[QXXZ], particleJ.octopole[QXYZ], particleJ.octopole[QXZZ]);
    Vec3 oyyJ        = Vec3(particleJ.octopole[QXYY], particleJ.octopole[QYYY], particleJ.octopole[QYYZ]);
    Vec3 oyzJ        = Vec3(particleJ.octopole[QXYZ], particleJ.octopole[QYYZ], particleJ.octopole[QYZZ]);
    Vec3 ozzJ        = Vec3(particleJ.octopole[QXZZ], particleJ.octopole[QYZZ], particleJ.octopole[QZZZ]);
    Vec3 oxJ         = Vec3(oxxJ.dot(deltaR), oxyJ.dot(deltaR), oxzJ.dot(deltaR));
    Vec3 oyJ         = Vec3(oxyJ.dot(deltaR), oyyJ.dot(deltaR), oyzJ.dot(deltaR));
    Vec3 ozJ         = Vec3(oxzJ.dot(deltaR), oyzJ.dot(deltaR), ozzJ.dot(deltaR));
    Vec3 oJ          = Vec3(oxJ.dot(deltaR), oyJ.dot(deltaR), ozJ.dot(deltaR));

    fim += -oJ*(3.0*bn3) + deltaR*bn4*oJ.dot(deltaR);
    fjm += -oI*(3.0*bn3) + deltaR*bn4*oI.dot(deltaR);
    fid += -oJ*(3.0*drr7) + deltaR*drr9*oJ.dot(deltaR);
    fjd += -oI*(3.0*drr7) + deltaR*drr9*oI.dot(deltaR);
    // increment the field at each site due to this interaction
    _fixedMultipoleField[iIndex] += fim - fid;
    _fixedMultipoleField[jIndex] += fjm - fjd;

}

void MPIDReferencePmeForce::calculateFixedMultipoleField(const vector<MultipoleParticleData>& particleData)
{

    // first calculate reciprocal space fixed multipole fields

    resizePmeArrays();
    computeMPIDBsplines(particleData);
    initializePmeGrid();
    spreadFixedMultipolesOntoGrid(particleData);
    fftpack_exec_3d(_fftplan, FFTPACK_FORWARD, _pmeGrid, _pmeGrid);
    performMPIDReciprocalConvolution();
    fftpack_exec_3d(_fftplan, FFTPACK_BACKWARD, _pmeGrid, _pmeGrid);
    computeFixedPotentialFromGrid();
    recordFixedMultipoleField();

    // include self-energy portion of the multipole field
    // and initialize _fixedMultipoleFieldPolar to _fixedMultipoleField

    double term = (4.0/3.0)*(_alphaEwald*_alphaEwald*_alphaEwald)/SQRT_PI;
    for (unsigned int jj = 0; jj < _numParticles; jj++) {
        Vec3 selfEnergy = particleData[jj].dipole*term;
        _fixedMultipoleField[jj] += selfEnergy;
    }

    // include direct space fixed multipole fields

    this->MPIDReferenceForce::calculateFixedMultipoleField(particleData);
}

#define ARRAY(x,y) array[(x)-1+((y)-1)*MPID_PME_ORDER]

/**
 * This is called from computeBsplines().  It calculates the spline coefficients for a single atom along a single axis.
 */
void MPIDReferencePmeForce::computeBSplinePoint(vector<double5>& thetai, double w)
{

    double array[MPID_PME_ORDER*MPID_PME_ORDER];

    // initialization to get to 2nd order recursion

    ARRAY(2,2) = w;
    ARRAY(2,1) = 1.0 - w;

    // perform one pass to get to 3rd order recursion

    ARRAY(3,3) = 0.5 * w * ARRAY(2,2);
    ARRAY(3,2) = 0.5 * ((1.0+w)*ARRAY(2,1)+(2.0-w)*ARRAY(2,2));
    ARRAY(3,1) = 0.5 * (1.0-w) * ARRAY(2,1);

    // compute standard B-spline recursion to desired order

    for (int i = 4; i <= MPID_PME_ORDER; i++) {
        int k = i - 1;
        double denom = 1.0 / k;
        ARRAY(i,i) = denom * w * ARRAY(k,k);
        for (int j = 1; j <= i-2; j++)
            ARRAY(i,i-j) = denom * ((w+j)*ARRAY(k,i-j-1)+(i-j-w)*ARRAY(k,i-j));
        ARRAY(i,1) = denom * (1.0-w) * ARRAY(k,1);
    }

    // get coefficients for the B-spline first derivative

    int k = MPID_PME_ORDER - 1;
    ARRAY(k,MPID_PME_ORDER) = ARRAY(k,MPID_PME_ORDER-1);
    for (int i = MPID_PME_ORDER-1; i >= 2; i--)
        ARRAY(k,i) = ARRAY(k,i-1) - ARRAY(k,i);
    ARRAY(k,1) = -ARRAY(k,1);

    // get coefficients for the B-spline second derivative

    k = MPID_PME_ORDER - 2;
    ARRAY(k,MPID_PME_ORDER-1) = ARRAY(k,MPID_PME_ORDER-2);
    for (int i = MPID_PME_ORDER-2; i >= 2; i--)
        ARRAY(k,i) = ARRAY(k,i-1) - ARRAY(k,i);
    ARRAY(k,1) = -ARRAY(k,1);
    ARRAY(k,MPID_PME_ORDER) = ARRAY(k,MPID_PME_ORDER-1);
    for (int i = MPID_PME_ORDER-1; i >= 2; i--)
        ARRAY(k,i) = ARRAY(k,i-1) - ARRAY(k,i);
    ARRAY(k,1) = -ARRAY(k,1);

    // get coefficients for the B-spline third derivative

    k = MPID_PME_ORDER - 3;
    ARRAY(k,MPID_PME_ORDER-2) = ARRAY(k,MPID_PME_ORDER-3);
    for (int i = MPID_PME_ORDER-3; i >= 2; i--)
        ARRAY(k,i) = ARRAY(k,i-1) - ARRAY(k,i);
    ARRAY(k,1) = -ARRAY(k,1);
    ARRAY(k,MPID_PME_ORDER-1) = ARRAY(k,MPID_PME_ORDER-2);
    for (int i = MPID_PME_ORDER-2; i >= 2; i--)
        ARRAY(k,i) = ARRAY(k,i-1) - ARRAY(k,i);
    ARRAY(k,1) = -ARRAY(k,1);
    ARRAY(k,MPID_PME_ORDER) = ARRAY(k,MPID_PME_ORDER-1);
    for (int i = MPID_PME_ORDER-1; i >= 2; i--)
        ARRAY(k,i) = ARRAY(k,i-1) - ARRAY(k,i);
    ARRAY(k,1) = -ARRAY(k,1);


    // get coefficients for the B-spline fourth derivative

    k = MPID_PME_ORDER - 4;
    ARRAY(k,MPID_PME_ORDER-3) = ARRAY(k,MPID_PME_ORDER-4);
    for (int i = MPID_PME_ORDER-4; i >= 2; i--)
        ARRAY(k,i) = ARRAY(k,i-1) - ARRAY(k,i);
    ARRAY(k,1) = -ARRAY(k,1);
    ARRAY(k,MPID_PME_ORDER-2) = ARRAY(k,MPID_PME_ORDER-3);
    for (int i = MPID_PME_ORDER-3; i >= 2; i--)
        ARRAY(k,i) = ARRAY(k,i-1) - ARRAY(k,i);
    ARRAY(k,1) = -ARRAY(k,1);
    ARRAY(k,MPID_PME_ORDER-1) = ARRAY(k,MPID_PME_ORDER-2);
    for (int i = MPID_PME_ORDER-2; i >= 2; i--)
        ARRAY(k,i) = ARRAY(k,i-1) - ARRAY(k,i);
    ARRAY(k,1) = -ARRAY(k,1);
    ARRAY(k,MPID_PME_ORDER) = ARRAY(k,MPID_PME_ORDER-1);
    for (int i = MPID_PME_ORDER-1; i >= 2; i--)
        ARRAY(k,i) = ARRAY(k,i-1) - ARRAY(k,i);
    ARRAY(k,1) = -ARRAY(k,1);

    // copy coefficients from temporary to permanent storage

    for (int i = 1; i <= MPID_PME_ORDER; i++)
        thetai[i-1] = double5(ARRAY(MPID_PME_ORDER,i), ARRAY(MPID_PME_ORDER-1,i), ARRAY(MPID_PME_ORDER-2,i), ARRAY(MPID_PME_ORDER-3,i), ARRAY(MPID_PME_ORDER-4, i));
}

/**
 * Compute b-spline coefficients.
 */
void MPIDReferencePmeForce::computeMPIDBsplines(const vector<MultipoleParticleData>& particleData)
{
    //  get the B-spline coefficients for each multipole site

    for (unsigned int ii = 0; ii < _numParticles; ii++) {
        Vec3 position  = particleData[ii].position;
        getPeriodicDelta(position);
        IntVec igrid;
        for (unsigned int jj = 0; jj < 3; jj++) {

            double w  = position[0]*_recipBoxVectors[0][jj]+position[1]*_recipBoxVectors[1][jj]+position[2]*_recipBoxVectors[2][jj];
            double fr = _pmeGridDimensions[jj]*(w-(int)(w+0.5)+0.5);
            int ifr   = static_cast<int>(floor(fr));
            w         = fr - ifr;
            igrid[jj] = ifr - MPID_PME_ORDER + 1;
            igrid[jj] += igrid[jj] < 0 ? _pmeGridDimensions[jj] : 0;
            vector<double5> thetaiTemp(MPID_PME_ORDER);
            computeBSplinePoint(thetaiTemp, w);
            for (unsigned int kk = 0; kk < MPID_PME_ORDER; kk++)
                _thetai[jj][ii*MPID_PME_ORDER+kk] = thetaiTemp[kk];
        }

        // Record the grid point.

        _iGrid[ii]               = igrid;
    }
}

void MPIDReferencePmeForce::transformMultipolesToFractionalCoordinates(const vector<MultipoleParticleData>& particleData) {
    // Build matrices for transforming the dipoles and quadrupoles.

    Vec3 a[3];
    for (int i = 0; i < 3; i++)
        for (int j = 0; j < 3; j++)
            a[j][i] = _pmeGridDimensions[j]*_recipBoxVectors[i][j];
    int index1[] = {0, 0, 0, 1, 1, 2};
    int index2[] = {0, 1, 2, 1, 2, 2};
    double b[6][6];
    for (int i = 0; i < 6; i++) {
        for (int j = 0; j < 6; j++) {
            b[i][j] = a[index1[i]][index1[j]]*a[index2[i]][index2[j]];
            if (index1[i] != index2[i])
                b[i][j] += a[index1[i]][index2[j]]*a[index2[i]][index1[j]];
        }
    }

    // Transform the multipoles.

    _transformed.resize(particleData.size());
    double quadScale[] = {1, 2, 2, 1, 2, 1};
    for (int i = 0; i < (int) particleData.size(); i++) {
        _transformed[i].charge = particleData[i].charge;
        _transformed[i].dipole = Vec3();
        for (int j = 0; j < 3; j++)
            for (int k = 0; k < 3; k++)
                _transformed[i].dipole[j] += a[j][k]*particleData[i].dipole[k];
        for (int j = 0; j < 6; j++) {
            _transformed[i].quadrupole[j] = 0;
            for (int k = 0; k < 6; k++)
                _transformed[i].quadrupole[j] += quadScale[k]*b[j][k]*particleData[i].quadrupole[k];
        }
    }

    for (int i = 0; i < (int) particleData.size(); i++) {
        double T1XXX = a[0][0]*particleData[i].octopole[QXXX] + a[0][1]*particleData[i].octopole[QXXY] + a[0][2]*particleData[i].octopole[QXXZ];
        double T1XXY = a[1][0]*particleData[i].octopole[QXXX] + a[1][1]*particleData[i].octopole[QXXY] + a[1][2]*particleData[i].octopole[QXXZ];
        double T1XXZ = a[2][0]*particleData[i].octopole[QXXX] + a[2][1]*particleData[i].octopole[QXXY] + a[2][2]*particleData[i].octopole[QXXZ];
        double T1XYX = a[0][0]*particleData[i].octopole[QXXY] + a[0][1]*particleData[i].octopole[QXYY] + a[0][2]*particleData[i].octopole[QXYZ];
        double T1XYY = a[1][0]*particleData[i].octopole[QXXY] + a[1][1]*particleData[i].octopole[QXYY] + a[1][2]*particleData[i].octopole[QXYZ];
        double T1XYZ = a[2][0]*particleData[i].octopole[QXXY] + a[2][1]*particleData[i].octopole[QXYY] + a[2][2]*particleData[i].octopole[QXYZ];
        double T1XZX = a[0][0]*particleData[i].octopole[QXXZ] + a[0][1]*particleData[i].octopole[QXYZ] + a[0][2]*particleData[i].octopole[QXZZ];
        double T1XZY = a[1][0]*particleData[i].octopole[QXXZ] + a[1][1]*particleData[i].octopole[QXYZ] + a[1][2]*particleData[i].octopole[QXZZ];
        double T1XZZ = a[2][0]*particleData[i].octopole[QXXZ] + a[2][1]*particleData[i].octopole[QXYZ] + a[2][2]*particleData[i].octopole[QXZZ];
        double T1YXX = a[0][0]*particleData[i].octopole[QXXY] + a[0][1]*particleData[i].octopole[QXYY] + a[0][2]*particleData[i].octopole[QXYZ];
        double T1YXY = a[1][0]*particleData[i].octopole[QXXY] + a[1][1]*particleData[i].octopole[QXYY] + a[1][2]*particleData[i].octopole[QXYZ];
        double T1YXZ = a[2][0]*particleData[i].octopole[QXXY] + a[2][1]*particleData[i].octopole[QXYY] + a[2][2]*particleData[i].octopole[QXYZ];
        double T1YYX = a[0][0]*particleData[i].octopole[QXYY] + a[0][1]*particleData[i].octopole[QYYY] + a[0][2]*particleData[i].octopole[QYYZ];
        double T1YYY = a[1][0]*particleData[i].octopole[QXYY] + a[1][1]*particleData[i].octopole[QYYY] + a[1][2]*particleData[i].octopole[QYYZ];
        double T1YYZ = a[2][0]*particleData[i].octopole[QXYY] + a[2][1]*particleData[i].octopole[QYYY] + a[2][2]*particleData[i].octopole[QYYZ];
        double T1YZX = a[0][0]*particleData[i].octopole[QXYZ] + a[0][1]*particleData[i].octopole[QYYZ] + a[0][2]*particleData[i].octopole[QYZZ];
        double T1YZY = a[1][0]*particleData[i].octopole[QXYZ] + a[1][1]*particleData[i].octopole[QYYZ] + a[1][2]*particleData[i].octopole[QYZZ];
        double T1YZZ = a[2][0]*particleData[i].octopole[QXYZ] + a[2][1]*particleData[i].octopole[QYYZ] + a[2][2]*particleData[i].octopole[QYZZ];
        double T1ZXX = a[0][0]*particleData[i].octopole[QXXZ] + a[0][1]*particleData[i].octopole[QXYZ] + a[0][2]*particleData[i].octopole[QXZZ];
        double T1ZXY = a[1][0]*particleData[i].octopole[QXXZ] + a[1][1]*particleData[i].octopole[QXYZ] + a[1][2]*particleData[i].octopole[QXZZ];
        double T1ZXZ = a[2][0]*particleData[i].octopole[QXXZ] + a[2][1]*particleData[i].octopole[QXYZ] + a[2][2]*particleData[i].octopole[QXZZ];
        double T1ZYX = a[0][0]*particleData[i].octopole[QXYZ] + a[0][1]*particleData[i].octopole[QYYZ] + a[0][2]*particleData[i].octopole[QYZZ];
        double T1ZYY = a[1][0]*particleData[i].octopole[QXYZ] + a[1][1]*particleData[i].octopole[QYYZ] + a[1][2]*particleData[i].octopole[QYZZ];
        double T1ZYZ = a[2][0]*particleData[i].octopole[QXYZ] + a[2][1]*particleData[i].octopole[QYYZ] + a[2][2]*particleData[i].octopole[QYZZ];
        double T1ZZX = a[0][0]*particleData[i].octopole[QXZZ] + a[0][1]*particleData[i].octopole[QYZZ] + a[0][2]*particleData[i].octopole[QZZZ];
        double T1ZZY = a[1][0]*particleData[i].octopole[QXZZ] + a[1][1]*particleData[i].octopole[QYZZ] + a[1][2]*particleData[i].octopole[QZZZ];
        double T1ZZZ = a[2][0]*particleData[i].octopole[QXZZ] + a[2][1]*particleData[i].octopole[QYZZ] + a[2][2]*particleData[i].octopole[QZZZ];
        double T2XXX = a[0][0]*T1XXX + a[0][1]*T1XYX + a[0][2]*T1XZX;
        double T2XXY = a[0][0]*T1XXY + a[0][1]*T1XYY + a[0][2]*T1XZY;
        double T2XXZ = a[0][0]*T1XXZ + a[0][1]*T1XYZ + a[0][2]*T1XZZ;
        double T2XYY = a[1][0]*T1XXY + a[1][1]*T1XYY + a[1][2]*T1XZY;
        double T2XYZ = a[1][0]*T1XXZ + a[1][1]*T1XYZ + a[1][2]*T1XZZ;
        double T2XZZ = a[2][0]*T1XXZ + a[2][1]*T1XYZ + a[2][2]*T1XZZ;
        double T2YXX = a[0][0]*T1YXX + a[0][1]*T1YYX + a[0][2]*T1YZX;
        double T2YXY = a[0][0]*T1YXY + a[0][1]*T1YYY + a[0][2]*T1YZY;
        double T2YXZ = a[0][0]*T1YXZ + a[0][1]*T1YYZ + a[0][2]*T1YZZ;
        double T2YYY = a[1][0]*T1YXY + a[1][1]*T1YYY + a[1][2]*T1YZY;
        double T2YYZ = a[1][0]*T1YXZ + a[1][1]*T1YYZ + a[1][2]*T1YZZ;
        double T2YZZ = a[2][0]*T1YXZ + a[2][1]*T1YYZ + a[2][2]*T1YZZ;
        double T2ZXX = a[0][0]*T1ZXX + a[0][1]*T1ZYX + a[0][2]*T1ZZX;
        double T2ZXY = a[0][0]*T1ZXY + a[0][1]*T1ZYY + a[0][2]*T1ZZY;
        double T2ZXZ = a[0][0]*T1ZXZ + a[0][1]*T1ZYZ + a[0][2]*T1ZZZ;
        double T2ZYY = a[1][0]*T1ZXY + a[1][1]*T1ZYY + a[1][2]*T1ZZY;
        double T2ZYZ = a[1][0]*T1ZXZ + a[1][1]*T1ZYZ + a[1][2]*T1ZZZ;
        double T2ZZZ = a[2][0]*T1ZXZ + a[2][1]*T1ZYZ + a[2][2]*T1ZZZ;
        _transformed[i].octopole[QXXX] =     (a[0][0]*T2XXX + a[0][1]*T2YXX + a[0][2]*T2ZXX);
        _transformed[i].octopole[QXXY] = 3.0*(a[0][0]*T2XXY + a[0][1]*T2YXY + a[0][2]*T2ZXY);
        _transformed[i].octopole[QXXZ] = 3.0*(a[0][0]*T2XXZ + a[0][1]*T2YXZ + a[0][2]*T2ZXZ);
        _transformed[i].octopole[QXYY] = 3.0*(a[0][0]*T2XYY + a[0][1]*T2YYY + a[0][2]*T2ZYY);
        _transformed[i].octopole[QXYZ] = 6.0*(a[0][0]*T2XYZ + a[0][1]*T2YYZ + a[0][2]*T2ZYZ);
        _transformed[i].octopole[QXZZ] = 3.0*(a[0][0]*T2XZZ + a[0][1]*T2YZZ + a[0][2]*T2ZZZ);
        _transformed[i].octopole[QYYY] =     (a[1][0]*T2XYY + a[1][1]*T2YYY + a[1][2]*T2ZYY);
        _transformed[i].octopole[QYYZ] = 3.0*(a[1][0]*T2XYZ + a[1][1]*T2YYZ + a[1][2]*T2ZYZ);
        _transformed[i].octopole[QYZZ] = 3.0*(a[1][0]*T2XZZ + a[1][1]*T2YZZ + a[1][2]*T2ZZZ);
        _transformed[i].octopole[QZZZ] =     (a[2][0]*T2XZZ + a[2][1]*T2YZZ + a[2][2]*T2ZZZ);
    }
}

void MPIDReferencePmeForce::transformPotentialToCartesianCoordinates(const vector<double>& fphi, vector<double>& cphi) const {
    // Build a matrix for transforming the potential.

    Vec3 a[3];
    for (int i = 0; i < 3; i++)
        for (int j = 0; j < 3; j++)
            a[i][j] = _pmeGridDimensions[j]*_recipBoxVectors[i][j];
    int index1[] = {0, 1, 2, 0, 0, 1};
    int index2[] = {0, 1, 2, 1, 2, 2};
    double b[6][6];
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 6; j++) {
            b[i][j] = a[index1[i]][index1[j]]*a[index2[i]][index2[j]];
            if (index1[j] != index2[j])
                b[i][j] *= 2;
        }
    }
    for (int i = 3; i < 6; i++) {
        for (int j = 0; j < 6; j++) {
            b[i][j] = a[index1[i]][index1[j]]*a[index2[i]][index2[j]];
            if (index1[j] != index2[j])
                b[i][j] += a[index1[i]][index2[j]]*a[index2[i]][index1[j]];
        }
    }

    // Transform the potential.

    for (int i = 0; i < _numParticles; i++) {
        cphi[20*i] = fphi[35*i];
        cphi[20*i+1] = a[0][0]*fphi[35*i+1] + a[0][1]*fphi[35*i+2] + a[0][2]*fphi[35*i+3];
        cphi[20*i+2] = a[1][0]*fphi[35*i+1] + a[1][1]*fphi[35*i+2] + a[1][2]*fphi[35*i+3];
        cphi[20*i+3] = a[2][0]*fphi[35*i+1] + a[2][1]*fphi[35*i+2] + a[2][2]*fphi[35*i+3];
        for (int j = 0; j < 6; j++) {
            cphi[20*i+4+j] = 0;
            for (int k = 0; k < 6; k++)
                cphi[20*i+4+j] += b[j][k]*fphi[35*i+4+k];
        }
    }

    for (int i = 0; i < _numParticles; i++) {
        double T1XXX = a[0][0]*fphi[35*i+10] + a[0][1]*fphi[35*i+13] + a[0][2]*fphi[35*i+14];
        double T1XXY = a[1][0]*fphi[35*i+10] + a[1][1]*fphi[35*i+13] + a[1][2]*fphi[35*i+14];
        double T1XXZ = a[2][0]*fphi[35*i+10] + a[2][1]*fphi[35*i+13] + a[2][2]*fphi[35*i+14];
        double T1XYX = a[0][0]*fphi[35*i+13] + a[0][1]*fphi[35*i+15] + a[0][2]*fphi[35*i+19];
        double T1XYY = a[1][0]*fphi[35*i+13] + a[1][1]*fphi[35*i+15] + a[1][2]*fphi[35*i+19];
        double T1XYZ = a[2][0]*fphi[35*i+13] + a[2][1]*fphi[35*i+15] + a[2][2]*fphi[35*i+19];
        double T1XZX = a[0][0]*fphi[35*i+14] + a[0][1]*fphi[35*i+19] + a[0][2]*fphi[35*i+17];
        double T1XZY = a[1][0]*fphi[35*i+14] + a[1][1]*fphi[35*i+19] + a[1][2]*fphi[35*i+17];
        double T1XZZ = a[2][0]*fphi[35*i+14] + a[2][1]*fphi[35*i+19] + a[2][2]*fphi[35*i+17];
        double T1YXX = a[0][0]*fphi[35*i+13] + a[0][1]*fphi[35*i+15] + a[0][2]*fphi[35*i+19];
        double T1YXY = a[1][0]*fphi[35*i+13] + a[1][1]*fphi[35*i+15] + a[1][2]*fphi[35*i+19];
        double T1YXZ = a[2][0]*fphi[35*i+13] + a[2][1]*fphi[35*i+15] + a[2][2]*fphi[35*i+19];
        double T1YYX = a[0][0]*fphi[35*i+15] + a[0][1]*fphi[35*i+11] + a[0][2]*fphi[35*i+16];
        double T1YYY = a[1][0]*fphi[35*i+15] + a[1][1]*fphi[35*i+11] + a[1][2]*fphi[35*i+16];
        double T1YYZ = a[2][0]*fphi[35*i+15] + a[2][1]*fphi[35*i+11] + a[2][2]*fphi[35*i+16];
        double T1YZX = a[0][0]*fphi[35*i+19] + a[0][1]*fphi[35*i+16] + a[0][2]*fphi[35*i+18];
        double T1YZY = a[1][0]*fphi[35*i+19] + a[1][1]*fphi[35*i+16] + a[1][2]*fphi[35*i+18];
        double T1YZZ = a[2][0]*fphi[35*i+19] + a[2][1]*fphi[35*i+16] + a[2][2]*fphi[35*i+18];
        double T1ZXX = a[0][0]*fphi[35*i+14] + a[0][1]*fphi[35*i+19] + a[0][2]*fphi[35*i+17];
        double T1ZXY = a[1][0]*fphi[35*i+14] + a[1][1]*fphi[35*i+19] + a[1][2]*fphi[35*i+17];
        double T1ZXZ = a[2][0]*fphi[35*i+14] + a[2][1]*fphi[35*i+19] + a[2][2]*fphi[35*i+17];
        double T1ZYX = a[0][0]*fphi[35*i+19] + a[0][1]*fphi[35*i+16] + a[0][2]*fphi[35*i+18];
        double T1ZYY = a[1][0]*fphi[35*i+19] + a[1][1]*fphi[35*i+16] + a[1][2]*fphi[35*i+18];
        double T1ZYZ = a[2][0]*fphi[35*i+19] + a[2][1]*fphi[35*i+16] + a[2][2]*fphi[35*i+18];
        double T1ZZX = a[0][0]*fphi[35*i+17] + a[0][1]*fphi[35*i+18] + a[0][2]*fphi[35*i+12];
        double T1ZZY = a[1][0]*fphi[35*i+17] + a[1][1]*fphi[35*i+18] + a[1][2]*fphi[35*i+12];
        double T1ZZZ = a[2][0]*fphi[35*i+17] + a[2][1]*fphi[35*i+18] + a[2][2]*fphi[35*i+12];
        double T2XXX = a[0][0]*T1XXX + a[0][1]*T1XYX + a[0][2]*T1XZX;
        double T2XXY = a[0][0]*T1XXY + a[0][1]*T1XYY + a[0][2]*T1XZY;
        double T2XXZ = a[0][0]*T1XXZ + a[0][1]*T1XYZ + a[0][2]*T1XZZ;
        double T2XYY = a[1][0]*T1XXY + a[1][1]*T1XYY + a[1][2]*T1XZY;
        double T2XYZ = a[1][0]*T1XXZ + a[1][1]*T1XYZ + a[1][2]*T1XZZ;
        double T2XZZ = a[2][0]*T1XXZ + a[2][1]*T1XYZ + a[2][2]*T1XZZ;
        double T2YXX = a[0][0]*T1YXX + a[0][1]*T1YYX + a[0][2]*T1YZX;
        double T2YXY = a[0][0]*T1YXY + a[0][1]*T1YYY + a[0][2]*T1YZY;
        double T2YXZ = a[0][0]*T1YXZ + a[0][1]*T1YYZ + a[0][2]*T1YZZ;
        double T2YYY = a[1][0]*T1YXY + a[1][1]*T1YYY + a[1][2]*T1YZY;
        double T2YYZ = a[1][0]*T1YXZ + a[1][1]*T1YYZ + a[1][2]*T1YZZ;
        double T2YZZ = a[2][0]*T1YXZ + a[2][1]*T1YYZ + a[2][2]*T1YZZ;
        double T2ZXX = a[0][0]*T1ZXX + a[0][1]*T1ZYX + a[0][2]*T1ZZX;
        double T2ZXY = a[0][0]*T1ZXY + a[0][1]*T1ZYY + a[0][2]*T1ZZY;
        double T2ZXZ = a[0][0]*T1ZXZ + a[0][1]*T1ZYZ + a[0][2]*T1ZZZ;
        double T2ZYY = a[1][0]*T1ZXY + a[1][1]*T1ZYY + a[1][2]*T1ZZY;
        double T2ZYZ = a[1][0]*T1ZXZ + a[1][1]*T1ZYZ + a[1][2]*T1ZZZ;
        double T2ZZZ = a[2][0]*T1ZXZ + a[2][1]*T1ZYZ + a[2][2]*T1ZZZ;
        cphi[20*i+10+QXXX] = a[0][0]*T2XXX + a[0][1]*T2YXX + a[0][2]*T2ZXX;
        cphi[20*i+10+QXXY] = a[0][0]*T2XXY + a[0][1]*T2YXY + a[0][2]*T2ZXY;
        cphi[20*i+10+QXXZ] = a[0][0]*T2XXZ + a[0][1]*T2YXZ + a[0][2]*T2ZXZ;
        cphi[20*i+10+QXYY] = a[0][0]*T2XYY + a[0][1]*T2YYY + a[0][2]*T2ZYY;
        cphi[20*i+10+QXYZ] = a[0][0]*T2XYZ + a[0][1]*T2YYZ + a[0][2]*T2ZYZ;
        cphi[20*i+10+QXZZ] = a[0][0]*T2XZZ + a[0][1]*T2YZZ + a[0][2]*T2ZZZ;
        cphi[20*i+10+QYYY] = a[1][0]*T2XYY + a[1][1]*T2YYY + a[1][2]*T2ZYY;
        cphi[20*i+10+QYYZ] = a[1][0]*T2XYZ + a[1][1]*T2YYZ + a[1][2]*T2ZYZ;
        cphi[20*i+10+QYZZ] = a[1][0]*T2XZZ + a[1][1]*T2YZZ + a[1][2]*T2ZZZ;
        cphi[20*i+10+QZZZ] = a[2][0]*T2XZZ + a[2][1]*T2YZZ + a[2][2]*T2ZZZ;
    }
}

void MPIDReferencePmeForce::spreadFixedMultipolesOntoGrid(const vector<MultipoleParticleData>& particleData)
{

    transformMultipolesToFractionalCoordinates(particleData);

    // Clear the grid.

    for (int gridIndex = 0; gridIndex < _totalGridSize; gridIndex++)
        _pmeGrid[gridIndex] = t_complex(0, 0);

    // Loop over atoms and spread them on the grid.

    for (int atomIndex = 0; atomIndex < _numParticles; atomIndex++) {
        double atomCharge = _transformed[atomIndex].charge;
        Vec3 atomDipole = Vec3(_transformed[atomIndex].dipole[0],
                               _transformed[atomIndex].dipole[1],
                               _transformed[atomIndex].dipole[2]);

        double atomQuadrupoleXX = _transformed[atomIndex].quadrupole[QXX];
        double atomQuadrupoleXY = _transformed[atomIndex].quadrupole[QXY];
        double atomQuadrupoleXZ = _transformed[atomIndex].quadrupole[QXZ];
        double atomQuadrupoleYY = _transformed[atomIndex].quadrupole[QYY];
        double atomQuadrupoleYZ = _transformed[atomIndex].quadrupole[QYZ];
        double atomQuadrupoleZZ = _transformed[atomIndex].quadrupole[QZZ];
        double atomOctopoleXXX  = _transformed[atomIndex].octopole[QXXX];
        double atomOctopoleXXY  = _transformed[atomIndex].octopole[QXXY];
        double atomOctopoleXXZ  = _transformed[atomIndex].octopole[QXXZ];
        double atomOctopoleXYY  = _transformed[atomIndex].octopole[QXYY];
        double atomOctopoleXYZ  = _transformed[atomIndex].octopole[QXYZ];
        double atomOctopoleXZZ  = _transformed[atomIndex].octopole[QXZZ];
        double atomOctopoleYYY  = _transformed[atomIndex].octopole[QYYY];
        double atomOctopoleYYZ  = _transformed[atomIndex].octopole[QYYZ];
        double atomOctopoleYZZ  = _transformed[atomIndex].octopole[QYZZ];
        double atomOctopoleZZZ  = _transformed[atomIndex].octopole[QZZZ];
        IntVec& gridPoint = _iGrid[atomIndex];
        for (int ix = 0; ix < MPID_PME_ORDER; ix++) {
            int x = (gridPoint[0]+ix) % _pmeGridDimensions[0];
            for (int iy = 0; iy < MPID_PME_ORDER; iy++) {
                int y = (gridPoint[1]+iy) % _pmeGridDimensions[1];
                for (int iz = 0; iz < MPID_PME_ORDER; iz++) {
                    int z = (gridPoint[2]+iz) % _pmeGridDimensions[2];
                    double5 t = _thetai[0][atomIndex*MPID_PME_ORDER+ix];
                    double5 u = _thetai[1][atomIndex*MPID_PME_ORDER+iy];
                    double5 v = _thetai[2][atomIndex*MPID_PME_ORDER+iz];
                    double term0 = atomCharge*u[0]*v[0] + atomDipole[1]*u[1]*v[0] + atomDipole[2]*u[0]*v[1]
                                 + atomQuadrupoleYY*u[2]*v[0] + atomQuadrupoleZZ*u[0]*v[2] + atomQuadrupoleYZ*u[1]*v[1]
                                 + atomOctopoleYYY*u[3]*v[0] + atomOctopoleYYZ*u[2]*v[1] + atomOctopoleYZZ*u[1]*v[2] + atomOctopoleZZZ*u[0]*v[3];
                    double term1 = atomDipole[0]*u[0]*v[0] + atomQuadrupoleXY*u[1]*v[0] + atomQuadrupoleXZ*u[0]*v[1]
                                 + atomOctopoleXYY*u[2]*v[0] + atomOctopoleXYZ*u[1]*v[1] + atomOctopoleXZZ*u[0]*v[2];
                    double term2 = atomQuadrupoleXX * u[0] * v[0]
                                 + atomOctopoleXXY*u[1]*v[0] + atomOctopoleXXZ*u[0]*v[1];
                    double term3 = atomOctopoleXXX*u[0]*v[0];
                    t_complex& gridValue = _pmeGrid[x*_pmeGridDimensions[1]*_pmeGridDimensions[2]+y*_pmeGridDimensions[2]+z];
                    gridValue.re += term0*t[0] + term1*t[1] + term2*t[2] + term3*t[3];
                }
            }
        }
    }
}

void MPIDReferencePmeForce::performMPIDReciprocalConvolution()
{

    double expFactor   = (M_PI*M_PI)/(_alphaEwald*_alphaEwald);
    double scaleFactor = 1.0/(M_PI*_periodicBoxVectors[0][0]*_periodicBoxVectors[1][1]*_periodicBoxVectors[2][2]);

    for (int index = 0; index < _totalGridSize; index++)
    {
        int kx = index/(_pmeGridDimensions[1]*_pmeGridDimensions[2]);
        int remainder = index-kx*_pmeGridDimensions[1]*_pmeGridDimensions[2];
        int ky = remainder/_pmeGridDimensions[2];
        int kz = remainder-ky*_pmeGridDimensions[2];

        if (kx == 0 && ky == 0 && kz == 0) {
            _pmeGrid[index].re = _pmeGrid[index].im = 0.0;
            continue;
        }

        int mx = (kx < (_pmeGridDimensions[0]+1)/2) ? kx : (kx-_pmeGridDimensions[0]);
        int my = (ky < (_pmeGridDimensions[1]+1)/2) ? ky : (ky-_pmeGridDimensions[1]);
        int mz = (kz < (_pmeGridDimensions[2]+1)/2) ? kz : (kz-_pmeGridDimensions[2]);

        double mhx = mx*_recipBoxVectors[0][0];
        double mhy = mx*_recipBoxVectors[1][0]+my*_recipBoxVectors[1][1];
        double mhz = mx*_recipBoxVectors[2][0]+my*_recipBoxVectors[2][1]+mz*_recipBoxVectors[2][2];

        double bx = _pmeBsplineModuli[0][kx];
        double by = _pmeBsplineModuli[1][ky];
        double bz = _pmeBsplineModuli[2][kz];

        double m2 = mhx*mhx+mhy*mhy+mhz*mhz;
        double denom = m2*bx*by*bz;
        double eterm = scaleFactor*exp(-expFactor*m2)/denom;

        _pmeGrid[index].re *= eterm;
        _pmeGrid[index].im *= eterm;
    }
}

void MPIDReferencePmeForce::computeFixedPotentialFromGrid()
{
    // extract the permanent multipole field at each site

    for (int m = 0; m < _numParticles; m++) {
        IntVec gridPoint = _iGrid[m];
        double tuv000 = 0.0;
        double tuv001 = 0.0;
        double tuv010 = 0.0;
        double tuv100 = 0.0;
        double tuv200 = 0.0;
        double tuv020 = 0.0;
        double tuv002 = 0.0;
        double tuv110 = 0.0;
        double tuv101 = 0.0;
        double tuv011 = 0.0;
        double tuv300 = 0.0;
        double tuv030 = 0.0;
        double tuv003 = 0.0;
        double tuv210 = 0.0;
        double tuv201 = 0.0;
        double tuv120 = 0.0;
        double tuv021 = 0.0;
        double tuv102 = 0.0;
        double tuv012 = 0.0;
        double tuv111 = 0.0;
        double tuv400 = 0.0;
        double tuv040 = 0.0;
        double tuv004 = 0.0;
        double tuv310 = 0.0;
        double tuv301 = 0.0;
        double tuv130 = 0.0;
        double tuv031 = 0.0;
        double tuv103 = 0.0;
        double tuv013 = 0.0;
        double tuv220 = 0.0;
        double tuv202 = 0.0;
        double tuv022 = 0.0;
        double tuv211 = 0.0;
        double tuv121 = 0.0;
        double tuv112 = 0.0;
        for (int iz = 0; iz < MPID_PME_ORDER; iz++) {
            int k = gridPoint[2]+iz-(gridPoint[2]+iz >= _pmeGridDimensions[2] ? _pmeGridDimensions[2] : 0);
            double5 v = _thetai[2][m*MPID_PME_ORDER+iz];
            double tu00 = 0.0;
            double tu10 = 0.0;
            double tu01 = 0.0;
            double tu20 = 0.0;
            double tu11 = 0.0;
            double tu02 = 0.0;
            double tu30 = 0.0;
            double tu21 = 0.0;
            double tu12 = 0.0;
            double tu03 = 0.0;
            double tu40 = 0.0;
            double tu04 = 0.0;
            double tu31 = 0.0;
            double tu13 = 0.0;
            double tu22 = 0.0;
            for (int iy = 0; iy < MPID_PME_ORDER; iy++) {
                int j = gridPoint[1]+iy-(gridPoint[1]+iy >= _pmeGridDimensions[1] ? _pmeGridDimensions[1] : 0);
                double5 u = _thetai[1][m*MPID_PME_ORDER+iy];
                double5 t = double5(0.0, 0.0, 0.0, 0.0, 0.0);
                for (int ix = 0; ix < MPID_PME_ORDER; ix++) {
                    int i = gridPoint[0]+ix-(gridPoint[0]+ix >= _pmeGridDimensions[0] ? _pmeGridDimensions[0] : 0);
                    int gridIndex = i*_pmeGridDimensions[1]*_pmeGridDimensions[2] + j*_pmeGridDimensions[2] + k;
                    double tq = _pmeGrid[gridIndex].re;
                    double5 tadd = _thetai[0][m*MPID_PME_ORDER+ix];
                    t[0] += tq*tadd[0];
                    t[1] += tq*tadd[1];
                    t[2] += tq*tadd[2];
                    t[3] += tq*tadd[3];
                    t[4] += tq*tadd[4];
                }
                tu00 += t[0]*u[0];
                tu10 += t[1]*u[0];
                tu01 += t[0]*u[1];
                tu20 += t[2]*u[0];
                tu11 += t[1]*u[1];
                tu02 += t[0]*u[2];
                tu30 += t[3]*u[0];
                tu21 += t[2]*u[1];
                tu12 += t[1]*u[2];
                tu03 += t[0]*u[3];
                tu40 += t[4]*u[0];
                tu04 += t[0]*u[4];
                tu31 += t[3]*u[1];
                tu13 += t[1]*u[3];
                tu22 += t[2]*u[2];
            }
            tuv000 += tu00*v[0];
            tuv100 += tu10*v[0];
            tuv010 += tu01*v[0];
            tuv001 += tu00*v[1];
            tuv200 += tu20*v[0];
            tuv020 += tu02*v[0];
            tuv002 += tu00*v[2];
            tuv110 += tu11*v[0];
            tuv101 += tu10*v[1];
            tuv011 += tu01*v[1];
            tuv300 += tu30*v[0];
            tuv030 += tu03*v[0];
            tuv003 += tu00*v[3];
            tuv210 += tu21*v[0];
            tuv201 += tu20*v[1];
            tuv120 += tu12*v[0];
            tuv021 += tu02*v[1];
            tuv102 += tu10*v[2];
            tuv012 += tu01*v[2];
            tuv111 += tu11*v[1];
            tuv400 += tu40*v[0];
            tuv040 += tu04*v[0];
            tuv004 += tu00*v[4];
            tuv310 += tu31*v[0];
            tuv301 += tu30*v[1];
            tuv130 += tu13*v[0];
            tuv031 += tu03*v[1];
            tuv103 += tu10*v[3];
            tuv013 += tu01*v[3];
            tuv220 += tu22*v[0];
            tuv202 += tu20*v[2];
            tuv022 += tu02*v[2];
            tuv211 += tu21*v[1];
            tuv121 += tu12*v[1];
            tuv112 += tu11*v[2];
        }
        _phi[35*m] = tuv000;
        _phi[35*m+1] = tuv100;
        _phi[35*m+2] = tuv010;
        _phi[35*m+3] = tuv001;
        _phi[35*m+4] = tuv200;
        _phi[35*m+5] = tuv020;
        _phi[35*m+6] = tuv002;
        _phi[35*m+7] = tuv110;
        _phi[35*m+8] = tuv101;
        _phi[35*m+9] = tuv011;
        _phi[35*m+10] = tuv300;
        _phi[35*m+11] = tuv030;
        _phi[35*m+12] = tuv003;
        _phi[35*m+13] = tuv210;
        _phi[35*m+14] = tuv201;
        _phi[35*m+15] = tuv120;
        _phi[35*m+16] = tuv021;
        _phi[35*m+17] = tuv102;
        _phi[35*m+18] = tuv012;
        _phi[35*m+19] = tuv111;
        _phi[35*m+20] = tuv400;
        _phi[35*m+21] = tuv040;
        _phi[35*m+22] = tuv004;
        _phi[35*m+23] = tuv310;
        _phi[35*m+24] = tuv301;
        _phi[35*m+25] = tuv130;
        _phi[35*m+26] = tuv031;
        _phi[35*m+27] = tuv103;
        _phi[35*m+28] = tuv013;
        _phi[35*m+29] = tuv220;
        _phi[35*m+30] = tuv202;
        _phi[35*m+31] = tuv022;
        _phi[35*m+32] = tuv211;
        _phi[35*m+33] = tuv121;
        _phi[35*m+34] = tuv112;
    }
}

void MPIDReferencePmeForce::spreadInducedDipolesOnGrid(const vector<Vec3>& inputInducedDipole) {
    // Create the matrix to convert from Cartesian to fractional coordinates.

    Vec3 cartToFrac[3];
    for (int i = 0; i < 3; i++)
        for (int j = 0; j < 3; j++)
            cartToFrac[j][i] = _pmeGridDimensions[j]*_recipBoxVectors[i][j];

    // Clear the grid.

    for (int gridIndex = 0; gridIndex < _totalGridSize; gridIndex++)
        _pmeGrid[gridIndex] = t_complex(0, 0);

    // Loop over atoms and spread them on the grid.

    for (int atomIndex = 0; atomIndex < _numParticles; atomIndex++) {
        Vec3 inducedDipole = Vec3(inputInducedDipole[atomIndex][0]*cartToFrac[0][0] + inputInducedDipole[atomIndex][1]*cartToFrac[0][1] + inputInducedDipole[atomIndex][2]*cartToFrac[0][2],
                                  inputInducedDipole[atomIndex][0]*cartToFrac[1][0] + inputInducedDipole[atomIndex][1]*cartToFrac[1][1] + inputInducedDipole[atomIndex][2]*cartToFrac[1][2],
                                  inputInducedDipole[atomIndex][0]*cartToFrac[2][0] + inputInducedDipole[atomIndex][1]*cartToFrac[2][1] + inputInducedDipole[atomIndex][2]*cartToFrac[2][2]);
        IntVec& gridPoint = _iGrid[atomIndex];
        for (int ix = 0; ix < MPID_PME_ORDER; ix++) {
            int x = (gridPoint[0]+ix) % _pmeGridDimensions[0];
            for (int iy = 0; iy < MPID_PME_ORDER; iy++) {
                int y = (gridPoint[1]+iy) % _pmeGridDimensions[1];
                for (int iz = 0; iz < MPID_PME_ORDER; iz++) {
                    int z = (gridPoint[2]+iz) % _pmeGridDimensions[2];

                    double5 t = _thetai[0][atomIndex*MPID_PME_ORDER+ix];
                    double5 u = _thetai[1][atomIndex*MPID_PME_ORDER+iy];
                    double5 v = _thetai[2][atomIndex*MPID_PME_ORDER+iz];

                    double term01 = inducedDipole[1]*u[1]*v[0] + inducedDipole[2]*u[0]*v[1];
                    double term11 = inducedDipole[0]*u[0]*v[0];

                    t_complex& gridValue = _pmeGrid[x*_pmeGridDimensions[1]*_pmeGridDimensions[2]+y*_pmeGridDimensions[2]+z];
                    gridValue.re += term01*t[0] + term11*t[1];
                    gridValue.im = 0.0;
                }
            }
        }
    }
}

void MPIDReferencePmeForce::computeInducedPotentialFromGrid()
{
    // extract the induced dipole field at each site

    for (int m = 0; m < _numParticles; m++) {
        IntVec gridPoint = _iGrid[m];
        double tuv000 = 0.0;
        double tuv001 = 0.0;
        double tuv010 = 0.0;
        double tuv100 = 0.0;
        double tuv200 = 0.0;
        double tuv020 = 0.0;
        double tuv002 = 0.0;
        double tuv110 = 0.0;
        double tuv101 = 0.0;
        double tuv011 = 0.0;
        double tuv300 = 0.0;
        double tuv030 = 0.0;
        double tuv003 = 0.0;
        double tuv210 = 0.0;
        double tuv201 = 0.0;
        double tuv120 = 0.0;
        double tuv021 = 0.0;
        double tuv102 = 0.0;
        double tuv012 = 0.0;
        double tuv111 = 0.0;
        double tuv400 = 0.0;
        double tuv040 = 0.0;
        double tuv004 = 0.0;
        double tuv310 = 0.0;
        double tuv301 = 0.0;
        double tuv130 = 0.0;
        double tuv031 = 0.0;
        double tuv103 = 0.0;
        double tuv013 = 0.0;
        double tuv220 = 0.0;
        double tuv202 = 0.0;
        double tuv022 = 0.0;
        double tuv211 = 0.0;
        double tuv121 = 0.0;
        double tuv112 = 0.0;
        for (int iz = 0; iz < MPID_PME_ORDER; iz++) {
            int k = gridPoint[2]+iz-(gridPoint[2]+iz >= _pmeGridDimensions[2] ? _pmeGridDimensions[2] : 0);
            double5 v = _thetai[2][m*MPID_PME_ORDER+iz];
            double tu00 = 0.0;
            double tu10 = 0.0;
            double tu01 = 0.0;
            double tu20 = 0.0;
            double tu11 = 0.0;
            double tu02 = 0.0;
            double tu30 = 0.0;
            double tu21 = 0.0;
            double tu12 = 0.0;
            double tu03 = 0.0;
            double tu40 = 0.0;
            double tu04 = 0.0;
            double tu31 = 0.0;
            double tu13 = 0.0;
            double tu22 = 0.0;
            for (int iy = 0; iy < MPID_PME_ORDER; iy++) {
                int j = gridPoint[1]+iy-(gridPoint[1]+iy >= _pmeGridDimensions[1] ? _pmeGridDimensions[1] : 0);
                double5 u = _thetai[1][m*MPID_PME_ORDER+iy];
                double5 t = double5(0.0, 0.0, 0.0, 0.0, 0.0);
                for (int ix = 0; ix < MPID_PME_ORDER; ix++) {
                    int i = gridPoint[0]+ix-(gridPoint[0]+ix >= _pmeGridDimensions[0] ? _pmeGridDimensions[0] : 0);
                    int gridIndex = i*_pmeGridDimensions[1]*_pmeGridDimensions[2] + j*_pmeGridDimensions[2] + k;
                    double tq = _pmeGrid[gridIndex].re;
                    double5 tadd = _thetai[0][m*MPID_PME_ORDER+ix];
                    t[0] += tq*tadd[0];
                    t[1] += tq*tadd[1];
                    t[2] += tq*tadd[2];
                    t[3] += tq*tadd[3];
                    t[4] += tq*tadd[4];
                }
                tu00 += t[0]*u[0];
                tu10 += t[1]*u[0];
                tu01 += t[0]*u[1];
                tu20 += t[2]*u[0];
                tu11 += t[1]*u[1];
                tu02 += t[0]*u[2];
                tu30 += t[3]*u[0];
                tu21 += t[2]*u[1];
                tu12 += t[1]*u[2];
                tu03 += t[0]*u[3];
                tu40 += t[4]*u[0];
                tu04 += t[0]*u[4];
                tu31 += t[3]*u[1];
                tu13 += t[1]*u[3];
                tu22 += t[2]*u[2];
            }
            tuv000 += tu00*v[0];
            tuv100 += tu10*v[0];
            tuv010 += tu01*v[0];
            tuv001 += tu00*v[1];
            tuv200 += tu20*v[0];
            tuv020 += tu02*v[0];
            tuv002 += tu00*v[2];
            tuv110 += tu11*v[0];
            tuv101 += tu10*v[1];
            tuv011 += tu01*v[1];
            tuv300 += tu30*v[0];
            tuv030 += tu03*v[0];
            tuv003 += tu00*v[3];
            tuv210 += tu21*v[0];
            tuv201 += tu20*v[1];
            tuv120 += tu12*v[0];
            tuv021 += tu02*v[1];
            tuv102 += tu10*v[2];
            tuv012 += tu01*v[2];
            tuv111 += tu11*v[1];
            tuv400 += tu40*v[0];
            tuv040 += tu04*v[0];
            tuv004 += tu00*v[4];
            tuv310 += tu31*v[0];
            tuv301 += tu30*v[1];
            tuv130 += tu13*v[0];
            tuv031 += tu03*v[1];
            tuv103 += tu10*v[3];
            tuv013 += tu01*v[3];
            tuv220 += tu22*v[0];
            tuv202 += tu20*v[2];
            tuv022 += tu02*v[2];
            tuv211 += tu21*v[1];
            tuv121 += tu12*v[1];
            tuv112 += tu11*v[2];
        }
        _phidp[35*m] = tuv000;
        _phidp[35*m+1] = tuv100;
        _phidp[35*m+2] = tuv010;
        _phidp[35*m+3] = tuv001;
        _phidp[35*m+4] = tuv200;
        _phidp[35*m+5] = tuv020;
        _phidp[35*m+6] = tuv002;
        _phidp[35*m+7] = tuv110;
        _phidp[35*m+8] = tuv101;
        _phidp[35*m+9] = tuv011;
        _phidp[35*m+10] = tuv300;
        _phidp[35*m+11] = tuv030;
        _phidp[35*m+12] = tuv003;
        _phidp[35*m+13] = tuv210;
        _phidp[35*m+14] = tuv201;
        _phidp[35*m+15] = tuv120;
        _phidp[35*m+16] = tuv021;
        _phidp[35*m+17] = tuv102;
        _phidp[35*m+18] = tuv012;
        _phidp[35*m+19] = tuv111;
        _phidp[35*m+20] = tuv400;
        _phidp[35*m+21] = tuv040;
        _phidp[35*m+22] = tuv004;
        _phidp[35*m+23] = tuv310;
        _phidp[35*m+24] = tuv301;
        _phidp[35*m+25] = tuv130;
        _phidp[35*m+26] = tuv031;
        _phidp[35*m+27] = tuv103;
        _phidp[35*m+28] = tuv013;
        _phidp[35*m+29] = tuv220;
        _phidp[35*m+30] = tuv202;
        _phidp[35*m+31] = tuv022;
        _phidp[35*m+32] = tuv211;
        _phidp[35*m+33] = tuv121;
        _phidp[35*m+34] = tuv112;
    }
}

double MPIDReferencePmeForce::computeReciprocalSpaceFixedMultipoleForceAndEnergy(const vector<MultipoleParticleData>& particleData,
                                                                                            vector<Vec3>& forces, vector<Vec3>& torques) const
{
    double multipole[20]; //                                  XXX XXY XXZ XYY XYZ XZZ YYY YYZ YZZ ZZZ
    const int deriv0[] = {0, 1, 2, 3,  4,  5,  6,  7,  8,  9,  10, 13, 14, 15, 19, 17, 11, 16, 18, 12};
    const int deriv1[] = {1, 4, 7, 8, 10, 15, 17, 13, 14, 19,  20, 23, 24, 29, 32, 30, 25, 33, 34, 27};
    const int deriv2[] = {2, 7, 5, 9, 13, 11, 18, 15, 19, 16,  23, 29, 32, 25, 33, 34, 21, 26, 31, 28};
    const int deriv3[] = {3, 8, 9, 6, 14, 16, 12, 19, 17, 18,  24, 32, 30, 33, 34, 27, 26, 31, 28, 22};
    vector<double> cphi(20*_numParticles);
    transformPotentialToCartesianCoordinates(_phi, cphi);
    Vec3 fracToCart[3];
    for (int i = 0; i < 3; i++)
        for (int j = 0; j < 3; j++)
            fracToCart[i][j] = _pmeGridDimensions[j]*_recipBoxVectors[i][j];
    double energy = 0.0;
    for (int i = 0; i < _numParticles; i++) {

        // Compute the torque.

        multipole[0] = particleData[i].charge;

        multipole[1] =  particleData[i].dipole[0];
        multipole[2] =  particleData[i].dipole[1];
        multipole[3] =  particleData[i].dipole[2];
        if(particleData[i].isAnisotropic){
            multipole[1] += _inducedDipole[i][0];
            multipole[2] += _inducedDipole[i][1];
            multipole[3] += _inducedDipole[i][2];
        }

        multipole[4] = particleData[i].quadrupole[QXX];
        multipole[5] = particleData[i].quadrupole[QYY];
        multipole[6] = particleData[i].quadrupole[QZZ];
        multipole[7] = particleData[i].quadrupole[QXY]*2.0;
        multipole[8] = particleData[i].quadrupole[QXZ]*2.0;
        multipole[9] = particleData[i].quadrupole[QYZ]*2.0;

        multipole[10] = particleData[i].octopole[QXXX];
        multipole[11] = particleData[i].octopole[QXXY]*3.0;
        multipole[12] = particleData[i].octopole[QXXZ]*3.0;
        multipole[13] = particleData[i].octopole[QXYY]*3.0;
        multipole[14] = particleData[i].octopole[QXYZ]*6.0;
        multipole[15] = particleData[i].octopole[QXZZ]*3.0;
        multipole[16] = particleData[i].octopole[QYYY];
        multipole[17] = particleData[i].octopole[QYYZ]*3.0;
        multipole[18] = particleData[i].octopole[QYZZ]*3.0;
        multipole[19] = particleData[i].octopole[QZZZ];

        const double* phi = &cphi[20*i];
        torques[i][0] += _electric*(multipole[3]*phi[2] - multipole[2]*phi[3]
                      + 2.0*(multipole[6]-multipole[5])*phi[9]
                      + multipole[8]*phi[7] + multipole[9]*phi[5]
                      - multipole[7]*phi[8] - multipole[9]*phi[6]
                      + phi[11]*(4.0*multipole[12] - multipole[17] - 3.0*multipole[19])/5.0
                      + phi[13]*multipole[14]
                      + phi[16]*(-multipole[12] + 4.0*multipole[17] - 3.0*multipole[19])/5.0
                      + phi[12]*(-4.0*multipole[11] + 3.0*multipole[16] + multipole[18])/5.0
                      + phi[14]*(-2.0*multipole[13] + 2.0*multipole[15])
                      + phi[17]*(multipole[11] - 12.0*multipole[16] + 11.0*multipole[18])/5.0
                      - phi[15]*multipole[14]
                      + phi[18]*(-multipole[12] - 11.0*multipole[17] + 12.0*multipole[19])/5.0
                      + phi[19]*(multipole[11] + 3.0*multipole[16] - 4.0*multipole[18])/5.0);

        torques[i][1] += _electric*(multipole[1]*phi[3] - multipole[3]*phi[1]
                      + 2.0*(multipole[4]-multipole[6])*phi[8]
                      + multipole[7]*phi[9] + multipole[8]*phi[6]
                      - multipole[8]*phi[4] - multipole[9]*phi[7]
                      + phi[10]*(-4.0*multipole[12] + multipole[17] + 3.0*multipole[19])/5.0
                      - phi[11]*multipole[14]
                      + phi[13]*(multipole[12] - 4.0*multipole[17] + 3.0*multipole[19])/5.0
                      + phi[12]*(12.0*multipole[10] - multipole[13] - 11.0*multipole[15])/5.0
                      + phi[14]*2.0*(multipole[11] - multipole[18])
                      + phi[17]*(-3.0*multipole[10] + 4.0*multipole[13] - multipole[15])/5.0
                      + phi[15]*(11.0*multipole[12] + multipole[17] - 12.0*multipole[19])/5.0
                      + phi[18]*multipole[14]
                      + phi[19]*(-3.0*multipole[10] - multipole[13] + 4.0*multipole[15])/5.0);

        torques[i][2] += _electric*(multipole[2]*phi[1] - multipole[1]*phi[2]
                      + 2.0*(multipole[5]-multipole[4])*phi[7]
                      + multipole[7]*phi[4] + multipole[9]*phi[8]
                      - multipole[7]*phi[5] - multipole[8]*phi[9]
                      + phi[10]*(4.0*multipole[11] - 3.0*multipole[16] - multipole[18])/5.0
                      + phi[11]*(-12.0*multipole[10] + 11.0*multipole[13] + multipole[15])/5.0
                      + phi[13]*(-11.0*multipole[11] + 12.0*multipole[16] - multipole[18])/5.0
                      + phi[16]*(3.0*multipole[10] - 4.0*multipole[13] + multipole[15])/5.0
                      + phi[12]*multipole[14]
                      + phi[14]*2.0*(-multipole[12] + multipole[17])
                      - phi[17]*multipole[14]
                      + phi[15]*(-multipole[11] - 3.0*multipole[16] + 4.0*multipole[18])/5.0
                      + phi[18]*(3.0*multipole[10] + multipole[13] - 4.0*multipole[15])/5.0);

        // Compute the force and energy.

        multipole[1] = _transformed[i].dipole[0];
        multipole[2] = _transformed[i].dipole[1];
        multipole[3] = _transformed[i].dipole[2];
        multipole[4] = _transformed[i].quadrupole[QXX];
        multipole[5] = _transformed[i].quadrupole[QYY];
        multipole[6] = _transformed[i].quadrupole[QZZ];
        multipole[7] = _transformed[i].quadrupole[QXY];
        multipole[8] = _transformed[i].quadrupole[QXZ];
        multipole[9] = _transformed[i].quadrupole[QYZ];
        multipole[10] = _transformed[i].octopole[QXXX];
        multipole[11] = _transformed[i].octopole[QXXY];
        multipole[12] = _transformed[i].octopole[QXXZ];
        multipole[13] = _transformed[i].octopole[QXYY];
        multipole[14] = _transformed[i].octopole[QXYZ];
        multipole[15] = _transformed[i].octopole[QXZZ];
        multipole[16] = _transformed[i].octopole[QYYY];
        multipole[17] = _transformed[i].octopole[QYYZ];
        multipole[18] = _transformed[i].octopole[QYZZ];
        multipole[19] = _transformed[i].octopole[QZZZ];

        Vec3 f = Vec3(0.0, 0.0, 0.0);
        for (int k = 0; k < 20; k++) {
            energy += multipole[k]*_phi[35*i+deriv0[k]];
            f[0]   += multipole[k]*_phi[35*i+deriv1[k]];
            f[1]   += multipole[k]*_phi[35*i+deriv2[k]];
            f[2]   += multipole[k]*_phi[35*i+deriv3[k]];
        }
        f              *= (_electric);
        forces[i]      -= Vec3(f[0]*fracToCart[0][0] + f[1]*fracToCart[0][1] + f[2]*fracToCart[0][2],
                               f[0]*fracToCart[1][0] + f[1]*fracToCart[1][1] + f[2]*fracToCart[1][2],
                               f[0]*fracToCart[2][0] + f[1]*fracToCart[2][1] + f[2]*fracToCart[2][2]);

    }
    return (0.5*_electric*energy);
}

/**
 * Compute the forces due to the reciprocal space PME calculation for induced dipoles.
 */
double MPIDReferencePmeForce::computeReciprocalSpaceInducedDipoleForceAndEnergy(MPIDReferenceForce::PolarizationType polarizationType,
                                                                                           const vector<MultipoleParticleData>& particleData,
                                                                                           vector<Vec3>& forces, vector<Vec3>& torques) const
{
    double inducedDipole[3];
    double multipole[20]; //                                  XXX XXY XXZ XYY XYZ XZZ YYY YYZ YZZ ZZZ
    const int deriv1[] = {1, 4, 7, 8, 10, 15, 17, 13, 14, 19,  20, 23, 24, 29, 32, 30, 25, 33, 34, 27};
    const int deriv2[] = {2, 7, 5, 9, 13, 11, 18, 15, 19, 16,  23, 29, 32, 25, 33, 34, 21, 26, 31, 28};
    const int deriv3[] = {3, 8, 9, 6, 14, 16, 12, 19, 17, 18,  24, 32, 30, 33, 34, 27, 26, 31, 28, 22};
    vector<double> cphi(20*_numParticles);
    transformPotentialToCartesianCoordinates(_phidp, cphi);
    Vec3 cartToFrac[3], fracToCart[3];
    for (int i = 0; i < 3; i++)
        for (int j = 0; j < 3; j++)
            cartToFrac[j][i] = fracToCart[i][j] = _pmeGridDimensions[j]*_recipBoxVectors[i][j];
    double energy = 0.0;
    for (int i = 0; i < _numParticles; i++) {

        // Compute the torque.

        unsigned int iIndex = particleData[i].particleIndex;

        multipole[0] = particleData[i].charge;

        multipole[1] = particleData[i].dipole[0];
        multipole[2] = particleData[i].dipole[1];
        multipole[3] = particleData[i].dipole[2];
        if (polarizationType == MPIDReferenceForce::Mutual && particleData[i].isAnisotropic){
            multipole[1] += _inducedDipole[i][0];
            multipole[2] += _inducedDipole[i][1];
            multipole[3] += _inducedDipole[i][2];
        }
        multipole[4] = particleData[i].quadrupole[QXX];
        multipole[5] = particleData[i].quadrupole[QYY];
        multipole[6] = particleData[i].quadrupole[QZZ];
        multipole[7] = particleData[i].quadrupole[QXY]*2.0;
        multipole[8] = particleData[i].quadrupole[QXZ]*2.0;
        multipole[9] = particleData[i].quadrupole[QYZ]*2.0;

        multipole[10] = particleData[i].octopole[QXXX];
        multipole[11] = particleData[i].octopole[QXXY]*3.0;
        multipole[12] = particleData[i].octopole[QXXZ]*3.0;
        multipole[13] = particleData[i].octopole[QXYY]*3.0;
        multipole[14] = particleData[i].octopole[QXYZ]*6.0;
        multipole[15] = particleData[i].octopole[QXZZ]*3.0;
        multipole[16] = particleData[i].octopole[QYYY];
        multipole[17] = particleData[i].octopole[QYYZ]*3.0;
        multipole[18] = particleData[i].octopole[QYZZ]*3.0;
        multipole[19] = particleData[i].octopole[QZZZ];

        const double* phi = &cphi[20*i];
        torques[iIndex][0] += _electric*(multipole[3]*phi[2] - multipole[2]*phi[3]
                      + 2.0*(multipole[6]-multipole[5])*phi[9]
                      + multipole[8]*phi[7] + multipole[9]*phi[5]
                      - multipole[7]*phi[8] - multipole[9]*phi[6]
                      + phi[11]*(4.0*multipole[12] - multipole[17] - 3.0*multipole[19])/5.0
                      + phi[13]*multipole[14]
                      + phi[16]*(-multipole[12] + 4.0*multipole[17] - 3.0*multipole[19])/5.0
                      + phi[12]*(-4.0*multipole[11] + 3.0*multipole[16] + multipole[18])/5.0
                      + phi[14]*(-2.0*multipole[13] + 2.0*multipole[15])
                      + phi[17]*(multipole[11] - 12.0*multipole[16] + 11.0*multipole[18])/5.0
                      - phi[15]*multipole[14]
                      + phi[18]*(-multipole[12] - 11.0*multipole[17] + 12.0*multipole[19])/5.0
                      + phi[19]*(multipole[11] + 3.0*multipole[16] - 4.0*multipole[18])/5.0);

        torques[iIndex][1] += _electric*(multipole[1]*phi[3] - multipole[3]*phi[1]
                      + 2.0*(multipole[4]-multipole[6])*phi[8]
                      + multipole[7]*phi[9] + multipole[8]*phi[6]
                      - multipole[8]*phi[4] - multipole[9]*phi[7]
                      + phi[10]*(-4.0*multipole[12] + multipole[17] + 3.0*multipole[19])/5.0
                      - phi[11]*multipole[14]
                      + phi[13]*(multipole[12] - 4.0*multipole[17] + 3.0*multipole[19])/5.0
                      + phi[12]*(12.0*multipole[10] - multipole[13] - 11.0*multipole[15])/5.0
                      + phi[14]*2.0*(multipole[11] - multipole[18])
                      + phi[17]*(-3.0*multipole[10] + 4.0*multipole[13] - multipole[15])/5.0
                      + phi[15]*(11.0*multipole[12] + multipole[17] - 12.0*multipole[19])/5.0
                      + phi[18]*multipole[14]
                      + phi[19]*(-3.0*multipole[10] - multipole[13] + 4.0*multipole[15])/5.0);

        torques[iIndex][2] += _electric*(multipole[2]*phi[1] - multipole[1]*phi[2]
                      + 2.0*(multipole[5]-multipole[4])*phi[7]
                      + multipole[7]*phi[4] + multipole[9]*phi[8]
                      - multipole[7]*phi[5] - multipole[8]*phi[9]
                      + phi[10]*(4.0*multipole[11] - 3.0*multipole[16] - multipole[18])/5.0
                      + phi[11]*(-12.0*multipole[10] + 11.0*multipole[13] + multipole[15])/5.0
                      + phi[13]*(-11.0*multipole[11] + 12.0*multipole[16] - multipole[18])/5.0
                      + phi[16]*(3.0*multipole[10] - 4.0*multipole[13] + multipole[15])/5.0
                      + phi[12]*multipole[14]
                      + phi[14]*2.0*(-multipole[12] + multipole[17])
                      - phi[17]*multipole[14]
                      + phi[15]*(-multipole[11] - 3.0*multipole[16] + 4.0*multipole[18])/5.0
                      + phi[18]*(3.0*multipole[10] + multipole[13] - 4.0*multipole[15])/5.0);
        // Compute the force and energy.

        multipole[1] = _transformed[i].dipole[0];
        multipole[2] = _transformed[i].dipole[1];
        multipole[3] = _transformed[i].dipole[2];
        multipole[4] = _transformed[i].quadrupole[QXX];
        multipole[5] = _transformed[i].quadrupole[QYY];
        multipole[6] = _transformed[i].quadrupole[QZZ];
        multipole[7] = _transformed[i].quadrupole[QXY];
        multipole[8] = _transformed[i].quadrupole[QXZ];
        multipole[9] = _transformed[i].quadrupole[QYZ];
        multipole[10] = _transformed[i].octopole[QXXX];
        multipole[11] = _transformed[i].octopole[QXXY];
        multipole[12] = _transformed[i].octopole[QXXZ];
        multipole[13] = _transformed[i].octopole[QXYY];
        multipole[14] = _transformed[i].octopole[QXYZ];
        multipole[15] = _transformed[i].octopole[QXZZ];
        multipole[16] = _transformed[i].octopole[QYYY];
        multipole[17] = _transformed[i].octopole[QYYZ];
        multipole[18] = _transformed[i].octopole[QYZZ];
        multipole[19] = _transformed[i].octopole[QZZZ];

        inducedDipole[0] = _inducedDipole[i][0]*cartToFrac[0][0] + _inducedDipole[i][1]*cartToFrac[0][1] + _inducedDipole[i][2]*cartToFrac[0][2];
        inducedDipole[1] = _inducedDipole[i][0]*cartToFrac[1][0] + _inducedDipole[i][1]*cartToFrac[1][1] + _inducedDipole[i][2]*cartToFrac[1][2];
        inducedDipole[2] = _inducedDipole[i][0]*cartToFrac[2][0] + _inducedDipole[i][1]*cartToFrac[2][1] + _inducedDipole[i][2]*cartToFrac[2][2];

        energy += 2.0*inducedDipole[0]*_phi[35*i+1];
        energy += 2.0*inducedDipole[1]*_phi[35*i+2];
        energy += 2.0*inducedDipole[2]*_phi[35*i+3];

        Vec3 f = Vec3(0.0, 0.0, 0.0);

        for (int k = 0; k < 3; k++) {

            int j1 = deriv1[k+1];
            int j2 = deriv2[k+1];
            int j3 = deriv3[k+1];

            f[0] += 2.0*inducedDipole[k]*_phi[35*i+j1];
            f[1] += 2.0*inducedDipole[k]*_phi[35*i+j2];
            f[2] += 2.0*inducedDipole[k]*_phi[35*i+j3];
 
            if (polarizationType == MPIDReferenceForce::Mutual) {
                f[0] += inducedDipole[k]*2.0*_phidp[35*i+j1];
                f[1] += inducedDipole[k]*2.0*_phidp[35*i+j2];
                f[2] += inducedDipole[k]*2.0*_phidp[35*i+j3];
            }
        }

        for (int k = 0; k < 20; k++) {
            f[0] += multipole[k]*2.0*_phidp[35*i+deriv1[k]];
            f[1] += multipole[k]*2.0*_phidp[35*i+deriv2[k]];
            f[2] += multipole[k]*2.0*_phidp[35*i+deriv3[k]];
        }

        f              *= (0.5*_electric);
        forces[iIndex] -= Vec3(f[0]*fracToCart[0][0] + f[1]*fracToCart[0][1] + f[2]*fracToCart[0][2],
                               f[0]*fracToCart[1][0] + f[1]*fracToCart[1][1] + f[2]*fracToCart[1][2],
                               f[0]*fracToCart[2][0] + f[1]*fracToCart[2][1] + f[2]*fracToCart[2][2]);
    }
    return (0.25*_electric*energy);
}

void MPIDReferencePmeForce::recordFixedMultipoleField()
{
    Vec3 fracToCart[3];
    for (int i = 0; i < 3; i++)
        for (int j = 0; j < 3; j++)
            fracToCart[i][j] = _pmeGridDimensions[j]*_recipBoxVectors[i][j];
    for (int i = 0; i < _numParticles; i++) {
        _fixedMultipoleField[i][0] = -(_phi[35*i+1]*fracToCart[0][0] + _phi[35*i+2]*fracToCart[0][1] + _phi[35*i+3]*fracToCart[0][2]);
        _fixedMultipoleField[i][1] = -(_phi[35*i+1]*fracToCart[1][0] + _phi[35*i+2]*fracToCart[1][1] + _phi[35*i+3]*fracToCart[1][2]);
        _fixedMultipoleField[i][2] = -(_phi[35*i+1]*fracToCart[2][0] + _phi[35*i+2]*fracToCart[2][1] + _phi[35*i+3]*fracToCart[2][2]);
    }
}

void MPIDReferencePmeForce::initializeInducedDipoles(vector<UpdateInducedDipoleFieldStruct>& updateInducedDipoleFields)
{

    this->MPIDReferenceForce::initializeInducedDipoles(updateInducedDipoleFields);
    calculateReciprocalSpaceInducedDipoleField(updateInducedDipoleFields);
}

void MPIDReferencePmeForce::recordInducedDipoleField(vector<Vec3>& field)
{
    Vec3 fracToCart[3];
    for (int i = 0; i < 3; i++)
        for (int j = 0; j < 3; j++)
            fracToCart[i][j] = _pmeGridDimensions[j]*_recipBoxVectors[i][j];
    for (int i = 0; i < _numParticles; i++) {

        field[i][0] -= _phidp[35*i+1]*fracToCart[0][0] + _phidp[35*i+2]*fracToCart[0][1] + _phidp[35*i+3]*fracToCart[0][2];
        field[i][1] -= _phidp[35*i+1]*fracToCart[1][0] + _phidp[35*i+2]*fracToCart[1][1] + _phidp[35*i+3]*fracToCart[1][2];
        field[i][2] -= _phidp[35*i+1]*fracToCart[2][0] + _phidp[35*i+2]*fracToCart[2][1] + _phidp[35*i+3]*fracToCart[2][2];
    }
}

void MPIDReferencePmeForce::calculateReciprocalSpaceInducedDipoleField(vector<UpdateInducedDipoleFieldStruct>& updateInducedDipoleFields)
{
    // Perform PME for the induced dipoles.

    initializePmeGrid();
    spreadInducedDipolesOnGrid(*updateInducedDipoleFields[0].inducedDipoles);
    fftpack_exec_3d(_fftplan, FFTPACK_FORWARD, _pmeGrid, _pmeGrid);
    performMPIDReciprocalConvolution();
    fftpack_exec_3d(_fftplan, FFTPACK_BACKWARD, _pmeGrid, _pmeGrid);
    computeInducedPotentialFromGrid();
    recordInducedDipoleField(updateInducedDipoleFields[0].inducedDipoleField);
}

void MPIDReferencePmeForce::calculateInducedDipoleFields(const vector<MultipoleParticleData>& particleData,
                                                                     vector<UpdateInducedDipoleFieldStruct>& updateInducedDipoleFields)
{
    // Initialize the fields to zero.

    Vec3 zeroVec(0.0, 0.0, 0.0);
    for (auto& field : updateInducedDipoleFields)
        std::fill(field.inducedDipoleField.begin(), field.inducedDipoleField.end(), zeroVec);

    // Add fields from direct space interactions.

    for (unsigned int ii = 0; ii < particleData.size(); ii++) {
        for (unsigned int jj = ii + 1; jj < particleData.size(); jj++) {
            calculateDirectInducedDipolePairIxns(particleData[ii], particleData[jj], updateInducedDipoleFields);
        }
    }

    // reciprocal space ixns

    calculateReciprocalSpaceInducedDipoleField(updateInducedDipoleFields);

    if(getPolarizationType() == MPIDReferenceForce::Extrapolated) {
        // While we have the reciprocal space (fractional coordinate) field gradient available, add it to the real space
        // terms computed above, after transforming to Cartesian coordinates.  This allows real and reciprocal space
        // dipole response force contributions to be computed together.
        Vec3 fracToCart[3];
        for (int i = 0; i < 3; i++)
            for (int j = 0; j < 3; j++)
                fracToCart[i][j] = _pmeGridDimensions[j]*_recipBoxVectors[i][j];


        for (int i = 0; i < _numParticles; i++) {
            double EmatD[3][3] = {
                { _phidp[35*i+4], _phidp[35*i+7], _phidp[35*i+8] },
                { _phidp[35*i+7], _phidp[35*i+5], _phidp[35*i+9] },
                { _phidp[35*i+8], _phidp[35*i+9], _phidp[35*i+6] }
            };

            double Exx = 0.0, Eyy = 0.0, Ezz = 0.0, Exy = 0.0, Exz = 0.0, Eyz = 0.0;
            for (int k = 0; k < 3; ++k) {
                for (int l = 0; l < 3; ++l) {
                    Exx += fracToCart[0][k] * EmatD[k][l] * fracToCart[0][l];
                    Eyy += fracToCart[1][k] * EmatD[k][l] * fracToCart[1][l];
                    Ezz += fracToCart[2][k] * EmatD[k][l] * fracToCart[2][l];
                    Exy += fracToCart[0][k] * EmatD[k][l] * fracToCart[1][l];
                    Exz += fracToCart[0][k] * EmatD[k][l] * fracToCart[2][l];
                    Eyz += fracToCart[1][k] * EmatD[k][l] * fracToCart[2][l];
                }
            }
            updateInducedDipoleFields[0].inducedDipoleFieldGradient[i][0] -= Exx;
            updateInducedDipoleFields[0].inducedDipoleFieldGradient[i][1] -= Eyy;
            updateInducedDipoleFields[0].inducedDipoleFieldGradient[i][2] -= Ezz;
            updateInducedDipoleFields[0].inducedDipoleFieldGradient[i][3] -= Exy;
            updateInducedDipoleFields[0].inducedDipoleFieldGradient[i][4] -= Exz;
            updateInducedDipoleFields[0].inducedDipoleFieldGradient[i][5] -= Eyz;
        }
    }

    // self ixn

    double term = (4.0/3.0)*(_alphaEwald*_alphaEwald*_alphaEwald)/SQRT_PI;
    for (auto& field : updateInducedDipoleFields) {
        vector<Vec3>& inducedDipoles = *field.inducedDipoles;
        vector<Vec3>& inducedDipoleField = field.inducedDipoleField;
        for (unsigned int jj = 0; jj < particleData.size(); jj++) {
            inducedDipoleField[jj] += inducedDipoles[jj]*term;
        }
    }
}

void MPIDReferencePmeForce::calculateDirectInducedDipolePairIxn(unsigned int iIndex, unsigned int jIndex,
                                                                           double preFactor1, double preFactor2,
                                                                           const Vec3& delta,
                                                                           const vector<Vec3>& inducedDipole,
                                                                           vector<Vec3>& field) const
{

    // field at i due induced dipole at j

    double dur  = inducedDipole[jIndex].dot(delta);
    field[iIndex]  += delta*(dur*preFactor2) + inducedDipole[jIndex]*preFactor1;

    // field at j due induced dipole at i

               dur  = inducedDipole[iIndex].dot(delta);
    field[jIndex]  += delta*(dur*preFactor2) + inducedDipole[iIndex]*preFactor1;
}

void MPIDReferencePmeForce::calculateDirectInducedDipolePairIxns(const MultipoleParticleData& particleI,
                                                                 const MultipoleParticleData& particleJ,
                                                                 vector<UpdateInducedDipoleFieldStruct>& updateInducedDipoleFields)
{

    // compute the real space portion of the Ewald summation

    double uscale = 1.0;
    double pscale = getMultipoleScaleFactor(particleI.particleIndex, particleJ.particleIndex, P_SCALE);

    Vec3 deltaR = particleJ.position - particleI.position;

    // periodic boundary conditions

    getPeriodicDelta(deltaR);
    double r2 = deltaR.dot(deltaR);

    if (r2 > _cutoffDistanceSquared)
        return;

    double r           = sqrt(r2);

    // calculate the error function damping terms

    double ralpha      = _alphaEwald*r;

    double bn0         = erfc(ralpha)/r;
    double alsq2       = 2.0*_alphaEwald*_alphaEwald;
    double alsq2n      = 1.0/(SQRT_PI*_alphaEwald);
    double exp2a       = exp(-(ralpha*ralpha));
    alsq2n            *= alsq2;
    double bn1         = (bn0+alsq2n*exp2a)/r2;

    alsq2n            *= alsq2;
    double bn2         = (3.0*bn1+alsq2n*exp2a)/r2;

    alsq2n            *= alsq2;
    double bn3         = (5.0*bn2+alsq2n*exp2a)/r2;

    // compute the error function scaled and unscaled terms

    double scale3      = 1.0;
    double scale5      = 1.0;
    double scale7      = 1.0;
    double damp        = particleI.dampingFactor*particleJ.dampingFactor;
    if (damp != 0.0) {

        double ratio = (r/damp);
        double pgamma  = pscale == 0.0 ? particleI.thole + particleJ.thole : _defaultTholeWidth;
               damp    = pgamma*ratio;
        if (damp < 50.0) {
            double expdamp = exp(-damp);
            scale3 = 1.0 - expdamp*(1.0 + damp + 0.5*damp*damp);
            scale5 = 1.0 - expdamp*(1.0 + damp + 0.5*damp*damp + damp*damp*damp/6.0);
            scale7 = 1.0 - expdamp*(1.0 + damp + 0.5*damp*damp + damp*damp*damp/6.0 + damp*damp*damp*damp/30.0);
        }
    }
    double dsc3        = uscale*scale3;
    double dsc5        = uscale*scale5;
    double dsc7        = uscale*scale7;

    double r3          = (r*r2);
    double r5          = (r3*r2);
    double r7          = (r5*r2);
    double rr3         = (1.0-dsc3)/r3;
    double rr5         = 3.0*(1.0-dsc5)/r5;
    double rr7         = 15.0*(1.0-dsc7)/r7;

    double preFactor1  = rr3 - bn1;
    double preFactor2  = bn2 - rr5;
    double preFactor3  = bn3 - rr7;

    for (auto& field : updateInducedDipoleFields) {
        calculateDirectInducedDipolePairIxn(particleI.particleIndex, particleJ.particleIndex, preFactor1, preFactor2, deltaR,
                                            *field.inducedDipoles, field.inducedDipoleField);
        if (getPolarizationType() == MPIDReferenceForce::Extrapolated) {
            // Compute and store the field gradient for later use.
            double dx = deltaR[0];
            double dy = deltaR[1];
            double dz = deltaR[2];

            OpenMM::Vec3 &dipolesI = (*field.inducedDipoles)[particleI.particleIndex];
            double xDipole = dipolesI[0];
            double yDipole = dipolesI[1];
            double zDipole = dipolesI[2];
            double muDotR = xDipole*dx + yDipole*dy + zDipole*dz;
            double Exx = muDotR*dx*dx*preFactor3 - (2.0*xDipole*dx + muDotR)*preFactor2;
            double Eyy = muDotR*dy*dy*preFactor3 - (2.0*yDipole*dy + muDotR)*preFactor2;
            double Ezz = muDotR*dz*dz*preFactor3 - (2.0*zDipole*dz + muDotR)*preFactor2;
            double Exy = muDotR*dx*dy*preFactor3 - (xDipole*dy + yDipole*dx)*preFactor2;
            double Exz = muDotR*dx*dz*preFactor3 - (xDipole*dz + zDipole*dx)*preFactor2;
            double Eyz = muDotR*dy*dz*preFactor3 - (yDipole*dz + zDipole*dy)*preFactor2;

            field.inducedDipoleFieldGradient[particleJ.particleIndex][0] -= Exx;
            field.inducedDipoleFieldGradient[particleJ.particleIndex][1] -= Eyy;
            field.inducedDipoleFieldGradient[particleJ.particleIndex][2] -= Ezz;
            field.inducedDipoleFieldGradient[particleJ.particleIndex][3] -= Exy;
            field.inducedDipoleFieldGradient[particleJ.particleIndex][4] -= Exz;
            field.inducedDipoleFieldGradient[particleJ.particleIndex][5] -= Eyz;

            OpenMM::Vec3 &dipolesJ = (*field.inducedDipoles)[particleJ.particleIndex];
            xDipole = dipolesJ[0];
            yDipole = dipolesJ[1];
            zDipole = dipolesJ[2];
            muDotR = xDipole*dx + yDipole*dy + zDipole*dz;
            Exx = muDotR*dx*dx*preFactor3 - (2.0*xDipole*dx + muDotR)*preFactor2;
            Eyy = muDotR*dy*dy*preFactor3 - (2.0*yDipole*dy + muDotR)*preFactor2;
            Ezz = muDotR*dz*dz*preFactor3 - (2.0*zDipole*dz + muDotR)*preFactor2;
            Exy = muDotR*dx*dy*preFactor3 - (xDipole*dy + yDipole*dx)*preFactor2;
            Exz = muDotR*dx*dz*preFactor3 - (xDipole*dz + zDipole*dx)*preFactor2;
            Eyz = muDotR*dy*dz*preFactor3 - (yDipole*dz + zDipole*dy)*preFactor2;

            field.inducedDipoleFieldGradient[particleI.particleIndex][0] += Exx;
            field.inducedDipoleFieldGradient[particleI.particleIndex][1] += Eyy;
            field.inducedDipoleFieldGradient[particleI.particleIndex][2] += Ezz;
            field.inducedDipoleFieldGradient[particleI.particleIndex][3] += Exy;
            field.inducedDipoleFieldGradient[particleI.particleIndex][4] += Exz;
            field.inducedDipoleFieldGradient[particleI.particleIndex][5] += Eyz;
        }
    }
}

double MPIDReferencePmeForce::calculatePmeSelfEnergy(const vector<MultipoleParticleData>& particleData) const
{
    double cii = 0.0;
    double dii = 0.0;
    double qii = 0.0;
    double oii = 0.0;
    for (unsigned int ii = 0; ii < _numParticles; ii++) {

        const MultipoleParticleData& particleI = particleData[ii];

        cii += particleI.charge*particleI.charge;

        Vec3 dipole(particleI.sphericalDipole[1], particleI.sphericalDipole[2], particleI.sphericalDipole[0]);
        dii += dipole.dot(dipole + _inducedDipole[ii]);

        qii += (particleI.sphericalQuadrupole[0]*particleI.sphericalQuadrupole[0]
               +particleI.sphericalQuadrupole[1]*particleI.sphericalQuadrupole[1]
               +particleI.sphericalQuadrupole[2]*particleI.sphericalQuadrupole[2]
               +particleI.sphericalQuadrupole[3]*particleI.sphericalQuadrupole[3]
               +particleI.sphericalQuadrupole[4]*particleI.sphericalQuadrupole[4]);

        oii += (particleI.sphericalOctopole[0]*particleI.sphericalOctopole[0]
               +particleI.sphericalOctopole[1]*particleI.sphericalOctopole[1]
               +particleI.sphericalOctopole[2]*particleI.sphericalOctopole[2]
               +particleI.sphericalOctopole[3]*particleI.sphericalOctopole[3]
               +particleI.sphericalOctopole[4]*particleI.sphericalOctopole[4]
               +particleI.sphericalOctopole[5]*particleI.sphericalOctopole[5]
               +particleI.sphericalOctopole[6]*particleI.sphericalOctopole[6]);
    }
    double prefac = -_alphaEwald * _electric / (_dielectric*SQRT_PI);
    double a2 = _alphaEwald * _alphaEwald;
    double a4 = a2*a2;
    double a6 = a4*a2;
    double energy = prefac*(cii + twoThirds*a2*dii + fourOverFifteen*a4*qii + a6*eightOverOneHundredFive*oii);
    return energy;
}

void MPIDReferencePmeForce::calculatePmeSelfTorque(const vector<MultipoleParticleData>& particleData,
                                                              vector<Vec3>& torques) const
{
    double term = (2.0/3.0)*(_electric/_dielectric)*(_alphaEwald*_alphaEwald*_alphaEwald)/SQRT_PI;
    for (unsigned int ii = 0; ii < _numParticles; ii++) {

        const MultipoleParticleData& particleI = particleData[ii];
        if(particleI.isAnisotropic) continue;
        Vec3 ui = _inducedDipole[ii]*2.0;
        Vec3 dipole(particleI.sphericalDipole[1], particleI.sphericalDipole[2], particleI.sphericalDipole[0]);
        Vec3 torque = dipole.cross(ui)*term;
        torques[ii] += torque;
    }
}

double MPIDReferencePmeForce::calculatePmeDirectElectrostaticPairIxn(const MultipoleParticleData& particleI,
                                                                                    const MultipoleParticleData& particleJ,
                                                                                    const vector<double>& scalingFactors,
                                                                                    vector<Vec3>& forces,
                                                                                    vector<Vec3>& torques) const
{

    unsigned int iIndex = particleI.particleIndex;
    unsigned int jIndex = particleJ.particleIndex;

    double energy;
    Vec3 deltaR = particleJ.position - particleI.position;
    getPeriodicDelta(deltaR);
    double r2 = deltaR.dot(deltaR);

    if (r2 > _cutoffDistanceSquared)
        return 0.0;

    double r = sqrt(r2);

    // Start by constructing rotation matrices to put dipoles and
    // quadrupoles into the QI frame, from the lab frame.
    double qiRotationMatrix1[3][3];
    formQIRotationMatrix(particleI.position, particleJ.position, deltaR, r, qiRotationMatrix1);
    double qiRotationMatrix2[5][5];
    buildSphericalQuadrupoleRotationMatrix(qiRotationMatrix1, qiRotationMatrix2);
    double qiRotationMatrix3[7][7];
    buildSphericalOctopoleRotationMatrix(qiRotationMatrix1, qiRotationMatrix2, qiRotationMatrix3);
    // The force rotation matrix rotates the QI forces into the lab
    // frame, and makes sure the result is in {x,y,z} ordering. Its
    // transpose is used to rotate the induced dipoles to the QI frame.
    double forceRotationMatrix[3][3];
    forceRotationMatrix[0][0] = qiRotationMatrix1[1][1];
    forceRotationMatrix[0][1] = qiRotationMatrix1[2][1];
    forceRotationMatrix[0][2] = qiRotationMatrix1[0][1];
    forceRotationMatrix[1][0] = qiRotationMatrix1[1][2];
    forceRotationMatrix[1][1] = qiRotationMatrix1[2][2];
    forceRotationMatrix[1][2] = qiRotationMatrix1[0][2];
    forceRotationMatrix[2][0] = qiRotationMatrix1[1][0];
    forceRotationMatrix[2][1] = qiRotationMatrix1[2][0];
    forceRotationMatrix[2][2] = qiRotationMatrix1[0][0];
    // For efficiency, we go ahead and cache that transposed version
    // now, because we need to do 4 rotations in total (I,J, and p,d).
    // We also fold in the factor of 0.5 needed to average the p and d
    // components.
    double inducedDipoleRotationMatrix[3][3];
    inducedDipoleRotationMatrix[0][0] = 0.5*qiRotationMatrix1[0][1];
    inducedDipoleRotationMatrix[0][1] = 0.5*qiRotationMatrix1[0][2];
    inducedDipoleRotationMatrix[0][2] = 0.5*qiRotationMatrix1[0][0];
    inducedDipoleRotationMatrix[1][0] = 0.5*qiRotationMatrix1[1][1];
    inducedDipoleRotationMatrix[1][1] = 0.5*qiRotationMatrix1[1][2];
    inducedDipoleRotationMatrix[1][2] = 0.5*qiRotationMatrix1[1][0];
    inducedDipoleRotationMatrix[2][0] = 0.5*qiRotationMatrix1[2][1];
    inducedDipoleRotationMatrix[2][1] = 0.5*qiRotationMatrix1[2][2];
    inducedDipoleRotationMatrix[2][2] = 0.5*qiRotationMatrix1[2][0];
    // Rotate the induced dipoles to the QI frame.
    double qiUindI[3], qiUindJ[3];
    for (int ii = 0; ii < 3; ii++) {
        double valID = 0.0;
        double valJD = 0.0;
        for (int jj = 0; jj < 3; jj++) {
            valID += inducedDipoleRotationMatrix[ii][jj] * _inducedDipole[iIndex][jj];
            valJD += inducedDipoleRotationMatrix[ii][jj] * _inducedDipole[jIndex][jj];
        }
        qiUindI[ii] = valID;
        qiUindJ[ii] = valJD;
    }

    // The Qtilde intermediates (QI frame multipoles) for atoms I and J
    double qiQI[16], qiQJ[16];
    // Rotate the permanent multipoles to the QI frame.
    qiQI[0] = particleI.charge;
    qiQJ[0] = particleJ.charge;
    for (int ii = 0; ii < 3; ii++) {
        double valI = 0.0;
        double valJ = 0.0;
        for (int jj = 0; jj < 3; jj++) {
            valI += qiRotationMatrix1[ii][jj] * particleI.sphericalDipole[jj];
            valJ += qiRotationMatrix1[ii][jj] * particleJ.sphericalDipole[jj];
        }
        qiQI[ii+1] = valI;
        qiQJ[ii+1] = valJ;
    }
    for (int ii = 0; ii < 5; ii++) {
        double valI = 0.0;
        double valJ = 0.0;
        for (int jj = 0; jj < 5; jj++) {
            valI += qiRotationMatrix2[ii][jj] * particleI.sphericalQuadrupole[jj];
            valJ += qiRotationMatrix2[ii][jj] * particleJ.sphericalQuadrupole[jj];
        }
        qiQI[ii+4] = valI;
        qiQJ[ii+4] = valJ;
    }
    for (int ii = 0; ii < 7; ii++) {
        double valI = 0.0;
        double valJ = 0.0;
        for (int jj = 0; jj < 7; jj++) {
            valI += qiRotationMatrix3[ii][jj] * particleI.sphericalOctopole[jj];
            valJ += qiRotationMatrix3[ii][jj] * particleJ.sphericalOctopole[jj];
        }
        qiQI[ii+9] = valI;
        qiQJ[ii+9] = valJ;
    }

    // The Qtilde{x,y,z} torque intermediates for atoms I and J, which are used to obtain the torques on the permanent moments.
    // 0    1    2    3    4    5    6    7    8    9   10   11   12   13   14   15
    // q    10  11c  11s   20  21c  21s  22c  22s  30   31c  31s  32c  32s  33c  33s
    double qiQIX[16] = {0.0, qiQI[3], 0.0, -qiQI[1], sqrtThree*qiQI[6], qiQI[8], -sqrtThree*qiQI[4] - qiQI[7], qiQI[6], -qiQI[5],
                        sqrtSix*qiQI[11], sqrtFiveHalves*qiQI[13], -sqrtSix*qiQI[9]-sqrtFiveHalves*qiQI[12],sqrtFiveHalves*qiQI[11]+sqrtThreeHalves*qiQI[15],
                        -sqrtFiveHalves*qiQI[10]-sqrtThreeHalves*qiQI[14], sqrtThreeHalves*qiQI[13], -sqrtThreeHalves*qiQI[12]};
    double qiQIY[16] = {0.0, -qiQI[2], qiQI[1], 0.0, -sqrtThree*qiQI[5], sqrtThree*qiQI[4] - qiQI[7], -qiQI[8], qiQI[5], qiQI[6],
                        -sqrtSix*qiQI[10], sqrtSix*qiQI[9]-sqrtFiveHalves*qiQI[12], -sqrtFiveHalves*qiQI[13], sqrtFiveHalves*qiQI[10]-sqrtThreeHalves*qiQI[14],
                        sqrtFiveHalves*qiQI[11]-sqrtThreeHalves*qiQI[15], sqrtThreeHalves*qiQI[12], sqrtThreeHalves*qiQI[13]};
    double qiQIZ[16] = {0.0, 0.0, -qiQI[3], qiQI[2], 0.0, -qiQI[6], qiQI[5], -2.0*qiQI[8], 2.0*qiQI[7],
                        0.0, -qiQI[11], qiQI[10], -2.0*qiQI[13], 2.0*qiQI[12], -3.0*qiQI[15], 3.0*qiQI[14]};
    double qiQJX[16] = {0.0, qiQJ[3], 0.0, -qiQJ[1], sqrtThree*qiQJ[6], qiQJ[8], -sqrtThree*qiQJ[4] - qiQJ[7], qiQJ[6], -qiQJ[5],
                        sqrtSix*qiQJ[11], sqrtFiveHalves*qiQJ[13], -sqrtSix*qiQJ[9]-sqrtFiveHalves*qiQJ[12],sqrtFiveHalves*qiQJ[11]+sqrtThreeHalves*qiQJ[15],
                        -sqrtFiveHalves*qiQJ[10]-sqrtThreeHalves*qiQJ[14], sqrtThreeHalves*qiQJ[13], -sqrtThreeHalves*qiQJ[12]};
    double qiQJY[16] = {0.0, -qiQJ[2], qiQJ[1], 0.0, -sqrtThree*qiQJ[5], sqrtThree*qiQJ[4] - qiQJ[7], -qiQJ[8], qiQJ[5], qiQJ[6],
                        -sqrtSix*qiQJ[10], sqrtSix*qiQJ[9]-sqrtFiveHalves*qiQJ[12], -sqrtFiveHalves*qiQJ[13], sqrtFiveHalves*qiQJ[10]-sqrtThreeHalves*qiQJ[14],
                        sqrtFiveHalves*qiQJ[11]-sqrtThreeHalves*qiQJ[15], sqrtThreeHalves*qiQJ[12], sqrtThreeHalves*qiQJ[13]};
    double qiQJZ[16] = {0.0, 0.0, -qiQJ[3], qiQJ[2], 0.0, -qiQJ[6], qiQJ[5], -2.0*qiQJ[8], 2.0*qiQJ[7],
                        0.0, -qiQJ[11], qiQJ[10], -2.0*qiQJ[13], 2.0*qiQJ[12], -3.0*qiQJ[15], 3.0*qiQJ[14]};

    // The field derivatives at I due to permanent and induced moments on J, and vice-versa.
    // Also, their derivatives w.r.t. R, which are needed for force calculations
    double Vij[16], Vji[16], VjiR[16], VijR[16];
    // The field derivatives at I due to only permanent moments on J, and vice-versa.
    double Vijd[3], Vjid[3];
    double rInvVec[9], alphaRVec[10], bVec[6];

    double prefac = (_electric/_dielectric);
    double rInv = 1.0 / r;

    // The rInvVec array is defined such that the ith element is R^-i, with the
    // dieleectric constant folded in, to avoid conversions later.
    rInvVec[1] = prefac * rInv;
    for (int i = 2; i < 9; ++i)
        rInvVec[i] = rInvVec[i-1] * rInv;

    // The alpharVec array is defined such that the ith element is (alpha R)^i,
    // where kappa (alpha in OpenMM parlance) is the Ewald attenuation parameter.
    alphaRVec[1] = _alphaEwald * r;
    for (int i = 2; i < 10; ++i)
        alphaRVec[i] = alphaRVec[i-1] * alphaRVec[1];

    double erfAlphaR = erf(alphaRVec[1]);
    double X = 2.0*exp(-alphaRVec[2])/SQRT_PI;
    double mScale = scalingFactors[M_SCALE];
    double pScale = scalingFactors[P_SCALE];
    double dScale = pScale;
    double uScale = 1.0;

    int doubleFactorial = 1, facCount = 1;
    double tmp = alphaRVec[1];
    bVec[1] = -erfAlphaR;
    for (int i=2; i < 6; ++i) {
        bVec[i] = bVec[i-1] + tmp * X / doubleFactorial;
        facCount = facCount + 2;
        doubleFactorial = doubleFactorial * facCount;
        tmp *= 2.0 * alphaRVec[2];
    }

    double dmp = particleI.dampingFactor*particleJ.dampingFactor;
    double a = pScale == 0.0 ? particleI.thole + particleJ.thole : _defaultTholeWidth;
    double u = std::abs(dmp) > 1.0E-5 ? r/dmp : 1E10;
    double au = a*u;
    double expau = au < 50.0 ? exp(-au) : 0.0;
    double au2 = au*au;
    double au3 = au2*au;
    double au4 = au3*au;
    double au5 = au4*au;
    double au6 = au5*au;
    // Thole damping factors for energies
    double thole_c   = 1.0 - expau*(1.0 + au + 0.5*au2);
    double thole_d0  = 1.0 - expau*(1.0 + au + 0.5*au2 + au3/4.0);
    double thole_d1  = 1.0 - expau*(1.0 + au + 0.5*au2);
    double thole_q0  = 1.0 - expau*(1.0 + au + 0.5*au2 + au3/6.0 + au4/18.0);
    double thole_q1  = 1.0 - expau*(1.0 + au + 0.5*au2 + au3/6.0);
    double thole_o0  = 1.0 - expau*(1.0 + au + 0.5*au2 + au3/6.0 + au4/24.0 + au5/120.0);
    double thole_o1  = 1.0 - expau*(1.0 + au + 0.5*au2 + au3/6.0 + au4/30.0);
    // Thole damping factors for derivatives
    double dthole_c  = 1.0 - expau*(1.0 + au + 0.5*au2 + au3/4.0);
    double dthole_d0 = 1.0 - expau*(1.0 + au + 0.5*au2 + au3/6.0 + au4/12.0);
    double dthole_d1 = 1.0 - expau*(1.0 + au + 0.5*au2 + au3/6.0);
    double dthole_q0 = 1.0 - expau*(1.0 + au + 0.5*au2 + au3/6.0 + au4/24.0 + au5/72.0);
    double dthole_q1 = 1.0 - expau*(1.0 + au + 0.5*au2 + au3/6.0 + au4/24.0);
    double dthole_o0 = 1.0 - expau*(1.0 + au + 0.5*au2 + au3/6.0 + au4/24.0 + au5/120.0 + au6/600.0);
    double dthole_o1 = 1.0 - expau*(1.0 + au + 0.5*au2 + au3/6.0 + au4/25.0 + au5/150.0);

    // Now we compute the (attenuated) Coulomb operator and its derivatives, contracted with
    // permanent moments and induced dipoles.  Note that the coefficient of the permanent force
    // terms is half of the expected value; this is because we compute the interaction of I with
    // the sum of induced and permanent moments on J, as well as the interaction of J with I's
    // permanent and induced moments; doing so double counts the permanent-permanent interaction.
    double ePermCoef, dPermCoef, eUindCoef, dUindCoef;

    // C-C terms (m=0)
    ePermCoef = rInvVec[1]*(mScale + bVec[2] - alphaRVec[1]*X);
    dPermCoef = -0.5*(mScale + bVec[2])*rInvVec[2];
    Vij[0]  = ePermCoef*qiQJ[0];
    Vji[0]  = ePermCoef*qiQI[0];
    VijR[0] = dPermCoef*qiQJ[0];
    VjiR[0] = dPermCoef*qiQI[0];

    // C-D and C-Uind terms (m=0)
    ePermCoef = rInvVec[2]*(mScale + bVec[2]);
    eUindCoef = 2.0*rInvVec[2]*(pScale*thole_c + bVec[2]);
    dPermCoef = -rInvVec[3]*(mScale + bVec[2] + alphaRVec[3]*X);
    dUindCoef = -4.0*rInvVec[3]*(dScale*dthole_c + bVec[2] + alphaRVec[3]*X);
    Vij[0]  += -(ePermCoef*qiQJ[1] + eUindCoef*qiUindJ[0]);
    Vji[1]   = -(ePermCoef*qiQI[0]);
    VijR[0] += -(dPermCoef*qiQJ[1] + dUindCoef*qiUindJ[0]);
    VjiR[1]  = -(dPermCoef*qiQI[0]);
    Vjid[0]  = -(eUindCoef*qiQI[0]);
    // D-C and Uind-C terms (m=0)
    Vij[1]   = ePermCoef*qiQJ[0];
    Vji[0]  += ePermCoef*qiQI[1] + eUindCoef*qiUindI[0];
    VijR[1]  = dPermCoef*qiQJ[0];
    VjiR[0] += dPermCoef*qiQI[1] + dUindCoef*qiUindI[0];
    Vijd[0]  = eUindCoef*qiQJ[0];

    // D-D and D-Uind terms (m=0)
    ePermCoef = -twoThirds*rInvVec[3]*(3.0*(mScale + bVec[3]) + alphaRVec[3]*X);
    eUindCoef = -2.0*twoThirds*rInvVec[3]*(3.0*(dScale*thole_d0 + bVec[3]) + alphaRVec[3]*X);
    dPermCoef = rInvVec[4]*(3.0*(mScale + bVec[3]) + 2.*alphaRVec[5]*X);
    dUindCoef = 2.0*rInvVec[4]*(6.0*(dScale*dthole_d0 + bVec[3]) + 4.0*alphaRVec[5]*X);
    Vij[1]  += ePermCoef*qiQJ[1] + eUindCoef*qiUindJ[0];
    Vji[1]  += ePermCoef*qiQI[1] + eUindCoef*qiUindI[0];
    VijR[1] += dPermCoef*qiQJ[1] + dUindCoef*qiUindJ[0];
    VjiR[1] += dPermCoef*qiQI[1] + dUindCoef*qiUindI[0];
    Vijd[0] += eUindCoef*qiQJ[1];
    Vjid[0] += eUindCoef*qiQI[1];
    // D-D and D-Uind terms (m=1)
    ePermCoef = rInvVec[3]*(mScale + bVec[3] - twoThirds*alphaRVec[3]*X);
    eUindCoef = 2.0*rInvVec[3]*(dScale*thole_d1 + bVec[3] - twoThirds*alphaRVec[3]*X);
    dPermCoef = -1.5*rInvVec[4]*(mScale + bVec[3]);
    dUindCoef = -6.0*rInvVec[4]*(dScale*dthole_d1 + bVec[3]);
    Vij[2]  = ePermCoef*qiQJ[2] + eUindCoef*qiUindJ[1];
    Vji[2]  = ePermCoef*qiQI[2] + eUindCoef*qiUindI[1];
    VijR[2] = dPermCoef*qiQJ[2] + dUindCoef*qiUindJ[1];
    VjiR[2] = dPermCoef*qiQI[2] + dUindCoef*qiUindI[1];
    Vij[3]  = ePermCoef*qiQJ[3] + eUindCoef*qiUindJ[2];
    Vji[3]  = ePermCoef*qiQI[3] + eUindCoef*qiUindI[2];
    VijR[3] = dPermCoef*qiQJ[3] + dUindCoef*qiUindJ[2];
    VjiR[3] = dPermCoef*qiQI[3] + dUindCoef*qiUindI[2];
    Vijd[1] = eUindCoef*qiQJ[2];
    Vjid[1] = eUindCoef*qiQI[2];
    Vijd[2] = eUindCoef*qiQJ[3];
    Vjid[2] = eUindCoef*qiQI[3];

    // C-Q terms (m=0)
    ePermCoef = (mScale + bVec[3])*rInvVec[3];
    dPermCoef = -oneThird*rInvVec[4]*(4.5*(mScale + bVec[3]) + 2.0*alphaRVec[5]*X);
    Vij[0]  += ePermCoef*qiQJ[4];
    Vji[4]   = ePermCoef*qiQI[0];
    VijR[0] += dPermCoef*qiQJ[4];
    VjiR[4]  = dPermCoef*qiQI[0];
    // Q-C terms (m=0)
    Vij[4]   = ePermCoef*qiQJ[0];
    Vji[0]  += ePermCoef*qiQI[4];
    VijR[4]  = dPermCoef*qiQJ[0];
    VjiR[0] += dPermCoef*qiQI[4];

    // D-Q and Uind-Q terms (m=0)
    ePermCoef = rInvVec[4]*(3.0*(mScale + bVec[3]) + fourThirds*alphaRVec[5]*X);
    eUindCoef = 2.0*rInvVec[4]*(3.0*(dScale*thole_q0 + bVec[3]) + fourThirds*alphaRVec[5]*X);
    dPermCoef = -fourThirds*rInvVec[5]*(4.5*(mScale + bVec[3]) + (1.0 + alphaRVec[2])*alphaRVec[5]*X);
    dUindCoef = -2.0*fourThirds*rInvVec[5]*(9.0*(dScale*dthole_q0 + bVec[3]) + 2.0*(1.0 + alphaRVec[2])*alphaRVec[5]*X);
    Vij[1]  += ePermCoef*qiQJ[4];
    Vji[4]  += ePermCoef*qiQI[1] + eUindCoef*qiUindI[0];
    VijR[1] += dPermCoef*qiQJ[4];
    VjiR[4] += dPermCoef*qiQI[1] + dUindCoef*qiUindI[0];
    Vijd[0] += eUindCoef*qiQJ[4];
    // Q-D and Q-Uind terms (m=0)
    Vij[4]  += -(ePermCoef*qiQJ[1] + eUindCoef*qiUindJ[0]);
    Vji[1]  += -(ePermCoef*qiQI[4]);
    VijR[4] += -(dPermCoef*qiQJ[1] + dUindCoef*qiUindJ[0]);
    VjiR[1] += -(dPermCoef*qiQI[4]);
    Vjid[0] += -(eUindCoef*qiQI[4]);

    // D-Q and Uind-Q terms (m=1)
    ePermCoef = -sqrtThree*rInvVec[4]*(mScale + bVec[3]);
    eUindCoef = -2.0*sqrtThree*rInvVec[4]*(pScale*thole_q1 + bVec[3]);
    dPermCoef = fourSqrtOneThird*rInvVec[5]*(1.5*(mScale + bVec[3]) + 0.5*alphaRVec[5]*X);
    dUindCoef = 2.0*fourSqrtOneThird*rInvVec[5]*(3.0*(dScale*dthole_q1 + bVec[3]) + alphaRVec[5]*X);
    Vij[2]  += ePermCoef*qiQJ[5];
    Vji[5]   = ePermCoef*qiQI[2] + eUindCoef*qiUindI[1];
    VijR[2] += dPermCoef*qiQJ[5];
    VjiR[5]  = dPermCoef*qiQI[2] + dUindCoef*qiUindI[1];
    Vij[3]  += ePermCoef*qiQJ[6];
    Vji[6]   = ePermCoef*qiQI[3] + eUindCoef*qiUindI[2];
    VijR[3] += dPermCoef*qiQJ[6];
    VjiR[6]  = dPermCoef*qiQI[3] + dUindCoef*qiUindI[2];
    Vijd[1] += eUindCoef*qiQJ[5];
    Vijd[2] += eUindCoef*qiQJ[6];
    // D-Q and Uind-Q terms (m=1)
    Vij[5]   = -(ePermCoef*qiQJ[2] + eUindCoef*qiUindJ[1]);
    Vji[2]  += -(ePermCoef*qiQI[5]);
    VijR[5]  = -(dPermCoef*qiQJ[2] + dUindCoef*qiUindJ[1]);
    VjiR[2] += -(dPermCoef*qiQI[5]);
    Vij[6]   = -(ePermCoef*qiQJ[3] + eUindCoef*qiUindJ[2]);
    Vji[3]  += -(ePermCoef*qiQI[6]);
    VijR[6]  = -(dPermCoef*qiQJ[3] + dUindCoef*qiUindJ[2]);
    VjiR[3] += -(dPermCoef*qiQI[6]);
    Vjid[1] += -(eUindCoef*qiQI[5]);
    Vjid[2] += -(eUindCoef*qiQI[6]);

    // Q-Q terms (m=0)
    ePermCoef = rInvVec[5]*(6.0*(mScale + bVec[4]) + fourOverFortyFive*(-3.0 + 10.0*alphaRVec[2])*alphaRVec[5]*X);
    dPermCoef = -oneNinth*rInvVec[6]*(135.0*(mScale + bVec[4]) + 4.0*(1.0 + 2.0*alphaRVec[2])*alphaRVec[7]*X);
    Vij[4]  += ePermCoef*qiQJ[4];
    Vji[4]  += ePermCoef*qiQI[4];
    VijR[4] += dPermCoef*qiQJ[4];
    VjiR[4] += dPermCoef*qiQI[4];
    // Q-Q terms (m=1)
    ePermCoef = -fourOverFifteen*rInvVec[5]*(15.0*(mScale + bVec[4]) + alphaRVec[5]*X);
    dPermCoef = rInvVec[6]*(10.0*(mScale + bVec[4]) + fourThirds*alphaRVec[7]*X);
    Vij[5]  += ePermCoef*qiQJ[5];
    Vji[5]  += ePermCoef*qiQI[5];
    VijR[5] += dPermCoef*qiQJ[5];
    VjiR[5] += dPermCoef*qiQI[5];
    Vij[6]  += ePermCoef*qiQJ[6];
    Vji[6]  += ePermCoef*qiQI[6];
    VijR[6] += dPermCoef*qiQJ[6];
    VjiR[6] += dPermCoef*qiQI[6];
    // Q-Q terms (m=2)
    ePermCoef = rInvVec[5]*(mScale + bVec[4] - fourOverFifteen*alphaRVec[5]*X);
    dPermCoef = -2.5*(mScale + bVec[4])*rInvVec[6];
    Vij[7]  = ePermCoef*qiQJ[7];
    Vji[7]  = ePermCoef*qiQI[7];
    VijR[7] = dPermCoef*qiQJ[7];
    VjiR[7] = dPermCoef*qiQI[7];
    Vij[8]  = ePermCoef*qiQJ[8];
    Vji[8]  = ePermCoef*qiQI[8];
    VijR[8] = dPermCoef*qiQJ[8];
    VjiR[8] = dPermCoef*qiQI[8];

    // C-O (m=0)
    ePermCoef = rInvVec[4]*(-mScale - bVec[3] - fourOverFifteen*alphaRVec[5]*X);
    dPermCoef = 0.5*fourOverFifteen*rInvVec[5]*(15.*(mScale+bVec[3])+2.*(2.*alphaRVec[5]+alphaRVec[7])*X);
    Vij[0]  += ePermCoef*qiQJ[9];
    Vji[9]   = ePermCoef*qiQI[0];
    VijR[0] += dPermCoef*qiQJ[9];
    VjiR[9]  = dPermCoef*qiQI[0];
    // O-C (m=0)
    Vij[9]   = -ePermCoef*qiQJ[0];
    Vji[0]  -=  ePermCoef*qiQI[9];
    VijR[9]  = -dPermCoef*qiQJ[0];
    VjiR[0] -=  dPermCoef*qiQI[9];

    // D-O and Uind-O (m=0)
    ePermCoef = -4.*rInvVec[5]*(mScale+bVec[4]+twoOverFifteen*alphaRVec[7]*X);
    eUindCoef = -8.*rInvVec[5]*(dScale*thole_o0+bVec[4]+twoOverFifteen*alphaRVec[7]*X);
    dPermCoef = 0.5*fourOverFifteen*rInvVec[6]*(75.*(mScale+bVec[4])+4.*(1.+alphaRVec[2])*alphaRVec[7]*X);
    dUindCoef = 2.0*fourOverFifteen*rInvVec[6]*(75.*(dScale*dthole_o0+bVec[4])+4.*(1.+alphaRVec[2])*alphaRVec[7]*X);
    Vij[1]  += ePermCoef*qiQJ[9];
    Vji[9]  += ePermCoef*qiQI[1] + eUindCoef*qiUindI[0];
    VijR[1] += dPermCoef*qiQJ[9];
    VjiR[9] += dPermCoef*qiQI[1] + dUindCoef*qiUindI[0];
    // O-D and O-Uind (m=0)
    Vij[9]  += ePermCoef*qiQJ[1] + eUindCoef*qiUindJ[0];
    Vji[1]  += ePermCoef*qiQI[9];
    VijR[9] += dPermCoef*qiQJ[1] + dUindCoef*qiUindJ[0];
    VjiR[1] += dPermCoef*qiQI[9];
    Vijd[0] += eUindCoef*qiQJ[9];
    Vjid[0] += eUindCoef*qiQI[9];
    // D-O and O-Uind (m=1)
    ePermCoef = sqrtSix*(mScale+bVec[4])*rInvVec[5];
    eUindCoef = 2.0*sqrtSix*(dScale*thole_o1+bVec[4])*rInvVec[5];
    dPermCoef = -0.5*0.2*sqrtTwoThirds*rInvVec[6]*(75.*(mScale+bVec[4])+8.*alphaRVec[7]*X);
    dUindCoef = -2.0*0.2*sqrtTwoThirds*rInvVec[6]*(75.*(dScale*dthole_o1+bVec[4])+8.*alphaRVec[7]*X);
    Vij[2]   += ePermCoef*qiQJ[10];
    Vji[10]   = ePermCoef*qiQI[2] + eUindCoef*qiUindI[1];
    VijR[2]  += dPermCoef*qiQJ[10];
    VjiR[10]  = dPermCoef*qiQI[2] + dUindCoef*qiUindI[1];
    Vij[3]   += ePermCoef*qiQJ[11];
    Vji[11]   = ePermCoef*qiQI[3] + eUindCoef*qiUindI[2];
    VijR[3]  += dPermCoef*qiQJ[11];
    VjiR[11]  = dPermCoef*qiQI[3] + dUindCoef*qiUindI[2];
    Vijd[1] += eUindCoef*qiQJ[10];
    Vijd[2] += eUindCoef*qiQJ[11];
    // O-D and O-Uind (m=1)
    Vij[10]   = ePermCoef*qiQJ[2] + eUindCoef*qiUindJ[1];
    Vji[2]   += ePermCoef*qiQI[10];
    VijR[10]  = dPermCoef*qiQJ[2] + dUindCoef*qiUindJ[1];
    VjiR[2]  += dPermCoef*qiQI[10];
    Vij[11]   = ePermCoef*qiQJ[3] + eUindCoef*qiUindJ[2];
    Vji[3]   += ePermCoef*qiQI[11];
    VijR[11]  = dPermCoef*qiQJ[3] + dUindCoef*qiUindJ[2];
    VjiR[3]  += dPermCoef*qiQI[11];
    Vjid[1] += eUindCoef*qiQI[10];
    Vjid[2] += eUindCoef*qiQI[11];

    // Q-O (m=0)
    ePermCoef = rInvVec[6]*(-10.*(mScale+bVec[4]) - eightOverFortyFive*(3.+2.*alphaRVec[2])*alphaRVec[7]*X);
    dPermCoef = 0.5*fourOverFortyFive*rInvVec[7]*(675.*(mScale+bVec[4])+2.*(27.+4.*alphaRVec[4])*alphaRVec[7]*X);
    Vij[4]  += ePermCoef*qiQJ[9];
    Vji[9]  += ePermCoef*qiQI[4];
    VijR[4] += dPermCoef*qiQJ[9];
    VjiR[9] += dPermCoef*qiQI[4];
    // O-Q (m=0)
    Vij[9]  -= ePermCoef*qiQJ[4];
    Vji[4]  -= ePermCoef*qiQI[9];
    VijR[9] -= dPermCoef*qiQJ[4];
    VjiR[4] -= dPermCoef*qiQI[9];
    // Q-O (m=1)
    ePermCoef = 5.0*sqrtTwo*rInvVec[6]*(mScale+bVec[4] + eightOverSeventyFive*alphaRVec[7]*X);
    dPermCoef = -0.5*sqrtEightOverFifteen*rInvVec[7]*(225.*(mScale+bVec[4])+8.*(2.+alphaRVec[2])*alphaRVec[7]*X);
    Vij[5]   += ePermCoef*qiQJ[10];
    Vji[10]  += ePermCoef*qiQI[5];
    VijR[5]  += dPermCoef*qiQJ[10];
    VjiR[10] += dPermCoef*qiQI[5];
    Vij[6]   += ePermCoef*qiQJ[11];
    Vji[11]  += ePermCoef*qiQI[6];
    VijR[6]  += dPermCoef*qiQJ[11];
    VjiR[11] += dPermCoef*qiQI[6];
    // O-Q (m=1)
    Vij[10]  -= ePermCoef*qiQJ[5];
    Vji[5]   -= ePermCoef*qiQI[10];
    VijR[10] -= dPermCoef*qiQJ[5];
    VjiR[5]  -= dPermCoef*qiQI[10];
    Vij[11]  -= ePermCoef*qiQJ[6];
    Vji[6]   -= ePermCoef*qiQI[11];
    VijR[11] -= dPermCoef*qiQJ[6];
    VjiR[6]  -= dPermCoef*qiQI[11];
    // Q-O (m=2)
    ePermCoef = -sqrtFive*(mScale+bVec[4])*rInvVec[6];
    dPermCoef = 0.5*twoSqrtFiveOverFifteen*rInvVec[7]*(45.*(mScale+bVec[4])+4.*alphaRVec[7]*X);
    Vij[7]  += ePermCoef*qiQJ[12];
    Vji[12]  = ePermCoef*qiQI[7];
    VijR[7] += dPermCoef*qiQJ[12];
    VjiR[12] = dPermCoef*qiQI[7];
    Vij[8]  += ePermCoef*qiQJ[13];
    Vji[13]  = ePermCoef*qiQI[8];
    VijR[8] += dPermCoef*qiQJ[13];
    VjiR[13] = dPermCoef*qiQI[8];
    // O-Q (m=2)
    Vij[12]  = -ePermCoef*qiQJ[7];
    Vji[7]  -=  ePermCoef*qiQI[12];
    VijR[12] = -dPermCoef*qiQJ[7];
    VjiR[7] -=  dPermCoef*qiQI[12];
    Vij[13]  = -ePermCoef*qiQJ[8];
    Vji[8]  -=  ePermCoef*qiQI[13];
    VijR[13] = -dPermCoef*qiQJ[8];
    VjiR[8] -=  dPermCoef*qiQI[13];

    // O-O (m=0)
    ePermCoef = rInvVec[7]*(-20.*(mScale+bVec[5]) - eightOverOneFiveSevenFive*(15.+28.*alphaRVec[2]+28.*alphaRVec[4])*alphaRVec[7]*X);
    dPermCoef = 0.5*fourOverTwoTwoFive*rInvVec[8]*(7875.*(mScale+bVec[5])+4.*(41. - 4.*alphaRVec[2]+4.*alphaRVec[4])*alphaRVec[9]*X);
    Vij[9]  += ePermCoef*qiQJ[9];
    Vji[9]  += ePermCoef*qiQI[9];
    VijR[9] += dPermCoef*qiQJ[9];
    VjiR[9] += dPermCoef*qiQI[9];
    // O-O (m=1)
    ePermCoef = rInvVec[7]*(15.*(mScale+bVec[5]) + eightOverFiveTwoFive*(-5. + 28.*alphaRVec[2])* alphaRVec[7]*X);
    dPermCoef = -0.5*twoOverOneFiveZero*rInvVec[8]*(7875.*(mScale+bVec[5]) + 32.*(3. + 2.* alphaRVec[2])*alphaRVec[9]*X);
    Vij[10]  += ePermCoef*qiQJ[10];
    Vji[10]  += ePermCoef*qiQI[10];
    VijR[10] += dPermCoef*qiQJ[10];
    VjiR[10] += dPermCoef*qiQI[10];
    Vij[11]  += ePermCoef*qiQJ[11];
    Vji[11]  += ePermCoef*qiQI[11];
    VijR[11] += dPermCoef*qiQJ[11];
    VjiR[11] += dPermCoef*qiQI[11];
    // O-O (m=2)
    ePermCoef = rInvVec[7]*(-6.*(mScale+bVec[5]) - eightOverOneHundredFive*alphaRVec[7]*X);
    dPermCoef = 0.5*rInvVec[8]*(42.*(mScale+bVec[5]) + sixteenOverFifteen*alphaRVec[9]*X);
    Vij[12]  += ePermCoef*qiQJ[12];
    Vji[12]  += ePermCoef*qiQI[12];
    VijR[12] += dPermCoef*qiQJ[12];
    VjiR[12] += dPermCoef*qiQI[12];
    Vij[13]  += ePermCoef*qiQJ[13];
    Vji[13]  += ePermCoef*qiQI[13];
    VijR[13] += dPermCoef*qiQJ[13];
    VjiR[13] += dPermCoef*qiQI[13];
    // O-O (m=3)
    ePermCoef = rInvVec[7]*((mScale+bVec[5]) - eightOverOneHundredFive*alphaRVec[7]*X);
    dPermCoef = -0.5*7.*(mScale+bVec[5])*rInvVec[8];
    Vij[14]  = ePermCoef*qiQJ[14];
    Vji[14]  = ePermCoef*qiQI[14];
    VijR[14] = dPermCoef*qiQJ[14];
    VjiR[14] = dPermCoef*qiQI[14];
    Vij[15]  = ePermCoef*qiQJ[15];
    Vji[15]  = ePermCoef*qiQI[15];
    VijR[15] = dPermCoef*qiQJ[15];
    VjiR[15] = dPermCoef*qiQI[15];

    // Evaluate the energies, forces and torques due to permanent+induced moments
    // interacting with just the permanent moments.
    energy = 0.5*(qiQI[0]*Vij[0] + qiQJ[0]*Vji[0]);
    double fIZ = qiQI[0]*VijR[0];
    double fJZ = qiQJ[0]*VjiR[0];
    double EIX = 0.0, EIY = 0.0, EIZ = 0.0, EJX = 0.0, EJY = 0.0, EJZ = 0.0;
    for (int i = 1; i < 16; ++i) {
        energy += 0.5*(qiQI[i]*Vij[i] + qiQJ[i]*Vji[i]);
        fIZ += qiQI[i]*VijR[i];
        fJZ += qiQJ[i]*VjiR[i];
        EIX += qiQIX[i]*Vij[i];
        EIY += qiQIY[i]*Vij[i];
        EIZ += qiQIZ[i]*Vij[i];
        EJX += qiQJX[i]*Vji[i];
        EJY += qiQJY[i]*Vji[i];
        EJZ += qiQJZ[i]*Vji[i];
    }
    // Define the torque intermediates for the induced dipoles. These are simply the induced dipole torque
    // intermediates dotted with the field due to permanent moments only, at each center. We inline the
    // induced dipole torque intermediates here, for simplicity. N.B. There are no torques on the dipoles
    // themselves, so we accumulate the torque intermediates into separate variables to allow them to be
    // used only in the force calculation.
    //
    // The torque about the x axis (needed to obtain the y force on the induced dipoles, below)
    //    qiUindIx[0] = qiQUindI[2];    qiUindIx[1] = 0;    qiUindIx[2] = -qiQUindI[0]
    double iEIX = qiUindI[2]*Vijd[0] - qiUindI[0]*Vijd[2];
    double iEJX = qiUindJ[2]*Vjid[0] - qiUindJ[0]*Vjid[2];
    // The torque about the y axis (needed to obtain the x force on the induced dipoles, below)
    //    qiUindIy[0] = -qiQUindI[1];   qiUindIy[1] = qiQUindI[0];    qiUindIy[2] = 0
    double iEIY = qiUindI[0]*Vijd[1] - qiUindI[1]*Vijd[0];
    double iEJY = qiUindJ[0]*Vjid[1] - qiUindJ[1]*Vjid[0];
    // The torque about the z axis (needed to obtain the x force on the induced dipoles, below)
    //    qiUindIz[0] = 0;  qiUindIz[1] = -qiQUindI[2];    qiUindIz[2] = qiQUindI[1]
    double iEIZ = qiUindI[1]*Vijd[2] - qiUindI[2]*Vijd[1];
    double iEJZ = qiUindJ[1]*Vjid[2] - qiUindJ[2]*Vjid[1];

    // Add in the induced-induced terms, if needed.
    if(getPolarizationType() == MPIDReferenceForce::Mutual) {
        // Uind-Uind terms (m=0)
        double eCoef = -2.0*fourThirds*rInvVec[3]*(3.0*(uScale*thole_d0 + bVec[3]) + alphaRVec[3]*X);
        double dCoef = 2.0*rInvVec[4]*(6.0*(uScale*dthole_d0 + bVec[3]) + 4.0*alphaRVec[5]*X);
        iEIX += eCoef*qiUindI[2]*qiUindJ[0];
        iEJX += eCoef*qiUindJ[2]*qiUindI[0];
        iEIY -= eCoef*qiUindI[1]*qiUindJ[0];
        iEJY -= eCoef*qiUindJ[1]*qiUindI[0];
        fIZ  += dCoef*qiUindI[0]*qiUindJ[0];
        fJZ  += dCoef*qiUindJ[0]*qiUindI[0];
        // Uind-Uind terms (m=1)
        eCoef = 4.0*rInvVec[3]*(uScale*thole_d1 + bVec[3] - twoThirds*alphaRVec[3]*X);
        dCoef = -6.0*rInvVec[4]*(uScale*dthole_d1 + bVec[3]);
        iEIX -= eCoef*qiUindI[0]*qiUindJ[2];
        iEJX -= eCoef*qiUindJ[0]*qiUindI[2];
        iEIY += eCoef*qiUindI[0]*qiUindJ[1];
        iEJY += eCoef*qiUindJ[0]*qiUindI[1];
        iEIZ += eCoef*qiUindI[1]*qiUindJ[2];
        iEJZ += eCoef*qiUindJ[1]*qiUindI[2];
        fIZ  += dCoef*(qiUindI[1]*qiUindJ[1] + qiUindI[2]*qiUindJ[2]);
        fJZ  += dCoef*(qiUindJ[1]*qiUindI[1] + qiUindJ[2]*qiUindI[2]);
    }

    // The quasi-internal frame forces and torques.  Note that the induced torque intermediates are
    // used in the force expression, but not in the torques; the induced dipoles are isotropic.
    double qiForce[3] = {rInv*(EIY+EJY+iEIY+iEJY), -rInv*(EIX+EJX+iEIX+iEJX), -(fJZ+fIZ)};
    double qiTorqueI[3] = {-EIX, -EIY, -EIZ};
    double qiTorqueJ[3] = {-EJX, -EJY, -EJZ};
    if(particleI.isAnisotropic){
        qiTorqueI[0] += -iEIX;
        qiTorqueI[1] += -iEIY;
        qiTorqueI[2] += -iEIZ;
    }
    if(particleJ.isAnisotropic){
        qiTorqueJ[0] += -iEJX;
        qiTorqueJ[1] += -iEJY;
        qiTorqueJ[2] += -iEJZ;
    }

    // Rotate the forces and torques back to the lab frame
    Vec3 tmpf, tmpi, tmpj;
    for (int ii = 0; ii < 3; ii++) {
        double forceVal = 0.0;
        double torqueIVal = 0.0;
        double torqueJVal = 0.0;
        for (int jj = 0; jj < 3; jj++) {
            forceVal   += forceRotationMatrix[ii][jj] * qiForce[jj];
            torqueIVal += forceRotationMatrix[ii][jj] * qiTorqueI[jj];
            torqueJVal += forceRotationMatrix[ii][jj] * qiTorqueJ[jj];
        }
        tmpi[ii] = torqueIVal;
        tmpj[ii] = torqueJVal;
        tmpf[ii] = forceVal;
        torques[iIndex][ii] += torqueIVal;
        torques[jIndex][ii] += torqueJVal;
        forces[iIndex][ii]  -= forceVal;
        forces[jIndex][ii]  += forceVal;
    }
    return energy;

}

double MPIDReferencePmeForce::calculateElectrostatic(const vector<MultipoleParticleData>& particleData,
                                                                vector<Vec3>& torques, vector<Vec3>& forces)
{
    double energy = 0.0;
    vector<double> scaleFactors(LAST_SCALE_TYPE_INDEX);
    for (auto& s : scaleFactors)
        s = 1.0;

    // loop over particle pairs for direct space interactions

    for (unsigned int ii = 0; ii < particleData.size(); ii++) {
        for (unsigned int jj = ii+1; jj < particleData.size(); jj++) {

            if (jj <= _maxScaleIndex[ii]) {
                getMultipoleScaleFactors(ii, jj, scaleFactors);
            }

            energy += calculatePmeDirectElectrostaticPairIxn(particleData[ii], particleData[jj], scaleFactors, forces, torques);

            if (jj <= _maxScaleIndex[ii]) {
                for (auto& s : scaleFactors)
                    s = 1.0;
            }
        }
    }

    // The polarization energy
    calculatePmeSelfTorque(particleData, torques);
    energy += computeReciprocalSpaceInducedDipoleForceAndEnergy(getPolarizationType(), particleData, forces, torques);
    energy += computeReciprocalSpaceFixedMultipoleForceAndEnergy(particleData, forces, torques);
    energy += calculatePmeSelfEnergy(particleData);

    // Now that both the direct and reciprocal space contributions have been added, we can compute the dipole
    // response contributions to the forces, if we're using the extrapolated polarization algorithm.
    if (getPolarizationType() == MPIDReferenceForce::Extrapolated) {
        double prefac = (_electric/_dielectric);
        for (int i = 0; i < _numParticles; i++) {
            // Compute the µ(m) T µ(n) force contributions here
            for (int l = 0; l < _maxPTOrder-1; ++l) {
                for (int m = 0; m < _maxPTOrder-1-l; ++m) {
                    double p = _extPartCoefficients[l+m+1];
                    if(std::fabs(p) < 1e-6) continue;
                    forces[i][0] += p*prefac*(_ptDipoleD[l][i][0]*_ptDipoleFieldGradientD[m][6*i+0]
                                            + _ptDipoleD[l][i][1]*_ptDipoleFieldGradientD[m][6*i+3]
                                            + _ptDipoleD[l][i][2]*_ptDipoleFieldGradientD[m][6*i+4]);
                    forces[i][1] += p*prefac*(_ptDipoleD[l][i][0]*_ptDipoleFieldGradientD[m][6*i+3]
                                            + _ptDipoleD[l][i][1]*_ptDipoleFieldGradientD[m][6*i+1]
                                            + _ptDipoleD[l][i][2]*_ptDipoleFieldGradientD[m][6*i+5]);
                    forces[i][2] += p*prefac*(_ptDipoleD[l][i][0]*_ptDipoleFieldGradientD[m][6*i+4]
                                            + _ptDipoleD[l][i][1]*_ptDipoleFieldGradientD[m][6*i+5]
                                            + _ptDipoleD[l][i][2]*_ptDipoleFieldGradientD[m][6*i+2]);
                    if(particleData[i].isAnisotropic){
                        torques[i][0] += p*prefac*(_ptDipoleD[l][i][1]*_ptDipoleFieldD[m][3*i+2]
                                                 - _ptDipoleD[l][i][2]*_ptDipoleFieldD[m][3*i+1]);
                        torques[i][1] += p*prefac*(_ptDipoleD[l][i][2]*_ptDipoleFieldD[m][3*i+0]
                                                 - _ptDipoleD[l][i][0]*_ptDipoleFieldD[m][3*i+2]);
                        torques[i][2] += p*prefac*(_ptDipoleD[l][i][0]*_ptDipoleFieldD[m][3*i+1]
                                                 - _ptDipoleD[l][i][1]*_ptDipoleFieldD[m][3*i+0]);
                    }
                }
            }
        }
    }
    return energy;
}
