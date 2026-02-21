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

#ifndef __MPIDReferenceForce_H__
#define __MPIDReferenceForce_H__

#include "openmm/MPIDForce.h"
#include "openmm/Vec3.h"
#include <map>
#include "fftpack.h"
#include <complex>
#include <vector>

using namespace std;

namespace OpenMM {

typedef std::map< unsigned int, double> MapIntRealOpenMM;
typedef MapIntRealOpenMM::iterator MapIntRealOpenMMI;
typedef MapIntRealOpenMM::const_iterator MapIntRealOpenMMCI;


// A few useful constants for the spherical harmonic multipole code.
const double oneThird = 1.0/3.0;
const double twoThirds = 2.0/3.0;
const double fourThirds = 4.0/3.0;
const double fourSqrtOneThird = 4.0/sqrt(3.0);
const double sqrtTwoThirds = sqrt(2.0/3.0);
const double sqrtFourThirds = 2.0/sqrt(3.0);
const double sqrtOneThird = 1.0/sqrt(3.0);
const double sqrtOneTenth = 1.0/sqrt(10.0);
const double sqrtTwo = sqrt(2.0);
const double sqrtThree = sqrt(3.0);
const double sqrtFive = sqrt(5.0);
const double sqrtSix = sqrt(6.0);
const double sqrtEightOverFifteen = sqrt(8.0)/15.0;
const double sqrtThreeHalves = sqrt(3.0/2.0);
const double sqrtFiveHalves = sqrt(5.0/2.0);
const double sqrtThreeFifths = sqrt(3.0/5.0);
const double twoSqrtFiveOverFifteen = 2.0*sqrt(5.0)/15.0;
const double oneNinth = 1.0/9.0;
const double fourOverFortyFive = 4.0/45.0;
const double fourOverTwoTwoFive = 4.0/225.0;
const double eightOverFortyFive = 8.0/45.0;
const double eightOverSeventyFive = 8.0/75.0;
const double eightOverFiveTwoFive = 8.0/525.0;
const double eightOverOneFiveSevenFive = 8.0/1575.0;
const double twoOverFifteen = 2.0/15.0;
const double twoOverOneFiveZero = 2.0/150.0;
const double fourOverFifteen = 4.0/15.0;
const double eightOverOneHundredFive = 8.0/105.0;
const double sixteenOverFifteen = 16.0/15.0;


/**
 * 2-dimensional int vector
 */
class int2 {
public:
    /**
     * Create a int2 whose elements are all 0.
     */
    int2() {
        data[0] = data[1] = 0;
    }
    /**
     * Create a int2 with specified x, y components.
     */
    int2(int x, int y) {
        data[0] = x;
        data[1] = y;
    }
    int operator[](int index) const {
        assert(index >= 0 && index < 2);
        return data[index];
    }
    int& operator[](int index) {
        assert(index >= 0 && index < 2);
        return data[index];
    }

    // Arithmetic operators

    // unary plus
    int2 operator+() const {
        return int2(*this);
    }

    // plus
    int2 operator+(const int2& rhs) const {
        const int2& lhs = *this;
        return int2(lhs[0] + rhs[0], lhs[1] + rhs[1]);
    }

    int2& operator+=(const int2& rhs) {
        data[0] += rhs[0];
        data[1] += rhs[1];
        return *this;
    }

    int2& operator-=(const int2& rhs) {
        data[0] -= rhs[0];
        data[1] -= rhs[1];
        return *this;
    }

private:
    int data[2];
};

/**
 * 3-dimensional int vector
 */
class IntVec {
public:
    /**
     * Create a IntVec whose elements are all 0.
     */
    IntVec() {
        data[0] = data[1] = data[2] = 0;
    }
    /**
     * Create a IntVec with specified x, y, z, w components.
     */
    IntVec(int x, int y, int z) {
        data[0] = x;
        data[1] = y;
        data[2] = z;
    }
    int operator[](int index) const {
        assert(index >= 0 && index < 3);
        return data[index];
    }
    int& operator[](int index) {
        assert(index >= 0 && index < 3);
        return data[index];
    }

    // Arithmetic operators

    // unary plus
    IntVec operator+() const {
        return IntVec(*this);
    }

    // plus
    IntVec operator+(const IntVec& rhs) const {
        const IntVec& lhs = *this;
        return IntVec(lhs[0] + rhs[0], lhs[1] + rhs[1], lhs[2] + rhs[2]);
    }

    IntVec& operator+=(const IntVec& rhs) {
        data[0] += rhs[0];
        data[1] += rhs[1];
        data[2] += rhs[2];
        return *this;
    }

    IntVec& operator-=(const IntVec& rhs) {
        data[0] -= rhs[0];
        data[1] -= rhs[1];
        data[2] -= rhs[2];
        return *this;
    }

private:
    int data[3];
};

/**
 * 4-dimensional double vector
 */
class double4 {
public:
    /**
     * Create a double4 whose elements are all 0.
     */
    double4() {
        data[0] = data[1] = data[2] = data[3] = 0.0;
    }
    /**
     * Create a double4 with specified x, y, z, w components.
     */
    double4(double x, double y, double z, double w) {
        data[0] = x;
        data[1] = y;
        data[2] = z;
        data[3] = w;
    }
    double operator[](int index) const {
        assert(index >= 0 && index < 4);
        return data[index];
    }
    double& operator[](int index) {
        assert(index >= 0 && index < 4);
        return data[index];
    }

    // Arithmetic operators

    // unary plus
    double4 operator+() const {
        return double4(*this);
    }

    // plus
    double4 operator+(const double4& rhs) const {
        const double4& lhs = *this;
        return double4(lhs[0] + rhs[0], lhs[1] + rhs[1], lhs[2] + rhs[2],lhs[3] + rhs[3]);
    }

    double4& operator+=(const double4& rhs) {
        data[0] += rhs[0];
        data[1] += rhs[1];
        data[2] += rhs[2];
        data[3] += rhs[3];
        return *this;
    }

    double4& operator-=(const double4& rhs) {
        data[0] -= rhs[0];
        data[1] -= rhs[1];
        data[2] -= rhs[2];
        data[3] -= rhs[3];
        return *this;
    }

private:
    double data[4];
};


/**
 * 5-dimensional double vector
 */
class double5 {
public:
    /**
     * Create a double5 whose elements are all 0.
     */
    double5() {
        data[0] = data[1] = data[2] = data[3] = data[4] = 0.0;
    }
    /**
     * Create a double4 with specified x, y, z, w components.
     */
    double5(double x, double y, double z, double w, double v) {
        data[0] = x;
        data[1] = y;
        data[2] = z;
        data[3] = w;
        data[4] = v;
    }
    double operator[](int index) const {
        assert(index >= 0 && index < 5);
        return data[index];
    }
    double& operator[](int index) {
        assert(index >= 0 && index < 5);
        return data[index];
    }

    // Arithmetic operators

    // unary plus
    double5 operator+() const {
        return double5(*this);
    }

    // plus
    double5 operator+(const double5& rhs) const {
        const double5& lhs = *this;
        return double5(lhs[0] + rhs[0], lhs[1] + rhs[1], lhs[2] + rhs[2], lhs[3] + rhs[3], lhs[4] + rhs[4]);
    }

    double5& operator+=(const double5& rhs) {
        data[0] += rhs[0];
        data[1] += rhs[1];
        data[2] += rhs[2];
        data[3] += rhs[3];
        data[4] += rhs[4];
        return *this;
    }

    double5& operator-=(const double5& rhs) {
        data[0] -= rhs[0];
        data[1] -= rhs[1];
        data[2] -= rhs[2];
        data[3] -= rhs[3];
        data[4] -= rhs[4];
        return *this;
    }

private:
    double data[5];
};


using namespace OpenMM;

class MPIDReferenceForce {

   /**
    * MPIDReferenceForce is base class for Force calculations
    * MPIDReferenceGeneralizedKirkwoodForce  is derived class for Generalized Kirkwood calculations
    * MPIDReferencePmeForce is derived class for PME calculations
    *
    * Below is a outline of the sequence of methods called to evaluate the force and energy 
    * for each scenario: Generalized Kirkwood (GK) and PME.
    *
    * If 'virtual' appears before the method name, the method is overridden in one or more of the derived classes.
    *
    * calculateForceAndEnergy()                            calculate forces  and energy
    *
    *    setup()                                           rotate molecular multipole moments to lab frame
    *                                                      setup scaling maps and calculate induced dipoles (see calculateInducedDipoles below)
    *
    *    virtual calculateElectrostatic()                  calculate forces and torques
    *
    *                                                      GK case includes the following calls:
    *
    *                                                          MPIDReferenceForce::calculateElectrostatic()
    *                                                               loop over particle pairs: calculateElectrostaticPairIxn()
    *
    *                                                          TINKER's egk1a: calculateKirkwoodPairIxn()
    *
    *                                                          SASA force and energy: calculateCavityTermEnergyAndForces()
    *
    *                                                          TINKER's born1 (Born chain rule term): loop over particle pairs: calculateGrycukChainRulePairIxn()
    *
    *                                                          TINKER's ediff1: loop over particle pairs: calculateKirkwoodEDiffPairIxn()
    *                                                       
    *                                                      PME case includes the following calls:
    *
    *                                                          reciprocal [computeReciprocalSpaceInducedDipoleForceAndEnergy(),
    *                                                                      computeReciprocalSpaceFixedForceAndEnergy]
    *
    *                                                          direct space calculations [calculatePmeDirectElectrostaticPairIxn()]
    *
    *                                                          self-energy [calculatePmeSelfEnergy()]
    *
    *                                                          torques [calculatePmeSelfTorque()]
    *
    *    mapTorqueToForce()                                map torques to forces
    * 
    * setup()
    *    loadParticleData()                                load particle data (polarity, multipole moments, Thole factors, ...)
    *    checkChiral()                                     if needed, invert multipole moments at chiral centers
    *    applyRotationMatrix()                             rotate molecular multipole moments to lab frame
    *    setupScaleMaps()                                  setup scaling maps
    *    calculateInducedDipoles()                         calculate induced dipoles
    * 
    * 
    * virtual calculateInducedDipoles()                    calculate induced dipoles:
    *                                                          field at each site due to fixed multipoles first calculated
    *                                                          if polarization type == Direct,
    *                                                          initial induced dipoles are calculated, but are not converged.
    *                                                          if polarization type == Mutual, then loop until
    *                                                          induce dipoles converge.
    *                                                       For GK, include gkField in setup
    *                                                       For PME, base class method is used
    *
    * 
    *     virtual zeroFixedMultipoleFields()                zero fixed multipole vectors; for GK includes zeroing of gkField vector
    * 
    *     virtual calculateFixedMultipoleField()            calculate fixed multipole field -- particle pair loop 
    *                                                       gkField also calculated for GK
    *                                                       for PME, reciprocal, direct space (particle pair loop) and self terms calculated
    *                                                       
    * 
    *         virtual calculateFixedMultipoleFieldPairIxn() pair ixn for fixed multipole
    *                                                       gkField also calculated for GK
    *                                                       for PME, direct space ixn calculated here
    * 
    *     virtual initializeInducedDipoles()                initialize induced dipoles; for PME, calculateReciprocalSpaceInducedDipoleField()
    *                                                       called in case polarization type == Direct
    *
    *     convergeInduceDipoles()                           loop until induced dipoles converge
    *
    *         updateInducedDipoleFields()                   update fields at each site due other induced dipoles
    *
    *           virtual calculateInducedDipoleFields()      calculate induced dipole field at each site by looping over particle pairs 
    *                                                       for PME includes reciprocal space calculation calculateReciprocalSpaceInducedDipoleField(), 
    *                                                       direct space calculateDirectInducedDipolePairIxns() and self terms
    *
    *              virtual calculateInducedDipolePairIxns() field at particle i due particle j's induced dipole and vice versa; for GK includes GK field
    */

public:

    /** 
     * This is an enumeration of the different methods that may be used for handling long range Multipole forces.
     */
    enum NonbondedMethod {

        /**
         * No cutoff is applied to the interactions.  The full set of N^2 interactions is computed exactly.
         * This necessarily means that periodic boundary conditions cannot be used.  This is the default.
         */
        NoCutoff = 0,

       /** 
         * Periodic boundary conditions are used, and Particle-Mesh Ewald (PME) summation is used to compute the interaction of each particle
         * with all periodic copies of every other particle.
         */
        PME = 1
    };

    enum PolarizationType {

        /** 
         * Mutual polarization
         */
        Mutual = 0,

        /**
         * Direct polarization
         */
        Direct = 1,

        /**
         * Extrapolated perturbation theory
         */
        Extrapolated = 2
    };

    /**
     * Constructor
     * 
     */
    MPIDReferenceForce();
 
    /**
     * Constructor
     * 
     * @param nonbondedMethod nonbonded method
     */
    MPIDReferenceForce(NonbondedMethod nonbondedMethod);
 
    /**
     * Destructor
     * 
     */
    virtual ~MPIDReferenceForce() {};
 
    /**
     * Get nonbonded method.
     * 
     * @return nonbonded method
     */
    NonbondedMethod getNonbondedMethod() const;

    /**
     * Set nonbonded method.
     * 
     * @param nonbondedMethod nonbonded method
     */
    void setNonbondedMethod(NonbondedMethod nonbondedMethod);

    /**
     * Get default Thole width.
     *
     * @return default Thole width (nm).
     */
    double getDefaultTholeWidth() const;

    /**
     * Set default Thole width.
     *
     * @param width The default (used for direct pairs) Thole width in nm
     */
    void setDefaultTholeWidth(double width);

    /**
     * Get polarization type.
     * 
     * @return polarization type
     */
    PolarizationType getPolarizationType() const;

    /**
     * Set polarization type.
     * 
     * @param  polarizationType polarization type
     */
    void setPolarizationType(PolarizationType polarizationType);

    /**
     * Get flag indicating if mutual induced dipoles are converged.
     *
     * @return nonzero if converged
     *
     */
    int getMutualInducedDipoleConverged() const;

    /**
     * Get the number of iterations used in computing mutual induced dipoles.
     *
     * @return number of iterations
     *
     */
    int getMutualInducedDipoleIterations() const;

    /**
     * Get the final epsilon for mutual induced dipoles.
     *
     *  @return epsilon
     *
     */
    double getMutualInducedDipoleEpsilon() const;

    /**
     * Set the coefficients for the µ_0, µ_1, µ_2, µ_n terms in the extrapolation
     * theory algorithm for induced dipoles
     *
     * @param optCoefficients a vector whose mth entry specifies the coefficient for µ_m
     *
     */
    void setExtrapolationCoefficients(const std::vector<double> &coefficients);

    /**
     * Set the target epsilon for converging mutual induced dipoles.
     *
     * @param targetEpsilon target epsilon for converging mutual induced dipoles
     *
     */
    void setMutualInducedDipoleTargetEpsilon(double targetEpsilon);

    /**
     * Get the target epsilon for converging mutual induced dipoles.
     *
     * @return target epsilon for converging mutual induced dipoles
     *
     */
    double getMutualInducedDipoleTargetEpsilon() const;

    /**
     * Set the scale permanent-permanent and permanent-induced scale factors
     * for 1-4 connected connected atoms
     *
     * @param scaleFactor the factor by which the 1-4 interactions are scaled.
     *
     */
    void set14ScaleFactor(double scaleFactor);

    /**
     * Get the scale permanent-permanent and permanent-induced scale factors
     * for 1-4 connected connected atoms
     *
     * @return the factor by which the 1-4 interactions are scaled.
     *
     */
    double get14ScaleFactor() const;

    /**
     * Set the maximum number of iterations to be executed in converging mutual induced dipoles.
     *
     * @param maximumMutualInducedDipoleIterations maximum number of iterations to be executed in converging mutual induced dipoles
     *
     */
    void setMaximumMutualInducedDipoleIterations(int maximumMutualInducedDipoleIterations);

    /**
     * Get the maximum number of iterations to be executed in converging mutual induced dipoles.
     *
     * @return maximum number of iterations to be executed in converging mutual induced dipoles
     * 
     */
    int getMaximumMutualInducedDipoleIterations() const;

    /**
     * Calculate force and energy.
     *
     * @param particlePositions         Cartesian coordinates of particles
     * @param charges                   scalar charges for each particle
     * @param dipoles                   molecular frame dipoles for each particle
     * @param quadrupoles               molecular frame quadrupoles for each particle
     * @param octopoles                 molecular frame octopoles for each particle
     * @param tholes                    Thole factors for each particle
     * @param dampingFactors            damping factors for each particle
     * @param polarity                  diagonal elements of polarizability tensor for each particle
     * @param axisTypes                 axis type (Z-then-X, ...) for each particle
     * @param multipoleAtomZs           indicies of particle specifying the molecular frame z-axis for each particle
     * @param multipoleAtomXs           indicies of particle specifying the molecular frame x-axis for each particle
     * @param multipoleAtomYs           indicies of particle specifying the molecular frame y-axis for each particle
     * @param multipoleAtomCovalentInfo covalent info needed to set scaling factors
     * @param forces                    add forces to this vector
     *
     * @return energy
     */
    double calculateForceAndEnergy(const std::vector<OpenMM::Vec3>& particlePositions,
                                   const std::vector<double>& charges,
                                   const std::vector<double>& dipoles,
                                   const std::vector<double>& quadrupoles,
                                   const std::vector<double> &octopoles,
                                   const std::vector<double>& tholes,
                                   const std::vector<double>& dampingFactors,
                                   const vector<std::vector<double> > &polarity,
                                   const std::vector<int>& axisTypes,
                                   const std::vector<int>& multipoleAtomZs,
                                   const std::vector<int>& multipoleAtomXs,
                                   const std::vector<int>& multipoleAtomYs,
                                   const std::vector< std::vector< std::vector<int> > >& multipoleAtomCovalentInfo,
                                   std::vector<OpenMM::Vec3>& forces);

    /**
     * Calculate particle induced dipoles.
     *
     * @param masses                    particle masses
     * @param particlePositions         Cartesian coordinates of particles
     * @param charges                   scalar charges for each particle
     * @param dipoles                   molecular frame dipoles for each particle
     * @param quadrupoles               molecular frame quadrupoles for each particle
     * @param octopoles                 molecular frame octopoles for each particle
     * @param tholes                    Thole factors for each particle
     * @param dampingFactors            dampling factors for each particle
     * @param polarity                  diagonal elements of the polarizability tensor for each particle
     * @param axisTypes                 axis type (Z-then-X, ...) for each particle
     * @param multipoleAtomZs           indicies of particle specifying the molecular frame z-axis for each particle
     * @param multipoleAtomXs           indicies of particle specifying the molecular frame x-axis for each particle
     * @param multipoleAtomYs           indicies of particle specifying the molecular frame y-axis for each particle
     * @param multipoleAtomCovalentInfo covalent info needed to set scaling factors
     * @param outputMultipoleMoments    output multipole moments
     */
    void calculateInducedDipoles(const std::vector<OpenMM::Vec3>& particlePositions,
                                 const std::vector<double>& charges,
                                 const std::vector<double>& dipoles,
                                 const std::vector<double>& quadrupoles,
                                 const std::vector<double>& octopoles,
                                 const std::vector<double>& tholes,
                                 const std::vector<double>& dampingFactors,
                                 const std::vector<std::vector<double> >& polarity,
                                 const std::vector<int>& axisTypes,
                                 const std::vector<int>& multipoleAtomZs,
                                 const std::vector<int>& multipoleAtomXs,
                                 const std::vector<int>& multipoleAtomYs,
                                 const std::vector< std::vector< std::vector<int> > >& multipoleAtomCovalentInfo,
                                 std::vector<Vec3>& outputInducedDipoles);

    /**
     * Calculate particle permanent dipoles rotated in the lab frame.
     *
     * @param masses                    particle masses
     * @param particlePositions         Cartesian coordinates of particles
     * @param charges                   scalar charges for each particle
     * @param dipoles                   molecular frame dipoles for each particle
     * @param quadrupoles               molecular frame quadrupoles for each particle
     * @param octopoles                 molecular frame octopoles for each particle
     * @param tholes                    Thole factors for each particle
     * @param dampingFactors            dampling factors for each particle
     * @param polarity                  diagonal elements of the polarizability tensor
     * @param axisTypes                 axis type (Z-then-X, ...) for each particle
     * @param multipoleAtomZs           indicies of particle specifying the molecular frame z-axis for each particle
     * @param multipoleAtomXs           indicies of particle specifying the molecular frame x-axis for each particle
     * @param multipoleAtomYs           indicies of particle specifying the molecular frame y-axis for each particle
     * @param multipoleAtomCovalentInfo covalent info needed to set scaling factors
     * @param outputMultipoleMoments    output multipole moments
     */

    void calculateLabFramePermanentDipoles(const std::vector<Vec3>& particlePositions,
                                           const std::vector<double>& charges,
                                           const std::vector<double>& dipoles,
                                           const std::vector<double>& quadrupoles,
                                           const std::vector<double>& octopoles,
                                           const std::vector<double>& tholes,
                                           const std::vector<double>& dampingFactors,
                                           const vector<std::vector<double> > &polarity,
                                           const std::vector<int>& axisTypes,
                                           const std::vector<int>& multipoleAtomZs,
                                           const std::vector<int>& multipoleAtomXs,
                                           const std::vector<int>& multipoleAtomYs,
                                           const std::vector< vector< vector<int> > >& multipoleAtomCovalentInfo,
                                           std::vector<Vec3>& outputRotatedPermanentDipoles);

    /**
     * Calculate particle total dipoles.
     *
     * @param masses                    particle masses
     * @param particlePositions         Cartesian coordinates of particles
     * @param charges                   scalar charges for each particle
     * @param dipoles                   molecular frame dipoles for each particle
     * @param quadrupoles               molecular frame quadrupoles for each particle
     * @param octopoles                 molecular frame octopoles for each particle
     * @param tholes                    Thole factors for each particle
     * @param dampingFactors            dampling factors for each particle
     * @param polarity                  diagonal elements of the polarizability tensor for each particle
     * @param axisTypes                 axis type (Z-then-X, ...) for each particle
     * @param multipoleAtomZs           indicies of particle specifying the molecular frame z-axis for each particle
     * @param multipoleAtomXs           indicies of particle specifying the molecular frame x-axis for each particle
     * @param multipoleAtomYs           indicies of particle specifying the molecular frame y-axis for each particle
     * @param multipoleAtomCovalentInfo covalent info needed to set scaling factors
     * @param outputMultipoleMoments    output multipole moments
     */


    void calculateTotalDipoles(const std::vector<Vec3>& particlePositions,
                               const std::vector<double>& charges,
                               const std::vector<double>& dipoles,
                               const std::vector<double>& quadrupoles,
                               const std::vector<double>& octopoles,
                               const std::vector<double>& tholes,
                               const std::vector<double>& dampingFactors,
                               const vector<std::vector<double> > &polarity,
                               const std::vector<int>& axisTypes,
                               const std::vector<int>& multipoleAtomZs,
                               const std::vector<int>& multipoleAtomXs,
                               const std::vector<int>& multipoleAtomYs,
                               const std::vector< vector< vector<int> > >& multipoleAtomCovalentInfo,
                               std::vector<Vec3>& outputRotatedPermanentDipoles);



    /**
     * Calculate system multipole moments.
     *
     * @param masses                    particle masses
     * @param particlePositions         Cartesian coordinates of particles
     * @param charges                   scalar charges for each particle
     * @param dipoles                   molecular frame dipoles for each particle
     * @param quadrupoles               molecular frame quadrupoles for each particle
     * @param octopoles                 molecular frame octopoles for each particle
     * @param tholes                    Thole factors for each particle
     * @param dampingFactors            dampling factors for each particle
     * @param polarity                  diagonal elements of the polarizability tensor for each particle
     * @param axisTypes                 axis type (Z-then-X, ...) for each particle
     * @param multipoleAtomZs           indicies of particle specifying the molecular frame z-axis for each particle
     * @param multipoleAtomXs           indicies of particle specifying the molecular frame x-axis for each particle
     * @param multipoleAtomYs           indicies of particle specifying the molecular frame y-axis for each particle
     * @param multipoleAtomCovalentInfo covalent info needed to set scaling factors
     * @param outputMultipoleMoments    output multipole moments
     */
    void calculateMPIDSystemMultipoleMoments(const std::vector<double>& masses,
                                               const std::vector<OpenMM::Vec3>& particlePositions,
                                               const std::vector<double>& charges,
                                               const std::vector<double>& dipoles,
                                               const std::vector<double>& quadrupoles,
                                               const std::vector<double>& octopoles,
                                               const std::vector<double>& tholes,
                                               const std::vector<double>& dampingFactors,
                                               const std::vector<std::vector<double> >& polarity,
                                               const std::vector<int>& axisTypes,
                                               const std::vector<int>& multipoleAtomZs,
                                               const std::vector<int>& multipoleAtomXs,
                                               const std::vector<int>& multipoleAtomYs,
                                               const std::vector< std::vector< std::vector<int> > >& multipoleAtomCovalentInfo,
                                               std::vector<double>& outputMultipoleMoments);

    /**
     * Calculate electrostatic potential at a set of grid points.
     *
     * @param particlePositions         Cartesian coordinates of particles
     * @param charges                   scalar charges for each particle
     * @param dipoles                   molecular frame dipoles for each particle
     * @param quadrupoles               molecular frame quadrupoles for each particle
     * @param octopoles                 molecular frame octopoles for each particle
     * @param tholes                    Thole factors for each particle
     * @param dampingFactors            dampling factors for each particle
     * @param polarity                  diagonal elements of the polarizability tensor for each particle
     * @param axisTypes                 axis type (Z-then-X, ...) for each particle
     * @param multipoleAtomZs           indicies of particle specifying the molecular frame z-axis for each particle
     * @param multipoleAtomXs           indicies of particle specifying the molecular frame x-axis for each particle
     * @param multipoleAtomYs           indicies of particle specifying the molecular frame y-axis for each particle
     * @param multipoleAtomCovalentInfo covalent info needed to set scaling factors
     * @param input grid                input grid points to compute potential
     * @param outputPotential           output electrostatic potential
     */
    void calculateElectrostaticPotential(const std::vector<OpenMM::Vec3>& particlePositions,
                                         const std::vector<double>& charges,
                                         const std::vector<double>& dipoles,
                                         const std::vector<double>& quadrupoles,
                                         const std::vector<double>& octopoles,
                                         const std::vector<double>& tholes,
                                         const std::vector<double>& dampingFactors,
                                         const std::vector<std::vector<double> >& polarity,
                                         const std::vector<int>& axisTypes,
                                         const std::vector<int>& multipoleAtomZs,
                                         const std::vector<int>& multipoleAtomXs,
                                         const std::vector<int>& multipoleAtomYs,
                                         const std::vector< std::vector< std::vector<int> > >& multipoleAtomCovalentInfo,
                                         const std::vector<Vec3>& inputGrid,
                                         std::vector<double>& outputPotential);

protected:

    enum MultipoleParticleDataEnum { PARTICLE_POSITION, PARTICLE_CHARGE, PARTICLE_DIPOLE, PARTICLE_QUADRUPOLE,
                                     PARTICLE_THOLE, PARTICLE_DAMPING_FACTOR, PARTICLE_POLARITY, PARTICLE_FIELD, 
                                     PARTICLE_FIELD_POLAR, GK_FIELD, PARTICLE_INDUCED_DIPOLE, PARTICLE_INDUCED_DIPOLE_POLAR };

    enum QuadrupoleIndices { QXX, QXY, QXZ, QYY, QYZ, QZZ };
    enum OctopoleIndices   { QXXX, QXXY, QXXZ, QXYY, QXYZ, QXZZ, QYYY, QYYZ, QYZZ, QZZZ };

    /* 
     * Particle parameters and coordinates
     */
    class MultipoleParticleData {
        public:
            unsigned int particleIndex;    
            Vec3 position;
            double charge;
            Vec3 dipole;
            double quadrupole[6];
            Vec3 sphericalDipole;
            double sphericalQuadrupole[5];
            double octopole[10];
            double sphericalOctopole[7];
            double thole;
            double dampingFactor;
            std::vector<double> polarity;
            double labPolarization[6];
            bool isAnisotropic;
    };
    
    /**
     * Particle parameters transformed into fractional coordinates
     */
    class TransformedMultipole {
    public:
        double charge;
        Vec3 dipole;
        double quadrupole[6];
        double octopole[10];
    };

    /* 
     * Helper class used in calculating induced dipoles
     */
    struct UpdateInducedDipoleFieldStruct {
            UpdateInducedDipoleFieldStruct(std::vector<OpenMM::Vec3>& inputFixed_E_Field, std::vector<OpenMM::Vec3>& inputInducedDipoles, std::vector<std::vector<Vec3> >& extrapolatedDipoles, std::vector<std::vector<double> >& extrapolatedDipoleField, std::vector<std::vector<double> >& extrapolatedDipoleFieldGradient);
            std::vector<OpenMM::Vec3>* fixedMultipoleField;
            std::vector<OpenMM::Vec3>* inducedDipoles;
            std::vector<std::vector<Vec3> >* extrapolatedDipoles;
            std::vector<std::vector<double> >* extrapolatedDipoleField;
            std::vector<std::vector<double> >* extrapolatedDipoleFieldGradient;
            std::vector<OpenMM::Vec3> inducedDipoleField;
            std::vector<std::vector<double> > inducedDipoleFieldGradient;
    };

    unsigned int _numParticles;

    NonbondedMethod _nonbondedMethod;
    PolarizationType _polarizationType;

    double _electric;
    double _dielectric;

    enum ScaleType { M_SCALE, P_SCALE, LAST_SCALE_TYPE_INDEX };
    std::vector<  std::vector< MapIntRealOpenMM > > _scaleMaps;
    std::vector<unsigned int> _maxScaleIndex;
    double _mScale[5];
    double _pScale[5];

    std::vector<TransformedMultipole> _transformed;
    std::vector<Vec3> _fixedMultipoleField;
    std::vector<Vec3> _inducedDipole;
    std::vector<std::vector<Vec3> > _ptDipoleD;
    std::vector<std::vector<double> > _ptDipoleFieldD;
    std::vector<std::vector<double> > _ptDipoleFieldGradientD;

    double _defaultTholeWidth;
    int _mutualInducedDipoleConverged;
    int _mutualInducedDipoleIterations;
    int _maximumMutualInducedDipoleIterations;
    int _maxPTOrder;
    std::vector<double>  _extrapolationCoefficients;
    std::vector<double>  _extPartCoefficients;
    double  _mutualInducedDipoleEpsilon;
    double  _mutualInducedDipoleTargetEpsilon;
    double  _polarSOR;
    double  _debye;

    /**
     * Helper constructor method to centralize initialization of objects.
     *
     */
    void initialize();

    /**
     * Load particle data.
     *
     * @param particlePositions   particle coordinates
     * @param charges             charges
     * @param dipoles             dipoles
     * @param quadrupoles         quadrupoles
     * @param octopoles           octopoles
     * @param tholes              Thole parameters
     * @param dampingFactors      dampming factors
     * @param polarity            diagonal elements of the polarizability tensor
     * @param particleData        output data struct
     *
     */
    void loadParticleData(const std::vector<OpenMM::Vec3>& particlePositions,
                          const std::vector<double>& charges,
                          const std::vector<double>& dipoles,
                          const std::vector<double>& quadrupoles,
                          const std::vector<double>& octopoles,
                          const std::vector<double>& tholes,
                          const std::vector<double>& dampingFactors,
                          const vector<std::vector<double> > &polarity,
                          std::vector<MultipoleParticleData>& particleData) const;

    /**
     * Calculate fixed multipole fields.
     *
     * @param particleData vector of particle data
     * 
     */
    virtual void calculateFixedMultipoleField(const vector<MultipoleParticleData>& particleData);

    /**
     * Set flag indicating if mutual induced dipoles are converged.
     * 
     * @param converged nonzero if converged
     *
     */
    void setMutualInducedDipoleConverged(int converged);

    /**
     * Set number of iterations used in computing mutual induced dipoles.
     * 
     * @param  number of iterations
     * 
     */
    void setMutualInducedDipoleIterations(int iterations);

    /**
     * Set the final epsilon for mutual induced dipoles.
     * 
     * @param epsilon
     *
     */
    void setMutualInducedDipoleEpsilon(double epsilon);

    /**
     * Setup scale factors given covalent info.
     *
     * @param  multipoleAtomCovalentInfo vector of vectors containing the covalent info
     *
     */
    void setupScaleMaps(const std::vector< std::vector< std::vector<int> > >& multipoleAtomCovalentInfo);

    /**
     * Get multipole scale factor for particleI & particleJ
     * 
     * @param  particleI           index of particleI whose scale factor is to be retrieved
     * @param  particleJ           index of particleJ whose scale factor is to be retrieved
     * @param  scaleType           scale type (D_SCALE, P_SCALE, M_SCAL)
     *
     * @return scaleFactor 
     */
    double getMultipoleScaleFactor(unsigned int particleI, unsigned int particleJ, ScaleType scaleType) const;

    /**
     * Get scale factor for particleI & particleJ
     * 
     * @param  particleI           index of particleI whose scale factor is to be retrieved
     * @param  particleJ           index of particleJ whose scale factor is to be retrieved
     * @param  scaleType           scale type (D_SCALE, P_SCALE, M_SCAL)
     *
     * @return array of scaleFactors 
     */
    void getMultipoleScaleFactors(unsigned int particleI, unsigned int particleJ, std::vector<double>& scaleFactors) const;

    /**
     * Get p- and d-scale factors for particleI & particleJ ixn
     *
     * @param  particleI           index of particleI 
     * @param  particleJ           index of particleJ
     * @param  dScale              output d-scale factor
     * @param  pScale              output p-scale factor
     */
    void getDScaleAndPScale(unsigned int particleI, unsigned int particleJ, double& dScale, double& pScale) const;
    
    /**
     * Calculate damped powers of 1/r.
     *
     * @param  particleI           index of particleI 
     * @param  particleJ           index of particleJ
     * @param  dScale              output d-scale factor
     * @param  pScale              output p-scale factor
     */
    void getAndScaleInverseRs(double dampI, double dampJ, double pscale, double tholeI, double tholeJ,
                              double r, std::vector<double>& rrI) const;

    /**
     * Check if multipoles at chiral site should be inverted.
     *
     * @param  particleI            particleI data 
     * @param  axisType             axis type
     * @param  particleZ            z-axis particle to particleI
     * @param  particleX            x-axis particle to particleI
     * @param  particleY            y-axis particle to particleI
     *
     */
    void checkChiralCenterAtParticle(MultipoleParticleData& particleI, int axisType, MultipoleParticleData& particleZ,
                                     MultipoleParticleData& particleX, MultipoleParticleData& particleY) const;

    /**
     * Invert multipole moments (dipole[Y], quadrupole[XY] and quadrupole[YZ]) if chiral center inverted.
     * 
     * @param particleData            vector of parameters (charge, labFrame dipoles, quadrupoles, ...) for particles
     * @param multipoleAtomXs         vector of z-particle indices used to map molecular frame to lab frame
     * @param multipoleAtomYs         vector of x-particle indices used to map molecular frame to lab frame
     * @param multipoleAtomZs         vector of y-particle indices used to map molecular frame to lab frame
     * @param axisType                axis type
     */
    void checkChiral(std::vector<MultipoleParticleData>& particleData, 
                     const std::vector<int>& multipoleAtomXs,
                     const std::vector<int>& multipoleAtomYs,
                     const std::vector<int>& multipoleAtomZs,
                     const std::vector<int>& axisTypes) const;
    /**
     * Apply rotation matrix to molecular dipole/quadrupoles to get corresponding lab frame values
     * for particle I.
     * 
     * @param  particleI            particleI data
     * @param  particleJ            particleI data
     * @param  axisType             axis type
     */
    void applyRotationMatrixToParticle(      MultipoleParticleData& particleI,
                                       const MultipoleParticleData* particleZ,
                                       const MultipoleParticleData* particleX,
                                             MultipoleParticleData* particleY, int axisType) const;

    /**
     * Forms the rotation matrix for the quasi-internal coordinate system,
     * which is the rotation matrix that describes the orientation of the
     * internuclear vector for a given pair (I,J) in lab frame.
     *
     * @param particleI             particleI position
     * @param particleJ             particleJ position
     * @param deltaR                the internuclear vector, corrected for periodic boundary conditions
     * @param r                     the bond length between atoms I and J
     * @param rotationmatrix        the output rotation matrix for a 3-vector
     */
    void formQIRotationMatrix(const Vec3& iPosition,
                              const Vec3& jPosition,
                              const Vec3 &deltaR,
                              double r,
                              double (&rotationMatrix)[3][3]) const;


    /**
     * Constructs a rotation matrix for spherical harmonic quadrupoles, using the dipole rotation matrix.
     *
     * @param D1                    The input spherical harmonic dipole rotation matrix
     * @param D2                    The output spherical harmonic quadrupole rotation matrix
     */
     void buildSphericalQuadrupoleRotationMatrix(const double (&D1)[3][3], double (&D2)[5][5]) const;

     /**
      * Constructs a rotation matrix for spherical harmonic octopoles, using the dipole rotation matrix.
      *
      * @param D1                    The input spherical harmonic dipole rotation matrix
      * @param D2                    The input spherical harmonic quadrupole rotation matrix
      * @param D3                    The output spherical harmonic octopole rotation matrix
      */
      void buildSphericalOctopoleRotationMatrix(const double (&D1)[3][3], const double (&D2)[5][5], double (&D3)[7][7]) const;

     /**
      * Constructs a rotation matrix for spherical harmonic quadrupoles, using the dipole rotation matrix.
      * Only the m={0,1c,1s} terms are constructed; these are the only terms needed to evaluate the field.
      *
      * @param D1                    The input spherical harmonic dipole rotation matrix
      * @param D2                    The output spherical harmonic quadrupole rotation matrix
      */
      void buildPartialSphericalQuadrupoleRotationMatrix(const double (&D1)[3][3], double (&D2)[3][5]) const;

    /**
     * Apply rotation matrix to molecular dipole/quadrupoles to get corresponding lab frame values.
     * 
     * @param particleData            vector of parameters (charge, labFrame dipoles, quadrupoles, ...) for particles
     *                                dipole and quadrupole entries are modified
     * @param multipoleAtomXs         vector of z-particle indices used to map molecular frame to lab frame
     * @param multipoleAtomYs         vector of x-particle indices used to map molecular frame to lab frame
     * @param multipoleAtomZs         vector of y-particle indices used to map molecular frame to lab frame
     * @param axisType                axis type
     */
    void applyRotationMatrix(std::vector<MultipoleParticleData>& particleData, 
                             const std::vector<int>& multipoleAtomXs,
                             const std::vector<int>& multipoleAtomYs,
                             const std::vector<int>& multipoleAtomZs,
                             const std::vector<int>& axisTypes) const;
    /**
     * Zero fixed multipole fields.
     */
    virtual void zeroFixedMultipoleFields();

    /**
     * Calculate electric field at particle I due fixed multipoles at particle J and vice versa
     * (field at particle J due fixed multipoles at particle I).
     * 
     * @param particleI               positions and parameters (charge, labFrame dipoles, quadrupoles, ...) for particle I
     * @param particleJ               positions and parameters (charge, labFrame dipoles, quadrupoles, ...) for particle J
     * @param dScale                  d-scale value for i-j interaction
     * @param pScale                  p-scale value for i-j interaction
     */
    virtual void calculateFixedMultipoleFieldPairIxn(const MultipoleParticleData& particleI, const MultipoleParticleData& particleJ,
                                                     double dScale, double pScale);

    /**
     * Initialize induced dipoles
     *
     * @param updateInducedDipoleFields vector of UpdateInducedDipoleFieldStruct containing input induced dipoles and output fields
     */
    virtual void initializeInducedDipoles(std::vector<UpdateInducedDipoleFieldStruct>& updateInducedDipoleFields); 

    /**
     * Calculate field at particle I due induced dipole at particle J and vice versa
     * (field at particle J due induced dipole at particle I).
     * 
     * @param particleI               index of particle I
     * @param particleJ               index of particle J
     * @param rr3                     damped 1/r^3 factor
     * @param rr5                     damped 1/r^5 factor
     * @param delta                   delta of particle positions: particleJ.x - particleI.x, ...
     * @param inducedDipole           vector of induced dipoles
     * @param field                   vector of induced dipole fields
     */
    void calculateInducedDipolePairIxn(unsigned int particleI, unsigned int particleJ,
                                       double rr3, double rr5, const Vec3& delta,
                                       const std::vector<Vec3>& inducedDipole,
                                       std::vector<Vec3>& field) const;

    /**
     * Calculate fields due induced dipoles at each site.
     *
     * @param particleI                 positions and parameters (charge, labFrame dipoles, quadrupoles, ...) for particle I
     * @param particleJ                 positions and parameters (charge, labFrame dipoles, quadrupoles, ...) for particle J
     * @param updateInducedDipoleFields vector of UpdateInducedDipoleFieldStruct containing input induced dipoles and output fields
     */
    virtual void calculateInducedDipolePairIxns(const MultipoleParticleData& particleI, const MultipoleParticleData& particleJ,
                                                std::vector<UpdateInducedDipoleFieldStruct>& updateInducedDipoleFields);

    /**
     * Calculate induced dipole fields.
     * 
     * @param particleData              vector of particle positions and parameters (charge, labFrame dipoles, quadrupoles, ...)
     * @param updateInducedDipoleFields vector of UpdateInducedDipoleFieldStruct containing input induced dipoles and output fields
     */
    virtual void calculateInducedDipoleFields(const std::vector<MultipoleParticleData>& particleData,
                                              std::vector<UpdateInducedDipoleFieldStruct>& updateInducedDipoleFields);
    /**
     * Calculated induced dipoles using extrapolated perturbation theory.
     *
     * @param particleData              vector of particle positions and parameters (charge, labFrame dipoles, quadrupoles, ...)
     * @param updateInducedDipoleFields vector of UpdateInducedDipoleFieldStruct containing input induced dipoles and output fields
     */
    void convergeInduceDipolesByExtrapolation(const std::vector<MultipoleParticleData>& particleData,
                                              std::vector<UpdateInducedDipoleFieldStruct>& calculateInducedDipoleField);
    /**
     * Converge induced dipoles.
     * 
     * @param particleData              vector of particle positions and parameters (charge, labFrame dipoles, quadrupoles, ...)
     * @param updateInducedDipoleFields vector of UpdateInducedDipoleFieldStruct containing input induced dipoles and output fields
     */
    void convergeInduceDipolesBySOR(const std::vector<MultipoleParticleData>& particleData,
                                    std::vector<UpdateInducedDipoleFieldStruct>& calculateInducedDipoleField);
    /**
     * Converge induced dipoles.
     * 
     * @param particleData              vector of particle positions and parameters (charge, labFrame dipoles, quadrupoles, ...)
     * @param updateInducedDipoleFields vector of UpdateInducedDipoleFieldStruct containing input induced dipoles and output fields
     */
    void convergeInduceDipolesByDIIS(const std::vector<MultipoleParticleData>& particleData,
                                     std::vector<UpdateInducedDipoleFieldStruct>& calculateInducedDipoleField);
    
    /**
     * Use DIIS to compute the weighting coefficients for the new induced dipoles.
     * 
     * @param prevErrors    the vector of errors from previous iterations
     * @param coefficients  the coefficients will be stored into this
     */
    void computeDIISCoefficients(const std::vector<std::vector<Vec3> >& prevErrors, std::vector<double>& coefficients) const;

    /**
     * Update fields due to induced dipoles for each particle.
     * 
     * @param particleData              vector of particle positions and parameters (charge, labFrame dipoles, quadrupoles, ...)
     * @param updateInducedDipoleFields vector of UpdateInducedDipoleFieldStruct containing input induced dipoles and output fields
     */
    double updateInducedDipoleFields(const std::vector<MultipoleParticleData>& particleData,
                                     std::vector<UpdateInducedDipoleFieldStruct>& calculateInducedDipoleField);

    /**
     * Update induced dipole for a particle given updated induced dipole field at the site.
     * 
     * @param particleI                 positions and parameters (charge, labFrame dipoles, quadrupoles, ...) for particle I
     * @param fixedMultipoleField       fields due fixed multipoles at each site
     * @param inducedDipoleField        fields due induced dipoles at each site
     * @param inducedDipoles            output vector of updated induced dipoles
     */
    double updateInducedDipole(const std::vector<MultipoleParticleData>& particleI,
                               const std::vector<Vec3>& fixedMultipoleField,
                               const std::vector<Vec3>& inducedDipoleField,
                               std::vector<Vec3>& inducedDipoles);

    /**
     * Calculate induced dipoles.
     * 
     * @param particleData      vector of particle positions and parameters (charge, labFrame dipoles, quadrupoles, ...)
     */
    virtual void calculateInducedDipoles(const std::vector<MultipoleParticleData>& particleData);

    /**
     * Setup: 
     *        if needed invert multipole moments at chiral centers
     *        rotate molecular multipole moments to lab frame 
     *        setup scaling maps and 
     *        calculate induced dipoles (see calculateInducedDipoles below)
     *
     * @param particlePositions         Cartesian coordinates of particles
     * @param charges                   scalar charges for each particle
     * @param dipoles                   molecular frame dipoles for each particle
     * @param quadrupoles               molecular frame quadrupoles for each particle
     * @param octopoles                 molecular frame octopoles for each particle
     * @param tholes                    Thole factors for each particle
     * @param dampingFactors            dampling factors for each particle
     * @param polarity                  diagonal elements of the polarizability tensor for each particle
     * @param axisTypes                 axis type (Z-then-X, ...) for each particle
     * @param multipoleAtomZs           indicies of particle specifying the molecular frame z-axis for each particle
     * @param multipoleAtomXs           indicies of particle specifying the molecular frame x-axis for each particle
     * @param multipoleAtomYs           indicies of particle specifying the molecular frame y-axis for each particle
     * @param multipoleAtomCovalentInfo covalent info needed to set scaling factors
     * @param particleData              output vector of parameters (charge, labFrame dipoles, quadrupoles, ...) for particles
     *
     */
    void setup(const std::vector<OpenMM::Vec3>& particlePositions,
               const std::vector<double>& charges,
               const std::vector<double>& dipoles,
               const std::vector<double>& quadrupoles,
               const std::vector<double>& octopoles,
               const std::vector<double>& tholes,
               const std::vector<double>& dampingFactors,
               const std::vector<std::vector<double> >& polarity,
               const std::vector<int>& axisTypes,
               const std::vector<int>& multipoleAtomZs,
               const std::vector<int>& multipoleAtomXs,
               const std::vector<int>& multipoleAtomYs,
               const std::vector< std::vector< std::vector<int> > >& multipoleAtomCovalentInfo,
               std::vector<MultipoleParticleData>& particleData);

    /**
     * Calculate electrostatic interaction between particles I and K.
     * 
     * @param particleI         positions and parameters (charge, labFrame dipoles, quadrupoles, ...) for particle I
     * @param particleK         positions and parameters (charge, labFrame dipoles, quadrupoles, ...) for particle K
     * @param scalingFactors    scaling factors for interaction
     * @param forces            vector of particle forces to be updated
     * @param torque            vector of particle torques to be updated
     */
    double calculateElectrostaticPairIxn(const MultipoleParticleData& particleI, const MultipoleParticleData& particleK,
                                         const std::vector<double>& scalingFactors, std::vector<OpenMM::Vec3>& forces, std::vector<Vec3>& torque) const;

    /**
     * Map particle torque to force.
     * 
     * @param particleI               particle whose torque is to be mapped
     * @param particleU               particle1 of lab frame for particleI 
     * @param particleV               particle2 of lab frame for particleI 
     * @param particleW               particle3 of lab frame for particleI 
     * @param axisType                axis type (Bisector/Z-then-X, ...)
     * @param torque                  torque on particle I
     * @param forces                  vector of particle forces to be updated
     */
    void mapTorqueToForceForParticle(const MultipoleParticleData& particleI,
                                     const MultipoleParticleData& particleU,
                                     const MultipoleParticleData& particleV,
                                           MultipoleParticleData* particleW,
                                     int axisType, const Vec3& torque, std::vector<OpenMM::Vec3>& forces) const;

    /**
     * Map torques to forces.
     * 
     * @param particleData            vector of parameters (charge, labFrame dipoles, quadrupoles, ...) for particles
     * @param multipoleAtomZs         vector of z-particle indices used to map molecular frame to lab frame
     * @param multipoleAtomXs         vector of x-particle indices used to map molecular frame to lab frame
     * @param multipoleAtomYs         vector of y-particle indices used to map molecular frame to lab frame
     * @param axisType                vector of axis types (Bisector/Z-then-X, ...) for particles
     * @param torques                 output torques
     * @param forces                  output forces 
     */
    void mapTorqueToForce(std::vector<MultipoleParticleData>& particleData, 
                          const std::vector<int>& multipoleAtomXs,
                          const std::vector<int>& multipoleAtomYs,
                          const std::vector<int>& multipoleAtomZs,
                          const std::vector<int>& axisTypes,
                          std::vector<OpenMM::Vec3>& torques,
                          std::vector<OpenMM::Vec3>& forces) const;

    /**
     * Calculate electrostatic forces
     * 
     * @param particleData            vector of parameters (charge, labFrame dipoles, quadrupoles, ...) for particles
     * @param torques                 output torques
     * @param forces                  output forces 
     *
     * @return energy
     */
    virtual double calculateElectrostatic(const std::vector<MultipoleParticleData>& particleData, 
                                          std::vector<OpenMM::Vec3>& torques,
                                          std::vector<OpenMM::Vec3>& forces);

    /**
     * Normalize a Vec3
     *
     * @param vectorToNormalize vector to normalize
     *
     * @return norm of vector on input
     * 
     */
    double normalizeVec3(Vec3& vectorToNormalize) const;

    /**
     * Initialize vector of double (size=numParticles)
     *
     * @param vectorToInitialize vector to initialize
     * 
     */
    void initializeRealOpenMMVector(vector<double>& vectorToInitialize) const;

    /**
     * Initialize vector of Vec3 (size=numParticles)
     *
     * @param vectorToInitialize vector to initialize
     * 
     */
    void initializeVec3Vector(vector<Vec3>& vectorToInitialize) const;

    /**
     * Copy vector of Vec3
     *
     * @param inputVector  vector to copy
     * @param outputVector output vector
     * 
     */
    void copyVec3Vector(const std::vector<OpenMM::Vec3>& inputVector, std::vector<OpenMM::Vec3>& outputVector) const;

    /**
     * Calculate potential at grid point due to a particle
     *
     * @param particleData            vector of parameters (charge, labFrame dipoles, quadrupoles, ...) for particles
     * @param gridPoint               grid point
     *
     * @return potential at grid point
     * 
     */
    double calculateElectrostaticPotentialForParticleGridPoint(const MultipoleParticleData& particleI, const Vec3& gridPoint) const;

    /**
     * Apply periodic boundary conditions to difference in positions
     * 
     * @param deltaR  difference in particle positions; modified on output after applying PBC
     * 
     */
    virtual void getPeriodicDelta(Vec3& deltaR) const {};
};


class MPIDReferencePmeForce : public MPIDReferenceForce {

public:

    /**
     * Constructor
     * 
     */
    MPIDReferencePmeForce();
 
    /**
     * Destructor
     * 
     */
    ~MPIDReferencePmeForce();
 
    /**
     * Get cutoff distance.
     *
     * @return cutoff distance
     *
     */
    double getCutoffDistance() const;

    /**
     * Set cutoff distance.
     *
     * @return cutoff distance
     *
     */
    void setCutoffDistance(double cutoffDistance);

    /**
     * Get alpha used in Ewald summation.
     *
     * @return alpha
     *
     */
    double getAlphaEwald() const;

    /**
     * Set alpha used in Ewald summation.
     *
     * @return alpha
     *
     */
    void setAlphaEwald(double alphaEwald);

    /**
     * Get PME grid dimensions.
     *
     * @param pmeGridDimensions contains PME grid dimensions upon return

     *
     */
    void getPmeGridDimensions(std::vector<int>& pmeGridDimensions) const;

    /**
     * Set PME grid dimensions.
     *
     * @param pmeGridDimensions input PME grid dimensions 
     *
     */
    void setPmeGridDimensions(std::vector<int>& pmeGridDimensions);

    /**
     * Set periodic box size.
     *
     * @param vectors    the vectors defining the periodic box
     */
     void setPeriodicBoxSize(OpenMM::Vec3* vectors);

private:

    static const int MPID_PME_ORDER;
    static const double SQRT_PI;

    double _alphaEwald;
    double _cutoffDistance;
    double _cutoffDistanceSquared;

    Vec3 _recipBoxVectors[3];
    Vec3 _periodicBoxVectors[3];

    int _totalGridSize;
    IntVec _pmeGridDimensions;

    fftpack_t   _fftplan;

    unsigned int _pmeGridSize;
    t_complex* _pmeGrid;
 
    std::vector<double> _pmeBsplineModuli[3];
    std::vector<double5> _thetai[3];
    std::vector<IntVec> _iGrid;
    std::vector<double> _phi;
    std::vector<double> _phidp;
    std::vector<double4> _pmeBsplineTheta;
    std::vector<double4> _pmeBsplineDtheta;

    /**
     * Resize PME arrays.
     * 
     */
    void resizePmeArrays();

    /**
     * Zero Pme grid.
     */
    void initializePmeGrid();

    /**
     * Modify input vector of differences in particle positions for periodic boundary conditions.
     * 
     * @param delta                   input vector of difference in particle positions; on output adjusted for
     *                                periodic boundary conditions
     */
    void getPeriodicDelta(Vec3& deltaR) const;

    /**
     * Calculate damped inverse distances.
     * 
     * @param particleI               positions and parameters (charge, labFrame dipoles, quadrupoles, ...) for particle I
     * @param particleJ               positions and parameters (charge, labFrame dipoles, quadrupoles, ...) for particle J
     * @param dScale                  d-scale value for i-j interaction
     * @param pScale                  p-scale value for i-j interaction
     * @param dampedDInverseDistances damped inverse distances (drr3,drr5,drr7 in udirect2a() in TINKER)
     * @param dampedPInverseDistances damped inverse distances (prr3,prr5,prr7 in udirect2a() in TINKER)
     */
    void getDampedInverseDistances(const MultipoleParticleData& particleI, const MultipoleParticleData& particleJ,
                                   double dscale, double pscale, double r,
                                   std::vector<double> &dampedDInverseDistances, std::vector<double> &dampedPInverseDistances) const;
    
    /**
     * Initialize B-spline moduli.
     * 
     */
    void initializeBSplineModuli();

    /**
     * Calculate direct-space field at site I due fixed multipoles at site J and vice versa.
     * 
     * @param particleI               positions and parameters (charge, labFrame dipoles, quadrupoles, ...) for particle I
     * @param particleJ               positions and parameters (charge, labFrame dipoles, quadrupoles, ...) for particle J
     * @param dScale                  d-scale value for i-j interaction
     * @param pScale                  p-scale value for i-j interaction
     */
    void calculateFixedMultipoleFieldPairIxn(const MultipoleParticleData& particleI, const MultipoleParticleData& particleJ,
                                             double dscale, double pscale);
    
    /**
     * Calculate fixed multipole fields.
     *
     * @param particleData vector particle data
     * 
     */
    void calculateFixedMultipoleField(const vector<MultipoleParticleData>& particleData);

    /**
     * This is called from computeMPIDBsplines().  It calculates the spline coefficients for a single atom along a single axis.
     * 
     * @param thetai output spline coefficients
     * @param w offset from grid point
     */
    void computeBSplinePoint(vector<double5> &thetai, double w);
    
    /**
     * Compute bspline coefficients.
     *
     * @param particleData   vector of particle positions and parameters (charge, labFrame dipoles, quadrupoles, ...)
     */
    void computeMPIDBsplines(const std::vector<MultipoleParticleData>& particleData);

    /**
     * Transform multipoles from cartesian coordinates to fractional coordinates.
     */
    void transformMultipolesToFractionalCoordinates(const vector<MultipoleParticleData>& particleData);

    /**
     * Transform potential from fractional coordinates to cartesian coordinates.
     */
    void transformPotentialToCartesianCoordinates(const std::vector<double>& fphi, std::vector<double>& cphi) const;

    /**
     * Spread fixed multipoles onto PME grid.
     * 
     * @param particleData vector of particle positions and parameters (charge, labFrame dipoles, quadrupoles, ...)
     */
    void spreadFixedMultipolesOntoGrid(const vector<MultipoleParticleData>& particleData);

    /**
     * Perform reciprocal convolution.
     * 
     */
    void performMPIDReciprocalConvolution();

    /**
     * Compute reciprocal potential due fixed multipoles at each particle site.
     * 
     */
    void computeFixedPotentialFromGrid(void);

    /**
     * Compute reciprocal potential due fixed multipoles at each particle site.
     * 
     */
    void computeInducedPotentialFromGrid();

    /**
     * Calculate reciprocal space energy and force due to fixed multipoles.
     * 
     * @param particleData    vector of particle positions and parameters (charge, labFrame dipoles, quadrupoles, ...)
     * @param forces          upon return updated vector of forces
     * @param torques         upon return updated vector of torques
     *
     * @return energy
     */
    double computeReciprocalSpaceFixedMultipoleForceAndEnergy(const std::vector<MultipoleParticleData>& particleData,
                                                              std::vector<Vec3>& forces, std::vector<Vec3>& torques) const;

    /**
     * Set reciprocal space fixed multipole fields.
     * 
     */
    void recordFixedMultipoleField();

    /**
     * Compute the potential due to the reciprocal space PME calculation for induced dipoles.
     *
     * @param updateInducedDipoleFields vector of UpdateInducedDipoleFieldStruct containing input induced dipoles and output fields
     */
    void calculateReciprocalSpaceInducedDipoleField(std::vector<UpdateInducedDipoleFieldStruct>& updateInducedDipoleFields);

    /**
     * Calculate field at particleI due to induced dipole at particle J and vice versa.
     *
     * @param iIndex        particle I index
     * @param jIndex        particle J index
     * @param preFactor1    first factor used in calculating field
     * @param preFactor2    second factor used in calculating field
     * @param delta         delta in particle positions after adjusting for periodic boundary conditions
     * @param inducedDipole vector of induced dipoles
     * @param field         vector of field at each particle due induced dipole of other particles
     */
    void calculateDirectInducedDipolePairIxn(unsigned int iIndex, unsigned int jIndex,
                                             double preFactor1, double preFactor2, const Vec3& delta,
                                             const std::vector<Vec3>& inducedDipole,
                                             std::vector<Vec3>& field) const;

    /**
     * Calculate direct space field at particleI due to induced dipole at particle J and vice versa for
     * inducedDipole and inducedDipolePolar.
     * 
     * @param particleI                 positions and parameters (charge, labFrame dipoles, quadrupoles, ...) for particle I
     * @param particleJ                 positions and parameters (charge, labFrame dipoles, quadrupoles, ...) for particle J
     * @param updateInducedDipoleFields vector of UpdateInducedDipoleFieldStruct containing input induced dipoles and output fields
     */
    void calculateDirectInducedDipolePairIxns(const MultipoleParticleData& particleI,
                                              const MultipoleParticleData& particleJ,
                                              std::vector<UpdateInducedDipoleFieldStruct>& updateInducedDipoleFields);

    /**
     * Initialize induced dipoles
     *
     * @param updateInducedDipoleFields vector of UpdateInducedDipoleFieldStruct containing input induced dipoles and output fields
     */
    void initializeInducedDipoles(std::vector<UpdateInducedDipoleFieldStruct>& updateInducedDipoleFields); 

    /**
     * Spread induced dipoles onto grid.
     *
     * @param inputInducedDipole      induced dipole value
     */
    void spreadInducedDipolesOnGrid(const std::vector<Vec3>& inputInducedDipole);

    /**
     * Calculate induced dipole fields.
     * 
     * @param particleData              vector of particle positions and parameters (charge, labFrame dipoles, quadrupoles, ...)
     * @param updateInducedDipoleFields vector of UpdateInducedDipoleFieldStruct containing input induced dipoles and output fields
     */
    void calculateInducedDipoleFields(const std::vector<MultipoleParticleData>& particleData,
                                      std::vector<UpdateInducedDipoleFieldStruct>& updateInducedDipoleFields);

    /**
     * Set reciprocal space induced dipole fields. 
     *
     * @param field       reciprocal space output induced dipole field value at each site
     * 
     */
    void recordInducedDipoleField(vector<Vec3>& field);

    /**
     * Compute Pme self energy.
     *
     * @param particleData            vector of parameters (charge, labFrame dipoles, quadrupoles, ...) for particles
     */
    double calculatePmeSelfEnergy(const std::vector<MultipoleParticleData>& particleData) const;

    /**
     * Compute the self torques.
     *
     * @param particleData            vector of parameters (charge, labFrame dipoles, quadrupoles, ...) for particles
     * @param torques                 vector of torques
     */
    void calculatePmeSelfTorque(const std::vector<MultipoleParticleData>& particleData, std::vector<Vec3>& torques) const;

    /**
     * Calculate direct space electrostatic interaction between particles I and J.
     * 
     * @param particleI         positions and parameters (charge, labFrame dipoles, quadrupoles, ...) for particle I
     * @param particleJ         positions and parameters (charge, labFrame dipoles, quadrupoles, ...) for particle J
     * @param scalingFactors    scaling factors for interaction
     * @param forces            vector of particle forces to be updated
     * @param torques           vector of particle torques to be updated
     */
    double calculatePmeDirectElectrostaticPairIxn(const MultipoleParticleData& particleI, const MultipoleParticleData& particleJ,
                                                  const std::vector<double>& scalingFactors,
                                                  std::vector<Vec3>& forces, std::vector<Vec3>& torques) const;

    /**
     * Calculate reciprocal space energy/force/torque for dipole interaction.
     * 
     * @param polarizationType  if 'Direct' polariztion, only initial induced dipoles calculated
     *                          if 'Mutual' polariztion, induced dipoles converged to specified tolerance
     * @param particleData      vector of particle positions and parameters (charge, labFrame dipoles, quadrupoles, ...)
     * @param forces            vector of particle forces to be updated
     * @param torques           vector of particle torques to be updated
     */
     double computeReciprocalSpaceInducedDipoleForceAndEnergy(MPIDReferenceForce::PolarizationType polarizationType,
                                                              const std::vector<MultipoleParticleData>& particleData,
                                                              std::vector<Vec3>& forces, std::vector<Vec3>& torques) const;

    /**
     * Calculate electrostatic forces.
     * 
     * @param particleData            vector of parameters (charge, labFrame dipoles, quadrupoles, ...) for particles
     * @param torques                 output torques
     * @param forces                  output forces 
     *
     * @return energy
     */
    double calculateElectrostatic(const std::vector<MultipoleParticleData>& particleData, 
                                  std::vector<OpenMM::Vec3>& torques,
                                  std::vector<OpenMM::Vec3>& forces);

};

} // namespace OpenMM

#endif // _MPIDReferenceForce___
