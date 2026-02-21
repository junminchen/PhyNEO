/* -------------------------------------------------------------------------- *
 *                                   OpenMMMPID                             *
 * -------------------------------------------------------------------------- *
 * This is part of the OpenMM molecular simulation toolkit originating from   *
 * Simbios, the NIH National Center for Physics-Based Simulation of           *
 * Biological Structures at Stanford, funded under the NIH Roadmap for        *
 * Medical Research, grant U54 GM072970. See https://simtk.org.               *
 *                                                                            *
 * Portions copyright (c) 2008-2012 Stanford University and the Authors.      *
 * Authors: Mark Friedrichs                                                   *
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

/**
 * This tests the CUDA implementation of MPIDForce.
 */

#include "openmm/internal/AssertionUtilities.h"
#include "openmm/Context.h"
#include "OpenMMMPID.h"
#include "openmm/NonbondedForce.h"
#include "openmm/System.h"
#include "openmm/Units.h"
#include "openmm/VerletIntegrator.h"
#include "openmm/MPIDForce.h"
#include "openmm/LangevinIntegrator.h"
#include "openmm/Vec3.h"
#include <iostream>
#include <iomanip>
#include <vector>
#include <stdlib.h>
#include <stdio.h>

#define ASSERT_EQUAL_TOL_MOD(expected, found, tol, testname) {double _scale_ = std::abs(expected) > 1.0 ? std::abs(expected) : 1.0; if (!(std::abs((expected)-(found))/_scale_ <= (tol))) {std::stringstream details; details << testname << " Expected "<<(expected)<<", found "<<(found); throwException(__FILE__, __LINE__, details.str());}};

#define ASSERT_EQUAL_VEC_MOD(expected, found, tol,testname) {ASSERT_EQUAL_TOL_MOD((expected)[0], (found)[0], (tol),(testname)); ASSERT_EQUAL_TOL_MOD((expected)[1], (found)[1], (tol),(testname)); ASSERT_EQUAL_TOL_MOD((expected)[2], (found)[2], (tol),(testname));};


using namespace OpenMM;
using namespace std;

extern "C" void registerMPIDCudaKernelFactories();

const double TOL = 1e-4;

void make_charge_square(double boxEdgeLength, vector<Vec3> &positions, MPIDForce *forceField, System &system)
{
    positions.clear();

    // 0  3
    // |  |
    // 1--2

    // Atom 0
    positions.push_back(Vec3(0.1, 0.0, 0.0));
    forceField->addMultipole(1.0, {0,0,0}, {0,0,0,0,0,0}, {0,0,0,0,0,0,0,0,0,0}, MPIDForce::NoAxisType, 0, 0, 0, 0.0, {0.0, 0.0, 0.0});
    system.addParticle(1.0);
    forceField->setCovalentMap(0, MPIDForce::Covalent12, {1});
    forceField->setCovalentMap(0, MPIDForce::Covalent13, {2});
    forceField->setCovalentMap(0, MPIDForce::Covalent14, {3});

    // Atom 1
    positions.push_back(Vec3(0.0, 0.0, 0.0));
    forceField->addMultipole(1.0, {0,0,0}, {0,0,0,0,0,0}, {0,0,0,0,0,0,0,0,0,0}, MPIDForce::NoAxisType, 0, 0, 0, 0.0, {0.0, 0.0, 0.0});
    system.addParticle(1.0);
    forceField->setCovalentMap(1, MPIDForce::Covalent12, {0, 2});
    forceField->setCovalentMap(1, MPIDForce::Covalent13, {3});

    // Atom 2
    positions.push_back(Vec3(0.0, 0.1, 0.0));
    forceField->addMultipole(-1.0, {0,0,0}, {0,0,0,0,0,0}, {0,0,0,0,0,0,0,0,0,0}, MPIDForce::NoAxisType, 0, 0, 0, 0.0, {0.0, 0.0, 0.0});
    system.addParticle(1.0);
    forceField->setCovalentMap(2, MPIDForce::Covalent12, {1, 3});
    forceField->setCovalentMap(2, MPIDForce::Covalent13, {0});

    // Atom 3
    positions.push_back(Vec3(0.1, 0.1, 0.0));
    forceField->addMultipole(-1.0, {0,0,0}, {0,0,0,0,0,0}, {0,0,0,0,0,0,0,0,0,0}, MPIDForce::NoAxisType, 0, 0, 0, 0.0, {0.0, 0.0, 0.0});
    system.addParticle(1.0);
    forceField->setCovalentMap(3, MPIDForce::Covalent12, {2});
    forceField->setCovalentMap(3, MPIDForce::Covalent13, {1});
    forceField->setCovalentMap(3, MPIDForce::Covalent14, {0});

    system.setDefaultPeriodicBoxVectors(Vec3(boxEdgeLength, 0, 0),
                                        Vec3(0, boxEdgeLength, 0),
                                        Vec3(0, 0, boxEdgeLength));

}


void make_waterbox(int natoms, double boxEdgeLength, MPIDForce *forceField,  vector<Vec3> &positions, System &system,
                   bool do_charge = true, bool do_dpole = true, bool do_qpole = true, bool do_opole = true, bool do_pol = true)
{
    std::map < std::string, double > tholemap;
    std::map < std::string, std::vector<double>> polarmap;
    std::map < std::string, double > chargemap;
    std::map < std::string, std::vector<double> > dipolemap;
    std::map < std::string, std::vector<double> > quadrupolemap;
    std::map < std::string, std::vector<double> > octopolemap;
    std::map < std::string, MPIDForce::MultipoleAxisTypes > axesmap;
    std::map < std::string, std::vector<int> > anchormap;
    std::map < std::string, double > massmap;
    std::map < std::string, std::vector<int> > polgrpmap;
    std::map < std::string, std::vector<int> > cov12map;
    std::map < std::string, std::vector<int> > cov13map;

    axesmap["O"]  = MPIDForce::Bisector;
    axesmap["H1"] = MPIDForce::ZThenX;
    axesmap["H2"] = MPIDForce::ZThenX;

    chargemap["O"]  = -0.51966;
    chargemap["H1"] = 0.25983;
    chargemap["H2"] = 0.25983;
    if(!do_charge){
        chargemap["O"]  = 0.0;
        chargemap["H1"] = 0.0;
        chargemap["H2"] = 0.0;
    }

    int oanc[3] = {1, 2, 0};
    int h1anc[3] = {-1, 1, 0};
    int h2anc[3] = {-2, -1, 0};
    std::vector<int> oancv(&oanc[0], &oanc[3]);
    std::vector<int> h1ancv(&h1anc[0], &h1anc[3]);
    std::vector<int> h2ancv(&h2anc[0], &h2anc[3]);
    anchormap["O"]  = oancv;
    anchormap["H1"] = h1ancv;
    anchormap["H2"] = h2ancv;

    double od[3] = {0.0, 0.0, 0.00755612136146};
    double hd[3] = {-0.00204209484795, 0.0, -0.00307875299958};
    std::vector<double> odv(&od[0], &od[3]);
    std::vector<double> hdv(&hd[0], &hd[3]);
    if(!do_dpole){
        odv.assign(3, 0);
        hdv.assign(3, 0);
    }
    dipolemap["O"]  = odv;
    dipolemap["H1"] = hdv;
    dipolemap["H2"] = hdv;

    double oq[6] = {0.000354030721139, 0.0, -0.000390257077096, 0.0, 0.0,  3.62263559571e-05};
    double hq[6] = {-3.42848248983e-05, 0.0, -0.000100240875193, -1.89485963908e-06, 0.0,  0.000134525700091};

    std::vector<double> oqv(&oq[0], &oq[6]);
    std::vector<double> hqv(&hq[0], &hq[6]);
    if(!do_qpole){
        oqv.assign(6, 0);
        hqv.assign(6, 0);
    }
    quadrupolemap["O"]  = oqv;
    quadrupolemap["H1"] = hqv;
    quadrupolemap["H2"] = hqv;

    double oo[10] = { 0, 0, 0, 0, -6.285758282686837e-07, 0, -9.452653225954594e-08, 0, 0, 7.231018665791977e-07};
    double ho[10] = { -2.405600937552608e-07, 0, -6.415084018183151e-08, 0, -1.152422607026746e-06,
                      0,  -2.558537436767218e-06, 3.047102424084479e-07, 0, 3.710960043793964e-06 };
    std::vector<double> oov(&oo[0], &oo[10]);
    std::vector<double> hov(&ho[0], &ho[10]);
    if(!do_opole){
        oov.assign(10, 0);
        hov.assign(10, 0);
    }
    octopolemap["O"]  = oov;
    octopolemap["H1"] = hov;
    octopolemap["H2"] = hov;

    polarmap["O"]  = std::vector<double>{0.000837, 0.000837, 0.000837};
    polarmap["H1"] = std::vector<double>{0.000496, 0.000496, 0.000496};
    polarmap["H2"] = std::vector<double>{0.000496, 0.000496, 0.000496};

    tholemap["O"]  = 0.3900;
    tholemap["H1"] = 0.3900;
    tholemap["H2"] = 0.3900;

    massmap["O"]  = 15.999;
    massmap["H1"] = 1.0080000;
    massmap["H2"] = 1.0080000;

    int opg[3] = {0,1,2};
    int h1pg[3] = {-1,0,1};
    int h2pg[3] = {-2,-1,0};
    std::vector<int> opgv(&opg[0], &opg[3]);
    std::vector<int> h1pgv(&h1pg[0], &h1pg[3]);
    std::vector<int> h2pgv(&h2pg[0], &h2pg[3]);
    polgrpmap["O"] = opgv;
    polgrpmap["H1"] = h1pgv;
    polgrpmap["H2"] = h2pgv;

    int cov12o[2] = {1,2};
    int cov12h1[1] = {-1};
    int cov12h2[1] = {-2};
    std::vector<int> cov12ov(&cov12o[0], &cov12o[2]);
    std::vector<int> cov12h1v(&cov12h1[0], &cov12h1[1]);
    std::vector<int> cov12h2v(&cov12h2[0], &cov12h2[1]);
    cov12map["O"] = cov12ov;
    cov12map["H1"] = cov12h1v;
    cov12map["H2"] = cov12h2v;

    int cov13h1[1] = {1};
    int cov13h2[1] = {-1};
    std::vector<int> cov13h1v(&cov13h1[0], &cov13h1[1]);
    std::vector<int> cov13h2v(&cov13h2[0], &cov13h2[1]);
    cov13map["O"] = std::vector<int>();
    cov13map["H1"] = cov13h1v;
    cov13map["H2"] = cov13h2v;
    positions.clear();
    if (natoms == 6) {
        const double coords[6][3] = {
            {  2.000000, 2.000000, 2.000000},
            {  2.500000, 2.000000, 3.000000},
            {  1.500000, 2.000000, 3.000000},
            {  0.000000, 0.000000, 0.000000},
            {  0.500000, 0.000000, 1.000000},
            { -0.500000, 0.000000, 1.000000}
        };
        for (int atom = 0; atom < natoms; ++atom)
            positions.push_back(Vec3(coords[atom][0], coords[atom][1], coords[atom][2])*OpenMM::NmPerAngstrom);
    }
    else if (natoms == 375) {
        const double coords[375][3] = {
            { -6.22, -6.25, -6.24 },
            { -5.32, -6.03, -6.00 },
            { -6.75, -5.56, -5.84 },
            { -3.04, -6.23, -6.19 },
            { -3.52, -5.55, -5.71 },
            { -3.59, -6.43, -6.94 },
            {  0.02, -6.23, -6.14 },
            { -0.87, -5.97, -6.37 },
            {  0.53, -6.03, -6.93 },
            {  3.10, -6.20, -6.27 },
            {  3.87, -6.35, -5.72 },
            {  2.37, -6.11, -5.64 },
            {  6.18, -6.14, -6.20 },
            {  6.46, -6.66, -5.44 },
            {  6.26, -6.74, -6.94 },
            { -6.21, -3.15, -6.24 },
            { -6.23, -3.07, -5.28 },
            { -6.02, -2.26, -6.55 },
            { -3.14, -3.07, -6.16 },
            { -3.38, -3.63, -6.90 },
            { -2.18, -3.05, -6.17 },
            { -0.00, -3.16, -6.23 },
            { -0.03, -2.30, -6.67 },
            {  0.05, -2.95, -5.29 },
            {  3.08, -3.11, -6.14 },
            {  2.65, -2.55, -6.79 },
            {  3.80, -3.53, -6.62 },
            {  6.16, -3.14, -6.16 },
            {  7.04, -3.32, -6.51 },
            {  5.95, -2.27, -6.51 },
            { -6.20, -0.04, -6.15 },
            { -5.43,  0.32, -6.59 },
            { -6.95,  0.33, -6.62 },
            { -3.10, -0.06, -6.19 },
            { -3.75,  0.42, -6.69 },
            { -2.46,  0.60, -5.93 },
            {  0.05, -0.01, -6.17 },
            { -0.10,  0.02, -7.12 },
            { -0.79,  0.16, -5.77 },
            {  3.03,  0.00, -6.19 },
            {  3.54,  0.08, -7.01 },
            {  3.69, -0.22, -5.53 },
            {  6.17,  0.05, -6.19 },
            {  5.78, -0.73, -6.57 },
            {  7.09, -0.17, -6.04 },
            { -6.20,  3.15, -6.25 },
            { -6.59,  3.18, -5.37 },
            { -5.87,  2.25, -6.33 },
            { -3.09,  3.04, -6.17 },
            { -3.88,  3.58, -6.26 },
            { -2.41,  3.54, -6.63 },
            {  0.00,  3.06, -6.26 },
            { -0.71,  3.64, -6.00 },
            {  0.65,  3.15, -5.55 },
            {  3.14,  3.06, -6.23 },
            {  3.11,  3.31, -5.30 },
            {  2.38,  3.49, -6.63 },
            {  6.19,  3.14, -6.25 },
            {  6.82,  3.25, -5.54 },
            {  5.76,  2.30, -6.07 },
            { -6.22,  6.26, -6.19 },
            { -6.22,  5.74, -7.00 },
            { -5.89,  5.67, -5.52 },
            { -3.04,  6.24, -6.20 },
            { -3.08,  5.28, -6.17 },
            { -3.96,  6.52, -6.25 },
            { -0.05,  6.21, -6.16 },
            {  0.82,  6.58, -6.06 },
            {  0.01,  5.64, -6.93 },
            {  3.10,  6.25, -6.15 },
            {  3.64,  5.47, -6.31 },
            {  2.46,  6.24, -6.87 },
            {  6.22,  6.20, -6.27 },
            {  5.37,  6.42, -5.88 },
            {  6.80,  6.07, -5.51 },
            { -6.19, -6.15, -3.13 },
            { -6.37, -7.01, -3.51 },
            { -6.25, -6.29, -2.18 },
            { -3.10, -6.27, -3.11 },
            { -2.29, -5.77, -2.99 },
            { -3.80, -5.62, -2.98 },
            { -0.03, -6.18, -3.15 },
            { -0.07, -7.05, -2.75 },
            {  0.68, -5.74, -2.70 },
            {  3.10, -6.14, -3.07 },
            {  2.35, -6.72, -3.23 },
            {  3.86, -6.65, -3.37 },
            {  6.22, -6.20, -3.16 },
            {  6.82, -6.36, -2.43 },
            {  5.35, -6.13, -2.75 },
            { -6.26, -3.13, -3.12 },
            { -6.16, -2.27, -2.70 },
            { -5.36, -3.47, -3.18 },
            { -3.11, -3.05, -3.14 },
            { -3.31, -3.96, -3.34 },
            { -2.77, -3.06, -2.24 },
            {  0.00, -3.13, -3.16 },
            {  0.48, -2.37, -2.81 },
            { -0.57, -3.40, -2.44 },
            {  3.09, -3.09, -3.16 },
            {  2.41, -3.19, -2.49 },
            {  3.91, -3.07, -2.67 },
            {  6.19, -3.04, -3.08 },
            {  5.64, -3.61, -3.61 },
            {  6.93, -3.58, -2.82 },
            { -6.18, -0.00, -3.04 },
            { -6.00, -0.59, -3.78 },
            { -6.79,  0.64, -3.39 },
            { -3.05, -0.03, -3.07 },
            { -2.95,  0.80, -3.52 },
            { -4.00, -0.20, -3.07 },
            { -0.03,  0.03, -3.06 },
            { -0.33, -0.37, -3.87 },
            {  0.89, -0.21, -2.99 },
            {  3.13, -0.05, -3.10 },
            {  3.44,  0.81, -3.34 },
            {  2.21,  0.07, -2.86 },
            {  6.20, -0.05, -3.13 },
            {  6.89,  0.60, -3.20 },
            {  5.58,  0.30, -2.49 },
            { -6.23,  3.09, -3.16 },
            { -5.62,  3.79, -2.94 },
            { -6.33,  2.60, -2.33 },
            { -3.10,  3.08, -3.04 },
            { -3.84,  3.47, -3.51 },
            { -2.40,  3.01, -3.69 },
            {  0.01,  3.04, -3.11 },
            { -0.56,  3.59, -3.64 },
            {  0.28,  3.60, -2.38 },
            {  3.04,  3.11, -3.09 },
            {  3.49,  2.30, -2.87 },
            {  3.70,  3.66, -3.51 },
            {  6.15,  3.14, -3.11 },
            {  6.52,  2.52, -3.74 },
            {  6.72,  3.06, -2.34 },
            { -6.22,  6.15, -3.13 },
            { -5.49,  6.21, -2.51 },
            { -6.56,  7.04, -3.18 },
            { -3.11,  6.24, -3.05 },
            { -3.76,  5.83, -3.62 },
            { -2.26,  5.92, -3.37 },
            {  0.03,  6.25, -3.07 },
            {  0.34,  5.63, -3.73 },
            { -0.87,  6.00, -2.91 },
            {  3.07,  6.15, -3.08 },
            {  3.29,  6.92, -2.56 },
            {  3.39,  6.35, -3.96 },
            {  6.22,  6.14, -3.12 },
            {  5.79,  6.38, -2.29 },
            {  6.25,  6.96, -3.62 },
            { -6.21, -6.20, -0.06 },
            { -5.79, -6.87,  0.48 },
            { -6.43, -5.50,  0.54 },
            { -3.16, -6.21, -0.02 },
            { -2.50, -6.87,  0.20 },
            { -2.77, -5.37,  0.23 },
            { -0.00, -6.14, -0.00 },
            {  0.68, -6.72, -0.33 },
            { -0.64, -6.73,  0.38 },
            {  3.03, -6.20, -0.01 },
            {  3.77, -6.56, -0.51 },
            {  3.43, -5.85,  0.78 },
            {  6.25, -6.16, -0.00 },
            {  5.36, -6.09, -0.36 },
            {  6.24, -6.97,  0.49 },
            { -6.24, -3.05, -0.01 },
            { -6.35, -3.64,  0.73 },
            { -5.42, -3.33, -0.42 },
            { -3.09, -3.06,  0.05 },
            { -2.44, -3.62, -0.38 },
            { -3.90, -3.21, -0.43 },
            {  0.05, -3.10,  0.02 },
            { -0.31, -2.35, -0.43 },
            { -0.63, -3.77,  0.01 },
            {  3.05, -3.09, -0.04 },
            {  3.28, -3.90,  0.41 },
            {  3.65, -2.43,  0.30 },
            {  6.20, -3.04, -0.03 },
            {  5.66, -3.31,  0.71 },
            {  6.78, -3.79, -0.19 },
            { -6.18,  0.04, -0.04 },
            { -6.73, -0.73, -0.15 },
            { -5.98,  0.06,  0.89 },
            { -3.11, -0.04, -0.04 },
            { -3.36, -0.08,  0.87 },
            { -2.70,  0.81, -0.14 },
            { -0.02, -0.02, -0.05 },
            { -0.45,  0.28,  0.75 },
            {  0.90,  0.15,  0.07 },
            {  3.04,  0.02, -0.01 },
            {  3.26, -0.82,  0.38 },
            {  3.89,  0.45, -0.13 },
            {  6.19,  0.05, -0.03 },
            {  5.52, -0.56,  0.25 },
            {  7.01, -0.29,  0.32 },
            { -6.14,  3.08,  0.00 },
            { -6.83,  2.82,  0.61 },
            { -6.59,  3.64, -0.64 },
            { -3.05,  3.09, -0.04 },
            { -3.79,  2.50,  0.09 },
            { -3.18,  3.80,  0.59 },
            {  0.02,  3.14,  0.04 },
            { -0.89,  3.04, -0.19 },
            {  0.49,  2.57, -0.57 },
            {  3.14,  3.15,  0.00 },
            {  3.28,  2.28,  0.37 },
            {  2.30,  3.08, -0.45 },
            {  6.27,  3.08, -0.00 },
            {  5.55,  2.54, -0.33 },
            {  5.83,  3.87,  0.34 },
            { -6.18,  6.15, -0.03 },
            { -6.45,  6.21,  0.88 },
            { -6.26,  7.05, -0.36 },
            { -3.06,  6.19, -0.05 },
            { -2.84,  6.64,  0.76 },
            { -3.99,  5.96,  0.03 },
            { -0.00,  6.20,  0.06 },
            { -0.67,  5.99, -0.59 },
            {  0.76,  6.46, -0.44 },
            {  3.10,  6.26, -0.03 },
            {  3.57,  6.09,  0.78 },
            {  2.57,  5.47, -0.18 },
            {  6.26,  6.18,  0.02 },
            {  5.53,  5.64, -0.29 },
            {  5.95,  7.08, -0.06 },
            { -6.26, -6.21,  3.07 },
            { -5.98, -6.38,  3.97 },
            { -5.46, -5.94,  2.62 },
            { -3.10, -6.24,  3.04 },
            { -2.69, -6.51,  3.87 },
            { -3.43, -5.35,  3.21 },
            { -0.03, -6.16,  3.06 },
            {  0.83, -6.00,  3.42 },
            { -0.30, -6.99,  3.45 },
            {  3.15, -6.25,  3.11 },
            {  2.77, -5.60,  3.72 },
            {  2.68, -6.10,  2.28 },
            {  6.20, -6.21,  3.16 },
            {  5.75, -6.73,  2.50 },
            {  6.69, -5.56,  2.66 },
            { -6.17, -3.10,  3.04 },
            { -6.82, -2.44,  3.28 },
            { -6.12, -3.69,  3.80 },
            { -3.08, -3.04,  3.11 },
            { -3.59, -3.56,  3.72 },
            { -2.97, -3.61,  2.34 },
            {  0.01, -3.04,  3.11 },
            { -0.86, -3.41,  3.20 },
            {  0.56, -3.78,  2.86 },
            {  3.07, -3.07,  3.15 },
            {  3.81, -3.68,  3.13 },
            {  2.80, -2.98,  2.23 },
            {  6.20, -3.04,  3.13 },
            {  5.48, -3.64,  2.92 },
            {  6.98, -3.49,  2.81 },
            { -6.18, -0.05,  3.12 },
            { -6.41,  0.66,  3.69 },
            { -6.33,  0.28,  2.23 },
            { -3.05,  0.03,  3.10 },
            { -3.46, -0.42,  3.83 },
            { -3.57, -0.19,  2.33 },
            {  0.03, -0.02,  3.15 },
            {  0.23, -0.08,  2.21 },
            { -0.81,  0.41,  3.18 },
            {  3.09,  0.00,  3.03 },
            {  2.48, -0.29,  3.71 },
            {  3.91,  0.16,  3.51 },
            {  6.19, -0.06,  3.11 },
            {  6.05,  0.47,  2.33 },
            {  6.59,  0.52,  3.74 },
            { -6.20,  3.05,  3.05 },
            { -6.87,  3.73,  3.17 },
            { -5.55,  3.24,  3.73 },
            { -3.11,  3.06,  3.15 },
            { -3.64,  3.74,  2.71 },
            { -2.32,  3.00,  2.62 },
            {  0.02,  3.05,  3.06 },
            { -0.87,  3.14,  3.38 },
            {  0.48,  3.82,  3.42 },
            {  3.07,  3.10,  3.16 },
            {  3.95,  3.44,  2.97 },
            {  2.76,  2.73,  2.32 },
            {  6.19,  3.07,  3.16 },
            {  7.02,  3.30,  2.72 },
            {  5.52,  3.27,  2.51 },
            { -6.19,  6.24,  3.15 },
            { -5.56,  5.88,  2.52 },
            { -7.05,  5.96,  2.83 },
            { -3.10,  6.14,  3.08 },
            { -2.34,  6.69,  3.27 },
            { -3.86,  6.69,  3.29 },
            { -0.04,  6.24,  3.13 },
            {  0.63,  6.54,  2.53 },
            {  0.08,  5.29,  3.18 },
            {  3.12,  6.24,  3.14 },
            {  3.57,  5.82,  2.40 },
            {  2.23,  5.90,  3.12 },
            {  6.25,  6.19,  3.06 },
            {  5.55,  5.59,  3.32 },
            {  6.08,  6.99,  3.55 },
            { -6.20, -6.16,  6.15 },
            { -6.29, -5.99,  7.09 },
            { -6.09, -7.11,  6.09 },
            { -3.09, -6.19,  6.27 },
            { -2.56, -5.90,  5.52 },
            { -3.80, -6.69,  5.87 },
            {  0.02, -6.25,  6.24 },
            { -0.70, -5.70,  6.51 },
            {  0.25, -5.93,  5.36 },
            {  3.11, -6.18,  6.14 },
            {  3.76, -6.54,  6.74 },
            {  2.29, -6.20,  6.64 },
            {  6.22, -6.17,  6.15 },
            {  6.61, -6.98,  6.47 },
            {  5.56, -5.94,  6.81 },
            { -6.21, -3.10,  6.14 },
            { -6.76, -2.66,  6.78 },
            { -5.51, -3.50,  6.65 },
            { -3.13, -3.05,  6.18 },
            { -2.19, -3.14,  6.34 },
            { -3.50, -3.89,  6.43 },
            {  0.01, -3.06,  6.15 },
            { -0.06, -2.81,  7.07 },
            { -0.25, -3.98,  6.13 },
            {  3.04, -3.09,  6.17 },
            {  3.84, -3.51,  5.84 },
            {  3.25, -2.85,  7.08 },
            {  6.26, -3.13,  6.19 },
            {  6.01, -2.20,  6.09 },
            {  5.47, -3.55,  6.54 },
            { -6.20,  0.01,  6.27 },
            { -5.79, -0.70,  5.78 },
            { -6.67,  0.51,  5.60 },
            { -3.13,  0.01,  6.14 },
            { -3.53, -0.35,  6.94 },
            { -2.21,  0.17,  6.39 },
            { -0.04, -0.04,  6.20 },
            {  0.26,  0.47,  5.46 },
            {  0.51,  0.22,  6.93 },
            {  3.10, -0.05,  6.23 },
            {  2.33,  0.44,  5.95 },
            {  3.85,  0.45,  5.92 },
            {  6.19, -0.01,  6.26 },
            {  7.05,  0.16,  5.88 },
            {  5.58,  0.02,  5.52 },
            { -6.22,  3.04,  6.17 },
            { -5.45,  3.57,  5.95 },
            { -6.62,  3.50,  6.92 },
            { -3.09,  3.16,  6.21 },
            { -3.71,  2.75,  5.61 },
            { -2.60,  2.43,  6.59 },
            { -0.02,  3.10,  6.26 },
            {  0.89,  3.27,  6.05 },
            { -0.44,  2.94,  5.41 },
            {  3.12,  3.04,  6.23 },
            {  2.31,  3.53,  6.43 },
            {  3.59,  3.60,  5.60 },
            {  6.23,  3.05,  6.24 },
            {  5.92,  3.91,  6.54 },
            {  6.02,  3.03,  5.30 },
            { -6.15,  6.21,  6.24 },
            { -6.27,  6.46,  5.32 },
            { -7.00,  5.85,  6.51 },
            { -3.07,  6.15,  6.22 },
            { -3.98,  6.27,  5.94 },
            { -2.66,  7.01,  6.10 },
            {  0.04,  6.20,  6.25 },
            { -0.38,  5.50,  5.75 },
            { -0.36,  7.00,  5.93 },
            {  3.12,  6.15,  6.24 },
            {  3.66,  6.88,  5.93 },
            {  2.25,  6.33,  5.86 },
            {  6.20,  6.27,  6.19 },
            {  5.46,  5.65,  6.19 },
            {  6.97,  5.73,  6.39 }
        };
        for (int atom = 0; atom < natoms; ++atom)
            positions.push_back(Vec3(coords[atom][0], coords[atom][1], coords[atom][2])*OpenMM::NmPerAngstrom);
    }
    else
        throw exception();

    system.setDefaultPeriodicBoxVectors(Vec3(boxEdgeLength, 0, 0),
                                        Vec3(0, boxEdgeLength, 0),
                                        Vec3(0, 0, boxEdgeLength));

    const char* atom_types[3] = {"O", "H1", "H2"};
    for(int atom = 0; atom < natoms; ++atom){
        const char* element = atom_types[atom%3];
        std::vector<double> &alpha = polarmap[element];
        if(!do_pol) alpha = std::vector<double> {0, 0, 0};
        int atomz = atom + anchormap[element][0];
        int atomx = atom + anchormap[element][1];
        int atomy = anchormap[element][2]==0 ? -1 : atom + anchormap[element][2];
        forceField->addMultipole(chargemap[element], dipolemap[element], quadrupolemap[element], octopolemap[element],
                                 axesmap[element], atomz, atomx, atomy, tholemap[element], alpha);
        system.addParticle(massmap[element]);
        // Polarization groups
        std::vector<int> tmppol;
        std::vector<int>& polgrps = polgrpmap[element];
        for(int i=0; i < polgrps.size(); ++i)
            tmppol.push_back(polgrps[i]+atom);
        if(!tmppol.empty())
           forceField->setCovalentMap(atom, MPIDForce::PolarizationCovalent11, tmppol);
        // 1-2 covalent groups
        std::vector<int> tmp12;
        std::vector<int>& cov12s = cov12map[element];
        for(int i=0; i < cov12s.size(); ++i)
            tmp12.push_back(cov12s[i]+atom);
        if(!tmp12.empty())
           forceField->setCovalentMap(atom, MPIDForce::Covalent12, tmp12);
        // 1-3 covalent groups
        std::vector<int> tmp13;
        std::vector<int>& cov13s = cov13map[element];
        for(int i=0; i < cov13s.size(); ++i)
            tmp13.push_back(cov13s[i]+atom);
        if(!tmp13.empty())
           forceField->setCovalentMap(atom, MPIDForce::Covalent13, tmp13);
    }
}


void make_methanolbox(int natoms, double boxEdgeLength, MPIDForce *forceField,  vector<Vec3> &positions, System &system,
                      bool do_charge = true, bool do_dpole  = true, bool do_qpole = true, bool do_opole = true, bool do_pol = true)
{
    std::map < std::string, double > tholemap;
    std::map < std::string, std::vector<double>> polarmap;
    std::map < std::string, double > chargemap;
    std::map < std::string, std::vector<double> > dipolemap;
    std::map < std::string, std::vector<double> > quadrupolemap;
    std::map < std::string, std::vector<double> > octopolemap;
    std::map < std::string, MPIDForce::MultipoleAxisTypes > axesmap;
    std::map < std::string, std::vector<int> > anchormap;
    std::map < std::string, double > massmap;
    std::map < std::string, std::vector<int> > cov12map;
    std::map < std::string, std::vector<int> > cov13map;

    const char* atom_types[6] = {"C1", "O1", "HO1", "H1A", "H1B", "H1C"};
    massmap["C1"]  = 12.01100;
    massmap["O1"]  = 15.999;
    massmap["HO1"] = 1.0080000;
    massmap["H1A"] = 1.0080000;
    massmap["H1B"] = 1.0080000;
    massmap["H1C"] = 1.0080000;

    axesmap["C1"]  = MPIDForce::ZOnly;
    axesmap["O1"]  = MPIDForce::ZThenX;
    axesmap["HO1"] = MPIDForce::ZOnly;
    axesmap["H1A"] = MPIDForce::ZOnly;
    axesmap["H1B"] = MPIDForce::ZOnly;
    axesmap["H1C"] = MPIDForce::ZOnly;


    chargemap["C1"]  = -0.140;
    chargemap["O1"]  = -0.460;
    chargemap["HO1"] =  0.360;
    chargemap["H1A"] =  0.080;
    chargemap["H1B"] =  0.080;
    chargemap["H1C"] =  0.080;
    if(!do_charge){
        chargemap["C1"]  =  0.0;
        chargemap["O1"]  =  0.0;
        chargemap["HO1"] =  0.0;
        chargemap["H1A"] =  0.0;
        chargemap["H1B"] =  0.0;
        chargemap["H1C"] =  0.0;
    }

    int c1anc[3] = { 1, 0, 0};
    int o1anc[3] = {-1, 1, 0};
    int ho1anc[3] = {-1, 0, 0};
    int h1aanc[3] = {-3, 0, 0};
    int h1banc[3] = {-4, 0, 0};
    int h1canc[3] = {-5, 0, 0};
    std::vector<int> c1ancv(&c1anc[0], &c1anc[3]);
    std::vector<int> o1ancv(&o1anc[0], &o1anc[3]);
    std::vector<int> ho1ancv(&ho1anc[0], &ho1anc[3]);
    std::vector<int> h1aancv(&h1aanc[0], &h1aanc[3]);
    std::vector<int> h1bancv(&h1banc[0], &h1banc[3]);
    std::vector<int> h1cancv(&h1canc[0], &h1canc[3]);
    anchormap["C1"]  = c1ancv;
    anchormap["O1"]  = o1ancv;
    anchormap["HO1"] = ho1ancv;
    anchormap["H1A"] = h1aancv;
    anchormap["H1B"] = h1bancv;
    anchormap["H1C"] = h1cancv;

    double od[3] = {
            0.00026405942708641,                      0,    0.00550661803258754
    };
    double zerod[3] = {
                              0,                      0,                      0
    };
    std::vector<double> odv(&od[0], &od[3]);
    std::vector<double> zerodv(&zerod[0], &zerod[3]);
    if(!do_dpole){
        odv.assign(3, 0);
    }
    dipolemap["C1"]  = zerodv;
    dipolemap["O1"]  = odv;
    dipolemap["HO1"] = zerodv;
    dipolemap["H1A"] = zerodv;
    dipolemap["H1B"] = zerodv;
    dipolemap["H1C"] = zerodv;

    double oq[6] = { 9.383755641232907e-05, 0, -0.0001547997648007625, -1.577493985246555e-06, 0, 6.096220838843343e-05 };
    double zeroq[6] = { 0, 0, 0, 0, 0, 0 };

    std::vector<double> oqv(&oq[0], &oq[6]);
    std::vector<double> zeroqv(&zeroq[0], &zeroq[6]);
    if(!do_qpole){
        oqv.assign(6, 0);
    }
    quadrupolemap["C1"]  = zeroqv;
    quadrupolemap["O1"]  = oqv;
    quadrupolemap["HO1"] = zeroqv;
    quadrupolemap["H1A"] = zeroqv;
    quadrupolemap["H1B"] = zeroqv;
    quadrupolemap["H1C"] = zeroqv;

    double oo[10] = { -3.230426667733178e-08, 0,  3.684859776955582e-08, 0, -2.245492298396793e-07,
                      0, 7.675967953604524e-07,  -4.445541285871346e-09, 0, -5.43047565520773e-07};
    double zeroo[10] = { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 };
    std::vector<double> oov(&oo[0], &oo[10]);
    std::vector<double> zeroov(&zeroo[0], &zeroo[10]);
    if(!do_opole){
        oov.assign(10, 0);
    }
    octopolemap["C1"]  = zeroov;
    octopolemap["O1"]  = oov;
    octopolemap["HO1"] = zeroov;
    octopolemap["H1A"] = zeroov;
    octopolemap["H1B"] = zeroov;
    octopolemap["H1C"] = zeroov;

    std::vector<double> c1pol{0.00100000, 0.00100000, 0.00100000};
    std::vector<double> o1pol{0.00100024, 0.00125025, 0.00083350};
    std::vector<double> zeropol{0.0, 0.0, 0.0};
    polarmap["C1"]  = c1pol;
    polarmap["O1"]  = o1pol;
    polarmap["HO1"] = zeropol;
    polarmap["H1A"] = zeropol;
    polarmap["H1B"] = zeropol;
    polarmap["H1C"] = zeropol;

    tholemap["C1"]  = 1.3;
    tholemap["O1"]  = 1.3;
    tholemap["HO1"] = 0.0;
    tholemap["H1A"] = 0.0;
    tholemap["H1B"] = 0.0;
    tholemap["H1C"] = 0.0;

    int cov12c1[4] = {1,3,4,5};
    int cov12o1[2] = {-1,1};
    int cov12h01[1] = {-1};
    int cov12h1a[1] = {-3};
    int cov12h1b[1] = {-4};
    int cov12h1c[1] = {-5};
    std::vector<int> cov12c1v(&cov12c1[0], &cov12c1[4]);
    std::vector<int> cov12o1v(&cov12o1[0], &cov12o1[2]);
    std::vector<int> cov12h01v(&cov12h01[0], &cov12h01[1]);
    std::vector<int> cov12h1av(&cov12h1a[0], &cov12h1a[1]);
    std::vector<int> cov12h1bv(&cov12h1b[0], &cov12h1b[1]);
    std::vector<int> cov12h1cv(&cov12h1c[0], &cov12h1c[1]);
    cov12map["C1"]  = cov12c1v;
    cov12map["O1"]  = cov12o1v;
    cov12map["HO1"] = cov12h01v;
    cov12map["H1A"] = cov12h1av;
    cov12map["H1B"] = cov12h1bv;
    cov12map["H1C"] = cov12h1cv;

    int cov13c1[1] = {2};
    int cov13o1[3] = {2,3,4};
    int cov13h01[1] = {-1};
    int cov13h1a[3] = {-2,1,2};
    int cov13h1b[3] = {-3,-1,1};
    int cov13h1c[3] = {-4,-2,-1};
    std::vector<int> cov13c1v(&cov13c1[0], &cov13c1[1]);
    std::vector<int> cov13o1v(&cov13o1[0], &cov13o1[3]);
    std::vector<int> cov13h01v(&cov13h01[0], &cov13h01[1]);
    std::vector<int> cov13h1av(&cov13h1a[0], &cov13h1a[3]);
    std::vector<int> cov13h1bv(&cov13h1b[0], &cov13h1b[3]);
    std::vector<int> cov13h1cv(&cov13h1c[0], &cov13h1c[3]);
    cov13map["C1"]  = cov13c1v;
    cov13map["O1"]  = cov13o1v;
    cov13map["HO1"] = cov13h01v;
    cov13map["H1A"] = cov13h1av;
    cov13map["H1B"] = cov13h1bv;
    cov13map["H1C"] = cov13h1cv;
    positions.clear();
    if (natoms == 12) {
        const double coords[12][3] = {
            {  1.6118739816,  -7.7986654421,  -9.3388011053},
            {  0.4344388195,  -8.6290855266,  -9.4591523136},
            { -0.2932869802,  -8.1452383606,  -9.0381926002},
            {  1.5101797393,  -6.7361319725,  -9.6470249020},
            {  1.8800020341,  -7.7312323778,  -8.2627522535},
            {  2.3828031354,  -8.2685172700,  -9.9862796759},
            { -2.3016642008,  -3.3801483374,  -4.5239842701},
            { -2.6774345292,  -3.8370280231,  -3.2318499504},
            { -1.9568218092,  -3.4707595618,  -2.6956925837},
            { -1.4748236015,  -3.9573461155,  -4.9903514535},
            { -3.2561339708,  -3.3690912389,  -5.0924789477},
            { -1.9925289806,  -2.3413378186,  -4.2797935219},
        };
        for (int atom = 0; atom < natoms; ++atom)
            positions.push_back(Vec3(coords[atom][0], coords[atom][1], coords[atom][2])*OpenMM::NmPerAngstrom);
    }
    else
        throw exception();

    system.setDefaultPeriodicBoxVectors(Vec3(boxEdgeLength, 0, 0),
                                        Vec3(0, boxEdgeLength, 0),
                                        Vec3(0, 0, boxEdgeLength));

    for(int atom = 0; atom < natoms; ++atom){
        const char* element = atom_types[atom%6];
        std::vector<double> alphas = polarmap[element];
        if(!do_pol) alphas = std::vector<double>{0, 0, 0};
        int atomz = atom + anchormap[element][0];
        int atomx = anchormap[element][1]==0 ? -1 : atom + anchormap[element][1];
        int atomy = anchormap[element][2]==0 ? -1 : atom + anchormap[element][2];
        forceField->addMultipole(chargemap[element], dipolemap[element], quadrupolemap[element], octopolemap[element],
                                 axesmap[element], atomz, atomx, atomy, tholemap[element], alphas);
        system.addParticle(massmap[element]);
        // 1-2 covalent groups
        std::vector<int> tmp12;
        std::vector<int>& cov12s = cov12map[element];
        for(int i=0; i < cov12s.size(); ++i)
            tmp12.push_back(cov12s[i]+atom);
        if(!tmp12.empty())
           forceField->setCovalentMap(atom, MPIDForce::Covalent12, tmp12);
        // 1-3 covalent groups
        std::vector<int> tmp13;
        std::vector<int>& cov13s = cov13map[element];
        for(int i=0; i < cov13s.size(); ++i)
            tmp13.push_back(cov13s[i]+atom);
        if(!tmp13.empty())
           forceField->setCovalentMap(atom, MPIDForce::Covalent13, tmp13);
    }
}

static void check_full_finite_differences(vector<Vec3> analytic_forces, Context &context, vector<Vec3> positions, double stepSize = 1e-4, double tol=1e-4)
{
    // Take a small step in the direction of the energy gradient and see whether the potential energy changes by the expected amount.

    int natoms = analytic_forces.size();
    vector<Vec3> coords;
    std::copy(positions.begin(), positions.end(), std::back_inserter(coords));
    State state;
    for (int atom = 0; atom < natoms; ++atom) {
        for(int xyz = 0; xyz < 3; ++xyz){
            // Positive displacement
            coords[atom][xyz] += stepSize;
            context.setPositions(coords);
            state = context.getState(State::Energy);
            double Ep = state.getPotentialEnergy();
            coords[atom][xyz] -= stepSize;
            // Negative displacement
            coords[atom][xyz] -= stepSize;
            context.setPositions(coords);
            state = context.getState(State::Energy);
            double Em = state.getPotentialEnergy();
            coords[atom][xyz] += stepSize;
            // Check gradient component
            double findif = (Em - Ep) / (2.0*stepSize);
            cout << findif << "  " << analytic_forces[atom][xyz] <<  "  " << findif-analytic_forces[atom][xyz] << endl;
            ASSERT_EQUAL_TOL(findif, analytic_forces[atom][xyz], tol);
        }
    }
}


static void check_finite_differences(vector<Vec3> analytic_forces, Context &context, vector<Vec3> positions)
{
    // This is not a good test to perform in single precision
    if (context.getPlatform().getPropertyValue(context, "Precision") != "double") return;

    // Take a small step in the direction of the energy gradient and see whether the potential energy changes by the expected amount.
    double norm = 0.0;
    for (auto& f : analytic_forces)
        norm += f.dot(f);
    norm = std::sqrt(norm);
    const double stepSize = 1e-3;
    double step = 0.5*stepSize/norm;
    vector<Vec3> positions2(analytic_forces.size()), positions3(analytic_forces.size());
    for (int i = 0; i < (int) positions.size(); ++i) {
        Vec3 p = positions[i];
        Vec3 f = analytic_forces[i];
        positions2[i] = Vec3(p[0]-f[0]*step, p[1]-f[1]*step, p[2]-f[2]*step);
        positions3[i] = Vec3(p[0]+f[0]*step, p[1]+f[1]*step, p[2]+f[2]*step);
    }
    context.setPositions(positions2);
    State state2 = context.getState(State::Energy);
    context.setPositions(positions3);
    State state3 = context.getState(State::Energy);
    ASSERT_EQUAL_TOL(norm, (state2.getPotentialEnergy()-state3.getPotentialEnergy())/stepSize, 1e-4);
}


#define FMT(x) std::setprecision(10) << std::setw(16) << (x)
void print_energy_and_forces(double energy, const vector<Vec3>&forces)
{
    std::cout << "AKMA Units:" << std::endl;
    std::cout << "Energy:" << energy*OpenMM::KcalPerKJ << std::endl;
    size_t natoms = forces.size();
    std::cout << "Forces:" << std::endl;
    double sf = -OpenMM::KcalPerKJ/10.0;
    for(int i = 0; i < natoms; ++i){
        std::cout << i+1 << "\t" << FMT(forces[i][0]*sf) <<
                                    FMT(forces[i][1]*sf) <<
                                    FMT(forces[i][2]*sf) << std::endl;
    }
    std::cout << "SI Units:" << std::endl;
    std::cout << "Energy:" << energy << std::endl;
    std::cout << "Forces:" << std::endl;
    for(int i = 0; i < natoms; ++i){
        std::cout << i+1 << "\t" << FMT(forces[i][0]) <<
                                    FMT(forces[i][1]) <<
                                    FMT(forces[i][2]) << std::endl;
    }
    std::cout << "double refenergy = " << energy << ";" << endl;
    std::cout << "vector<Vec3> refforces(" << natoms << ");" << endl;
    int width = natoms>9 ? 2 : 1;
    for(int i = 0; i < natoms; ++i){
        cout << "refforces[" << std::setw(width) << i << "] = Vec3(" <<
                                    FMT(forces[i][0]) << "," <<
                                    FMT(forces[i][1]) << "," <<
                                    FMT(forces[i][2]) << ");" << std::endl;
    }

}


void testMethanolDimerEnergyAndForcesPMEDirect() {
    // Methanol box with anisotropic induced dipoles
    const double cutoff = 12.0*OpenMM::NmPerAngstrom;
    double boxEdgeLength = 24.61817*OpenMM::NmPerAngstrom;
    const double alpha = 4.5;
    const int grid = 64;
    MPIDForce* forceField = new MPIDForce();

    vector<Vec3> positions;
    System system;

    const int numAtoms = 12;

    bool do_charge = true;
    bool do_dpole  = true;
    bool do_qpole  = true;
    bool do_opole  = true;
    bool do_pol    = true;
    make_methanolbox(numAtoms, boxEdgeLength, forceField,  positions,  system,
                     do_charge, do_dpole, do_qpole, do_opole, do_pol);
    forceField->setNonbondedMethod(OpenMM::MPIDForce::PME);
    forceField->setPMEParameters(alpha, grid, grid, grid);
    forceField->setDefaultTholeWidth(3.0);
    forceField->setCutoffDistance(cutoff);
    forceField->setPolarizationType(MPIDForce::Direct);
    system.addForce(forceField);

    VerletIntegrator integrator(0.01);
    Context context(system, integrator, Platform::getPlatformByName("CUDA"));
    context.setPositions(positions);

    double refenergy = 100.048119;
    vector<Vec3> refforces(12);
    refforces[ 0] = Vec3(    0.4407512632,    0.9533272891,    0.2662227116);
    refforces[ 1] = Vec3(     2.777987186,     1.840858052,    -1.333640734);
    refforces[ 2] = Vec3(    -171.6344629,    -55.43492185,     12.32086698);
    refforces[ 3] = Vec3(     54.15942246,     41.78995026,    -18.66314014);
    refforces[ 4] = Vec3(      67.2048408,     12.09783528,     23.99010209);
    refforces[ 5] = Vec3(     46.52456681,    -2.469100554,    -16.59349719);
    refforces[ 6] = Vec3(    0.9792294678,    -1.511490917,    -2.188302524);
    refforces[ 7] = Vec3(     2.487119928,    -3.695492757,    -6.600407624);
    refforces[ 8] = Vec3(     12.53120547,    -46.68073149,        204.8161);
    refforces[ 9] = Vec3(     13.32630506,    -12.47148185,    -64.76480367);
    refforces[10] = Vec3(    -26.51166625,     3.163119144,    -46.15867739);
    refforces[11] = Vec3(     -2.28426176,     62.41964403,    -85.09330594);
    State state = context.getState(State::Forces | State::Energy);
    double energy = state.getPotentialEnergy();
    const vector<Vec3>& forces = state.getForces();
    //print_energy_and_forces(energy, forces);
    ASSERT_EQUAL_TOL(refenergy, energy, 1E-4);
    for (int n = 0; n < numAtoms; ++n)
        ASSERT_EQUAL_VEC(refforces[n], forces[n], 1E-4);
    check_finite_differences(forces, context, positions);
    //check_full_finite_differences(forces, context, positions, 1E-3, 1E-4);
}


void testMethanolDimerEnergyAndForcesNoCutDirect() {
    // Methanol box with anisotropic induced dipoles
    double boxEdgeLength = 24.61817*OpenMM::NmPerAngstrom;
    MPIDForce* forceField = new MPIDForce();

    vector<Vec3> positions;
    System system;

    const int numAtoms = 12;

    bool do_charge = true;
    bool do_dpole  = true;
    bool do_qpole  = true;
    bool do_opole  = true;
    bool do_pol    = true;
    make_methanolbox(numAtoms, boxEdgeLength, forceField,  positions,  system,
                     do_charge, do_dpole, do_qpole, do_opole, do_pol);
    forceField->setNonbondedMethod(OpenMM::MPIDForce::NoCutoff);
    forceField->setDefaultTholeWidth(3.0);
    forceField->setPolarizationType(MPIDForce::Direct);
    system.addForce(forceField);

    VerletIntegrator integrator(0.01);
    Context context(system, integrator, Platform::getPlatformByName("CUDA"));
    context.setPositions(positions);

    double refenergy = 100.1426571;
    vector<Vec3> refforces(12);
    refforces[ 0] = Vec3(    0.6174848862,     1.244067416,     0.348981665);
    refforces[ 1] = Vec3(     3.137044182,     3.021402996,    -1.106973251);
    refforces[ 2] = Vec3(    -171.9428202,    -56.29908471,     12.15602242);
    refforces[ 3] = Vec3(     54.08049294,     41.61038079,    -18.72605327);
    refforces[ 4] = Vec3(     67.10967846,     11.90263439,     23.93203305);
    refforces[ 5] = Vec3(     46.43698088,    -2.657546533,    -16.65203133);
    refforces[ 6] = Vec3(       1.1456374,    -1.273419433,    -2.195068492);
    refforces[ 7] = Vec3(     3.279289766,    -2.747123834,     -6.42588385);
    refforces[ 8] = Vec3(     11.97737104,    -47.37054551,     204.7348033);
    refforces[ 9] = Vec3(     13.21305881,    -12.64877074,    -64.79230192);
    refforces[10] = Vec3(    -26.64547941,     2.986613971,    -46.17381233);
    refforces[11] = Vec3(    -2.408738783,     62.23139119,      -85.099716);
    State state = context.getState(State::Forces | State::Energy);
    double energy = state.getPotentialEnergy();
    const vector<Vec3>& forces = state.getForces();
//    print_energy_and_forces(energy, forces);
    ASSERT_EQUAL_TOL(refenergy, energy, 1E-4);
    for (int n = 0; n < numAtoms; ++n)
        ASSERT_EQUAL_VEC(refforces[n], forces[n], 1E-4);
    check_finite_differences(forces, context, positions);
//    check_full_finite_differences(forces, context, positions, 1E-4, 1E-4);
}


void testMethanolDimerEnergyAndForcesPMEMutual() {
    // Methanol box with anisotropic induced dipoles
    const double cutoff = 12.0*OpenMM::NmPerAngstrom;
    double boxEdgeLength = 24.61817*OpenMM::NmPerAngstrom;
    const double alpha = 4.5;
    const int grid = 64;
    MPIDForce* forceField = new MPIDForce();

    vector<Vec3> positions;
    System system;

    const int numAtoms = 12;

    bool do_charge = true;
    bool do_dpole  = true;
    bool do_qpole  = true;
    bool do_opole  = true;
    bool do_pol    = true;
    make_methanolbox(numAtoms, boxEdgeLength, forceField,  positions,  system,
                     do_charge, do_dpole, do_qpole, do_opole, do_pol);
    forceField->setNonbondedMethod(OpenMM::MPIDForce::PME);
    forceField->setPMEParameters(alpha, grid, grid, grid);
    forceField->setDefaultTholeWidth(3.0);
    forceField->setCutoffDistance(cutoff);
    forceField->setPolarizationType(MPIDForce::Mutual);
    forceField->setMutualInducedTargetEpsilon(1e-9);
    system.addForce(forceField);

    VerletIntegrator integrator(0.01);
    Context context(system, integrator, Platform::getPlatformByName("CUDA"));
    context.setPositions(positions);

    double refenergy = 100.0480699;
    vector<Vec3> refforces(12);
    refforces[ 0] = Vec3(    0.4388490535,    0.9521698628,    0.2671848666);
    refforces[ 1] = Vec3(      2.77743403,     1.838052623,    -1.324369675);
    refforces[ 2] = Vec3(    -171.6330632,    -55.43091584,     12.31209076);
    refforces[ 3] = Vec3(     54.15929787,     41.79141719,    -18.66382841);
    refforces[ 4] = Vec3(     67.20544012,     12.09872923,     23.98850965);
    refforces[ 5] = Vec3(     46.52468443,    -2.468565193,    -16.59413613);
    refforces[ 6] = Vec3(    0.9812786184,    -1.522494372,     -2.19205581);
    refforces[ 7] = Vec3(     2.482142255,    -3.688423769,    -6.596946686);
    refforces[ 8] = Vec3(     12.53264302,     -46.6801623,     204.8169521);
    refforces[ 9] = Vec3(     13.32737364,    -12.47126727,    -64.76450554);
    refforces[10] = Vec3(    -26.51129625,      3.16340949,    -46.15822949);
    refforces[11] = Vec3(    -2.283744668,     62.41956649,    -85.09314563);
    State state = context.getState(State::Forces | State::Energy);
    double energy = state.getPotentialEnergy();
    const vector<Vec3>& forces = state.getForces();
//    print_energy_and_forces(energy, forces);
    ASSERT_EQUAL_TOL(refenergy, energy, 1E-4);
    for (int n = 0; n < numAtoms; ++n)
        ASSERT_EQUAL_VEC(refforces[n], forces[n], 1E-4);
    check_finite_differences(forces, context, positions);
//    check_full_finite_differences(forces, context, positions, 1E-7, 5E-4);
}


void testMethanolDimerEnergyAndForcesNoCutMutual() {
    // Methanol box with anisotropic induced dipoles
    double boxEdgeLength = 24.61817*OpenMM::NmPerAngstrom;
    MPIDForce* forceField = new MPIDForce();

    vector<Vec3> positions;
    System system;

    const int numAtoms = 12;

    bool do_charge = true;
    bool do_dpole  = true;
    bool do_qpole  = true;
    bool do_opole  = true;
    bool do_pol    = true;
    make_methanolbox(numAtoms, boxEdgeLength, forceField,  positions,  system,
                     do_charge, do_dpole, do_qpole, do_opole, do_pol);
    forceField->setNonbondedMethod(OpenMM::MPIDForce::NoCutoff);
    forceField->setDefaultTholeWidth(3.0);
    forceField->setPolarizationType(MPIDForce::Mutual);
    forceField->setMutualInducedTargetEpsilon(1e-9);
    system.addForce(forceField);

    VerletIntegrator integrator(0.01);
    Context context(system, integrator, Platform::getPlatformByName("CUDA"));
    context.setPositions(positions);

    double refenergy = 100.1424251;
    vector<Vec3> refforces(12);
    refforces[ 0] = Vec3(    0.6146835636,      1.24282615,    0.3501711642);
    refforces[ 1] = Vec3(      3.13797926,     3.015595591,    -1.097188673);
    refforces[ 2] = Vec3(    -171.9420969,    -56.29281618,     12.14742654);
    refforces[ 3] = Vec3(     54.08018498,     41.61211644,    -18.72680649);
    refforces[ 4] = Vec3(     67.11010165,     11.90386444,     23.93032756);
    refforces[ 5] = Vec3(     46.43703861,    -2.656708806,    -16.65281219);
    refforces[ 6] = Vec3(     1.148286492,    -1.284971833,    -2.197342953);
    refforces[ 7] = Vec3(     3.272556789,    -2.742411944,    -6.424646877);
    refforces[ 8] = Vec3(     11.97974305,    -47.36853963,     204.7356438);
    refforces[ 9] = Vec3(     13.21447088,    -12.64805524,    -64.79191248);
    refforces[10] = Vec3(    -26.64493967,     2.987395526,    -46.17328869);
    refforces[11] = Vec3(     -2.40800871,     62.23170548,    -85.09957069);
    State state = context.getState(State::Forces | State::Energy);
    double energy = state.getPotentialEnergy();
    const vector<Vec3>& forces = state.getForces();
//    print_energy_and_forces(energy, forces);
    ASSERT_EQUAL_TOL(refenergy, energy, 1E-4);
    for (int n = 0; n < numAtoms; ++n)
        ASSERT_EQUAL_VEC(refforces[n], forces[n], 1E-4);
    check_finite_differences(forces, context, positions);
//    check_full_finite_differences(forces, context, positions, 1E-7, 5E-4);
}


void testMethanolDimerEnergyAndForcesPMEExtrapolated() {
    // Methanol box with anisotropic induced dipoles
    const double cutoff = 12.0*OpenMM::NmPerAngstrom;
    double boxEdgeLength = 24.61817*OpenMM::NmPerAngstrom;
    const double alpha = 4.5;
    const int grid = 64;
    MPIDForce* forceField = new MPIDForce();

    vector<Vec3> positions;
    System system;

    const int numAtoms = 12;

    bool do_charge = true;
    bool do_dpole  = true;
    bool do_qpole  = true;
    bool do_opole  = true;
    bool do_pol    = true;
    make_methanolbox(numAtoms, boxEdgeLength, forceField,  positions,  system,
                     do_charge, do_dpole, do_qpole, do_opole, do_pol);
    forceField->setNonbondedMethod(OpenMM::MPIDForce::PME);
    forceField->setPMEParameters(alpha, grid, grid, grid);
    forceField->setDefaultTholeWidth(3.0);
    forceField->setCutoffDistance(cutoff);
    forceField->setPolarizationType(MPIDForce::Extrapolated);
    system.addForce(forceField);

    VerletIntegrator integrator(0.01);
    Context context(system, integrator, Platform::getPlatformByName("CUDA"));
    context.setPositions(positions);

    double refenergy = 100.0480906;
    vector<Vec3> refforces(12);
    refforces[ 0] = Vec3(     0.438635498,    0.9521799034,    0.2673223945);
    refforces[ 1] = Vec3(     2.777143984,     1.837985034,     -1.32278651);
    refforces[ 2] = Vec3(    -171.6327822,    -55.43082349,     12.31056078);
    refforces[ 3] = Vec3(     54.15935455,     41.79153141,    -18.66398402);
    refforces[ 4] = Vec3(     67.20558687,      12.0987565,      23.9882874);
    refforces[ 5] = Vec3(     46.52473894,     -2.46854161,    -16.59424569);
    refforces[ 6] = Vec3(    0.9814929295,    -1.523627818,    -2.192349756);
    refforces[ 7] = Vec3(     2.481269398,    -3.686996821,    -6.596448702);
    refforces[ 8] = Vec3(     12.53309724,    -46.68066444,     204.8169435);
    refforces[ 9] = Vec3(     13.32746646,    -12.47126075,    -64.76446756);
    refforces[10] = Vec3(    -26.51126553,     3.163426109,    -46.15818436);
    refforces[11] = Vec3(    -2.283699129,     62.41955233,    -85.09312721);
    State state = context.getState(State::Forces | State::Energy);
    double energy = state.getPotentialEnergy();
    const vector<Vec3>& forces = state.getForces();
    //print_energy_and_forces(energy, forces);
    ASSERT_EQUAL_TOL(refenergy, energy, 1E-4);
    for (int n = 0; n < numAtoms; ++n)
        ASSERT_EQUAL_VEC(refforces[n], forces[n], 1E-4);
    check_finite_differences(forces, context, positions);
//    check_full_finite_differences(forces, context, positions, 1E-4, 1E-4);
}


void testMethanolDimerEnergyAndForcesNoCutExtrapolated() {
    // Methanol box with anisotropic induced dipoles
    double boxEdgeLength = 24.61817*OpenMM::NmPerAngstrom;
    MPIDForce* forceField = new MPIDForce();

    vector<Vec3> positions;
    System system;

    const int numAtoms = 12;

    bool do_charge = true;
    bool do_dpole  = true;
    bool do_qpole  = true;
    bool do_opole  = true;
    bool do_pol    = true;
    make_methanolbox(numAtoms, boxEdgeLength, forceField,  positions,  system,
                     do_charge, do_dpole, do_qpole, do_opole, do_pol);
    forceField->setNonbondedMethod(OpenMM::MPIDForce::NoCutoff);
    forceField->setDefaultTholeWidth(3.0);
    forceField->setPolarizationType(MPIDForce::Extrapolated);
    system.addForce(forceField);

    VerletIntegrator integrator(0.01);
    Context context(system, integrator, Platform::getPlatformByName("CUDA"));
    context.setPositions(positions);

    double refenergy = 100.1424271;
    vector<Vec3> refforces(12);
    refforces[ 0] = Vec3(    0.6144183532,     1.242877198,    0.3502970714);
    refforces[ 1] = Vec3(      3.13789228,     3.015096318,    -1.095457146);
    refforces[ 2] = Vec3(     -171.941967,    -56.29241757,     12.14584015);
    refforces[ 3] = Vec3(     54.08021516,     41.61225418,    -18.72696219);
    refforces[ 4] = Vec3(     67.11022091,     11.90392367,     23.93009922);
    refforces[ 5] = Vec3(     46.43708221,    -2.656656187,    -16.65293107);
    refforces[ 6] = Vec3(     1.148617423,    -1.286157531,    -2.197592114);
    refforces[ 7] = Vec3(     3.271428661,    -2.741098117,    -6.424340583);
    refforces[ 8] = Vec3(     11.98035816,    -47.36903306,     204.7356612);
    refforces[ 9] = Vec3(     13.21458611,    -12.64799137,    -64.79184765);
    refforces[10] = Vec3(    -26.64490265,     2.987465959,    -46.17322487);
    refforces[11] = Vec3(     -2.40794957,      62.2317365,    -85.09954205);
    State state = context.getState(State::Forces | State::Energy);
    double energy = state.getPotentialEnergy();
    const vector<Vec3>& forces = state.getForces();
//    print_energy_and_forces(energy, forces);
    ASSERT_EQUAL_TOL(refenergy, energy, 1E-4);
    for (int n = 0; n < numAtoms; ++n)
        ASSERT_EQUAL_VEC(refforces[n], forces[n], 1E-4);
    check_finite_differences(forces, context, positions);
//    check_full_finite_differences(forces, context, positions, 1E-4, 1E-4);
}


void testWaterDimerEnergyAndForcesNoCutDirect() {
    // Water box with isotropic induced dipoles
    const double cutoff = 6.0*OpenMM::NmPerAngstrom;
    double boxEdgeLength = 20*OpenMM::NmPerAngstrom;
    const int grid = 64;
    MPIDForce* forceField = new MPIDForce();

    vector<Vec3> positions;
    System system;

    const int numAtoms = 6;

    bool do_charge = true;
    bool do_dpole  = true;
    bool do_qpole  = true;
    bool do_opole  = true;
    bool do_pol    = true;
    make_waterbox(numAtoms, boxEdgeLength, forceField,  positions, system,
                  do_charge, do_dpole, do_qpole, do_opole, do_pol);
    forceField->setNonbondedMethod(OpenMM::MPIDForce::NoCutoff);
    forceField->setDefaultTholeWidth(3.0);
    forceField->setPolarizationType(MPIDForce::Direct);
    system.addForce(forceField);

    VerletIntegrator integrator(0.01);
    Context context(system, integrator, Platform::getPlatformByName("CUDA"));
    context.setPositions(positions);

    double refenergy = -1.949902453;
    vector<Vec3> refforces(6);
    refforces[0] = Vec3(    -138.7310812,    -182.9709838,     35.70961618);
    refforces[1] = Vec3(      37.1153441,    -5.548490702,      5.04277195);
    refforces[2] = Vec3(     41.13860764,     118.8270727,     31.47279046);
    refforces[3] = Vec3(    -116.4297925,     -100.864177,    -27.61517965);
    refforces[4] = Vec3(     126.6370205,     165.8966158,    -19.33373258);
    refforces[5] = Vec3(     50.26990146,     4.659962993,    -25.27626636);

    State state = context.getState(State::Forces | State::Energy);
    double energy = state.getPotentialEnergy();
    const vector<Vec3>& forces = state.getForces();
//    print_energy_and_forces(energy, forces);
    ASSERT_EQUAL_TOL(refenergy, energy, 1E-4);
    for (int n = 0; n < numAtoms; ++n)
        ASSERT_EQUAL_VEC(refforces[n], forces[n], 1E-4);
    check_finite_differences(forces, context, positions);
//    check_full_finite_differences(forces, context, positions, 1E-4, 1E-4);
}


void testWaterDimerEnergyAndForcesPMEDirect() {
    // Water box with isotropic induced dipoles
    const double cutoff = 6.0*OpenMM::NmPerAngstrom;
    double boxEdgeLength = 20*OpenMM::NmPerAngstrom;
    const double alpha = 3.0001;
    const int grid = 64;
    MPIDForce* forceField = new MPIDForce();

    vector<Vec3> positions;
    System system;

    const int numAtoms = 6;

    bool do_charge = true;
    bool do_dpole  = true;
    bool do_qpole  = true;
    bool do_opole  = true;
    bool do_pol    = true;
    make_waterbox(numAtoms, boxEdgeLength, forceField,  positions, system,
                  do_charge, do_dpole, do_qpole, do_opole, do_pol);
    forceField->setNonbondedMethod(OpenMM::MPIDForce::PME);
    forceField->setPMEParameters(alpha, grid, grid, grid);
    forceField->setDefaultTholeWidth(3.0);
    forceField->setCutoffDistance(cutoff);
    forceField->setPolarizationType(MPIDForce::Direct);
    system.addForce(forceField);

    VerletIntegrator integrator(0.01);
    Context context(system, integrator, Platform::getPlatformByName("CUDA"));

    context.setPositions(positions);

    double refenergy = -2.523318862;
    vector<Vec3> refforces(6);
    refforces[0] = Vec3(    -138.9578383,    -183.3187212,     31.05996292);
    refforces[1] = Vec3(     36.78883138,    -5.591080652,     7.601999899);
    refforces[2] = Vec3(     41.46403045,     118.9693325,     34.16137849);
    refforces[3] = Vec3(    -116.5222458,    -100.9480058,    -32.82501978);
    refforces[4] = Vec3(     126.6226866,     166.1966239,    -17.03839082);
    refforces[5] = Vec3(      50.6045891,     4.691845173,    -22.96008637);

    State state = context.getState(State::Forces | State::Energy);
    double energy = state.getPotentialEnergy();
    const vector<Vec3>& forces = state.getForces();
    //print_energy_and_forces(energy, forces);
    ASSERT_EQUAL_TOL(refenergy, energy, 1E-4);
    for (int n = 0; n < numAtoms; ++n)
        ASSERT_EQUAL_VEC(refforces[n], forces[n], 1E-4);
    check_finite_differences(forces, context, positions);
    //check_full_finite_differences(forces, context, positions, 1E-4, 1E-4);
}

void testWaterDimerEnergyAndForcesNoCutMutual() {
    // Water box with isotropic induced dipoles
    const double cutoff = 6.0*OpenMM::NmPerAngstrom;
    double boxEdgeLength = 20*OpenMM::NmPerAngstrom;
    MPIDForce* forceField = new MPIDForce();

    vector<Vec3> positions;

    System system;

    const int numAtoms = 6;

    bool do_charge = true;
    bool do_dpole  = true;
    bool do_qpole  = true;
    bool do_opole  = true;
    bool do_pol    = true;
    make_waterbox(numAtoms, boxEdgeLength, forceField,  positions, system,
                  do_charge, do_dpole, do_qpole, do_opole, do_pol);
    forceField->setNonbondedMethod(OpenMM::MPIDForce::NoCutoff);
    forceField->setDefaultTholeWidth(3.0);
    forceField->setPolarizationType(MPIDForce::Mutual);
    forceField->setMutualInducedTargetEpsilon(1e-8);
    system.addForce(forceField);

    VerletIntegrator integrator(0.01);
    Context context(system, integrator, Platform::getPlatformByName("CUDA"));
    context.setPositions(positions);

    double refenergy = -1.952917117;
    vector<Vec3> refforces(6);
    refforces[0] = Vec3(    -139.7835608,    -184.4337529,     35.62953533);
    refforces[1] = Vec3(       37.434981,    -5.522902943,      5.11681405);
    refforces[2] = Vec3(     41.23101208,     119.3674074,     31.61700973);
    refforces[3] = Vec3(    -116.9476192,    -101.4714619,    -27.86430037);
    refforces[4] = Vec3(     127.7709383,     167.4188741,    -19.23010497);
    refforces[5] = Vec3(     50.29424862,     4.641836191,    -25.26895377);
    State state = context.getState(State::Forces | State::Energy);
    double energy = state.getPotentialEnergy();
    const vector<Vec3>& forces = state.getForces();
//    print_energy_and_forces(energy, forces);
    ASSERT_EQUAL_TOL(refenergy, energy, 1E-4);
    for (int n = 0; n < numAtoms; ++n)
        ASSERT_EQUAL_VEC(refforces[n], forces[n], 1E-4);
    check_finite_differences(forces, context, positions);
//    check_full_finite_differences(forces, context, positions, 1E-4, 1E-4);
}

void testWaterDimerEnergyAndForcesPMEMutual() {
    // Water box with isotropic induced dipoles
    const double cutoff = 6.0*OpenMM::NmPerAngstrom;
    double boxEdgeLength = 20*OpenMM::NmPerAngstrom;
    const double alpha = 3;
    const int grid = 64;
    MPIDForce* forceField = new MPIDForce();

    vector<Vec3> positions;

    System system;

    const int numAtoms = 6;

    bool do_charge = true;
    bool do_dpole  = true;
    bool do_qpole  = true;
    bool do_opole  = true;
    bool do_pol    = true;
    make_waterbox(numAtoms, boxEdgeLength, forceField,  positions, system,
                  do_charge, do_dpole, do_qpole, do_opole, do_pol);
    forceField->setNonbondedMethod(OpenMM::MPIDForce::PME);
    forceField->setPMEParameters(alpha, grid, grid, grid);
    forceField->setDefaultTholeWidth(3.0);
    forceField->setCutoffDistance(cutoff);
    forceField->setPolarizationType(MPIDForce::Mutual);
    forceField->setMutualInducedTargetEpsilon(1e-8);
    system.addForce(forceField);

    VerletIntegrator integrator(0.01);
    Context context(system, integrator, Platform::getPlatformByName("CUDA"));
    context.setPositions(positions);

    double refenergy = -2.533082539;
    vector<Vec3> refforces(6);
    refforces[0] = Vec3(    -140.0801113,    -184.8502938,     30.90206227);
    refforces[1] = Vec3(     37.10990648,    -5.575145037,     7.692659824);
    refforces[2] = Vec3(     41.55181662,       119.50422,     34.31895915);
    refforces[3] = Vec3(    -117.0338412,    -101.5516429,    -33.10846732);
    refforces[4] = Vec3(     127.7947382,     167.7834207,    -16.87461724);
    refforces[5] = Vec3(     50.65754476,     4.689434353,    -22.93075376);

    State state = context.getState(State::Forces | State::Energy);
    double energy = state.getPotentialEnergy();
    const vector<Vec3>& forces = state.getForces();
    //print_energy_and_forces(energy, forces);
    ASSERT_EQUAL_TOL(refenergy, energy, 1E-4);
    for (int n = 0; n < numAtoms; ++n)
        ASSERT_EQUAL_VEC(refforces[n], forces[n], 1E-4);
    check_finite_differences(forces, context, positions);
    //check_full_finite_differences(forces, context, positions, 5E-5, 1E-4);
}


void testWaterDimerEnergyAndForcesPMEExtrapolated() {
    // Water box with isotropic induced dipoles
    const double cutoff = 6.0*OpenMM::NmPerAngstrom;
    double boxEdgeLength = 20*OpenMM::NmPerAngstrom;
    const double alpha = 3.0;
    const int grid = 64;
    MPIDForce* forceField = new MPIDForce();

    vector<Vec3> positions;

    System system;

    const int numAtoms = 6;

    bool do_charge = true;
    bool do_dpole  = true;
    bool do_qpole  = true;
    bool do_opole  = true;
    bool do_pol    = true;
    make_waterbox(numAtoms, boxEdgeLength, forceField,  positions, system,
                  do_charge, do_dpole, do_qpole, do_opole, do_pol);
    forceField->setNonbondedMethod(OpenMM::MPIDForce::PME);
    forceField->setPMEParameters(alpha, grid, grid, grid);
    forceField->setDefaultTholeWidth(3.0);
    forceField->setCutoffDistance(cutoff);
    forceField->setPolarizationType(MPIDForce::Extrapolated);
    system.addForce(forceField);

    VerletIntegrator integrator(0.01);
    Context context(system, integrator, Platform::getPlatformByName("CUDA"));
    context.setPositions(positions);

    double refenergy = -2.527846018;
    vector<Vec3> refforces(6);
    refforces[0] = Vec3(     -140.156749,    -184.9802098,     30.95576142);
    refforces[1] = Vec3(     37.14821556,    -5.561204369,     7.691990976);
    refforces[2] = Vec3(     41.56730743,     119.5695275,     34.32030003);
    refforces[3] = Vec3(    -117.1007878,    -101.6308286,    -33.11577527);
    refforces[4] = Vec3(     127.9092349,      167.929753,    -16.90691988);
    refforces[5] = Vec3(     50.63283256,     4.672955505,    -22.94551458);

    State state = context.getState(State::Forces | State::Energy);
    double energy = state.getPotentialEnergy();
    const vector<Vec3>& forces = state.getForces();
//    print_energy_and_forces(energy, forces);
    ASSERT_EQUAL_TOL(refenergy, energy, 1E-4);
    for (int n = 0; n < numAtoms; ++n)
        ASSERT_EQUAL_VEC(refforces[n], forces[n], 1E-4);
    check_finite_differences(forces, context, positions);
//    check_full_finite_differences(forces, context, positions, 1E-4, 1E-4);
}


void testWaterDimerEnergyAndForcesNoCutExtrapolated() {
    // Water box with isotropic induced dipoles
    double boxEdgeLength = 20*OpenMM::NmPerAngstrom;
    MPIDForce* forceField = new MPIDForce();

    vector<Vec3> positions;

    System system;

    const int numAtoms = 6;

    bool do_charge = true;
    bool do_dpole  = true;
    bool do_qpole  = true;
    bool do_opole  = true;
    bool do_pol    = true;
    make_waterbox(numAtoms, boxEdgeLength, forceField,  positions, system,
                  do_charge, do_dpole, do_qpole, do_opole, do_pol);
    forceField->setNonbondedMethod(OpenMM::MPIDForce::NoCutoff);
    forceField->setDefaultTholeWidth(3.0);
    forceField->setPolarizationType(MPIDForce::Extrapolated);
    system.addForce(forceField);

    VerletIntegrator integrator(0.01);
    Context context(system, integrator, Platform::getPlatformByName("CUDA"));
    context.setPositions(positions);

    double refenergy = -1.94668563;
    vector<Vec3> refforces(6);
    refforces[0] = Vec3(    -139.8529084,    -184.5568497,     35.69566243);
    refforces[1] = Vec3(     37.47391144,    -5.507167568,     5.113662284);
    refforces[2] = Vec3(     41.24807762,     119.4349691,     31.61668322);
    refforces[3] = Vec3(    -117.0172298,    -101.5532625,    -27.86681578);
    refforces[4] = Vec3(     127.8817331,     167.5584812,    -19.27180201);
    refforces[5] = Vec3(     50.26641605,     4.623829403,    -25.28739015);

    State state = context.getState(State::Forces | State::Energy);
    double energy = state.getPotentialEnergy();
    const vector<Vec3>& forces = state.getForces();
//    print_energy_and_forces(energy, forces);
    ASSERT_EQUAL_TOL(refenergy, energy, 1E-4);
    for (int n = 0; n < numAtoms; ++n)
        ASSERT_EQUAL_VEC(refforces[n], forces[n], 1E-4);
    check_finite_differences(forces, context, positions);
//    check_full_finite_differences(forces, context, positions, 1E-4, 1E-4);
}

void test14ScalingNoCutoff() {
    // Water box with isotropic induced dipoles
    double boxEdgeLength = 20*OpenMM::NmPerAngstrom;
    MPIDForce* forceField1 = new MPIDForce();
    MPIDForce* forceField2 = new MPIDForce();
    MPIDForce* forceField3 = new MPIDForce();
    vector<Vec3> positions;

    double energy;
    System system1, system2, system3;
    State state;

    make_charge_square(boxEdgeLength, positions, forceField1, system1);
    forceField1->setNonbondedMethod(OpenMM::MPIDForce::NoCutoff);
    system1.addForce(forceField1);
    VerletIntegrator integrator1(0.01);
    Context context1(system1, integrator1, Platform::getPlatformByName("CUDA"));
    context1.setPositions(positions);
    state = context1.getState(State::Forces | State::Energy);
    energy = state.getPotentialEnergy();
    ASSERT_EQUAL_TOL(energy, -1389.35, 1E-5);

    make_charge_square(boxEdgeLength, positions, forceField2, system2);
    forceField2->setNonbondedMethod(OpenMM::MPIDForce::NoCutoff);
    forceField2->set14ScaleFactor(0.5);
    system2.addForce(forceField2);
    VerletIntegrator integrator2(0.01);
    Context context2(system2, integrator2, Platform::getPlatformByName("CUDA"));
    context2.setPositions(positions);
    state = context2.getState(State::Forces | State::Energy);
    energy = state.getPotentialEnergy();
    ASSERT_EQUAL_TOL(energy, -1389.35/2, 1E-5);

    make_charge_square(boxEdgeLength, positions, forceField3, system3);
    forceField3->setNonbondedMethod(OpenMM::MPIDForce::NoCutoff);
    forceField3->set14ScaleFactor(0.0);
    system3.addForce(forceField3);
    VerletIntegrator integrator3(0.01);
    Context context3(system3, integrator3, Platform::getPlatformByName("CUDA"));
    context3.setPositions(positions);
    state = context3.getState(State::Forces | State::Energy);
    energy = state.getPotentialEnergy();
    ASSERT_EQUAL_TOL(energy, 0.0, 1E-5);
}

void test14ScalingPME() {
    // Water box with isotropic induced dipoles
    const double cutoff = 4.0*OpenMM::NmPerAngstrom;
    double boxEdgeLength = 30*OpenMM::NmPerAngstrom;
    const int grid = 64;
    MPIDForce* forceField1 = new MPIDForce();
    MPIDForce* forceField2 = new MPIDForce();
    MPIDForce* forceField3 = new MPIDForce();
    vector<Vec3> positions;

    double energy;
    System system1, system2, system3;
    State state;

    make_charge_square(boxEdgeLength, positions, forceField1, system1);
    forceField1->setNonbondedMethod(OpenMM::MPIDForce::PME);
    forceField1->setCutoffDistance(cutoff);
    forceField1->setPMEParameters(0.001, grid, grid, grid);
    system1.addForce(forceField1);
    VerletIntegrator integrator1(0.01);
    Context context1(system1, integrator1, Platform::getPlatformByName("CUDA"));
    context1.setPositions(positions);
    state = context1.getState(State::Forces | State::Energy);
    energy = state.getPotentialEnergy();
    ASSERT_EQUAL_TOL(energy, -1389.35, 1E-3);

    make_charge_square(boxEdgeLength, positions, forceField2, system2);
    forceField2->setNonbondedMethod(OpenMM::MPIDForce::PME);
    forceField2->setCutoffDistance(cutoff);
    forceField2->setPMEParameters(0.001, grid, grid, grid);
    forceField2->set14ScaleFactor(0.5);
    system2.addForce(forceField2);
    VerletIntegrator integrator2(0.01);
    Context context2(system2, integrator2, Platform::getPlatformByName("CUDA"));
    context2.setPositions(positions);
    state = context2.getState(State::Forces | State::Energy);
    energy = state.getPotentialEnergy();
    ASSERT_EQUAL_TOL(energy, -1389.35/2, 1E-3);

    make_charge_square(boxEdgeLength, positions, forceField3, system3);
    forceField3->setNonbondedMethod(OpenMM::MPIDForce::PME);
    forceField3->setCutoffDistance(cutoff);
    forceField3->setPMEParameters(0.001, grid, grid, grid);
    forceField3->set14ScaleFactor(0.0);
    system3.addForce(forceField3);
    VerletIntegrator integrator3(0.01);
    Context context3(system3, integrator3, Platform::getPlatformByName("CUDA"));
    context3.setPositions(positions);
    state = context3.getState(State::Forces | State::Energy);
    energy = state.getPotentialEnergy();
    ASSERT_EQUAL_TOL(energy, 0.0, 1E-3);
}



int main(int argc, char* argv[]) {
    try {
        std::cout << "TestCudaMPIDForce running test..." << std::endl;
        registerMPIDCudaKernelFactories();
        if (argc > 1)
            Platform::getPlatformByName("CUDA").setPropertyDefaultValue("Precision", std::string(argv[1]));

        test14ScalingNoCutoff();
        test14ScalingPME();
        testWaterDimerEnergyAndForcesPMEDirect();
        testWaterDimerEnergyAndForcesNoCutDirect();
        testWaterDimerEnergyAndForcesPMEMutual();
        testWaterDimerEnergyAndForcesNoCutMutual();
        testWaterDimerEnergyAndForcesPMEExtrapolated();
        testWaterDimerEnergyAndForcesNoCutExtrapolated();
        testMethanolDimerEnergyAndForcesPMEExtrapolated();
        testMethanolDimerEnergyAndForcesNoCutExtrapolated();
        testMethanolDimerEnergyAndForcesPMEDirect();
        testMethanolDimerEnergyAndForcesNoCutDirect();
        testMethanolDimerEnergyAndForcesPMEMutual();
        testMethanolDimerEnergyAndForcesNoCutMutual();
    } catch(const std::exception& e) {
        std::cout << "exception: " << e.what() << std::endl;
        std::cout << "FAIL - ERROR.  Test failed." << std::endl;
        return 1;
    }
    std::cout << "Done" << std::endl;
    return 0;
}
