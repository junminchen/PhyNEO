/* -------------------------------------------------------------------------- *
 *                                OpenMMMPID                                *
 * -------------------------------------------------------------------------- *
 * This is part of the OpenMM molecular simulation toolkit originating from   *
 * Simbios, the NIH National Center for Physics-Based Simulation of           *
 * Biological Structures at Stanford, funded under the NIH Roadmap for        *
 * Medical Research, grant U54 GM072970. See https://simtk.org.               *
 *                                                                            *
 * Portions copyright (c) 2010-2016 Stanford University and the Authors.      *
 * Authors: Peter Eastman                                                     *
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

#include "openmm/serialization/MPIDForceProxy.h"
#include "openmm/serialization/SerializationNode.h"
#include "openmm/Force.h"
#include "openmm/MPIDForce.h"
#include <sstream>

using namespace OpenMM;
using namespace std;

MPIDForceProxy::MPIDForceProxy() : SerializationProxy("MPIDForce") {
}

static void getCovalentTypes(std::vector<std::string>& covalentTypes) {

    covalentTypes.push_back("Covalent12");
    covalentTypes.push_back("Covalent13");
    covalentTypes.push_back("Covalent14");
    covalentTypes.push_back("Covalent15");

    covalentTypes.push_back("PolarizationCovalent11");
    covalentTypes.push_back("PolarizationCovalent12");
    covalentTypes.push_back("PolarizationCovalent13");
    covalentTypes.push_back("PolarizationCovalent14");
}

static void addCovalentMap(SerializationNode& particleExclusions, int particleIndex, std::string mapName, std::vector< int > covalentMap) {
    SerializationNode& map   = particleExclusions.createChildNode(mapName);
    for (unsigned int ii = 0; ii < covalentMap.size(); ii++) {
        map.createChildNode("Cv").setIntProperty("v", covalentMap[ii]);
    }
}

void loadCovalentMap(const SerializationNode& map, std::vector< int >& covalentMap) {
    for (unsigned int ii = 0; ii < map.getChildren().size(); ii++) {
        covalentMap.push_back(map.getChildren()[ii].getIntProperty("v"));
    }
}

void MPIDForceProxy::serialize(const void* object, SerializationNode& node) const {
    node.setIntProperty("version", 0);
    const MPIDForce& force = *reinterpret_cast<const MPIDForce*>(object);

    node.setIntProperty("forceGroup", force.getForceGroup());
    node.setIntProperty("nonbondedMethod",                  force.getNonbondedMethod());
    node.setIntProperty("polarizationType",                 force.getPolarizationType());
    node.setIntProperty("mutualInducedMaxIterations",       force.getMutualInducedMaxIterations());

    node.setDoubleProperty("cutoffDistance",                force.getCutoffDistance());
    double alpha;
    int nx, ny, nz;
    force.getPMEParameters(alpha, nx, ny, nz);
    node.setDoubleProperty("aEwald",                        alpha);
    node.setDoubleProperty("mutualInducedTargetEpsilon",    force.getMutualInducedTargetEpsilon());
    node.setDoubleProperty("ewaldErrorTolerance",           force.getEwaldErrorTolerance());
    node.setDoubleProperty("scaleFactor14",                 force.get14ScaleFactor());

    SerializationNode& gridDimensionsNode  = node.createChildNode("MultipoleParticleGridDimension");
    gridDimensionsNode.setIntProperty("d0", nx).setIntProperty("d1", ny).setIntProperty("d2", nz); 
    
    SerializationNode& coefficients = node.createChildNode("ExtrapolationCoefficients");
    vector<double> coeff = force.getExtrapolationCoefficients();
    for (int i = 0; i < coeff.size(); i++) {
        stringstream key;
        key << "c" << i;
        coefficients.setDoubleProperty(key.str(), coeff[i]);
    }

    std::vector<std::string> covalentTypes;
    getCovalentTypes(covalentTypes);

    SerializationNode& particles = node.createChildNode("MultipoleParticles");
    for (unsigned int ii = 0; ii < static_cast<unsigned int>(force.getNumMultipoles()); ii++) {

        int axisType, multipoleAtomZ, multipoleAtomX, multipoleAtomY;
        double charge, thole, dampingFactor;

        std::vector<double> molecularDipole;
        std::vector<double> molecularQuadrupole;
        std::vector<double> molecularOctopole;
        std::vector<double> alphas;
        force.getMultipoleParameters(ii, charge, molecularDipole, molecularQuadrupole, molecularOctopole,
                                     axisType, multipoleAtomZ, multipoleAtomX, multipoleAtomY, thole, alphas);

        SerializationNode& particle    = particles.createChildNode("Particle");
        particle.setIntProperty("axisType", axisType).setIntProperty("multipoleAtomZ", multipoleAtomZ).setIntProperty("multipoleAtomX", multipoleAtomX).setIntProperty("multipoleAtomY", multipoleAtomY);
        particle.setDoubleProperty("charge", charge).setDoubleProperty("thole", thole).setDoubleProperty("damp", dampingFactor).setDoubleProperty("polarizabilityXX", alphas[0]).setDoubleProperty("polarizabilityYY", alphas[1]).setDoubleProperty("polarizabilityZZ", alphas[2]);

        SerializationNode& dipole      = particle.createChildNode("Dipole");
        dipole.setDoubleProperty("dX", molecularDipole[0]);
        dipole.setDoubleProperty("dY", molecularDipole[1]);
        dipole.setDoubleProperty("dZ", molecularDipole[2]);

        SerializationNode& quadrupole  = particle.createChildNode("Quadrupole");
        quadrupole.setDoubleProperty("qXX", molecularQuadrupole[0]);
        quadrupole.setDoubleProperty("qXY", molecularQuadrupole[1]);
        quadrupole.setDoubleProperty("qYY", molecularQuadrupole[2]);
        quadrupole.setDoubleProperty("qXZ", molecularQuadrupole[3]);
        quadrupole.setDoubleProperty("qYZ", molecularQuadrupole[4]);
        quadrupole.setDoubleProperty("qZZ", molecularQuadrupole[5]);

        SerializationNode& octopole  = particle.createChildNode("Octopole");
        octopole.setDoubleProperty("oXXX", molecularOctopole[0]);
        octopole.setDoubleProperty("oXXY", molecularOctopole[1]);
        octopole.setDoubleProperty("oXYY", molecularOctopole[2]);
        octopole.setDoubleProperty("oYYY", molecularOctopole[3]);
        octopole.setDoubleProperty("oXXZ", molecularOctopole[4]);
        octopole.setDoubleProperty("oXYZ", molecularOctopole[5]);
        octopole.setDoubleProperty("oYYZ", molecularOctopole[6]);
        octopole.setDoubleProperty("oXZZ", molecularOctopole[7]);
        octopole.setDoubleProperty("oYZZ", molecularOctopole[8]);
        octopole.setDoubleProperty("oZZZ", molecularOctopole[9]);

        for (unsigned int jj = 0; jj < covalentTypes.size(); jj++) {
            std::vector< int > covalentMap;
            force.getCovalentMap(ii, static_cast<MPIDForce::CovalentType>(jj), covalentMap);
            addCovalentMap(particle, ii, covalentTypes[jj], covalentMap);
        }
    }
}

void* MPIDForceProxy::deserialize(const SerializationNode& node) const {
    int version = node.getIntProperty("version");
    if (version < 0 || version > 0)
        throw OpenMMException("Unsupported version number");
    MPIDForce* force = new MPIDForce();

    try {
        force->setForceGroup(node.getIntProperty("forceGroup", 0));
        force->setNonbondedMethod(static_cast<MPIDForce::NonbondedMethod>(node.getIntProperty("nonbondedMethod")));
        force->setPolarizationType(static_cast<MPIDForce::PolarizationType>(node.getIntProperty("polarizationType")));
        force->setMutualInducedMaxIterations(node.getIntProperty("mutualInducedMaxIterations"));

        force->setCutoffDistance(node.getDoubleProperty("cutoffDistance"));
        force->setMutualInducedTargetEpsilon(node.getDoubleProperty("mutualInducedTargetEpsilon"));
        force->setEwaldErrorTolerance(node.getDoubleProperty("ewaldErrorTolerance"));
        force->set14ScaleFactor(node.getDoubleProperty("scaleFactor14"));

        const SerializationNode& gridDimensionsNode  = node.getChildNode("MultipoleParticleGridDimension");
        force->setPMEParameters(node.getDoubleProperty("aEwald"), gridDimensionsNode.getIntProperty("d0"), gridDimensionsNode.getIntProperty("d1"), gridDimensionsNode.getIntProperty("d2"));
    
        const SerializationNode& coefficients = node.getChildNode("ExtrapolationCoefficients");
        vector<double> coeff;
        for (int i = 0; ; i++) {
            stringstream key;
            key << "c" << i;
            if (coefficients.getProperties().find(key.str()) == coefficients.getProperties().end())
                break;
            coeff.push_back(coefficients.getDoubleProperty(key.str()));
        }
        force->setExtrapolationCoefficients(coeff);
        std::vector<std::string> covalentTypes;
        getCovalentTypes(covalentTypes);

        const SerializationNode& particles = node.getChildNode("MultipoleParticles");
        for (unsigned int ii = 0; ii < particles.getChildren().size(); ii++) {

            const SerializationNode& particle = particles.getChildren()[ii];

            std::vector<double> molecularDipole;
            const SerializationNode& dipole = particle.getChildNode("Dipole");
            molecularDipole.push_back(dipole.getDoubleProperty("dX"));
            molecularDipole.push_back(dipole.getDoubleProperty("dY"));
            molecularDipole.push_back(dipole.getDoubleProperty("dZ"));

            std::vector<double> molecularQuadrupole;
            const SerializationNode& quadrupole = particle.getChildNode("Quadrupole");
            molecularQuadrupole.push_back(quadrupole.getDoubleProperty("qXX"));
            molecularQuadrupole.push_back(quadrupole.getDoubleProperty("qXY"));
            molecularQuadrupole.push_back(quadrupole.getDoubleProperty("qYY"));
            molecularQuadrupole.push_back(quadrupole.getDoubleProperty("qXZ"));
            molecularQuadrupole.push_back(quadrupole.getDoubleProperty("qYZ"));
            molecularQuadrupole.push_back(quadrupole.getDoubleProperty("qZZ"));

            std::vector<double> molecularOctopole;
            const SerializationNode& octopole = particle.getChildNode("Octopole");
            molecularOctopole.push_back(octopole.getDoubleProperty("oXXX"));
            molecularOctopole.push_back(octopole.getDoubleProperty("oXXY"));
            molecularOctopole.push_back(octopole.getDoubleProperty("oXYY"));
            molecularOctopole.push_back(octopole.getDoubleProperty("oYYY"));
            molecularOctopole.push_back(octopole.getDoubleProperty("oXXZ"));
            molecularOctopole.push_back(octopole.getDoubleProperty("oXYZ"));
            molecularOctopole.push_back(octopole.getDoubleProperty("oYYZ"));
            molecularOctopole.push_back(octopole.getDoubleProperty("oXZZ"));
            molecularOctopole.push_back(octopole.getDoubleProperty("oYZZ"));
            molecularOctopole.push_back(octopole.getDoubleProperty("oZZZ"));
            std::vector<double> polarizability;
            polarizability.push_back(particle.getDoubleProperty("polarizabilityXX"));
            polarizability.push_back(particle.getDoubleProperty("polarizabilityYY"));
            polarizability.push_back(particle.getDoubleProperty("polarizabilityZZ"));
            force->addMultipole(particle.getDoubleProperty("charge"), molecularDipole, molecularQuadrupole, molecularOctopole,
                                particle.getIntProperty("axisType"),
                                particle.getIntProperty("multipoleAtomZ"),
                                particle.getIntProperty("multipoleAtomX"),
                                particle.getIntProperty("multipoleAtomY"),
                                particle.getDoubleProperty("thole"),
                                polarizability);

            // covalent maps 

            for (unsigned int jj = 0; jj < covalentTypes.size(); jj++) {
                std::vector< int > covalentMap;
                loadCovalentMap(particle.getChildNode(covalentTypes[jj]), covalentMap);
                force->setCovalentMap(ii, static_cast<MPIDForce::CovalentType>(jj), covalentMap);
            }
        }

    }
    catch (...) {
        delete force;
        throw;
    }

    return force;
}
