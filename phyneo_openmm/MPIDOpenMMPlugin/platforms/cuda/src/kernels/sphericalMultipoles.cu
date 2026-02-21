__device__ void buildQIRotationMatrix(real3 deltaR, real rInv, real (&rotationMatrix)[3][3]) {
    real3 vectorZ = deltaR*rInv;
    real3 vectorX = vectorZ;
    if (deltaR.y != 0 || deltaR.z != 0)
        vectorX.x += 1;
    else
        vectorX.y += 1;

    vectorX -= vectorZ*dot(vectorX, vectorZ);
    vectorX = normalize(vectorX);
    real3 vectorY = cross(vectorZ, vectorX);

    // Reorder the Cartesian {x,y,z} dipole rotation matrix, to account
    // for spherical harmonic ordering {z,x,y}.
    rotationMatrix[0][0] = vectorZ.z;
    rotationMatrix[0][1] = vectorZ.x;
    rotationMatrix[0][2] = vectorZ.y;
    rotationMatrix[1][0] = vectorX.z;
    rotationMatrix[1][1] = vectorX.x;
    rotationMatrix[1][2] = vectorX.y;
    rotationMatrix[2][0] = vectorY.z;
    rotationMatrix[2][1] = vectorY.x;
    rotationMatrix[2][2] = vectorY.y;
}

__device__ real3 rotateDipole(real3& dipole, const real (&rotationMatrix)[3][3]) {
    return make_real3(rotationMatrix[0][0]*dipole.x + rotationMatrix[0][1]*dipole.y + rotationMatrix[0][2]*dipole.z,
                      rotationMatrix[1][0]*dipole.x + rotationMatrix[1][1]*dipole.y + rotationMatrix[1][2]*dipole.z,
                      rotationMatrix[2][0]*dipole.x + rotationMatrix[2][1]*dipole.y + rotationMatrix[2][2]*dipole.z);
}


__device__ void rotateQuadupoles(const real (&rotationMatrix)[3][3], const real* quad1, const real* quad2, real* rotated1, real* rotated2) {
    real sqrtThree = SQRT((real) 3);
    real element;
    element = 0.5f*(3.0f*rotationMatrix[0][0]*rotationMatrix[0][0] - 1.0f);
    rotated1[0] += quad1[0]*element;
    rotated2[0] += quad2[0]*element;
    element = sqrtThree*rotationMatrix[0][0]*rotationMatrix[0][1];
    rotated1[0] += quad1[1]*element;
    rotated2[0] += quad2[1]*element;
    element = sqrtThree*rotationMatrix[0][0]*rotationMatrix[0][2];
    rotated1[0] += quad1[2]*element;
    rotated2[0] += quad2[2]*element;
    element = 0.5f*sqrtThree*(rotationMatrix[0][1]*rotationMatrix[0][1] - rotationMatrix[0][2]*rotationMatrix[0][2]);
    rotated1[0] += quad1[3]*element;
    rotated2[0] += quad2[3]*element;
    element = sqrtThree*rotationMatrix[0][1]*rotationMatrix[0][2];
    rotated1[0] += quad1[4]*element;
    rotated2[0] += quad2[4]*element;
    element = sqrtThree*rotationMatrix[0][0]*rotationMatrix[1][0];
    rotated1[1] += quad1[0]*element;
    rotated2[1] += quad2[0]*element;
    element = rotationMatrix[1][0]*rotationMatrix[0][1] + rotationMatrix[0][0]*rotationMatrix[1][1];
    rotated1[1] += quad1[1]*element;
    rotated2[1] += quad2[1]*element;
    element = rotationMatrix[1][0]*rotationMatrix[0][2] + rotationMatrix[0][0]*rotationMatrix[1][2];
    rotated1[1] += quad1[2]*element;
    rotated2[1] += quad2[2]*element;
    element = rotationMatrix[0][1]*rotationMatrix[1][1] - rotationMatrix[0][2]*rotationMatrix[1][2];
    rotated1[1] += quad1[3]*element;
    rotated2[1] += quad2[3]*element;
    element = rotationMatrix[1][1]*rotationMatrix[0][2] + rotationMatrix[0][1]*rotationMatrix[1][2];
    rotated1[1] += quad1[4]*element;
    rotated2[1] += quad2[4]*element;
    element = sqrtThree*rotationMatrix[0][0]*rotationMatrix[2][0];
    rotated1[2] += quad1[0]*element;
    rotated2[2] += quad2[0]*element;
    element = rotationMatrix[2][0]*rotationMatrix[0][1] + rotationMatrix[0][0]*rotationMatrix[2][1];
    rotated1[2] += quad1[1]*element;
    rotated2[2] += quad2[1]*element;
    element = rotationMatrix[2][0]*rotationMatrix[0][2] + rotationMatrix[0][0]*rotationMatrix[2][2];
    rotated1[2] += quad1[2]*element;
    rotated2[2] += quad2[2]*element;
    element = rotationMatrix[0][1]*rotationMatrix[2][1] - rotationMatrix[0][2]*rotationMatrix[2][2];
    rotated1[2] += quad1[3]*element;
    rotated2[2] += quad2[3]*element;
    element = rotationMatrix[2][1]*rotationMatrix[0][2] + rotationMatrix[0][1]*rotationMatrix[2][2];
    rotated1[2] += quad1[4]*element;
    rotated2[2] += quad2[4]*element;
    element = 0.5f*sqrtThree*(rotationMatrix[1][0]*rotationMatrix[1][0] - rotationMatrix[2][0]*rotationMatrix[2][0]);
    rotated1[3] += quad1[0]*element;
    rotated2[3] += quad2[0]*element;
    element = rotationMatrix[1][0]*rotationMatrix[1][1] - rotationMatrix[2][0]*rotationMatrix[2][1];
    rotated1[3] += quad1[1]*element;
    rotated2[3] += quad2[1]*element;
    element = rotationMatrix[1][0]*rotationMatrix[1][2] - rotationMatrix[2][0]*rotationMatrix[2][2];
    rotated1[3] += quad1[2]*element;
    rotated2[3] += quad2[2]*element;
    element = 0.5f*(rotationMatrix[1][1]*rotationMatrix[1][1] - rotationMatrix[2][1]*rotationMatrix[2][1] - rotationMatrix[1][2]*rotationMatrix[1][2] + rotationMatrix[2][2]*rotationMatrix[2][2]);
    rotated1[3] += quad1[3]*element;
    rotated2[3] += quad2[3]*element;
    element = rotationMatrix[1][1]*rotationMatrix[1][2] - rotationMatrix[2][1]*rotationMatrix[2][2];
    rotated1[3] += quad1[4]*element;
    rotated2[3] += quad2[4]*element;
    element = sqrtThree*rotationMatrix[1][0]*rotationMatrix[2][0];
    rotated1[4] += quad1[0]*element;
    rotated2[4] += quad2[0]*element;
    element = rotationMatrix[2][0]*rotationMatrix[1][1] + rotationMatrix[1][0]*rotationMatrix[2][1];
    rotated1[4] += quad1[1]*element;
    rotated2[4] += quad2[1]*element;
    element = rotationMatrix[2][0]*rotationMatrix[1][2] + rotationMatrix[1][0]*rotationMatrix[2][2];
    rotated1[4] += quad1[2]*element;
    rotated2[4] += quad2[2]*element;
    element = rotationMatrix[1][1]*rotationMatrix[2][1] - rotationMatrix[1][2]*rotationMatrix[2][2];
    rotated1[4] += quad1[3]*element;
    rotated2[4] += quad2[3]*element;
    element = rotationMatrix[2][1]*rotationMatrix[1][2] + rotationMatrix[1][1]*rotationMatrix[2][2];
    rotated1[4] += quad1[4]*element;
    rotated2[4] += quad2[4]*element;
}

__device__ void rotateOctopoles(const real (&rotationMatrix)[3][3], const real* oct1, const real* oct2, real* rotated1, real* rotated2) {
    const real sqrtSix = SQRT((real) 6);
    const real sqrtTen = SQRT((real) 10);
    const real sqrtFifteen = SQRT((real) 15);
    const real sqrtThreeHalves = SQRT((real) 1.5f);
    const real sqrtFiveHalves = SQRT((real) 2.5f);
    const real sqrtTenth = SQRT((real) 0.1f);
    const real sqrtFourTenths = SQRT((real) 0.4f);
    const real sqrtSixTenths = SQRT((real) 0.6f);
    real element;
    element = (-3*rotationMatrix[0][1]*rotationMatrix[0][1]*rotationMatrix[0][0])/2 - (3*rotationMatrix[0][2]*rotationMatrix[0][2]*rotationMatrix[0][0])/2 + rotationMatrix[0][0]*rotationMatrix[0][0]*rotationMatrix[0][0];
    rotated1[0] += oct1[0]*element;
    rotated2[0] += oct2[0]*element;
    element = -(sqrtThreeHalves*rotationMatrix[0][1]*rotationMatrix[0][1]*rotationMatrix[0][1])/2 - (sqrtThreeHalves*rotationMatrix[0][1]*rotationMatrix[0][2]*rotationMatrix[0][2])/2  + sqrtSix*rotationMatrix[0][1]*rotationMatrix[0][0]*rotationMatrix[0][0];
    rotated1[0] += oct1[1]*element;
    rotated2[0] += oct2[1]*element;
    element = -(sqrtThreeHalves*rotationMatrix[0][1]*rotationMatrix[0][1]*rotationMatrix[0][2])/2 - (sqrtThreeHalves*rotationMatrix[0][2]*rotationMatrix[0][2]*rotationMatrix[0][2])/2 + sqrtSix*rotationMatrix[0][2]*rotationMatrix[0][0]*rotationMatrix[0][0];
    rotated1[0] += oct1[2]*element;
    rotated2[0] += oct2[2]*element;
    element = (sqrtFifteen*rotationMatrix[0][1]*rotationMatrix[0][1]*rotationMatrix[0][0])/2 - (sqrtFifteen*rotationMatrix[0][2]*rotationMatrix[0][2]*rotationMatrix[0][0])/2;
    rotated1[0] += oct1[3]*element;
    rotated2[0] += oct2[3]*element;
    element = sqrtFifteen*rotationMatrix[0][1]*rotationMatrix[0][2]*rotationMatrix[0][0];
    rotated1[0] += oct1[4]*element;
    rotated2[0] += oct2[4]*element;
    element = (sqrtFiveHalves*rotationMatrix[0][1]*rotationMatrix[0][1]*rotationMatrix[0][1])/2 - (3*sqrtFiveHalves*rotationMatrix[0][1]*rotationMatrix[0][2]*rotationMatrix[0][2])/2;
    rotated1[0] += oct1[5]*element;
    rotated2[0] += oct2[5]*element;
    element = (3*sqrtFiveHalves*rotationMatrix[0][1]*rotationMatrix[0][1]*rotationMatrix[0][2])/2 - (sqrtFiveHalves*rotationMatrix[0][2]*rotationMatrix[0][2]*rotationMatrix[0][2])/2;
    rotated1[0] += oct1[6]*element;
    rotated2[0] += oct2[6]*element;

    element = -(sqrtThreeHalves*rotationMatrix[0][1]*rotationMatrix[0][1]*rotationMatrix[1][0])/2 - (sqrtThreeHalves*rotationMatrix[0][2]*rotationMatrix[0][2]*rotationMatrix[1][0])/2 - sqrtThreeHalves*rotationMatrix[1][1]*rotationMatrix[0][1]*rotationMatrix[0][0] - sqrtThreeHalves*rotationMatrix[1][2]*rotationMatrix[0][2]*rotationMatrix[0][0] + sqrtThreeHalves*rotationMatrix[1][0]*rotationMatrix[0][0]*rotationMatrix[0][0];
    rotated1[1] += oct1[0]*element;
    rotated2[1] += oct2[0]*element;
    element = (-3*rotationMatrix[1][1]*rotationMatrix[0][1]*rotationMatrix[0][1])/4 - (rotationMatrix[0][1]*rotationMatrix[1][2]*rotationMatrix[0][2])/2 - (rotationMatrix[1][1]*rotationMatrix[0][2]*rotationMatrix[0][2])/4 + 2*rotationMatrix[0][1]*rotationMatrix[1][0]*rotationMatrix[0][0] + rotationMatrix[1][1]*rotationMatrix[0][0]*rotationMatrix[0][0];
    rotated1[1] += oct1[1]*element;
    rotated2[1] += oct2[1]*element;
    element = -(rotationMatrix[0][1]*rotationMatrix[0][1]*rotationMatrix[1][2])/4 - (rotationMatrix[1][1]*rotationMatrix[0][1]*rotationMatrix[0][2])/2 - (3*rotationMatrix[1][2]*rotationMatrix[0][2]*rotationMatrix[0][2])/4 + 2*rotationMatrix[0][2]*rotationMatrix[1][0]*rotationMatrix[0][0] + rotationMatrix[1][2]*rotationMatrix[0][0]*rotationMatrix[0][0];
    rotated1[1] += oct1[2]*element;
    rotated2[1] += oct2[2]*element;
    element = (sqrtFiveHalves*rotationMatrix[0][1]*rotationMatrix[0][1]*rotationMatrix[1][0])/2 - (sqrtFiveHalves*rotationMatrix[0][2]*rotationMatrix[0][2]*rotationMatrix[1][0])/2 + sqrtFiveHalves*rotationMatrix[1][1]*rotationMatrix[0][1]*rotationMatrix[0][0] - sqrtFiveHalves*rotationMatrix[1][2]*rotationMatrix[0][2]*rotationMatrix[0][0];
    rotated1[1] += oct1[3]*element;
    rotated2[1] += oct2[3]*element;
    element = sqrtFiveHalves*rotationMatrix[0][1]*rotationMatrix[0][2]*rotationMatrix[1][0] + sqrtFiveHalves*rotationMatrix[0][1]*rotationMatrix[1][2]*rotationMatrix[0][0] + sqrtFiveHalves*rotationMatrix[1][1]*rotationMatrix[0][2]*rotationMatrix[0][0];
    rotated1[1] += oct1[4]*element;
    rotated2[1] += oct2[4]*element;
    element = sqrtFifteen*rotationMatrix[1][1]*rotationMatrix[0][1]*rotationMatrix[0][1]/4 - (sqrtFifteen*rotationMatrix[0][1]*rotationMatrix[1][2]*rotationMatrix[0][2])/2 - (sqrtFifteen*rotationMatrix[1][1]*rotationMatrix[0][2]*rotationMatrix[0][2])/4;
    rotated1[1] += oct1[5]*element;
    rotated2[1] += oct2[5]*element;
    element = sqrtFifteen*rotationMatrix[0][1]*rotationMatrix[0][1]*rotationMatrix[1][2]/4 + (sqrtFifteen*rotationMatrix[1][1]*rotationMatrix[0][1]*rotationMatrix[0][2])/2 - (sqrtFifteen*rotationMatrix[1][2]*rotationMatrix[0][2]*rotationMatrix[0][2])/4;
    rotated1[1] += oct1[6]*element;
    rotated2[1] += oct2[6]*element;

    //rotatedOctopole[2] = 
    element = -(sqrtThreeHalves*rotationMatrix[0][1]*rotationMatrix[0][1]*rotationMatrix[2][0])/2 - (sqrtThreeHalves*rotationMatrix[0][2]*rotationMatrix[0][2]*rotationMatrix[2][0])/2 - sqrtThreeHalves*rotationMatrix[2][1]*rotationMatrix[0][1]*rotationMatrix[0][0] - sqrtThreeHalves*rotationMatrix[2][2]*rotationMatrix[0][2]*rotationMatrix[0][0] + sqrtThreeHalves*rotationMatrix[2][0]*rotationMatrix[0][0]*rotationMatrix[0][0];
    rotated1[2] += oct1[0]*element;
    rotated2[2] += oct2[0]*element;
    element = (-3*rotationMatrix[2][1]*rotationMatrix[0][1]*rotationMatrix[0][1])/4 - (rotationMatrix[0][1]*rotationMatrix[2][2]*rotationMatrix[0][2])/2 - (rotationMatrix[2][1]*rotationMatrix[0][2]*rotationMatrix[0][2])/4 + 2*rotationMatrix[0][1]*rotationMatrix[2][0]*rotationMatrix[0][0] + rotationMatrix[2][1]*rotationMatrix[0][0]*rotationMatrix[0][0];
    rotated1[2] += oct1[1]*element;
    rotated2[2] += oct2[1]*element;
    element = -(rotationMatrix[0][1]*rotationMatrix[0][1]*rotationMatrix[2][2])/4 - (rotationMatrix[2][1]*rotationMatrix[0][1]*rotationMatrix[0][2])/2 - (3*rotationMatrix[2][2]*rotationMatrix[0][2]*rotationMatrix[0][2])/4 + 2*rotationMatrix[0][2]*rotationMatrix[2][0]*rotationMatrix[0][0] + rotationMatrix[2][2]*rotationMatrix[0][0]*rotationMatrix[0][0];
    rotated1[2] += oct1[2]*element;
    rotated2[2] += oct2[2]*element;
    element = (sqrtFiveHalves*rotationMatrix[0][1]*rotationMatrix[0][1]*rotationMatrix[2][0])/2 - (sqrtFiveHalves*rotationMatrix[0][2]*rotationMatrix[0][2]*rotationMatrix[2][0])/2 + sqrtFiveHalves*rotationMatrix[2][1]*rotationMatrix[0][1]*rotationMatrix[0][0] - sqrtFiveHalves*rotationMatrix[2][2]*rotationMatrix[0][2]*rotationMatrix[0][0];
    rotated1[2] += oct1[3]*element;
    rotated2[2] += oct2[3]*element;
    element = sqrtFiveHalves*rotationMatrix[0][1]*rotationMatrix[0][2]*rotationMatrix[2][0] + sqrtFiveHalves*rotationMatrix[0][1]*rotationMatrix[2][2]*rotationMatrix[0][0] + sqrtFiveHalves*rotationMatrix[2][1]*rotationMatrix[0][2]*rotationMatrix[0][0];
    rotated1[2] += oct1[4]*element;
    rotated2[2] += oct2[4]*element;
    element = (sqrtFifteen*rotationMatrix[2][1]*rotationMatrix[0][1]*rotationMatrix[0][1])/4 - (sqrtFifteen*rotationMatrix[0][1]*rotationMatrix[2][2]*rotationMatrix[0][2])/2 - (sqrtFifteen*rotationMatrix[2][1]*rotationMatrix[0][2]*rotationMatrix[0][2])/4;
    rotated1[2] += oct1[5]*element;
    rotated2[2] += oct2[5]*element;
    element = (sqrtFifteen*rotationMatrix[0][1]*rotationMatrix[0][1]*rotationMatrix[2][2])/4 + (sqrtFifteen*rotationMatrix[2][1]*rotationMatrix[0][1]*rotationMatrix[0][2])/2 - (sqrtFifteen*rotationMatrix[2][2]*rotationMatrix[0][2]*rotationMatrix[0][2])/4;
    rotated1[2] += oct1[6]*element;
    rotated2[2] += oct2[6]*element;

    //rotatedOctopole[3] = 
    element = -(sqrtSixTenths*rotationMatrix[1][1]*rotationMatrix[0][1]*rotationMatrix[1][0]) - sqrtSixTenths*rotationMatrix[1][2]*rotationMatrix[0][2]*rotationMatrix[1][0] + sqrtSixTenths*rotationMatrix[2][1]*rotationMatrix[0][1]*rotationMatrix[2][0] + sqrtSixTenths*rotationMatrix[2][2]*rotationMatrix[0][2]*rotationMatrix[2][0] - (sqrtSixTenths*rotationMatrix[1][1]*rotationMatrix[1][1]*rotationMatrix[0][0])/2 + (sqrtSixTenths*rotationMatrix[2][1]*rotationMatrix[2][1]*rotationMatrix[0][0])/2 - (sqrtSixTenths*rotationMatrix[1][2]*rotationMatrix[1][2]*rotationMatrix[0][0])/2 + (sqrtSixTenths*rotationMatrix[2][2]*rotationMatrix[2][2]*rotationMatrix[0][0])/2 + sqrtSixTenths*rotationMatrix[1][0]*rotationMatrix[1][0]*rotationMatrix[0][0] - sqrtSixTenths*rotationMatrix[2][0]*rotationMatrix[2][0]*rotationMatrix[0][0];
    rotated1[3] += oct1[0]*element;
    rotated2[3] += oct2[0]*element;
    element = (-3*rotationMatrix[1][1]*rotationMatrix[1][1]*rotationMatrix[0][1])/(2*sqrtTen) + (3*rotationMatrix[2][1]*rotationMatrix[2][1]*rotationMatrix[0][1])/(2*sqrtTen) - (rotationMatrix[0][1]*rotationMatrix[1][2]*rotationMatrix[1][2])/(2*sqrtTen) + (rotationMatrix[0][1]*rotationMatrix[2][2]*rotationMatrix[2][2])/(2*sqrtTen) - (rotationMatrix[1][1]*rotationMatrix[1][2]*rotationMatrix[0][2])/sqrtTen + (rotationMatrix[2][1]*rotationMatrix[2][2]*rotationMatrix[0][2])/sqrtTen + sqrtFourTenths*rotationMatrix[0][1]*rotationMatrix[1][0]*rotationMatrix[1][0] - sqrtFourTenths*rotationMatrix[0][1]*rotationMatrix[2][0]*rotationMatrix[2][0] + 2*sqrtFourTenths*rotationMatrix[1][1]*rotationMatrix[1][0]*rotationMatrix[0][0] - 2*sqrtFourTenths*rotationMatrix[2][1]*rotationMatrix[2][0]*rotationMatrix[0][0];
    rotated1[3] += oct1[1]*element;
    rotated2[3] += oct2[1]*element;
    element = -((rotationMatrix[1][1]*rotationMatrix[0][1]*rotationMatrix[1][2])/sqrtTen) + (rotationMatrix[2][1]*rotationMatrix[0][1]*rotationMatrix[2][2])/sqrtTen - (rotationMatrix[1][1]*rotationMatrix[1][1]*rotationMatrix[0][2])/(2*sqrtTen) + (rotationMatrix[2][1]*rotationMatrix[2][1]*rotationMatrix[0][2])/(2*sqrtTen) - (3*rotationMatrix[1][2]*rotationMatrix[1][2]*rotationMatrix[0][2])/(2*sqrtTen) + (3*rotationMatrix[2][2]*rotationMatrix[2][2]*rotationMatrix[0][2])/(2*sqrtTen) + sqrtFourTenths*rotationMatrix[0][2]*rotationMatrix[1][0]*rotationMatrix[1][0] - sqrtFourTenths*rotationMatrix[0][2]*rotationMatrix[2][0]*rotationMatrix[2][0] + 2*sqrtFourTenths*rotationMatrix[1][2]*rotationMatrix[1][0]*rotationMatrix[0][0] - 2*sqrtFourTenths*rotationMatrix[2][2]*rotationMatrix[2][0]*rotationMatrix[0][0];
    rotated1[3] += oct1[2]*element;
    rotated2[3] += oct2[2]*element;
    element = rotationMatrix[1][1]*rotationMatrix[0][1]*rotationMatrix[1][0] - rotationMatrix[1][2]*rotationMatrix[0][2]*rotationMatrix[1][0] - rotationMatrix[2][1]*rotationMatrix[0][1]*rotationMatrix[2][0] + rotationMatrix[2][2]*rotationMatrix[0][2]*rotationMatrix[2][0] + (rotationMatrix[1][1]*rotationMatrix[1][1]*rotationMatrix[0][0])/2 - (rotationMatrix[2][1]*rotationMatrix[2][1]*rotationMatrix[0][0])/2 - (rotationMatrix[1][2]*rotationMatrix[1][2]*rotationMatrix[0][0])/2 + (rotationMatrix[2][2]*rotationMatrix[2][2]*rotationMatrix[0][0])/2;
    rotated1[3] += oct1[3]*element;
    rotated2[3] += oct2[3]*element;
    element = rotationMatrix[0][1]*rotationMatrix[1][2]*rotationMatrix[1][0] + rotationMatrix[1][1]*rotationMatrix[0][2]*rotationMatrix[1][0] - rotationMatrix[0][1]*rotationMatrix[2][2]*rotationMatrix[2][0] - rotationMatrix[2][1]*rotationMatrix[0][2]*rotationMatrix[2][0] + rotationMatrix[1][1]*rotationMatrix[1][2]*rotationMatrix[0][0] - rotationMatrix[2][1]*rotationMatrix[2][2]*rotationMatrix[0][0];
    rotated1[3] += oct1[4]*element;
    rotated2[3] += oct2[4]*element;
    element = (sqrtThreeHalves*rotationMatrix[1][1]*rotationMatrix[1][1]*rotationMatrix[0][1])/2 - (sqrtThreeHalves*rotationMatrix[2][1]*rotationMatrix[2][1]*rotationMatrix[0][1])/2 - (sqrtThreeHalves*rotationMatrix[0][1]*rotationMatrix[1][2]*rotationMatrix[1][2])/2 + (sqrtThreeHalves*rotationMatrix[0][1]*rotationMatrix[2][2]*rotationMatrix[2][2])/2 - sqrtThreeHalves*rotationMatrix[1][1]*rotationMatrix[1][2]*rotationMatrix[0][2] + sqrtThreeHalves*rotationMatrix[2][1]*rotationMatrix[2][2]*rotationMatrix[0][2];
    rotated1[3] += oct1[5]*element;
    rotated2[3] += oct2[5]*element;
    element = sqrtThreeHalves*rotationMatrix[1][1]*rotationMatrix[0][1]*rotationMatrix[1][2] - sqrtThreeHalves*rotationMatrix[2][1]*rotationMatrix[0][1]*rotationMatrix[2][2] + (sqrtThreeHalves*rotationMatrix[1][1]*rotationMatrix[1][1]*rotationMatrix[0][2])/2 - (sqrtThreeHalves*rotationMatrix[2][1]*rotationMatrix[2][1]*rotationMatrix[0][2])/2 - (sqrtThreeHalves*rotationMatrix[1][2]*rotationMatrix[1][2]*rotationMatrix[0][2])/2 + (sqrtThreeHalves*rotationMatrix[2][2]*rotationMatrix[2][2]*rotationMatrix[0][2])/2;
    rotated1[3] += oct1[6]*element;
    rotated2[3] += oct2[6]*element;

    //rotatedOctopole[4] = 
    element = -(sqrtSixTenths*rotationMatrix[2][1]*rotationMatrix[0][1]*rotationMatrix[1][0]) - sqrtSixTenths*rotationMatrix[2][2]*rotationMatrix[0][2]*rotationMatrix[1][0] - sqrtSixTenths*rotationMatrix[1][1]*rotationMatrix[0][1]*rotationMatrix[2][0] - sqrtSixTenths*rotationMatrix[1][2]*rotationMatrix[0][2]*rotationMatrix[2][0] - sqrtSixTenths*rotationMatrix[1][1]*rotationMatrix[2][1]*rotationMatrix[0][0] - sqrtSixTenths*rotationMatrix[1][2]*rotationMatrix[2][2]*rotationMatrix[0][0] + 2*sqrtSixTenths*rotationMatrix[1][0]*rotationMatrix[2][0]*rotationMatrix[0][0];
    rotated1[4] += oct1[0]*element;
    rotated2[4] += oct2[0]*element;
    element = (-3*rotationMatrix[1][1]*rotationMatrix[2][1]*rotationMatrix[0][1])/sqrtTen - (rotationMatrix[0][1]*rotationMatrix[1][2]*rotationMatrix[2][2])/sqrtTen - (rotationMatrix[2][1]*rotationMatrix[1][2]*rotationMatrix[0][2])/sqrtTen - (rotationMatrix[1][1]*rotationMatrix[2][2]*rotationMatrix[0][2])/sqrtTen + 2*sqrtFourTenths*rotationMatrix[0][1]*rotationMatrix[1][0]*rotationMatrix[2][0] + 2*sqrtFourTenths*rotationMatrix[2][1]*rotationMatrix[1][0]*rotationMatrix[0][0] + 2*sqrtFourTenths*rotationMatrix[1][1]*rotationMatrix[2][0]*rotationMatrix[0][0];
    rotated1[4] += oct1[1]*element;
    rotated2[4] += oct2[1]*element;
    element = -((rotationMatrix[2][1]*rotationMatrix[0][1]*rotationMatrix[1][2])/sqrtTen) - (rotationMatrix[1][1]*rotationMatrix[0][1]*rotationMatrix[2][2])/sqrtTen - (rotationMatrix[1][1]*rotationMatrix[2][1]*rotationMatrix[0][2])/sqrtTen - (3*rotationMatrix[1][2]*rotationMatrix[2][2]*rotationMatrix[0][2])/sqrtTen + 2*sqrtFourTenths*rotationMatrix[0][2]*rotationMatrix[1][0]*rotationMatrix[2][0] + 2*sqrtFourTenths*rotationMatrix[2][2]*rotationMatrix[1][0]*rotationMatrix[0][0] + 2*sqrtFourTenths*rotationMatrix[1][2]*rotationMatrix[2][0]*rotationMatrix[0][0];
    rotated1[4] += oct1[2]*element;
    rotated2[4] += oct2[2]*element;
    element = rotationMatrix[2][1]*rotationMatrix[0][1]*rotationMatrix[1][0] - rotationMatrix[2][2]*rotationMatrix[0][2]*rotationMatrix[1][0] + rotationMatrix[1][1]*rotationMatrix[0][1]*rotationMatrix[2][0] - rotationMatrix[1][2]*rotationMatrix[0][2]*rotationMatrix[2][0] + rotationMatrix[1][1]*rotationMatrix[2][1]*rotationMatrix[0][0] - rotationMatrix[1][2]*rotationMatrix[2][2]*rotationMatrix[0][0];
    rotated1[4] += oct1[3]*element;
    rotated2[4] += oct2[3]*element;
    element = rotationMatrix[0][1]*rotationMatrix[2][2]*rotationMatrix[1][0] + rotationMatrix[2][1]*rotationMatrix[0][2]*rotationMatrix[1][0] + rotationMatrix[0][1]*rotationMatrix[1][2]*rotationMatrix[2][0] + rotationMatrix[1][1]*rotationMatrix[0][2]*rotationMatrix[2][0] + rotationMatrix[2][1]*rotationMatrix[1][2]*rotationMatrix[0][0] + rotationMatrix[1][1]*rotationMatrix[2][2]*rotationMatrix[0][0];
    rotated1[4] += oct1[4]*element;
    rotated2[4] += oct2[4]*element;
    element = sqrtThreeHalves*rotationMatrix[1][1]*rotationMatrix[2][1]*rotationMatrix[0][1] - sqrtThreeHalves*rotationMatrix[0][1]*rotationMatrix[1][2]*rotationMatrix[2][2] - sqrtThreeHalves*rotationMatrix[2][1]*rotationMatrix[1][2]*rotationMatrix[0][2] - sqrtThreeHalves*rotationMatrix[1][1]*rotationMatrix[2][2]*rotationMatrix[0][2];
    rotated1[4] += oct1[5]*element;
    rotated2[4] += oct2[5]*element;
    element = sqrtThreeHalves*rotationMatrix[2][1]*rotationMatrix[0][1]*rotationMatrix[1][2] + sqrtThreeHalves*rotationMatrix[1][1]*rotationMatrix[0][1]*rotationMatrix[2][2] + sqrtThreeHalves*rotationMatrix[1][1]*rotationMatrix[2][1]*rotationMatrix[0][2] - sqrtThreeHalves*rotationMatrix[1][2]*rotationMatrix[2][2]*rotationMatrix[0][2];
    rotated1[4] += oct1[6]*element;
    rotated2[4] += oct2[6]*element;

    //rotatedOctopole[5] = 
    element = (-3*rotationMatrix[1][1]*rotationMatrix[1][1]*rotationMatrix[1][0])/(2*sqrtTen) + (3*rotationMatrix[2][1]*rotationMatrix[2][1]*rotationMatrix[1][0])/(2*sqrtTen) - (3*rotationMatrix[1][2]*rotationMatrix[1][2]*rotationMatrix[1][0])/(2*sqrtTen) + (3*rotationMatrix[2][2]*rotationMatrix[2][2]*rotationMatrix[1][0])/(2*sqrtTen) + rotationMatrix[1][0]*rotationMatrix[1][0]*rotationMatrix[1][0]/sqrtTen + (3*rotationMatrix[1][1]*rotationMatrix[2][1]*rotationMatrix[2][0])/sqrtTen + (3*rotationMatrix[1][2]*rotationMatrix[2][2]*rotationMatrix[2][0])/sqrtTen - (3*rotationMatrix[1][0]*rotationMatrix[2][0]*rotationMatrix[2][0])/sqrtTen;
    rotated1[5] += oct1[0]*element;
    rotated2[5] += oct2[0]*element;
    element = -(sqrtSixTenths*rotationMatrix[1][1]*rotationMatrix[1][1]*rotationMatrix[1][1])/4 + (3*sqrtSixTenths*rotationMatrix[1][1]*rotationMatrix[2][1]*rotationMatrix[2][1])/4 - (sqrtSixTenths*rotationMatrix[1][1]*rotationMatrix[1][2]*rotationMatrix[1][2])/4 + (sqrtSixTenths*rotationMatrix[2][1]*rotationMatrix[1][2]*rotationMatrix[2][2])/2 + (sqrtSixTenths*rotationMatrix[1][1]*rotationMatrix[2][2]*rotationMatrix[2][2])/4 + sqrtSixTenths*rotationMatrix[1][1]*rotationMatrix[1][0]*rotationMatrix[1][0] - 2*sqrtSixTenths*rotationMatrix[2][1]*rotationMatrix[1][0]*rotationMatrix[2][0] - sqrtSixTenths*rotationMatrix[1][1]*rotationMatrix[2][0]*rotationMatrix[2][0];
    rotated1[5] += oct1[1]*element;
    rotated2[5] += oct2[1]*element;
    element = -(sqrtSixTenths*rotationMatrix[1][1]*rotationMatrix[1][1]*rotationMatrix[1][2])/4 + (sqrtSixTenths*rotationMatrix[2][1]*rotationMatrix[2][1]*rotationMatrix[1][2])/4 - (sqrtSixTenths*rotationMatrix[1][2]*rotationMatrix[1][2]*rotationMatrix[1][2])/4 + (sqrtSixTenths*rotationMatrix[1][1]*rotationMatrix[2][1]*rotationMatrix[2][2])/2 + (3*sqrtSixTenths*rotationMatrix[1][2]*rotationMatrix[2][2]*rotationMatrix[2][2])/4 + sqrtSixTenths*rotationMatrix[1][2]*rotationMatrix[1][0]*rotationMatrix[1][0] - 2*sqrtSixTenths*rotationMatrix[2][2]*rotationMatrix[1][0]*rotationMatrix[2][0] - sqrtSixTenths*rotationMatrix[1][2]*rotationMatrix[2][0]*rotationMatrix[2][0];
    rotated1[5] += oct1[2]*element;
    rotated2[5] += oct2[2]*element;
    element = (sqrtThreeHalves*rotationMatrix[1][1]*rotationMatrix[1][1]*rotationMatrix[1][0])/2 - (sqrtThreeHalves*rotationMatrix[2][1]*rotationMatrix[2][1]*rotationMatrix[1][0])/2 - (sqrtThreeHalves*rotationMatrix[1][2]*rotationMatrix[1][2]*rotationMatrix[1][0])/2 + (sqrtThreeHalves*rotationMatrix[2][2]*rotationMatrix[2][2]*rotationMatrix[1][0])/2 - sqrtThreeHalves*rotationMatrix[1][1]*rotationMatrix[2][1]*rotationMatrix[2][0] + sqrtThreeHalves*rotationMatrix[1][2]*rotationMatrix[2][2]*rotationMatrix[2][0];
    rotated1[5] += oct1[3]*element;
    rotated2[5] += oct2[3]*element;
    element = sqrtThreeHalves*rotationMatrix[1][1]*rotationMatrix[1][2]*rotationMatrix[1][0] - sqrtThreeHalves*rotationMatrix[2][1]*rotationMatrix[2][2]*rotationMatrix[1][0] - sqrtThreeHalves*rotationMatrix[2][1]*rotationMatrix[1][2]*rotationMatrix[2][0] - sqrtThreeHalves*rotationMatrix[1][1]*rotationMatrix[2][2]*rotationMatrix[2][0];
    rotated1[5] += oct1[4]*element;
    rotated2[5] += oct2[4]*element;
    element = rotationMatrix[1][1]*rotationMatrix[1][1]*rotationMatrix[1][1]/4 - (3*rotationMatrix[1][1]*rotationMatrix[2][1]*rotationMatrix[2][1])/4 - (3*rotationMatrix[1][1]*rotationMatrix[1][2]*rotationMatrix[1][2])/4 + (3*rotationMatrix[2][1]*rotationMatrix[1][2]*rotationMatrix[2][2])/2 + (3*rotationMatrix[1][1]*rotationMatrix[2][2]*rotationMatrix[2][2])/4;
    rotated1[5] += oct1[5]*element;
    rotated2[5] += oct2[5]*element;
    element = (3*rotationMatrix[1][1]*rotationMatrix[1][1]*rotationMatrix[1][2])/4 - (3*rotationMatrix[2][1]*rotationMatrix[2][1]*rotationMatrix[1][2])/4 - rotationMatrix[1][2]*rotationMatrix[1][2]*rotationMatrix[1][2]/4 - (3*rotationMatrix[1][1]*rotationMatrix[2][1]*rotationMatrix[2][2])/2 + (3*rotationMatrix[1][2]*rotationMatrix[2][2]*rotationMatrix[2][2])/4;
    rotated1[5] += oct1[6]*element;
    rotated2[5] += oct2[6]*element;

    //rotatedOctopole[6] = 
    element = (-3*rotationMatrix[1][1]*rotationMatrix[2][1]*rotationMatrix[1][0])/sqrtTen - (3*rotationMatrix[1][2]*rotationMatrix[2][2]*rotationMatrix[1][0])/sqrtTen - (3*rotationMatrix[1][1]*rotationMatrix[1][1]*rotationMatrix[2][0])/(2*sqrtTen) + (3*rotationMatrix[2][1]*rotationMatrix[2][1]*rotationMatrix[2][0])/(2*sqrtTen) - (3*rotationMatrix[1][2]*rotationMatrix[1][2]*rotationMatrix[2][0])/(2*sqrtTen) + (3*rotationMatrix[2][2]*rotationMatrix[2][2]*rotationMatrix[2][0])/(2*sqrtTen) + (3*rotationMatrix[1][0]*rotationMatrix[1][0]*rotationMatrix[2][0])/sqrtTen - rotationMatrix[2][0]*rotationMatrix[2][0]*rotationMatrix[2][0]/sqrtTen;
    rotated1[6] += oct1[0]*element;
    rotated2[6] += oct2[0]*element;
    element = (-3*sqrtSixTenths*rotationMatrix[1][1]*rotationMatrix[1][1]*rotationMatrix[2][1])/4 + (sqrtSixTenths*rotationMatrix[2][1]*rotationMatrix[2][1]*rotationMatrix[2][1])/4 - (sqrtSixTenths*rotationMatrix[2][1]*rotationMatrix[1][2]*rotationMatrix[1][2])/4 - (sqrtSixTenths*rotationMatrix[1][1]*rotationMatrix[1][2]*rotationMatrix[2][2])/2 + (sqrtSixTenths*rotationMatrix[2][1]*rotationMatrix[2][2]*rotationMatrix[2][2])/4 + sqrtSixTenths*rotationMatrix[2][1]*rotationMatrix[1][0]*rotationMatrix[1][0] + 2*sqrtSixTenths*rotationMatrix[1][1]*rotationMatrix[1][0]*rotationMatrix[2][0] - sqrtSixTenths*rotationMatrix[2][1]*rotationMatrix[2][0]*rotationMatrix[2][0];
    rotated1[6] += oct1[1]*element;
    rotated2[6] += oct2[1]*element;
    element = -(sqrtSixTenths*rotationMatrix[1][1]*rotationMatrix[2][1]*rotationMatrix[1][2])/2 - (sqrtSixTenths*rotationMatrix[1][1]*rotationMatrix[1][1]*rotationMatrix[2][2])/4 + (sqrtSixTenths*rotationMatrix[2][1]*rotationMatrix[2][1]*rotationMatrix[2][2])/4 - (3*sqrtSixTenths*rotationMatrix[1][2]*rotationMatrix[1][2]*rotationMatrix[2][2])/4 + (sqrtSixTenths*rotationMatrix[2][2]*rotationMatrix[2][2]*rotationMatrix[2][2])/4 + sqrtSixTenths*rotationMatrix[2][2]*rotationMatrix[1][0]*rotationMatrix[1][0] + 2*sqrtSixTenths*rotationMatrix[1][2]*rotationMatrix[1][0]*rotationMatrix[2][0] - sqrtSixTenths*rotationMatrix[2][2]*rotationMatrix[2][0]*rotationMatrix[2][0];
    rotated1[6] += oct1[2]*element;
    rotated2[6] += oct2[2]*element;
    element = sqrtThreeHalves*rotationMatrix[1][1]*rotationMatrix[2][1]*rotationMatrix[1][0] - sqrtThreeHalves*rotationMatrix[1][2]*rotationMatrix[2][2]*rotationMatrix[1][0] + (sqrtThreeHalves*rotationMatrix[1][1]*rotationMatrix[1][1]*rotationMatrix[2][0])/2 - (sqrtThreeHalves*rotationMatrix[2][1]*rotationMatrix[2][1]*rotationMatrix[2][0])/2 - (sqrtThreeHalves*rotationMatrix[1][2]*rotationMatrix[1][2]*rotationMatrix[2][0])/2 + (sqrtThreeHalves*rotationMatrix[2][2]*rotationMatrix[2][2]*rotationMatrix[2][0])/2;
    rotated1[6] += oct1[3]*element;
    rotated2[6] += oct2[3]*element;
    element = sqrtThreeHalves*rotationMatrix[2][1]*rotationMatrix[1][2]*rotationMatrix[1][0] + sqrtThreeHalves*rotationMatrix[1][1]*rotationMatrix[2][2]*rotationMatrix[1][0] + sqrtThreeHalves*rotationMatrix[1][1]*rotationMatrix[1][2]*rotationMatrix[2][0] - sqrtThreeHalves*rotationMatrix[2][1]*rotationMatrix[2][2]*rotationMatrix[2][0];
    rotated1[6] += oct1[4]*element;
    rotated2[6] += oct2[4]*element;
    element = (3*rotationMatrix[1][1]*rotationMatrix[1][1]*rotationMatrix[2][1])/4 - rotationMatrix[2][1]*rotationMatrix[2][1]*rotationMatrix[2][1]/4 - (3*rotationMatrix[2][1]*rotationMatrix[1][2]*rotationMatrix[1][2])/4 - (3*rotationMatrix[1][1]*rotationMatrix[1][2]*rotationMatrix[2][2])/2 + (3*rotationMatrix[2][1]*rotationMatrix[2][2]*rotationMatrix[2][2])/4;
    rotated1[6] += oct1[5]*element;
    rotated2[6] += oct2[5]*element;
    element = (3*rotationMatrix[1][1]*rotationMatrix[2][1]*rotationMatrix[1][2])/2 + (3*rotationMatrix[1][1]*rotationMatrix[1][1]*rotationMatrix[2][2])/4 - (3*rotationMatrix[2][1]*rotationMatrix[2][1]*rotationMatrix[2][2])/4 - (3*rotationMatrix[1][2]*rotationMatrix[1][2]*rotationMatrix[2][2])/4 + rotationMatrix[2][2]*rotationMatrix[2][2]*rotationMatrix[2][2]/4;
    rotated1[6] += oct1[6]*element;
    rotated2[6] += oct2[6]*element;
}

#if 0
    //rotatedOctopole[0] = 
    element = (-3*vectorX.z*vectorX.z*vectorZ.z)/2 - (3*vectorY.z*vectorY.z*vectorZ.z)/2 + vectorZ.z*vectorZ.z*vectorZ.z;
    element = -(sqrtThreeHalves*vectorX.z*vectorX.z*vectorX.z)/2 - (sqrtThreeHalves*vectorX.z*vectorY.z*vectorY.z)/2  + sqrtSix*vectorX.z*vectorZ.z*vectorZ.z;
    element = -(sqrtThreeHalves*vectorX.z*vectorX.z*vectorY.z)/2 - (sqrtThreeHalves*vectorY.z*vectorY.z*vectorY.z)/2 + sqrtSix*vectorY.z*vectorZ.z*vectorZ.z;
    element = (sqrtFifteen*vectorX.z*vectorX.z*vectorZ.z)/2 - (sqrtFifteen*vectorY.z*vectorY.z*vectorZ.z)/2;
    element = sqrtFifteen*vectorX.z*vectorY.z*vectorZ.z
    element = (sqrtFiveHalves*vectorX.z*vectorX.z*vectorX.z)/2 - (3*sqrtFiveHalves*vectorX.z*vectorY.z*vectorY.z)/2;
    element = (3*sqrtFiveHalves*vectorX.z*vectorX.z*vectorY.z)/2 - (sqrtFiveHalves*vectorY.z*vectorY.z*vectorY.z)/2;

    //rotatedOctopole[1] = 
    element = -(sqrtThreeHalves*vectorX.z*vectorX.z*vectorZ.x)/2 - (sqrtThreeHalves*vectorY.z*vectorY.z*vectorZ.x)/2 - sqrtThreeHalves*vectorX.x*vectorX.z*vectorZ.z - sqrtThreeHalves*vectorY.x*vectorY.z*vectorZ.z + sqrtThreeHalves*vectorZ.x*vectorZ.z*vectorZ.z;
    element = (-3*vectorX.x*vectorX.z*vectorX.z)/4 - (vectorX.z*vectorY.x*vectorY.z)/2 - (vectorX.x*vectorY.z*vectorY.z)/4 + 2*vectorX.z*vectorZ.x*vectorZ.z + vectorX.x*vectorZ.z*vectorZ.z;
    element = -(vectorX.z*vectorX.z*vectorY.x)/4 - (vectorX.x*vectorX.z*vectorY.z)/2 - (3*vectorY.x*vectorY.z*vectorY.z)/4 + 2*vectorY.z*vectorZ.x*vectorZ.z + vectorY.x*vectorZ.z*vectorZ.z
    element = (sqrtFiveHalves*vectorX.z*vectorX.z*vectorZ.x)/2 - (sqrtFiveHalves*vectorY.z*vectorY.z*vectorZ.x)/2 + sqrtFiveHalves*vectorX.x*vectorX.z*vectorZ.z - sqrtFiveHalves*vectorY.x*vectorY.z*vectorZ.z;
    element = sqrtFiveHalves*vectorX.z*vectorY.z*vectorZ.x + sqrtFiveHalves*vectorX.z*vectorY.x*vectorZ.z + sqrtFiveHalves*vectorX.x*vectorY.z*vectorZ.z;
    element = sqrtFifteen*vectorX.x*vectorX.z*vectorX.z/4 - (sqrtFifteen*vectorX.z*vectorY.x*vectorY.z)/2 - (sqrtFifteen*vectorX.x*vectorY.z*vectorY.z)/4;
    element = sqrtFifteen*vectorX.z*vectorX.z*vectorY.x/4 + (sqrtFifteen*vectorX.x*vectorX.z*vectorY.z)/2 - (sqrtFifteen*vectorY.x*vectorY.z*vectorY.z)/4;

    //rotatedOctopole[2] = 
    element = -(sqrtThreeHalves*vectorX.z*vectorX.z*vectorZ.y)/2 - (sqrtThreeHalves*vectorY.z*vectorY.z*vectorZ.y)/2 - sqrtThreeHalves*vectorX.y*vectorX.z*vectorZ.z - sqrtThreeHalves*vectorY.y*vectorY.z*vectorZ.z + sqrtThreeHalves*vectorZ.y*vectorZ.z*vectorZ.z;
    element = (-3*vectorX.y*vectorX.z*vectorX.z)/4 - (vectorX.z*vectorY.y*vectorY.z)/2 - (vectorX.y*vectorY.z*vectorY.z)/4 + 2*vectorX.z*vectorZ.y*vectorZ.z + vectorX.y*vectorZ.z*vectorZ.z;
    element = -(vectorX.z*vectorX.z*vectorY.y)/4 - (vectorX.y*vectorX.z*vectorY.z)/2 - (3*vectorY.y*vectorY.z*vectorY.z)/4 + 2*vectorY.z*vectorZ.y*vectorZ.z + vectorY.y*vectorZ.z*vectorZ.z;
    element = (sqrtFiveHalves*vectorX.z*vectorX.z*vectorZ.y)/2 - (sqrtFiveHalves*vectorY.z*vectorY.z*vectorZ.y)/2 + sqrtFiveHalves*vectorX.y*vectorX.z*vectorZ.z - sqrtFiveHalves*vectorY.y*vectorY.z*vectorZ.z;
    element = sqrtFiveHalves*vectorX.z*vectorY.z*vectorZ.y + sqrtFiveHalves*vectorX.z*vectorY.y*vectorZ.z + sqrtFiveHalves*vectorX.y*vectorY.z*vectorZ.z;
    element = (sqrtFifteen*vectorX.y*vectorX.z*vectorX.z)/4 - (sqrtFifteen*vectorX.z*vectorY.y*vectorY.z)/2 - (sqrtFifteen*vectorX.y*vectorY.z*vectorY.z)/4;
    element = (sqrtFifteen*vectorX.z*vectorX.z*vectorY.y)/4 + (sqrtFifteen*vectorX.y*vectorX.z*vectorY.z)/2 - (sqrtFifteen*vectorY.y*vectorY.z*vectorY.z)/4;

    //rotatedOctopole[3] = 
    element = -(sqrtSixTenths*vectorX.x*vectorX.z*vectorZ.x) - sqrtSixTenths*vectorY.x*vectorY.z*vectorZ.x + sqrtSixTenths*vectorX.y*vectorX.z*vectorZ.y + sqrtSixTenths*vectorY.y*vectorY.z*vectorZ.y - (sqrtSixTenths*vectorX.x*vectorX.x*vectorZ.z)/2 + (sqrtSixTenths*vectorX.y*vectorX.y*vectorZ.z)/2 - (sqrtSixTenths*vectorY.x*vectorY.x*vectorZ.z)/2 + (sqrtSixTenths*vectorY.y*vectorY.y*vectorZ.z)/2 + sqrtSixTenths*vectorZ.x*vectorZ.x*vectorZ.z - sqrtSixTenths*vectorZ.y*vectorZ.y*vectorZ.z;
    element = (-3*vectorX.x*vectorX.x*vectorX.z)/(2*sqrtTen) + (3*vectorX.y*vectorX.y*vectorX.z)/(2*sqrtTen) - (vectorX.z*vectorY.x*vectorY.x)/(2*sqrtTen) + (vectorX.z*vectorY.y*vectorY.y)/(2*sqrtTen) - (vectorX.x*vectorY.x*vectorY.z)/sqrtTen + (vectorX.y*vectorY.y*vectorY.z)/sqrtTen + sqrtFourTenths*vectorX.z*vectorZ.x*vectorZ.x - sqrtFourTenths*vectorX.z*vectorZ.y*vectorZ.y + 2*sqrtFourTenths*vectorX.x*vectorZ.x*vectorZ.z - 2*sqrtFourTenths*vectorX.y*vectorZ.y*vectorZ.z;
    element = -((vectorX.x*vectorX.z*vectorY.x)/sqrtTen) + (vectorX.y*vectorX.z*vectorY.y)/sqrtTen - (vectorX.x*vectorX.x*vectorY.z)/(2*sqrtTen) + (vectorX.y*vectorX.y*vectorY.z)/(2*sqrtTen) - (3*vectorY.x*vectorY.x*vectorY.z)/(2*sqrtTen) + (3*vectorY.y*vectorY.y*vectorY.z)/(2*sqrtTen) + sqrtFourTenths*vectorY.z*vectorZ.x*vectorZ.x - sqrtFourTenths*vectorY.z*vectorZ.y*vectorZ.y + 2*sqrtFourTenths*vectorY.x*vectorZ.x*vectorZ.z - 2*sqrtFourTenths*vectorY.y*vectorZ.y*vectorZ.z;
    element = vectorX.x*vectorX.z*vectorZ.x - vectorY.x*vectorY.z*vectorZ.x - vectorX.y*vectorX.z*vectorZ.y + vectorY.y*vectorY.z*vectorZ.y + (vectorX.x*vectorX.x*vectorZ.z)/2 - (vectorX.y*vectorX.y*vectorZ.z)/2 - (vectorY.x*vectorY.x*vectorZ.z)/2 + (vectorY.y*vectorY.y*vectorZ.z)/2;
    element = vectorX.z*vectorY.x*vectorZ.x + vectorX.x*vectorY.z*vectorZ.x - vectorX.z*vectorY.y*vectorZ.y - vectorX.y*vectorY.z*vectorZ.y + vectorX.x*vectorY.x*vectorZ.z - vectorX.y*vectorY.y*vectorZ.z;
    element = (sqrtThreeHalves*vectorX.x*vectorX.x*vectorX.z)/2 - (sqrtThreeHalves*vectorX.y*vectorX.y*vectorX.z)/2 - (sqrtThreeHalves*vectorX.z*vectorY.x*vectorY.x)/2 + (sqrtThreeHalves*vectorX.z*vectorY.y*vectorY.y)/2 - sqrtThreeHalves*vectorX.x*vectorY.x*vectorY.z + sqrtThreeHalves*vectorX.y*vectorY.y*vectorY.z;
    element = sqrtThreeHalves*vectorX.x*vectorX.z*vectorY.x - sqrtThreeHalves*vectorX.y*vectorX.z*vectorY.y + (sqrtThreeHalves*vectorX.x*vectorX.x*vectorY.z)/2 - (sqrtThreeHalves*vectorX.y*vectorX.y*vectorY.z)/2 - (sqrtThreeHalves*vectorY.x*vectorY.x*vectorY.z)/2 + (sqrtThreeHalves*vectorY.y*vectorY.y*vectorY.z)/2;

    //rotatedOctopole[4] = 
    element = -(sqrtSixTenths*vectorX.y*vectorX.z*vectorZ.x) - sqrtSixTenths*vectorY.y*vectorY.z*vectorZ.x - sqrtSixTenths*vectorX.x*vectorX.z*vectorZ.y - sqrtSixTenths*vectorY.x*vectorY.z*vectorZ.y - sqrtSixTenths*vectorX.x*vectorX.y*vectorZ.z - sqrtSixTenths*vectorY.x*vectorY.y*vectorZ.z + 2*sqrtSixTenths*vectorZ.x*vectorZ.y*vectorZ.z;
    element = (-3*vectorX.x*vectorX.y*vectorX.z)/sqrtTen - (vectorX.z*vectorY.x*vectorY.y)/sqrtTen - (vectorX.y*vectorY.x*vectorY.z)/sqrtTen - (vectorX.x*vectorY.y*vectorY.z)/sqrtTen + 2*sqrtFourTenths*vectorX.z*vectorZ.x*vectorZ.y + 2*sqrtFourTenths*vectorX.y*vectorZ.x*vectorZ.z + 2*sqrtFourTenths*vectorX.x*vectorZ.y*vectorZ.z;
    element = -((vectorX.y*vectorX.z*vectorY.x)/sqrtTen) - (vectorX.x*vectorX.z*vectorY.y)/sqrtTen - (vectorX.x*vectorX.y*vectorY.z)/sqrtTen - (3*vectorY.x*vectorY.y*vectorY.z)/sqrtTen + 2*sqrtFourTenths*vectorY.z*vectorZ.x*vectorZ.y + 2*sqrtFourTenths*vectorY.y*vectorZ.x*vectorZ.z + 2*sqrtFourTenths*vectorY.x*vectorZ.y*vectorZ.z;
    element = vectorX.y*vectorX.z*vectorZ.x - vectorY.y*vectorY.z*vectorZ.x + vectorX.x*vectorX.z*vectorZ.y - vectorY.x*vectorY.z*vectorZ.y + vectorX.x*vectorX.y*vectorZ.z - vectorY.x*vectorY.y*vectorZ.z;
    element = vectorX.z*vectorY.y*vectorZ.x + vectorX.y*vectorY.z*vectorZ.x + vectorX.z*vectorY.x*vectorZ.y + vectorX.x*vectorY.z*vectorZ.y + vectorX.y*vectorY.x*vectorZ.z + vectorX.x*vectorY.y*vectorZ.z;
    element = sqrtThreeHalves*vectorX.x*vectorX.y*vectorX.z - sqrtThreeHalves*vectorX.z*vectorY.x*vectorY.y - sqrtThreeHalves*vectorX.y*vectorY.x*vectorY.z - sqrtThreeHalves*vectorX.x*vectorY.y*vectorY.z;
    element = sqrtThreeHalves*vectorX.y*vectorX.z*vectorY.x + sqrtThreeHalves*vectorX.x*vectorX.z*vectorY.y + sqrtThreeHalves*vectorX.x*vectorX.y*vectorY.z - sqrtThreeHalves*vectorY.x*vectorY.y*vectorY.z;

    //rotatedOctopole[5] = 
    element = (-3*vectorX.x*vectorX.x*vectorZ.x)/(2*sqrtTen) + (3*vectorX.y*vectorX.y*vectorZ.x)/(2*sqrtTen) - (3*vectorY.x*vectorY.x*vectorZ.x)/(2*sqrtTen) + (3*vectorY.y*vectorY.y*vectorZ.x)/(2*sqrtTen) + vectorZ.x*vectorZ.x*vectorZ.x/sqrtTen + (3*vectorX.x*vectorX.y*vectorZ.y)/sqrtTen + (3*vectorY.x*vectorY.y*vectorZ.y)/sqrtTen - (3*vectorZ.x*vectorZ.y*vectorZ.y)/sqrtTen;
    element = -(sqrtSixTenths*vectorX.x*vectorX.x*vectorX.x)/4 + (3*sqrtSixTenths*vectorX.x*vectorX.y*vectorX.y)/4 - (sqrtSixTenths*vectorX.x*vectorY.x*vectorY.x)/4 + (sqrtSixTenths*vectorX.y*vectorY.x*vectorY.y)/2 + (sqrtSixTenths*vectorX.x*vectorY.y*vectorY.y)/4 + sqrtSixTenths*vectorX.x*vectorZ.x*vectorZ.x - 2*sqrtSixTenths*vectorX.y*vectorZ.x*vectorZ.y - sqrtSixTenths*vectorX.x*vectorZ.y*vectorZ.y;
    element = -(sqrtSixTenths*vectorX.x*vectorX.x*vectorY.x)/4 + (sqrtSixTenths*vectorX.y*vectorX.y*vectorY.x)/4 - (sqrtSixTenths*vectorY.x*vectorY.x*vectorY.x)/4 + (sqrtSixTenths*vectorX.x*vectorX.y*vectorY.y)/2 + (3*sqrtSixTenths*vectorY.x*vectorY.y*vectorY.y)/4 + sqrtSixTenths*vectorY.x*vectorZ.x*vectorZ.x - 2*sqrtSixTenths*vectorY.y*vectorZ.x*vectorZ.y - sqrtSixTenths*vectorY.x*vectorZ.y*vectorZ.y;
    element = (sqrtThreeHalves*vectorX.x*vectorX.x*vectorZ.x)/2 - (sqrtThreeHalves*vectorX.y*vectorX.y*vectorZ.x)/2 - (sqrtThreeHalves*vectorY.x*vectorY.x*vectorZ.x)/2 + (sqrtThreeHalves*vectorY.y*vectorY.y*vectorZ.x)/2 - sqrtThreeHalves*vectorX.x*vectorX.y*vectorZ.y + sqrtThreeHalves*vectorY.x*vectorY.y*vectorZ.y;
    element = sqrtThreeHalves*vectorX.x*vectorY.x*vectorZ.x - sqrtThreeHalves*vectorX.y*vectorY.y*vectorZ.x - sqrtThreeHalves*vectorX.y*vectorY.x*vectorZ.y - sqrtThreeHalves*vectorX.x*vectorY.y*vectorZ.y;
    element = vectorX.x*vectorX.x*vectorX.x/4 - (3*vectorX.x*vectorX.y*vectorX.y)/4 - (3*vectorX.x*vectorY.x*vectorY.x)/4 + (3*vectorX.y*vectorY.x*vectorY.y)/2 + (3*vectorX.x*vectorY.y*vectorY.y)/4;
    element = (3*vectorX.x*vectorX.x*vectorY.x)/4 - (3*vectorX.y*vectorX.y*vectorY.x)/4 - vectorY.x*vectorY.x*vectorY.x/4 - (3*vectorX.x*vectorX.y*vectorY.y)/2 + (3*vectorY.x*vectorY.y*vectorY.y)/4;

    //rotatedOctopole[6] = 
    element = (-3*vectorX.x*vectorX.y*vectorZ.x)/sqrtTen - (3*vectorY.x*vectorY.y*vectorZ.x)/sqrtTen - (3*vectorX.x*vectorX.x*vectorZ.y)/(2*sqrtTen) + (3*vectorX.y*vectorX.y*vectorZ.y)/(2*sqrtTen) - (3*vectorY.x*vectorY.x*vectorZ.y)/(2*sqrtTen) + (3*vectorY.y*vectorY.y*vectorZ.y)/(2*sqrtTen) + (3*vectorZ.x*vectorZ.x*vectorZ.y)/sqrtTen - vectorZ.y*vectorZ.y*vectorZ.y/sqrtTen;
    element = (-3*sqrtSixTenths*vectorX.x*vectorX.x*vectorX.y)/4 + (sqrtSixTenths*vectorX.y*vectorX.y*vectorX.y)/4 - (sqrtSixTenths*vectorX.y*vectorY.x*vectorY.x)/4 - (sqrtSixTenths*vectorX.x*vectorY.x*vectorY.y)/2 + (sqrtSixTenths*vectorX.y*vectorY.y*vectorY.y)/4 + sqrtSixTenths*vectorX.y*vectorZ.x*vectorZ.x + 2*sqrtSixTenths*vectorX.x*vectorZ.x*vectorZ.y - sqrtSixTenths*vectorX.y*vectorZ.y*vectorZ.y;
    element = -(sqrtSixTenths*vectorX.x*vectorX.y*vectorY.x)/2 - (sqrtSixTenths*vectorX.x*vectorX.x*vectorY.y)/4 + (sqrtSixTenths*vectorX.y*vectorX.y*vectorY.y)/4 - (3*sqrtSixTenths*vectorY.x*vectorY.x*vectorY.y)/4 + (sqrtSixTenths*vectorY.y*vectorY.y*vectorY.y)/4 + sqrtSixTenths*vectorY.y*vectorZ.x*vectorZ.x + 2*sqrtSixTenths*vectorY.x*vectorZ.x*vectorZ.y - sqrtSixTenths*vectorY.y*vectorZ.y*vectorZ.y;
    element = sqrtThreeHalves*vectorX.x*vectorX.y*vectorZ.x - sqrtThreeHalves*vectorY.x*vectorY.y*vectorZ.x + (sqrtThreeHalves*vectorX.x*vectorX.x*vectorZ.y)/2 - (sqrtThreeHalves*vectorX.y*vectorX.y*vectorZ.y)/2 - (sqrtThreeHalves*vectorY.x*vectorY.x*vectorZ.y)/2 + (sqrtThreeHalves*vectorY.y*vectorY.y*vectorZ.y)/2;
    element = sqrtThreeHalves*vectorX.y*vectorY.x*vectorZ.x + sqrtThreeHalves*vectorX.x*vectorY.y*vectorZ.x + sqrtThreeHalves*vectorX.x*vectorY.x*vectorZ.y - sqrtThreeHalves*vectorX.y*vectorY.y*vectorZ.y;
    element = (3*vectorX.x*vectorX.x*vectorX.y)/4 - vectorX.y*vectorX.y*vectorX.y/4 - (3*vectorX.y*vectorY.x*vectorY.x)/4 - (3*vectorX.x*vectorY.x*vectorY.y)/2 + (3*vectorX.y*vectorY.y*vectorY.y)/4;
    element = (3*vectorX.x*vectorX.y*vectorY.x)/2 + (3*vectorX.x*vectorX.x*vectorY.y)/4 - (3*vectorX.y*vectorX.y*vectorY.y)/4 - (3*vectorY.x*vectorY.x*vectorY.y)/4 + vectorY.y*vectorY.y*vectorY.y/4;
#endif
