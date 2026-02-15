#pragma once

#include <cuda_runtime.h>

#include "kernels.h"

struct DeviceArrays {
  float *x = nullptr;
  float *y = nullptr;
  float *z = nullptr;
  float *vx = nullptr;
  float *vy = nullptr;
  float *vz = nullptr;
  float *m = nullptr;
  float *rad = nullptr;
  float *ax = nullptr;
  float *ay = nullptr;
  float *az = nullptr;
  int *partner = nullptr;
  int2 *pairs = nullptr;
  int *pairCount = nullptr;
  int maxPairs = 0;
  float *energy = nullptr;
  float *minDist = nullptr;
  int *minPartner = nullptr;
  int *minBody = nullptr;
  float *tmpDist = nullptr;
  int *tmpIdx = nullptr;
  CollisionDebug *debug = nullptr;
  double *reduceA = nullptr;
  double *reduceB = nullptr;
  double *reduceAx = nullptr;
  double *reduceAy = nullptr;
  double *reduceAz = nullptr;
  int n = 0;
};

void allocateDeviceArrays(DeviceArrays &d, int n, bool collisions, bool energy, bool minDist,
                          bool debug, bool totals, bool angmom);
void freeDeviceArrays(DeviceArrays &d);

void launchComputeAccel(const DeviceArrays &d, int n, float G, float eps2, int blockSize);
void launchUpdateVelHalf(const DeviceArrays &d, int n, float dt, int blockSize);
void launchUpdatePos(const DeviceArrays &d, int n, float dt, int blockSize);
void launchFinalizeVel(const DeviceArrays &d, int n, float dt, int blockSize);

void launchResetPairCount(const DeviceArrays &d);
void launchDetectCollisionPairs(const DeviceArrays &d, int n, int blockSize);
void launchResolveCollisionPairs(const DeviceArrays &d, int pairCount, float restitution, bool debug);

void launchComputeEnergy(const DeviceArrays &d, int n, float G, float eps2, int blockSize);
void launchComputeMinDist(const DeviceArrays &d, int n, int blockSize);
void computeGlobalMinPair(const DeviceArrays &d, int n, int blockSize,
                          float *outMinDist, int *outMinBody, int *outMinPartner);
void launchResetCollisionDebug(const DeviceArrays &d);

bool computeTotalEnergy(const DeviceArrays &d, int n, float G, float eps2, int blockSize,
                        double *outK, double *outU);
bool computeAngularMomentum(const DeviceArrays &d, int n, int blockSize,
                            double *outLx, double *outLy, double *outLz);
