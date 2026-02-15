#include "integrator.h"

#include "kernels.h"
#include "utils.h"

#include <algorithm>
#include <climits>

void allocateDeviceArrays(DeviceArrays &d, int n, bool collisions, bool energy, bool minDist,
                          bool debug, bool totals, bool angmom) {
  d.n = n;
  CUDA_CHECK(cudaMalloc(&d.x, n * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d.y, n * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d.z, n * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d.vx, n * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d.vy, n * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d.vz, n * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d.m, n * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d.ax, n * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d.ay, n * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d.az, n * sizeof(float)));

  if (collisions) {
    CUDA_CHECK(cudaMalloc(&d.rad, n * sizeof(float)));
    d.partner = nullptr;
    size_t pairCap = static_cast<size_t>(n) * static_cast<size_t>(n - 1) / 2;
    if (pairCap > static_cast<size_t>(INT_MAX)) {
      pairCap = static_cast<size_t>(INT_MAX);
    }
    d.maxPairs = static_cast<int>(pairCap);
    CUDA_CHECK(cudaMalloc(&d.pairs, d.maxPairs * sizeof(int2)));
    CUDA_CHECK(cudaMalloc(&d.pairCount, sizeof(int)));
    if (debug) {
      CUDA_CHECK(cudaMalloc(&d.debug, sizeof(CollisionDebug)));
    }
  } else {
    d.rad = nullptr;
    d.partner = nullptr;
    d.pairs = nullptr;
    d.pairCount = nullptr;
    d.maxPairs = 0;
  }

  if (energy) {
    CUDA_CHECK(cudaMalloc(&d.energy, n * sizeof(float)));
  }
  if (minDist) {
    CUDA_CHECK(cudaMalloc(&d.minDist, n * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d.minPartner, n * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d.minBody, n * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d.tmpDist, n * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d.tmpIdx, n * sizeof(int)));
  }

  if (totals || angmom) {
    CUDA_CHECK(cudaMalloc(&d.reduceA, n * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d.reduceB, n * sizeof(double)));
  }
  if (angmom) {
    CUDA_CHECK(cudaMalloc(&d.reduceAx, n * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d.reduceAy, n * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d.reduceAz, n * sizeof(double)));
  }
}

void freeDeviceArrays(DeviceArrays &d) {
  if (d.x) CUDA_CHECK(cudaFree(d.x));
  if (d.y) CUDA_CHECK(cudaFree(d.y));
  if (d.z) CUDA_CHECK(cudaFree(d.z));
  if (d.vx) CUDA_CHECK(cudaFree(d.vx));
  if (d.vy) CUDA_CHECK(cudaFree(d.vy));
  if (d.vz) CUDA_CHECK(cudaFree(d.vz));
  if (d.m) CUDA_CHECK(cudaFree(d.m));
  if (d.rad) CUDA_CHECK(cudaFree(d.rad));
  if (d.ax) CUDA_CHECK(cudaFree(d.ax));
  if (d.ay) CUDA_CHECK(cudaFree(d.ay));
  if (d.az) CUDA_CHECK(cudaFree(d.az));
  if (d.partner) CUDA_CHECK(cudaFree(d.partner));
  if (d.pairs) CUDA_CHECK(cudaFree(d.pairs));
  if (d.pairCount) CUDA_CHECK(cudaFree(d.pairCount));
  if (d.energy) CUDA_CHECK(cudaFree(d.energy));
  if (d.minDist) CUDA_CHECK(cudaFree(d.minDist));
  if (d.minPartner) CUDA_CHECK(cudaFree(d.minPartner));
  if (d.minBody) CUDA_CHECK(cudaFree(d.minBody));
  if (d.tmpDist) CUDA_CHECK(cudaFree(d.tmpDist));
  if (d.tmpIdx) CUDA_CHECK(cudaFree(d.tmpIdx));
  if (d.debug) CUDA_CHECK(cudaFree(d.debug));
  if (d.reduceA) CUDA_CHECK(cudaFree(d.reduceA));
  if (d.reduceB) CUDA_CHECK(cudaFree(d.reduceB));
  if (d.reduceAx) CUDA_CHECK(cudaFree(d.reduceAx));
  if (d.reduceAy) CUDA_CHECK(cudaFree(d.reduceAy));
  if (d.reduceAz) CUDA_CHECK(cudaFree(d.reduceAz));
  d = DeviceArrays();
}

void launchComputeAccel(const DeviceArrays &d, int n, float G, float eps2, int blockSize) {
  int grid = divUp(n, blockSize);
  size_t shmem = 4 * blockSize * sizeof(float);
  compute_accel_tiled<<<grid, blockSize, shmem>>>(d.x, d.y, d.z, d.m, d.ax, d.ay, d.az, n, G, eps2);
  CUDA_CHECK_LAST("compute_accel_tiled");
}

void launchUpdateVelHalf(const DeviceArrays &d, int n, float dt, int blockSize) {
  int grid = divUp(n, blockSize);
  update_vel_half<<<grid, blockSize>>>(d.vx, d.vy, d.vz, d.ax, d.ay, d.az, n, dt);
  CUDA_CHECK_LAST("update_vel_half");
}

void launchUpdatePos(const DeviceArrays &d, int n, float dt, int blockSize) {
  int grid = divUp(n, blockSize);
  update_pos<<<grid, blockSize>>>(d.x, d.y, d.z, d.vx, d.vy, d.vz, n, dt);
  CUDA_CHECK_LAST("update_pos");
}

void launchFinalizeVel(const DeviceArrays &d, int n, float dt, int blockSize) {
  int grid = divUp(n, blockSize);
  finalize_vel<<<grid, blockSize>>>(d.vx, d.vy, d.vz, d.ax, d.ay, d.az, n, dt);
  CUDA_CHECK_LAST("finalize_vel");
}

void launchResetPairCount(const DeviceArrays &d) {
  if (!d.pairCount) {
    return;
  }
  CUDA_CHECK(cudaMemset(d.pairCount, 0, sizeof(int)));
}

void launchDetectCollisionPairs(const DeviceArrays &d, int n, int blockSize) {
  if (!d.pairs || !d.pairCount || !d.rad) {
    return;
  }
  int grid = divUp(n, blockSize);
  detect_collision_pairs<<<grid, blockSize>>>(d.x, d.y, d.z, d.rad, n, d.pairs, d.pairCount, d.maxPairs);
  CUDA_CHECK_LAST("detect_collision_pairs");
}

void launchResolveCollisionPairs(const DeviceArrays &d, int pairCount, float restitution, bool debug) {
  if (!d.pairs || !d.rad) {
    return;
  }
  resolve_collision_pairs<<<1, 1>>>(d.x, d.y, d.z, d.vx, d.vy, d.vz, d.m, d.rad,
                                    d.pairs, pairCount, restitution, d.debug, debug ? 1 : 0);
  CUDA_CHECK_LAST("resolve_collision_pairs");
}

void launchComputeEnergy(const DeviceArrays &d, int n, float G, float eps2, int blockSize) {
  if (!d.energy) {
    return;
  }
  int grid = divUp(n, blockSize);
  size_t shmem = 4 * blockSize * sizeof(float);
  compute_energy_tiled<<<grid, blockSize, shmem>>>(d.x, d.y, d.z, d.vx, d.vy, d.vz, d.m, d.energy, n, G, eps2);
  CUDA_CHECK_LAST("compute_energy_tiled");
}

void launchComputeMinDist(const DeviceArrays &d, int n, int blockSize) {
  if (!d.minDist || !d.minPartner || !d.minBody) {
    return;
  }
  int grid = divUp(n, blockSize);
  compute_min_dist<<<grid, blockSize>>>(d.x, d.y, d.z, d.minDist, d.minPartner, d.minBody, n);
  CUDA_CHECK_LAST("compute_min_dist");
}

namespace {
double reduceDouble(double *in, double *scratch, int n, int blockSize) {
  double *input = in;
  double *output = scratch;
  int current = n;
  size_t shmem = blockSize * sizeof(double);
  while (current > 1) {
    int grid = divUp(current, blockSize);
    reduce_sum_double<<<grid, blockSize, shmem>>>(input, output, current);
    CUDA_CHECK_LAST("reduce_sum_double");
    current = grid;
    std::swap(input, output);
  }
  double result = 0.0;
  CUDA_CHECK(cudaMemcpy(&result, input, sizeof(double), cudaMemcpyDeviceToHost));
  return result;
}
} // namespace

void computeGlobalMinPair(const DeviceArrays &d, int n, int blockSize,
                          float *outMinDist, int *outMinBody, int *outMinPartner) {
  if (!d.minDist || !d.minBody || !d.minPartner || !d.tmpDist || !d.tmpIdx || n <= 0) {
    if (outMinDist) *outMinDist = 0.0f;
    if (outMinBody) *outMinBody = -1;
    if (outMinPartner) *outMinPartner = -1;
    return;
  }

  const float *distA = d.minDist;
  const int *idxA = d.minBody;
  float *distB = d.tmpDist;
  int *idxB = d.tmpIdx;
  int currentN = n;

  while (currentN > 1) {
    int grid = divUp(currentN, blockSize);
    size_t shmem = sizeof(float) * blockSize + sizeof(int) * blockSize;
    reduce_min_pair<<<grid, blockSize, shmem>>>(distA, idxA, distB, idxB, currentN);
    CUDA_CHECK_LAST("reduce_min_pair");
    currentN = grid;
    const float *tmpDist = distA;
    distA = distB;
    distB = const_cast<float *>(tmpDist);
    const int *tmpIdx = idxA;
    idxA = idxB;
    idxB = const_cast<int *>(tmpIdx);
  }

  int minBody = -1;
  float minDist = 0.0f;
  CUDA_CHECK(cudaMemcpy(&minDist, distA, sizeof(float), cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(&minBody, idxA, sizeof(int), cudaMemcpyDeviceToHost));
  int minPartner = -1;
  if (minBody >= 0) {
    CUDA_CHECK(cudaMemcpy(&minPartner, d.minPartner + minBody, sizeof(int), cudaMemcpyDeviceToHost));
  }
  if (outMinDist) *outMinDist = minDist;
  if (outMinBody) *outMinBody = minBody;
  if (outMinPartner) *outMinPartner = minPartner;
}

void launchResetCollisionDebug(const DeviceArrays &d) {
  if (!d.debug) {
    return;
  }
  reset_collision_debug<<<1, 1>>>(d.debug);
  CUDA_CHECK_LAST("reset_collision_debug");
}

bool computeTotalEnergy(const DeviceArrays &d, int n, float G, float eps2, int blockSize,
                        double *outK, double *outU) {
  if (!d.reduceA || !d.reduceB) {
    return false;
  }
  int grid = divUp(n, blockSize);
  size_t shmem = blockSize * sizeof(double);
  kinetic_blocksum<<<grid, blockSize, shmem>>>(d.vx, d.vy, d.vz, d.m, d.reduceA, n);
  CUDA_CHECK_LAST("kinetic_blocksum");
  double totalK = reduceDouble(d.reduceA, d.reduceB, grid, blockSize);

  size_t shmemPot = 4 * blockSize * sizeof(float) + blockSize * sizeof(double) + sizeof(double);
  potential_blocksum_tiled<<<grid, blockSize, shmemPot>>>(d.x, d.y, d.z, d.m, d.reduceA, n, G, eps2);
  CUDA_CHECK_LAST("potential_blocksum_tiled");
  double totalPotDouble = reduceDouble(d.reduceA, d.reduceB, grid, blockSize);
  double totalU = 0.5 * totalPotDouble;

  if (outK) *outK = totalK;
  if (outU) *outU = totalU;
  return true;
}

bool computeAngularMomentum(const DeviceArrays &d, int n, int blockSize,
                            double *outLx, double *outLy, double *outLz) {
  if (!d.reduceAx || !d.reduceAy || !d.reduceAz || !d.reduceB) {
    return false;
  }
  int grid = divUp(n, blockSize);
  size_t shmem = 3 * blockSize * sizeof(double);
  angmom_blocksum<<<grid, blockSize, shmem>>>(d.x, d.y, d.z, d.vx, d.vy, d.vz, d.m,
                                              d.reduceAx, d.reduceAy, d.reduceAz, n);
  CUDA_CHECK_LAST("angmom_blocksum");

  double lx = reduceDouble(d.reduceAx, d.reduceB, grid, blockSize);
  double ly = reduceDouble(d.reduceAy, d.reduceB, grid, blockSize);
  double lz = reduceDouble(d.reduceAz, d.reduceB, grid, blockSize);
  if (outLx) *outLx = lx;
  if (outLy) *outLy = ly;
  if (outLz) *outLz = lz;
  return true;
}
