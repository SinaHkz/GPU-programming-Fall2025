#pragma once

#include <cuda_runtime.h>

struct CollisionDebug {
  int valid;
  int i;
  int j;
  float mi;
  float mj;
  float preVi[3];
  float preVj[3];
  float postVi[3];
  float postVj[3];
  float preP[3];
  float postP[3];
};

__global__ void compute_accel_tiled(const float *x, const float *y, const float *z,
                                   const float *m, float *ax, float *ay, float *az,
                                   int n, float G, float eps2);

__global__ void compute_energy_tiled(const float *x, const float *y, const float *z,
                                     const float *vx, const float *vy, const float *vz,
                                     const float *m, float *energy,
                                     int n, float G, float eps2);

__global__ void update_vel_half(float *vx, float *vy, float *vz,
                                const float *ax, const float *ay, const float *az,
                                int n, float dt);

__global__ void update_pos(float *x, float *y, float *z,
                           const float *vx, const float *vy, const float *vz,
                           int n, float dt);

__global__ void finalize_vel(float *vx, float *vy, float *vz,
                             const float *ax, const float *ay, const float *az,
                             int n, float dt);

__global__ void detect_collision_pairs(const float *x, const float *y, const float *z,
                                       const float *rad, int n,
                                       int2 *pairs, int *pairCount, int maxPairs);

__global__ void resolve_collision_pairs(float *x, float *y, float *z,
                                        float *vx, float *vy, float *vz,
                                        const float *m, const float *rad,
                                        const int2 *pairs, int pairCount,
                                        float restitution,
                                        CollisionDebug *debug, int debugEnabled);

__global__ void reset_collision_debug(CollisionDebug *debug);

__global__ void compute_min_dist(const float *x, const float *y, const float *z,
                                 float *minDist, int *minPartner, int *minBody,
                                 int n);

__global__ void reduce_min_pair(const float *inDist, const int *inIdx,
                                float *outDist, int *outIdx, int n);

__global__ void kinetic_blocksum(const float *vx, const float *vy, const float *vz,
                                 const float *m, double *out, int n);

__global__ void potential_blocksum_tiled(const float *x, const float *y, const float *z,
                                         const float *m, double *out,
                                         int n, float G, float eps2);

__global__ void angmom_blocksum(const float *x, const float *y, const float *z,
                                const float *vx, const float *vy, const float *vz,
                                const float *m, double *outX, double *outY, double *outZ,
                                int n);

__global__ void reduce_sum_double(const double *in, double *out, int n);
