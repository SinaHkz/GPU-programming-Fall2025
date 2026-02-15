#include "kernels.h"

#include <cuda_runtime.h>
#include <stdint.h>

__global__ void compute_energy_tiled(const float *x, const float *y, const float *z,
                                     const float *vx, const float *vy, const float *vz,
                                     const float *m, float *energy,
                                     int n, float G, float eps2) {
  extern __shared__ float sh[];
  float *sx = sh;
  float *sy = sh + blockDim.x;
  float *sz = sh + 2 * blockDim.x;
  float *sm = sh + 3 * blockDim.x;

  int i = blockIdx.x * blockDim.x + threadIdx.x;
  float xi = 0.0f;
  float yi = 0.0f;
  float zi = 0.0f;
  float vxi = 0.0f;
  float vyi = 0.0f;
  float vzi = 0.0f;
  float mi = 0.0f;
  if (i < n) {
    xi = x[i];
    yi = y[i];
    zi = z[i];
    vxi = vx[i];
    vyi = vy[i];
    vzi = vz[i];
    mi = m[i];
  }

  float pot = 0.0f;
  int tiles = (n + blockDim.x - 1) / blockDim.x;
  for (int t = 0; t < tiles; ++t) {
    int j = t * blockDim.x + threadIdx.x;
    if (j < n) {
      sx[threadIdx.x] = x[j];
      sy[threadIdx.x] = y[j];
      sz[threadIdx.x] = z[j];
      sm[threadIdx.x] = m[j];
    } else {
      sx[threadIdx.x] = 0.0f;
      sy[threadIdx.x] = 0.0f;
      sz[threadIdx.x] = 0.0f;
      sm[threadIdx.x] = 0.0f;
    }
    __syncthreads();

    if (i < n) {
      int limit = blockDim.x;
      if ((t + 1) * blockDim.x > n) {
        limit = n - t * blockDim.x;
      }
      for (int k = 0; k < limit; ++k) {
        int idx = t * blockDim.x + k;
        if (idx == i) {
          continue;
        }
        float dx = sx[k] - xi;
        float dy = sy[k] - yi;
        float dz = sz[k] - zi;
        float dist2 = dx * dx + dy * dy + dz * dz + eps2;
        float invDist = rsqrtf(dist2);
        pot += -G * mi * sm[k] * invDist;
      }
    }
    __syncthreads();
  }

  if (i < n) {
    float kin = 0.5f * mi * (vxi * vxi + vyi * vyi + vzi * vzi);
    energy[i] = kin + pot;
  }
}

__global__ void compute_accel_tiled(const float *x, const float *y, const float *z,
                                   const float *m, float *ax, float *ay, float *az,
                                   int n, float G, float eps2) {
  extern __shared__ float sh[];
  float *sx = sh;
  float *sy = sh + blockDim.x;
  float *sz = sh + 2 * blockDim.x;
  float *sm = sh + 3 * blockDim.x;

  int i = blockIdx.x * blockDim.x + threadIdx.x;
  float xi = 0.0f;
  float yi = 0.0f;
  float zi = 0.0f;
  if (i < n) {
    xi = x[i];
    yi = y[i];
    zi = z[i];
  }

  float axi = 0.0f;
  float ayi = 0.0f;
  float azi = 0.0f;

  int tiles = (n + blockDim.x - 1) / blockDim.x;
  for (int t = 0; t < tiles; ++t) {
    int j = t * blockDim.x + threadIdx.x;
    if (j < n) {
      sx[threadIdx.x] = x[j];
      sy[threadIdx.x] = y[j];
      sz[threadIdx.x] = z[j];
      sm[threadIdx.x] = m[j];
    } else {
      sx[threadIdx.x] = 0.0f;
      sy[threadIdx.x] = 0.0f;
      sz[threadIdx.x] = 0.0f;
      sm[threadIdx.x] = 0.0f;
    }
    __syncthreads();

    if (i < n) {
      int limit = blockDim.x;
      if ((t + 1) * blockDim.x > n) {
        limit = n - t * blockDim.x;
      }
      for (int k = 0; k < limit; ++k) {
        int idx = t * blockDim.x + k;
        if (idx == i) {
          continue;
        }
        float dx = sx[k] - xi;
        float dy = sy[k] - yi;
        float dz = sz[k] - zi;
        float dist2 = dx * dx + dy * dy + dz * dz + eps2;
        float invDist = rsqrtf(dist2);
        float invDist3 = invDist * invDist * invDist;
        float s = G * sm[k] * invDist3;
        axi += dx * s;
        ayi += dy * s;
        azi += dz * s;
      }
    }
    __syncthreads();
  }

  if (i < n) {
    ax[i] = axi;
    ay[i] = ayi;
    az[i] = azi;
  }
}

__global__ void update_vel_half(float *vx, float *vy, float *vz,
                                const float *ax, const float *ay, const float *az,
                                int n, float dt) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    float half = 0.5f * dt;
    vx[i] += ax[i] * half;
    vy[i] += ay[i] * half;
    vz[i] += az[i] * half;
  }
}

__global__ void update_pos(float *x, float *y, float *z,
                           const float *vx, const float *vy, const float *vz,
                           int n, float dt) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    x[i] += vx[i] * dt;
    y[i] += vy[i] * dt;
    z[i] += vz[i] * dt;
  }
}

__global__ void finalize_vel(float *vx, float *vy, float *vz,
                             const float *ax, const float *ay, const float *az,
                             int n, float dt) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    float half = 0.5f * dt;
    vx[i] += ax[i] * half;
    vy[i] += ay[i] * half;
    vz[i] += az[i] * half;
  }
}

__global__ void detect_collision_pairs(const float *x, const float *y, const float *z,
                                       const float *rad, int n,
                                       int2 *pairs, int *pairCount, int maxPairs) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) {
    return;
  }
  float xi = x[i];
  float yi = y[i];
  float zi = z[i];
  float ri = rad[i];
  for (int j = i + 1; j < n; ++j) {
    float dx = x[j] - xi;
    float dy = y[j] - yi;
    float dz = z[j] - zi;
    float rsum = ri + rad[j];
    if (dx * dx + dy * dy + dz * dz < rsum * rsum) {
      int idx = atomicAdd(pairCount, 1);
      if (idx < maxPairs) {
        pairs[idx] = make_int2(i, j);
      }
    }
  }
}

__global__ void resolve_collision_pairs(float *x, float *y, float *z,
                                        float *vx, float *vy, float *vz,
                                        const float *m, const float *rad,
                                        const int2 *pairs, int pairCount,
                                        float restitution,
                                        CollisionDebug *debug, int debugEnabled) {
  if (blockIdx.x != 0 || threadIdx.x != 0) {
    return;
  }
  for (int p = 0; p < pairCount; ++p) {
    int i = pairs[p].x;
    int j = pairs[p].y;
    if (i < 0 || j < 0) {
      continue;
    }
    float xi = x[i];
    float yi = y[i];
    float zi = z[i];
    float xj = x[j];
    float yj = y[j];
    float zj = z[j];

    float dx = xj - xi;
    float dy = yj - yi;
    float dz = zj - zi;
    float dist2 = dx * dx + dy * dy + dz * dz;
    float dist = sqrtf(dist2 + 1e-12f);
    float rsum = rad[i] + rad[j];
    if (dist >= rsum) {
      continue;
    }
    if (m[i] <= 0.0f || m[j] <= 0.0f) {
      continue;
    }

    float nx = dx / dist;
    float ny = dy / dist;
    float nz = dz / dist;

    float vix = vx[i];
    float viy = vy[i];
    float viz = vz[i];
    float vjx = vx[j];
    float vjy = vy[j];
    float vjz = vz[j];

    float rvx = vix - vjx;
    float rvy = viy - vjy;
    float rvz = viz - vjz;
    float vn = rvx * nx + rvy * ny + rvz * nz;
    if (vn >= 0.0f) {
      continue;
    }

    float invMi = 1.0f / m[i];
    float invMj = 1.0f / m[j];
    float J = -(1.0f + restitution) * vn / (invMi + invMj);

    bool capture = false;
    CollisionDebug local{};
    if (debugEnabled && debug) {
      if (atomicCAS(&debug->valid, 0, 1) == 0) {
        capture = true;
        local.i = i;
        local.j = j;
        local.mi = m[i];
        local.mj = m[j];
        local.preVi[0] = vix; local.preVi[1] = viy; local.preVi[2] = viz;
        local.preVj[0] = vjx; local.preVj[1] = vjy; local.preVj[2] = vjz;
        float prePx = m[i] * vix + m[j] * vjx;
        float prePy = m[i] * viy + m[j] * vjy;
        float prePz = m[i] * viz + m[j] * vjz;
        local.preP[0] = prePx; local.preP[1] = prePy; local.preP[2] = prePz;
      }
    }

    vx[i] = vix + (J * invMi) * nx;
    vy[i] = viy + (J * invMi) * ny;
    vz[i] = viz + (J * invMi) * nz;

    vx[j] = vjx - (J * invMj) * nx;
    vy[j] = vjy - (J * invMj) * ny;
    vz[j] = vjz - (J * invMj) * nz;

    if (capture) {
      float nvix = vx[i];
      float nviy = vy[i];
      float nviz = vz[i];
      float nvjx = vx[j];
      float nvjy = vy[j];
      float nvjz = vz[j];
      local.postVi[0] = nvix; local.postVi[1] = nviy; local.postVi[2] = nviz;
      local.postVj[0] = nvjx; local.postVj[1] = nvjy; local.postVj[2] = nvjz;
      float postPx = m[i] * nvix + m[j] * nvjx;
      float postPy = m[i] * nviy + m[j] * nvjy;
      float postPz = m[i] * nviz + m[j] * nvjz;
      local.postP[0] = postPx; local.postP[1] = postPy; local.postP[2] = postPz;
    }

    float penetration = rsum - dist;
    if (penetration > 0.0f) {
      float correction = 0.5f * penetration;
      x[i] = xi - correction * nx;
      y[i] = yi - correction * ny;
      z[i] = zi - correction * nz;

      x[j] = xj + correction * nx;
      y[j] = yj + correction * ny;
      z[j] = zj + correction * nz;
    }

    if (capture && debug) {
      debug->i = local.i;
      debug->j = local.j;
      debug->mi = local.mi;
      debug->mj = local.mj;
      debug->preVi[0] = local.preVi[0]; debug->preVi[1] = local.preVi[1]; debug->preVi[2] = local.preVi[2];
      debug->preVj[0] = local.preVj[0]; debug->preVj[1] = local.preVj[1]; debug->preVj[2] = local.preVj[2];
      debug->postVi[0] = local.postVi[0]; debug->postVi[1] = local.postVi[1]; debug->postVi[2] = local.postVi[2];
      debug->postVj[0] = local.postVj[0]; debug->postVj[1] = local.postVj[1]; debug->postVj[2] = local.postVj[2];
      debug->preP[0] = local.preP[0]; debug->preP[1] = local.preP[1]; debug->preP[2] = local.preP[2];
      debug->postP[0] = local.postP[0]; debug->postP[1] = local.postP[1]; debug->postP[2] = local.postP[2];
    }
  }
}

__global__ void reset_collision_debug(CollisionDebug *debug) {
  if (threadIdx.x == 0 && blockIdx.x == 0) {
    debug->valid = 0;
  }
}

__global__ void compute_min_dist(const float *x, const float *y, const float *z,
                                 float *minDist, int *minPartner, int *minBody,
                                 int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) {
    return;
  }
  float xi = x[i];
  float yi = y[i];
  float zi = z[i];
  float best = 1e30f;
  int bestIdx = -1;
  for (int j = 0; j < n; ++j) {
    if (j == i) continue;
    float dx = x[j] - xi;
    float dy = y[j] - yi;
    float dz = z[j] - zi;
    float d2 = dx * dx + dy * dy + dz * dz;
    if (d2 < best) {
      best = d2;
      bestIdx = j;
    }
  }
  minDist[i] = best;
  minPartner[i] = bestIdx;
  minBody[i] = i;
}

__global__ void reduce_min_pair(const float *inDist, const int *inIdx,
                                float *outDist, int *outIdx, int n) {
  extern __shared__ float sdist[];
  int *sidx = (int *)(sdist + blockDim.x);

  int tid = threadIdx.x;
  int i = blockIdx.x * blockDim.x + tid;
  float val = 1e30f;
  int idx = -1;
  if (i < n) {
    val = inDist[i];
    idx = inIdx[i];
  }
  sdist[tid] = val;
  sidx[tid] = idx;
  __syncthreads();

  for (int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (tid < s) {
      float other = sdist[tid + s];
      int oidx = sidx[tid + s];
      if (other < sdist[tid]) {
        sdist[tid] = other;
        sidx[tid] = oidx;
      }
    }
    __syncthreads();
  }

  if (tid == 0) {
    outDist[blockIdx.x] = sdist[0];
    outIdx[blockIdx.x] = sidx[0];
  }
}

__global__ void kinetic_blocksum(const float *vx, const float *vy, const float *vz,
                                 const float *m, double *out, int n) {
  extern __shared__ double ssum[];
  int tid = threadIdx.x;
  int i = blockIdx.x * blockDim.x + tid;
  double val = 0.0;
  if (i < n) {
    double vxi = static_cast<double>(vx[i]);
    double vyi = static_cast<double>(vy[i]);
    double vzi = static_cast<double>(vz[i]);
    double mi = static_cast<double>(m[i]);
    val = 0.5 * mi * (vxi * vxi + vyi * vyi + vzi * vzi);
  }
  ssum[tid] = val;
  __syncthreads();
  for (int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (tid < s) {
      ssum[tid] += ssum[tid + s];
    }
    __syncthreads();
  }
  if (tid == 0) {
    out[blockIdx.x] = ssum[0];
  }
}

__global__ void potential_blocksum_tiled(const float *x, const float *y, const float *z,
                                         const float *m, double *out,
                                         int n, float G, float eps2) {
  extern __shared__ unsigned char shmem[];
  float *sx = reinterpret_cast<float *>(shmem);
  float *sy = sx + blockDim.x;
  float *sz = sy + blockDim.x;
  float *sm = sz + blockDim.x;
  uintptr_t base = reinterpret_cast<uintptr_t>(sm + blockDim.x);
  base = (base + sizeof(double) - 1) & ~(sizeof(double) - 1);
  double *ssum = reinterpret_cast<double *>(base);

  int tid = threadIdx.x;
  int i = blockIdx.x * blockDim.x + tid;
  double pot = 0.0;
  float xi = 0.0f;
  float yi = 0.0f;
  float zi = 0.0f;
  float mi = 0.0f;
  if (i < n) {
    xi = x[i];
    yi = y[i];
    zi = z[i];
    mi = m[i];
  }

  int tiles = (n + blockDim.x - 1) / blockDim.x;
  for (int t = 0; t < tiles; ++t) {
    int j = t * blockDim.x + tid;
    if (j < n) {
      sx[tid] = x[j];
      sy[tid] = y[j];
      sz[tid] = z[j];
      sm[tid] = m[j];
    } else {
      sx[tid] = 0.0f;
      sy[tid] = 0.0f;
      sz[tid] = 0.0f;
      sm[tid] = 0.0f;
    }
    __syncthreads();

    if (i < n) {
      int limit = blockDim.x;
      if ((t + 1) * blockDim.x > n) {
        limit = n - t * blockDim.x;
      }
      for (int k = 0; k < limit; ++k) {
        int idx = t * blockDim.x + k;
        if (idx == i) {
          continue;
        }
        float dx = sx[k] - xi;
        float dy = sy[k] - yi;
        float dz = sz[k] - zi;
        float dist2 = dx * dx + dy * dy + dz * dz + eps2;
        float invDist = rsqrtf(dist2);
        pot += -static_cast<double>(G) * static_cast<double>(mi) * static_cast<double>(sm[k]) *
               static_cast<double>(invDist);
      }
    }
    __syncthreads();
  }

  ssum[tid] = pot;
  __syncthreads();
  for (int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (tid < s) {
      ssum[tid] += ssum[tid + s];
    }
    __syncthreads();
  }
  if (tid == 0) {
    out[blockIdx.x] = ssum[0];
  }
}

__global__ void angmom_blocksum(const float *x, const float *y, const float *z,
                                const float *vx, const float *vy, const float *vz,
                                const float *m, double *outX, double *outY, double *outZ,
                                int n) {
  extern __shared__ double sdata[];
  double *sx = sdata;
  double *sy = sdata + blockDim.x;
  double *sz = sdata + 2 * blockDim.x;

  int tid = threadIdx.x;
  int i = blockIdx.x * blockDim.x + tid;
  double lx = 0.0;
  double ly = 0.0;
  double lz = 0.0;
  if (i < n) {
    double xi = static_cast<double>(x[i]);
    double yi = static_cast<double>(y[i]);
    double zi = static_cast<double>(z[i]);
    double vxi = static_cast<double>(vx[i]);
    double vyi = static_cast<double>(vy[i]);
    double vzi = static_cast<double>(vz[i]);
    double mi = static_cast<double>(m[i]);
    double px = mi * vxi;
    double py = mi * vyi;
    double pz = mi * vzi;
    lx = yi * pz - zi * py;
    ly = zi * px - xi * pz;
    lz = xi * py - yi * px;
  }
  sx[tid] = lx;
  sy[tid] = ly;
  sz[tid] = lz;
  __syncthreads();
  for (int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (tid < s) {
      sx[tid] += sx[tid + s];
      sy[tid] += sy[tid + s];
      sz[tid] += sz[tid + s];
    }
    __syncthreads();
  }
  if (tid == 0) {
    outX[blockIdx.x] = sx[0];
    outY[blockIdx.x] = sy[0];
    outZ[blockIdx.x] = sz[0];
  }
}

__global__ void reduce_sum_double(const double *in, double *out, int n) {
  extern __shared__ double ssum[];
  int tid = threadIdx.x;
  int i = blockIdx.x * blockDim.x + tid;
  double val = 0.0;
  if (i < n) {
    val = in[i];
  }
  ssum[tid] = val;
  __syncthreads();
  for (int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (tid < s) {
      ssum[tid] += ssum[tid + s];
    }
    __syncthreads();
  }
  if (tid == 0) {
    out[blockIdx.x] = ssum[0];
  }
}
