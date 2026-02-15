#include "sim.h"

#include "utils.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <iostream>
#include <vector>

void stepSimulation(const DeviceArrays &d, int n, float G, float eps2, float dt,
                    bool collisions, float restitution, bool debugCollision,
                    bool cudaDebug, int blockSize) {
  // Leapfrog / Velocity-Verlet:
  // 1) v(t+dt/2) = v(t) + a(t) * dt/2
  // 2) r(t+dt)   = r(t) + v(t+dt/2) * dt
  // 3) a(t+dt)   = accel(r(t+dt))
  // 4) v(t+dt)   = v(t+dt/2) + a(t+dt) * dt/2
  launchUpdateVelHalf(d, n, dt, blockSize);
  if (cudaDebug) CUDA_CHECK(cudaDeviceSynchronize());
  launchUpdatePos(d, n, dt, blockSize);
  if (cudaDebug) CUDA_CHECK(cudaDeviceSynchronize());
  if (collisions) {
    // Collision resolution is non-Hamiltonian and breaks strict symplectic behavior.
    launchResetPairCount(d);
    if (cudaDebug) CUDA_CHECK(cudaDeviceSynchronize());
    launchDetectCollisionPairs(d, n, blockSize);
    if (cudaDebug) CUDA_CHECK(cudaDeviceSynchronize());
    int pairCount = 0;
    CUDA_CHECK(cudaMemcpy(&pairCount, d.pairCount, sizeof(int), cudaMemcpyDeviceToHost));
    if (pairCount > d.maxPairs) {
      std::cerr << "Warning: collision buffer overflow (" << pairCount
                << " > " << d.maxPairs << "); truncating.\n";
      pairCount = d.maxPairs;
    }
    launchResolveCollisionPairs(d, pairCount, restitution, debugCollision);
    if (cudaDebug) CUDA_CHECK(cudaDeviceSynchronize());
  }
  launchComputeAccel(d, n, G, eps2, blockSize);
  if (cudaDebug) CUDA_CHECK(cudaDeviceSynchronize());
  launchFinalizeVel(d, n, dt, blockSize);
  if (cudaDebug) CUDA_CHECK(cudaDeviceSynchronize());
}

static void resolveElasticHost(HostArrays &h, int i, int j, float restitution) {
  float xi = h.x[i];
  float yi = h.y[i];
  float zi = h.z[i];
  float xj = h.x[j];
  float yj = h.y[j];
  float zj = h.z[j];
  float dx = xj - xi;
  float dy = yj - yi;
  float dz = zj - zi;
  float dist2 = dx * dx + dy * dy + dz * dz;
  float dist = std::sqrt(dist2 + 1e-12f);
  if (dist <= 1e-6f) {
    return;
  }
  float nx = dx / dist;
  float ny = dy / dist;
  float nz = dz / dist;
  float rvx = h.vx[i] - h.vx[j];
  float rvy = h.vy[i] - h.vy[j];
  float rvz = h.vz[i] - h.vz[j];
  float vn = rvx * nx + rvy * ny + rvz * nz;
  if (vn >= 0.0f) {
    return;
  }
  float invMi = 1.0f / h.m[i];
  float invMj = 1.0f / h.m[j];
  float J = -(1.0f + restitution) * vn / (invMi + invMj);
  h.vx[i] += (J * invMi) * nx;
  h.vy[i] += (J * invMi) * ny;
  h.vz[i] += (J * invMi) * nz;
  h.vx[j] -= (J * invMj) * nx;
  h.vy[j] -= (J * invMj) * ny;
  h.vz[j] -= (J * invMj) * nz;
  float penetration = (h.rad[i] + h.rad[j]) - dist;
  if (penetration > 0.0f) {
    float correction = 0.5f * penetration;
    h.x[i] = xi - correction * nx;
    h.y[i] = yi - correction * ny;
    h.z[i] = zi - correction * nz;
    h.x[j] = xj + correction * nx;
    h.y[j] = yj + correction * ny;
    h.z[j] = zj + correction * nz;
  }
}

bool stepSimulationMerge(DeviceArrays &d, HostArrays &h, std::vector<int2> &pairs,
                         const Config &cfg, std::mt19937 &rng, int &debrisActive,
                         float dt, int blockSize, bool debugCollision) {
  bool changed = false;
  launchUpdateVelHalf(d, cfg.n, dt, blockSize);
  if (cfg.cudaDebug) CUDA_CHECK(cudaDeviceSynchronize());
  launchUpdatePos(d, cfg.n, dt, blockSize);
  if (cfg.cudaDebug) CUDA_CHECK(cudaDeviceSynchronize());

  // Collision resolution happens after r(t+dt) and before a(t+dt).
  launchResetPairCount(d);
  if (cfg.cudaDebug) CUDA_CHECK(cudaDeviceSynchronize());
  launchDetectCollisionPairs(d, cfg.n, blockSize);
  if (cfg.cudaDebug) CUDA_CHECK(cudaDeviceSynchronize());

  int pairCount = 0;
  CUDA_CHECK(cudaMemcpy(&pairCount, d.pairCount, sizeof(int), cudaMemcpyDeviceToHost));
  if (pairCount > d.maxPairs) {
    pairCount = d.maxPairs;
  }
  if (cfg.maxCollisionPairs > 0 && pairCount > cfg.maxCollisionPairs) {
    pairCount = cfg.maxCollisionPairs;
  }
  if (pairCount > 0) {
    pairs.resize(pairCount);
    CUDA_CHECK(cudaMemcpy(pairs.data(), d.pairs, pairCount * sizeof(int2), cudaMemcpyDeviceToHost));
  } else {
    pairs.clear();
  }

  CUDA_CHECK(cudaMemcpy(h.x.data(), d.x, cfg.n * sizeof(float), cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(h.y.data(), d.y, cfg.n * sizeof(float), cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(h.z.data(), d.z, cfg.n * sizeof(float), cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(h.vx.data(), d.vx, cfg.n * sizeof(float), cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(h.vy.data(), d.vy, cfg.n * sizeof(float), cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(h.vz.data(), d.vz, cfg.n * sizeof(float), cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(h.m.data(), d.m, cfg.n * sizeof(float), cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(h.rad.data(), d.rad, cfg.n * sizeof(float), cudaMemcpyDeviceToHost));
  if (cfg.cudaDebug) CUDA_CHECK(cudaDeviceSynchronize());

  float sunMass = 0.0f;
  for (int i = 0; i < cfg.n; ++i) {
    if (h.type[i] == 0) {
      sunMass = h.m[i];
      break;
    }
  }
  std::vector<int> freeSlots;
  freeSlots.reserve(cfg.n);
  debrisActive = 0;
  for (int i = 0; i < cfg.n; ++i) {
    if (h.m[i] <= 0.0f) {
      freeSlots.push_back(i);
    } else if (h.type[i] == 4) {
      debrisActive++;
    }
  }
  std::vector<char> used(cfg.n, 0);
  std::uniform_real_distribution<float> spread(-1.0f, 1.0f);

  for (const auto &p : pairs) {
    int ia = p.x;
    int ib = p.y;
    if (ia < 0 || ib < 0 || ia >= cfg.n || ib >= cfg.n) continue;
    if (used[ia] || used[ib]) continue;
    if (h.m[ia] <= 0.0f || h.m[ib] <= 0.0f) continue;

    bool planetA = (h.type[ia] == 1 || h.type[ia] == 0);
    bool planetB = (h.type[ib] == 1 || h.type[ib] == 0);
    bool minorA = (h.type[ia] >= 2);
    bool minorB = (h.type[ib] >= 2);

    if (planetA || planetB) {
      if (cfg.allowPlanetAccretion && (planetA ^ planetB)) {
        int planet = planetA ? ia : ib;
        int other = planetA ? ib : ia;
        if (minorA || minorB) {
          float totalMass = h.m[planet] + h.m[other];
          if (totalMass > 1e-8f) {
            float invM = 1.0f / totalMass;
            float vcomx = (h.m[planet] * h.vx[planet] + h.m[other] * h.vx[other]) * invM;
            float vcomy = (h.m[planet] * h.vy[planet] + h.m[other] * h.vy[other]) * invM;
            float vcomz = (h.m[planet] * h.vz[planet] + h.m[other] * h.vz[other]) * invM;
            float xcom = (h.m[planet] * h.x[planet] + h.m[other] * h.x[other]) * invM;
            float ycom = (h.m[planet] * h.y[planet] + h.m[other] * h.y[other]) * invM;
            float zcom = (h.m[planet] * h.z[planet] + h.m[other] * h.z[other]) * invM;
            h.m[planet] = totalMass;
            h.vx[planet] = vcomx;
            h.vy[planet] = vcomy;
            h.vz[planet] = vcomz;
            h.x[planet] = xcom;
            h.y[planet] = ycom;
            h.z[planet] = zcom;
            h.rad[planet] = cbrtf(h.rad[planet] * h.rad[planet] * h.rad[planet] +
                                  h.rad[other] * h.rad[other] * h.rad[other]);
            h.m[other] = 0.0f;
            h.rad[other] = 0.0f;
            h.vrad[other] = 0.0f;
            h.vx[other] = h.vy[other] = h.vz[other] = 0.0f;
            used[planet] = 1;
            used[other] = 1;
            changed = true;
            continue;
          }
        }
      }
      resolveElasticHost(h, ia, ib, cfg.collisionRestitution);
      changed = true;
      used[ia] = used[ib] = 1;
      continue;
    }

    float totalMass = h.m[ia] + h.m[ib];
    if (totalMass <= 1e-8f) continue;
    float invM = 1.0f / totalMass;
    float vcomx = (h.m[ia] * h.vx[ia] + h.m[ib] * h.vx[ib]) * invM;
    float vcomy = (h.m[ia] * h.vy[ia] + h.m[ib] * h.vy[ib]) * invM;
    float vcomz = (h.m[ia] * h.vz[ia] + h.m[ib] * h.vz[ib]) * invM;
    float xcom = (h.m[ia] * h.x[ia] + h.m[ib] * h.x[ib]) * invM;
    float ycom = (h.m[ia] * h.y[ia] + h.m[ib] * h.y[ib]) * invM;
    float zcom = (h.m[ia] * h.z[ia] + h.m[ib] * h.z[ib]) * invM;

    float rvx = h.vx[ia] - h.vx[ib];
    float rvy = h.vy[ia] - h.vy[ib];
    float rvz = h.vz[ia] - h.vz[ib];
    float vrel2 = rvx * rvx + rvy * rvy + rvz * rvz;
    float mu = (h.m[ia] * h.m[ib]) * invM;
    float energy = 0.5f * mu * vrel2;

    bool doFragment = cfg.fragmentation && energy > cfg.fragmentEnergyThreshold;
    if (doFragment) {
      if (debrisActive >= cfg.maxDebrisActive) {
        doFragment = false;
      }
    }
    if (doFragment && freeSlots.size() < static_cast<size_t>(cfg.fragmentsPerCollision)) {
      doFragment = false;
    }

    if (doFragment) {
      int k = std::min(cfg.fragmentsPerCollision, static_cast<int>(freeSlots.size()));
      int allowed = cfg.maxDebrisActive > 0 ? (cfg.maxDebrisActive - debrisActive) : k;
      k = std::min(k, allowed);
      if (k < 2) {
        doFragment = false;
      } else {
        float r = std::sqrt(xcom * xcom + ycom * ycom + zcom * zcom);
        float localOrb = (r > 1e-6f && sunMass > 0.0f) ? std::sqrt(cfg.G * sunMass / r) : 0.0f;
        float spreadScale = std::max(1e-3f, std::sqrt(vrel2));
        float maxSpread = cfg.fragmentSpeedClamp * std::max(1e-3f, localOrb);
        if (maxSpread > 0.0f) {
          spreadScale = std::min(spreadScale, maxSpread);
        }
        std::vector<std::array<float, 3>> jitter(k);
        float avgx = 0.0f, avgy = 0.0f, avgz = 0.0f;
        for (int i = 0; i < k; ++i) {
          float rx = spread(rng);
          float ry = spread(rng);
          float rz = spread(rng);
          jitter[i] = {rx, ry, rz};
          avgx += rx;
          avgy += ry;
          avgz += rz;
        }
        avgx /= static_cast<float>(k);
        avgy /= static_cast<float>(k);
        avgz /= static_cast<float>(k);
        float fragMass = totalMass / static_cast<float>(k);
        for (int i = 0; i < k; ++i) {
          int slot = freeSlots.back();
          freeSlots.pop_back();
          float jx = jitter[i][0] - avgx;
          float jy = jitter[i][1] - avgy;
          float jz = jitter[i][2] - avgz;
          h.x[slot] = xcom + jx * cfg.eps * 0.5f;
          h.y[slot] = ycom + jy * cfg.eps * 0.5f;
          h.z[slot] = zcom + jz * cfg.eps * 0.5f;
          h.vx[slot] = vcomx + jx * spreadScale;
          h.vy[slot] = vcomy + jy * spreadScale;
          h.vz[slot] = vcomz + jz * spreadScale;
          h.m[slot] = fragMass;
          h.rad[slot] = 0.02f;
          h.vrad[slot] = 0.03f;
          h.type[slot] = 4;
        }
        debrisActive += k;
        h.m[ia] = 0.0f;
        h.m[ib] = 0.0f;
        h.rad[ia] = h.rad[ib] = 0.0f;
        h.vrad[ia] = h.vrad[ib] = 0.0f;
        h.vx[ia] = h.vy[ia] = h.vz[ia] = 0.0f;
        h.vx[ib] = h.vy[ib] = h.vz[ib] = 0.0f;
        used[ia] = used[ib] = 1;
        changed = true;
        continue;
      }
    }

    float radMerged = cbrtf(h.rad[ia] * h.rad[ia] * h.rad[ia] +
                            h.rad[ib] * h.rad[ib] * h.rad[ib]);
    h.x[ia] = xcom;
    h.y[ia] = ycom;
    h.z[ia] = zcom;
    h.vx[ia] = vcomx;
    h.vy[ia] = vcomy;
    h.vz[ia] = vcomz;
    h.m[ia] = totalMass;
    h.rad[ia] = radMerged;
    h.type[ia] = (h.type[ia] == 2 || h.type[ib] == 2) ? 2 : 3;
    h.m[ib] = 0.0f;
    h.rad[ib] = 0.0f;
    h.vrad[ib] = 0.0f;
    h.vx[ib] = h.vy[ib] = h.vz[ib] = 0.0f;
    used[ia] = used[ib] = 1;
    changed = true;
  }

  if (changed) {
    CUDA_CHECK(cudaMemcpy(d.x, h.x.data(), cfg.n * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d.y, h.y.data(), cfg.n * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d.z, h.z.data(), cfg.n * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d.vx, h.vx.data(), cfg.n * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d.vy, h.vy.data(), cfg.n * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d.vz, h.vz.data(), cfg.n * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d.m, h.m.data(), cfg.n * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d.rad, h.rad.data(), cfg.n * sizeof(float), cudaMemcpyHostToDevice));
    if (cfg.cudaDebug) CUDA_CHECK(cudaDeviceSynchronize());
  }

  launchComputeAccel(d, cfg.n, cfg.G, cfg.eps * cfg.eps, blockSize);
  if (cfg.cudaDebug) CUDA_CHECK(cudaDeviceSynchronize());
  launchFinalizeVel(d, cfg.n, dt, blockSize);
  if (cfg.cudaDebug) CUDA_CHECK(cudaDeviceSynchronize());
  return changed;
}
