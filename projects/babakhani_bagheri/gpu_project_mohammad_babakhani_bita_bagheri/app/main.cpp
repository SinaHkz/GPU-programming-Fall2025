#include "bench.h"
#include "cli.h"
#include "config.h"
#include "integrator.h"
#include "init.h"
#include "render_utils.h"
#include "renderer.h"
#include "sim.h"
#include "utils.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <random>
#include <string>
#include <vector>

namespace {
constexpr float kPi = 3.14159265f;
constexpr float kTwoPi = 2.0f * kPi;
}  // namespace

struct CollisionEvent {
  int a = -1;
  int b = -1;
  float timer = 0.0f;
};

int main(int argc, char **argv) {
  Config cfg;
  if (!parseArgs(argc, argv, cfg)) {
    printUsage();
    return 1;
  }
  if (cfg.showHelp) {
    printUsage();
    return 0;
  }

  const int blockSize = 256;

  if (cfg.benchmark) {
    cfg.render = false;
    cfg.trails = false;
    cfg.trailLength = 0;
    cfg.highlightCloseEncounters = false;
    cfg.showEnergyColor = false;
    cfg.showVelocityVectors = false;
    cfg.autoFocus = false;
    cfg.debugCollision = false;
    cfg.centerOnCOM = false;
    runBenchmark(cfg, blockSize);
    return 0;
  }

  if (cfg.demoCollision) {
    cfg.collisions = true;
    cfg.timeScale *= 0.2f;
    cfg.trails = false;
    cfg.trailLength = 0;
  }

  InitConfig initCfg;
  initCfg.n = cfg.n;
  initCfg.seed = cfg.seed;
  initCfg.G = cfg.G;
  initCfg.massScale = cfg.massScale;
  initCfg.sunMass = cfg.sunMass;
  initCfg.planetMass = cfg.planetMass;
  initCfg.cometMass = cfg.cometMass;
  initCfg.minorMass = cfg.minorMass;
  initCfg.stableInit = cfg.stableInit;
  initCfg.zeroMomentum = cfg.zeroMomentum;
  initCfg.realSolarSystem = cfg.realSolarSystem;
  initCfg.numPlanets = cfg.numPlanets;
  initCfg.planetRadiiPreset = cfg.planetRadiiPreset;
  initCfg.planetEccMax = cfg.planetEccMax;
  initCfg.planetIncDegMax = cfg.planetIncDegMax;
  initCfg.asteroidBelt = cfg.asteroidBelt;
  initCfg.asteroidBeltRmin = cfg.asteroidBeltRmin;
  initCfg.asteroidBeltRmax = cfg.asteroidBeltRmax;
  initCfg.asteroidTangentialJitter = cfg.asteroidTangentialJitter;
  initCfg.asteroidRadialJitter = cfg.asteroidRadialJitter;
  initCfg.beltCount = cfg.beltCount;
  initCfg.minorCount = cfg.minorCount;
  initCfg.hideMinors = cfg.hideMinors;
  initCfg.numComets = cfg.numComets;
  initCfg.cometEccMin = cfg.cometEccMin;
  initCfg.cometEccMax = cfg.cometEccMax;
  initCfg.cometIncDegMax = cfg.cometIncDegMax;
  initCfg.slingshotTargetPlanet = cfg.slingshotTargetPlanet;
  initCfg.fragmentation = cfg.fragmentation;
  initCfg.maxDebris = cfg.maxDebris;
  initCfg.demoCollision = cfg.demoCollision;
  initCfg.demoSlingshot = cfg.demoSlingshot;
  initCfg.demoCollisionSpeed = cfg.demoCollisionSpeed;
  initCfg.fragmentation = cfg.fragmentation;

  HostArrays h;
  float maxRadius = 0.0f;
  initSolarSystem(h, initCfg, maxRadius);

  if (cfg.normalizeOrbits) {
    float sunMass = 0.0f;
    for (int i = 0; i < cfg.n; ++i) {
      if (h.type[i] == 0) {
        sunMass = h.m[i];
        break;
      }
    }
    if (sunMass > 0.0f) {
      for (int i = 0; i < cfg.n; ++i) {
        if (h.type[i] != 1) {
          continue;
        }
        float x = h.x[i];
        float y = h.y[i];
        float r = std::sqrt(x * x + y * y);
        if (r < 1e-6f) {
          continue;
        }
        float vCirc = std::sqrt(cfg.G * sunMass / r) * cfg.orbitSpeedScale;
        float rx = x / r;
        float ry = y / r;
        float tx = -ry;
        float ty = rx;
        float vRad = h.vx[i] * rx + h.vy[i] * ry;
        float vTan = h.vx[i] * tx + h.vy[i] * ty;
        float sign = (vTan >= 0.0f) ? 1.0f : -1.0f;
        h.vx[i] = vRad * rx + sign * vCirc * tx;
        h.vy[i] = vRad * ry + sign * vCirc * ty;
      }
    }
    if (cfg.zeroMomentum) {
      float cX = 0.0f, cY = 0.0f, cZ = 0.0f;
      float vX = 0.0f, vY = 0.0f, vZ = 0.0f;
      computeCOM(h, cX, cY, cZ, vX, vY, vZ);
      for (int i = 0; i < cfg.n; ++i) {
        h.vx[i] -= vX;
        h.vy[i] -= vY;
        h.vz[i] -= vZ;
      }
    }
  }

  int cometIndex = -1;
  int targetPlanetIndex = -1;
  std::vector<int> cometIndices;
  if (cfg.demoSlingshot) {
    float best = 1e30f;
    int planetCounter = 0;
    int desiredPlanet = cfg.slingshotTargetPlanet;
    for (int i = 0; i < cfg.n; ++i) {
      if (h.type[i] == 2) {
        if (cometIndex < 0) {
          cometIndex = i;
        }
        cometIndices.push_back(i);
      } else if (h.type[i] == 1) {
        if (desiredPlanet >= 0 && planetCounter == desiredPlanet) {
          targetPlanetIndex = i;
        }
        planetCounter++;
        if (desiredPlanet < 0) {
          float r = std::sqrt(h.x[i] * h.x[i] + h.y[i] * h.y[i]);
          float d = std::abs(r - 4.5f);
          if (d < best) {
            best = d;
            targetPlanetIndex = i;
          }
        }
      }
    }
    if (cfg.cometTrailWidth <= 1.0f) {
      cfg.cometTrailWidth = 2.0f;
    }
  }
  if (!cfg.demoSlingshot) {
    for (int i = 0; i < cfg.n; ++i) {
      if (h.type[i] == 2) {
        cometIndices.push_back(i);
      }
    }
  }

  bool needEnergy = cfg.showEnergyColor;
  bool needMinDist = cfg.highlightCloseEncounters || cfg.autoFocus;
  bool needDebug = cfg.collisions && cfg.debugCollision;
  bool needTotals = cfg.printTotalEnergy;
  bool needAngMom = cfg.printAngularMomentum;

  DeviceArrays d;
  allocateDeviceArrays(d, cfg.n, cfg.collisions, needEnergy, needMinDist, needDebug,
                       needTotals, needAngMom);
  CUDA_CHECK(cudaMemcpy(d.x, h.x.data(), cfg.n * sizeof(float), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d.y, h.y.data(), cfg.n * sizeof(float), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d.z, h.z.data(), cfg.n * sizeof(float), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d.vx, h.vx.data(), cfg.n * sizeof(float), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d.vy, h.vy.data(), cfg.n * sizeof(float), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d.vz, h.vz.data(), cfg.n * sizeof(float), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d.m, h.m.data(), cfg.n * sizeof(float), cudaMemcpyHostToDevice));
  if (cfg.collisions) {
    CUDA_CHECK(cudaMemcpy(d.rad, h.rad.data(), cfg.n * sizeof(float), cudaMemcpyHostToDevice));
  }
  if (needDebug) {
    launchResetCollisionDebug(d);
  }

  float eps2 = cfg.eps * cfg.eps;
  launchComputeAccel(d, cfg.n, cfg.G, eps2, blockSize);

  const int energyPrintInterval = kEnergyPrintInterval;
  double energy0 = 0.0;
  bool energy0Set = false;
  if (needTotals) {
    double totalK = 0.0;
    double totalU = 0.0;
    if (computeTotalEnergy(d, cfg.n, cfg.G, eps2, blockSize, &totalK, &totalU)) {
      energy0 = totalK + totalU;
      energy0Set = true;
      std::cout << "TotalEnergy step=0 K=" << totalK << " U=" << totalU
                << " E=" << energy0 << " drift=0\n";
    }
  }
  if (needAngMom) {
    double lx = 0.0, ly = 0.0, lz = 0.0;
    if (computeAngularMomentum(d, cfg.n, blockSize, &lx, &ly, &lz)) {
      std::cout << "AngularMomentum step=0 L=(" << lx << "," << ly << "," << lz << ")\n";
    }
  }

  if (!cfg.render) {
    int steps = cfg.steps > 0 ? cfg.steps : 1000;
    for (int i = 0; i < steps; ++i) {
      float effectiveDt = cfg.dt * cfg.timeScale;
      if (needDebug) {
        launchResetCollisionDebug(d);
      }
      stepSimulation(d, cfg.n, cfg.G, eps2, effectiveDt, cfg.collisions,
                     cfg.collisionRestitution, needDebug, cfg.cudaDebug, blockSize);
      int stepIndex = i + 1;
      if (needTotals && (stepIndex % energyPrintInterval == 0)) {
        double totalK = 0.0;
        double totalU = 0.0;
        if (computeTotalEnergy(d, cfg.n, cfg.G, eps2, blockSize, &totalK, &totalU)) {
          double totalE = totalK + totalU;
          double drift = energy0Set && std::abs(energy0) > 0.0 ? std::abs(totalE - energy0) / std::abs(energy0) : 0.0;
          std::cout << "TotalEnergy step=" << stepIndex
                    << " K=" << totalK << " U=" << totalU
                    << " E=" << totalE << " drift=" << drift << "\n";
        }
      }
      if (needAngMom && (stepIndex % energyPrintInterval == 0)) {
        double lx = 0.0, ly = 0.0, lz = 0.0;
        if (computeAngularMomentum(d, cfg.n, blockSize, &lx, &ly, &lz)) {
          std::cout << "AngularMomentum step=" << stepIndex
                    << " L=(" << lx << "," << ly << "," << lz << ")\n";
        }
      }
      if (cfg.printCOM && (stepIndex % energyPrintInterval == 0)) {
        CUDA_CHECK(cudaMemcpy(h.x.data(), d.x, cfg.n * sizeof(float), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(h.y.data(), d.y, cfg.n * sizeof(float), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(h.z.data(), d.z, cfg.n * sizeof(float), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(h.vx.data(), d.vx, cfg.n * sizeof(float), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(h.vy.data(), d.vy, cfg.n * sizeof(float), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(h.vz.data(), d.vz, cfg.n * sizeof(float), cudaMemcpyDeviceToHost));
        float cX = 0.0f, cY = 0.0f, cZ = 0.0f;
        float vX = 0.0f, vY = 0.0f, vZ = 0.0f;
        computeCOM(h, cX, cY, cZ, vX, vY, vZ);
        float vmag = std::sqrt(vX * vX + vY * vY + vZ * vZ);
        std::cout << "COM step=" << stepIndex
                  << " R=(" << cX << "," << cY << "," << cZ << ")"
                  << " V=(" << vX << "," << vY << "," << vZ << ")"
                  << " |V|=" << vmag << "\n";
      }
    }
    freeDeviceArrays(d);
    return 0;
  }

  std::vector<float> baseColors;
  fillColors(h, baseColors);
  std::vector<float> baseSizes;
  float comX0 = 0.0f;
  float comY0 = 0.0f;
  float comZ0 = 0.0f;
  float vcomX0 = 0.0f;
  float vcomY0 = 0.0f;
  float vcomZ0 = 0.0f;
  computeCOM(h, comX0, comY0, comZ0, vcomX0, vcomY0, vcomZ0);

  float rMaxSimAll = 0.0f;
  float rMaxSimMajor = 0.0f;
  for (int i = 0; i < cfg.n; ++i) {
    float dx = h.x[i] - comX0;
    float dy = h.y[i] - comY0;
    float dz = h.z[i] - comZ0;
    float r = std::sqrt(dx * dx + dy * dy + dz * dz);
    rMaxSimAll = std::max(rMaxSimAll, r);
    if (h.type[i] == 0 || h.type[i] == 1 || h.type[i] == 2) {
      rMaxSimMajor = std::max(rMaxSimMajor, r);
    }
  }
  float rMaxSim = cfg.autoFitMajorOnly ? rMaxSimMajor : rMaxSimAll;
  if (rMaxSim <= 1e-6f) {
    rMaxSim = rMaxSimAll;
  }

  if (cfg.autoFit && !cfg.worldScaleSet) {
    if (rMaxSim > 1e-6f) {
      cfg.worldScale = 1.0f / rMaxSim;
    } else {
      cfg.worldScale = 1.0f;
    }
  }
  if (cfg.worldScale <= 0.0f) {
    cfg.worldScale = 1.0f;
  }

  int trailLen = (cfg.trails && cfg.trailLength > 0) ? cfg.trailLength : 0;

  float rMaxRender = rMaxSim * cfg.worldScale;
  float viewRadius = cfg.autoFit ? (rMaxRender / 0.8f) : (maxRadius * 1.3f * cfg.worldScale);
  if (viewRadius < 0.01f) {
    viewRadius = 0.01f;
  }
  const int initWidth = 1280;
  const int initHeight = 720;
  float pointScale = cfg.pointSizeScale * cfg.visualScale;
  float minZoom = cfg.minZoomSet ? cfg.minZoom : std::max(0.2f * viewRadius, 0.01f);
  float maxZoom = cfg.maxZoomSet ? cfg.maxZoom : std::max(5.0f * viewRadius, minZoom + 1.0f);
  float initialZoom = std::clamp(viewRadius, minZoom, maxZoom);
  float initialPixelScale = 0.5f * static_cast<float>(initHeight) * (initialZoom / viewRadius);
  computeRenderSizes(h, cfg, cfg.worldScale, initialPixelScale, pointScale, baseSizes);

  Renderer renderer;
  if (!renderer.init(initWidth, initHeight, "CUDA N-Body", cfg.n, baseColors, baseSizes, trailLen,
                     viewRadius, pointScale)) {
    std::fprintf(stderr, "Renderer initialization failed\n");
    freeDeviceArrays(d);
    return 1;
  }
  renderer.setZoomLimits(minZoom, maxZoom);
  renderer.setCamera(0.0f, 0.0f, 0.0f, initialZoom, viewRadius);
  const float resetPanX = 0.0f;
  const float resetPanY = 0.0f;
  const float resetPanZ = 0.0f;
  const float resetZoom = initialZoom;
  const float resetViewRadius = viewRadius;
  const float resetViewRadiusClamped = std::clamp(resetViewRadius, cfg.minViewRadius, cfg.maxViewRadius);
  renderer.setTrailsEnabled(cfg.trails);
  renderer.setTrailAlpha(cfg.trailAlpha);
  renderer.setVelocityLineAlpha(cfg.velocityLineAlpha);
  renderer.setCloseLineAlpha(cfg.closeLineAlpha);
  renderer.setAuxLineAlpha(cfg.cometTailAlpha);
  renderer.setAdditiveBlend(cfg.cinematic || cfg.bloomBoost > 0.0f);
  renderer.setBloom(cfg.bloomBoost, cfg.bloomThreshold);
  renderer.setPointSizeScale(pointScale);
  float impostorPx = std::max(cfg.minPlanetPixels, cfg.minCometPixels + 2.0f);
  renderer.setImpostorThreshold(impostorPx / std::max(pointScale, 1e-6f));
  renderer.setPlanetStyle(cfg.planetStyle);
  renderer.setGlDebug(cfg.glDebug);
  if (cfg.demoSlingshot && cometIndex >= 0) {
    renderer.setCometTrail(cometIndex, cfg.cometTrailWidth);
  } else {
    renderer.setCometTrail(-1, 1.0f);
  }

  std::vector<float> positions;
  std::vector<float> velocities;
  std::vector<float> energies;
  std::vector<int> closeFlags;
  std::vector<int> closePartners;
  std::vector<float> dynamicColors;
  std::vector<float> velocityLines;
  std::vector<float> closeLines;
  std::vector<float> cometTailLines;
  std::vector<float> collisionFlash;
  std::vector<float> dynamicSizes;
  std::vector<int2> collisionPairs;
  std::vector<CollisionEvent> collisionEvents;
  int lastCollisionA = -1;
  int lastCollisionB = -1;
  float lastCollisionFocusTimer = 0.0f;
  float collisionFocusCooldown = 0.0f;
  float lastShockTimer = 0.0f;
  int debrisSpawned = 0;
  std::mt19937 fragRng(cfg.seed + 1337u);
  float collisionHoldTimer = 0.0f;
  float collisionSlowTimer = 0.0f;
  bool sizesDirty = false;
  float lastPixelScale = -1.0f;
  int sanityFrame = 0;

  float renderComX0 = cfg.centerOnCOM ? comX0 : 0.0f;
  float renderComY0 = cfg.centerOnCOM ? comY0 : 0.0f;
  float renderComZ0 = cfg.centerOnCOM ? comZ0 : 0.0f;
  interleavePositions(h, renderComX0, renderComY0, renderComZ0, cfg.worldScale, positions);
  renderer.updatePositions(positions);

  std::vector<float> trailBuffer;
  std::vector<float> trailRender;
  int trailHead = 0;
  int trailCount = 0;
  if (trailLen > 0) {
    trailBuffer.resize(trailLen * cfg.n * 3, 0.0f);
    trailRender.resize(trailLen * cfg.n * 3, 0.0f);
  }

  float timeScale = cfg.timeScale;
  bool paused = false;
  int selectedIndex = 0;
  float focusThreshold = (cfg.focusThreshold > 0.0f) ? cfg.focusThreshold : (3.0f * cfg.eps);
  float slingshotMinDist = 1e30f;
  float slingshotSpeedBefore = 0.0f;
  float slingshotSpeedAfter = 0.0f;
  bool slingshotArmed = false;
  bool slingshotReported = false;
  float simTime = 0.0f;

  int step = 0;
  while (!renderer.shouldClose()) {
    bool pairsReady = false;
    if (cfg.steps > 0 && step >= cfg.steps) {
      break;
    }

    renderer.pollEvents();
    renderer.updateInput();
    RenderActions actions = renderer.consumeActions();
    if (actions.pauseToggle) {
      paused = !paused;
    }
    if (actions.timeScaleDelta > 0.0f) {
      timeScale *= 1.1f;
    } else if (actions.timeScaleDelta < 0.0f) {
      timeScale *= 0.9f;
      if (timeScale < 1e-4f) {
        timeScale = 1e-4f;
      }
    }
    if (actions.selectDelta != 0) {
      selectedIndex = (selectedIndex + actions.selectDelta) % cfg.n;
      if (selectedIndex < 0) {
        selectedIndex += cfg.n;
      }
    }
    if (actions.resetCamera) {
      renderer.setCamera(resetPanX, resetPanY, resetPanZ, resetZoom, resetViewRadiusClamped);
    }

    if (!paused) {
      float baseDt = cfg.dt * timeScale;
      if (cfg.collisions && cfg.slowOnCollision && collisionHoldTimer > 0.0f) {
        collisionHoldTimer = std::max(0.0f, collisionHoldTimer - baseDt);
        simTime += baseDt;
      } else {
        float effectiveDt = baseDt;
        if (cfg.collisions && cfg.slowOnCollision && collisionSlowTimer > 0.0f) {
          effectiveDt = baseDt * std::max(0.0f, cfg.collisionSlowFactor);
          collisionSlowTimer = std::max(0.0f, collisionSlowTimer - baseDt);
        }
        if (needDebug) {
          launchResetCollisionDebug(d);
        }
        if (cfg.collisions && cfg.collisionModel == 0) {
          bool changed = stepSimulationMerge(d, h, collisionPairs, cfg, fragRng, debrisSpawned,
                                             effectiveDt, blockSize, needDebug);
          pairsReady = true;
          if (changed) {
            sizesDirty = true;
          }
        } else {
          stepSimulation(d, cfg.n, cfg.G, eps2, effectiveDt, cfg.collisions,
                         cfg.collisionRestitution, needDebug, cfg.cudaDebug, blockSize);
        }
        simTime += effectiveDt;
      }
    }

    if (needTotals && (step % energyPrintInterval == 0)) {
      double totalK = 0.0;
      double totalU = 0.0;
      if (computeTotalEnergy(d, cfg.n, cfg.G, eps2, blockSize, &totalK, &totalU)) {
        double totalE = totalK + totalU;
        if (!energy0Set) {
          energy0 = totalE;
          energy0Set = true;
        }
        double drift = std::abs(energy0) > 0.0 ? std::abs(totalE - energy0) / std::abs(energy0) : 0.0;
        std::cout << "TotalEnergy step=" << step
                  << " K=" << totalK << " U=" << totalU
                  << " E=" << totalE << " drift=" << drift << "\n";
      }
    }
    if (needAngMom && (step % energyPrintInterval == 0)) {
      double lx = 0.0, ly = 0.0, lz = 0.0;
      if (computeAngularMomentum(d, cfg.n, blockSize, &lx, &ly, &lz)) {
        std::cout << "AngularMomentum step=" << step
                  << " L=(" << lx << "," << ly << "," << lz << ")\n";
      }
    }

    CUDA_CHECK(cudaMemcpy(h.x.data(), d.x, cfg.n * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h.y.data(), d.y, cfg.n * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h.z.data(), d.z, cfg.n * sizeof(float), cudaMemcpyDeviceToHost));
    if (cfg.cudaDebug) CUDA_CHECK(cudaDeviceSynchronize());

    float comX = 0.0f;
    float comY = 0.0f;
    float comZ = 0.0f;
    float vcomX = 0.0f;
    float vcomY = 0.0f;
    float vcomZ = 0.0f;
    computeCOM(h, comX, comY, comZ, vcomX, vcomY, vcomZ);
    if (!cfg.centerOnCOM) {
      comX = 0.0f;
      comY = 0.0f;
      comZ = 0.0f;
    }

    interleavePositions(h, comX, comY, comZ, cfg.worldScale, positions);
    renderer.updatePositions(positions);

    bool needVelCopy = cfg.showVelocityVectors || cfg.showEnergyColor || cfg.demoSlingshot || cfg.printCOM || cfg.cometTail;
    if (needVelCopy) {
      velocities.resize(cfg.n * 3);
      CUDA_CHECK(cudaMemcpy(h.vx.data(), d.vx, cfg.n * sizeof(float), cudaMemcpyDeviceToHost));
      CUDA_CHECK(cudaMemcpy(h.vy.data(), d.vy, cfg.n * sizeof(float), cudaMemcpyDeviceToHost));
      CUDA_CHECK(cudaMemcpy(h.vz.data(), d.vz, cfg.n * sizeof(float), cudaMemcpyDeviceToHost));
      if (cfg.cudaDebug) CUDA_CHECK(cudaDeviceSynchronize());
      for (int i = 0; i < cfg.n; ++i) {
        velocities[3 * i + 0] = h.vx[i];
        velocities[3 * i + 1] = h.vy[i];
        velocities[3 * i + 2] = h.vz[i];
      }
    }

    if (cfg.sanityChecks) {
      sanityFrame++;
      bool doPrint = (sanityFrame % 60 == 0);
      int countNaN = 0;
      float maxAbsPos = 0.0f;
      float maxAbsVel = 0.0f;
      std::array<int, 5> badIdx{};
      int badFilled = 0;
      for (int i = 0; i < cfg.n; ++i) {
        float xi = h.x[i];
        float yi = h.y[i];
        float zi = h.z[i];
        float ax = std::max(std::abs(xi), std::max(std::abs(yi), std::abs(zi)));
        if (!std::isfinite(xi) || !std::isfinite(yi) || !std::isfinite(zi)) {
          countNaN++;
          if (badFilled < 5) badIdx[badFilled++] = i;
        }
        if (ax > maxAbsPos) maxAbsPos = ax;
      }
      if (!velocities.empty()) {
        for (int i = 0; i < cfg.n; ++i) {
          float vx = velocities[3 * i + 0];
          float vy = velocities[3 * i + 1];
          float vz = velocities[3 * i + 2];
          float av = std::max(std::abs(vx), std::max(std::abs(vy), std::abs(vz)));
          if (av > maxAbsVel) maxAbsVel = av;
        }
      }
      if (doPrint && (cfg.solarCollisionDemo || cfg.solarCollision)) {
        std::cout << "Sanity frame=" << sanityFrame
                  << " NaN=" << countNaN
                  << " maxAbsPos=" << maxAbsPos
                  << " maxAbsVel=" << maxAbsVel << "\n";
      }
      if (countNaN > 0 || maxAbsPos > cfg.maxAbsPosClamp) {
        std::cerr << "Sanity check failure: NaN=" << countNaN
                  << " maxAbsPos=" << maxAbsPos << " clamp=" << cfg.maxAbsPosClamp << "\n";
        for (int k = 0; k < badFilled; ++k) {
          int idx = badIdx[k];
          std::cerr << "  bad[" << idx << "] = (" << h.x[idx] << "," << h.y[idx]
                    << "," << h.z[idx] << ")\n";
        }
        for (int i = 0; i < cfg.n; ++i) {
          float xi = h.x[i];
          float yi = h.y[i];
          float zi = h.z[i];
          if (!std::isfinite(xi) || !std::isfinite(yi) || !std::isfinite(zi)) {
            h.x[i] = h.y[i] = h.z[i] = 0.0f;
            h.vx[i] = h.vy[i] = h.vz[i] = 0.0f;
            h.m[i] = 0.0f;
            h.rad[i] = 0.0f;
          }
        }
        CUDA_CHECK(cudaMemcpy(d.x, h.x.data(), cfg.n * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d.y, h.y.data(), cfg.n * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d.z, h.z.data(), cfg.n * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d.vx, h.vx.data(), cfg.n * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d.vy, h.vy.data(), cfg.n * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d.vz, h.vz.data(), cfg.n * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d.m, h.m.data(), cfg.n * sizeof(float), cudaMemcpyHostToDevice));
        if (cfg.collisions) {
          CUDA_CHECK(cudaMemcpy(d.rad, h.rad.data(), cfg.n * sizeof(float), cudaMemcpyHostToDevice));
        }
        if (cfg.cudaDebug) CUDA_CHECK(cudaDeviceSynchronize());
        cfg.collisions = false;
        paused = true;
        sizesDirty = true;
      }
    }

    if (cfg.printCOM && (step % energyPrintInterval == 0)) {
      float cX = 0.0f, cY = 0.0f, cZ = 0.0f;
      float vX = 0.0f, vY = 0.0f, vZ = 0.0f;
      computeCOM(h, cX, cY, cZ, vX, vY, vZ);
      float vmag = std::sqrt(vX * vX + vY * vY + vZ * vZ);
      std::cout << "COM step=" << step
                << " R=(" << cX << "," << cY << "," << cZ << ")"
                << " V=(" << vX << "," << vY << "," << vZ << ")"
                << " |V|=" << vmag << "\n";
    }

    if (cfg.demoSlingshot && cometIndex >= 0 && targetPlanetIndex >= 0 && !velocities.empty()) {
      float dx = h.x[cometIndex] - h.x[targetPlanetIndex];
      float dy = h.y[cometIndex] - h.y[targetPlanetIndex];
      float dz = h.z[cometIndex] - h.z[targetPlanetIndex];
      float dist = std::sqrt(dx * dx + dy * dy + dz * dz);
      float vx = velocities[3 * cometIndex + 0];
      float vy = velocities[3 * cometIndex + 1];
      float vz = velocities[3 * cometIndex + 2];
      float speed = std::sqrt(vx * vx + vy * vy + vz * vz);
      if (dist < slingshotMinDist) {
        slingshotMinDist = dist;
      }
      if (!slingshotArmed && dist < 6.0f) {
        slingshotSpeedBefore = speed;
        slingshotArmed = true;
      }
      if (slingshotArmed && !slingshotReported && dist > slingshotMinDist * 1.5f) {
        slingshotSpeedAfter = speed;
        slingshotReported = true;
        std::cout << "Slingshot: minDist=" << slingshotMinDist
                  << " speedBefore=" << slingshotSpeedBefore
                  << " speedAfter=" << slingshotSpeedAfter << "\n";
      }
      float highlightDist = std::max(1.0f, 5.0f * cfg.eps);
      if (cfg.slingshotHighlight && dist < highlightDist) {
        if (collisionFlash.size() != static_cast<size_t>(cfg.n)) {
          collisionFlash.assign(cfg.n, 0.0f);
        }
        float flashDuration = std::max(0.1f, cfg.collisionFlashDuration);
        collisionFlash[cometIndex] = flashDuration;
        collisionFlash[targetPlanetIndex] = flashDuration;
        renderer.setCometTrail(cometIndex, cfg.cometTrailWidth * 2.0f);
      } else if (cfg.demoSlingshot && cometIndex >= 0) {
        renderer.setCometTrail(cometIndex, cfg.cometTrailWidth);
      }
    }

    if (needEnergy) {
      launchComputeEnergy(d, cfg.n, cfg.G, eps2, blockSize);
      energies.resize(cfg.n);
      CUDA_CHECK(cudaMemcpy(energies.data(), d.energy, cfg.n * sizeof(float), cudaMemcpyDeviceToHost));
      if (cfg.cudaDebug) CUDA_CHECK(cudaDeviceSynchronize());
      if (step % 200 == 0) {
        double totalEnergy = 0.0;
        double totalKinetic = 0.0;
        for (int i = 0; i < cfg.n; ++i) {
          totalEnergy += energies[i];
          double vx = static_cast<double>(h.vx[i]);
          double vy = static_cast<double>(h.vy[i]);
          double vz = static_cast<double>(h.vz[i]);
          double v2 = vx * vx + vy * vy + vz * vz;
          totalKinetic += 0.5 * static_cast<double>(h.m[i]) * v2;
        }
        double totalPotential = 0.5 * (totalEnergy - totalKinetic);
        std::cout << "Energy: K=" << totalKinetic << " U=" << totalPotential
                  << " E=" << totalEnergy << "\n";
      }
    }

    int minBody = -1;
    int minPartner = -1;
    float minDist2 = 1e30f;
    if (needMinDist) {
      launchComputeMinDist(d, cfg.n, blockSize);
      computeGlobalMinPair(d, cfg.n, blockSize, &minDist2, &minBody, &minPartner);
    }

    if (cfg.collisions) {
      if (!pairsReady) {
        int pairCount = 0;
        if (d.pairCount) {
          CUDA_CHECK(cudaMemcpy(&pairCount, d.pairCount, sizeof(int), cudaMemcpyDeviceToHost));
          if (pairCount > d.maxPairs) {
            pairCount = d.maxPairs;
          }
          if (cfg.maxCollisionPairs > 0 && pairCount > cfg.maxCollisionPairs) {
            pairCount = cfg.maxCollisionPairs;
          }
          if (pairCount > 0) {
            collisionPairs.resize(pairCount);
            CUDA_CHECK(cudaMemcpy(collisionPairs.data(), d.pairs, pairCount * sizeof(int2), cudaMemcpyDeviceToHost));
            if (cfg.cudaDebug) CUDA_CHECK(cudaDeviceSynchronize());
          } else {
            collisionPairs.clear();
          }
        }
      }
      if (collisionFlash.size() != static_cast<size_t>(cfg.n)) {
        collisionFlash.assign(cfg.n, 0.0f);
      }
      if (!paused) {
        float decay = cfg.dt * timeScale;
        for (int i = 0; i < cfg.n; ++i) {
          collisionFlash[i] = std::max(0.0f, collisionFlash[i] - decay);
        }
        for (auto &evt : collisionEvents) {
          evt.timer = std::max(0.0f, evt.timer - decay);
        }
        if (lastCollisionFocusTimer > 0.0f) {
          lastCollisionFocusTimer = std::max(0.0f, lastCollisionFocusTimer - decay);
        }
        if (collisionFocusCooldown > 0.0f) {
          collisionFocusCooldown = std::max(0.0f, collisionFocusCooldown - decay);
        }
        if (lastShockTimer > 0.0f) {
          lastShockTimer = std::max(0.0f, lastShockTimer - decay);
        }
      }
      collisionEvents.erase(std::remove_if(collisionEvents.begin(), collisionEvents.end(),
                                           [](const CollisionEvent &e) { return e.timer <= 0.0f; }),
                            collisionEvents.end());
      const float flashDuration = std::max(0.0f, cfg.collisionFlashDuration);
      if (!collisionPairs.empty()) {
        int addedThisFrame = 0;
        for (const auto &p : collisionPairs) {
          if (static_cast<int>(collisionEvents.size()) >= cfg.maxHighlightedCollisions) {
            break;
          }
          if (addedThisFrame >= cfg.maxHighlightedCollisions) {
            break;
          }
          bool exists = false;
          for (const auto &evt : collisionEvents) {
            if ((evt.a == p.x && evt.b == p.y) || (evt.a == p.y && evt.b == p.x)) {
              exists = true;
              break;
            }
          }
          if (!exists) {
            collisionEvents.push_back({p.x, p.y, flashDuration});
            lastCollisionA = p.x;
            lastCollisionB = p.y;
            if (collisionFocusCooldown <= 0.0f) {
              lastCollisionFocusTimer = std::max(0.0f, cfg.collisionFocusSeconds);
              collisionFocusCooldown = std::max(0.0f, cfg.collisionFocusCooldownSeconds);
            }
            lastShockTimer = std::max(0.0f, cfg.collisionFlashDuration);
            if (cfg.slowOnCollision) {
              collisionHoldTimer = std::max(collisionHoldTimer, cfg.collisionHoldSeconds);
              collisionSlowTimer = std::max(collisionSlowTimer, cfg.collisionSlowSeconds);
            }
            addedThisFrame++;
          }
        }
      }

      for (const auto &evt : collisionEvents) {
        if (evt.a >= 0 && evt.a < cfg.n) collisionFlash[evt.a] = std::max(collisionFlash[evt.a], evt.timer);
        if (evt.b >= 0 && evt.b < cfg.n) collisionFlash[evt.b] = std::max(collisionFlash[evt.b], evt.timer);
      }

      if (cfg.collisionModel == 1 && cfg.fragmentation && !collisionPairs.empty() &&
          cfg.maxDebris > 0 && debrisSpawned < cfg.maxDebris) {
        std::vector<int> available;
        available.reserve(cfg.maxDebris);
        for (int i = 0; i < cfg.n && static_cast<int>(available.size()) < cfg.maxDebris; ++i) {
          if (h.m[i] == 0.0f) {
            available.push_back(i);
          }
        }
        std::uniform_real_distribution<float> spread(-1.0f, 1.0f);
        for (const auto &p : collisionPairs) {
          if (debrisSpawned >= cfg.maxDebris) break;
          int ia = p.x;
          int ib = p.y;
          if (ia < 0 || ib < 0 || ia >= cfg.n || ib >= cfg.n) continue;
          if (h.m[ia] <= 0.0f || h.m[ib] <= 0.0f) continue;
          if (!cfg.allowPlanetFragment && (h.type[ia] <= 1 || h.type[ib] <= 1)) continue;
          int k = std::min(cfg.fragmentsPerCollision, static_cast<int>(available.size()));
          if (k < 2) break;
          float totalMass = h.m[ia] + h.m[ib];
          if (totalMass <= 1e-8f) {
            continue;
          }
          float invMass = 1.0f / totalMass;
          float vcomx = (h.m[ia] * h.vx[ia] + h.m[ib] * h.vx[ib]) * invMass;
          float vcomy = (h.m[ia] * h.vy[ia] + h.m[ib] * h.vy[ib]) * invMass;
          float vcomz = (h.m[ia] * h.vz[ia] + h.m[ib] * h.vz[ib]) * invMass;
          float vcomMag = std::sqrt(vcomx * vcomx + vcomy * vcomy + vcomz * vcomz);
          if (!std::isfinite(vcomMag)) {
            continue;
          }
          float spreadScale = std::max(1e-3f, vcomMag) * cfg.fragmentSpeedSpread;
          float maxSpread = std::max(1e-3f, vcomMag) * cfg.fragmentSpeedClamp;
          if (spreadScale > maxSpread) {
            spreadScale = maxSpread;
          }
          float cx = 0.5f * (h.x[ia] + h.x[ib]);
          float cy = 0.5f * (h.y[ia] + h.y[ib]);
          float cz = 0.5f * (h.z[ia] + h.z[ib]);

          std::vector<std::array<float, 3>> jitter(k);
          float avgx = 0.0f, avgy = 0.0f, avgz = 0.0f;
          for (int i = 0; i < k; ++i) {
            float rx = spread(fragRng);
            float ry = spread(fragRng);
            float rz = spread(fragRng);
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
            int slot = available.back();
            available.pop_back();
            float jx = jitter[i][0] - avgx;
            float jy = jitter[i][1] - avgy;
            float jz = jitter[i][2] - avgz;
            h.x[slot] = cx + jx * cfg.eps * 0.5f;
            h.y[slot] = cy + jy * cfg.eps * 0.5f;
            h.z[slot] = cz + jz * cfg.eps * 0.5f;
            h.vx[slot] = vcomx + jx * spreadScale;
            h.vy[slot] = vcomy + jy * spreadScale;
            h.vz[slot] = vcomz + jz * spreadScale;
            h.m[slot] = fragMass;
            h.rad[slot] = 0.02f;
            h.vrad[slot] = 0.03f;
            h.type[slot] = 4;
            CUDA_CHECK(cudaMemcpy(d.x + slot, &h.x[slot], sizeof(float), cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(d.y + slot, &h.y[slot], sizeof(float), cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(d.z + slot, &h.z[slot], sizeof(float), cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(d.vx + slot, &h.vx[slot], sizeof(float), cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(d.vy + slot, &h.vy[slot], sizeof(float), cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(d.vz + slot, &h.vz[slot], sizeof(float), cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(d.m + slot, &h.m[slot], sizeof(float), cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(d.rad + slot, &h.rad[slot], sizeof(float), cudaMemcpyHostToDevice));
            if (cfg.cudaDebug) CUDA_CHECK(cudaDeviceSynchronize());
            collisionFlash[slot] = flashDuration;
          }
          debrisSpawned += k;
          h.m[ia] = 0.0f;
          h.m[ib] = 0.0f;
          h.rad[ia] = 0.0f;
          h.rad[ib] = 0.0f;
          h.vrad[ia] = 0.0f;
          h.vrad[ib] = 0.0f;
          h.vx[ia] = h.vy[ia] = h.vz[ia] = 0.0f;
          h.vx[ib] = h.vy[ib] = h.vz[ib] = 0.0f;
          CUDA_CHECK(cudaMemcpy(d.m + ia, &h.m[ia], sizeof(float), cudaMemcpyHostToDevice));
          CUDA_CHECK(cudaMemcpy(d.m + ib, &h.m[ib], sizeof(float), cudaMemcpyHostToDevice));
          CUDA_CHECK(cudaMemcpy(d.rad + ia, &h.rad[ia], sizeof(float), cudaMemcpyHostToDevice));
          CUDA_CHECK(cudaMemcpy(d.rad + ib, &h.rad[ib], sizeof(float), cudaMemcpyHostToDevice));
          if (cfg.cudaDebug) CUDA_CHECK(cudaDeviceSynchronize());
          sizesDirty = true;
        }
      }
    }

    float renderScale = cfg.worldScale;
    closeFlags.clear();
    closeLines.clear();
    if (cfg.highlightCloseEncounters) {
      closeFlags.assign(cfg.n, 0);
      float threshold2 = 9.0f * cfg.eps * cfg.eps;
      for (int i = 0; i < cfg.n; ++i) {
        for (int j = i + 1; j < cfg.n; ++j) {
          float dx = h.x[j] - h.x[i];
          float dy = h.y[j] - h.y[i];
          float dz = h.z[j] - h.z[i];
          float d2 = dx * dx + dy * dy + dz * dz;
          if (d2 < threshold2) {
            closeFlags[i] = 1;
            closeFlags[j] = 1;
            closeLines.push_back((h.x[i] - comX) * renderScale);
            closeLines.push_back((h.y[i] - comY) * renderScale);
            closeLines.push_back((h.z[i] - comZ) * renderScale);
            closeLines.push_back((h.x[j] - comX) * renderScale);
            closeLines.push_back((h.y[j] - comY) * renderScale);
            closeLines.push_back((h.z[j] - comZ) * renderScale);
          }
        }
      }
    }
    if (cfg.collisions && cfg.collisionShockwave && lastShockTimer > 0.0f &&
        lastCollisionA >= 0 && lastCollisionB >= 0) {
      float flashDuration = std::max(0.001f, cfg.collisionFlashDuration);
      float t = 1.0f - (lastShockTimer / flashDuration);
      t = std::clamp(t, 0.0f, 1.0f);
      float mx = (0.5f * (h.x[lastCollisionA] + h.x[lastCollisionB]) - comX) * renderScale;
      float my = (0.5f * (h.y[lastCollisionA] + h.y[lastCollisionB]) - comY) * renderScale;
      float mz = (0.5f * (h.z[lastCollisionA] + h.z[lastCollisionB]) - comZ) * renderScale;
      float base = cfg.eps * 2.0f * renderScale;
      float radius = base + t * cfg.eps * 6.0f * renderScale;
      const int segments = 32;
      for (int s = 0; s < segments; ++s) {
        float a0 = kTwoPi * (static_cast<float>(s) / segments);
        float a1 = kTwoPi * (static_cast<float>(s + 1) / segments);
        closeLines.push_back(mx + std::cos(a0) * radius);
        closeLines.push_back(my + std::sin(a0) * radius);
        closeLines.push_back(mz);
        closeLines.push_back(mx + std::cos(a1) * radius);
        closeLines.push_back(my + std::sin(a1) * radius);
        closeLines.push_back(mz);
      }
    }
    if (cfg.collisions && !collisionEvents.empty()) {
      for (const auto &evt : collisionEvents) {
        if (evt.a < 0 || evt.b < 0 || evt.a >= cfg.n || evt.b >= cfg.n) {
          continue;
        }
        closeLines.push_back((h.x[evt.a] - comX) * renderScale);
        closeLines.push_back((h.y[evt.a] - comY) * renderScale);
        closeLines.push_back((h.z[evt.a] - comZ) * renderScale);
        closeLines.push_back((h.x[evt.b] - comX) * renderScale);
        closeLines.push_back((h.y[evt.b] - comY) * renderScale);
        closeLines.push_back((h.z[evt.b] - comZ) * renderScale);
      }
    }
    size_t maxCloseFloats = static_cast<size_t>(cfg.n) * 2u * 3u;
    if (closeLines.size() > maxCloseFloats) {
      closeLines.resize(maxCloseFloats);
    }
    renderer.updateCloseLines(closeLines);

    if (cfg.showVelocityVectors && !velocities.empty()) {
      velocityLines.clear();
      velocityLines.reserve(cfg.n * 6 * 3);
      float headLen = std::max(0.0f, cfg.arrowHeadLength) * renderScale;
      float headWidth = std::max(0.0f, cfg.arrowHeadWidth) * renderScale;
      if (headLen <= 0.0f) {
        headWidth = 0.0f;
      }
      for (int i = 0; i < cfg.n; ++i) {
        bool major = (cfg.velocityMassThreshold <= 0.0f) || (h.m[i] >= cfg.velocityMassThreshold);
        bool close = (!closeFlags.empty() && closeFlags[i]);
        bool flash = (!collisionFlash.empty() && collisionFlash[i] > 0.0f);
        if (!major && !close && !flash) {
          continue;
        }
        float px = (h.x[i] - comX) * renderScale;
        float py = (h.y[i] - comY) * renderScale;
        float pz = (h.z[i] - comZ) * renderScale;
        float vx = velocities[3 * i + 0];
        float vy = velocities[3 * i + 1];
        float vz = velocities[3 * i + 2];
        float vlen = std::sqrt(vx * vx + vy * vy + vz * vz);
        float tipx = px;
        float tipy = py;
        float tipz = pz;
        float dirx = 0.0f;
        float diry = 0.0f;
        float dirz = 0.0f;
        float scaledLen = 0.0f;
        if (vlen > 1e-6f) {
          dirx = vx / vlen;
          diry = vy / vlen;
          dirz = vz / vlen;
          scaledLen = vlen * cfg.velocityVectorScale * renderScale;
          tipx = px + dirx * scaledLen;
          tipy = py + diry * scaledLen;
          tipz = pz + dirz * scaledLen;
        }

        float hl = headLen;
        if (scaledLen > 0.0f && hl > scaledLen) {
          hl = scaledLen;
        }

        float upx = 0.0f;
        float upy = 0.0f;
        float upz = 1.0f;
        float cx = diry * upz - dirz * upy;
        float cy = dirz * upx - dirx * upz;
        float cz = dirx * upy - diry * upx;
        float clen = std::sqrt(cx * cx + cy * cy + cz * cz);
        if (clen < 1e-4f) {
          upx = 0.0f;
          upy = 1.0f;
          upz = 0.0f;
          cx = diry * upz - dirz * upy;
          cy = dirz * upx - dirx * upz;
          cz = dirx * upy - diry * upx;
          clen = std::sqrt(cx * cx + cy * cy + cz * cz);
        }
        float perpx = 0.0f;
        float perpy = 0.0f;
        float perpz = 0.0f;
        if (clen > 1e-6f) {
          perpx = cx / clen;
          perpy = cy / clen;
          perpz = cz / clen;
        }

        float leftx = tipx - dirx * hl + perpx * headWidth;
        float lefty = tipy - diry * hl + perpy * headWidth;
        float leftz = tipz - dirz * hl + perpz * headWidth;
        float rightx = tipx - dirx * hl - perpx * headWidth;
        float righty = tipy - diry * hl - perpy * headWidth;
        float rightz = tipz - dirz * hl - perpz * headWidth;

        velocityLines.push_back(px);
        velocityLines.push_back(py);
        velocityLines.push_back(pz);
        velocityLines.push_back(tipx);
        velocityLines.push_back(tipy);
        velocityLines.push_back(tipz);

        velocityLines.push_back(tipx);
        velocityLines.push_back(tipy);
        velocityLines.push_back(tipz);
        velocityLines.push_back(leftx);
        velocityLines.push_back(lefty);
        velocityLines.push_back(leftz);

        velocityLines.push_back(tipx);
        velocityLines.push_back(tipy);
        velocityLines.push_back(tipz);
        velocityLines.push_back(rightx);
        velocityLines.push_back(righty);
        velocityLines.push_back(rightz);
      }
      renderer.updateVelocityLines(velocityLines);
    } else {
      velocityLines.clear();
      renderer.updateVelocityLines(velocityLines);
    }

    cometTailLines.clear();
    if (cfg.cometTail && !cometIndices.empty() && !velocities.empty()) {
      cometTailLines.reserve(static_cast<size_t>(cometIndices.size()) * 2 * 3);
      float influence = std::max(3.0f, cfg.asteroidBeltRmax * 2.0f);
      for (int idxComet : cometIndices) {
        if (idxComet < 0 || idxComet >= cfg.n) {
          continue;
        }
        float vx = velocities[3 * idxComet + 0];
        float vy = velocities[3 * idxComet + 1];
        float vz = velocities[3 * idxComet + 2];
        float vlen = std::sqrt(vx * vx + vy * vy + vz * vz);
        if (vlen < 1e-6f) {
          continue;
        }
        float rx = h.x[idxComet];
        float ry = h.y[idxComet];
        float rz = h.z[idxComet];
        float dist = std::sqrt(rx * rx + ry * ry + rz * rz);
        float factor = 1.0f - std::min(1.0f, dist / influence);
        float tailLen = cfg.cometTailLength * (0.5f + 1.5f * factor);
        float dirx = -vx / vlen;
        float diry = -vy / vlen;
        float dirz = -vz / vlen;
        float px = (rx - comX) * renderScale;
        float py = (ry - comY) * renderScale;
        float pz = (rz - comZ) * renderScale;
        float tx = px + dirx * tailLen * renderScale;
        float ty = py + diry * tailLen * renderScale;
        float tz = pz + dirz * tailLen * renderScale;
        cometTailLines.push_back(px);
        cometTailLines.push_back(py);
        cometTailLines.push_back(pz);
        cometTailLines.push_back(tx);
        cometTailLines.push_back(ty);
        cometTailLines.push_back(tz);
      }
      renderer.updateAuxLines(cometTailLines);
    } else {
      renderer.updateAuxLines(cometTailLines);
    }

    if (cfg.showEnergyColor || cfg.highlightCloseEncounters || !collisionFlash.empty()) {
      updateDynamicColors(baseColors, energies, closeFlags, collisionFlash, cfg.showEnergyColor, dynamicColors);
      renderer.updateColors(dynamicColors);
    } else {
      renderer.updateColors(baseColors);
    }

    int winW = 0, winH = 0;
    renderer.getWindowSize(winW, winH);
    float camPanX = 0.0f, camPanY = 0.0f, camPanZ = 0.0f;
    float camYaw = 0.0f, camPitch = 0.0f, camZoom = 1.0f, camView = 1.0f;
    renderer.getCamera(camPanX, camPanY, camPanZ, camYaw, camPitch, camZoom, camView);
    float pixelScale = 0.5f * static_cast<float>(std::max(1, winH)) * (camZoom / std::max(camView, 1e-6f));
    bool sizeUpdate = sizesDirty || (std::abs(pixelScale - lastPixelScale) > 1e-3f);
    if (sizeUpdate) {
      computeRenderSizes(h, cfg, cfg.worldScale, pixelScale, pointScale, baseSizes);
      lastPixelScale = pixelScale;
      sizesDirty = false;
    }

    if (!collisionFlash.empty() && cfg.collisionFlashScale > 0.0f) {
      dynamicSizes = baseSizes;
      float denom = std::max(0.001f, cfg.collisionFlashDuration);
      for (int i = 0; i < cfg.n; ++i) {
        if (collisionFlash[i] > 0.0f) {
          float t = collisionFlash[i] / denom;
          dynamicSizes[i] = baseSizes[i] * (1.0f + cfg.collisionFlashScale * t);
        }
      }
      renderer.updateSizes(dynamicSizes);
    } else {
      if (sizeUpdate) {
        renderer.updateSizes(baseSizes);
      } else if (!dynamicSizes.empty()) {
        renderer.updateSizes(baseSizes);
      }
      dynamicSizes.clear();
    }

    if (trailLen > 0 && renderer.trailsEnabled()) {
      int offset = trailHead * cfg.n * 3;
      for (int i = 0; i < cfg.n; ++i) {
        trailBuffer[offset + 3 * i + 0] = h.x[i];
        trailBuffer[offset + 3 * i + 1] = h.y[i];
        trailBuffer[offset + 3 * i + 2] = h.z[i];
      }
      trailHead = (trailHead + 1) % trailLen;
      trailCount = std::min(trailCount + 1, trailLen);

      for (int i = 0; i < cfg.n; ++i) {
        for (int t = 0; t < trailCount; ++t) {
          int idx = (trailHead + t) % trailLen;
          int src = idx * cfg.n * 3 + 3 * i;
          int dst = i * trailLen * 3 + 3 * t;
          trailRender[dst + 0] = (trailBuffer[src + 0] - comX) * renderScale;
          trailRender[dst + 1] = (trailBuffer[src + 1] - comY) * renderScale;
          trailRender[dst + 2] = (trailBuffer[src + 2] - comZ) * renderScale;
        }
      }
      renderer.updateTrails(trailRender, trailCount);
    }

    auto smoothFocus = [&](float tx, float ty, float tz, float radius) {
      float panX = 0.0f, panY = 0.0f, panZ = 0.0f, yaw = 0.0f, pitch = 0.0f, zoom = 1.0f, viewRadius = 1.0f;
      renderer.getCamera(panX, panY, panZ, yaw, pitch, zoom, viewRadius);
      float desiredPanX = -tx;
      float desiredPanY = -ty;
      float desiredPanZ = -tz;
      float alpha = std::clamp(cfg.cameraLerp, 0.0f, 1.0f);
      float beta = std::clamp(cfg.zoomLerp, 0.0f, 1.0f);
      if (!std::isfinite(radius)) {
        radius = cfg.maxFocusRadius;
      }
      float targetRadius = std::clamp(radius, cfg.minFocusRadius, cfg.maxFocusRadius);
      panX = panX + (desiredPanX - panX) * alpha;
      panY = panY + (desiredPanY - panY) * alpha;
      panZ = panZ + (desiredPanZ - panZ) * alpha;
      viewRadius = viewRadius + (targetRadius - viewRadius) * beta;
      viewRadius = std::clamp(viewRadius, cfg.minViewRadius, cfg.maxViewRadius);
      renderer.setCamera(panX, panY, panZ, zoom, viewRadius);
    };

    if (actions.focusSelected && selectedIndex >= 0 && selectedIndex < cfg.n) {
      smoothFocus((h.x[selectedIndex] - comX) * cfg.worldScale,
                  (h.y[selectedIndex] - comY) * cfg.worldScale,
                  (h.z[selectedIndex] - comZ) * cfg.worldScale,
                  cfg.eps * 10.0f * cfg.worldScale);
    }
    if (actions.focusClosest && minBody >= 0 && minPartner >= 0) {
      float mx = (0.5f * (h.x[minBody] + h.x[minPartner]) - comX) * cfg.worldScale;
      float my = (0.5f * (h.y[minBody] + h.y[minPartner]) - comY) * cfg.worldScale;
      float mz = (0.5f * (h.z[minBody] + h.z[minPartner]) - comZ) * cfg.worldScale;
      float radius = std::sqrt(minDist2) * 3.0f * cfg.worldScale + cfg.eps * cfg.worldScale;
      smoothFocus(mx, my, mz, radius);
    }
    bool focusAllowed = simTime >= cfg.initialFocusDelay;
    if (focusAllowed && cfg.collisions && cfg.collisionFocus &&
        lastCollisionFocusTimer > 0.0f && lastCollisionA >= 0 && lastCollisionB >= 0) {
      float mx = (0.5f * (h.x[lastCollisionA] + h.x[lastCollisionB]) - comX) * cfg.worldScale;
      float my = (0.5f * (h.y[lastCollisionA] + h.y[lastCollisionB]) - comY) * cfg.worldScale;
      float mz = (0.5f * (h.z[lastCollisionA] + h.z[lastCollisionB]) - comZ) * cfg.worldScale;
      float dx = h.x[lastCollisionA] - h.x[lastCollisionB];
      float dy = h.y[lastCollisionA] - h.y[lastCollisionB];
      float dz = h.z[lastCollisionA] - h.z[lastCollisionB];
      float radius = std::sqrt(dx * dx + dy * dy + dz * dz) * 2.0f * cfg.worldScale + cfg.eps * cfg.worldScale;
      smoothFocus(mx, my, mz, radius);
    } else if (focusAllowed && cfg.autoFocus && minBody >= 0 && minPartner >= 0) {
      float minDist = std::sqrt(minDist2);
      if (minDist < focusThreshold) {
        float mx = (0.5f * (h.x[minBody] + h.x[minPartner]) - comX) * cfg.worldScale;
        float my = (0.5f * (h.y[minBody] + h.y[minPartner]) - comY) * cfg.worldScale;
        float mz = (0.5f * (h.z[minBody] + h.z[minPartner]) - comZ) * cfg.worldScale;
        float radius = minDist * 3.0f * cfg.worldScale + cfg.eps * cfg.worldScale;
        smoothFocus(mx, my, mz, radius);
      }
    }

    {
      float panX = 0.0f, panY = 0.0f, panZ = 0.0f, yaw = 0.0f, pitch = 0.0f, zoom = 1.0f, view = 1.0f;
      renderer.getCamera(panX, panY, panZ, yaw, pitch, zoom, view);
      bool invalid = !std::isfinite(panX) || !std::isfinite(panY) || !std::isfinite(panZ)
                     || !std::isfinite(zoom) || !std::isfinite(view);
      float minView = std::max(1e-4f, cfg.minViewRadius);
      if (view < minView * 0.5f || view > cfg.maxViewRadius * 2.0f) {
        invalid = true;
      }
      if (invalid) {
        renderer.setCamera(resetPanX, resetPanY, resetPanZ, resetZoom, resetViewRadiusClamped);
      }
    }

    if (needDebug) {
      CollisionDebug debug{};
      CUDA_CHECK(cudaMemcpy(&debug, d.debug, sizeof(CollisionDebug), cudaMemcpyDeviceToHost));
      if (cfg.cudaDebug) CUDA_CHECK(cudaDeviceSynchronize());
      if (debug.valid) {
        std::cout << "Collision " << debug.i << "<->" << debug.j << "\n";
        std::cout << "  masses=(" << debug.mi << "," << debug.mj << ")\n";
        std::cout << "  preVi=(" << debug.preVi[0] << "," << debug.preVi[1] << "," << debug.preVi[2] << ")"
                  << " preVj=(" << debug.preVj[0] << "," << debug.preVj[1] << "," << debug.preVj[2] << ")\n";
        std::cout << "  postVi=(" << debug.postVi[0] << "," << debug.postVi[1] << "," << debug.postVi[2] << ")"
                  << " postVj=(" << debug.postVj[0] << "," << debug.postVj[1] << "," << debug.postVj[2] << ")\n";
        std::cout << "  preP=(" << debug.preP[0] << "," << debug.preP[1] << "," << debug.preP[2] << ")"
                  << " postP=(" << debug.postP[0] << "," << debug.postP[1] << "," << debug.postP[2] << ")\n";
      }
    }

    renderer.render();
    step++;
  }

  renderer.shutdown();
  freeDeviceArrays(d);
  return 0;
}
