#include "bench.h"

#include "init.h"
#include "integrator.h"
#include "sim.h"
#include "utils.h"

#include <iomanip>
#include <iostream>
#include <vector>

void runBenchmark(const Config &cfg, int blockSize) {
  std::vector<int> sizes = {1024, 2048, 4096, 8192, 16384};
  int steps = cfg.steps > 0 ? cfg.steps : 100;

  std::cout << "Benchmark mode (render disabled)\n";
  std::cout << "steps=" << steps << " dt=" << cfg.dt << " eps=" << cfg.eps
            << " blockSize=" << blockSize << " collisions=" << (cfg.collisions ? 1 : 0) << "\n\n";
  std::cout << "N | dt | eps | blockSize | time_per_step_ms | IPS\n";

  for (int n : sizes) {
    InitConfig initCfg;
    initCfg.n = n;
    initCfg.seed = cfg.seed;
    initCfg.G = cfg.G;
    initCfg.massScale = cfg.massScale;
    initCfg.sunMass = cfg.sunMass;
    initCfg.planetMass = cfg.planetMass;
    initCfg.cometMass = cfg.cometMass;
    initCfg.minorMass = cfg.minorMass;
    initCfg.numPlanets = cfg.numPlanets;
    initCfg.planetRadiiPreset = cfg.planetRadiiPreset;
    initCfg.planetEccMax = cfg.planetEccMax;
    initCfg.planetIncDegMax = cfg.planetIncDegMax;
    initCfg.asteroidBelt = cfg.asteroidBelt;
    initCfg.asteroidBeltRmin = cfg.asteroidBeltRmin;
    initCfg.asteroidBeltRmax = cfg.asteroidBeltRmax;
    initCfg.asteroidTangentialJitter = cfg.asteroidTangentialJitter;
    initCfg.asteroidRadialJitter = cfg.asteroidRadialJitter;
    initCfg.numComets = cfg.numComets;
    initCfg.cometEccMin = cfg.cometEccMin;
    initCfg.cometEccMax = cfg.cometEccMax;
    initCfg.cometIncDegMax = cfg.cometIncDegMax;
    initCfg.slingshotTargetPlanet = cfg.slingshotTargetPlanet;
    initCfg.fragmentation = cfg.fragmentation;
    initCfg.maxDebris = cfg.maxDebris;
    initCfg.zeroMomentum = cfg.zeroMomentum;
    initCfg.realSolarSystem = cfg.realSolarSystem;
    initCfg.demoCollision = cfg.demoCollision;
    initCfg.demoSlingshot = cfg.demoSlingshot;

    HostArrays h;
    float maxRadius = 0.0f;
    initSolarSystem(h, initCfg, maxRadius);

    DeviceArrays d;
    allocateDeviceArrays(d, n, cfg.collisions, false, false, cfg.debugCollision, false, false);
    CUDA_CHECK(cudaMemcpy(d.x, h.x.data(), n * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d.y, h.y.data(), n * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d.z, h.z.data(), n * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d.vx, h.vx.data(), n * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d.vy, h.vy.data(), n * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d.vz, h.vz.data(), n * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d.m, h.m.data(), n * sizeof(float), cudaMemcpyHostToDevice));
    if (cfg.collisions) {
      CUDA_CHECK(cudaMemcpy(d.rad, h.rad.data(), n * sizeof(float), cudaMemcpyHostToDevice));
    }

    float eps2 = cfg.eps * cfg.eps;
    launchComputeAccel(d, n, cfg.G, eps2, blockSize);

    for (int i = 0; i < 5; ++i) {
      stepSimulation(d, n, cfg.G, eps2, cfg.dt, cfg.collisions, cfg.collisionRestitution,
                     cfg.debugCollision, cfg.cudaDebug, blockSize);
    }

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    CUDA_CHECK(cudaEventRecord(start));
    for (int i = 0; i < steps; ++i) {
      stepSimulation(d, n, cfg.G, eps2, cfg.dt, cfg.collisions, cfg.collisionRestitution,
                     cfg.debugCollision, cfg.cudaDebug, blockSize);
    }
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
    float timePerStepMs = ms / static_cast<float>(steps);
    double timePerStepSec = timePerStepMs * 1.0e-3;
    double ips = static_cast<double>(n) * static_cast<double>(n - 1) / timePerStepSec;

    std::cout << n << " | " << std::fixed << std::setprecision(4) << cfg.dt
              << " | " << std::fixed << std::setprecision(4) << cfg.eps
              << " | " << blockSize
              << " | " << std::fixed << std::setprecision(3) << timePerStepMs
              << " | " << std::scientific << std::setprecision(3) << ips << "\n";

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    freeDeviceArrays(d);
  }
}
