#pragma once

#include "config.h"
#include "init.h"
#include "integrator.h"

#include <random>
#include <vector>

void stepSimulation(const DeviceArrays &d, int n, float G, float eps2, float dt,
                    bool collisions, float restitution, bool debugCollision,
                    bool cudaDebug, int blockSize);

bool stepSimulationMerge(DeviceArrays &d, HostArrays &h, std::vector<int2> &pairs,
                         const Config &cfg, std::mt19937 &rng, int &debrisActive,
                         float dt, int blockSize, bool debugCollision);
