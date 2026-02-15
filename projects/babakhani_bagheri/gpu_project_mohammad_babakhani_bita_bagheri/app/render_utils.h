#pragma once

#include "config.h"
#include "init.h"

#include <vector>

void fillColors(const HostArrays &h, std::vector<float> &colors);
void fillTrailColors(const std::vector<float> &baseColors, int trailLen,
                     std::vector<float> &trailColors);
void interleavePositions(const HostArrays &h, float ox, float oy, float oz, float scale,
                         std::vector<float> &out);
void computeRenderSizes(const HostArrays &h, const Config &cfg, float worldScale,
                        float pixelScale, float pointScale, std::vector<float> &sizes);
void computeCOM(const HostArrays &h, float &comX, float &comY, float &comZ,
                float &vcomX, float &vcomY, float &vcomZ);
void updateDynamicColors(const std::vector<float> &baseColors,
                         const std::vector<float> &energies,
                         const std::vector<int> &closeFlags,
                         const std::vector<float> &collisionFlash,
                         bool showEnergyColor,
                         std::vector<float> &outColors);
