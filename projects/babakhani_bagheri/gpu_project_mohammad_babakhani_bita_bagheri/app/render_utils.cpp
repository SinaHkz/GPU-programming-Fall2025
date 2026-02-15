#include "render_utils.h"

#include <algorithm>
#include <array>
#include <cmath>

void fillColors(const HostArrays &h, std::vector<float> &colors) {
  colors.resize(h.type.size() * 4);
  const std::array<std::array<float, 3>, 8> planetColors = {{
      {0.6f, 0.6f, 0.6f},  // Mercury
      {0.9f, 0.85f, 0.6f}, // Venus
      {0.2f, 0.55f, 1.0f}, // Earth
      {0.9f, 0.35f, 0.2f}, // Mars
      {0.85f, 0.75f, 0.55f}, // Jupiter
      {0.8f, 0.7f, 0.5f},  // Saturn
      {0.5f, 0.8f, 0.9f},  // Uranus
      {0.2f, 0.3f, 0.9f}   // Neptune
  }};
  int planetIndex = 0;
  for (size_t i = 0; i < h.type.size(); ++i) {
    float r = 0.8f;
    float g = 0.8f;
    float b = 0.8f;
    switch (h.type[i]) {
      case 0: r = 1.0f; g = 0.9f; b = 0.2f; break; // Sun
      case 1: {
        int idx = std::min(planetIndex, static_cast<int>(planetColors.size()) - 1);
        r = planetColors[idx][0];
        g = planetColors[idx][1];
        b = planetColors[idx][2];
        planetIndex++;
        break;
      }
      case 2: r = 0.85f; g = 0.9f; b = 1.0f; break; // Comet
      case 3: r = 0.6f; g = 0.6f; b = 0.6f; break; // Minor
      case 4: r = 0.7f; g = 0.5f; b = 0.5f; break; // Debris
      default: break;
    }
    colors[4 * i + 0] = r;
    colors[4 * i + 1] = g;
    colors[4 * i + 2] = b;
    colors[4 * i + 3] = 1.0f;
  }
}

void fillTrailColors(const std::vector<float> &baseColors, int trailLen,
                     std::vector<float> &trailColors) {
  int n = static_cast<int>(baseColors.size() / 4);
  trailColors.resize(trailLen * n * 4);
  for (int t = 0; t < trailLen; ++t) {
    for (int i = 0; i < n; ++i) {
      trailColors[(t * n + i) * 4 + 0] = baseColors[i * 4 + 0];
      trailColors[(t * n + i) * 4 + 1] = baseColors[i * 4 + 1];
      trailColors[(t * n + i) * 4 + 2] = baseColors[i * 4 + 2];
      trailColors[(t * n + i) * 4 + 3] = 1.0f;
    }
  }
}

void interleavePositions(const HostArrays &h, float ox, float oy, float oz, float scale,
                         std::vector<float> &out) {
  int n = static_cast<int>(h.x.size());
  out.resize(n * 3);
  for (int i = 0; i < n; ++i) {
    out[3 * i + 0] = (h.x[i] - ox) * scale;
    out[3 * i + 1] = (h.y[i] - oy) * scale;
    out[3 * i + 2] = (h.z[i] - oz) * scale;
  }
}

void computeRenderSizes(const HostArrays &h, const Config &cfg, float worldScale,
                        float pixelScale, float pointScale,
                        std::vector<float> &sizes) {
  sizes.resize(h.m.size());
  float scale = std::max(worldScale, 1e-6f);
  float pxScale = std::max(pixelScale, 1e-6f);
  float point = std::max(pointScale, 1e-6f);
  float sunCapPx = 0.0f;
  for (size_t i = 0; i < h.m.size(); ++i) {
    if (h.type[i] != 0) continue;
    float sizeWorld = h.vrad[i] * cfg.sunVisualScale * cfg.sunVisualBoost * scale;
    float sizePx = sizeWorld * pxScale;
    sizePx = std::max(sizePx, cfg.minSunPixels);
    sunCapPx = sizePx * 0.95f;
    break;
  }
  for (size_t i = 0; i < h.m.size(); ++i) {
    if (h.m[i] <= 0.0f && h.vrad[i] <= 0.0f) {
      sizes[i] = 0.0f;
      continue;
    }
    float sizeWorld = 0.0f;
    float minPixels = cfg.minMinorPixels;
    float maxPixels = -1.0f;
    if (h.type[i] == 0) {
      sizeWorld = h.vrad[i] * cfg.sunVisualScale * cfg.sunVisualBoost;
      minPixels = cfg.minSunPixels;
    } else if (h.type[i] == 1) {
      sizeWorld = h.vrad[i] * cfg.planetVisualScale * cfg.planetVisualBoost;
      minPixels = cfg.minPlanetPixels;
    } else if (h.type[i] == 2) {
      sizeWorld = h.vrad[i] * cfg.cometVisualScale * cfg.cometVisualBoost;
      minPixels = cfg.minCometPixels;
    } else {
      sizeWorld = h.vrad[i] * cfg.minorVisualScale * cfg.minorVisualBoost;
      minPixels = cfg.minMinorPixels;
      maxPixels = cfg.maxMinorPixels;
    }
    sizeWorld *= scale;
    float sizePx = sizeWorld * pxScale;
    if (minPixels > 0.0f && sizePx < minPixels) {
      sizePx = minPixels;
    }
    if (sunCapPx > 0.0f && h.type[i] == 1 && sizePx > sunCapPx) {
      sizePx = sunCapPx;
    }
    if (maxPixels > 0.0f && sizePx > maxPixels) {
      sizePx = maxPixels;
    }
    sizes[i] = sizePx / point;
  }
}

void computeCOM(const HostArrays &h, float &comX, float &comY, float &comZ,
                float &vcomX, float &vcomY, float &vcomZ) {
  double px = 0.0;
  double py = 0.0;
  double pz = 0.0;
  double totalM = 0.0;
  double sx = 0.0;
  double sy = 0.0;
  double sz = 0.0;
  for (size_t i = 0; i < h.m.size(); ++i) {
    double mi = static_cast<double>(h.m[i]);
    if (mi <= 0.0) continue;
    totalM += mi;
    sx += mi * static_cast<double>(h.x[i]);
    sy += mi * static_cast<double>(h.y[i]);
    sz += mi * static_cast<double>(h.z[i]);
    px += mi * static_cast<double>(h.vx[i]);
    py += mi * static_cast<double>(h.vy[i]);
    pz += mi * static_cast<double>(h.vz[i]);
  }
  if (totalM <= 0.0) {
    comX = comY = comZ = 0.0f;
    vcomX = vcomY = vcomZ = 0.0f;
    return;
  }
  comX = static_cast<float>(sx / totalM);
  comY = static_cast<float>(sy / totalM);
  comZ = static_cast<float>(sz / totalM);
  vcomX = static_cast<float>(px / totalM);
  vcomY = static_cast<float>(py / totalM);
  vcomZ = static_cast<float>(pz / totalM);
}

void updateDynamicColors(const std::vector<float> &baseColors,
                         const std::vector<float> &energies,
                         const std::vector<int> &closeFlags,
                         const std::vector<float> &collisionFlash,
                         bool showEnergyColor,
                         std::vector<float> &outColors) {
  int n = static_cast<int>(baseColors.size() / 4);
  outColors = baseColors;
  if (showEnergyColor && energies.size() == static_cast<size_t>(n)) {
    for (int i = 0; i < n; ++i) {
      if (energies[i] < 0.0f) {
        outColors[4 * i + 0] = 0.2f;
        outColors[4 * i + 1] = 0.4f;
        outColors[4 * i + 2] = 1.0f;
      } else {
        outColors[4 * i + 0] = 1.0f;
        outColors[4 * i + 1] = 0.9f;
        outColors[4 * i + 2] = 0.2f;
      }
    }
  }

  if (!closeFlags.empty()) {
    for (int i = 0; i < n; ++i) {
      if (closeFlags[i]) {
        outColors[4 * i + 0] = 1.0f;
        outColors[4 * i + 1] = 0.1f;
        outColors[4 * i + 2] = 0.1f;
      }
    }
  }

  if (!collisionFlash.empty()) {
    for (int i = 0; i < n; ++i) {
      if (collisionFlash[i] > 0.0f) {
        outColors[4 * i + 0] = 1.0f;
        outColors[4 * i + 1] = 0.15f;
        outColors[4 * i + 2] = 0.05f;
      }
    }
  }
}
