#include "init.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <random>

namespace {
float randf(std::mt19937 &rng, float a, float b) {
  std::uniform_real_distribution<float> dist(a, b);
  return dist(rng);
}

constexpr float kPi = 3.14159265358979323846f;
constexpr float kDegToRad = 3.14159265358979323846f / 180.0f;
}

void initSolarSystem(HostArrays &h, const InitConfig &cfg, float &maxRadius) {
  int n = cfg.n;
  h.x.assign(n, 0.0f);
  h.y.assign(n, 0.0f);
  h.z.assign(n, 0.0f);
  h.vx.assign(n, 0.0f);
  h.vy.assign(n, 0.0f);
  h.vz.assign(n, 0.0f);
  h.m.assign(n, 0.0f);
  h.rad.assign(n, 0.0f);
  h.vrad.assign(n, 0.0f);
  h.type.assign(n, 0);

  maxRadius = 1.0f;
  if (n <= 0) {
    return;
  }

  std::mt19937 rng(cfg.seed);

  int idx = 0;
  float massScale = cfg.massScale;

  auto applyZeroMomentum = [&]() {
    if (!cfg.zeroMomentum) {
      return;
    }
    double px = 0.0;
    double py = 0.0;
    double pz = 0.0;
    double totalM = 0.0;
    for (int i = 0; i < n; ++i) {
      double mi = static_cast<double>(h.m[i]);
      if (mi <= 0.0) {
        continue;
      }
      totalM += mi;
      px += mi * static_cast<double>(h.vx[i]);
      py += mi * static_cast<double>(h.vy[i]);
      pz += mi * static_cast<double>(h.vz[i]);
    }
    if (totalM <= 0.0) {
      return;
    }
    float vcomx = static_cast<float>(px / totalM);
    float vcomy = static_cast<float>(py / totalM);
    float vcomz = static_cast<float>(pz / totalM);
    for (int i = 0; i < n; ++i) {
      h.vx[i] -= vcomx;
      h.vy[i] -= vcomy;
      h.vz[i] -= vcomz;
    }
  };

  if (cfg.demoCollision) {
    float mass = cfg.cometMass * massScale;
    float radius = 0.15f;
    float speed = cfg.demoCollisionSpeed;
    int count = std::min(n, 8);
    std::vector<std::array<float, 3>> positions = {
        { -2.0f, 0.0f, 0.0f },
        {  2.0f, 0.0f, 0.0f },
        {  0.0f, -2.0f, 0.0f },
        {  0.0f,  2.0f, 0.0f },
        { -1.4f, -1.4f, 0.0f },
        {  1.4f,  1.4f, 0.0f },
        { -1.4f,  1.4f, 0.0f },
        {  1.4f, -1.4f, 0.0f }
    };
    for (int i = 0; i < count; ++i) {
      float px = positions[i][0];
      float py = positions[i][1];
      float pz = positions[i][2];
      float dx = -px;
      float dy = -py;
      float dz = -pz;
      float len = std::sqrt(dx * dx + dy * dy + dz * dz) + 1e-6f;
      dx /= len;
      dy /= len;
      dz /= len;

      h.x[i] = px;
      h.y[i] = py;
      h.z[i] = pz;
      h.vx[i] = dx * speed;
      h.vy[i] = dy * speed;
      h.vz[i] = dz * speed;
      h.m[i] = mass;
      h.rad[i] = radius;
      h.vrad[i] = 0.08f;
      h.type[i] = 2;
      maxRadius = std::max(maxRadius, len);
    }
    if (cfg.fragmentation && n > count) {
      for (int i = count; i < n; ++i) {
        h.x[i] = 0.0f;
        h.y[i] = 0.0f;
        h.z[i] = 0.0f;
        h.vx[i] = 0.0f;
        h.vy[i] = 0.0f;
        h.vz[i] = 0.0f;
        h.m[i] = 0.0f;
        h.rad[i] = 0.0f;
        h.vrad[i] = 0.0f;
        h.type[i] = 3;
      }
    }
    applyZeroMomentum();
    return;
  }
  // Sun
  h.x[idx] = 0.0f;
  h.y[idx] = 0.0f;
  h.z[idx] = 0.0f;
  h.vx[idx] = 0.0f;
  h.vy[idx] = 0.0f;
  h.vz[idx] = 0.0f;
  h.m[idx] = cfg.sunMass * massScale;
  h.rad[idx] = 0.4f;
  h.vrad[idx] = 1.0f;
  h.type[idx] = 0;
  idx++;

  const float defaultPlanetRadii[] = {0.7f, 1.0f, 1.4f, 2.1f, 3.6f, 5.0f, 6.5f, 7.8f};
  const float defaultPlanetMasses[] = {0.25f, 0.8f, 1.0f, 0.3f, 5.0f, 4.0f, 2.0f, 2.5f};
  const float realPlanetRadii[] = {0.7f, 1.0f, 1.4f, 2.1f, 3.6f, 5.0f, 6.5f, 7.8f};
  const float realPlanetMassRatios[] = {1.7e-7f, 2.4e-6f, 3.0e-6f, 3.2e-7f, 9.5e-4f, 2.8e-4f, 4.4e-5f, 5.1e-5f};
  const float planetVisualRadii[] = {0.12f, 0.18f, 0.18f, 0.14f, 0.35f, 0.30f, 0.22f, 0.22f};
  const float *planetRadii = cfg.realSolarSystem ? realPlanetRadii : defaultPlanetRadii;
  const float *planetMasses = cfg.realSolarSystem ? realPlanetMassRatios : defaultPlanetMasses;
  int presetPlanets = static_cast<int>(sizeof(defaultPlanetRadii) / sizeof(float));
  int numPlanets = std::min(std::max(0, cfg.numPlanets), presetPlanets);
  numPlanets = std::min(numPlanets, n - 1);
  int numComets = std::min(std::max(0, cfg.numComets), n - 1 - numPlanets);
  int remaining = n - 1 - numPlanets - numComets;
  int reserveDebris = (cfg.fragmentation && cfg.maxDebris > 0) ? std::min(cfg.maxDebris, std::max(0, remaining / 3)) : 0;
  int desiredBelt = cfg.beltCount > 0 ? cfg.beltCount : std::max(0, remaining / 2);
  int desiredMinor = cfg.minorCount > 0 ? cfg.minorCount : std::max(0, remaining - desiredBelt - reserveDebris);
  int numBelt = cfg.hideMinors ? 0 : std::min(desiredBelt, std::max(0, remaining - reserveDebris));
  int numMinors = cfg.hideMinors ? 0 : std::max(0, remaining - reserveDebris - numBelt);

  int targetPlanet = -1;
  for (int p = 0; p < numPlanets; ++p) {
    if (std::abs(planetRadii[p] - 3.6f) < 0.3f) {
      targetPlanet = p;
      break;
    }
  }
  float thetaOffset = 0.0f;
  if (cfg.stableInit && targetPlanet >= 0) {
    float base = 2.0f * kPi * static_cast<float>(targetPlanet) / static_cast<float>(numPlanets);
    thetaOffset = -base;
  }

  for (int p = 0; p < numPlanets; ++p) {
    float R = cfg.planetRadiiPreset ? planetRadii[p] : (1.2f + 0.9f * static_cast<float>(p));
    float theta = 0.0f;
    if (cfg.stableInit) {
      float base = 2.0f * kPi * static_cast<float>(p) / static_cast<float>(numPlanets);
      theta = base + thetaOffset;
    } else {
      theta = randf(rng, 0.0f, 2.0f * kPi);
    }
    float ecc = cfg.planetEccMax > 0.0f ? randf(rng, 0.0f, cfg.planetEccMax) : 0.0f;
    float inc = cfg.planetIncDegMax > 0.0f ? randf(rng, 0.0f, cfg.planetIncDegMax) * kDegToRad : 0.0f;
    float r = R * (1.0f - ecc);
    float v = std::sqrt(cfg.G * cfg.sunMass * massScale * (1.0f + ecc) / (R * (1.0f - ecc)));
    float x = r * std::cos(theta);
    float y = r * std::sin(theta);
    float z = 0.0f;
    float vx = -v * std::sin(theta);
    float vy = v * std::cos(theta);
    float vz = 0.0f;

    float cosi = std::cos(inc);
    float sini = std::sin(inc);
    float y2 = y * cosi - z * sini;
    float z2 = y * sini + z * cosi;
    float vy2 = vy * cosi - vz * sini;
    float vz2 = vy * sini + vz * cosi;
    y = y2;
    z = z2;
    vy = vy2;
    vz = vz2;

    h.x[idx] = x;
    h.y[idx] = y;
    h.z[idx] = z;
    h.vx[idx] = vx;
    h.vy[idx] = vy;
    h.vz[idx] = vz;
    if (cfg.realSolarSystem) {
      h.m[idx] = cfg.sunMass * planetMasses[p] * massScale;
    } else {
      h.m[idx] = cfg.planetMass * planetMasses[p] * massScale;
    }
    h.rad[idx] = 0.12f + 0.02f * static_cast<float>(p);
    if (p < 8) {
      h.vrad[idx] = planetVisualRadii[p];
    } else {
      h.vrad[idx] = 0.18f;
    }
    h.type[idx] = 1;
    maxRadius = std::max(maxRadius, R);
    idx++;
  }

  for (int c = 0; c < numComets; ++c) {
    float e = randf(rng, cfg.cometEccMin, cfg.cometEccMax);
    float q = randf(rng, 1.5f, 4.5f);
    if (c == 0 && (cfg.demoSlingshot || cfg.slingshotTargetPlanet >= 0) && numPlanets > 0) {
      int tp = cfg.slingshotTargetPlanet;
      if (tp < 0 || tp >= numPlanets) {
        tp = std::min(numPlanets - 1, std::max(0, targetPlanet));
      }
      q = planetRadii[tp] * 1.05f;
    }
    float a = q / (1.0f - e);
    float rAp = a * (1.0f + e);
    float vAp = std::sqrt(cfg.G * cfg.sunMass * massScale * (1.0f - e) / (a * (1.0f + e)));

    float x = -rAp;
    float y = 0.0f;
    float z = 0.0f;
    float vx = 0.0f;
    float vy = vAp;
    float vz = 0.0f;

    float inc = cfg.cometIncDegMax > 0.0f ? randf(rng, 0.0f, cfg.cometIncDegMax) * kDegToRad : 0.0f;
    float phi = randf(rng, 0.0f, 2.0f * kPi);
    float cosphi = std::cos(phi);
    float sinphi = std::sin(phi);
    float x2 = x * cosphi - y * sinphi;
    float y2 = x * sinphi + y * cosphi;
    float vx2 = vx * cosphi - vy * sinphi;
    float vy2 = vx * sinphi + vy * cosphi;

    float cosi = std::cos(inc);
    float sini = std::sin(inc);
    float y3 = y2 * cosi - z * sini;
    float z3 = y2 * sini + z * cosi;
    float vy3 = vy2 * cosi - vz * sini;
    float vz3 = vy2 * sini + vz * cosi;

    h.x[idx] = x2;
    h.y[idx] = y3;
    h.z[idx] = z3;
    h.vx[idx] = vx2;
    h.vy[idx] = vy3;
    h.vz[idx] = vz3;
    h.m[idx] = cfg.cometMass * massScale;
    h.rad[idx] = 0.06f;
    h.vrad[idx] = 0.06f;
    h.type[idx] = 2;
    maxRadius = std::max(maxRadius, rAp);
    idx++;
  }

  for (int i = 0; i < numBelt; ++i) {
    float R = cfg.asteroidBelt ? randf(rng, cfg.asteroidBeltRmin, cfg.asteroidBeltRmax)
                               : randf(rng, 6.5f, 10.5f);
    float theta = randf(rng, 0.0f, 2.0f * kPi);
    float z = randf(rng, -0.03f, 0.03f);
    float baseV = std::sqrt(cfg.G * cfg.sunMass * massScale / R);
    float tangential = 1.0f + randf(rng, -cfg.asteroidTangentialJitter, cfg.asteroidTangentialJitter);
    float radial = randf(rng, -cfg.asteroidRadialJitter, cfg.asteroidRadialJitter);
    float vx = -baseV * tangential * std::sin(theta) + radial * baseV * std::cos(theta);
    float vy = baseV * tangential * std::cos(theta) + radial * baseV * std::sin(theta);

    h.x[idx] = R * std::cos(theta);
    h.y[idx] = R * std::sin(theta);
    h.z[idx] = z;
    h.vx[idx] = vx;
    h.vy[idx] = vy;
    h.vz[idx] = 0.0f;
    float jitter = randf(rng, 0.6f, 1.4f);
    float minorBase = cfg.realSolarSystem ? cfg.sunMass * 1.0e-7f : cfg.minorMass;
    h.m[idx] = minorBase * jitter * massScale;
    h.rad[idx] = 0.02f;
    h.vrad[idx] = 0.04f;
    h.type[idx] = 3;
    maxRadius = std::max(maxRadius, R);
    idx++;
  }

  for (int i = 0; i < numMinors; ++i) {
    float R = randf(rng, 6.5f, 10.5f);
    float theta = randf(rng, 0.0f, 2.0f * kPi);
    float z = randf(rng, -0.05f, 0.05f);
    float baseV = std::sqrt(cfg.G * cfg.sunMass * massScale / R);
    float v = baseV * (1.0f + randf(rng, -0.02f, 0.02f));
    h.x[idx] = R * std::cos(theta);
    h.y[idx] = R * std::sin(theta);
    h.z[idx] = z;
    h.vx[idx] = -v * std::sin(theta);
    h.vy[idx] = v * std::cos(theta);
    h.vz[idx] = 0.0f;
    float jitter = randf(rng, 0.6f, 1.4f);
    float minorBase = cfg.realSolarSystem ? cfg.sunMass * 1.0e-7f : cfg.minorMass;
    h.m[idx] = minorBase * jitter * massScale;
    h.rad[idx] = 0.02f;
    h.vrad[idx] = 0.04f;
    h.type[idx] = 3;
    maxRadius = std::max(maxRadius, R);
    idx++;
  }
  for (int i = 0; i < reserveDebris; ++i) {
    h.x[idx] = 0.0f;
    h.y[idx] = 0.0f;
    h.z[idx] = 0.0f;
    h.vx[idx] = 0.0f;
    h.vy[idx] = 0.0f;
    h.vz[idx] = 0.0f;
    h.m[idx] = 0.0f;
    h.rad[idx] = 0.0f;
    h.vrad[idx] = 0.0f;
    h.type[idx] = 3;
    idx++;
  }

  applyZeroMomentum();
}
