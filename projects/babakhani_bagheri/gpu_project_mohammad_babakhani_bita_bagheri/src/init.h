#pragma once

#include <vector>

struct HostArrays {
  std::vector<float> x;
  std::vector<float> y;
  std::vector<float> z;
  std::vector<float> vx;
  std::vector<float> vy;
  std::vector<float> vz;
  std::vector<float> m;
  std::vector<float> rad;
  std::vector<float> vrad;
  std::vector<int> type;
};

struct InitConfig {
  int n = 4096;
  unsigned int seed = 1;
  float G = 1.0f;
  float massScale = 1.0f;
  float sunMass = 10000.0f;
  float planetMass = 12.0f;
  float cometMass = 1.0f;
  float minorMass = 0.03f;
  bool stableInit = true;
  bool zeroMomentum = true;
  bool realSolarSystem = false;
  int numPlanets = 8;
  bool planetRadiiPreset = true;
  float planetEccMax = 0.03f;
  float planetIncDegMax = 2.0f;
  bool asteroidBelt = true;
  float asteroidBeltRmin = 2.2f;
  float asteroidBeltRmax = 3.4f;
  float asteroidTangentialJitter = 0.02f;
  float asteroidRadialJitter = 0.02f;
  int beltCount = 0;
  int minorCount = 0;
  bool hideMinors = false;
  int numComets = 4;
  float cometEccMin = 0.6f;
  float cometEccMax = 0.95f;
  float cometIncDegMax = 30.0f;
  int slingshotTargetPlanet = -1;
  bool fragmentation = false;
  int maxDebris = 0;
  bool demoCollision = false;
  bool demoSlingshot = false;
  float demoCollisionSpeed = 1.0f;
};

void initSolarSystem(HostArrays &h, const InitConfig &cfg, float &maxRadius);
