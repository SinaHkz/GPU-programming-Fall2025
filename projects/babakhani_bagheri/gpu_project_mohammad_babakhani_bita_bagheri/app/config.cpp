#include "config.h"

#include <cstdio>

const char *kUsage =
    "Usage: ./nbody --N <int> --dt <float> --timeScale <float> --G <float> --eps <float> "
    "--massScale <float> --sunMass <float> --planetMass <float> --cometMass <float> --minorMass <float> "
    "--steps <int> --seed <int> --render 0/1 --trails 0/1 --trailLength <int> --benchmark 0/1 "
    "--collisions 0/1 --collisionRestitution <float> --highlightCloseEncounters 0/1 "
    "--showVelocityVectors 0/1 --showEnergyColor 0/1 --autoFocus 0/1 --centerOnCOM 0/1 --zeroMomentum 0/1 "
    "--autoFit 0/1 --autoFitTarget all|major --worldScale <float> --minZoom <float> --maxZoom <float> "
    "--stableInit 0/1 --realSolarSystem 0/1 --numPlanets <int> --planetRadiiPreset 0/1 "
    "--planetEccMax <float> --planetIncDegMax <float> --asteroidBelt 0/1 "
    "--asteroidBeltRmin <float> --asteroidBeltRmax <float> --asteroidTangentialJitter <float> "
    "--asteroidRadialJitter <float> --beltCount <int> --minorCount <int> --hideMinors 0/1 "
    "--numComets <int> --cometEccMin <float> --cometEccMax <float> "
    "--cometIncDegMax <float> --cometTail 0/1 --cometTailLength <float> --cometTailAlpha <float> "
    "--maxHighlightedCollisions <int> --maxHotCollisions <int> --fragmentsPerCollision <int> --fragmentSpeedSpread <float> --maxDebris <int> "
    "--maxCollisionPairs <int> --maxDebrisActive <int> --fragmentEnergyThreshold <float> "
    "--solarCollisionDemo 0/1 --solarCollision 0/1 "
    "--focusThreshold <float> --cameraLerp <float> --zoomLerp <float> "
    "--initialFocusDelay <float> --initialFocusDelaySeconds <float> --pointSizeScale <float> --visualScale <float> "
    "--planetScale <float> --sunScale <float> --sunVisualScale <float> --planetVisualScale <float> "
    "--minorVisualScale <float> --cometVisualScale <float> --sunVisualBoost <float> --planetVisualBoost <float> "
    "--minorVisualBoost <float> --cometVisualBoost <float> --planetStyle 0/1 "
    "--minPlanetPixels <float> --minSunPixels <float> --minCometPixels <float> "
    "--minMinorPixels <float> --maxMinorPixels <float> "
    "--bloomBoost <float> --bloomThreshold <float> "
    "--velocityVectorScale <float> --arrowHeadLength <float> --arrowHeadWidth <float> "
    "--trailAlpha <float> --velocityLineAlpha <float> --closeLineAlpha <float> "
    "--velocityMassThreshold <float> --collisionFlashDuration <float> --collisionFlashSeconds <float> --collisionFlashScale <float> "
    "--collisionFocus 0/1 --collisionFocusSeconds <float> --collisionFocusCooldownSeconds <float> "
    "--minFocusRadius <float> --maxFocusRadius <float> --minViewRadius <float> --maxViewRadius <float> "
    "--slowOnCollision 0/1 --collisionSlowFactor <float> "
    "--collisionHoldSeconds <float> --collisionSlowSeconds <float> --cometTrailWidth <float> --collisionShockwave 0/1 "
    "--slingshotTargetPlanet <int> --slingshotHighlight 0/1 "
    "--glDebug 0/1 --cudaDebug 0/1 --sanityChecks 0/1 --maxAbsPosClamp <float> "
    "--collisionDtScale <float> --collisionEpsMin <float> "
    "--present 0/1 --presentCollision 0/1 --cinematic 0/1 "
    "--demoCollision 0/1 --demoSlingshot 0/1 --demoCollisionSpeed <float> --fragmentation 0/1 --allowPlanetFragment 0/1 "
    "--allowPlanetAccretion 0/1 --collisionModel merge|elastic --normalizeOrbits 0/1 --orbitSpeedScale <float> "
    "--collisionHotSeconds <float> --fragmentSpeedClamp <float> "
    "--debugCollision 0/1 --printCOM 0/1 --printTotalEnergy 0/1 --printAngularMomentum 0/1\n";

void printUsage() {
  std::fprintf(stderr, "%s", kUsage);
}
