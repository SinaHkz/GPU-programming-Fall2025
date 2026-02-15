#include "cli.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <string>

bool parseArgs(int argc, char **argv, Config &cfg) {
  bool setTimeScale = false;
  bool setHighlight = false;
  bool setAutoFocus = false;
  bool setEnergyColor = false;
  bool setVelocityVectors = false;
  bool setTrails = false;
  bool setTrailLength = false;
  bool setCollisions = false;
  bool setDebugCollision = false;
  bool setN = false;
  bool setDt = false;
  bool setFragmentation = false;
  bool setNumPlanets = false;
  bool setAsteroidBelt = false;
  bool setNumComets = false;
  bool setCometTail = false;
  bool setMinPlanetPixels = false;
  bool setMinSunPixels = false;

  for (int i = 1; i < argc; ++i) {
    std::string arg = argv[i];
    auto requireValue = [&](const char *name) -> const char * {
      if (i + 1 >= argc) {
        std::fprintf(stderr, "Missing value for %s\n", name);
        return nullptr;
      }
      return argv[++i];
    };

    if (arg == "--help" || arg == "-h") {
      cfg.showHelp = true;
    } else if (arg == "--N") {
      const char *val = requireValue("--N");
      if (!val) return false;
      cfg.n = std::atoi(val);
      setN = true;
    } else if (arg == "--dt") {
      const char *val = requireValue("--dt");
      if (!val) return false;
      cfg.dt = std::atof(val);
      setDt = true;
    } else if (arg == "--timeScale") {
      const char *val = requireValue("--timeScale");
      if (!val) return false;
      cfg.timeScale = std::atof(val);
      setTimeScale = true;
    } else if (arg == "--G") {
      const char *val = requireValue("--G");
      if (!val) return false;
      cfg.G = std::atof(val);
    } else if (arg == "--eps") {
      const char *val = requireValue("--eps");
      if (!val) return false;
      cfg.eps = std::atof(val);
    } else if (arg == "--massScale") {
      const char *val = requireValue("--massScale");
      if (!val) return false;
      cfg.massScale = std::atof(val);
    } else if (arg == "--sunMass") {
      const char *val = requireValue("--sunMass");
      if (!val) return false;
      cfg.sunMass = std::atof(val);
    } else if (arg == "--planetMass") {
      const char *val = requireValue("--planetMass");
      if (!val) return false;
      cfg.planetMass = std::atof(val);
    } else if (arg == "--cometMass") {
      const char *val = requireValue("--cometMass");
      if (!val) return false;
      cfg.cometMass = std::atof(val);
    } else if (arg == "--minorMass") {
      const char *val = requireValue("--minorMass");
      if (!val) return false;
      cfg.minorMass = std::atof(val);
    } else if (arg == "--steps") {
      const char *val = requireValue("--steps");
      if (!val) return false;
      cfg.steps = std::atoi(val);
    } else if (arg == "--seed") {
      const char *val = requireValue("--seed");
      if (!val) return false;
      cfg.seed = static_cast<unsigned int>(std::atoi(val));
    } else if (arg == "--render") {
      const char *val = requireValue("--render");
      if (!val) return false;
      cfg.render = std::atoi(val) != 0;
    } else if (arg == "--trails") {
      const char *val = requireValue("--trails");
      if (!val) return false;
      cfg.trails = std::atoi(val) != 0;
      setTrails = true;
    } else if (arg == "--trailLength") {
      const char *val = requireValue("--trailLength");
      if (!val) return false;
      cfg.trailLength = std::atoi(val);
      setTrailLength = true;
    } else if (arg == "--benchmark") {
      const char *val = requireValue("--benchmark");
      if (!val) return false;
      cfg.benchmark = std::atoi(val) != 0;
    } else if (arg == "--collisions") {
      const char *val = requireValue("--collisions");
      if (!val) return false;
      cfg.collisions = std::atoi(val) != 0;
      setCollisions = true;
    } else if (arg == "--collisionRestitution") {
      const char *val = requireValue("--collisionRestitution");
      if (!val) return false;
      cfg.collisionRestitution = std::atof(val);
    } else if (arg == "--highlightCloseEncounters") {
      const char *val = requireValue("--highlightCloseEncounters");
      if (!val) return false;
      cfg.highlightCloseEncounters = std::atoi(val) != 0;
      setHighlight = true;
    } else if (arg == "--showVelocityVectors") {
      const char *val = requireValue("--showVelocityVectors");
      if (!val) return false;
      cfg.showVelocityVectors = std::atoi(val) != 0;
      setVelocityVectors = true;
    } else if (arg == "--showEnergyColor") {
      const char *val = requireValue("--showEnergyColor");
      if (!val) return false;
      cfg.showEnergyColor = std::atoi(val) != 0;
      setEnergyColor = true;
    } else if (arg == "--autoFocus") {
      const char *val = requireValue("--autoFocus");
      if (!val) return false;
      cfg.autoFocus = std::atoi(val) != 0;
      setAutoFocus = true;
    } else if (arg == "--centerOnCOM") {
      const char *val = requireValue("--centerOnCOM");
      if (!val) return false;
      cfg.centerOnCOM = std::atoi(val) != 0;
    } else if (arg == "--zeroMomentum") {
      const char *val = requireValue("--zeroMomentum");
      if (!val) return false;
      cfg.zeroMomentum = std::atoi(val) != 0;
    } else if (arg == "--autoFit") {
      const char *val = requireValue("--autoFit");
      if (!val) return false;
      cfg.autoFit = std::atoi(val) != 0;
    } else if (arg == "--autoFitTarget") {
      const char *val = requireValue("--autoFitTarget");
      if (!val) return false;
      std::string mode = val;
      if (mode == "all") {
        cfg.autoFitMajorOnly = false;
      } else if (mode == "major") {
        cfg.autoFitMajorOnly = true;
      } else {
        std::fprintf(stderr, "Invalid --autoFitTarget (use all|major)\n");
        return false;
      }
    } else if (arg == "--worldScale") {
      const char *val = requireValue("--worldScale");
      if (!val) return false;
      cfg.worldScale = std::atof(val);
      cfg.worldScaleSet = true;
    } else if (arg == "--minZoom") {
      const char *val = requireValue("--minZoom");
      if (!val) return false;
      cfg.minZoom = std::atof(val);
      cfg.minZoomSet = true;
    } else if (arg == "--maxZoom") {
      const char *val = requireValue("--maxZoom");
      if (!val) return false;
      cfg.maxZoom = std::atof(val);
      cfg.maxZoomSet = true;
    } else if (arg == "--stableInit") {
      const char *val = requireValue("--stableInit");
      if (!val) return false;
      cfg.stableInit = std::atoi(val) != 0;
    } else if (arg == "--realSolarSystem") {
      const char *val = requireValue("--realSolarSystem");
      if (!val) return false;
      cfg.realSolarSystem = std::atoi(val) != 0;
    } else if (arg == "--numPlanets") {
      const char *val = requireValue("--numPlanets");
      if (!val) return false;
      cfg.numPlanets = std::atoi(val);
      setNumPlanets = true;
    } else if (arg == "--planetRadiiPreset") {
      const char *val = requireValue("--planetRadiiPreset");
      if (!val) return false;
      cfg.planetRadiiPreset = std::atoi(val) != 0;
    } else if (arg == "--planetEccMax") {
      const char *val = requireValue("--planetEccMax");
      if (!val) return false;
      cfg.planetEccMax = std::atof(val);
    } else if (arg == "--planetIncDegMax") {
      const char *val = requireValue("--planetIncDegMax");
      if (!val) return false;
      cfg.planetIncDegMax = std::atof(val);
    } else if (arg == "--asteroidBelt") {
      const char *val = requireValue("--asteroidBelt");
      if (!val) return false;
      cfg.asteroidBelt = std::atoi(val) != 0;
      setAsteroidBelt = true;
    } else if (arg == "--asteroidBeltRmin") {
      const char *val = requireValue("--asteroidBeltRmin");
      if (!val) return false;
      cfg.asteroidBeltRmin = std::atof(val);
    } else if (arg == "--asteroidBeltRmax") {
      const char *val = requireValue("--asteroidBeltRmax");
      if (!val) return false;
      cfg.asteroidBeltRmax = std::atof(val);
    } else if (arg == "--asteroidTangentialJitter") {
      const char *val = requireValue("--asteroidTangentialJitter");
      if (!val) return false;
      cfg.asteroidTangentialJitter = std::atof(val);
    } else if (arg == "--asteroidRadialJitter") {
      const char *val = requireValue("--asteroidRadialJitter");
      if (!val) return false;
      cfg.asteroidRadialJitter = std::atof(val);
    } else if (arg == "--beltCount") {
      const char *val = requireValue("--beltCount");
      if (!val) return false;
      cfg.beltCount = std::atoi(val);
    } else if (arg == "--minorCount") {
      const char *val = requireValue("--minorCount");
      if (!val) return false;
      cfg.minorCount = std::atoi(val);
    } else if (arg == "--hideMinors") {
      const char *val = requireValue("--hideMinors");
      if (!val) return false;
      cfg.hideMinors = std::atoi(val) != 0;
    } else if (arg == "--numComets") {
      const char *val = requireValue("--numComets");
      if (!val) return false;
      cfg.numComets = std::atoi(val);
      setNumComets = true;
    } else if (arg == "--cometEccMin") {
      const char *val = requireValue("--cometEccMin");
      if (!val) return false;
      cfg.cometEccMin = std::atof(val);
    } else if (arg == "--cometEccMax") {
      const char *val = requireValue("--cometEccMax");
      if (!val) return false;
      cfg.cometEccMax = std::atof(val);
    } else if (arg == "--cometIncDegMax") {
      const char *val = requireValue("--cometIncDegMax");
      if (!val) return false;
      cfg.cometIncDegMax = std::atof(val);
    } else if (arg == "--cometTail") {
      const char *val = requireValue("--cometTail");
      if (!val) return false;
      cfg.cometTail = std::atoi(val) != 0;
      setCometTail = true;
    } else if (arg == "--cometTailLength") {
      const char *val = requireValue("--cometTailLength");
      if (!val) return false;
      cfg.cometTailLength = std::atof(val);
    } else if (arg == "--cometTailAlpha") {
      const char *val = requireValue("--cometTailAlpha");
      if (!val) return false;
      cfg.cometTailAlpha = std::atof(val);
    } else if (arg == "--planetStyle") {
      const char *val = requireValue("--planetStyle");
      if (!val) return false;
      cfg.planetStyle = std::atoi(val) != 0;
    } else if (arg == "--maxHighlightedCollisions") {
      const char *val = requireValue("--maxHighlightedCollisions");
      if (!val) return false;
      cfg.maxHighlightedCollisions = std::atoi(val);
    } else if (arg == "--maxHotCollisions") {
      const char *val = requireValue("--maxHotCollisions");
      if (!val) return false;
      cfg.maxHighlightedCollisions = std::atoi(val);
    } else if (arg == "--fragmentsPerCollision") {
      const char *val = requireValue("--fragmentsPerCollision");
      if (!val) return false;
      cfg.fragmentsPerCollision = std::atoi(val);
    } else if (arg == "--fragmentSpread") {
      const char *val = requireValue("--fragmentSpread");
      if (!val) return false;
      cfg.fragmentSpeedSpread = std::atof(val);
    } else if (arg == "--fragmentSpeedSpread") {
      const char *val = requireValue("--fragmentSpeedSpread");
      if (!val) return false;
      cfg.fragmentSpeedSpread = std::atof(val);
    } else if (arg == "--fragmentSpeedClamp") {
      const char *val = requireValue("--fragmentSpeedClamp");
      if (!val) return false;
      cfg.fragmentSpeedClamp = std::atof(val);
    } else if (arg == "--maxDebris") {
      const char *val = requireValue("--maxDebris");
      if (!val) return false;
      cfg.maxDebris = std::atoi(val);
    } else if (arg == "--maxCollisionPairs") {
      const char *val = requireValue("--maxCollisionPairs");
      if (!val) return false;
      cfg.maxCollisionPairs = std::atoi(val);
    } else if (arg == "--maxDebrisActive") {
      const char *val = requireValue("--maxDebrisActive");
      if (!val) return false;
      cfg.maxDebrisActive = std::atoi(val);
    } else if (arg == "--fragmentEnergyThreshold") {
      const char *val = requireValue("--fragmentEnergyThreshold");
      if (!val) return false;
      cfg.fragmentEnergyThreshold = std::atof(val);
    } else if (arg == "--allowPlanetFragment") {
      const char *val = requireValue("--allowPlanetFragment");
      if (!val) return false;
      cfg.allowPlanetFragment = std::atoi(val) != 0;
    } else if (arg == "--allowPlanetAccretion") {
      const char *val = requireValue("--allowPlanetAccretion");
      if (!val) return false;
      cfg.allowPlanetAccretion = std::atoi(val) != 0;
    } else if (arg == "--solarCollisionDemo") {
      const char *val = requireValue("--solarCollisionDemo");
      if (!val) return false;
      cfg.solarCollisionDemo = std::atoi(val) != 0;
    } else if (arg == "--solarCollision") {
      const char *val = requireValue("--solarCollision");
      if (!val) return false;
      cfg.solarCollision = std::atoi(val) != 0;
    } else if (arg == "--focusThreshold") {
      const char *val = requireValue("--focusThreshold");
      if (!val) return false;
      cfg.focusThreshold = std::atof(val);
    } else if (arg == "--cameraLerp") {
      const char *val = requireValue("--cameraLerp");
      if (!val) return false;
      cfg.cameraLerp = std::atof(val);
    } else if (arg == "--zoomLerp") {
      const char *val = requireValue("--zoomLerp");
      if (!val) return false;
      cfg.zoomLerp = std::atof(val);
    } else if (arg == "--normalizeOrbits") {
      const char *val = requireValue("--normalizeOrbits");
      if (!val) return false;
      cfg.normalizeOrbits = std::atoi(val) != 0;
    } else if (arg == "--orbitSpeedScale") {
      const char *val = requireValue("--orbitSpeedScale");
      if (!val) return false;
      cfg.orbitSpeedScale = std::atof(val);
    } else if (arg == "--collisionModel") {
      const char *val = requireValue("--collisionModel");
      if (!val) return false;
      std::string mode = val;
      if (mode == "merge") {
        cfg.collisionModel = 0;
      } else if (mode == "elastic") {
        cfg.collisionModel = 1;
      } else {
        std::fprintf(stderr, "Invalid --collisionModel (use merge|elastic)\n");
        return false;
      }
    } else if (arg == "--initialFocusDelay") {
      const char *val = requireValue("--initialFocusDelay");
      if (!val) return false;
      cfg.initialFocusDelay = std::atof(val);
    } else if (arg == "--initialFocusDelaySeconds") {
      const char *val = requireValue("--initialFocusDelaySeconds");
      if (!val) return false;
      cfg.initialFocusDelay = std::atof(val);
    } else if (arg == "--pointSizeScale") {
      const char *val = requireValue("--pointSizeScale");
      if (!val) return false;
      cfg.pointSizeScale = std::atof(val);
    } else if (arg == "--visualScale") {
      const char *val = requireValue("--visualScale");
      if (!val) return false;
      cfg.visualScale = std::atof(val);
    } else if (arg == "--planetScale") {
      const char *val = requireValue("--planetScale");
      if (!val) return false;
      cfg.planetScale = std::atof(val);
      cfg.planetVisualScale = cfg.planetScale;
    } else if (arg == "--sunScale") {
      const char *val = requireValue("--sunScale");
      if (!val) return false;
      cfg.sunScale = std::atof(val);
      cfg.sunVisualScale = cfg.sunScale;
    } else if (arg == "--sunVisualScale") {
      const char *val = requireValue("--sunVisualScale");
      if (!val) return false;
      cfg.sunVisualScale = std::atof(val);
    } else if (arg == "--sunVisualBoost") {
      const char *val = requireValue("--sunVisualBoost");
      if (!val) return false;
      cfg.sunVisualBoost = std::atof(val);
    } else if (arg == "--planetVisualScale") {
      const char *val = requireValue("--visualPlanetScale");
      if (!val) return false;
      cfg.planetVisualScale = std::atof(val);
    } else if (arg == "--planetVisualBoost") {
      const char *val = requireValue("--planetVisualBoost");
      if (!val) return false;
      cfg.planetVisualBoost = std::atof(val);
    } else if (arg == "--minorVisualScale") {
      const char *val = requireValue("--minorVisualScale");
      if (!val) return false;
      cfg.minorVisualScale = std::atof(val);
    } else if (arg == "--minorVisualBoost") {
      const char *val = requireValue("--minorVisualBoost");
      if (!val) return false;
      cfg.minorVisualBoost = std::atof(val);
    } else if (arg == "--cometVisualScale") {
      const char *val = requireValue("--cometVisualScale");
      if (!val) return false;
      cfg.cometVisualScale = std::atof(val);
    } else if (arg == "--cometVisualBoost") {
      const char *val = requireValue("--cometVisualBoost");
      if (!val) return false;
      cfg.cometVisualBoost = std::atof(val);
    } else if (arg == "--visualPlanetScale") {
      const char *val = requireValue("--visualPlanetScale");
      if (!val) return false;
      cfg.planetVisualScale = std::atof(val);
    } else if (arg == "--visualSunScale") {
      const char *val = requireValue("--visualSunScale");
      if (!val) return false;
      cfg.sunVisualScale = std::atof(val);
    } else if (arg == "--visualMinorScale") {
      const char *val = requireValue("--visualMinorScale");
      if (!val) return false;
      cfg.minorVisualScale = std::atof(val);
    } else if (arg == "--minPlanetPixels") {
      const char *val = requireValue("--minPlanetPixels");
      if (!val) return false;
      cfg.minPlanetPixels = std::atof(val);
      setMinPlanetPixels = true;
    } else if (arg == "--minSunPixels") {
      const char *val = requireValue("--minSunPixels");
      if (!val) return false;
      cfg.minSunPixels = std::atof(val);
      setMinSunPixels = true;
    } else if (arg == "--minCometPixels") {
      const char *val = requireValue("--minCometPixels");
      if (!val) return false;
      cfg.minCometPixels = std::atof(val);
    } else if (arg == "--maxMinorPixels") {
      const char *val = requireValue("--maxMinorPixels");
      if (!val) return false;
      cfg.maxMinorPixels = std::atof(val);
    } else if (arg == "--minMinorPixels") {
      const char *val = requireValue("--minMinorPixels");
      if (!val) return false;
      cfg.minMinorPixels = std::atof(val);
    } else if (arg == "--bloomBoost") {
      const char *val = requireValue("--bloomBoost");
      if (!val) return false;
      cfg.bloomBoost = std::atof(val);
    } else if (arg == "--bloomThreshold") {
      const char *val = requireValue("--bloomThreshold");
      if (!val) return false;
      cfg.bloomThreshold = std::atof(val);
    } else if (arg == "--velocityVectorScale") {
      const char *val = requireValue("--velocityVectorScale");
      if (!val) return false;
      cfg.velocityVectorScale = std::atof(val);
    } else if (arg == "--arrowHeadLength") {
      const char *val = requireValue("--arrowHeadLength");
      if (!val) return false;
      cfg.arrowHeadLength = std::atof(val);
    } else if (arg == "--arrowHeadWidth") {
      const char *val = requireValue("--arrowHeadWidth");
      if (!val) return false;
      cfg.arrowHeadWidth = std::atof(val);
    } else if (arg == "--trailAlpha") {
      const char *val = requireValue("--trailAlpha");
      if (!val) return false;
      cfg.trailAlpha = std::atof(val);
    } else if (arg == "--velocityLineAlpha") {
      const char *val = requireValue("--velocityLineAlpha");
      if (!val) return false;
      cfg.velocityLineAlpha = std::atof(val);
    } else if (arg == "--closeLineAlpha") {
      const char *val = requireValue("--closeLineAlpha");
      if (!val) return false;
      cfg.closeLineAlpha = std::atof(val);
    } else if (arg == "--velocityMassThreshold") {
      const char *val = requireValue("--velocityMassThreshold");
      if (!val) return false;
      cfg.velocityMassThreshold = std::atof(val);
    } else if (arg == "--collisionFlashDuration") {
      const char *val = requireValue("--collisionFlashDuration");
      if (!val) return false;
      cfg.collisionFlashDuration = std::atof(val);
    } else if (arg == "--collisionFlashSeconds") {
      const char *val = requireValue("--collisionFlashSeconds");
      if (!val) return false;
      cfg.collisionFlashDuration = std::atof(val);
      cfg.collisionHotSeconds = cfg.collisionFlashDuration;
    } else if (arg == "--collisionFlashScale") {
      const char *val = requireValue("--collisionFlashScale");
      if (!val) return false;
      cfg.collisionFlashScale = std::atof(val);
    } else if (arg == "--collisionHotSeconds") {
      const char *val = requireValue("--collisionHotSeconds");
      if (!val) return false;
      cfg.collisionHotSeconds = std::atof(val);
      cfg.collisionFlashDuration = cfg.collisionHotSeconds;
    } else if (arg == "--collisionFocus") {
      const char *val = requireValue("--collisionFocus");
      if (!val) return false;
      cfg.collisionFocus = std::atoi(val) != 0;
    } else if (arg == "--collisionFocusSeconds") {
      const char *val = requireValue("--collisionFocusSeconds");
      if (!val) return false;
      cfg.collisionFocusSeconds = std::atof(val);
    } else if (arg == "--collisionFocusCooldownSeconds") {
      const char *val = requireValue("--collisionFocusCooldownSeconds");
      if (!val) return false;
      cfg.collisionFocusCooldownSeconds = std::atof(val);
    } else if (arg == "--minFocusRadius") {
      const char *val = requireValue("--minFocusRadius");
      if (!val) return false;
      cfg.minFocusRadius = std::atof(val);
    } else if (arg == "--maxFocusRadius") {
      const char *val = requireValue("--maxFocusRadius");
      if (!val) return false;
      cfg.maxFocusRadius = std::atof(val);
    } else if (arg == "--minViewRadius") {
      const char *val = requireValue("--minViewRadius");
      if (!val) return false;
      cfg.minViewRadius = std::atof(val);
    } else if (arg == "--maxViewRadius") {
      const char *val = requireValue("--maxViewRadius");
      if (!val) return false;
      cfg.maxViewRadius = std::atof(val);
    } else if (arg == "--slowOnCollision") {
      const char *val = requireValue("--slowOnCollision");
      if (!val) return false;
      cfg.slowOnCollision = std::atoi(val) != 0;
    } else if (arg == "--collisionSlowFactor") {
      const char *val = requireValue("--collisionSlowFactor");
      if (!val) return false;
      cfg.collisionSlowFactor = std::atof(val);
    } else if (arg == "--collisionHoldSeconds") {
      const char *val = requireValue("--collisionHoldSeconds");
      if (!val) return false;
      cfg.collisionHoldSeconds = std::atof(val);
    } else if (arg == "--collisionSlowSeconds") {
      const char *val = requireValue("--collisionSlowSeconds");
      if (!val) return false;
      cfg.collisionSlowSeconds = std::atof(val);
    } else if (arg == "--glDebug") {
      const char *val = requireValue("--glDebug");
      if (!val) return false;
      cfg.glDebug = std::atoi(val) != 0;
    } else if (arg == "--cudaDebug") {
      const char *val = requireValue("--cudaDebug");
      if (!val) return false;
      cfg.cudaDebug = std::atoi(val) != 0;
    } else if (arg == "--sanityChecks") {
      const char *val = requireValue("--sanityChecks");
      if (!val) return false;
      cfg.sanityChecks = std::atoi(val) != 0;
    } else if (arg == "--maxAbsPosClamp") {
      const char *val = requireValue("--maxAbsPosClamp");
      if (!val) return false;
      cfg.maxAbsPosClamp = std::atof(val);
    } else if (arg == "--collisionDtScale") {
      const char *val = requireValue("--collisionDtScale");
      if (!val) return false;
      cfg.collisionDtScale = std::atof(val);
    } else if (arg == "--collisionEpsMin") {
      const char *val = requireValue("--collisionEpsMin");
      if (!val) return false;
      cfg.collisionEpsMin = std::atof(val);
    } else if (arg == "--cometTrailWidth") {
      const char *val = requireValue("--cometTrailWidth");
      if (!val) return false;
      cfg.cometTrailWidth = std::atof(val);
    } else if (arg == "--collisionShockwave") {
      const char *val = requireValue("--collisionShockwave");
      if (!val) return false;
      cfg.collisionShockwave = std::atoi(val) != 0;
    } else if (arg == "--present") {
      const char *val = requireValue("--present");
      if (!val) return false;
      cfg.present = std::atoi(val) != 0;
    } else if (arg == "--presentCollision") {
      const char *val = requireValue("--presentCollision");
      if (!val) return false;
      cfg.presentCollision = std::atoi(val) != 0;
    } else if (arg == "--cinematic") {
      const char *val = requireValue("--cinematic");
      if (!val) return false;
      cfg.cinematic = std::atoi(val) != 0;
    } else if (arg == "--demoCollision") {
      const char *val = requireValue("--demoCollision");
      if (!val) return false;
      cfg.demoCollision = std::atoi(val) != 0;
    } else if (arg == "--demoSlingshot") {
      const char *val = requireValue("--demoSlingshot");
      if (!val) return false;
      cfg.demoSlingshot = std::atoi(val) != 0;
    } else if (arg == "--demoCollisionSpeed") {
      const char *val = requireValue("--demoCollisionSpeed");
      if (!val) return false;
      cfg.demoCollisionSpeed = std::atof(val);
    } else if (arg == "--fragmentation") {
      const char *val = requireValue("--fragmentation");
      if (!val) return false;
      cfg.fragmentation = std::atoi(val) != 0;
      setFragmentation = true;
    } else if (arg == "--debugCollision") {
      const char *val = requireValue("--debugCollision");
      if (!val) return false;
      cfg.debugCollision = std::atoi(val) != 0;
      setDebugCollision = true;
    } else if (arg == "--slingshotTargetPlanet") {
      const char *val = requireValue("--slingshotTargetPlanet");
      if (!val) return false;
      cfg.slingshotTargetPlanet = std::atoi(val);
    } else if (arg == "--slingshotHighlight") {
      const char *val = requireValue("--slingshotHighlight");
      if (!val) return false;
      cfg.slingshotHighlight = std::atoi(val) != 0;
    } else if (arg == "--printCOM") {
      const char *val = requireValue("--printCOM");
      if (!val) return false;
      cfg.printCOM = std::atoi(val) != 0;
    } else if (arg == "--printTotalEnergy") {
      const char *val = requireValue("--printTotalEnergy");
      if (!val) return false;
      cfg.printTotalEnergy = std::atoi(val) != 0;
    } else if (arg == "--printAngularMomentum") {
      const char *val = requireValue("--printAngularMomentum");
      if (!val) return false;
      cfg.printAngularMomentum = std::atoi(val) != 0;
    } else {
      std::fprintf(stderr, "Unknown argument: %s\n", arg.c_str());
      return false;
    }
  }

  if (cfg.present) {
    if (!setTimeScale) {
      cfg.timeScale = 0.2f;
    }
    if (!setHighlight) {
      cfg.highlightCloseEncounters = true;
    }
    if (!setAutoFocus) {
      cfg.autoFocus = true;
    }
    if (!setEnergyColor) {
      cfg.showEnergyColor = true;
    }
    if (!setVelocityVectors) {
      cfg.showVelocityVectors = true;
    }
    if (!setTrails) {
      cfg.trails = true;
    }
    if (!setTrailLength) {
      cfg.trailLength = 80;
    }
    if (!setCollisions) {
      cfg.collisions = false;
    }
    if (!setDebugCollision) {
      cfg.debugCollision = false;
    }
    cfg.zeroMomentum = true;
    cfg.centerOnCOM = true;
    cfg.autoFit = true;
    cfg.autoFitMajorOnly = true;
    cfg.stableInit = true;
    cfg.realSolarSystem = true;
    cfg.normalizeOrbits = true;
    cfg.beltCount = cfg.beltCount > 0 ? cfg.beltCount : 1000;
    cfg.minorCount = cfg.minorCount > 0 ? cfg.minorCount : 300;
    cfg.demoCollision = false;
  }

  if (cfg.cinematic) {
    if (!setTimeScale) {
      cfg.timeScale = 0.5f;
    }
    if (!setTrails) {
      cfg.trails = true;
    }
    if (!setTrailLength) {
      cfg.trailLength = 120;
    }
    if (!setHighlight) {
      cfg.highlightCloseEncounters = true;
    }
    if (!setAutoFocus) {
      cfg.autoFocus = true;
    }
    cfg.zeroMomentum = true;
    cfg.realSolarSystem = true;
    cfg.stableInit = true;
    cfg.centerOnCOM = true;
    cfg.autoFit = true;
    cfg.autoFitMajorOnly = true;
    cfg.cameraLerp = 0.05f;
    cfg.zoomLerp = 0.08f;
    cfg.visualScale = 1.4f;
    cfg.planetVisualScale = 1.4f;
    cfg.sunVisualScale = 2.2f;
    cfg.minorVisualScale = 1.0f;
    cfg.cometVisualScale = 1.2f;
    cfg.bloomBoost = 0.7f;
    cfg.bloomThreshold = 1.0f;
    cfg.showVelocityVectors = true;
    cfg.velocityMassThreshold = std::max(1.0f, cfg.planetMass * cfg.massScale * 0.5f);
    if (cfg.minPlanetPixels < 8.0f) cfg.minPlanetPixels = 8.0f;
    if (cfg.minSunPixels < 12.0f) cfg.minSunPixels = 12.0f;
  }

  if (cfg.presentCollision) {
    cfg.collisions = true;
    cfg.timeScale = 1.0f;
    if (!setN) {
      cfg.n = 8;
    }
    if (!setDt) {
      cfg.dt = 0.002f;
    }
    cfg.debugCollision = true;
    cfg.demoCollision = true;
    cfg.showVelocityVectors = true;
    cfg.highlightCloseEncounters = true;
    cfg.autoFocus = true;
    cfg.pointSizeScale = 8.0f;
    cfg.demoCollisionSpeed = 5.0f;
    cfg.trails = false;
    cfg.trailLength = 0;
    cfg.collisionFlashScale = std::max(cfg.collisionFlashScale, 1.2f);
    cfg.collisionFocus = true;
    cfg.collisionShockwave = true;
    cfg.zeroMomentum = true;
    cfg.centerOnCOM = true;
    cfg.autoFit = true;
    cfg.autoFitMajorOnly = true;
    cfg.slowOnCollision = true;
    cfg.collisionHoldSeconds = std::max(cfg.collisionHoldSeconds, 0.3f);
    cfg.collisionSlowSeconds = std::max(cfg.collisionSlowSeconds, 2.0f);
    cfg.collisionSlowFactor = std::min(cfg.collisionSlowFactor, 0.05f);
    cfg.realSolarSystem = true;
    cfg.hideMinors = true;
  }

  if (cfg.realSolarSystem) {
    cfg.stableInit = true;
    cfg.autoFit = true;
    cfg.autoFitMajorOnly = true;
    cfg.centerOnCOM = true;
    cfg.zeroMomentum = true;
    if (!setNumPlanets) {
      cfg.numPlanets = 8;
    }
    cfg.planetRadiiPreset = true;
    if (!setAsteroidBelt) {
      cfg.asteroidBelt = true;
    }
    if (!setNumComets) {
      cfg.numComets = std::max(3, cfg.numComets);
    }
    if (!setCometTail) {
      cfg.cometTail = true;
    }
    if (!setMinPlanetPixels) {
      cfg.minPlanetPixels = std::max(cfg.minPlanetPixels, 12.0f);
    }
    if (!setMinSunPixels) {
      cfg.minSunPixels = std::max(cfg.minSunPixels, 20.0f);
    }
    cfg.minorMass = std::min(cfg.minorMass, cfg.planetMass * 0.001f);
    cfg.planetVisualScale = 1.2f;
    cfg.sunVisualScale = 1.6f;
    cfg.minorVisualScale = 0.6f;
    cfg.cometVisualScale = 0.8f;
    if (cfg.slingshotTargetPlanet < 0) {
      cfg.slingshotTargetPlanet = 4;
    }
    if (cfg.beltCount == 0) cfg.beltCount = 1200;
    if (cfg.minorCount == 0) cfg.minorCount = 400;
  }

  if (cfg.solarCollision) {
    cfg.solarCollisionDemo = true;
  }

  if (cfg.solarCollisionDemo) {
    cfg.realSolarSystem = true;
    cfg.collisions = true;
    cfg.collisionModel = 0;
    cfg.allowPlanetAccretion = true;
    cfg.fragmentation = true;
    cfg.collisionShockwave = true;
    cfg.collisionFocus = true;
    cfg.maxHighlightedCollisions = 10;
    cfg.trails = true;
    cfg.trailLength = 80;
    cfg.numComets = std::max(cfg.numComets, 4);
    cfg.asteroidBelt = true;
    cfg.autoFit = true;
    cfg.autoFitMajorOnly = true;
    cfg.centerOnCOM = true;
    cfg.zeroMomentum = true;
    cfg.stableInit = true;
    cfg.planetRadiiPreset = true;
    cfg.collisionFlashDuration = std::max(cfg.collisionFlashDuration, 2.0f);
    cfg.collisionHotSeconds = cfg.collisionFlashDuration;
    cfg.collisionFocusSeconds = std::max(cfg.collisionFocusSeconds, 2.5f);
    cfg.slowOnCollision = true;
    cfg.collisionHoldSeconds = 0.3f;
    cfg.collisionSlowSeconds = 2.0f;
    cfg.collisionSlowFactor = 0.05f;
    cfg.minorVisualScale = 0.5f;
    cfg.planetVisualScale = 1.2f;
    cfg.sunVisualScale = 1.6f;
    cfg.normalizeOrbits = true;
    cfg.orbitSpeedScale = 1.0f;
    if (!setN) {
      cfg.n = 4000;
    }
    if (cfg.beltCount == 0) cfg.beltCount = 3000;
    if (cfg.minorCount == 0) cfg.minorCount = 500;
    cfg.showVelocityVectors = false;
    cfg.glDebug = true;
    cfg.cudaDebug = true;
    cfg.sanityChecks = true;
    if (!setDt) {
      cfg.dt *= cfg.collisionDtScale;
    }
    if (cfg.dt > 0.0005f) {
      cfg.dt = 0.0005f;
    }
    if (cfg.eps < cfg.collisionEpsMin) {
      cfg.eps = cfg.collisionEpsMin;
    }
    if (!setFragmentation) {
      cfg.fragmentation = false;
    }
  }

  if (cfg.minViewRadius < 1e-4f) cfg.minViewRadius = 1e-4f;
  if (cfg.maxViewRadius <= cfg.minViewRadius) cfg.maxViewRadius = cfg.minViewRadius + 0.01f;
  if (cfg.minFocusRadius < 1e-4f) cfg.minFocusRadius = 1e-4f;
  if (cfg.maxFocusRadius <= cfg.minFocusRadius) cfg.maxFocusRadius = cfg.minFocusRadius + 0.01f;
  if (cfg.collisionFocusCooldownSeconds < 0.0f) cfg.collisionFocusCooldownSeconds = 0.0f;
  if (cfg.fragmentSpeedClamp < 0.0f) cfg.fragmentSpeedClamp = 0.0f;
  if (cfg.maxAbsPosClamp <= 0.0f) cfg.maxAbsPosClamp = 1.0e4f;
  if (cfg.collisionDtScale <= 0.0f) cfg.collisionDtScale = 0.5f;
  if (cfg.collisionEpsMin < 0.0f) cfg.collisionEpsMin = 0.0f;
  if (cfg.maxDebrisActive < 0) cfg.maxDebrisActive = 0;
  if (cfg.fragmentEnergyThreshold < 0.0f) cfg.fragmentEnergyThreshold = 0.0f;

  return true;
}
