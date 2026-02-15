# CUDA N-Body Simulation (Solar System + Comet)

A from-scratch CUDA/C++ N-body simulation with a solar-system-inspired setup, OpenGL visualization (GLFW + GLEW), and benchmark mode. Physics runs entirely on the GPU using a tiled O(N^2) gravity kernel and a Leapfrog/Velocity-Verlet integrator. Collisions are an optional runtime extension and are **off by default** (proposal-compliant).

## Build (Linux)

Dependencies:
- CUDA toolkit (nvcc)
- OpenGL
- GLFW
- GLEW
- CMake 3.18+

Example (Ubuntu/Debian):
```
sudo apt-get install cmake libglfw3-dev libglew-dev
```

Build:
```
mkdir -p build
cd build
cmake ..
cmake --build . -j
```

## Run

Default (rendering on, collisions off):
```
./build/nbody --N 4096 --dt 0.001 --timeScale 1.0 --G 1.0 --eps 0.01 --render 1 --trails 0 --trailLength 50 --benchmark 0 --collisions 0
```

Presentation preset (slow motion + highlights):
```
./build/nbody --present 1
```

Presentation collision preset:
```
./build/nbody --presentCollision 1
```

Cinematic preset:
```
./build/nbody --cinematic 1
```

Benchmark mode (rendering disabled):
```
./build/nbody --benchmark 1 --steps 100 --collisions 0
```

Enable optional elastic collisions:
```
./build/nbody --collisions 1 --collisionRestitution 1.0
```

Demo modes:
```
./build/nbody --demoCollision 1 --collisions 1
./build/nbody --demoSlingshot 1
```

Realistic solar system preset:
```
./build/nbody --realSolarSystem 1 --stableInit 1 --centerOnCOM 1
```

Solar-system collision demo (trackable collisions + fragmentation):
```
./build/nbody --solarCollision 1
```

Gold collision command (stable camera + visible collisions):
```
./build/nbody --solarCollision 1 --collisionFlashDuration 2.0 --collisionFocusCooldownSeconds 1.0
```

### Golden Commands (Quick Demos)
1. Presentation mode (clear planets + patterns):
```
./build/nbody --present 1
```
2. Solar collision demo (trackable collisions):
```
./build/nbody --solarCollision 1
```
3. Solar collision demo with explicit caps:
```
./build/nbody --solarCollision 1 --maxHotCollisions 10 --fragmentation 1
```
4. Benchmark mode (render off):
```
./build/nbody --benchmark 1 --steps 100 --collisions 0
```

### CLI Options
- `--N` number of bodies
- `--dt` timestep
- `--timeScale` time scaling factor (effective_dt = dt * timeScale)
- `--G` gravitational constant
- `--eps` softening epsilon
- `--massScale` multiplies all masses
- `--sunMass` sun mass
- `--planetMass` planet base mass
- `--cometMass` comet mass
- `--minorMass` minor body mass
- `--steps` number of steps (use `0` for infinite when rendering)
- `--seed` RNG seed
- `--render` 0/1
- `--trails` 0/1
- `--trailLength` trail length per body
- `--benchmark` 0/1
- `--collisions` 0/1 (default 0)
- `--collisionRestitution` restitution coefficient for elastic collisions
- `--highlightCloseEncounters` 0/1 (color + line when distance < 3*eps)
- `--showVelocityVectors` 0/1
- `--showEnergyColor` 0/1 (bound = blue, escaping = yellow)
- `--autoFocus` 0/1 (focus closest pair)
- `--centerOnCOM` 0/1 (rendering centered on center of mass)
- `--zeroMomentum` 0/1 (subtract COM velocity after init)
- `--autoFit` 0/1 (auto-fit camera and world scale)
- `--autoFitTarget` all|major (fit all bodies or only sun/planets/comets)
- `--worldScale` world-to-render scale (auto if not set)
- `--minZoom` minimum zoom clamp
- `--maxZoom` maximum zoom clamp
- `--stableInit` 0/1 (evenly spaced initial planets)
- `--realSolarSystem` 0/1 (realistic mass ratios preset)
- `--numPlanets` number of major planets (default 8)
- `--planetRadiiPreset` 0/1 (use preset radii list)
- `--planetEccMax` maximum planet eccentricity
- `--planetIncDegMax` maximum planet inclination (degrees)
- `--asteroidBelt` 0/1 (enable asteroid belt)
- `--asteroidBeltRmin` inner belt radius
- `--asteroidBeltRmax` outer belt radius
- `--asteroidTangentialJitter` tangential velocity jitter for belt
- `--asteroidRadialJitter` radial velocity jitter for belt
- `--beltCount` belt particle count
- `--minorCount` outer minor count
- `--hideMinors` 0/1 (hide minors for presentation)
- `--numComets` number of comets
- `--cometEccMin` minimum comet eccentricity
- `--cometEccMax` maximum comet eccentricity
- `--cometIncDegMax` maximum comet inclination (degrees)
- `--cometTail` 0/1 (enable comet tail overlay)
- `--cometTailLength` tail length (world units)
- `--cometTailAlpha` tail alpha
- `--maxHighlightedCollisions` max collision pairs highlighted (default 10)
- `--maxHotCollisions` alias for `--maxHighlightedCollisions`
- `--fragmentsPerCollision` fragments per collision (when fragmentation=1)
- `--fragmentSpeedSpread` debris velocity spread
- `--fragmentSpeedClamp` max debris speed multiplier relative to local speed
- `--maxDebris` maximum debris spawned
- `--maxDebrisActive` cap on active debris count
- `--fragmentEnergyThreshold` minimum collision energy for fragmentation
- `--maxCollisionPairs` cap on copied collision pairs per frame
- `--solarCollisionDemo` 0/1 (solar-system collision preset)
- `--solarCollision` 0/1 (alias of solar-system collision preset)
- `--numPlanets` number of major planets (default 8)
- `--planetRadiiPreset` 0/1 (use preset radii list)
- `--planetEccMax` maximum planet eccentricity
- `--planetIncDegMax` maximum planet inclination (degrees)
- `--asteroidBelt` 0/1 (enable asteroid belt)
- `--asteroidBeltRmin` inner belt radius
- `--asteroidBeltRmax` outer belt radius
- `--asteroidTangentialJitter` tangential velocity jitter for belt
- `--asteroidRadialJitter` radial velocity jitter for belt
- `--numComets` number of comets
- `--cometEccMin` minimum comet eccentricity
- `--cometEccMax` maximum comet eccentricity
- `--cometIncDegMax` maximum comet inclination (degrees)
- `--cometTail` 0/1 (enable comet tail overlay)
- `--cometTailLength` tail length (world units)
- `--cometTailAlpha` tail alpha
- `--maxHighlightedCollisions` max collision pairs highlighted (default 10)
- `--fragmentsPerCollision` fragments per collision (when fragmentation=1)
- `--fragmentSpeedSpread` debris velocity spread
- `--fragmentSpeedClamp` max debris speed multiplier relative to local speed
- `--maxDebris` maximum debris spawned
- `--solarCollisionDemo` 0/1 (solar-system collision preset)
- `--focusThreshold` minimum distance for auto-focus (default 3*eps)
- `--cameraLerp` smoothing factor for camera focus (0..1)
- `--zoomLerp` smoothing factor for zoom focus (0..1)
- `--initialFocusDelay` seconds to wait before auto-focus
- `--initialFocusDelaySeconds` same as `--initialFocusDelay`
- `--pointSizeScale` scalar for point size
- `--visualScale` extra scale factor for mass-based sizes
- `--planetScale` extra scale for planet render sizes (legacy)
- `--sunScale` extra scale for sun render size (legacy)
- `--sunVisualScale` extra scale for sun render sizes
- `--planetVisualScale` extra scale for planet render sizes
- `--minorVisualScale` extra scale for minor render sizes
- `--cometVisualScale` extra scale for comet render sizes
- `--sunVisualBoost` extra boost for sun size
- `--planetVisualBoost` extra boost for planet size
- `--minorVisualBoost` extra boost for minor size
- `--cometVisualBoost` extra boost for comet size
- `--planetStyle` 0/1 (procedural planet patterns)
- `--minPlanetPixels` minimum planet size in pixels
- `--minSunPixels` minimum sun size in pixels
- `--minCometPixels` minimum comet size in pixels
- `--minMinorPixels` minimum minor size in pixels
- `--maxMinorPixels` maximum minor size in pixels
- `--bloomBoost` bloom-like boost for large masses (0 disables)
- `--bloomThreshold` mass-size threshold for bloom
- `--velocityVectorScale` scales velocity vector length (world units per velocity unit)
- `--arrowHeadLength` arrowhead length (set to `0` to disable arrowheads)
- `--arrowHeadWidth` arrowhead half-width
- `--trailAlpha` alpha for trail lines
- `--velocityLineAlpha` alpha for velocity vectors
- `--closeLineAlpha` alpha for close-encounter lines
- `--velocityMassThreshold` only draw velocity vectors above this mass (or close interaction)
- `--collisionFlashDuration` flash duration in seconds
- `--collisionFlashSeconds` same as `--collisionFlashDuration`
- `--collisionFlashScale` size scale during collision flash
- `--collisionHotSeconds` duration for hot (red) collision highlight
- `--collisionFocus` 0/1 (smoothly focus on collisions)
- `--collisionFocusSeconds` duration to focus on collision
- `--collisionFocusCooldownSeconds` minimum seconds between focus retriggers
- `--minFocusRadius` clamp for focus target radius (render units)
- `--maxFocusRadius` clamp for focus target radius (render units)
- `--minViewRadius` clamp for camera view radius
- `--maxViewRadius` clamp for camera view radius
- `--glDebug` 0/1 (OpenGL error checks)
- `--cudaDebug` 0/1 (cudaDeviceSynchronize after kernels/memcpy)
- `--sanityChecks` 0/1 (NaN/Inf and range checks)
- `--maxAbsPosClamp` max |position| before triggering safety reset
- `--collisionDtScale` scale factor for collision preset dt
- `--collisionEpsMin` minimum epsilon for collision preset
- `--collisionModel` merge|elastic (merge default in collision presets)
- `--allowPlanetAccretion` 0/1 (planets accrete minor bodies in merge mode)
- `--normalizeOrbits` 0/1 (normalize major planet tangential speed)
- `--orbitSpeedScale` scale for normalized orbit speeds
- `--slowOnCollision` 0/1 (slow time on collisions)
- `--collisionSlowFactor` time scale factor during slow window
- `--collisionHoldSeconds` hold time on collision
- `--collisionSlowSeconds` slow duration on collision
- `--cometTrailWidth` line width for comet trail
- `--collisionShockwave` 0/1 (expanding ring on collisions)
- `--slingshotTargetPlanet` index of target planet for slingshot
- `--slingshotHighlight` 0/1 (highlight comet + target at closest approach)
- `--present` 0/1 (presentation preset)
- `--presentCollision` 0/1 (collision presentation preset)
- `--cinematic` 0/1 (cinematic visual preset: smooth camera, additive glow, trails, close-encounter highlights)
- `--demoCollision` 0/1
- `--demoSlingshot` 0/1
- `--demoCollisionSpeed` sets demo collision speed
- `--fragmentation` 0/1 (demo-only optional fragments)
- `--allowPlanetFragment` 0/1 (allow planet fragmentation in collision mode)
- `--debugCollision` 0/1 (prints one pair per step when collision occurs)
- `--printCOM` 0/1 (prints |Vcom| every 200 steps)
- `--printTotalEnergy` 0/1 (total energy + drift)
- `--printAngularMomentum` 0/1 (total angular momentum vector)

### Controls (Rendering)
- Arrow keys: pan
- Mouse wheel: zoom
- Middle mouse drag: pan
- Right mouse drag: rotate (3D)
- `+` / `-`: increase/decrease `timeScale`
- Space: pause/unpause
- `T`: toggle trails
- `F`: focus on selected body (use `[` / `]` to change selection)
- `N`: focus on closest interacting pair
- `R`: reset camera to fitted view
- `Esc`: quit

## Design Choices (Aligned to Proposal)

### Proposal-Compliant Mode (Default)
- Pure Newtonian gravity with softening only.
- Bodies are point masses (no physical collisions).
- Leapfrog / Velocity-Verlet integrator (symplectic).
- O(N^2) gravity with tiled shared memory on the GPU.
- Demonstrate stable orbits and comet slingshot with:
```
./build/nbody --demoSlingshot 1 --collisions 0 --timeScale 0.5 --trails 1 --trailLength 80
```

### Extended Collision Mode (Optional)
- Enable with `--collisions 1` (or `--presentCollision 1`).
- Elastic sphere collisions with restitution.
- Sequential resolution avoids race conditions but is intended for small-N demos.
- Collisions flash bodies orange/red and camera focuses on recent collisions.

### Physics Model
- Newtonian gravity with softening:
  a_i = sum_{j!=i} G m_j (r_j - r_i) / (|r_j-r_i|^2 + eps^2)^(3/2)
- Softening (`eps`) prevents singular forces at close range and stabilizes integration.
- Leapfrog / Velocity-Verlet integrator (exact step order):
  1. compute a(t)
  2. v(t+dt/2) = v(t) + a(t)*dt/2
  3. r(t+dt) = r(t) + v(t+dt/2)*dt
  4. compute a(t+dt)
  5. v(t+dt) = v(t+dt/2) + a(t+dt)*dt/2
- Advantage: symplectic integration provides good long-term energy behavior for orbital systems.

### CUDA Parallelization
- One thread per body.
- O(N^2) interaction kernel with **tiled shared memory** (positions + mass).
- Structure-of-Arrays (SoA) layout: `x[], y[], z[], vx[], vy[], vz[], m[], ax[], ay[], az[]`.
- Entire timestep loop runs on the GPU; CPU only initializes and visualizes.

### System Initialization
- Sun at origin with dominant mass.
- Planets on circular orbits: v = sqrt(G*M_sun/R).
- Comet on elongated orbit designed to pass near a planet and be visibly deflected.
- Minor bodies in a belt with small masses so they do not perturb planets.
- Categories only affect initialization and color; physics is identical for all bodies.

### Optional Collisions (Runtime Flag)
- **Default: off** (`--collisions 0`), matching the proposal (bodies pass through each other).
- When enabled (`--collisions 1`): elastic sphere collisions with impulse response and a minimal positional correction.
- Collision pipeline (approximate but stable):
  1) detect all colliding pairs into a GPU buffer
  2) resolve pairs sequentially on the GPU to avoid race conditions
- Limitation: collision buffer can be large for high N; collision mode is intended for small N demos.
- Note: collision response is non-Hamiltonian and breaks strict symplectic behavior.

### Visualization Features
- Soft-glow particles with point size scaled by `mass^(1/3)` and `--visualScale`.
- Procedural planet impostors for major bodies (no external textures required).
- Auto-fit camera and world scale (`--autoFit 1`) keeps the system inside the viewport.
- Optional comet tails for visibility (`--cometTail 1`).
- If you want textures, you can replace the procedural shading in `shaders/particle.frag`.
- Trails rendered as `GL_LINE_STRIP` per body.
- Optional velocity vectors (line segments).
- Optional close-encounter highlighting (red + line if distance < 3*eps).
- Optional energy coloring (blue bound / yellow escaping).
- Velocity vectors include arrowheads when `--arrowHeadLength > 0`.
- Cinematic mode enables additive glow blending, stronger bloom, and smoother camera motion.

### Scientific Validation
- `--printCOM 1` prints |Vcom| every 200 steps to verify drift removal.
- `--printTotalEnergy 1` prints total kinetic/potential/total energy and relative drift every 200 steps.
- `--printAngularMomentum 1` prints total angular momentum vector every 200 steps.

### Troubleshooting
- If the system drifts: `--zeroMomentum 1` removes COM drift and `--centerOnCOM 1` recenters rendering.
- If planets eject quickly: use `--stableInit 1`, a smaller `--dt`, or larger `--eps`.
- If the camera jumps: increase `--focusThreshold` or decrease `--cameraLerp`.
- If the system is off-screen or too small: enable `--autoFit 1` or adjust `--worldScale`, `--minZoom`, `--maxZoom`.
- If collisions are hard to see: increase `--collisionFlashScale` or `--collisionFlashDuration`.

### Benchmark Mode
- Disables rendering and runs sizes in `{1024, 2048, 4096, 8192, 16384}`.
- Prints time per step (CUDA events) and IPS:
  IPS = N*(N-1) / time_step_seconds

## How to Demo (Golden Commands)

1. **Realistic solar system (no collisions, proposal-compliant)**  
```
./build/nbody --realSolarSystem 1 --collisions 0
```

2. **Presentation mode**  
```
./build/nbody --present 1
```

3. **Presentation collision mode**  
```
./build/nbody --presentCollision 1
```

4. **Benchmark mode (render OFF)**  
```
./build/nbody --benchmark 1 --steps 100 --collisions 0
```

## File Structure
```
/app
  main.cpp
/src
  init.cpp
  init.h
  integrator.cu
  integrator.h
  kernels.cu
  kernels.h
  utils.cu
  utils.h
/shaders
  particle.vert
  particle.frag
  line.vert
  line.frag
/vis
  renderer.cpp
  renderer.h
CMakeLists.txt
README.md
```

## Notes
- Rendering uses GL_POINTS with soft glow; trails are line strips.
- Benchmark mode disables all visualization and measures only physics kernels.
