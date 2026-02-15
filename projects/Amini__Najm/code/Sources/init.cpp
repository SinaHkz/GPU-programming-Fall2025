#include "types.h"
#include <cmath>
#include <cstdlib>

const float G = 1.0f; 
const float SUN_MASS = 10000.0f;
const float PLANET_MASS_BASE = 0.001f; 
const float COMET_MASS = 0.001f;

void initSolarSystem(float4* positions, float3* velocities, int n) {
    
    // Seed
    srand(42); 

    // Sun
    positions[0] = {0.0f, 0.0f, 0.0f, SUN_MASS};
    velocities[0] = {0.0f, 0.0f, 0.0f};

    // Comet
    positions[1] = {250.0f, 150.0f, 0.0f, COMET_MASS};
    velocities[1] = {-25.0f, -10.0f, 0.0f}; 

    // Planets
    for (int i = 2; i < n; i++) {
        float radius = 50.0f + (rand() % 350); 
        float angle = (float)(rand() % 360) * M_PI / 180.0f;

        positions[i].x = radius * std::cos(angle);
        positions[i].y = radius * std::sin(angle);
        positions[i].z = ((rand() % 100) / 100.0f) * 2.0f - 1.0f; 
        
        positions[i].w = PLANET_MASS_BASE + ((rand() % 10) * 0.0001f);

        // Orbit
        float v = std::sqrt(G * SUN_MASS / radius);

        velocities[i].x = -v * std::sin(angle);
        velocities[i].y = v * std::cos(angle);
        velocities[i].z = 0.0f; 
    }
}