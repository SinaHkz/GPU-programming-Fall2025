#include "types.h"
#include <cmath>
#include <cstdlib>


const float SUN_MASS = 10000.0f;


const float PLANET_MASS_BASE = 0.001f; 
const float COMET_MASS = 0.001f;
const float G = 1.0f; 

void initSolarSystem(float4* positions, float3* velocities, int n) {
    srand(42); 

    // 1. The Sun (Body 0)
    positions[0] = {0.0f, 0.0f, 0.0f, SUN_MASS};
    velocities[0] = {0.0f, 0.0f, 0.0f};

    // 2. The Comet (Body 1) 
    positions[1] = {250.0f, 150.0f, 0.0f, COMET_MASS};
    velocities[1] = {-25.0f, -10.0f, 0.0f}; 

    // 3. The Planets (Bodies 2 to N-1)
    for (int i = 2; i < n; i++) {
        float r = 50.0f + (rand() % 350); 
        float angle = (float)(rand() % 360) * M_PI / 180.0f;

        positions[i].x = r * std::cos(angle);
        positions[i].y = r * std::sin(angle);
        positions[i].z = ((rand() % 100) / 100.0f) * 2.0f - 1.0f; 
        
     
        positions[i].w = PLANET_MASS_BASE + ((rand() % 10) * 0.0001f);

       
        float orbital_vel = std::sqrt(G * SUN_MASS / r);

        velocities[i].x = -orbital_vel * std::sin(angle);
        velocities[i].y = orbital_vel * std::cos(angle);
        velocities[i].z = 0.0f; 
    }
}