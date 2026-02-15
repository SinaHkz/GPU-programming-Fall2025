#include "physics.h"
#include <cmath>

void bodyForceCPU(float4* p, float3* v, float dt, int n) {
    // Forces
    for (int i = 0; i < n; i++) {
        float fx = 0.0f;
        float fy = 0.0f;
        float fz = 0.0f;

        for (int j = 0; j < n; j++) {
            float dx = p[j].x - p[i].x;
            float dy = p[j].y - p[i].y;
            float dz = p[j].z - p[i].z;
            
            // Softening
            float distSqr = dx*dx + dy*dy + dz*dz + 1e-9f; 
            float invDist = 1.0f / std::sqrt(distSqr);
            float invDist3 = invDist * invDist * invDist;

            // Math
            float F = p[j].w * invDist3;
            
            fx += dx * F;
            fy += dy * F;
            fz += dz * F;
        }

        // Velocity
        v[i].x += dt * fx;
        v[i].y += dt * fy;
        v[i].z += dt * fz;
    }
}

void integratePositionsCPU(float4* p, float3* v, float dt, int n) {
    // Update
    for (int i = 0; i < n; i++) {
        p[i].x += v[i].x * dt;
        p[i].y += v[i].y * dt;
        p[i].z += v[i].z * dt;
    }
}