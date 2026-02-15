#include "types.h"
#include "physics.h"
#include <cmath>


const float G = 1.0f; 

const float SOFTENING = 1e-9f; 

void bodyForceCPU(float4* p, float3* v, float dt, int n) {
    
    for (int i = 0; i < n; i++) {
        float Fx = 0.0f;
        float Fy = 0.0f;
        float Fz = 0.0f;

        for (int j = 0; j < n; j++) {
            
            float dx = p[j].x - p[i].x;
            float dy = p[j].y - p[i].y;
            float dz = p[j].z - p[i].z;
            
           
            float distSqr = dx*dx + dy*dy + dz*dz + SOFTENING;
            
            
            float invDist = 1.0f / std::sqrt(distSqr);
            float invDist3 = invDist * invDist * invDist;


            float force_magnitude = G * p[j].w * invDist3;


            Fx += dx * force_magnitude;
            Fy += dy * force_magnitude;
            Fz += dz * force_magnitude;
        }


        v[i].x += Fx * dt;
        v[i].y += Fy * dt;
        v[i].z += Fz * dt;
    }
}

void integratePositionsCPU(float4* p, float3* v, float dt, int n) {
    for (int i = 0; i < n; i++) {

        p[i].x += v[i].x * dt;
        p[i].y += v[i].y * dt;
        p[i].z += v[i].z * dt;
    }
}