#ifndef PHYSICS_H
#define PHYSICS_H

#include "types.h"

// CPU Baseline
void bodyForceCPU(float4* p, float3* v, float dt, int n);
void integratePositionsCPU(float4* p, float3* v, float dt, int n);

// GPU Wrappers
void runSimulationGPU_Naive(float4* h_p, float3* h_v, float dt, int nBodies, int nIters);
void runSimulationGPU_Tiled(float4* h_p, float3* h_v, float dt, int nBodies, int nIters);

// --- NEW: Streamed GPU Execution ---
void runSimulationGPU_Streamed(float4* h_p, float3* h_v, float dt, int nBodies, int nIters);

#endif // PHYSICS_H