#include <iostream>
#include <fstream>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <string>
#include "types.h"
#include "physics.h"

void initSolarSystem(float4* positions, float3* velocities, int n);

void savePositionsToCSV(std::ofstream& file, float4* p, int nBodies, int step) {
    for (int i = 0; i < nBodies; i++) {
        file << step << "," << i << "," << p[i].x << "," << p[i].y << "," << p[i].w << "\n";
    }
}

int main(int argc, char** argv) {
    int nBodies = 4096; 
    if (argc > 1) nBodies = std::atoi(argv[1]);

    bool export_data = false;
    if (argc > 2 && std::string(argv[2]) == "export") {
        export_data = true;
    }

    float dt = 0.002f;         
    int total_frames = 600;    
    int steps_per_frame = 20;  
    int nIters = 100;          

    int bytes_pos = nBodies * sizeof(float4);
    int bytes_vel = nBodies * sizeof(float3);
    
    float4* p = (float4*)malloc(bytes_pos);
    float3* v = (float3*)malloc(bytes_vel);

    if (export_data) {
        std::cout << "Exporting stable trajectory data to trajectory.csv...\n";
        std::ofstream outfile("trajectory.csv");
        outfile << "Step,BodyID,X,Y,Mass\n";

        initSolarSystem(p, v, nBodies);

        for (int frame = 0; frame < total_frames; frame++) {
            runSimulationGPU_Tiled(p, v, dt, nBodies, steps_per_frame); 
            savePositionsToCSV(outfile, p, nBodies, frame);
            if (frame % 50 == 0) std::cout << "Saved frame " << frame << " / " << total_frames << "\n";
        }
        outfile.close();
        std::cout << "Export complete! Run the Python visualizer to see the orbits.\n";

    } else {
        // --- THE ULTIMATE BENCHMARK SUITE ---
        std::cout << "Bodies: " << nBodies << " | Iterations: " << nIters << "\n";
        std::cout << "----------------------------------------\n";

        // 1. CPU BASELINE
        initSolarSystem(p, v, nBodies);
        auto start_cpu = std::chrono::high_resolution_clock::now();
        for (int step = 0; step < nIters; step++) {
            bodyForceCPU(p, v, dt, nBodies);
            integratePositionsCPU(p, v, dt, nBodies);
        }
        auto end_cpu = std::chrono::high_resolution_clock::now();
        double cpu_time = std::chrono::duration<double>(end_cpu - start_cpu).count();
        std::cout << "1. CPU Time:          " << cpu_time << " s\n";

        // 2. GPU NAIVE
        initSolarSystem(p, v, nBodies);
        auto start_naive = std::chrono::high_resolution_clock::now();
        runSimulationGPU_Naive(p, v, dt, nBodies, nIters);
        auto end_naive = std::chrono::high_resolution_clock::now();
        double naive_time = std::chrono::duration<double>(end_naive - start_naive).count();
        std::cout << "2. GPU Naive Time:    " << naive_time << " s\n";

        // 3. GPU TILED
        initSolarSystem(p, v, nBodies);
        auto start_tiled = std::chrono::high_resolution_clock::now();
        runSimulationGPU_Tiled(p, v, dt, nBodies, nIters);
        auto end_tiled = std::chrono::high_resolution_clock::now();
        double tiled_time = std::chrono::duration<double>(end_tiled - start_tiled).count();
        std::cout << "3. GPU Tiled Time:    " << tiled_time << " s\n";

        // 4. GPU STREAMED
        initSolarSystem(p, v, nBodies);
        auto start_stream = std::chrono::high_resolution_clock::now();
        runSimulationGPU_Streamed(p, v, dt, nBodies, nIters);
        auto end_stream = std::chrono::high_resolution_clock::now();
        double stream_time = std::chrono::duration<double>(end_stream - start_stream).count();
        std::cout << "4. GPU Streamed Time: " << stream_time << " s\n";
        std::cout << "----------------------------------------\n";

        // SPEEDUPS
        std::cout << "SPEEDUP (Naive vs CPU):    " << (cpu_time / naive_time) << "x\n";
        std::cout << "SPEEDUP (Tiled vs CPU):    " << (cpu_time / tiled_time) << "x\n";
        std::cout << "SPEEDUP (Streamed vs CPU): " << (cpu_time / stream_time) << "x\n";
        std::cout << "----------------------------------------\n";
    }

    free(p);
    free(v);
    return 0;
}