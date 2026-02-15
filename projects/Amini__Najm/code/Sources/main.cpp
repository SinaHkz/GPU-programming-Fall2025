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

    // Mem
    int b_pos = nBodies * sizeof(float4);
    int b_vel = nBodies * sizeof(float3);
    
    float4* p = (float4*)malloc(b_pos);
    float3* v = (float3*)malloc(b_vel);

    if (export_data) {
        std::cout << "Exporting stable trajectory data to trajectory.csv...\n";
        std::ofstream out("trajectory.csv");
        out << "Step,BodyID,X,Y,Mass\n";

        initSolarSystem(p, v, nBodies);

        // Export
        for (int f = 0; f < total_frames; f++) {
            runSimulationGPU_Tiled(p, v, dt, nBodies, steps_per_frame); 
            savePositionsToCSV(out, p, nBodies, f);
            if (f % 50 == 0) std::cout << "Saved frame " << f << " / " << total_frames << "\n";
        }
        out.close();
        std::cout << "Export complete! Run the Python visualizer to see the orbits.\n";

    } else {
        // Benchmark
        std::cout << "Bodies: " << nBodies << " | Iterations: " << nIters << "\n";
        std::cout << "----------------------------------------\n";

        // CPU
        initSolarSystem(p, v, nBodies);
        auto t1 = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < nIters; i++) {
            bodyForceCPU(p, v, dt, nBodies);
            integratePositionsCPU(p, v, dt, nBodies);
        }
        auto t2 = std::chrono::high_resolution_clock::now();
        double t_cpu = std::chrono::duration<double>(t2 - t1).count();
        std::cout << "1. CPU Time:          " << t_cpu << " s\n";

        // Naive
        initSolarSystem(p, v, nBodies);
        t1 = std::chrono::high_resolution_clock::now();
        runSimulationGPU_Naive(p, v, dt, nBodies, nIters);
        t2 = std::chrono::high_resolution_clock::now();
        double t_naive = std::chrono::duration<double>(t2 - t1).count();
        std::cout << "2. GPU Naive Time:    " << t_naive << " s\n";

        // Tiled
        initSolarSystem(p, v, nBodies);
        t1 = std::chrono::high_resolution_clock::now();
        runSimulationGPU_Tiled(p, v, dt, nBodies, nIters);
        t2 = std::chrono::high_resolution_clock::now();
        double t_tiled = std::chrono::duration<double>(t2 - t1).count();
        std::cout << "3. GPU Tiled Time:    " << t_tiled << " s\n";

        // Streamed
        initSolarSystem(p, v, nBodies);
        t1 = std::chrono::high_resolution_clock::now();
        runSimulationGPU_Streamed(p, v, dt, nBodies, nIters);
        t2 = std::chrono::high_resolution_clock::now();
        double t_stream = std::chrono::duration<double>(t2 - t1).count();
        std::cout << "4. GPU Streamed Time: " << t_stream << " s\n";
        std::cout << "----------------------------------------\n";

        // Speedup
        std::cout << "SPEEDUP (Naive vs CPU):    " << (t_cpu / t_naive) << "x\n";
        std::cout << "SPEEDUP (Tiled vs CPU):    " << (t_cpu / t_tiled) << "x\n";
        std::cout << "SPEEDUP (Streamed vs CPU): " << (t_cpu / t_stream) << "x\n";
        std::cout << "----------------------------------------\n";
    }

    // Cleanup
    free(p);
    free(v);
    return 0;
}