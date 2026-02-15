#ifndef TYPES_H
#define TYPES_H


#ifndef __CUDACC__
struct float4 {
    float x;
    float y;
    float z;
    float w; 
};

struct float3 {
    float x;
    float y;
    float z;
};
#else
#include <vector_types.h>
#endif

#endif // TYPES_H