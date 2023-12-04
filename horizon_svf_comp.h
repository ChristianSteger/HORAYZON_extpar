#ifndef TESTLIB_H
#define TESTLIB_H
#include <stdint.h>
#include <string>

void horizon_svf_comp(double* vlon, double* vlat, float* topography_v,
    int vertex, 
    double* clon, double* clat, int* vertex_of_cell, uint8_t* mask,
    int cell,
    float* horizon, float* skyview, int nhori, int svf_type);

#endif
