#ifndef TESTLIB_H
#define TESTLIB_H
#include <stdint.h>
#include <string>

void horizon_svf_comp(double* vlon, double* vlat, float* topography_v,
    int num_vertex, 
    double* clon, double* clat, int* vertex_of_cell,
    int num_cell,
    float* horizon, float* skyview, int nhori, int refine_factor,
    int svf_type);

#endif
