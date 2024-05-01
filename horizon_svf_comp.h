#ifndef TESTLIB_H
#define TESTLIB_H
#include <stdint.h>
#include <string>

void horizon_svf_comp(double* clon, double* clat, float* hsurf,
    int num_cell,
    double* vlon, double* vlat,
    int num_vertex,
    int* vertex_of_triangle, int num_triangle,  // temporary
    int* cells_of_vertex,
    float* horizon, float* skyview,
    int nhori,
    int refine_factor, int svf_type);

#endif
