#ifndef TESTLIB_H
#define TESTLIB_H

void horizon_svf_comp(double* vlon, double* vlat, 
    double* clon, double* clat, float* topography_v,
    int* vertex_of_cell, int vertex, int cell,
    float* horizon, float* skyview, int nhori, int* mask);

#endif
