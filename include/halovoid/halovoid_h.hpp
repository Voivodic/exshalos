/* Auxiliary types and functions shared by the 2D and 3D halo/void finders. */
#ifndef HALOVOID_H
#define HALOVOID_H

#ifdef DOUBLEPRECISION_FFTW
typedef double fft_real;
#define NP_OUT_TYPE NPY_DOUBLE
#else
typedef float fft_real;
#define NP_OUT_TYPE NPY_FLOAT
#endif

// Include headers for the Voronoi computations
#include <array>
#include <math.h>

// Define unsigned integers
using uint = unsigned int;

// Struct for the halos and voids
struct halo_void {
    fft_real x[3];
    fft_real rho;
};

// Get the cell index of a particle
size_t get_cell_index(const fft_real p[3], size_t Nd, fft_real scale);

// Organize the particles in a 3D grid
void create_container(fft_real *pos, size_t *offset, size_t Np, size_t Nd,
                      fft_real L);

#endif
