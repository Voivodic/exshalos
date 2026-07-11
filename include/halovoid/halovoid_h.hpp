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

// Define unsigned integers
using uint = unsigned int;

#ifdef __cplusplus
extern "C" {
#endif

#ifdef __cplusplus
}
#endif

#endif
