/* Decalarations of the 3D halo/void finder. */
#ifndef FINDER_H
#define FINDER_H

#include "halovoid_h.hpp"
#include "voro++.hh"

#ifdef __cplusplus
extern "C" {
#endif

int find_halos_voids(const fft_real *pos, size_t N, int find_halos,
                     int find_voids, fft_real delta_h, fft_real delta_v,
                     fft_real L, uint nd, fft_real r_max, fft_real **halos_out,
                     size_t *n_halos, fft_real **voids_out, size_t *n_voids);

#ifdef __cplusplus
}
#endif

#endif
