/* Decalarations of the 3D halo/void finder. */
#ifndef FINDER_H
#define FINDER_H

#include "cell.hh"
#include "halovoid_h.hpp"
#include <algorithm> // For std::sort

// Global compile-time constant array
constexpr std::array<std::array<std::array<int, 3>, 27>, 8>
compute_neighbor_order();

// Compute the total volume of all cells
double total_volume(const fft_real *p, const size_t *offset, size_t Np,
                    size_t Nd, fft_real L);

#endif
