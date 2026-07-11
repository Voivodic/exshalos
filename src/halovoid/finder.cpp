#include "finder.hpp"

// Main function that finds halos and voids in 3D
int find_halos_voids(const fft_real *pos, size_t N, int find_halos,
                     int find_voids, fft_real delta_h, fft_real delta_v,
                     fft_real L, uint nd, fft_real r_max, fft_real **halos_out,
                     size_t *n_halos, fft_real **voids_out, size_t *n_voids) {
    // Create the container structure
    uint **container;
    uint *n_container;
    container = new uint *[nd * nd * nd];
    n_container = new uint[nd * nd * nd];
    for (uint i = 0; i < nd * nd * nd; i++)
        n_container[i] = 0;

    // Create the container
    // create_container(pos, N, L, const_cast<uint>(nd), container,
    // n_container);

    // Free the container
    delete[] container;
    delete[] n_container;

    return 0;
}
