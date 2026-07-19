#include "halovoid_h.hpp"

// Get the cell index of a particle
size_t get_cell_index(const fft_real p[3], size_t Nd, fft_real scale) {
    size_t idx = (size_t)(p[0] * scale);
    if (idx >= Nd)
        idx = Nd - 1;
    size_t idy = (size_t)(p[1] * scale);
    if (idy >= Nd)
        idy = Nd - 1;
    size_t idz = (size_t)(p[2] * scale);
    if (idz >= Nd)
        idz = Nd - 1;

    return idx * Nd * Nd + idy * Nd + idz;
}

// Organize the particles in a 3D grid
void create_container(fft_real *pos, size_t *offset, size_t Np, size_t Nd,
                      fft_real L) {
    // Compute the total number of cells
    const size_t num_cells = Nd * Nd * Nd;

    // Allocate the array of particles and the number of particles per cell
    for (size_t i = 0; i < num_cells + 1; i++)
        offset[i] = 0;

    // Count the number of particles in each cell
    const fft_real scale = (float)Nd / L;
    for (size_t i = 0; i < Np; i++) {
        fft_real p[3] = {pos[3 * i], pos[3 * i + 1], pos[3 * i + 2]};
        size_t id = get_cell_index(p, Nd, scale);

        offset[id + 1]++;
    }

    // Compute the offset of each cell
    size_t current_offset = 0;
    size_t *write_ptr = new size_t[num_cells];
    for (size_t i = 0; i < num_cells; i++) {
        offset[i] = current_offset;
        write_ptr[i] = current_offset;
        current_offset += offset[i + 1];
    }
    offset[num_cells] = current_offset;

    // Fill the array of particles using in-place binning
    for (size_t b = 0; b < num_cells; b++) {
        // Loop while the current block still has unsorted slots
        while (write_ptr[b] < offset[b + 1]) {
            size_t i = write_ptr[b];
            fft_real p[3] = {pos[3 * i], pos[3 * i + 1], pos[3 * i + 2]};
            size_t target_b = get_cell_index(p, Nd, scale);

            // Check if the particle belongs to the current block
            if (target_b == b) {
                write_ptr[b]++;
            } else {
                size_t dest = write_ptr[target_b];

                // Swap the particles
                for (int dim = 0; dim < 3; dim++) {
                    pos[3 * i + dim] = pos[3 * dest + dim];
                    pos[3 * dest + dim] = p[dim];
                }

                // Increment the write pointer
                write_ptr[target_b]++;
            }
        }
    }

    delete[] write_ptr;
}
