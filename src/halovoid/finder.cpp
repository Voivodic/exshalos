#include "finder.hpp"

// Generates the lookup table at compile-time using standard C++23 features
constexpr std::array<std::array<std::array<int, 3>, 27>, 8>
compute_neighbor_order() {
    std::array<std::array<std::array<int, 3>, 27>, 8> result{};

    for (int octant = 0; octant < 8; ++octant) {
        // Octant center coordinates (scaled by 4 to keep integer math)
        int ox = (octant & 1) ? 1 : -1;
        int oy = (octant & 2) ? 1 : -1;
        int oz = (octant & 4) ? 1 : -1;

        std::array<std::array<int, 3>, 27> neighbors{};
        int n_idx = 0;

        // 1. Populate the 27 adjacent blocks
        for (int dz = -1; dz <= 1; ++dz) {
            for (int dy = -1; dy <= 1; ++dy) {
                for (int dx = -1; dx <= 1; ++dx) {
                    neighbors[n_idx++] = {dx, dy, dz};
                }
            }
        }

        // 2. Sort the array at compile-time using std::sort and a lambda
        std::sort(neighbors.begin(), neighbors.end(),
                  [ox, oy, oz](const std::array<int, 3> &a,
                               const std::array<int, 3> &b) {
                      auto dist_sq = [ox, oy, oz](const std::array<int, 3> &n) {
                          int rx = 4 * n[0] - ox;
                          int ry = 4 * n[1] - oy;
                          int rz = 4 * n[2] - oz;
                          return rx * rx + ry * ry + rz * rz;
                      };
                      return dist_sq(a) < dist_sq(b);
                  });

        // 3. Write sorted values directly to output
        for (int i = 0; i < 27; ++i) {
            result[octant][i] = {neighbors[i][0], neighbors[i][1],
                                 neighbors[i][2]};
        }
    }

    return result;
}

// Compile-time constant array with the neighbors order
constexpr auto NEIGHBOR_ORDER = compute_neighbor_order();

// Compute the total volume of all cells
double total_volume(const fft_real *p, const size_t *offset, size_t Np,
                    size_t Nd, fft_real L) {
    // Compute some constants
    const float Lcell = L / (float)Nd;
    const float scale = 1.0 / Lcell;
    const float shift_cell = 0.5 * Lcell;

    // Run over all particles
    voro::voronoicell cell;
    double total_volume = 0.0;
    for (size_t index = 0; index < Np; index++) {
        // Initialize the Voronoi cell
        cell.init(-L / 2.0, L / 2.0, -L / 2.0, L / 2.0, -L / 2.0, L / 2.0);

        // Get the position of the particle
        float px = p[3 * index];
        float py = p[3 * index + 1];
        float pz = p[3 * index + 2];

        // Get the cell index
        size_t idx = (size_t)(px * scale);
        if (idx >= Nd)
            idx = Nd - 1;
        size_t idy = (size_t)(py * scale);
        if (idy >= Nd)
            idy = Nd - 1;
        size_t idz = (size_t)(pz * scale);
        if (idz >= Nd)
            idz = Nd - 1;

        // Compute the local coordinates
        float dx_local = px - idx * Lcell;
        float dy_local = py - idy * Lcell;
        float dz_local = pz - idz * Lcell;

        // Get the offsets table
        const int octant = (dx_local - shift_cell > 0 ? 1 : 0) |
                           (dy_local - shift_cell > 0 ? 2 : 0) |
                           (dz_local - shift_cell > 0 ? 4 : 0);
        static const auto offsets = NEIGHBOR_ORDER[octant];

        // Compute the Voronoi volume of this particle
        double r2_max = cell.max_radius_squared();
        for (int o = 0; o < 27; o++) {
            int i = offsets[o][0];
            int j = offsets[o][1];
            int k = offsets[o][2];

            // x-direction
            int d_idx = idx + i;
            int wrap_x = 0;
            if (d_idx < 0) {
                d_idx += Nd;
                wrap_x = -1;
            } else if (d_idx >= (int)Nd) {
                d_idx -= Nd;
                wrap_x = 1;
            }
            float dist_x =
                (i == -1) ? dx_local : ((i == 1) ? Lcell - dx_local : 0.0);

            // y-direction
            int d_idy = idy + j;
            int wrap_y = 0;
            if (d_idy < 0) {
                d_idy += Nd;
                wrap_y = -1;
            } else if (d_idy >= (int)Nd) {
                d_idy -= Nd;
                wrap_y = 1;
            }
            float dist_y =
                (j == -1) ? dy_local : ((j == 1) ? Lcell - dy_local : 0.0);

            // z-direction
            int d_idz = idz + k;
            int wrap_z = 0;
            if (d_idz < 0) {
                d_idz += Nd;
                wrap_z = -1;
            } else if (d_idz >= (int)Nd) {
                d_idz -= Nd;
                wrap_z = 1;
            }
            float dist_z =
                (k == -1) ? dz_local : ((k == 1) ? Lcell - dz_local : 0.0);

            // If the block is too far away, skip all particles inside
            float block_dist2 =
                dist_x * dist_x + dist_y * dist_y + dist_z * dist_z;
            if (block_dist2 > 4.0 * r2_max) {
                continue;
            }

            // Compute the index of the block
            size_t c_index = d_idx * Nd * Nd + d_idy * Nd + d_idz;

            // Run over all particles in this block
            bool cell_cut = false;
            size_t np_in_block = offset[c_index + 1] - offset[c_index];
            for (size_t part = 0; part < np_in_block; part++) {
                float dx = p[3 * (offset[c_index] + part)] - px + wrap_x * L;
                float dy =
                    p[3 * (offset[c_index] + part) + 1] - py + wrap_y * L;
                float dz =
                    p[3 * (offset[c_index] + part) + 2] - pz + wrap_z * L;

                // Avoid self-particles and particles too far away
                float d2 = dx * dx + dy * dy + dz * dz;
                if (d2 < 1e-12 || d2 > 4.0 * r2_max)
                    continue;

                // Compute the plane
                if (cell.plane(dx, dy, dz)) {
                    cell_cut = true;
                }
            }

            // If the plane was cut, update the maximum radius
            if (cell_cut) {
                r2_max = cell.max_radius_squared();
            }
        }
        // Add the cell volume to the total volume
        total_volume += cell.volume();
    }

    return total_volume;
}

// Find the halos and voids in 3D
void find_halos_and_voids(const fft_real *p, const size_t *offset, size_t Np,
                    size_t Nd, fft_real L) {
    // Compute some constants
    const float Lcell = L / (float)Nd;
    const float scale = 1.0 / Lcell;
    const float shift_cell = 0.5 * Lcell;

    // Run over all particles
    voro::voronoicell cell;
    double total_volume = 0.0;
    for (size_t index = 0; index < Np; index++) {
        // Initialize the Voronoi cell
        cell.init(-L / 2.0, L / 2.0, -L / 2.0, L / 2.0, -L / 2.0, L / 2.0);

        // Get the position of the particle
        float px = p[3 * index];
        float py = p[3 * index + 1];
        float pz = p[3 * index + 2];

        // Get the cell index
        size_t idx = (size_t)(px * scale);
        if (idx >= Nd)
            idx = Nd - 1;
        size_t idy = (size_t)(py * scale);
        if (idy >= Nd)
            idy = Nd - 1;
        size_t idz = (size_t)(pz * scale);
        if (idz >= Nd)
            idz = Nd - 1;

        // Compute the local coordinates
        float dx_local = px - idx * Lcell;
        float dy_local = py - idy * Lcell;
        float dz_local = pz - idz * Lcell;

        // Get the offsets table
        const int octant = (dx_local - shift_cell > 0 ? 1 : 0) |
                           (dy_local - shift_cell > 0 ? 2 : 0) |
                           (dz_local - shift_cell > 0 ? 4 : 0);
        static const auto offsets = NEIGHBOR_ORDER[octant];

        // Compute the Voronoi volume of this particle
        double r2_max = cell.max_radius_squared();
        for (int o = 0; o < 27; o++) {
            int i = offsets[o][0];
            int j = offsets[o][1];
            int k = offsets[o][2];

            // x-direction
            int d_idx = idx + i;
            int wrap_x = 0;
            if (d_idx < 0) {
                d_idx += Nd;
                wrap_x = -1;
            } else if (d_idx >= (int)Nd) {
                d_idx -= Nd;
                wrap_x = 1;
            }
            float dist_x =
                (i == -1) ? dx_local : ((i == 1) ? Lcell - dx_local : 0.0);

            // y-direction
            int d_idy = idy + j;
            int wrap_y = 0;
            if (d_idy < 0) {
                d_idy += Nd;
                wrap_y = -1;
            } else if (d_idy >= (int)Nd) {
                d_idy -= Nd;
                wrap_y = 1;
            }
            float dist_y =
                (j == -1) ? dy_local : ((j == 1) ? Lcell - dy_local : 0.0);

            // z-direction
            int d_idz = idz + k;
            int wrap_z = 0;
            if (d_idz < 0) {
                d_idz += Nd;
                wrap_z = -1;
            } else if (d_idz >= (int)Nd) {
                d_idz -= Nd;
                wrap_z = 1;
            }
            float dist_z =
                (k == -1) ? dz_local : ((k == 1) ? Lcell - dz_local : 0.0);

            // If the block is too far away, skip all particles inside
            float block_dist2 =
                dist_x * dist_x + dist_y * dist_y + dist_z * dist_z;
            if (block_dist2 > 4.0 * r2_max) {
                continue;
            }

            // Compute the index of the block
            size_t c_index = d_idx * Nd * Nd + d_idy * Nd + d_idz;

            // Run over all particles in this block
            bool cell_cut = false;
            size_t np_in_block = offset[c_index + 1] - offset[c_index];
            for (size_t part = 0; part < np_in_block; part++) {
                float dx = p[3 * (offset[c_index] + part)] - px + wrap_x * L;
                float dy =
                    p[3 * (offset[c_index] + part) + 1] - py + wrap_y * L;
                float dz =
                    p[3 * (offset[c_index] + part) + 2] - pz + wrap_z * L;

                // Avoid self-particles and particles too far away
                float d2 = dx * dx + dy * dy + dz * dz;
                if (d2 < 1e-12 || d2 > 4.0 * r2_max)
                    continue;

                // Compute the plane
                if (cell.plane(dx, dy, dz)) {
                    cell_cut = true;
                }
            }

            // If the plane was cut, update the maximum radius
            if (cell_cut) {
                r2_max = cell.max_radius_squared();
            }
        }
        // Add the cell volume to the total volume
        total_volume += cell.volume();
    }
}
