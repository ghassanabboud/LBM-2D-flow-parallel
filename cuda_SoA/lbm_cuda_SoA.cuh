#ifndef LBM_CUDA_SOA_CUH
#define LBM_CUDA_SOA_CUH

#include <cstddef>
#include <cstdint>
#include <vector>

/**
 * @brief 2D Lattice Boltzmann solver, D2Q9 BGK, CUDA single-GPU implementation.
 *
 * Memory layout mirrors serial_fused_SoA: Structure of Arrays (SoA).
 *   d_f_[ i * (nx*ny) + (y*nx + x) ]   for direction i in [0, 9).
 * All cells for a given direction are contiguous, so warp-level streaming
 * reads are stride-1 within each direction plane — one 256-byte HBM2
 * transaction per 32 threads per direction.
 *
 * D2Q9 lattice constants (cx, cy, w, opp) are stored in __constant__ memory
 * so that every warp-level broadcast hits the constant cache rather than HBM2.
 * The solid mask is accessed via __ldg() to guarantee the read-only L1 path.
 *
 * Three kernels are launched per time step (in default CUDA stream order):
 *   1. lbm_collide_stream_kernel — fused pull-stream + BGK for all cells.
 *   2. lbm_inlet_kernel          — equilibrium reset at x = 0.
 *   3. lbm_outlet_kernel         — zero-gradient copy at x = nx-1.
 *
 * Host-side macroscopic accessors (rho, ux, uy, vorticity) operate on h_f_,
 * the host mirror of d_f_.  Call sync_to_host() before using them.
 * fetch_probe_uxy() uses cudaMemcpy2D to copy one cell's 9 populations without
 * a full device-to-host transfer; suitable for per-step probe recording.
 */
class LBM
{
public:
  static constexpr int Q = 9;

  /**
   * @param nx       Number of cells along x.
   * @param ny       Number of cells along y.
   * @param u_in     Inlet velocity in lattice units (must be << 1/sqrt(3)).
   * @param Re       Target Reynolds number based on cylinder diameter.
   * @param cyl_x    Cylinder centre along x, in cell units.
   * @param cyl_y    Cylinder centre along y, in cell units.
   * @param cyl_r    Cylinder radius, in cell units.
   */
  LBM(std::size_t nx, std::size_t ny,
      double u_in, double Re,
      double cyl_x, double cyl_y, double cyl_r);

  ~LBM();

  /// Add a second circular obstacle. No-op if r2 <= 0.
  void add_second_cylinder(double cx2, double cy2, double r2);

  /// Set d_f_ to the equilibrium distribution with rho=1, u=(u_in,0) on
  /// every fluid cell and u=(0,0) on solid cells.  Uploads result to device.
  void initialize();

  /// Advance the simulation by one time step: launches 3 kernels, swaps
  /// d_f_ / d_ftmp_ device pointers (no data copy).
  /// block_x/block_y: thread-block dimensions (1D: block_y=1, 2D: block_y>1).
  /// grid_x/grid_y:   grid dimensions; must satisfy
  ///                  grid_x * block_x >= nx and grid_y * block_y >= ny.
  void step(int block_x, int block_y, int grid_x, int grid_y);

  /// Copy d_f_ from device to h_f_ on host.  Must be called before using
  /// rho(), ux(), uy(), vorticity().
  void sync_to_host();

  /// Copy the 9 populations of cell (x,y) from device using cudaMemcpy2D,
  /// then compute and return ux and uy on the host.  No full sync required.
  void fetch_probe_uxy(std::size_t x, std::size_t y,
                       double & ux_out, double & uy_out) const;

  // Macroscopic accessors — operate on h_f_; require prior sync_to_host().
  double rho      (std::size_t x, std::size_t y) const;
  double ux       (std::size_t x, std::size_t y) const;
  double uy       (std::size_t x, std::size_t y) const;
  double vorticity(std::size_t x, std::size_t y) const;
  bool   is_solid (std::size_t x, std::size_t y) const;

  std::size_t nx()   const { return nx_; }
  std::size_t ny()   const { return ny_; }
  double      tau()  const { return tau_; }
  double      u_in() const { return u_in_; }

private:
  std::size_t idx(std::size_t x, std::size_t y) const { return y * nx_ + x; }

  void mark_obstacle(double cx, double cy, double r);

  std::size_t nx_, ny_, N_;   // N_ = nx_ * ny_
  double      u_in_, tau_;

  // Device arrays (cudaMalloc).
  double*  d_f_;       ///< Current distributions, SoA, Q*N_ doubles.
  double*  d_ftmp_;    ///< Double-buffer, Q*N_ doubles.
  uint8_t* d_solid_;   ///< Solid mask, N_ bytes.  Read-only during simulation.

  // Host mirrors.
  std::vector<double>  h_f_;      ///< Populated by sync_to_host().
  std::vector<uint8_t> h_solid_;  ///< Built at construction, uploaded once.
};

#endif  // LBM_CUDA_SOA_CUH
