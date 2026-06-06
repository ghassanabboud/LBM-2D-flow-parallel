#ifndef LBM_HH
#define LBM_HH

#include <cstddef>
#include <cstdint>
#include <vector>

#include <mpi.h>

/**
 * @brief 2D Lattice Boltzmann solver, D2Q9 lattice with BGK collision.
 *
 * Each MPI rank owns a rectangular tile of local_nx columns × local_ny rows,
 * plus one ghost layer on each of the four sides (filled by MPI_Sendrecv
 * before each collide_stream call).
 *
 * Local indexing (both axes share the same convention):
 *   local_x = 0               left ghost column   (data from west neighbour)
 *   local_x = 1 … local_nx   actual local data columns
 *   local_x = local_nx + 1   right ghost column   (data from east neighbour)
 *   (same convention for local_y / local_ny)
 *
 * SoA layout: f_[i * N_local + local_y * (local_nx+2) + local_x]
 *             N_local = (local_nx+2) * (local_ny+2).
 * Global x = x_start + local_x - 1.
 * Global y = y_start + local_y - 1.
 *
 * Process grid: MPI_Dims_create factorises nprocs into [prows, pcols].
 * Requirements: nx % pcols == 0,  ny % prows == 0.
 */
class LBM
{
public:
  static constexpr int Q = 9;

  static const int    cx[Q];
  static const int    cy[Q];
  static const double w[Q];
  static const int    opp[Q];
  static const char* const dir_name[Q];  ///< Human-readable direction labels (C,E,N,W,S,NE,NW,SW,SE).

  /**
   * @param nx, ny   Global grid dimensions.
   * @param u_in     Inlet velocity in lattice units.
   * @param Re       Reynolds number based on cylinder diameter.
   * @param cyl_x/y/r  First cylinder centre and radius in cell units.
   * @param comm     MPI communicator (typically MPI_COMM_WORLD).
   *
   * nx must be divisible by pcols and ny by prows (determined by MPI_Dims_create).
   */
  LBM(std::size_t nx, std::size_t ny,
      double u_in, double Re,
      double cyl_x, double cyl_y, double cyl_r,
      MPI_Comm comm);

  void add_second_cylinder(double cyl2_x, double cyl2_y, double cyl2_r);

  void initialize();

  /// Advance one time step: halo exchange → collide_stream → inlet → outlet.
  void step();

  // ---- Macroscopic accessors (local coords: local_x=1..local_nx_, local_y=1..local_ny_) ----
  double rho      (std::size_t local_x, std::size_t local_y) const;
  double ux       (std::size_t local_x, std::size_t local_y) const;
  double uy       (std::size_t local_x, std::size_t local_y) const;
  bool   is_solid (std::size_t local_x, std::size_t local_y) const;

  std::size_t nx()       const { return nx_; }
  std::size_t ny()       const { return ny_; }
  std::size_t local_nx() const { return local_nx_; }
  std::size_t local_ny() const { return local_ny_; }
  std::size_t x_start()  const { return x_start_; }
  std::size_t y_start()  const { return y_start_; }
  double      tau()      const { return tau_; }
  double      u_in()     const { return u_in_; }
  int         rank()     const { return rank_; }
  int         nprocs()   const { return nprocs_; }

  /**
   * Gather macroscopic fields to rank 0 (full domain, global row-major order).
   * Vorticity is computed on rank 0 from the gathered ux/uy arrays.
   * All ranks must call this collectively. Only rank 0's output vectors are
   * populated.
   */
  void gather_fields(std::vector<double>& rho_out,
                     std::vector<double>& ux_out,
                     std::vector<double>& uy_out,
                     std::vector<double>& vor_out) const;

  /// Gather solid mask to rank 0 (full domain, global row-major order).
  void gather_solid(std::vector<std::uint8_t>& solid_out) const;

  /// Debug: print solid mask at global boundary rows (rank-aware).
  void print_solid_boundary_rows() const;
  /// Debug: print solid mask stored in each rank's ghost rows.
  void print_solid_boundary_rows_ghost() const;
  /// Debug: print all 9 populations at a global (x_global, y_global) point.
  void print_point_populations(std::size_t x_global, std::size_t y_global) const;
  /// Debug: print post-stream pre-collision populations at a global point.
  /// Must be called after exchange_halos() so ghost rows are fresh.
  void print_post_stream_pre_collision(std::size_t x_global, std::size_t y_global) const;

private:
  // Local flat index: local_x in [0, local_nx+1], local_y in [0, local_ny+1].
  std::size_t lidx (std::size_t local_x, std::size_t local_y) const
  { return local_y * (local_nx_ + 2) + local_x; }
  // SoA population index within local storage.
  std::size_t fidx (int i, std::size_t local_x, std::size_t local_y) const
  { return std::size_t(i) * N_local_ + lidx(local_x, local_y); }

  void mark_obstacle  (double c_x, double c_y, double r);
  void exchange_halos ();
  void collide_stream ();
  void apply_inlet    ();
  void apply_outlet   ();
  void gather_streamed(std::size_t local_x, std::size_t local_y, double f_local[Q]) const;

  std::size_t nx_, ny_;       ///< Global dimensions.
  std::size_t local_nx_;      ///< Data columns owned by this rank.
  std::size_t local_ny_;      ///< Data rows owned by this rank.
  std::size_t x_start_;       ///< Global x of this rank's first data column.
  std::size_t y_start_;       ///< Global y of this rank's first data row.
  std::size_t N_local_;       ///< (local_nx+2)*(local_ny+2): cells per direction plane.
  double u_in_;
  double tau_;

  int      rank_, nprocs_;
  int      rank_below_, rank_above_;  ///< y-axis neighbours (south/north, may be MPI_PROC_NULL).
  int      rank_west_,  rank_east_;   ///< x-axis neighbours (may be MPI_PROC_NULL).
  int      cart_dims_[2];             ///< [prows, pcols].
  int      cart_coords_[2];           ///< [row_coord, col_coord] of this rank.
  MPI_Comm comm_;                     ///< Cartesian communicator.

  std::vector<double>       f_;       ///< Q * N_local doubles, SoA with ghost layers.
  std::vector<double>       ftmp_;
  std::vector<std::uint8_t> solid_;   ///< N_local bytes (ghost layers initialised 0).
};

#endif  // LBM_HH
