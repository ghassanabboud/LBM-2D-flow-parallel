#ifndef LBM_HH
#define LBM_HH

#include <cstddef>
#include <cstdint>
#include <vector>

#include <mpi.h>

/**
 * @brief 2D Lattice Boltzmann solver, D2Q9 lattice with BGK collision.
 *
 * Each MPI rank owns a horizontal slab of local_ny rows plus one ghost row
 * on each side (filled by MPI_Sendrecv before each collide_stream call).
 *
 * Local y-indexing:
 *   local_y = 0               bottom ghost row  (data from rank-1)
 *   local_y = 1 … local_ny   actual local data rows
 *   local_y = local_ny + 1   top ghost row     (data from rank+1)
 *
 * SoA layout: f_[i * N_local + local_y * nx + x], N_local = nx*(local_ny+2).
 * Global y = y_start + local_y - 1.
 *
 * Requirement: ny % nprocs == 0.
 */
class LBM
{
public:
  static constexpr int Q = 9;

  static const int    cx[Q];
  static const int    cy[Q];
  static const double w[Q];
  static const int    opp[Q];

  /**
   * @param nx, ny   Global grid dimensions.
   * @param u_in     Inlet velocity in lattice units.
   * @param Re       Reynolds number based on cylinder diameter.
   * @param cyl_x/y/r  First cylinder centre and radius in cell units.
   * @param comm     MPI communicator (typically MPI_COMM_WORLD).
   *
   * ny must be divisible by the communicator size.
   */
  LBM(std::size_t nx, std::size_t ny,
      double u_in, double Re,
      double cyl_x, double cyl_y, double cyl_r,
      MPI_Comm comm);

  void add_second_cylinder(double cyl2_x, double cyl2_y, double cyl2_r);

  void initialize();

  /// Advance one time step: halo exchange → collide_stream → inlet → outlet.
  void step();

  // ---- Macroscopic accessors (use local y coords: local_y = 1..local_ny_) ----
  double rho      (std::size_t x, std::size_t local_y) const;
  double ux       (std::size_t x, std::size_t local_y) const;
  double uy       (std::size_t x, std::size_t local_y) const;
  bool   is_solid (std::size_t x, std::size_t local_y) const;

  std::size_t nx()       const { return nx_; }
  std::size_t ny()       const { return ny_; }   ///< Global ny.
  std::size_t local_ny() const { return local_ny_; }
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

private:
  // Local flat index: local_y in [0, local_ny+1], x in [0, nx).
  std::size_t lidx (std::size_t x, std::size_t local_y) const { return local_y * nx_ + x; }
  // SoA population index within local storage.
  std::size_t fidx (int i, std::size_t x, std::size_t local_y) const
  { return std::size_t(i) * N_local_ + lidx(x, local_y); }

  void mark_obstacle  (double c_x, double c_y, double r);
  void exchange_halos ();
  void collide_stream ();
  void apply_inlet    ();
  void apply_outlet   ();

  std::size_t nx_, ny_;      ///< Global dimensions.
  std::size_t local_ny_;     ///< Data rows owned by this rank.
  std::size_t y_start_;      ///< Global y of this rank's first data row.
  std::size_t N_local_;      ///< nx * (local_ny + 2): cells per direction plane.
  double u_in_;
  double tau_;

  int      rank_, nprocs_;
  int      rank_below_, rank_above_;  ///< Neighbour ranks (may be MPI_PROC_NULL).
  MPI_Comm comm_;

  std::vector<double>   f_;      ///< Q * N_local doubles, SoA with ghost rows.
  std::vector<double>   ftmp_;
  std::vector<std::uint8_t> solid_;  ///< N_local bytes (ghost rows initialised 0).
};

#endif  // LBM_HH
