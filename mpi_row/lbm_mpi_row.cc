#include "lbm_mpi_row.hh"

#include <algorithm>
#include <cassert>
#include <cstring>
#include <stdexcept>
#include <stdio.h>
#include <iostream>

// D2Q9 lattice constants.
//   0: rest   1: E    2: N    3: W    4: S
//   5: NE     6: NW   7: SW   8: SE
const int LBM::cx[9]  = { 0,  1,  0, -1,  0,  1, -1, -1,  1};
const int LBM::cy[9]  = { 0,  0,  1,  0, -1,  1,  1, -1, -1};

const double LBM::w[9] = {
  4.0 / 9.0,
  1.0 / 9.0,  1.0 / 9.0,  1.0 / 9.0,  1.0 / 9.0,
  1.0 / 36.0, 1.0 / 36.0, 1.0 / 36.0, 1.0 / 36.0
};

const int LBM::opp[9] = {0, 3, 4, 1, 2, 7, 8, 5, 6};

const char* const LBM::dir_name[9] = {"C","E","N","W","S","NE","NW","SW","SE"};

// ---------------------------------------------------------------------------

LBM::LBM(std::size_t nx, std::size_t ny,
         double u_in, double Re,
         double cyl_x, double cyl_y, double cyl_r,
         MPI_Comm comm)
  : nx_(nx), ny_(ny), local_ny_(0), y_start_(0), N_local_(0),
    u_in_(u_in), tau_(0.0),
    rank_(0), nprocs_(1),
    rank_below_(MPI_PROC_NULL), rank_above_(MPI_PROC_NULL),
    comm_(comm)
{
  MPI_Comm_rank(comm_, &rank_);
  MPI_Comm_size(comm_, &nprocs_);

  if (ny % std::size_t(nprocs_) != 0)
    throw std::runtime_error("ny must be divisible by the number of MPI ranks");

  local_ny_ = ny / std::size_t(nprocs_);
  y_start_  = std::size_t(rank_) * local_ny_;
  N_local_  = nx_ * (local_ny_ + 2); // 2 additional rows for ghost data

  rank_below_ = (rank_ > 0)           ? rank_ - 1 : MPI_PROC_NULL;
  rank_above_ = (rank_ < nprocs_ - 1) ? rank_ + 1 : MPI_PROC_NULL;

  std::cout << "Rank " << rank_ << " / " << nprocs_
            << " : local_ny = " << local_ny_
            << ", y range = [" << y_start_ << ", " << (y_start_ + local_ny_ - 1) << "]\n"
            << ", rank_below = " << rank_below_ << ", rank_above = " << rank_above_ << "\n";

  // allocate and zero-initialise vectors to respect first-touch principle 
  f_    .assign(Q * N_local_, 0.0);
  ftmp_ .assign(Q * N_local_, 0.0);
  solid_.assign(N_local_, 0);

  const double nu = u_in_ * (2.0 * cyl_r) / Re;
  tau_ = 3.0 * nu + 0.5;

  // Mark top and bottom physical walls using global y.
  // Rank 0 owns global y = 0 (bottom wall) at local_y = 1.
  // Last rank owns global y = ny-1 (top wall) at local_y = local_ny_.
  for (std::size_t x = 0; x < nx_; ++x) {
    if (y_start_ == 0)
      solid_[lidx(x, 1)] = 1;                       // bottom wall marked at local_y = 1 because local_y = 0 is the ghost row representing the outside
    if (y_start_ + local_ny_ == ny_)
      solid_[lidx(x, local_ny_)] = 1;               // top wall marked at local_y = local_ny_ because local_y = local_ny_ + 1 is the ghost row representing the outside
  }

  mark_obstacle(cyl_x, cyl_y, cyl_r);
}

void
LBM::add_second_cylinder(double cyl2_x, double cyl2_y, double cyl2_r)
{
  if (cyl2_r > 0.0) mark_obstacle(cyl2_x, cyl2_y, cyl2_r);
}

void
LBM::mark_obstacle(double c_x, double c_y, double r)
{
  const double r2 = r * r;
  bool has_local = false;
  // includes ghost rows (local_y = 0 and local_ny_+1) so bounce-back works
  // correctly across rank boundaries — ghost global-y may be out of domain but
  // the cylinder equation is still valid there.
  for (std::size_t local_y = 0; local_y <= local_ny_ + 1; ++local_y) {
    const double gy = double(long(y_start_) + long(local_y) - 1);
    for (std::size_t x = 0; x < nx_; ++x) {
      const double dx = double(x) - c_x;
      const double dy = gy - c_y;
      if (dx * dx + dy * dy <= r2){
        solid_[lidx(x, local_y)] = 1;
        has_local = true;
    }
  }
}
  if (has_local) {
    std::cout << "Rank " << rank_ << " has part of the cylinder in its local domain.\n";
  }
}

void
LBM::initialize()
{
  for (std::size_t local_y = 1; local_y <= local_ny_; ++local_y) {
    for (std::size_t x = 0; x < nx_; ++x) {
      const std::size_t k  = lidx(x, local_y);
      const double      ux = solid_[k] ? 0.0 : u_in_;
      const double      uy = 0.0;
      const double      u2 = ux * ux;
      for (int i = 0; i < Q; ++i) {
        const double cu  = cx[i] * ux + cy[i] * uy;
        const double feq = w[i] * (1.0 + 3.0 * cu + 4.5 * cu * cu - 1.5 * u2);
        f_[std::size_t(i) * N_local_ + k] = feq;
      }
    }
  }
}

void
LBM::step()
{
  exchange_halos();
  if (std::getenv("DEBUG_PRINTS")) {
    const std::size_t x_mid = nx_ / 2;
    for (int r = 0; r < nprocs_; ++r) {
      if (rank_ == r)
        for (std::size_t gy : {std::size_t(0), std::size_t(1), ny_ - 2, ny_ - 1})
          print_post_stream_pre_collision(x_mid, gy);
      MPI_Barrier(comm_);
    }
  }
  collide_stream();
  apply_inlet();
  apply_outlet();
}

// ---------------------------------------------------------------------------
// Halo exchange
//
// Tags: 0 = upward data (sent from rank r to rank r+1)
//       1 = downward data (sent from rank r to rank r-1)
//
// Each MPI_Sendrecv packs / unpacks Q * nx doubles (all direction planes for
// one row). MPI_PROC_NULL makes boundary ranks' calls no-ops.
// ---------------------------------------------------------------------------
void
LBM::exchange_halos()
{
  const std::size_t row_len = nx_;  
  const std::size_t buf_len = Q * row_len;

  std::vector<double> send_down(buf_len), recv_from_below(buf_len, 0.0);
  std::vector<double> send_up  (buf_len), recv_from_above(buf_len, 0.0);

  // Pack bottom data row (local_y = 1) to send to rank_below.
  for (int i = 0; i < Q; ++i)
    std::memcpy(&send_down[std::size_t(i) * row_len], // this respects the SoA layout
                &f_[std::size_t(i) * N_local_ + 1 * nx_], // i * N_local_ respects the SoA layout, + 1 * nx_ to get to local_y = 1
                row_len * sizeof(double));

  // Pack top data row (local_y = local_ny_) to send to rank_above.
  for (int i = 0; i < Q; ++i)
    std::memcpy(&send_up[std::size_t(i) * row_len],
                &f_[std::size_t(i) * N_local_ + local_ny_ * nx_],
                row_len * sizeof(double));

  MPI_Status status;

  // Exchange with lower neighbour: send down (tag 1), receive upward (tag 0).
  MPI_Sendrecv(send_down.data(),      int(buf_len), MPI_DOUBLE, rank_below_, 1,
               recv_from_below.data(), int(buf_len), MPI_DOUBLE, rank_below_, 0,
               comm_, &status);

  // Exchange with upper neighbour: send up (tag 0), receive downward (tag 1).
  MPI_Sendrecv(send_up.data(),        int(buf_len), MPI_DOUBLE, rank_above_, 0,
               recv_from_above.data(), int(buf_len), MPI_DOUBLE, rank_above_, 1,
               comm_, &status);

  // Unpack into ghost rows.
  for (int i = 0; i < Q; ++i) {
    // Bottom ghost row (local_y = 0): data sent upward from rank_below.
    std::memcpy(&f_[std::size_t(i) * N_local_ + 0 * nx_], // i * N_local_ respects the SoA layout, + 0 * nx_ to get to local_y = 0 (the bottom ghost row)
                &recv_from_below[std::size_t(i) * row_len],
                row_len * sizeof(double));
    // Top ghost row (local_y = local_ny_ + 1): data sent downward from rank_above.
    std::memcpy(&f_[std::size_t(i) * N_local_ + (local_ny_ + 1) * nx_],
                &recv_from_above[std::size_t(i) * row_len],
                row_len * sizeof(double));
  }
}


void
LBM::collide_stream()
{
  // Fused pull-stream + BGK collision in a single pass over the grid.
  // The SoA layout means all cells for a given direction are contiguous in
  // memory, so each streaming gather reads stride-1 within its direction plane.
  const double inv_tau = 1.0 / tau_;

  for (std::size_t local_y = 1; local_y <= local_ny_; ++local_y) {
    for (std::size_t x = 0; x < nx_; ++x) {
      const std::size_t k = lidx(x, local_y);

      // this implementation does streaming and then collision in a single pass
      // first, we gather for cell k, the neighbouring position streams that are coming our way. There are three cases then:
      // - if the neighbour is fluid and inside the domain, we take the value from that cell and population
      // - if the neighbour is solid and inside the domain, we take the value from the opposite population of that cell (bounce-back)
      // - if the neighbour is outside the domain, we keep the current value in f_local, which will be overwritten if on inlet/outlet and is never used otherwise (e.g. on the top/bottom walls)
      // MPI SPECIFIC NOTE: we check both x and global-y out-of-bounds. Middle-rank ghost rows
      // contain valid halo data so their global y is in [0, ny_) and the check passes.
      // Boundary ghost rows (rank 0 bottom ghost at global y=-1, last rank top ghost at global
      // y=ny_) are outside the domain and must fall back to keeping the current value, matching
      // the SoA treatment of out-of-domain neighbours.


      // once we have gathered all of this into f_local, we have all the data we need to do the collision step for cell k before writing the post-collision values to ftmp_
      // we still need to have two arrays because changing f_ directly would change the values used for the next cells within one step
      double f_local[Q];
      gather_streamed(x, local_y, f_local);
      
      double rho = 0.0, mx = 0.0, my = 0.0;
      for (int i = 0; i < Q; ++i) {
        rho += f_local[i];
        mx  += cx[i] * f_local[i];
        my  += cy[i] * f_local[i];
      }

      const double ux = (rho > 0.0) ? mx / rho : 0.0;
      const double uy = (rho > 0.0) ? my / rho : 0.0;
      const double u2 = ux * ux + uy * uy;

      for (int i = 0; i < Q; ++i) {
        const double cu  = cx[i] * ux + cy[i] * uy;
        const double feq = w[i] * rho * (1.0 + 3.0 * cu + 4.5 * cu * cu - 1.5 * u2);
        ftmp_[std::size_t(i) * N_local_ + k] = f_local[i] - inv_tau * (f_local[i] - feq);
      }
    }
  }

  f_.swap(ftmp_);
}

void
LBM::apply_inlet()
{
  const std::size_t x = 0;
  for (std::size_t local_y = 1; local_y <= local_ny_; ++local_y) {
    const std::size_t k = lidx(x, local_y);
    if (solid_[k]) continue;
    const double ux = u_in_;
    const double u2 = ux * ux;
    for (int i = 0; i < Q; ++i) {
      const double cu = cx[i] * ux;
      f_[std::size_t(i) * N_local_ + k] = w[i] * (1.0 + 3.0 * cu + 4.5 * cu * cu - 1.5 * u2);
    }
  }
}

void
LBM::apply_outlet()
{
  if (nx_ < 2) return;
  const std::size_t x  = nx_ - 1;
  const std::size_t xs = nx_ - 2;
  for (std::size_t local_y = 1; local_y <= local_ny_; ++local_y) {
    const std::size_t k  = lidx(x,  local_y);
    const std::size_t ks = lidx(xs, local_y);
    for (int i = 0; i < Q; ++i)
      f_[std::size_t(i) * N_local_ + k] = f_[std::size_t(i) * N_local_ + ks];
  }
}

double
LBM::rho(std::size_t x, std::size_t local_y) const
{
  const std::size_t k = lidx(x, local_y);
  double r = 0.0;
  for (int i = 0; i < Q; ++i) r += f_[std::size_t(i) * N_local_ + k];
  return r;
}

double
LBM::ux(std::size_t x, std::size_t local_y) const
{
  const std::size_t k = lidx(x, local_y);
  double r = 0.0, m = 0.0;
  for (int i = 0; i < Q; ++i) {
    const double fi = f_[std::size_t(i) * N_local_ + k];
    r += fi;
    m += cx[i] * fi;
  }
  return (r > 0.0) ? m / r : 0.0;
}

double
LBM::uy(std::size_t x, std::size_t local_y) const
{
  const std::size_t k = lidx(x, local_y);
  double r = 0.0, m = 0.0;
  for (int i = 0; i < Q; ++i) {
    const double fi = f_[std::size_t(i) * N_local_ + k];
    r += fi;
    m += cy[i] * fi;
  }
  return (r > 0.0) ? m / r : 0.0;
}

bool
LBM::is_solid(std::size_t x, std::size_t local_y) const
{
  return solid_[lidx(x, local_y)] != 0;
}

// ---------------------------------------------------------------------------
// Collective gather operations — all ranks must call.
// ---------------------------------------------------------------------------

void
LBM::gather_fields(std::vector<double>& rho_out,
                   std::vector<double>& ux_out,
                   std::vector<double>& uy_out,
                   std::vector<double>& vor_out) const
{
  const std::size_t slab = local_ny_ * nx_;  // elements per rank per field

  // Compute local macro fields for rows 1..local_ny_ (contiguous in send buf).
  std::vector<double> loc_rho(slab), loc_ux(slab), loc_uy(slab);
  for (std::size_t local_y = 1; local_y <= local_ny_; ++local_y) {
    const std::size_t row_off = (local_y - 1) * nx_;
    for (std::size_t x = 0; x < nx_; ++x) {
      const std::size_t k = lidx(x, local_y);
      double r = 0.0, mx = 0.0, my = 0.0;
      for (int i = 0; i < Q; ++i) {
        const double fi = f_[std::size_t(i) * N_local_ + k];
        r  += fi;
        mx += cx[i] * fi;
        my += cy[i] * fi;
      }
      loc_rho[row_off + x] = r;
      loc_ux [row_off + x] = (r > 0.0) ? mx / r : 0.0;
      loc_uy [row_off + x] = (r > 0.0) ? my / r : 0.0;
    }
  }

  const int recv_count = int(slab);

  if (rank_ == 0) {
    rho_out.resize(nx_ * ny_);
    ux_out .resize(nx_ * ny_);
    uy_out .resize(nx_ * ny_);
    vor_out.resize(nx_ * ny_, 0.0);
  }

  MPI_Gather(loc_rho.data(), recv_count, MPI_DOUBLE,
             rank_ == 0 ? rho_out.data() : nullptr, recv_count, MPI_DOUBLE,
             0, comm_);
  MPI_Gather(loc_ux .data(), recv_count, MPI_DOUBLE,
             rank_ == 0 ? ux_out.data()  : nullptr, recv_count, MPI_DOUBLE,
             0, comm_);
  MPI_Gather(loc_uy .data(), recv_count, MPI_DOUBLE,
             rank_ == 0 ? uy_out.data()  : nullptr, recv_count, MPI_DOUBLE,
             0, comm_);

  // Rank 0 computes vorticity from the full ux/uy arrays.
  if (rank_ == 0) {
    for (std::size_t gy = 0; gy < ny_; ++gy) {
      for (std::size_t x = 0; x < nx_; ++x) {
        if (x == 0 || x == nx_ - 1 || gy == 0 || gy == ny_ - 1) continue;
        const std::size_t k  = gy * nx_ + x;
        // ∂uy/∂x − ∂ux/∂y (centred differences)
        vor_out[k] = 0.5 * ((uy_out[k + 1]  - uy_out[k - 1]) -
                             (ux_out[k + nx_] - ux_out[k - nx_]));
      }
    }
  }
}

void
LBM::gather_solid(std::vector<std::uint8_t>& solid_out) const
{
  const std::size_t slab = local_ny_ * nx_;

  // Pack data rows (local_y = 1..local_ny_) contiguously.
  std::vector<std::uint8_t> loc(slab);
  for (std::size_t local_y = 1; local_y <= local_ny_; ++local_y)
    std::memcpy(&loc[(local_y - 1) * nx_], &solid_[lidx(0, local_y)], nx_);

  if (rank_ == 0) solid_out.resize(nx_ * ny_);

  MPI_Gather(loc.data(), int(slab), MPI_UNSIGNED_CHAR,
             rank_ == 0 ? solid_out.data() : nullptr, int(slab), MPI_UNSIGNED_CHAR,
             0, comm_);
}

// ---------------------------------------------------------------------------
// Debug print helpers
// ---------------------------------------------------------------------------

// Pull-streaming for one cell, used in collide stream but isolated for debug examinations of post-stream 
// pre-collision populations using print_post_stream_pre_collision. Ghost rows must be fresh (after exchange_halos).
void
LBM::gather_streamed(std::size_t x, std::size_t local_y, double f_local[Q]) const
{
  const std::size_t k = lidx(x, local_y);
  for (int i = 0; i < Q; ++i) {
    const long sx        = long(x)       - cx[i];
    const long sy        = long(local_y) - cy[i];
    const long global_sy = long(y_start_) + sy - 1;
    if (sx >= 0 && sx < long(nx_) && global_sy >= 0 && global_sy < long(ny_)) {
      const std::size_t sk = std::size_t(sy) * nx_ + std::size_t(sx);
      f_local[i] = solid_[sk] ? f_[std::size_t(opp[i]) * N_local_ + sk]
                               : f_[std::size_t(i)      * N_local_ + sk];
    } else {
      f_local[i] = f_[std::size_t(i) * N_local_ + k];
    }
  }
}

void
LBM::print_solid_boundary_rows() const
{
  // Bottom wall lives at local_y=1 on rank 0 (global y=0).
  if (y_start_ == 0) {
    std::printf("[MPI rank %d] solid bottom wall (global y=0, local_y=1):", rank_);
    for (std::size_t x = 0; x < nx_; ++x)
      std::printf(" x=%zu:%d", x, solid_[lidx(x, 1)]);
    std::printf("\n");
    std::printf("[MPI rank %d] first fluid row  (global y=1, local_y=2):", rank_);
    for (std::size_t x = 0; x < nx_; ++x)
      std::printf(" x=%zu:%d", x, solid_[lidx(x, 2)]);
    std::printf("\n");
    std::fflush(stdout);
  }
  // Top wall lives at local_y=local_ny_ on the last rank (global y=ny-1).
  if (y_start_ + local_ny_ == ny_) {
    std::printf("[MPI rank %d] last fluid row   (global y=%zu, local_y=%zu):",
                rank_, ny_ - 2, local_ny_ - 1);
    for (std::size_t x = 0; x < nx_; ++x)
      std::printf(" x=%zu:%d", x, solid_[lidx(x, local_ny_ - 1)]);
    std::printf("\n");
    std::printf("[MPI rank %d] solid top wall   (global y=%zu, local_y=%zu):",
                rank_, ny_ - 1, local_ny_);
    for (std::size_t x = 0; x < nx_; ++x)
      std::printf(" x=%zu:%d", x, solid_[lidx(x, local_ny_)]);
    std::printf("\n");
    std::fflush(stdout);
  }
}

void
LBM::print_solid_boundary_rows_ghost() const
{
  // Ghost rows are initialised by each rank independently (not exchanged).
  // global y of bottom ghost = y_start_ - 1 (may be negative for rank 0).
  std::printf("[MPI rank %d] bottom ghost solid (local_y=0, global y=%ld):",
              rank_, long(y_start_) - 1);
  for (std::size_t x = 0; x < nx_; ++x)
    std::printf(" x=%zu:%d", x, solid_[lidx(x, 0)]);
  std::printf("\n");

  std::printf("[MPI rank %d] top ghost solid    (local_y=%zu, global y=%zu):",
              rank_, local_ny_ + 1, y_start_ + local_ny_);
  for (std::size_t x = 0; x < nx_; ++x)
    std::printf(" x=%zu:%d", x, solid_[lidx(x, local_ny_ + 1)]);
  std::printf("\n");
  std::fflush(stdout);
}

void
LBM::print_point_populations(std::size_t x, std::size_t y_global) const
{
  if (y_global < y_start_ || y_global >= y_start_ + local_ny_) return;
  const std::size_t local_y = y_global - y_start_ + 1;
  std::printf("[MPI rank %d] f at (x=%zu, global_y=%zu, local_y=%zu):",
              rank_, x, y_global, local_y);
  for (int i = 0; i < Q; ++i)
    std::printf(" f%s=%.6f", dir_name[i], f_[fidx(i, x, local_y)]);
  std::printf("\n");
  std::fflush(stdout);
}

void
LBM::print_post_stream_pre_collision(std::size_t x, std::size_t y_global) const
{
  if (y_global < y_start_ || y_global >= y_start_ + local_ny_) return;
  const std::size_t local_y = y_global - y_start_ + 1;
  double f_local[Q];
  gather_streamed(x, local_y, f_local);
  std::printf("[MPI rank %d] f_streamed at (x=%zu, global_y=%zu, local_y=%zu):",
              rank_, x, y_global, local_y);
  for (int i = 0; i < Q; ++i)
    std::printf(" f%s=%.6f", dir_name[i], f_local[i]);
  std::printf("\n");
  std::fflush(stdout);
}