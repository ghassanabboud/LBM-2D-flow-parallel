#include "lbm_mpi_cart.hh"

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
  : nx_(nx), ny_(ny),
    local_nx_(0), local_ny_(0),
    x_start_(0),  y_start_(0),
    N_local_(0),
    u_in_(u_in), tau_(0.0),
    rank_(0), nprocs_(0),
    rank_below_(MPI_PROC_NULL), rank_above_(MPI_PROC_NULL),
    rank_west_ (MPI_PROC_NULL), rank_east_ (MPI_PROC_NULL),
    comm_(MPI_COMM_NULL)
{
  MPI_Comm_size(comm, &nprocs_);

  // Find a balanced 2D process grid [prows, pcols].
  cart_dims_[0] = cart_dims_[1] = 0;
  MPI_Dims_create(nprocs_, 2, cart_dims_);

  if (ny % std::size_t(cart_dims_[0]) != 0 ||
      nx % std::size_t(cart_dims_[1]) != 0)
    throw std::runtime_error(
      "nx must be divisible by pcols and ny by prows (from MPI_Dims_create)");

  //  Create Cartesian communicator; reorder=0 keeps rank numbers stable, which simplifies the tile-reordering in gather_fields.
  int periods[2] = {0, 0};
  MPI_Cart_create(comm, 2, cart_dims_, periods, /*reorder=*/0, &comm_);
  MPI_Comm_rank(comm_, &rank_);

  // Get this rank's position in the 2D grid: [row_coord, col_coord] row_coord maps to y (vertical), col_coord maps to x (horizontal).
  MPI_Cart_coords(comm_, rank_, 2, cart_coords_);

  // Compute local tile size and global offsets.
  local_ny_ = ny / std::size_t(cart_dims_[0]);
  local_nx_ = nx / std::size_t(cart_dims_[1]);
  y_start_  = std::size_t(cart_coords_[0]) * local_ny_;
  x_start_  = std::size_t(cart_coords_[1]) * local_nx_;
  // Each tile has one ghost layer on every side.
  N_local_  = (local_nx_ + 2) * (local_ny_ + 2);

  // Find 4 axis-aligned neighbours; MPI_PROC_NULL for domain boundaries.
  MPI_Cart_shift(comm_, 0, 1, &rank_below_, &rank_above_);  // y-axis: south/north
  MPI_Cart_shift(comm_, 1, 1, &rank_west_,  &rank_east_);   // x-axis: west/east

  std::cout << "Rank " << rank_ << " / " << nprocs_
            << " : cart [" << cart_coords_[0] << "," << cart_coords_[1] << "]"
            << "  local_ny = " << local_ny_ << ", y range = ["
            << y_start_ << ", " << (y_start_ + local_ny_ - 1) << "]"
            << "  local_nx = " << local_nx_ << ", x range = ["
            << x_start_ << ", " << (x_start_ + local_nx_ - 1) << "]\n"
            << "  neighbours: S=" << rank_below_ << " N=" << rank_above_
            << " W=" << rank_west_  << " E=" << rank_east_ << "\n";

  // Allocate and zero-initialise (respects first-touch principle).
  f_    .assign(Q * N_local_, 0.0);
  ftmp_ .assign(Q * N_local_, 0.0);
  solid_.assign(N_local_, 0);

  const double nu = u_in_ * (2.0 * cyl_r) / Re;
  tau_ = 3.0 * nu + 0.5;

  // Mark top and bottom physical walls using global y.
  // Only the southernmost row of ranks owns global y=0 (bottom wall),
  // and only the northernmost row owns global y=ny-1 (top wall).
  // The wall cells sit at local_y=1 and local_y=local_ny_ respectively
  // because local_y=0 and local_y=local_ny_+1 are ghost layers.
  for (std::size_t local_x = 0; local_x <= local_nx_ + 1; ++local_x) {
    if (y_start_ == 0)
      solid_[lidx(local_x, 1)] = 1;
    if (y_start_ + local_ny_ == ny_)
      solid_[lidx(local_x, local_ny_)] = 1;
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
  // Includes all ghost layers so bounce-back works correctly across rank
  // boundaries — ghost global coords may be outside the domain but the
  // cylinder equation is still valid there.
  for (std::size_t local_y = 0; local_y <= local_ny_ + 1; ++local_y) {
    const double gy = double(long(y_start_) + long(local_y) - 1);
    for (std::size_t local_x = 0; local_x <= local_nx_ + 1; ++local_x) {
      const double gx = double(long(x_start_) + long(local_x) - 1);
      const double dx = gx - c_x;
      const double dy = gy - c_y;
      if (dx * dx + dy * dy <= r2) {
        solid_[lidx(local_x, local_y)] = 1;
        has_local = true;
      }
    }
  }
  if (has_local)
    std::cout << "Rank " << rank_ << " has part of the cylinder in its local domain.\n";
}

void
LBM::initialize()
{
  for (std::size_t local_y = 1; local_y <= local_ny_; ++local_y) {
    for (std::size_t local_x = 1; local_x <= local_nx_; ++local_x) {
      const std::size_t k  = lidx(local_x, local_y);
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
// Halo exchange — 4 directions (N/S then E/W), two MPI_Sendrecv pairs.
//
// Y-axis (N/S) first: tags 0=upward, 1=downward.
//   Each buffer holds Q * local_nx_ doubles (one row, local_x=1..local_nx_,
//   which is contiguous in memory).
//
// X-axis (E/W) second, including ghost rows: tags 2=eastward, 3=westward.
//   Columns are strided so we pack/unpack manually.
//   Buffer covers local_y=0..local_ny_+1 (col_len = local_ny_+2) so that
//   after the N/S exchange the already-populated ghost rows are also sent
//   east/west. This populates corner ghost cells needed by diagonal D2Q9
//   directions (NE/NW/SE/SW) without requiring separate diagonal exchanges.
//
// MPI_PROC_NULL makes boundary-rank calls no-ops automatically.
// ---------------------------------------------------------------------------
void
LBM::exchange_halos()
{
  const std::size_t row_len = local_nx_;        // doubles per N/S row per direction
  const std::size_t col_len = local_ny_ + 2;    // full column incl. ghost rows
  const std::size_t ns_buf  = Q * row_len;
  const std::size_t ew_buf  = Q * col_len;

  std::vector<double> send_down(ns_buf), recv_from_below(ns_buf, 0.0);
  std::vector<double> send_up  (ns_buf), recv_from_above(ns_buf, 0.0);
  std::vector<double> send_west(ew_buf), recv_from_west (ew_buf, 0.0);
  std::vector<double> send_east(ew_buf), recv_from_east (ew_buf, 0.0);

  // --- Step 1: N/S exchange ---
  // Pack bottom data row (local_y=1) and top data row (local_y=local_ny_).
  // local_x=1..local_nx_ is contiguous within a direction plane row.
  for (int i = 0; i < Q; ++i) {
    std::memcpy(&send_down[std::size_t(i) * row_len],
                &f_[std::size_t(i) * N_local_ + lidx(1, 1)],
                row_len * sizeof(double));
    std::memcpy(&send_up[std::size_t(i) * row_len],
                &f_[std::size_t(i) * N_local_ + lidx(1, local_ny_)],
                row_len * sizeof(double));
  }

  MPI_Status status;

  MPI_Sendrecv(send_down.data(),      int(ns_buf), MPI_DOUBLE, rank_below_, 1,
               recv_from_below.data(), int(ns_buf), MPI_DOUBLE, rank_below_, 0,
               comm_, &status);
  MPI_Sendrecv(send_up.data(),        int(ns_buf), MPI_DOUBLE, rank_above_, 0,
               recv_from_above.data(), int(ns_buf), MPI_DOUBLE, rank_above_, 1,
               comm_, &status);

  // Unpack N/S: ghost rows local_y=0 and local_y=local_ny_+1 at local_x=1..local_nx_.
  for (int i = 0; i < Q; ++i) {
    std::memcpy(&f_[std::size_t(i) * N_local_ + lidx(1, 0)],
                &recv_from_below[std::size_t(i) * row_len],
                row_len * sizeof(double));
    std::memcpy(&f_[std::size_t(i) * N_local_ + lidx(1, local_ny_ + 1)],
                &recv_from_above[std::size_t(i) * row_len],
                row_len * sizeof(double));
  }

  // --- Step 2: E/W exchange (full column including ghost rows) ---
  // Packing ly=0..local_ny_+1 means the ghost rows filled in step 1 are also
  // forwarded to diagonal neighbours, populating corner ghost cells.
  for (int i = 0; i < Q; ++i) {
    for (std::size_t ly = 0; ly <= local_ny_ + 1; ++ly) { //this inner loop is needed because columns are not contiguous in memory
      send_west[std::size_t(i) * col_len + ly] =
          f_[std::size_t(i) * N_local_ + lidx(1,         ly)];
      send_east[std::size_t(i) * col_len + ly] =
          f_[std::size_t(i) * N_local_ + lidx(local_nx_, ly)];
    }
  }

  MPI_Sendrecv(send_west.data(),      int(ew_buf), MPI_DOUBLE, rank_west_,  3,
               recv_from_west.data(),  int(ew_buf), MPI_DOUBLE, rank_west_,  2,
               comm_, &status);
  MPI_Sendrecv(send_east.data(),      int(ew_buf), MPI_DOUBLE, rank_east_,  2,
               recv_from_east.data(),  int(ew_buf), MPI_DOUBLE, rank_east_,  3,
               comm_, &status);

  // Unpack E/W: ghost columns local_x=0 and local_x=local_nx_+1 for all ly.
  for (int i = 0; i < Q; ++i) {
    for (std::size_t ly = 0; ly <= local_ny_ + 1; ++ly) {
      f_[std::size_t(i) * N_local_ + lidx(0,           ly)] =
          recv_from_west[std::size_t(i) * col_len + ly];
      f_[std::size_t(i) * N_local_ + lidx(local_nx_+1, ly)] =
          recv_from_east[std::size_t(i) * col_len + ly];
    }
  }
}


void
LBM::collide_stream()
{
  // Fused pull-stream + BGK collision in a single pass over the local tile.
  // The SoA layout means all cells for a given direction are contiguous in
  // memory, so each streaming gather reads stride-1 within its direction plane.
  const double inv_tau = 1.0 / tau_;

  for (std::size_t local_y = 1; local_y <= local_ny_; ++local_y) {
    for (std::size_t local_x = 1; local_x <= local_nx_; ++local_x) {
      const std::size_t k = lidx(local_x, local_y);

      // Gather streamed populations for cell (local_x, local_y).
      // For each direction i, pull from the neighbour that streams into k:
      //   - fluid neighbour inside domain: take f[i] from that cell
      //   - solid neighbour inside domain: bounce-back from opposite population
      //   - neighbour outside domain: keep current f_[i*N_local_+k]
      // Ghost layers hold fresh halo data so cross-rank neighbours work correctly.
      double f_local[Q];
      gather_streamed(local_x, local_y, f_local);

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
  // Only ranks whose tile touches global x=0 apply the inlet BC.
  if (x_start_ != 0) return;
  const std::size_t local_x = 1;  // global x=0 maps to local_x=1
  for (std::size_t local_y = 1; local_y <= local_ny_; ++local_y) {
    const std::size_t k = lidx(local_x, local_y);
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
  // Only ranks whose tile touches global x=nx_-1 apply the outlet BC.
  if (x_start_ + local_nx_ != nx_) return;
  if (local_nx_ < 2) return;
  const std::size_t local_x  = local_nx_;      // last data column (global x=nx_-1)
  const std::size_t local_xs = local_nx_ - 1;  // second-to-last data column
  for (std::size_t local_y = 1; local_y <= local_ny_; ++local_y) {
    const std::size_t k  = lidx(local_x,  local_y);
    const std::size_t ks = lidx(local_xs, local_y);
    for (int i = 0; i < Q; ++i)
      f_[std::size_t(i) * N_local_ + k] = f_[std::size_t(i) * N_local_ + ks];
  }
}

double
LBM::rho(std::size_t local_x, std::size_t local_y) const
{
  const std::size_t k = lidx(local_x, local_y);
  double r = 0.0;
  for (int i = 0; i < Q; ++i) r += f_[std::size_t(i) * N_local_ + k];
  return r;
}

double
LBM::ux(std::size_t local_x, std::size_t local_y) const
{
  const std::size_t k = lidx(local_x, local_y);
  double r = 0.0, m = 0.0;
  for (int i = 0; i < Q; ++i) {
    const double fi = f_[std::size_t(i) * N_local_ + k];
    r += fi;
    m += cx[i] * fi;
  }
  return (r > 0.0) ? m / r : 0.0;
}

double
LBM::uy(std::size_t local_x, std::size_t local_y) const
{
  const std::size_t k = lidx(local_x, local_y);
  double r = 0.0, m = 0.0;
  for (int i = 0; i < Q; ++i) {
    const double fi = f_[std::size_t(i) * N_local_ + k];
    r += fi;
    m += cy[i] * fi;
  }
  return (r > 0.0) ? m / r : 0.0;
}

bool
LBM::is_solid(std::size_t local_x, std::size_t local_y) const
{
  return solid_[lidx(local_x, local_y)] != 0;
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
  const std::size_t tile = local_nx_ * local_ny_;  // elements per rank per field

  // Compute local macro fields for the data tile (row-major, x varies fastest).
  std::vector<double> loc_rho(tile), loc_ux(tile), loc_uy(tile);
  for (std::size_t local_y = 1; local_y <= local_ny_; ++local_y) {
    const std::size_t row_off = (local_y - 1) * local_nx_;
    for (std::size_t local_x = 1; local_x <= local_nx_; ++local_x) {
      const std::size_t k = lidx(local_x, local_y);
      double r = 0.0, mx = 0.0, my = 0.0;
      for (int i = 0; i < Q; ++i) {
        const double fi = f_[std::size_t(i) * N_local_ + k];
        r  += fi;
        mx += cx[i] * fi;
        my += cy[i] * fi;
      }
      loc_rho[row_off + (local_x - 1)] = r;
      loc_ux [row_off + (local_x - 1)] = (r > 0.0) ? mx / r : 0.0;
      loc_uy [row_off + (local_x - 1)] = (r > 0.0) ? my / r : 0.0;
    }
  }

  const int recv_count = int(tile);

  // Gather all tiles to rank 0.  With reorder=0 in MPI_Cart_create, rank r in
  // comm_ corresponds to cart coords given by MPI_Cart_coords(comm_, r, ...).
  std::vector<double> all_rho, all_ux, all_uy;
  if (rank_ == 0) {
    rho_out.resize(nx_ * ny_);
    ux_out .resize(nx_ * ny_);
    uy_out .resize(nx_ * ny_);
    vor_out.resize(nx_ * ny_, 0.0);
    all_rho.resize(std::size_t(nprocs_) * tile);
    all_ux .resize(std::size_t(nprocs_) * tile);
    all_uy .resize(std::size_t(nprocs_) * tile);
  }

  MPI_Gather(loc_rho.data(), recv_count, MPI_DOUBLE,
             rank_ == 0 ? all_rho.data() : nullptr, recv_count, MPI_DOUBLE,
             0, comm_);
  MPI_Gather(loc_ux .data(), recv_count, MPI_DOUBLE,
             rank_ == 0 ? all_ux.data()  : nullptr, recv_count, MPI_DOUBLE,
             0, comm_);
  MPI_Gather(loc_uy .data(), recv_count, MPI_DOUBLE,
             rank_ == 0 ? all_uy.data()  : nullptr, recv_count, MPI_DOUBLE,
             0, comm_);

  // Rank 0: place each tile at its correct global position, then compute vorticity.
  if (rank_ == 0) {
    for (int r = 0; r < nprocs_; ++r) {
      int coords[2];
      MPI_Cart_coords(comm_, r, 2, coords);  // [row_coord, col_coord]
      const std::size_t gx0 = std::size_t(coords[1]) * local_nx_;  // global x start of tile
      const std::size_t gy0 = std::size_t(coords[0]) * local_ny_;  // global y start of tile
      const double* src_rho = all_rho.data() + std::size_t(r) * tile;
      const double* src_ux  = all_ux .data() + std::size_t(r) * tile;
      const double* src_uy  = all_uy .data() + std::size_t(r) * tile;
      for (std::size_t ly = 0; ly < local_ny_; ++ly) {
        for (std::size_t lx = 0; lx < local_nx_; ++lx) {
          const std::size_t gk = (gy0 + ly) * nx_ + (gx0 + lx);
          rho_out[gk] = src_rho[ly * local_nx_ + lx];
          ux_out [gk] = src_ux [ly * local_nx_ + lx];
          uy_out [gk] = src_uy [ly * local_nx_ + lx];
        }
      }
    }

    // ∂uy/∂x − ∂ux/∂y (centred differences)
    for (std::size_t gy = 0; gy < ny_; ++gy) {
      for (std::size_t x = 0; x < nx_; ++x) {
        if (x == 0 || x == nx_ - 1 || gy == 0 || gy == ny_ - 1) continue;
        const std::size_t k  = gy * nx_ + x;
        vor_out[k] = 0.5 * ((uy_out[k + 1]   - uy_out[k - 1]) -
                             (ux_out[k + nx_] - ux_out[k - nx_]));
      }
    }
  }
}

void
LBM::gather_solid(std::vector<std::uint8_t>& solid_out) const
{
  const std::size_t tile = local_nx_ * local_ny_;

  // Pack data columns (local_x=1..local_nx_) row by row into a contiguous buffer.
  std::vector<std::uint8_t> loc(tile);
  for (std::size_t local_y = 1; local_y <= local_ny_; ++local_y)
    for (std::size_t local_x = 1; local_x <= local_nx_; ++local_x)
      loc[(local_y - 1) * local_nx_ + (local_x - 1)] = solid_[lidx(local_x, local_y)];

  std::vector<std::uint8_t> all_solid;
  if (rank_ == 0) {
    solid_out.resize(nx_ * ny_);
    all_solid.resize(std::size_t(nprocs_) * tile);
  }

  MPI_Gather(loc.data(), int(tile), MPI_UNSIGNED_CHAR,
             rank_ == 0 ? all_solid.data() : nullptr, int(tile), MPI_UNSIGNED_CHAR,
             0, comm_);

  if (rank_ == 0) {
    for (int r = 0; r < nprocs_; ++r) {
      int coords[2];
      MPI_Cart_coords(comm_, r, 2, coords);
      const std::size_t gx0 = std::size_t(coords[1]) * local_nx_;
      const std::size_t gy0 = std::size_t(coords[0]) * local_ny_;
      const std::uint8_t* src = all_solid.data() + std::size_t(r) * tile;
      for (std::size_t ly = 0; ly < local_ny_; ++ly)
        for (std::size_t lx = 0; lx < local_nx_; ++lx)
          solid_out[(gy0 + ly) * nx_ + (gx0 + lx)] = src[ly * local_nx_ + lx];
    }
  }
}

// ---------------------------------------------------------------------------
// Debug print helpers (rank-aware)
// ---------------------------------------------------------------------------

// Pull-streaming for one cell, used in collide stream but isolated for debug examinations of post-stream 
// pre-collision populations using print_post_stream_pre_collision. Ghost rows must be fresh (after exchange_halos).
void
LBM::gather_streamed(std::size_t local_x, std::size_t local_y, double f_local[Q]) const
{
  const std::size_t k = lidx(local_x, local_y);
  for (int i = 0; i < Q; ++i) {
    const long sx        = long(local_x) - cx[i];
    const long sy        = long(local_y) - cy[i];
    const long global_sx = long(x_start_) + sx - 1;
    const long global_sy = long(y_start_) + sy - 1;
    // In-domain check covers both data cells and valid ghost cells.
    // Ghost cells from boundary ranks (global coord < 0 or >= nx_/ny_) are
    // outside the domain and fall back to keeping the current value.
    if (global_sx >= 0 && global_sx < long(nx_) &&
        global_sy >= 0 && global_sy < long(ny_)) {
      const std::size_t sk = lidx(std::size_t(sx), std::size_t(sy));
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
  if (y_start_ == 0) {
    std::printf("[MPI rank %d] solid bottom wall (global y=0, local_y=1):", rank_);
    for (std::size_t lx = 1; lx <= local_nx_; ++lx)
      std::printf(" x=%zu:%d", x_start_ + lx - 1, solid_[lidx(lx, 1)]);
    std::printf("\n");
    std::fflush(stdout);
  }
  if (y_start_ + local_ny_ == ny_) {
    std::printf("[MPI rank %d] solid top wall (global y=%zu, local_y=%zu):",
                rank_, ny_ - 1, local_ny_);
    for (std::size_t lx = 1; lx <= local_nx_; ++lx)
      std::printf(" x=%zu:%d", x_start_ + lx - 1, solid_[lidx(lx, local_ny_)]);
    std::printf("\n");
    std::fflush(stdout);
  }
}

void
LBM::print_solid_boundary_rows_ghost() const
{
  std::printf("[MPI rank %d] bottom ghost solid (local_y=0, global y=%ld):",
              rank_, long(y_start_) - 1);
  for (std::size_t lx = 1; lx <= local_nx_; ++lx)
    std::printf(" x=%zu:%d", x_start_ + lx - 1, solid_[lidx(lx, 0)]);
  std::printf("\n");

  std::printf("[MPI rank %d] top ghost solid (local_y=%zu, global y=%zu):",
              rank_, local_ny_ + 1, y_start_ + local_ny_);
  for (std::size_t lx = 1; lx <= local_nx_; ++lx)
    std::printf(" x=%zu:%d", x_start_ + lx - 1, solid_[lidx(lx, local_ny_ + 1)]);
  std::printf("\n");
  std::fflush(stdout);
}

void
LBM::print_point_populations(std::size_t x_global, std::size_t y_global) const
{
  if (x_global < x_start_ || x_global >= x_start_ + local_nx_) return;
  if (y_global < y_start_ || y_global >= y_start_ + local_ny_) return;
  const std::size_t local_x = x_global - x_start_ + 1;
  const std::size_t local_y = y_global - y_start_ + 1;
  std::printf("[MPI rank %d] f at (global_x=%zu, global_y=%zu, lx=%zu, ly=%zu):",
              rank_, x_global, y_global, local_x, local_y);
  for (int i = 0; i < Q; ++i)
    std::printf(" f%s=%.6f", dir_name[i], f_[fidx(i, local_x, local_y)]);
  std::printf("\n");
  std::fflush(stdout);
}

void
LBM::print_post_stream_pre_collision(std::size_t x_global, std::size_t y_global) const
{
  if (x_global < x_start_ || x_global >= x_start_ + local_nx_) return;
  if (y_global < y_start_ || y_global >= y_start_ + local_ny_) return;
  const std::size_t local_x = x_global - x_start_ + 1;
  const std::size_t local_y = y_global - y_start_ + 1;
  double f_local[Q];
  gather_streamed(local_x, local_y, f_local);
  std::printf("[MPI rank %d] f_streamed at (global_x=%zu, global_y=%zu, lx=%zu, ly=%zu):",
              rank_, x_global, y_global, local_x, local_y);
  for (int i = 0; i < Q; ++i)
    std::printf(" f%s=%.6f", dir_name[i], f_local[i]);
  std::printf("\n");
  std::fflush(stdout);
}
