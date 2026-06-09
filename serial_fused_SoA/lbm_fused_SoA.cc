#include "lbm_fused_SoA.hh"

#include <algorithm>
#include <cstdio>

// D2Q9 lattice constants. Indexing convention used throughout:
//   0: rest         5: NE
//   1: E            6: NW
//   2: N            7: SW
//   3: W            8: SE
//   4: S
const int LBM::cx[9] = { 0,  1,  0, -1,  0,  1, -1, -1,  1};
const int LBM::cy[9] = { 0,  0,  1,  0, -1,  1,  1, -1, -1};

const double LBM::w[9] = {
  4.0 / 9.0,
  1.0 / 9.0,  1.0 / 9.0,  1.0 / 9.0,  1.0 / 9.0,
  1.0 / 36.0, 1.0 / 36.0, 1.0 / 36.0, 1.0 / 36.0
};

const int LBM::opp[9] = {0, 3, 4, 1, 2, 7, 8, 5, 6};

const char* const LBM::dir_name[9] = {"C", "E", "N", "W", "S", "NE", "NW", "SW", "SE"};

LBM::LBM(std::size_t nx, std::size_t ny,
         double u_in, double Re,
         double cyl_x, double cyl_y, double cyl_r)
  : nx_(nx), ny_(ny), u_in_(u_in), tau_(0.0), // this is the constructor, initializing attributes ending in "_"
    f_   (Q * nx * ny, 0.0), //this is equivalent to a call to the vector constructor std::vector<double>(count, value))
    ftmp_(Q * nx * ny, 0.0),
    solid_(nx * ny, 0)
{
  // ν = c_s^2 (τ - 1/2) with c_s^2 = 1/3, and Re = u_in * D / ν.
  // the Reynolds number to be simulated is given as argument because it is a physically meaningful quantity
  // for the BGK-LBM scheme, the relaxation time tau for the updates is calculated from the viscosity nu, which is in turn calculated from the Reynolds number
  // inlet velocity and cylinder diameter. However, tau must be > 0.5 for stability, which limits the minimum viscosity and maximum Reynolds number that can be simulated at a given resolution and inlet velocity.
  const double nu = u_in_ * (2.0 * cyl_r) / Re;
  tau_ = 3.0 * nu + 0.5;

  // No-slip top and bottom walls.
  for (std::size_t x = 0; x < nx_; ++x) {
    solid_[idx(x, 0)]        = 1;
    solid_[idx(x, ny_ - 1)]  = 1;
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
  //this function iterates over all the grid points and marks as solid those that are within the radius of the cylinder centered at (c_x, c_y)
  const double r2 = r * r;
  for (std::size_t y = 0; y < ny_; ++y) {
    for (std::size_t x = 0; x < nx_; ++x) {
      const double dx = double(x) - c_x;
      const double dy = double(y) - c_y;
      if (dx * dx + dy * dy <= r2) solid_[idx(x, y)] = 1;
    }
  }
}

void
LBM::initialize()
{
  for (std::size_t y = 0; y < ny_; ++y) {
    for (std::size_t x = 0; x < nx_; ++x) {
      // we initialize the entire domain to a horizontal velocity matching the inlet
      // this allows to reach the vortex regime faster.
      const double rho = 1.0;
      const double ux  = solid_[idx(x, y)] ? 0.0 : u_in_; // No initial flow in solid cells.
      const double uy  = 0.0;
      const double u2  = ux * ux + uy * uy;
      const std::size_t k = idx(x, y);
      for (int i = 0; i < Q; ++i) {
        const double cu  = cx[i] * ux + cy[i] * uy;
        const double feq = w[i] * rho * (1.0 + 3.0 * cu + 4.5 * cu * cu - 1.5 * u2);
        // SoA layout: all cells for direction i are contiguous at f_[i*N + k]
        f_[std::size_t(i) * nx_ * ny_ + k] = feq;
      }
    }
  }
}

void
LBM::step()
{
  collide_stream();
  apply_inlet();
  apply_outlet();
}

void
LBM::collide_stream()
{
  // Fused pull-stream + BGK collision in a single pass over the grid.
  // The SoA layout means all cells for a given direction are contiguous in
  // memory, so each streaming gather reads stride-1 within its direction plane.
  const double inv_tau = 1.0 / tau_;

  for (std::size_t y = 0; y < ny_; ++y) {
    for (std::size_t x = 0; x < nx_; ++x) {
      const std::size_t k = idx(x, y);

      // this implementation does streaming and then collision in a single pass
      // first, we gather for cell k, the neighbouring position streams that are coming our way. There are three cases then:
      // - if the neighbour is fluid and inside the domain, we take the value from that cell and population
      // - if the neighbour is solid and inside the domain, we take the value from the opposite population of that cell (bounce-back)
      // - if the neighbour is outside the domain, we keep the current value in f_local, which will be overwritten if on inlet/outlet and is never used otherwise (e.g. on the top/bottom walls)


      // once we have gathered all of this into f_local, we have all the data we need to do the collision step for cell k before writing the post-collision values to ftmp_
      // we still need to have two arrays because changing f_ directly would change the values used for the next cells within one step
      

      double f_local[Q];

      // when we move k by one, all neighbours move by one as well, aka subsequent iterations of k create 9 sliding "streams"
      // the 9 accesses within one cell iteration jump between planes, but each plane forms a stride-1 stream across cell iterations.
      gather_streamed(x, y, f_local);

      // Collision over the new populations we just pulled
      double rho = 0.0, mx = 0.0, my = 0.0;
      for (int i = 0; i < Q; ++i) {
        rho += f_local[i];           //zeroth moment of distribution, aka density
        mx  += cx[i] * f_local[i];  //momentum in x direction, first moment of distribution
        my  += cy[i] * f_local[i];  //momentum in y direction, first moment of distribution
      }

      // macroscopic velocity components obtained by dividing momentum by density
      const double ux = (rho > 0.0) ? mx / rho : 0.0;
      const double uy = (rho > 0.0) ? my / rho : 0.0;
      const double u2 = ux * ux + uy * uy;

      // Write post-collision populations to ftmp_[i*N + k] (SoA).
      // same comments about sliding streams here
      for (int i = 0; i < Q; ++i) {
        const double cu  = cx[i] * ux + cy[i] * uy;
        const double feq = w[i] * rho * (1.0 + 3.0 * cu + 4.5 * cu * cu - 1.5 * u2);
        ftmp_[std::size_t(i) * nx_ * ny_ + k] = f_local[i] - inv_tau * (f_local[i] - feq);
      }
    }
  }

  f_.swap(ftmp_); //swap pointers to reuse ftmp_ as the new distribution array without copying data
}

void
LBM::apply_inlet()
{
  // Reset the inlet column to the equilibrium distribution corresponding
  // to a prescribed uniform velocity (u_in, 0) and unit density.
  const std::size_t x = 0;
  for (std::size_t y = 0; y < ny_; ++y) {
    if (solid_[idx(x, y)]) continue;
    const double rho = 1.0;
    const double ux  = u_in_;
    const double uy  = 0.0;
    const double u2  = ux * ux + uy * uy;
    const std::size_t k = idx(x, y);
    for (int i = 0; i < Q; ++i) {
      const double cu  = cx[i] * ux + cy[i] * uy;
      f_[std::size_t(i) * nx_ * ny_ + k] = w[i] * rho * (1.0 + 3.0 * cu + 4.5 * cu * cu - 1.5 * u2);
    }
  }
}

void
LBM::apply_outlet()
{
  // Zero-gradient outlet: copy the second-to-last column into the last.
  if (nx_ < 2) return;
  const std::size_t x  = nx_ - 1;
  const std::size_t xs = nx_ - 2;
  for (std::size_t y = 0; y < ny_; ++y) {
    const std::size_t k  = idx(x,  y);
    const std::size_t ks = idx(xs, y);
    for (int i = 0; i < Q; ++i) {
      f_[std::size_t(i) * nx_ * ny_ + k] = f_[std::size_t(i) * nx_ * ny_ + ks];
    }
  }
}

double
LBM::rho(std::size_t x, std::size_t y) const
{
  const std::size_t k = idx(x, y);
  double r = 0.0;
  for (int i = 0; i < Q; ++i) r += f_[std::size_t(i) * nx_ * ny_ + k];
  return r;
}

double
LBM::ux(std::size_t x, std::size_t y) const
{
  const std::size_t k = idx(x, y);
  double r = 0.0, m = 0.0;
  for (int i = 0; i < Q; ++i) {
    const double fi = f_[std::size_t(i) * nx_ * ny_ + k];
    r += fi;
    m += cx[i] * fi;
  }
  return (r > 0.0) ? m / r : 0.0;
}

double
LBM::uy(std::size_t x, std::size_t y) const
{
  const std::size_t k = idx(x, y);
  double r = 0.0, m = 0.0;
  for (int i = 0; i < Q; ++i) {
    const double fi = f_[std::size_t(i) * nx_ * ny_ + k];
    r += fi;
    m += cy[i] * fi;
  }
  return (r > 0.0) ? m / r : 0.0;
}

double
LBM::vorticity(std::size_t x, std::size_t y) const
{
  if (x == 0 || x == nx_ - 1 || y == 0 || y == ny_ - 1) return 0.0;
  return 0.5 * ((uy(x + 1, y) - uy(x - 1, y)) - (ux(x, y + 1) - ux(x, y - 1)));
}

bool
LBM::is_solid(std::size_t x, std::size_t y) const
{
  return solid_[idx(x, y)] != 0;
}

void
LBM::print_solid_boundary_rows() const
{
  std::printf("[SoA] solid bottom row (y=0):  ");
  for (std::size_t x = 0; x < nx_; ++x)
    std::printf(" x=%zu:%d", x, solid_[idx(x, 0)]);
  std::printf("\n");

  std::printf("[SoA] first fluid row on bottom (y=1):  ");
  for (std::size_t x = 0; x < nx_; ++x)
    std::printf(" x=%zu:%d", x, solid_[idx(x, 1)]);
  std::printf("\n");

  std::printf("[SoA] solid top row (y=%zu): ", ny_ - 1);
  for (std::size_t x = 0; x < nx_; ++x)
    std::printf(" x=%zu:%d", x, solid_[idx(x, ny_ - 1)]);
  std::printf("\n");
  std::fflush(stdout);

  std::printf("[SoA] last fluid row on top (y=%zu): ", ny_ - 2);
  for (std::size_t x = 0; x < nx_; ++x)
    std::printf(" x=%zu:%d", x, solid_[idx(x, ny_ - 2)]);
  std::printf("\n");
  std::fflush(stdout);
}

void
LBM::print_point_populations(std::size_t x, std::size_t y) const
{
  std::printf("[SoA] f at (x=%zu, y=%zu):", x, y);
  for (int i = 0; i < Q; ++i)
    std::printf(" f%s=%.6f", dir_name[i], f_[fidx(i, x, y)]);
  std::printf("\n");
  std::fflush(stdout);
}

void
LBM::gather_streamed(std::size_t x, std::size_t y, double f_local[Q]) const
{
  const std::size_t k = idx(x, y);
  for (int i = 0; i < Q; ++i) {
    // cast to signed so that e.g. x=0, cx[i]=1 gives sx=-1 rather than wrapping unsigned
    const long sx = long(x) - cx[i];
    const long sy = long(y) - cy[i];
    if (sx >= 0 && sx < long(nx_) && sy >= 0 && sy < long(ny_)) {
      const std::size_t sk = idx(std::size_t(sx), std::size_t(sy));
      f_local[i] = solid_[sk] ? f_[std::size_t(opp[i]) * nx_ * ny_ + sk]
                              : f_[std::size_t(i)      * nx_ * ny_ + sk];
    } else {
      f_local[i] = f_[std::size_t(i) * nx_ * ny_ + k];
    }
  }
}

void
LBM::print_post_stream_pre_collision(std::size_t x, std::size_t y) const
{
  double f_local[Q];
  gather_streamed(x, y, f_local);
  std::printf("[SoA] f_streamed at (x=%zu, y=%zu):", x, y);
  for (int i = 0; i < Q; ++i)
    std::printf(" f%s=%.6f", dir_name[i], f_local[i]);
  std::printf("\n");
  std::fflush(stdout);
}
