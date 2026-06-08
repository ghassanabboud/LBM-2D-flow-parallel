#include "lbm_cuda_SoA.cuh"

#include <cstdio>
#include <cstdlib>
#include <cstring>

// ---------------------------------------------------------------------------
// Error-checking macro for CUDA API calls.
// ---------------------------------------------------------------------------
#define CUDA_CHECK(call)                                                        \
  do {                                                                          \
    cudaError_t _e = (call);                                                    \
    if (_e != cudaSuccess) {                                                    \
      std::fprintf(stderr, "CUDA error at %s:%d — %s\n",                       \
                   __FILE__, __LINE__, cudaGetErrorString(_e));                 \
      std::exit(1);                                                             \
    }                                                                           \
  } while (0)

// ---------------------------------------------------------------------------
// D2Q9 lattice constants in __constant__ memory.
// All 32 threads in a warp read the same index each iteration, so every
// access hits the constant-cache broadcast path — zero extra HBM2 traffic.
// Indexing convention:
//   0: rest   1: E   2: N   3: W   4: S   5: NE   6: NW   7: SW   8: SE
// ---------------------------------------------------------------------------
__constant__ int    d_cx[9]  = { 0,  1,  0, -1,  0,  1, -1, -1,  1};
__constant__ int    d_cy[9]  = { 0,  0,  1,  0, -1,  1,  1, -1, -1};
__constant__ double d_w[9]   = {
  4.0 / 9.0,
  1.0 / 9.0,  1.0 / 9.0,  1.0 / 9.0,  1.0 / 9.0,
  1.0 / 36.0, 1.0 / 36.0, 1.0 / 36.0, 1.0 / 36.0
};
__constant__ int    d_opp[9] = {0, 3, 4, 1, 2, 7, 8, 5, 6};

// Host-side copies used to initialise the __constant__ arrays and to
// compute moments in fetch_probe_uxy() without duplicating the values.
static const int    h_cx[9]  = { 0,  1,  0, -1,  0,  1, -1, -1,  1};
static const int    h_cy[9]  = { 0,  0,  1,  0, -1,  1,  1, -1, -1};
static const double h_w[9]   = {
  4.0 / 9.0,
  1.0 / 9.0,  1.0 / 9.0,  1.0 / 9.0,  1.0 / 9.0,
  1.0 / 36.0, 1.0 / 36.0, 1.0 / 36.0, 1.0 / 36.0
};
static const int    h_opp[9] = {0, 3, 4, 1, 2, 7, 8, 5, 6};

// ---------------------------------------------------------------------------
// Kernel 1: fused pull-stream + BGK collision.
//
// One thread per cell.  Supports both 1D and 2D block/grid configurations:
//
//   1D (block_y = 1): thread k = blockIdx.x * blockDim.x + threadIdx.x maps
//     to cell k; x = k % nx, y = k / nx.  Consecutive threads in a warp
//     have consecutive k → consecutive x at fixed y, giving stride-1 reads.
//
//   2D (block_y > 1): thread (tx, ty) in block (bx, by) maps to cell
//     (x = bx*blockDim.x + tx, y = by*blockDim.y + ty).  Coalescing is
//     preserved when blockDim.x is a multiple of 32; if it is not (e.g.
//     16×16 blocks), each warp wraps two rows and reads are uncoalesced.
//
// The SoA layout d_f[i*N + k] ensures stride-1 access in x for both modes
// when blockDim.x >= 32.
//
// d_solid is accessed via __ldg() which routes through the read-only L1 cache.
// ---------------------------------------------------------------------------
__global__ void
lbm_collide_stream_kernel(const double* __restrict__ d_f,
                           double*       __restrict__ d_ftmp,
                           const uint8_t*             d_solid,
                           int nx, int ny, int N,
                           double inv_tau)
{
  const int x = blockIdx.x * blockDim.x + threadIdx.x;
  const int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x >= nx || y >= ny) return;

  const int k = y * nx + x;

  // Pull streaming: for each direction i we read from the upstream neighbour.
  // Three cases (matching serial gather_streamed):
  //   - neighbour inside domain and fluid  → take f[i, neighbour]
  //   - neighbour inside domain and solid  → bounce-back: take f[opp[i], neighbour]
  //   - neighbour outside domain           → keep current f[i, k] (overwritten
  //                                          for inlet/outlet by the next kernels)
  double f_local[9];

  for (int i = 0; i < 9; ++i) {
    const int sx = x - d_cx[i];
    const int sy = y - d_cy[i];
    if (sx >= 0 && sx < nx && sy >= 0 && sy < ny) {
      const int sk = sy * nx + sx;
      f_local[i] = __ldg(&d_solid[sk])
                   ? d_f[d_opp[i] * N + sk]   // bounce-back
                   : d_f[i       * N + sk];   // fluid pull
    } else {
      f_local[i] = d_f[i * N + k];            // out-of-domain: keep current
    }
  }

  // Compute macroscopic moments from the gathered populations.
  double rho = 0.0, mx = 0.0, my = 0.0;
  for (int i = 0; i < 9; ++i) {
    rho += f_local[i];
    mx  += d_cx[i] * f_local[i];
    my  += d_cy[i] * f_local[i];
  }

  const double ux = (rho > 0.0) ? mx / rho : 0.0;
  const double uy = (rho > 0.0) ? my / rho : 0.0;
  const double u2 = ux * ux + uy * uy;

  // BGK relaxation towards equilibrium; write post-collision values to d_ftmp.
  // SoA write: d_ftmp[i*N + k] — stride-1 across consecutive k values.
  for (int i = 0; i < 9; ++i) {
    const double cu  = d_cx[i] * ux + d_cy[i] * uy;
    const double feq = d_w[i] * rho * (1.0 + 3.0*cu + 4.5*cu*cu - 1.5*u2);
    d_ftmp[i * N + k] = f_local[i] - inv_tau * (f_local[i] - feq);
  }
}

// ---------------------------------------------------------------------------
// Kernel 2: inlet boundary condition (x = 0).
//
// Resets the left column of d_f to the equilibrium distribution for a uniform
// horizontal velocity (u_in, 0) at unit density, matching the serial
// apply_inlet().  Operates on d_ftmp (the output buffer of kernel 1).
// ---------------------------------------------------------------------------
__global__ void
lbm_inlet_kernel(double*       d_f,
                 const uint8_t* d_solid,
                 int nx, int ny, int N,
                 double u_in)
{
  const int y = blockIdx.x * blockDim.x + threadIdx.x;
  if (y >= ny) return;

  const int k = y * nx;   // x = 0
  if (__ldg(&d_solid[k])) return;

  const double rho = 1.0;
  const double ux  = u_in;
  const double u2  = ux * ux;   // uy = 0

  for (int i = 0; i < 9; ++i) {
    const double cu  = d_cx[i] * ux;   // d_cy[i] * uy = 0
    d_f[i * N + k] = d_w[i] * rho * (1.0 + 3.0*cu + 4.5*cu*cu - 1.5*u2);
  }
}

// ---------------------------------------------------------------------------
// Kernel 3: outlet boundary condition (x = nx-1).
//
// Zero-gradient: copies populations from the column at x = nx-2 into
// x = nx-1, matching the serial apply_outlet().  Operates on d_ftmp.
// ---------------------------------------------------------------------------
__global__ void
lbm_outlet_kernel(double* d_f,
                  int nx, int ny, int N)
{
  const int y = blockIdx.x * blockDim.x + threadIdx.x;
  if (y >= ny) return;

  const int k  = y * nx + (nx - 1);
  const int ks = y * nx + (nx - 2);

  for (int i = 0; i < 9; ++i)
    d_f[i * N + k] = d_f[i * N + ks];
}

// ===========================================================================
// LBM class implementation
// ===========================================================================

LBM::LBM(std::size_t nx, std::size_t ny,
         double u_in, double Re,
         double cyl_x, double cyl_y, double cyl_r)
  : nx_(nx), ny_(ny), N_(nx * ny),
    u_in_(u_in), tau_(0.0),
    d_f_(nullptr), d_ftmp_(nullptr), d_solid_(nullptr),
    h_f_(Q * nx * ny, 0.0),
    h_solid_(nx * ny, 0)
{
  // ν = c_s^2 (τ - 1/2) with c_s^2 = 1/3, and Re = u_in * D / ν.
  const double nu = u_in_ * (2.0 * cyl_r) / Re;
  tau_ = 3.0 * nu + 0.5;

  // Upload D2Q9 constants to __constant__ memory once.
  CUDA_CHECK(cudaMemcpyToSymbol(d_cx,  h_cx,  9 * sizeof(int)));
  CUDA_CHECK(cudaMemcpyToSymbol(d_cy,  h_cy,  9 * sizeof(int)));
  CUDA_CHECK(cudaMemcpyToSymbol(d_w,   h_w,   9 * sizeof(double)));
  CUDA_CHECK(cudaMemcpyToSymbol(d_opp, h_opp, 9 * sizeof(int)));

  // Allocate device arrays.
  CUDA_CHECK(cudaMalloc(&d_f_,     Q * N_ * sizeof(double)));
  CUDA_CHECK(cudaMalloc(&d_ftmp_,  Q * N_ * sizeof(double)));
  CUDA_CHECK(cudaMalloc(&d_solid_, N_ * sizeof(uint8_t)));

  // Build solid mask on host.
  for (std::size_t x = 0; x < nx_; ++x) {
    h_solid_[idx(x, 0)]       = 1;   // bottom wall
    h_solid_[idx(x, ny_ - 1)] = 1;   // top wall
  }
  mark_obstacle(cyl_x, cyl_y, cyl_r);

  // Upload solid mask to device.
  CUDA_CHECK(cudaMemcpy(d_solid_, h_solid_.data(),
                        N_ * sizeof(uint8_t), cudaMemcpyHostToDevice));
}

LBM::~LBM()
{
  cudaFree(d_f_);
  cudaFree(d_ftmp_);
  cudaFree(d_solid_);
}

void
LBM::add_second_cylinder(double cx2, double cy2, double r2)
{
  if (r2 > 0.0) {
    mark_obstacle(cx2, cy2, r2);
    CUDA_CHECK(cudaMemcpy(d_solid_, h_solid_.data(),
                          N_ * sizeof(uint8_t), cudaMemcpyHostToDevice));
  }
}

void
LBM::mark_obstacle(double c_x, double c_y, double r)
{
  const double r2 = r * r;
  for (std::size_t y = 0; y < ny_; ++y)
    for (std::size_t x = 0; x < nx_; ++x) {
      const double dx = double(x) - c_x;
      const double dy = double(y) - c_y;
      if (dx * dx + dy * dy <= r2) h_solid_[idx(x, y)] = 1;
    }
}

void
LBM::initialize()
{
  // Initialise the entire domain to equilibrium at (rho=1, ux=u_in, uy=0) for
  // fluid cells and (rho=1, u=0) for solid cells, to reach vortex shedding
  // faster than starting from rest.
  for (std::size_t y = 0; y < ny_; ++y) {
    for (std::size_t x = 0; x < nx_; ++x) {
      const double rho = 1.0;
      const double ux  = h_solid_[idx(x, y)] ? 0.0 : u_in_;
      const double uy  = 0.0;
      const double u2  = ux * ux + uy * uy;
      const std::size_t k = idx(x, y);
      for (int i = 0; i < Q; ++i) {
        const double cu  = h_cx[i] * ux + h_cy[i] * uy;
        const double feq = h_w[i] * rho * (1.0 + 3.0*cu + 4.5*cu*cu - 1.5*u2);
        h_f_[std::size_t(i) * N_ + k] = feq;
      }
    }
  }
  CUDA_CHECK(cudaMemcpy(d_f_, h_f_.data(),
                        Q * N_ * sizeof(double), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_ftmp_, h_f_.data(),
                        Q * N_ * sizeof(double), cudaMemcpyHostToDevice));
}

void
LBM::step(int block_x, int block_y, int grid_x, int grid_y)
{
  const dim3 block(block_x, block_y);
  const dim3 grid (grid_x,  grid_y);

  lbm_collide_stream_kernel<<<grid, block>>>(
      d_f_, d_ftmp_, d_solid_,
      int(nx_), int(ny_), int(N_), 1.0 / tau_);

  // Inlet and outlet are 1D over y: flatten block to total threads per block
  // so they work regardless of whether the caller chose a 1D or 2D block.
  const int bc_block = int(block.x) * int(block.y);
  const int bc_grid  = (int(ny_) + bc_block - 1) / bc_block;
  lbm_inlet_kernel <<<bc_grid, bc_block>>>(d_ftmp_, d_solid_,
                                            int(nx_), int(ny_), int(N_), u_in_);
  lbm_outlet_kernel<<<bc_grid, bc_block>>>(d_ftmp_,
                                            int(nx_), int(ny_), int(N_));

  // Swap device pointers — no data movement, O(1) cost.
  double* tmp = d_f_;
  d_f_        = d_ftmp_;
  d_ftmp_     = tmp;
}

void
LBM::sync_to_host()
{
  CUDA_CHECK(cudaMemcpy(h_f_.data(), d_f_,
                        Q * N_ * sizeof(double), cudaMemcpyDeviceToHost));
}

void
LBM::fetch_probe_uxy(std::size_t x, std::size_t y,
                     double & ux_out, double & uy_out) const
{
  // The 9 populations for cell k are at d_f_[i*N_ + k] for i=0..8.
  // In SoA layout these are stride-N_ apart.  cudaMemcpy2D copies them
  // into a contiguous host buffer without fetching the whole f array.
  const std::size_t k = idx(x, y);
  double f_local[Q];
  CUDA_CHECK(cudaMemcpy2D(
      f_local,              sizeof(double),          // dst, dst_pitch
      d_f_ + k,             N_ * sizeof(double),     // src, src_pitch (stride between directions)
      sizeof(double), Q,                             // width (bytes per row), height (rows = Q)
      cudaMemcpyDeviceToHost));

  double rho = 0.0, mx = 0.0, my = 0.0;
  for (int i = 0; i < Q; ++i) {
    rho += f_local[i];
    mx  += h_cx[i] * f_local[i];
    my  += h_cy[i] * f_local[i];
  }
  ux_out = (rho > 0.0) ? mx / rho : 0.0;
  uy_out = (rho > 0.0) ? my / rho : 0.0;
}

// ---------------------------------------------------------------------------
// Macroscopic accessors — operate on h_f_ (populated by sync_to_host()).
// ---------------------------------------------------------------------------

double
LBM::rho(std::size_t x, std::size_t y) const
{
  const std::size_t k = idx(x, y);
  double r = 0.0;
  for (int i = 0; i < Q; ++i) r += h_f_[std::size_t(i) * N_ + k];
  return r;
}

double
LBM::ux(std::size_t x, std::size_t y) const
{
  const std::size_t k = idx(x, y);
  double r = 0.0, m = 0.0;
  for (int i = 0; i < Q; ++i) {
    const double fi = h_f_[std::size_t(i) * N_ + k];
    r += fi;
    m += h_cx[i] * fi;
  }
  return (r > 0.0) ? m / r : 0.0;
}

double
LBM::uy(std::size_t x, std::size_t y) const
{
  const std::size_t k = idx(x, y);
  double r = 0.0, m = 0.0;
  for (int i = 0; i < Q; ++i) {
    const double fi = h_f_[std::size_t(i) * N_ + k];
    r += fi;
    m += h_cy[i] * fi;
  }
  return (r > 0.0) ? m / r : 0.0;
}

double
LBM::vorticity(std::size_t x, std::size_t y) const
{
  if (x == 0 || x == nx_ - 1 || y == 0 || y == ny_ - 1) return 0.0;
  return 0.5 * ((uy(x+1,y) - uy(x-1,y)) - (ux(x,y+1) - ux(x,y-1)));
}

bool
LBM::is_solid(std::size_t x, std::size_t y) const
{
  return h_solid_[idx(x, y)] != 0;
}
