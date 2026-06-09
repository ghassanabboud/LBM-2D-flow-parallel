#include "lbm_cuda_SoA.cuh"
#include "output_cuda_SoA.hh"

#include <chrono>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <unordered_map>

namespace {

using Args = std::unordered_map<std::string, std::string>;

Args
parse_args(int argc, char ** argv)
{
  Args kv;
  for (int i = 1; i < argc; ++i) {
    const std::string a = argv[i];
    const auto eq = a.find('=');
    if (eq == std::string::npos) {
      std::cerr << "Bad argument '" << a << "' (expected key=value)\n";
      std::exit(1);
    }
    kv[a.substr(0, eq)] = a.substr(eq + 1);
  }
  return kv;
}

template <typename T>
T
get(const Args & kv, const std::string & key, T def)
{
  auto it = kv.find(key);
  if (it == kv.end()) return def;
  std::istringstream iss(it->second);
  T v; iss >> v;
  return v;
}

std::string
get_string(const Args & kv, const std::string & key, const std::string & def)
{
  auto it = kv.find(key);
  return (it == kv.end()) ? def : it->second;
}

}  // namespace

int
main(int argc, char ** argv)
{
  const Args kv = parse_args(argc, argv);

  // Grid + physics.
  const std::size_t nx    = get<std::size_t>(kv, "nx",    800);
  const std::size_t ny    = get<std::size_t>(kv, "ny",    400);
  const double      Re    = get<double>     (kv, "re",    100.0);
  const double      u_in  = get<double>     (kv, "u_in",  0.05);
  const std::size_t steps = get<std::size_t>(kv, "steps", 60000);

  // Cylinder geometry. At Re = 100 with diameter = ny/20 and u_in = 0.05
  // this reproduces the classical Strouhal number St ~ 0.16.
  const double cx0 = get<double>(kv, "cyl_x", double(nx) * 0.25);
  const double cy0 = get<double>(kv, "cyl_y", double(ny) * 0.50);
  const double cr0 = get<double>(kv, "cyl_r", double(ny) * 0.025);

  // Optional second cylinder. Disabled if cyl2_r <= 0.
  const double cx1 = get<double>(kv, "cyl2_x", -1.0);
  const double cy1 = get<double>(kv, "cyl2_y", -1.0);
  const double cr1 = get<double>(kv, "cyl2_r", -1.0);

  // CUDA launch configuration.
  // block_x / block_y: thread-block dimensions.
  //   1D mode: block_y=1 (default), grid_y=1 → each thread maps to one cell
  //            via k = blockIdx.x * blockDim.x + threadIdx.x.
  //   2D mode: block_y > 1 → thread (tx,ty) maps to (x,y); coalescing is
  //            preserved only when block_x is a multiple of 32.
  // grid_x / grid_y: grid dimensions. Defaults to the minimum required to
  //   cover the full domain; may be overridden to experiment with padding.
  const int block_x = get<int>(kv, "block_x", 256);
  const int block_y = get<int>(kv, "block_y", 1);

  const int default_grid_x = (int(nx) + block_x - 1) / block_x; // ceiling division
  const int default_grid_y = (int(ny) + block_y - 1) / block_y;
  const int grid_x  = get<int>(kv, "grid_x",  default_grid_x);
  const int grid_y  = get<int>(kv, "grid_y",  default_grid_y);

  if (grid_x < default_grid_x || grid_y < default_grid_y) {
    std::cerr << "Error: grid (" << grid_x << "," << grid_y
              << ") is too small to cover domain (" << nx << "," << ny
              << ") with block (" << block_x << "," << block_y << ").\n";
    return 1;
  }

  // Output.
  const std::size_t every     = get<std::size_t>(kv, "every", 500);
  const std::string out_pref  = get_string(kv, "out", "out/lbm");
  const std::string probe_csv = get_string(kv, "probe", "probe.csv");

  // Probe location: about 4 diameters downstream, on the cylinder centreline.
  const std::size_t px = get<std::size_t>(kv, "probe_x",
                                          std::size_t(cx0 + 8.0 * cr0));
  const std::size_t py = get<std::size_t>(kv, "probe_y", std::size_t(cy0));

  LBM solver(nx, ny, u_in, Re, cx0, cy0, cr0);
  if (cr1 > 0.0) solver.add_second_cylinder(cx1, cy1, cr1);
  solver.initialize();

  std::cout << "LBM 2D D2Q9 BGK — CUDA SoA\n"
            << "  domain        : " << nx << " x " << ny << "\n"
            << "  block         : " << block_x << " x " << block_y
            << (block_y == 1 ? " (1-D)" : " (2-D)") << "\n"
            << "  grid          : " << grid_x  << " x " << grid_y
            << (grid_y  == 1 ? " (1-D)" : " (2-D)") << "\n"
            << "  Re            : " << Re << "\n"
            << "  u_in          : " << u_in << "\n"
            << "  tau           : " << solver.tau() << "\n"
            << "  cylinder      : (" << cx0 << ", " << cy0
            << "), r = " << cr0 << "\n"
            << "  diameter      : " << 2.0 * cr0 << "\n";
  if (cr1 > 0.0)
    std::cout << "  cylinder #2   : (" << cx1 << ", " << cy1
              << "), r = " << cr1 << "\n";
  std::cout << "  steps         : " << steps << "\n"
            << "  output every  : " << every << " (0 = off)\n"
            << "  output prefix : " << out_pref << "\n"
            << "  probe at      : (" << px << ", " << py << ")\n"
            << "  probe csv     : " << probe_csv << "\n";

  std::ofstream probe(probe_csv);
  probe << "step,ux,uy\n";

  XDMFWriter writer(out_pref, nx, ny);
  if (every > 0) writer.write_mask(solver);

  using clk = std::chrono::high_resolution_clock;
  const auto t0 = clk::now();

  for (std::size_t step = 1; step <= steps; ++step) {

    solver.step(block_x, block_y, grid_x, grid_y);

    // fetch_probe_uxy uses cudaMemcpy2D (9 doubles) and implicitly
    // synchronises the stream, so the GPU is idle between steps when
    // probe output is enabled.  For pure throughput benchmarks, redirect
    // probe_csv to /dev/null and the sync overhead disappears.
    double pux = 0.0, puy = 0.0;
    solver.fetch_probe_uxy(px, py, pux, puy);
    probe << step << ',' << pux << ',' << puy << '\n';

    if (every > 0 && step % every == 0) {
      // Copy full f array to host before computing macroscopic fields.
      solver.sync_to_host();
      writer.write_snapshot(solver, double(step));
      std::cout << "\r  step " << step << " / " << steps << std::flush;
    }
  }
  if (every > 0) std::cout << "\n";

  // The last fetch_probe_uxy already synchronised the device; chrono
  // measurement is accurate without an additional cudaDeviceSynchronize().
  const double dt    = std::chrono::duration<double>(clk::now() - t0).count();
  const double mlups = double(nx) * double(ny) * double(steps) / dt / 1.0e6;
  std::cout << "Wall time : " << dt    << " s\n"
            << "MLUPS     : " << mlups << "\n";
  return 0;
}
