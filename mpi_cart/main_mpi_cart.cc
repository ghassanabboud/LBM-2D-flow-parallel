#include "lbm_mpi_cart.hh"
#include "output_mpi_cart.hh"

#include <mpi.h>

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
      MPI_Abort(MPI_COMM_WORLD, 1);
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
  MPI_Init(&argc, &argv);

  int rank, nprocs;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

  std::cout << "Hello from rank " << rank << " / " << nprocs << "\n" << std::flush;
  MPI_Barrier(MPI_COMM_WORLD);

  const Args kv = parse_args(argc, argv);

  const std::size_t nx    = get<std::size_t>(kv, "nx",    800);
  const std::size_t ny    = get<std::size_t>(kv, "ny",    400);
  const double      Re    = get<double>     (kv, "re",    100.0);
  const double      u_in  = get<double>     (kv, "u_in",  0.05);
  const std::size_t steps = get<std::size_t>(kv, "steps", 60000);

  const double cx0 = get<double>(kv, "cyl_x", double(nx) * 0.25);
  const double cy0 = get<double>(kv, "cyl_y", double(ny) * 0.50);
  const double cr0 = get<double>(kv, "cyl_r", double(ny) * 0.025);

  const double cx1 = get<double>(kv, "cyl2_x", -1.0);
  const double cy1 = get<double>(kv, "cyl2_y", -1.0);
  const double cr1 = get<double>(kv, "cyl2_r", -1.0);

  const std::size_t every     = get<std::size_t>(kv, "every", 500);
  const std::string out_pref  = get_string(kv, "out", "out/lbm");
  const std::string probe_csv = get_string(kv, "probe", "probe.csv");

  const std::size_t px = get<std::size_t>(kv, "probe_x",
                                          std::size_t(cx0 + 8.0 * cr0));
  const std::size_t py = get<std::size_t>(kv, "probe_y", std::size_t(cy0));

  LBM solver(nx, ny, u_in, Re, cx0, cy0, cr0, MPI_COMM_WORLD);
  if (cr1 > 0.0) solver.add_second_cylinder(cx1, cy1, cr1);
  solver.initialize();

  const bool debug_prints = std::getenv("DEBUG_PRINTS") != nullptr;
  if (debug_prints) {
    if (rank == 0) std::cout << "[MPI] === after initialize ===\n" << std::flush;
    MPI_Barrier(MPI_COMM_WORLD);
    const std::size_t x_mid = nx / 2;
    for (int r = 0; r < nprocs; ++r) {
      if (rank == r) {
        solver.print_solid_boundary_rows();
        solver.print_solid_boundary_rows_ghost();
        for (std::size_t gy : {std::size_t(0), std::size_t(1), ny - 2, ny - 1})
          solver.print_point_populations(x_mid, gy);
      }
      MPI_Barrier(MPI_COMM_WORLD);
    }
  }

  // Determine if this rank owns the probe point (px, py) and its local coords.
  // With 2D decomposition each rank checks its own tile; no single probe_rank needed.
  const std::size_t local_ny   = solver.local_ny();
  const std::size_t local_nx   = solver.local_nx();
  const bool owns_probe = (px >= solver.x_start() && px < solver.x_start() + local_nx &&
                           py >= solver.y_start() && py < solver.y_start() + local_ny);
  const std::size_t probe_lx = owns_probe ? px - solver.x_start() + 1 : 0;
  const std::size_t probe_ly = owns_probe ? py - solver.y_start() + 1 : 0;

  if (rank == 0) {
    std::cout << "LBM 2D D2Q9 BGK  (MPI Cart, " << nprocs << " rank"
              << (nprocs > 1 ? "s" : "") << ")\n"
              << "  grid          : " << nx << " x " << ny << "\n"
              << "  local tile    : " << local_nx << " x " << local_ny << "\n"
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
  }

  std::ofstream probe;
  if (rank == 0) {
    probe.open(probe_csv);
    probe << "step,ux,uy\n";
  }

  XDMFWriter * writer = nullptr;
  if (rank == 0) writer = new XDMFWriter(out_pref, nx, ny);

  if (every > 0) {
    std::vector<std::uint8_t> full_solid;
    solver.gather_solid(full_solid);
    if (rank == 0) writer->write_mask(full_solid);
  }

  using clk = std::chrono::high_resolution_clock;
  const auto t0 = clk::now();

  for (std::size_t step = 1; step <= steps; ++step) {
    if (debug_prints && rank == 0)
      std::cout << "[MPI] === before step " << step
                << " (post-stream pre-collision) ===\n" << std::flush;
    // step() serializes print_post_stream_pre_collision rank-by-rank between
    // exchange_halos and collide_stream when DEBUG_PRINTS is set.
    solver.step();

    if (debug_prints) {
      if (rank == 0) std::cout << "[MPI] === after step " << step << " ===\n" << std::flush;
      MPI_Barrier(MPI_COMM_WORLD);
      const std::size_t x_mid = nx / 2;
      for (int r = 0; r < nprocs; ++r) {
        if (rank == r)
          for (std::size_t gy : {std::size_t(0), std::size_t(1), ny - 2, ny - 1})
            solver.print_point_populations(x_mid, gy);
        MPI_Barrier(MPI_COMM_WORLD);
      }
    }

    // Probe: owning rank (the one whose tile contains (px, py)) contributes;
    // MPI_Reduce sums across all ranks so rank 0 receives the value.
    double loc_ux = 0.0, loc_uy = 0.0;
    if (owns_probe) {
      loc_ux = solver.ux(probe_lx, probe_ly);
      loc_uy = solver.uy(probe_lx, probe_ly);
    }
    double probe_ux = 0.0, probe_uy = 0.0;
    MPI_Reduce(&loc_ux, &probe_ux, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(&loc_uy, &probe_uy, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    if (rank == 0) probe << step << ',' << probe_ux << ',' << probe_uy << '\n';

    if (every > 0 && step % every == 0) {
      std::vector<double> rho_v, ux_v, uy_v, vor_v;
      solver.gather_fields(rho_v, ux_v, uy_v, vor_v);
      if (rank == 0) {
        writer->write_snapshot(rho_v, ux_v, uy_v, vor_v, double(step));
        std::cout << "\r  step " << step << " / " << steps << std::flush;
      }
    }
  }
  if (every > 0 && rank == 0) std::cout << "\n";

  const double dt_local = std::chrono::duration<double>(clk::now() - t0).count();
  double dt_max = 0.0;
  MPI_Reduce(&dt_local, &dt_max, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

  if (rank == 0) {
    const double mlups = double(nx) * double(ny) * double(steps) / dt_max / 1.0e6;
    std::cout << "Wall time : " << dt_max << " s\n"
              << "MLUPS     : " << mlups << "\n";
  }

  delete writer;
  MPI_Finalize();
  return 0;
}
