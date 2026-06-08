#ifndef OUTPUT_CUDA_AOA_HH
#define OUTPUT_CUDA_AOA_HH

#include <cstddef>
#include <string>
#include <vector>

class LBM;

/**
 * @brief Writes LBM snapshots to HDF5 + a single XDMF index file
 *        for visualization in ParaView or PyVista.
 *
 * Identical interface to the serial XDMFWriter.  write_snapshot() calls the
 * LBM macroscopic accessors (rho, ux, uy, vorticity), which operate on the
 * host mirror h_f_.  The caller must invoke solver.sync_to_host() before
 * each write_snapshot() call.
 */
class XDMFWriter
{
public:
  XDMFWriter(const std::string & prefix, std::size_t nx, std::size_t ny);

  /// Write the static solid mask once (uses is_solid(), always available).
  void write_mask(const LBM & solver);

  /// Append a snapshot at simulation time t.
  /// Requires solver.sync_to_host() to have been called beforehand.
  void write_snapshot(const LBM & solver, double t);

private:
  void write_root_xdmf() const;

  std::string  prefix_;
  std::string  basename_;
  std::size_t  nx_, ny_;
  bool         has_mask_;
  std::vector<double> times_;
};

#endif  // OUTPUT_CUDA_AOA_HH
