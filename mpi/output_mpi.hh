#ifndef LBM_OUTPUT_HH
#define LBM_OUTPUT_HH

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

class LBM;

/**
 * @brief Writes LBM snapshots to HDF5 + a single XDMF index file.
 *
 * Two write_snapshot overloads are provided:
 *   - Vector-based: used by the MPI driver (rank 0 passes pre-gathered arrays).
 *   - LBM-reference: kept for the serial driver.
 */
class XDMFWriter
{
public:
  XDMFWriter(const std::string & prefix, std::size_t nx, std::size_t ny);

  /// Write static solid mask from a pre-gathered byte array (row-major, ny*nx).
  void write_mask(const std::vector<std::uint8_t> & mask);

  /// Write static solid mask by querying the solver directly (serial path).
  void write_mask(const LBM & solver);

  /// Append a snapshot from pre-gathered field arrays (row-major, ny*nx each).
  void write_snapshot(const std::vector<double> & rho,
                      const std::vector<double> & ux,
                      const std::vector<double> & uy,
                      const std::vector<double> & vor,
                      double t);

  /// Append a snapshot by querying the solver directly (serial path).
  void write_snapshot(const LBM & solver, double t);

private:
  void write_root_xdmf() const;

  std::string  prefix_;
  std::string  basename_;
  std::size_t  nx_, ny_;
  bool         has_mask_;
  std::vector<double> times_;
};

#endif  // LBM_OUTPUT_HH
