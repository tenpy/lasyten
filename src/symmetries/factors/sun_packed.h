#pragma once

// Internal header (not installed): reader for F / R symbol files in the "packed" layout of the
// SU(N) data generator. Instead of one tiny HDF5 dataset per key (``F_sym/F[2, 0][4, 0]...``) a
// packed file holds four arrays in the group ``F_packed`` (or ``R_packed``):
//
//   labels  (n, k, N) int32   irrep labels of the keys, rows sorted lexicographically
//   shapes  (n, rank) int64   shape of every block (rank 4 for F, 1 for R)
//   offsets (n + 1,)  int64   block i is data[offsets[i] : offsets[i + 1]]
//   data    (offsets[n],) complex128
//
// and the root attribute ``Layout = "packed_v1"``. A file without the attribute uses the folder
// layout.

#include <lasyten/symmetries/fusion_symbol.h>
#include <lasyten/symmetries/sector.h>
#include <lasyten/tools/lru_cache.h>

#include <hdf5_io/h5_ops.h>

#include <highfive/highfive.hpp>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <vector>

namespace lasyten::sun_packed {

/// Whether the symbol file is in the packed layout (``Layout`` attribute ``"packed_v1"``).
/// A file without the attribute uses the folder layout. Any other value is rejected, since a
/// future layout version must not be guessed. `path` is only used for the error message.
[[nodiscard]] bool is_packed_layout(HighFive::File& file, std::string const& path);

/// Random access to the blocks of a packed F or R group: the index (labels, shapes, offsets) is
/// held in RAM in a compact form, the block data either too (small files) or is read per lookup.
class PackedIndex
{
  public:
    /// Read and validate the index of the group ``<kind>_packed`` (kind 'F' or 'R') of `file`.
    /// With `read_data == false` the block data is never loaded (sanity checks only).
    [[nodiscard]] static std::shared_ptr<PackedIndex const> load(
      std::shared_ptr<HighFive::File> file,
      std::string const& path,
      char kind,
      int N,
      bool read_data = true);

    ~PackedIndex();
    PackedIndex(PackedIndex const&) = delete;
    PackedIndex& operator=(PackedIndex const&) = delete;

    /// Index of the block with the given labels (6 sectors for F, 3 for R), if it is stored.
    [[nodiscard]] std::optional<std::size_t> find(std::span<Sector const> sectors) const;

    /// The block `i` as a `FusionSymbol`, identical to the dataset of the folder layout.
    [[nodiscard]] FusionSymbol read(std::size_t i) const;

    /// The checks done when opening a folder file: a key for the all-trivial sectors exists and a
    /// key contains the highest weight twice. Throws ``std::invalid_argument``.
    void check(int64 highest_weight) const;

    [[nodiscard]] std::size_t size() const noexcept { return n_; }
    [[nodiscard]] bool data_in_ram() const noexcept { return data_ptr_ != nullptr; }

  private:
    PackedIndex() = default;

    std::shared_ptr<HighFive::File> file_;
    std::string path_;
    char kind_ = 'F';
    int k_ = 6; // labels per key
    int N_ = 2; // length of a label
    int rank_ = 4;
    std::size_t n_ = 0;
    std::vector<std::int16_t> labels_;  // n * k * N, rows sorted and unique
    std::vector<std::uint32_t> shapes_; // n * rank
    std::vector<std::int64_t> offsets_; // n + 1
    hdf5_io::Hdf5Buffer data_buffer_;   // all block data, if small enough
    complex128 const* data_ptr_ = nullptr;
    hid_t data_dataset_ = -1; // for reading one block at a time
    hid_t data_type_ = -1;
    std::size_t index_bytes_ = 0;
    std::size_t data_bytes_in_ram_ = 0;
    bool counted_ = false; // included in the process-wide statistics
};

} // namespace lasyten::sun_packed
