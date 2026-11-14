#pragma once

/// Native TeNPy-format HDF5 helpers used by lasyten ``save_hdf5`` / ``from_hdf5``.

#include <lasyten/backends/block_inds.h>
#include <lasyten/backends/tensor_backend.h>
#include <lasyten/block_backend/block_backend.h>
#include <lasyten/block_backend/dtypes.h>
#include <lasyten/symmetries/sector.h>
#include <lasyten/symmetries/spaces.h>
#include <lasyten/symmetries/symmetry.h>
#include <lasyten/symmetries/symmetry_factor.h>
#include <lasyten/tensors/labels.h>
#include <lasyten/tools/hdf5.h>

#include <hdf5_io/constants.h>
#include <hdf5_io/h5_ops.h>

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace lasyten::hdf5_export {

inline constexpr char const* kModule = "lasyten._core";

[[nodiscard]] hdf5_io::Hdf5Buffer i64_vector_to_buffer(std::vector<std::int64_t> const& v);
[[nodiscard]] std::vector<std::int64_t> i64_vector_from_buffer(hdf5_io::Hdf5Buffer const& buf);

[[nodiscard]] hdf5_io::Hdf5Buffer i64_matrix_to_buffer(std::vector<std::int64_t> const& flat,
                                                       std::size_t rows,
                                                       std::size_t cols);
[[nodiscard]] std::pair<std::vector<std::int64_t>, std::pair<std::size_t, std::size_t>>
i64_matrix_from_buffer(hdf5_io::Hdf5Buffer const& buf);

[[nodiscard]] hdf5_io::Hdf5Buffer f64_vector_to_buffer(std::vector<double> const& v);
[[nodiscard]] std::vector<double> f64_vector_from_buffer(hdf5_io::Hdf5Buffer const& buf);

void save_i64_vector(lasyten::hdf5::Saver& saver,
                     std::string const& path,
                     std::vector<std::int64_t> const& v);
[[nodiscard]] std::vector<std::int64_t> load_i64_vector(lasyten::hdf5::Loader& loader,
                                                        std::string const& path);

void save_f64_vector(lasyten::hdf5::Saver& saver,
                     std::string const& path,
                     std::vector<double> const& v);
[[nodiscard]] std::vector<double> load_f64_vector(lasyten::hdf5::Loader& loader,
                                                  std::string const& path);

void save_optional_i64_vector(lasyten::hdf5::Saver& saver,
                              std::string const& path,
                              std::optional<std::vector<std::int64_t>> const& v);
[[nodiscard]] std::optional<std::vector<std::int64_t>> load_optional_i64_vector(
  lasyten::hdf5::Loader& loader,
  std::string const& path);

void save_leg_labels(lasyten::hdf5::Saver& saver,
                     std::string const& path,
                     LegLabelList const& labels);
[[nodiscard]] LegLabelList load_leg_labels(lasyten::hdf5::Loader& loader, std::string const& path);

void save_dtype_string(lasyten::hdf5::Saver& saver, std::string const& path, Dtype dt);
[[nodiscard]] Dtype load_dtype_string(lasyten::hdf5::Loader& loader, std::string const& path);

void save_sector(lasyten::hdf5::Saver& saver, std::string const& path, Sector const& sector);
[[nodiscard]] Sector load_sector(lasyten::hdf5::Loader& loader, std::string const& path);

void save_sector_array(lasyten::hdf5::Saver& saver,
                       std::string const& path,
                       SectorArray const& sectors);
[[nodiscard]] SectorArray load_sector_array(lasyten::hdf5::Loader& loader,
                                            std::string const& path);

void save_block_inds(lasyten::hdf5::Saver& saver, std::string const& path, BlockInds const& bi);
[[nodiscard]] BlockInds load_block_inds(lasyten::hdf5::Loader& loader, std::string const& path);

void save_symmetry_factor(lasyten::hdf5::Saver& saver,
                          std::string const& path,
                          SymmetryFactor::CPtr const& factor);
[[nodiscard]] SymmetryFactor::Ptr load_symmetry_factor(lasyten::hdf5::Loader& loader,
                                                       std::string const& path);

void save_symmetry(lasyten::hdf5::Saver& saver,
                   std::string const& path,
                   Symmetry::CPtr const& symmetry);
[[nodiscard]] Symmetry::Ptr load_symmetry(lasyten::hdf5::Loader& loader, std::string const& path);

void save_elementary_space(lasyten::hdf5::Saver& saver,
                           std::string const& path,
                           ElementarySpace::CPtr const& space);
[[nodiscard]] ElementarySpace::Ptr load_elementary_space(lasyten::hdf5::Loader& loader,
                                                         std::string const& path);

void save_leg(lasyten::hdf5::Saver& saver, std::string const& path, Leg::CPtr const& leg);
[[nodiscard]] Leg::Ptr load_leg(lasyten::hdf5::Loader& loader, std::string const& path);

void save_tensor_product(lasyten::hdf5::Saver& saver,
                         std::string const& path,
                         TensorProduct::CPtr const& tp);
[[nodiscard]] TensorProduct::Ptr load_tensor_product(lasyten::hdf5::Loader& loader,
                                                     std::string const& path);

void save_block(lasyten::hdf5::Saver& saver, std::string const& path, BlockCPtr const& block);
[[nodiscard]] BlockPtr load_block(lasyten::hdf5::Loader& loader, std::string const& path);

void save_block_backend(lasyten::hdf5::Saver& saver,
                        std::string const& path,
                        std::shared_ptr<BlockBackend const> const& backend);
[[nodiscard]] std::shared_ptr<BlockBackend> load_block_backend(lasyten::hdf5::Loader& loader,
                                                               std::string const& path);

void save_tensor_backend(lasyten::hdf5::Saver& saver,
                         std::string const& path,
                         TensorBackend::CPtr const& backend);
[[nodiscard]] TensorBackend::Ptr load_tensor_backend(lasyten::hdf5::Loader& loader,
                                                     std::string const& path);

void save_tensor_backend_data(lasyten::hdf5::Saver& saver,
                              std::string const& path,
                              TensorBackend::DataCPtr const& data);
[[nodiscard]] TensorBackend::DataPtr load_tensor_backend_data(lasyten::hdf5::Loader& loader,
                                                              std::string const& path);

/// Open child path; if already memoized as ``T``, return it. Otherwise call ``make(group, sub)``.
template<typename T, typename Make>
std::shared_ptr<T>
load_instance(lasyten::hdf5::Loader& loader, std::string const& path, Make&& make)
{
    hid_t id = loader.open(path);
    if (auto memo = loader.lookup_memo(id)) {
        H5Idec_ref(id);
        return std::static_pointer_cast<T>(memo);
    }
    HighFive::Group group = hdf5_io::group_from_hid(id);
    return make(group, lasyten::hdf5::ensure_slash(path));
}

[[nodiscard]] std::string class_attr(HighFive::Group& h5gr);

/// NumPy ↔ ``Hdf5Buffer`` (used by numpy/torch/array_api block backends).
[[nodiscard]] hdf5_io::Hdf5Buffer buffer_from_numpy(py::array arr);
[[nodiscard]] py::array numpy_from_buffer(hdf5_io::Hdf5Buffer const& buf);

} // namespace lasyten::hdf5_export
