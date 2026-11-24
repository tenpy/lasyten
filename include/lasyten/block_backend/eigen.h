#pragma once

#include <complex>
#include <cstddef>
#include <cstdint>
#include <lasyten/block_backend/block_backend.h>
#include <lasyten/tools/hdf5.h>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

namespace lasyten {

namespace dtype {

/// Compile-time map from a C++ scalar type to the lasyten Dtype.
///
/// ``bool``, ``int64``, ``float32``, ``float64``, ``complex64`` and ``complex128`` are supported.
template<typename T>
constexpr Dtype
of()
{
    if constexpr (std::is_same_v<T, bool>)
        return Dtype::Bool;
    else if constexpr (std::is_same_v<T, int64>)
        return Dtype::Int64;
    else if constexpr (std::is_same_v<T, float32>)
        return Dtype::Float32;
    else if constexpr (std::is_same_v<T, complex64>)
        return Dtype::Complex64;
    else if constexpr (std::is_same_v<T, float64>)
        return Dtype::Float64;
    else if constexpr (std::is_same_v<T, complex128>)
        return Dtype::Complex128;
    else
        static_assert(!sizeof(T), "unsupported scalar type for lasyten::Dtype");
}

/// Size in bytes of one entry of the given dtype.
std::size_t itemsize(Dtype dtype);

} // namespace dtype

/// A native C++ block-backend using Eigen and LAPACK(E) on contiguous CPU buffers.
///
/// Blocks own their (C-ordered, contiguous) memory; no numpy / torch objects are used as
/// storage and all linear algebra is done with Eigen (products, elementwise) and LAPACKE
/// (SVD, QR, eigh, eig).
///
/// Python objects are only touched at the boundary: ``to_numpy`` / ``block_from_numpy``,
/// ``as_block`` / ``as_scalar(py::object, ...)``, HDF5 import / export and
/// ``_block_repr_lines``.
///
/// No constructor available, use from_factory instead.
/// Not to be subclassed.
class EigenBlockBackend : public BlockBackend
{
  public:
    /// Block that owns a contiguous C-ordered CPU buffer.
    class LASYTEN_EXPORT Block : public BlockBackend::Block
    {
      public:
        /// New block filled with zeros.
        Block(Dtype dtype, std::vector<int64> shape);
        /// New block taking ownership of ``data`` (raw bytes, C-order, ``numel * itemsize``).
        Block(Dtype dtype, std::vector<int64> shape, std::vector<std::byte> data);
#if LASYTEN_HAS_PYTHON
        /// Copy a numpy array (any memory layout) into owning storage.
        explicit Block(const py::array& arr);
#endif // LASYTEN_HAS_PYTHON
        virtual ~Block() = default;

        /// Number of entries.
        int64 numel() const;
        /// Raw bytes of the (contiguous, C-ordered) buffer.
        const std::vector<std::byte>& bytes() const { return data_; }
        std::vector<std::byte>& bytes() { return data_; }

        /// Typed access to the buffer. ``T`` must match ``dtype()``.
        template<typename T>
        const T* data() const
        {
            check_type<T>();
            return reinterpret_cast<const T*>(data_.data());
        }
        template<typename T>
        T* data()
        {
            check_type<T>();
            return reinterpret_cast<T*>(data_.data());
        }

        BlockBackend* get_backend() const override;

        std::vector<int64> shape() const override { return shape_; }
        Dtype dtype() const override { return dtype_; }
        const std::string& device() const override;
#if LASYTEN_HAS_PYTHON
        /// Copy the data into a new numpy array.
        py::array to_numpy() const override;
        py::array to_numpy(Dtype dtype) const override;

        BlockPtr get_item(py::object key) override;
        BlockCPtr get_item(py::object key) const override;
#endif // LASYTEN_HAS_PYTHON
       // Python-free equivalents: shape()/dtype()/data<T>()/bytes() above and the
       // ``get_item`` / ``set_item`` overloads taking ``BlockIndex`` below.
        BlockPtr get_item(std::span<const BlockIndex> key) override;
        BlockCPtr get_item(std::span<const BlockIndex> key) const override;
#if LASYTEN_HAS_PYTHON
        void set_item(py::object key, py::object value) override;
#endif // LASYTEN_HAS_PYTHON
        void set_item(std::span<const BlockIndex> key, const BlockBackend::Block& value) override;
        using BlockBackend::Block::set_item;
        void set_item(const std::vector<int64>& key, const Scalar& value) override;
        void set_item(int64 idx, const Scalar& value) override;

        complex128 _item_as_complex128() const override;
        int64 _item_as_int64() const override;

        BlockPtr operator+(const BlockBackend::Block& other) const override;
        BlockPtr operator-(const BlockBackend::Block& other) const override;
        BlockPtr operator*(const BlockBackend::Block& other) const override;
        BlockPtr operator/(const BlockBackend::Block& other) const override;
        BlockPtr operator<(const BlockBackend::Block& other) const override;
        BlockPtr operator<=(const BlockBackend::Block& other) const override;
        BlockPtr operator>(const BlockBackend::Block& other) const override;
        BlockPtr operator>=(const BlockBackend::Block& other) const override;
        BlockPtr operator==(const BlockBackend::Block& other) const override;
        BlockPtr operator!=(const BlockBackend::Block& other) const override;
        BlockPtr pow(const BlockBackend::Scalar& exponent) const override;
        BlockPtr pow(const BlockBackend::Block& exponent) const override;

        void save_hdf5(lasyten::hdf5::Saver& saver,
                       HighFive::Group& h5gr,
                       std::string const& subpath) override;
        static std::shared_ptr<Block> from_hdf5(lasyten::hdf5::Loader& loader,
                                                HighFive::Group& h5gr,
                                                std::string const& subpath);

      protected:
        template<typename T>
        void check_type() const
        {
            if (dtype_ != dtype::of<T>())
                throw std::logic_error("EigenBlockBackend::Block: dtype mismatch in typed access");
        }

        Dtype dtype_;
        std::vector<int64> shape_;
        std::vector<std::byte> data_;
    };

  public:
    Scalar as_scalar(complex128 value, Dtype dtype) override;
#if LASYTEN_HAS_PYTHON
    Scalar as_scalar(py::object value, Dtype dtype) override;
#endif // LASYTEN_HAS_PYTHON
    Scalar as_scalar(bool b) override;
    Scalar as_scalar(int64 x) override;
    Scalar as_scalar(float32 x) override;
    Scalar as_scalar(float64 x) override;
    Scalar as_scalar(complex64 z) override;
    Scalar as_scalar(complex128 z) override;

  public:
    /// Get the backend instance (singleton). ``device`` must be ``"cpu"`` or ``"eigen"``.
    static EigenBlockBackend* from_factory(const std::string& device = "cpu");
    /// Get a shared_ptr to the backend (e.g. for from_hdf5 or Python reference).
    static std::shared_ptr<EigenBlockBackend> from_factory_shared(
      const std::string& device = "cpu");

    /// Seed the random number generator used by `random_normal` and `random_uniform`.
    static void seed(std::uint64_t seed);

  protected:
    explicit EigenBlockBackend();

  public:
    static std::shared_ptr<EigenBlockBackend> from_hdf5(lasyten::hdf5::Loader& loader,
                                                        HighFive::Group& h5gr,
                                                        std::string const& subpath);

    std::string get_backend_name() const override;

    BlockPtr apply_axis_permutations(const BlockCPtr& block,
                                     const std::vector<std::vector<int64>>& perms) override;
#if LASYTEN_HAS_PYTHON
    BlockPtr apply_leg_permutations(const BlockCPtr& block,
                                    const std::vector<py::array_t<int64>>& perms) override;
    BlockPtr as_block(py::object a,
                      std::optional<Dtype> dtype,
                      std::optional<std::string> device) override;
#endif // LASYTEN_HAS_PYTHON
    std::string as_device(std::optional<std::string> device) override;
    std::vector<int64> abs_argmax(const BlockCPtr& block) override;
    std::vector<int64> argmin(const BlockCPtr& block) override;
    BlockPtr abs(const BlockCPtr& a) override;
    BlockPtr add_axis(const BlockCPtr& a, int64 pos) override;
    bool all(const BlockCPtr& a) override;
    bool allclose(const BlockCPtr& a, const BlockCPtr& b, float64 rtol, float64 atol) override;
    BlockPtr angle(const BlockCPtr& a) override;
    bool any(const BlockCPtr& a) override;
    BlockPtr apply_mask(const BlockCPtr& block, const BlockCPtr& mask, int64 ax) override;
    BlockPtr _argsort(const BlockCPtr& block, int64 axis) override;
    BlockPtr conj(const BlockCPtr& a) override;
    BlockPtr copy_block(const BlockCPtr& a, std::optional<std::string> device) override;
    BlockPtr cutoff_inverse(const BlockCPtr& a, float64 cutoff) override;
    std::tuple<BlockPtr, BlockPtr> eigh(const BlockCPtr& block,
                                        std::optional<std::string> sort) override;
    std::tuple<BlockPtr, BlockPtr> eig(const BlockCPtr& block,
                                       std::optional<std::string> sort) override;
    BlockPtr eigvalsh(const BlockCPtr& block, std::optional<std::string> sort) override;
    BlockPtr eigvals(const BlockCPtr& block, std::optional<std::string> sort) override;
    BlockPtr enlarge_leg(const BlockCPtr& block, const BlockCPtr& mask, int64 axis) override;
    BlockPtr exp(const BlockCPtr& a) override;
    BlockPtr block_from_diagonal(const BlockCPtr& diag) override;
    BlockPtr block_from_mask(const BlockCPtr& mask, Dtype dtype) override;
#if LASYTEN_HAS_PYTHON
    BlockPtr block_from_numpy(const py::array& a,
                              std::optional<Dtype> dtype,
                              std::optional<std::string> device) override;
#endif // LASYTEN_HAS_PYTHON
    BlockPtr get_diagonal(const BlockCPtr& a, std::optional<float64> tol) override;
    BlockPtr imag(const BlockCPtr& a) override;
    Scalar inner(const BlockCPtr& a, const BlockCPtr& b, bool do_dagger) override;
    Scalar item(const BlockCPtr& a) override;
    BlockPtr kron(const BlockCPtr& a, const BlockCPtr& b) override;
    BlockPtr linear_combination(const Scalar& a_coef,
                                const BlockCPtr& v,
                                const Scalar& b_coef,
                                const BlockCPtr& w) override;
    BlockPtr log(const BlockCPtr& a) override;
    Scalar max(const BlockCPtr& a) override;
    Scalar max_abs(const BlockCPtr& a) override;
    Scalar min(const BlockCPtr& a) override;
    BlockPtr mul(const Scalar& a, const BlockCPtr& b) override;
    Scalar norm(const BlockCPtr& a, float64 order, std::optional<int64> axis) override;
    BlockPtr outer(const BlockCPtr& a, const BlockCPtr& b) override;
    BlockPtr permute_axes(const BlockCPtr& a, const std::vector<int64>& permutation) override;
    BlockPtr random_normal(const std::vector<int64>& dims,
                           Dtype dtype,
                           float64 sigma,
                           std::optional<std::string> device) override;
    BlockPtr random_uniform(const std::vector<int64>& dims,
                            Dtype dtype,
                            std::optional<std::string> device) override;
    BlockPtr real(const BlockCPtr& a) override;
    BlockPtr real_if_close(const BlockCPtr& a, float64 tol) override;
    BlockPtr scale_axis(const BlockCPtr& block, const BlockCPtr& factors, int64 axis) override;
    BlockPtr tile(const BlockCPtr& a, int64 repeats) override;
    std::vector<std::string> _block_repr_lines(const BlockCPtr& a,
                                               const std::string& indent,
                                               int64 max_width,
                                               int64 max_lines) override;
    BlockPtr reshape(const BlockCPtr& a, const std::vector<int64>& shape) override;
    BlockPtr sqrt(const BlockCPtr& a) override;
    BlockPtr squeeze_axes(const BlockCPtr& a, const std::vector<int64>& idcs) override;
    BlockPtr stable_log(const BlockCPtr& block, float64 cutoff) override;
    BlockPtr sum(const BlockCPtr& a, int64 ax) override;
    Scalar sum_all(const BlockCPtr& a) override;
    BlockPtr multiply_blocks(const BlockCPtr& a, const BlockCPtr& b) override;
    BlockPtr minimum_blocks(const BlockCPtr& a, const BlockCPtr& b) override;
    BlockPtr maximum_blocks(const BlockCPtr& a, const BlockCPtr& b) override;
    BlockPtr tdot(const BlockCPtr& a,
                  const BlockCPtr& b,
                  const std::vector<int64>& idcs_a,
                  const std::vector<int64>& idcs_b) override;
    BlockPtr to_dtype(const BlockCPtr& a, Dtype dtype) override;
    Scalar trace_full(const BlockCPtr& a) override;
    BlockPtr trace_partial(const BlockCPtr& a,
                           const std::vector<int64>& idcs1,
                           const std::vector<int64>& idcs2,
                           const std::vector<int64>& remaining_idcs) override;
    BlockPtr eye_matrix(int64 dim, Dtype dtype, std::optional<std::string> device) override;
    Scalar get_block_element(const BlockCPtr& a, const std::vector<int64>& idcs) override;
    BlockPtr matrix_dot(const BlockCPtr& a, const BlockCPtr& b) override;
    BlockPtr matrix_exp(const BlockCPtr& matrix) override;
    std::tuple<BlockPtr, BlockPtr> matrix_qr(const BlockCPtr& a, bool full) override;
    /// SVD via LAPACKE: ``"gesdd"`` (default, divide and conquer), ``"gesvd"``,
    /// ``"robust"`` (gesdd, then gesvd on failure) or ``"robust_silent"``.
    std::tuple<BlockPtr, BlockPtr, BlockPtr> matrix_svd(
      const BlockCPtr& a,
      std::optional<std::string> algorithm) override;
    const std::vector<std::string>& possible_svd_algorithms() const override;
    BlockPtr ones_block(const std::vector<int64>& shape,
                        Dtype dtype,
                        std::optional<std::string> device) override;
    BlockPtr zeros(const std::vector<int64>& shape,
                   Dtype dtype,
                   std::optional<std::string> device) override;

  protected:
    bool is_correct_block_type(const BlockCPtr& block) const override;
};

} // namespace lasyten
