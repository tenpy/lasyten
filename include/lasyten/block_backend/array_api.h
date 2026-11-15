#pragma once

#include <lasyten/block_backend/block_backend.h>
#include <lasyten/detail/python_api.h>

#if !LASYTEN_HAS_PYTHON
#error                                                                                            \
  "ArrayApiBlockBackend needs a Python interpreter: configure with -DLASYTEN_WITH_PYTHON=ON and -DLASYTEN_WITH_NUMPY_BACKEND=ON"
#endif
#if defined(LASYTEN_WITH_NUMPY_BACKEND) && !LASYTEN_WITH_NUMPY_BACKEND
#error                                                                                            \
  "ArrayApiBlockBackend is compiled out (-DLASYTEN_WITH_NUMPY_BACKEND=OFF); use EigenBlockBackend"
#endif
#include <lasyten/tools/hdf5.h>
#include <map>
#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace lasyten {

/// A block-backend based on a generic Array API compliant library.
///
/// Construct with an Array-API namespace (e.g. ``numpy``) and optional default device.
/// Intended to be subclassed from Python to fill in missing operations.
class ArrayApiBlockBackend : public BlockBackend
{
  public:
    /// Block that holds an Array-API array as ``py::object``.
    class LASYTEN_EXPORT Block : public BlockBackend::Block
    {
      public:
        explicit Block(py::object arr, ArrayApiBlockBackend* backend);
        virtual ~Block() = default;

        BlockBackend* get_backend() const override;

        std::vector<int64> shape() const override;
        Dtype dtype() const override;
        const std::string& device() const override;
        py::array to_numpy() const override;
        py::array to_numpy(Dtype dtype) const override;

        /// Access the underlying Array-API array object.
        py::object& obj() { return arr_; }
        const py::object& obj() const { return arr_; }

        BlockPtr get_item(py::object key) override;
        BlockCPtr get_item(py::object key) const override;
        BlockPtr get_item(std::span<const BlockIndex> key) override;
        BlockCPtr get_item(std::span<const BlockIndex> key) const override;
        void set_item(py::object key, py::object value) override;
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
        py::object arr_;
        ArrayApiBlockBackend* backend_; // non-owning; backend outlives blocks in practice
        std::string device_;
    };

  private:
    Scalar as_scalar(py::object value);

  public:
    Scalar as_scalar(complex128 value, Dtype dtype) override;
    Scalar as_scalar(py::object value, Dtype dtype) override;
    Scalar as_scalar(bool b) override;
    Scalar as_scalar(int64 x) override;
    Scalar as_scalar(float32 x) override;
    Scalar as_scalar(float64 x) override;
    Scalar as_scalar(complex64 z) override;
    Scalar as_scalar(complex128 z) override;
    using BlockBackend::as_scalar;

  public:
    /// Construct from an Array API namespace module/object and default device.
    /// Public so Python subclasses can call ``super().__init__(api, device)``.
    explicit ArrayApiBlockBackend(py::object api_namespace,
                                  const std::string& default_device = "cpu");

    static std::shared_ptr<ArrayApiBlockBackend> from_hdf5(lasyten::hdf5::Loader& loader,
                                                           HighFive::Group& h5gr,
                                                           std::string const& subpath);

    /// The Array API namespace this backend dispatches to.
    py::object api() const { return api_; }

    std::string get_backend_name() const override;

    bool operator==(BlockBackend const& other) const override;

    BlockPtr apply_leg_permutations(const BlockCPtr& block,
                                    const std::vector<py::array_t<int64>>& perms) override;
    BlockPtr as_block(py::object a,
                      std::optional<Dtype> dtype,
                      std::optional<std::string> device) override;
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
    BlockPtr block_from_numpy(const py::array& a,
                              std::optional<Dtype> dtype,
                              std::optional<std::string> device) override;
    BlockPtr get_diagonal(const BlockCPtr& a, std::optional<float64> tol) override;
    BlockPtr imag(const BlockCPtr& a) override;
    Scalar item(const BlockCPtr& a) override;
    BlockPtr kron(const BlockCPtr& a, const BlockCPtr& b) override;
    BlockPtr log(const BlockCPtr& a) override;
    Scalar max(const BlockCPtr& a) override;
    Scalar max_abs(const BlockCPtr& a) override;
    Scalar min(const BlockCPtr& a) override;
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

    /// Map Array-API dtype object → lasyten Dtype.
    Dtype dtype_from_api(py::object api_dtype) const;
    /// Map lasyten Dtype → Array-API dtype object.
    py::object dtype_to_api(Dtype dtype) const;

  private:
    static const ArrayApiBlockBackend::Block* ptr(const BlockCPtr& b);
    static py::object obj(const BlockCPtr& b);
    BlockPtr wrap(py::object arr);

    py::object api_;
    std::map<Dtype, py::object> backend_dtype_map_;
    // keyed by id(api_dtype) as py::handle for lookup; also store reverse map via dtype_from_api
    std::map<std::uintptr_t, Dtype> lasyten_dtype_map_;
};

} // namespace lasyten

// pybind11 casters for nested bound types, delegating to nanobind (py_nanobind_bridge.h).
LASYTEN_PY_BRIDGE_CLASS(::lasyten::ArrayApiBlockBackend::Block)
