#pragma once

#include <lasyten/block_backend/block_backend.h>
#include <lasyten/detail/python_api.h>

#if !LASYTEN_HAS_PYTHON
#error "TorchBlockBackend is only built with LASYTEN_WITH_PYTHON=ON (torch.cpp still uses py::)"
#endif
#include <lasyten/tools/hdf5.h>
#include <memory>
#include <optional>
#include <string>
#include <torch/torch.h>
#include <vector>

namespace lasyten {

namespace dtype {

/// Map lasyten Dtype to torch ScalarType.
c10::ScalarType to_torch_dtype(Dtype dtype);
/// Map torch ScalarType to lasyten Dtype.
Dtype from_torch_dtype(c10::ScalarType torch_dtype);

} // namespace dtype

/// A block-backend using PyTorch.
///
/// No constructor available, use from_factory instead.
/// Not to be subclassed.
class TorchBlockBackend : public BlockBackend
{
  public:
    /// Block that holds a torch::Tensor.
    class LASYTEN_EXPORT Block : public BlockBackend::Block
    {
      public:
        explicit Block(torch::Tensor tensor);
        virtual ~Block() = default;

        BlockBackend* get_backend() const override;

        std::vector<int64> shape() const override;
        Dtype dtype() const override;
        const std::string& device() const override;
        py::array to_numpy() const override;
        py::array to_numpy(Dtype dtype) const override;

        /// Access the underlying torch tensor (mutable view / shared storage).
        torch::Tensor& tensor() { return tensor_; }
        const torch::Tensor& tensor() const { return tensor_; }

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
        torch::Tensor tensor_;
        std::string device_;
    };

  private:
    Scalar as_scalar(const torch::Tensor& value);

  public:
    Scalar as_scalar(complex128 value, Dtype dtype) override;
    Scalar as_scalar(py::object value, Dtype dtype) override;
    Scalar as_scalar(bool b) override;
    Scalar as_scalar(int64 x) override;
    Scalar as_scalar(float32 x) override;
    Scalar as_scalar(float64 x) override;
    Scalar as_scalar(complex64 z) override;
    Scalar as_scalar(complex128 z) override;

  public:
    /// Get the backend instance for the given device (nearly-singleton per device).
    static TorchBlockBackend* from_factory(const std::string& device = "cpu:0");
    /// Get a shared_ptr to the backend (e.g. for from_hdf5 or Python reference).
    static std::shared_ptr<TorchBlockBackend> from_factory_shared(
      const std::string& device = "cpu:0");

    /// Map an HDF5-saved device string to one that is usable on this machine.
    ///
    /// If ``device`` is available, return its canonical form (e.g. ``cuda:0``).
    /// Otherwise fall back to ``fallback`` (default: factory default, typically ``cpu:0``)
    /// and emit a UserWarning.
    static std::string device_for_hdf5_load(std::string const& device,
                                            std::optional<std::string> fallback = std::nullopt);

  protected:
    explicit TorchBlockBackend(const std::string& default_device);

  public:
    static std::shared_ptr<TorchBlockBackend> from_hdf5(lasyten::hdf5::Loader& loader,
                                                        HighFive::Group& h5gr,
                                                        std::string const& subpath);

    std::string get_backend_name() const override;

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

  private:
    static const TorchBlockBackend::Block* ptr(const BlockCPtr& b);
    static const torch::Tensor& tens(const BlockCPtr& b);
    static BlockPtr wrap(torch::Tensor tensor);

    /// Promote a and b to a common dtype (optionally at least ``at_least``).
    std::pair<torch::Tensor, torch::Tensor> to_same_dtype(
      const torch::Tensor& a,
      const torch::Tensor& b,
      std::optional<c10::ScalarType> at_least = std::nullopt) const;

    torch::Device parse_device(const std::string& device) const;
};

} // namespace lasyten
