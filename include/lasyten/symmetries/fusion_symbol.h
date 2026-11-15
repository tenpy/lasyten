#pragma once

#include "../block_backend/block_backend.h"
#include "../block_backend/dtypes.h"
#include "../lasyten.h"

#include <array>
#include <complex>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <utility>
#include <variant>
#include <vector>

namespace lasyten {

/// Owning dense array for topological symmetry data (F/R/C/B symbols, fusion tensors, …).
///
/// Rank is at most 4; dtype is Float64 or Complex128.
class FusionSymbol
{
  public:
    using Shape = std::array<std::size_t, 4>;
    using Data = std::variant<std::vector<float64>, std::vector<complex128>>;

    FusionSymbol() = default;

    /// Construct with given rank (1..4), shape (unused axes must be 1), and dtype.
    FusionSymbol(std::uint8_t rank, Shape shape, Dtype dtype);

    [[nodiscard]] Dtype dtype() const noexcept { return dtype_; }
    [[nodiscard]] std::uint8_t rank() const noexcept { return rank_; }
    [[nodiscard]] Shape const& shape() const noexcept { return shape_; }
    [[nodiscard]] std::size_t extent(std::uint8_t axis) const;
    [[nodiscard]] std::size_t size() const noexcept;
    [[nodiscard]] bool empty() const noexcept { return size() == 0; }
    [[nodiscard]] bool is_real() const noexcept { return dtype::is_real(dtype_); }

    [[nodiscard]] std::vector<int64> shape_as_int64() const;

    /// Flat C-order index from up to 4 coordinates (unused axes ignored / must be 0).
    [[nodiscard]] std::size_t offset(std::size_t i0,
                                     std::size_t i1 = 0,
                                     std::size_t i2 = 0,
                                     std::size_t i3 = 0) const;

    [[nodiscard]] complex128 get_complex(std::size_t i0,
                                         std::size_t i1 = 0,
                                         std::size_t i2 = 0,
                                         std::size_t i3 = 0) const;
    void set(std::size_t i0, complex128 value);
    void set(std::size_t i0, std::size_t i1, complex128 value);
    void set(std::size_t i0, std::size_t i1, std::size_t i2, complex128 value);
    void set(std::size_t i0, std::size_t i1, std::size_t i2, std::size_t i3, complex128 value);

    /// View / mutate storage (throws if dtype mismatch).
    [[nodiscard]] std::span<float64> as_float64();
    [[nodiscard]] std::span<float64 const> as_float64() const;
    [[nodiscard]] std::span<complex128> as_complex128();
    [[nodiscard]] std::span<complex128 const> as_complex128() const;

    /// Reshape in place (same number of elements); new_rank in 1..4.
    FusionSymbol& reshape(std::uint8_t new_rank, Shape new_shape);

    [[nodiscard]] FusionSymbol reshaped(std::uint8_t new_rank, Shape new_shape) const;

    /// Elementwise complex conjugate (real arrays returned unchanged / copied).
    [[nodiscard]] FusionSymbol conj() const;

    /// Promote Float64 → Complex128 if needed; Complex128 unchanged.
    [[nodiscard]] FusionSymbol as_complex() const;

    /// Cast to target dtype (Float64 or Complex128).
    [[nodiscard]] FusionSymbol as_dtype(Dtype target) const;

    [[nodiscard]] complex128 sum() const;

    /// Scale all entries by a scalar (promotes to complex if scalar is complex and array is real).
    [[nodiscard]] FusionSymbol operator*(complex128 scale) const;
    [[nodiscard]] FusionSymbol operator*(float64 scale) const;
    friend FusionSymbol operator*(complex128 scale, FusionSymbol const& a) { return a * scale; }
    friend FusionSymbol operator*(float64 scale, FusionSymbol const& a) { return a * scale; }

    /// Elementwise multiply with broadcasting (NumPy-style, ranks must match after pad).
    [[nodiscard]] FusionSymbol multiply(FusionSymbol const& other) const;

    /// Slice first two axes: result rank-2 array ``self[i0, i1, :, :]`` (requires rank == 4).
    [[nodiscard]] FusionSymbol slice2d(std::size_t i0, std::size_t i1) const;

    /// Slice last two axes fixed: ``self[:, :, i2, i3]`` as rank-2 (requires rank == 4).
    [[nodiscard]] FusionSymbol slice2d_trailing(std::size_t i2, std::size_t i3) const;

    /// Transpose: permutation of length ``rank`` (0-based axis indices).
    [[nodiscard]] FusionSymbol transpose(std::array<std::uint8_t, 4> const& axes) const;

    /// Copy of the leading 2D block ``self[0,0,:,:]`` for rank-4, or full array if rank-2.
    [[nodiscard]] FusionSymbol take_leading_matrix() const;

    void fill(complex128 value);

    /// Iterate all entries of a rank-2 array as complex.
    void for_each2d(std::function<void(std::size_t, std::size_t, complex128)> const& fn) const;

    // --- factories ---

    [[nodiscard]] static FusionSymbol zeros(std::uint8_t rank, Shape shape, Dtype dtype);
    [[nodiscard]] static FusionSymbol ones(std::uint8_t rank, Shape shape, Dtype dtype);
    [[nodiscard]] static FusionSymbol full(std::uint8_t rank,
                                           Shape shape,
                                           complex128 value,
                                           Dtype dtype);
    [[nodiscard]] static FusionSymbol from_float64(std::uint8_t rank,
                                                   Shape shape,
                                                   std::vector<float64> data);
    [[nodiscard]] static FusionSymbol from_complex128(std::uint8_t rank,
                                                      Shape shape,
                                                      std::vector<complex128> data);

    /// Rank-1 length-1 array containing ``value``.
    [[nodiscard]] static FusionSymbol scalar1d(complex128 value, Dtype dtype = Dtype::Complex128);
    [[nodiscard]] static FusionSymbol scalar1d(float64 value, Dtype dtype = Dtype::Float64);

    // Common topo ones (always Float64).
    [[nodiscard]] static FusionSymbol one_1D();
    [[nodiscard]] static FusionSymbol one_2D();
    [[nodiscard]] static FusionSymbol one_4D();

    [[nodiscard]] static std::size_t product(Shape const& shape);

  private:
    Dtype dtype_ = Dtype::Float64;
    std::uint8_t rank_ = 1;
    Shape shape_{ { 1, 1, 1, 1 } };
    Data data_ = std::vector<float64>{ 0.0 };

    void validate() const;
    void ensure_size();
    [[nodiscard]] static Dtype check_symbol_dtype(Dtype dtype);
};

/// Kronecker product of two FusionSymbols of equal rank (result same rank).
[[nodiscard]] FusionSymbol kron(FusionSymbol const& a, FusionSymbol const& b);

#if LASYTEN_HAS_PYTHON
/// NumPy bridge (bindings / interim Block conversion).
[[nodiscard]] py::array fusion_symbol_to_numpy(FusionSymbol const& src);
[[nodiscard]] FusionSymbol fusion_symbol_from_numpy(py::array arr);

#endif // LASYTEN_HAS_PYTHON

/// Convert via the backend: native for `EigenBlockBackend`; other backends go through
/// ``block_from_numpy`` / ``to_numpy`` (needs ``LASYTEN_WITH_PYTHON``).
[[nodiscard]] BlockBackend::BlockPtr block_from_fusion_symbol(
  BlockBackend& backend,
  FusionSymbol const& arr,
  std::optional<Dtype> dtype = std::nullopt,
  std::optional<std::string> device = std::nullopt);

[[nodiscard]] FusionSymbol fusion_symbol_from_block(BlockBackend::BlockCPtr const& block);

} // namespace lasyten
