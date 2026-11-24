#pragma once

#include <lasyten/block_backend/block_backend.h>
#include <lasyten/lasyten.h>
#include <lasyten/tensors/forward_declare.h>

#include <optional>

namespace lasyten {

/// The angle of a complex number, applied elementwise on diagonal tensors.
///
/// The counterclockwise angle from the positive real axis on the complex plane in the
/// range @f$(-\pi, \pi]@f$ with a real dtype. The angle of `0.` is `0.`.
///
/// @param x Input diagonal tensor or scalar.
/// @returns The elementwise argument (phase angle).
[[nodiscard]] DiagonalTensorPtr angle(DiagonalTensorCPtr x);
[[nodiscard]] BlockBackend::Scalar angle(BlockBackend::Scalar const& x);

/// Elementwise cutoff inverse.
///
/// The cutoff-inverse for a number `x` is `1 / x` if `abs(x) >= cutoff`, otherwise `0`.
///
/// @param x Input diagonal tensor or scalar.
/// @param cutoff Threshold below which the result is zero (default `1e-15`).
/// @returns The elementwise cutoff inverse.
[[nodiscard]] DiagonalTensorPtr cutoff_inverse(DiagonalTensorCPtr x, float64 cutoff = 1e-15);
[[nodiscard]] BlockBackend::Scalar cutoff_inverse(BlockBackend::Scalar const& x,
                                                  float64 cutoff = 1e-15);

/// Complex conjugation, applied elementwise on diagonal tensors.
///
/// @param x Input diagonal tensor or scalar.
/// @returns The complex conjugate.
[[nodiscard]] DiagonalTensorPtr complex_conj(DiagonalTensorCPtr x);
[[nodiscard]] BlockBackend::Scalar complex_conj(BlockBackend::Scalar const& x);

/// The imaginary part of a complex number, applied elementwise on diagonal tensors.
///
/// @param x Input diagonal tensor or scalar.
/// @returns The imaginary part.
[[nodiscard]] DiagonalTensorPtr imag(DiagonalTensorCPtr x);
[[nodiscard]] BlockBackend::Scalar imag(BlockBackend::Scalar const& x);

/// The real part of a complex number, applied elementwise on diagonal tensors.
///
/// @param x Input diagonal tensor or scalar.
/// @returns The real part.
[[nodiscard]] DiagonalTensorPtr real(DiagonalTensorCPtr x);
[[nodiscard]] BlockBackend::Scalar real(BlockBackend::Scalar const& x);

/// If close to real, return the real part; applied elementwise on diagonal tensors.
///
/// @param x Input complex number(s).
/// @param tol Precision for considering the imaginary part close to zero, in multiples of
///     machine epsilon for the dtype of `x` (default `100`).
/// @returns If `x` is close to real, the real part of `x`; otherwise the original complex `x`.
[[nodiscard]] DiagonalTensorPtr real_if_close(DiagonalTensorCPtr x, float64 tol = 100.);
[[nodiscard]] BlockBackend::Scalar real_if_close(BlockBackend::Scalar const& x,
                                                 float64 tol = 100.);

/// The square root of a number, applied elementwise on diagonal tensors.
///
/// @param x Input diagonal tensor or scalar.
/// @returns The elementwise square root.
[[nodiscard]] DiagonalTensorPtr sqrt(DiagonalTensorCPtr x);
[[nodiscard]] BlockBackend::Scalar sqrt(BlockBackend::Scalar const& x);

/// Stabilized logarithm, applied elementwise on diagonal tensors.
///
/// For values `> cutoff`, this is the standard natural logarithm. For values smaller than the
/// cutoff, return `0`.
///
/// @param x Input diagonal tensor or scalar.
/// @param cutoff Threshold below which the result is zero (default `1e-30`).
/// @returns The elementwise stabilized log.
[[nodiscard]] DiagonalTensorPtr stable_log(DiagonalTensorCPtr x, float64 cutoff = 1e-30);
[[nodiscard]] BlockBackend::Scalar stable_log(BlockBackend::Scalar const& x,
                                              float64 cutoff = 1e-30);

/// Elementwise minimum of two diagonal tensors or a diagonal tensor and scalar.
[[nodiscard]] DiagonalTensorPtr minimum(DiagonalTensorCPtr x, DiagonalTensorCPtr y);
[[nodiscard]] DiagonalTensorPtr minimum(DiagonalTensorCPtr x, BlockBackend::Scalar const& y);
[[nodiscard]] DiagonalTensorPtr minimum(BlockBackend::Scalar const& x, DiagonalTensorCPtr y);

/// Elementwise maximum of two diagonal tensors or a diagonal tensor and scalar.
[[nodiscard]] DiagonalTensorPtr maximum(DiagonalTensorCPtr x, DiagonalTensorCPtr y);
[[nodiscard]] DiagonalTensorPtr maximum(DiagonalTensorCPtr x, BlockBackend::Scalar const& y);
[[nodiscard]] DiagonalTensorPtr maximum(BlockBackend::Scalar const& x, DiagonalTensorCPtr y);

/// The exponential function.
///
/// For a tensor, viewed as a linear map from its domain to its codomain, the exponential
/// function is defined via its power series. For a diagonal tensor, this is equivalent to
/// the elementwise exponential function.
///
/// @param obj Input tensor (as a linear map).
/// @returns The matrix exponential of `obj`.
[[nodiscard]] TensorPtr exp(TensorCPtr obj);

} // namespace lasyten
