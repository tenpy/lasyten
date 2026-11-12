#pragma once

#include <lasyten/lasyten.h>
#include <lasyten/tensors/forward_declare.h>
#include <lasyten/tensors/labels.h>
#include <lasyten/tensors/ops_algebra.h>

#include <optional>
#include <string>
#include <tuple>
#include <vector>

namespace lasyten {

/// Apply a mask to *both* legs of a diagonal tensor.
///
/// The mask must be a projection, i.e. its large leg is in the domain, at the top.
/// We apply the mask via map composition::
///
///     |     ┏━━┷━━┓
///     |     ┃ M.hc┃
///     |     ┗━━┯━━┛
///     |     ┏━━┷━━┓
///     |     ┃  D  ┃
///     |     ┗━━┯━━┛
///     |     ┏━━┷━━┓
///     |     ┃  M  ┃
///     |     ┗━━┯━━┛
///
/// where `M.hc == dagger(M)`.
///
/// @param tensor The diagonal tensor to project.
/// @param mask A *projection* mask. Its large leg must equal `tensor.leg`.
/// @returns A masked diagonal tensor whose `leg` is `mask.small_leg`. Labels match `tensor`.
///
/// See also: `apply_mask`.
[[nodiscard]] DiagonalTensorPtr apply_mask_DiagonalTensor(DiagonalTensorCPtr tensor,
                                                          MaskCPtr mask);

/// The eigen-decomposition of a hermitian tensor.
///
/// A tensor decomposition `tensor ~ V @ W @ dagger(V)` with the following properties:
///
/// - `V` is unitary: `dagger(V) @ V ~ eye` and `V @ dagger(V) ~ eye`.
/// - `W` is a `DiagonalTensor` with the real eigenvalues of `tensor`.
///
/// *Assumes* that `tensor` is hermitian: `dagger(tensor) ~ tensor`, which requires in particular
/// that `tensor.domain == tensor.codomain`.
///
/// Graphically::
///
///     |                                 │   │   │   │
///     |                                ┏┷━━━┷━━━┷━━━┷┓
///     |                                ┃  dagger(V)  ┃
///     |        │   │   │   │           ┗━━━━━━┯━━━━━━┛
///     |       ┏┷━━━┷━━━┷━━━┷┓               ┏━┷━┓
///     |       ┃   tensor    ┃    ==         ┃ W ┃
///     |       ┗┯━━━┯━━━┯━━━┯┛               ┗━┯━┛
///     |        │   │   │   │           ┏━━━━━━┷━━━━━━┓
///     |                                ┃      V      ┃
///     |                                ┗┯━━━┯━━━┯━━━┯┛
///     |                                 │   │   │   │
///
/// @param tensor The hermitian tensor to decompose.
/// @param new_labels Labels for the new legs. Three labels `[a, b, c]` result in
///     `V.labels[-1] == a` and `W.labels == [b, c]`. Two labels `[a, b]` are equivalent to
///     `[a, b, a]`. A single label `a` is equivalent to `[a, a*, a]`.
/// @param new_leg_dual If the new leg should be a ket space (`false`) or bra space (`true`).
/// @param sort How the eigenvalues are sorted *within* each charge block.
///     One of `"m>"`, `"m<"`, `">"`, `"<"`, `"LI"`, `"SI"`, or `nullopt`.
///     Defaults to `nullopt`, which is the same as `"<"`. See `argsort` for details.
/// @returns `(W, V)`: real eigenvalues and orthonormal eigenvectors.
[[nodiscard]] std::tuple<DiagonalTensorPtr, TensorPtr> eigh(
  TensorCPtr tensor,
  LegLabelList new_labels,
  bool new_leg_dual = false,
  std::optional<std::string> sort = std::nullopt);

/// The eigen-decomposition of a general (not necessarily hermitian) tensor.
///
/// A tensor decomposition `tensor ~ V @ W @ pinv(V)` with the following properties:
///
/// - `W` is a `DiagonalTensor` with the (generally complex) eigenvalues of `tensor`.
/// - `V` contains the corresponding right eigenvectors. It is in general not unitary.
///
/// Requires that `tensor.domain == tensor.codomain`.
///
/// @param tensor The tensor to decompose.
/// @param new_labels Labels for the new legs. Three labels `[a, b, c]` result in
///     `V.labels[-1] == a` and `W.labels == [b, c]`. Two labels `[a, b]` are equivalent to
///     `[a, b, a]`. A single label `a` is equivalent to `[a, a*, a]`.
/// @param new_leg_dual If the new leg should be a ket space (`false`) or bra space (`true`).
/// @param sort How the eigenvalues are sorted *within* each charge block.
///     One of `"m>"`, `"m<"`, `">"`, `"<"`, `"LI"`, `"SI"`, or `nullopt`.
///     Defaults to `nullopt`, which leaves the backend default order (unsorted for `eig`).
///     See `argsort` for details.
/// @returns `(W, V)`: eigenvalues and right eigenvectors.
[[nodiscard]] std::tuple<DiagonalTensorPtr, TensorPtr> eig(
  TensorCPtr tensor,
  LegLabelList new_labels,
  bool new_leg_dual = false,
  std::optional<std::string> sort = std::nullopt);

/// Eigenvalues of a hermitian tensor, without eigenvectors.
///
/// *Assumes* that `tensor` is hermitian: `dagger(tensor) ~ tensor`, which requires in particular
/// that `tensor.domain == tensor.codomain`.
///
/// @param tensor The hermitian tensor.
/// @param new_labels Labels for the eigenvalue tensor `W` only. One label `a` is equivalent to
///     `[a, a*]`. Two labels `[b, c]` set `W.labels == [b, c]`.
/// @param new_leg_dual If the new leg should be a ket space (`false`) or bra space (`true`).
/// @param sort How the eigenvalues are sorted *within* each charge block. See `argsort` for
///     details. Defaults to `nullopt`, which is the same as `"<"`.
/// @returns `W`: real eigenvalues as a `DiagonalTensor`.
[[nodiscard]] DiagonalTensorPtr eigvalsh(TensorCPtr tensor,
                                         LegLabelList new_labels,
                                         bool new_leg_dual = false,
                                         std::optional<std::string> sort = std::nullopt);

/// Eigenvalues of a general tensor, without eigenvectors.
///
/// Requires that `tensor.domain == tensor.codomain`.
///
/// @param tensor The tensor.
/// @param new_labels Labels for the eigenvalue tensor `W` only. One label `a` is equivalent to
///     `[a, a*]`. Two labels `[b, c]` set `W.labels == [b, c]`.
/// @param new_leg_dual If the new leg should be a ket space (`false`) or bra space (`true`).
/// @param sort How the eigenvalues are sorted *within* each charge block. See `argsort` for
///     details. Defaults to `nullopt`, which leaves the backend default order (unsorted).
/// @returns `W`: (generally complex) eigenvalues as a `DiagonalTensor`.
[[nodiscard]] DiagonalTensorPtr eigvals(TensorCPtr tensor,
                                        LegLabelList new_labels,
                                        bool new_leg_dual = false,
                                        std::optional<std::string> sort = std::nullopt);

/// The entropy of a probability distribution.
///
/// Assumes that `p` is a probability distribution, i.e. real, non-negative and normalized to
/// `p.sum() == 1.`.
///
/// For `n == 1`, we compute the von-Neumann entropy
/// @f$S_\text{vN} = -\mathrm{Tr}[p \mathrm{log} p]@f$.
/// Otherwise, we compute the Renyi entropy
/// @f$S_n = \frac{1}{1 - n} \mathrm{log} \mathrm{Tr}[p^n]@f$.
///
/// For non-abelian symmetries and anyonic gradings we have
/// @f$p = \bigotimes_a \rho_a \mathbb{1}_a@f$ with @f$\rho_a \ge 0@f$
/// and @f$\sum_a d_a \rho_a = 1@f$. The entropy is then obtained as
/// @f$S_\text{vN} = \sum_a d_a \rho_a \mathrm{log} \rho_a@f$ or
/// @f$S_n = \frac{1}{1 - n} \mathrm{log} \sum_a d_a \rho_a^n@f$ where @f$d_a@f$
/// is the quantum dimension of sector @f$a@f$ (see `Symmetry::qdim`).
///
/// @param p Probability distribution as a diagonal tensor.
/// @param n Renyi index; `n == 1` selects von-Neumann entropy (default `1`).
/// @returns The entropy scalar.
[[nodiscard]] BlockBackend::Scalar entropy(DiagonalTensorCPtr p, float64 n = 1);

/// The LQ decomposition of a tensor.
///
/// A tensor decomposition `tensor ~ L @ Q` with the following properties:
///
/// - `L` has a lower triangular structure *in the coupled basis*.
/// - `Q` is an isometry: `dagger(Q) @ Q ~ eye`.
///
/// Graphically::
///
///     |                                 │   │   │   │
///     |                                ┏┷━━━┷━━━┷━━━┷┓
///     |        │   │   │   │           ┃      Q      ┃
///     |       ┏┷━━━┷━━━┷━━━┷┓          ┗━━━━━━┯━━━━━━┛
///     |       ┃   tensor    ┃    ==           │
///     |       ┗━━┯━━━┯━━━┯━━┛          ┏━━━━━━┷━━━━━━┓
///     |          │   │   │             ┃      L      ┃
///     |                                ┗━━┯━━━┯━━━┯━━┛
///     |                                   │   │   │
///
/// We always compute the "reduced", a.k.a. "economic" version.
/// To group the legs differently, use `permute_legs` or `combine_to_matrix` first.
///
/// @param tensor The tensor to decompose.
/// @param new_labels Labels for the new legs. Either two labels `[a, b]` such that
///     `L.labels[-1] == a` and `Q.labels[0] == b`. A single label `a` is equivalent to
///     `[a, a*]`. `nullopt` leaves the new legs unlabeled.
/// @param new_leg_dual If the new leg should be a ket space (`false`) or bra space (`true`).
/// @param charge_leg_top For a `ChargedTensor` input: if `true`, the charge leg ends up on the
///     top factor `Q`; if `false`, on the bottom factor `L`. Ignored otherwise.
/// @returns `(L, Q)`.
[[nodiscard]] std::tuple<TensorPtr, TensorPtr> lq(
  TensorCPtr tensor,
  std::optional<LegLabelList> new_labels = std::nullopt,
  bool new_leg_dual = false,
  bool charge_leg_top = true);

/// The QR decomposition of a tensor.
///
/// A tensor decomposition `tensor ~ Q @ R` with the following properties:
///
/// - `Q` is an isometry: `dagger(Q) @ Q ~ eye`.
/// - `R` has an upper triangular structure *in the coupled basis*.
///
/// Graphically::
///
///     |                                 │   │   │   │
///     |                                ┏┷━━━┷━━━┷━━━┷┓
///     |        │   │   │   │           ┃      R      ┃
///     |       ┏┷━━━┷━━━┷━━━┷┓          ┗━━━━━━┯━━━━━━┛
///     |       ┃   tensor    ┃    ==           │
///     |       ┗━━┯━━━┯━━━┯━━┛          ┏━━━━━━┷━━━━━━┓
///     |          │   │   │             ┃      Q      ┃
///     |                                ┗━━┯━━━┯━━━┯━━┛
///     |                                   │   │   │
///
/// We always compute the "reduced", a.k.a. "economic" version.
/// To group the legs differently, use `permute_legs` or `combine_to_matrix` first.
///
/// @param tensor The tensor to decompose.
/// @param new_labels Labels for the new legs. Either two labels `[a, b]` such that
///     `Q.labels[-1] == a` and `R.labels[0] == b`. A single label `a` is equivalent to
///     `[a, a*]`. `nullopt` leaves the new legs unlabeled.
/// @param new_leg_dual If the new leg should be a ket space (`false`) or bra space (`true`).
/// @param charge_leg_top For a `ChargedTensor` input: if `true`, the charge leg ends up on the
///     top factor `R`; if `false`, on the bottom factor `Q`. Ignored otherwise.
/// @returns `(Q, R)`.
[[nodiscard]] std::tuple<TensorPtr, TensorPtr> qr(
  TensorCPtr tensor,
  std::optional<LegLabelList> new_labels = std::nullopt,
  bool new_leg_dual = false,
  bool charge_leg_top = true);

/// The singular value decomposition (SVD) of a tensor.
///
/// A tensor decomposition `tensor ~ U @ S @ Vh` with the following properties:
///
/// - `Vh` and `U` are isometries: `dagger(U) @ U ~ eye ~ Vh @ dagger(Vh)`.
/// - `S` is a `DiagonalTensor` with real, non-negative entries.
/// - If `tensor` is a matrix (exactly one leg each in domain and codomain), it reproduces the
///   usual matrix SVD.
///
/// The basis for the newly generated leg is chosen arbitrarily, and in particular, unlike e.g.
/// `numpy.linalg.svd`, it is not guaranteed that `S.diag_numpy` is sorted.
///
/// Graphically::
///
///     |                                 │   │   │   │
///     |                                ┏┷━━━┷━━━┷━━━┷┓
///     |                                ┃      Vh     ┃
///     |        │   │   │   │           ┗━━━━━━┯━━━━━━┛
///     |       ┏┷━━━┷━━━┷━━━┷┓               ┏━┷━┓
///     |       ┃   tensor    ┃    ==         ┃ S ┃
///     |       ┗━━┯━━━┯━━━┯━━┛               ┗━┯━┛
///     |          │   │   │             ┏━━━━━━┷━━━━━━┓
///     |                                ┃      U      ┃
///     |                                ┗━━┯━━━┯━━━┯━━┛
///     |                                   │   │   │
///
/// We always compute the "reduced", a.k.a. "economic" version of SVD, where the isometries are
/// (in general) not full unitaries.
///
/// To group the legs differently, use `permute_legs` or `combine_to_matrix` first.
///
/// @param tensor The tensor to decompose.
/// @param new_labels Labels for the new legs. Four labels `[a, b, c, d]` result in
///     `U.labels[-1] == a`, `S.labels == [b, c]` and `Vh.labels[0] == d`.
///     Two labels `[a, b]` are equivalent to `[a, b, a, b]`.
///     A single label `a` is equivalent to `[a, a*, a, a*]`.
///     `nullopt` leaves the new legs unlabeled.
/// @param new_leg_dual If the new leg should be a ket space (`false`) or bra space (`true`).
/// @param charge_leg_top For a `ChargedTensor` input: if `true`, the charge leg ends up on the
///     top factor `Vh`; if `false`, on the bottom factor `U`. Ignored otherwise.
/// @param algorithm Algorithm (driver) for the block-wise SVD. Choices are backend-specific;
///     see `BlockBackend::svd_algorithms` of `tensor.backend.block_backend`. `nullopt` uses the
///     backend default.
/// @returns `(U, S, Vh)`.
[[nodiscard]] std::tuple<TensorPtr, DiagonalTensorPtr, TensorPtr> svd(
  TensorCPtr tensor,
  std::optional<LegLabelList> new_labels = std::nullopt,
  bool new_leg_dual = false,
  bool charge_leg_top = true,
  std::optional<std::string> algorithm = std::nullopt);

/// Truncate an existing SVD by applying a projection mask to `U`, `S`, and `Vh`.
///
/// @param U,S,Vh Factors of an existing SVD (`tensor ~ U @ S @ Vh`).
/// @param mask Projection mask selecting which singular values / bond sectors to keep.
/// @returns The truncated `(U, S, Vh)`.
[[nodiscard]] std::tuple<TensorPtr, DiagonalTensorPtr, TensorPtr>
svd_apply_mask(TensorCPtr U, DiagonalTensorCPtr S, TensorCPtr Vh, MaskCPtr mask);

/// Given *normalized* singular values, determine which to keep.
///
/// In the case of non-Abelian group symmetries, the quantum dimensions need to be considered when
/// truncating. Each independent entry @f$S_i@f$ in `S` is associated with a sector @f$a@f$
/// of the symmetry, e.g. the spin 1 sector of an SU(2) symmetry. It represents an entire multiplet
/// (e.g. a triplet in the spin 1 case) of degenerate singular values in that charge sector, with
/// the degeneracy given by the quantum dimension @f$d_a@f$ of the sector. When converting to a
/// non-symmetric representation, e.g. via `S.diagonal_to_numpy()`, that value @f$S_i@f$ will
/// appear @f$d_a@f$ times. In particular, the error that we get by truncating some of the
/// @f$S_i@f$ is given by @f$\epsilon = \sum_{i\,\mathrm{discarded}} d_{a_i} S_i^2@f$, such that
/// the quantum dimensions need to be considered when choosing which singular values to keep for an
/// optimal truncation error.
///
/// This is why the singular values are prioritized by largest @f$d_{a_i} S_i^2@f$.
///
/// For anyonic symmetries we lose the interpretation as a multiplet, since @f$d_a@f$ is in
/// general not integer, but the formula for the error holds, and the considerations for selecting
/// which singular values to keep apply just the same.
/// For abelian groups or for fermions these considerations become trivial, since all sectors are
/// one-dimensional.
///
/// @param S Singular values, normalized to `S.norm() == 1.`.
/// @param chi_max Keep at most this many singular values. `nullopt` means no constraint.
/// @param chi_min Keep at least this many singular values (default `1`).
/// @param degeneracy_tol Do not split (nearly) degenerate singular values.
///     We count `S[i]` and `S[j]` as nearly degenerate if `|log(S[i]/S[j])| < degeneracy_tol`,
///     or equivalently if `|S[i] - S[j]|/S[j] < exp(degeneracy_tol) - 1 ~= degeneracy_tol`.
///     In that case, either both are kept or both are truncated.
/// @param trunc_cut A *lower* bound on the incurred truncation error: as long as the error remains
///     below this threshold, singular values will be truncated. In particular, as long as
///     `sum_{i discarded} d[i] S[i] ** 2 <= trunc_cut ** 2`, where `d[i]` is the quantum dimension
///     (always one for abelian symmetries).
/// @param svd_min Discard all singular values below this threshold `S[i] < svd_min`.
///     Intended to exclude singular values that cannot be distinguished from zero at the given
///     precision. It does *not* directly bound the truncation error; use `trunc_cut` for that.
/// @param minimize_error If true, minimize truncation error by keeping as many singular values as
///     allowed by the other constraints; otherwise keep as few as possible.
/// @param mask_labels Labels for the returned mask. Either two string labels or `nullopt`
///     (default). By default the mask has labels `[S.labels[0], dual_label(S.labels[0])]`.
/// @returns `(mask, err, new_norm)` where `mask` indicates singular values to keep,
///     `err == norm(S[not mask])` is the truncation error (distance to the un-normalized
///     approximation), and `new_norm == norm(S[mask])`.
[[nodiscard]] std::tuple<MaskPtr, float64, float64> truncate_singular_values(
  DiagonalTensorCPtr S,
  std::optional<int64> chi_max = std::nullopt,
  int64 chi_min = 1,
  float64 degeneracy_tol = 0.,
  float64 trunc_cut = 0.,
  float64 svd_min = 0.,
  bool minimize_error = true,
  std::optional<LegLabelList> mask_labels = std::nullopt);

/// Truncated version of `svd`.
///
/// Given a normalized SVD `tensor = norm(tensor) * X @ Y @ Z`, we keep only some of the singular
/// values ``Y_k`` and discard ``Y_d`` such that ``Y = [[Y_k 0] [0 Y_d]]``.
/// The resulting closes approximation is `T_a = norm(tensor) * X_k @ Y_k @ Z_k`.
/// The relative approximation error is `err = norm(tensor - T_a) / norm(tensor) = norm(Y_d)`.
/// The norm of the approximation is `norm(T_a) = norm(tensor) * sqrt(1 - err ** 2) <=
/// norm(tensor)`` and is in particular reduced compared to the original tensor. It can therefore
/// be of interest to rescale the result.
///
/// @param tensor,new_labels,new_leg_dual,charge_leg_top,algorithm Same as for the non-truncated
///     `svd`.
/// @param normalize_to If `nullopt` (default), the resulting truncated singular values are not
///     renormalized, and we get ``norm(S) = norm(T) sqrt(1 - err ** 2) <= norm(T)``.
///     If set, we get ``norm(S) == normalize_to``.
/// @param chi_max,chi_min,degeneracy_tol,trunc_cut,svd_min Truncation options; see
///     `truncate_singular_values`.
/// @returns `(U, S, Vh, err, renormalize)` such that
///     `U` are the kept columns `X_k` of the left isometry
///     `S` are the kept singular values,
///         either unscaled `S = norm(tensor) * Y_k` if `normalize_to = nullopt`
///         or scaled `S = normalize_to / norm(Y_k) * Y_k` to target norm otherwise.
///     `Vh` are the kept rows `Y_k` of the right isometry
///     `err` is the *relative* two-norm approximation error `err = norm(tensor - T_a) /
///     norm(tensor)`.
///         it corresponds to the discarded weight `err = norm(Y_d)` of normalized singular values.
///     `renormalize` is the factor such that `T_a = renormalize * U @ S @ Vh` is the closest
///     approximation
///         if `normalize_to = nullopt` we have `renormalize = 1`
///         otherwise `renormalize = norm(tensor) / normalize_to * norm(Y_k)
///
/// See also: `svd`.
[[nodiscard]] std::tuple<TensorPtr, DiagonalTensorPtr, TensorPtr, float64, float64> truncated_svd(
  TensorCPtr tensor,
  std::optional<LegLabelList> new_labels = std::nullopt,
  bool new_leg_dual = false,
  bool charge_leg_top = true,
  std::optional<std::string> algorithm = std::nullopt,
  std::optional<float64> normalize_to = std::nullopt,
  std::optional<int64> chi_max = std::nullopt,
  int64 chi_min = 1,
  float64 degeneracy_tol = 0.,
  float64 trunc_cut = 0.,
  float64 svd_min = 0.);

} // namespace lasyten
