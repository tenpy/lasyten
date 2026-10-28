#pragma once

#include <lasyten/backends/tensor_backend.h>
#include <lasyten/block_backend/dtypes.h>
#include <lasyten/lasyten.h>
#include <lasyten/symmetries/spaces.h>
#include <lasyten/tensors/forward_declare.h>
#include <lasyten/tensors/labels.h>
#include <lasyten/tensors/ops_algebra.h>

#include <optional>
#include <string>
#include <vector>

namespace lasyten {

/// The identity tensor on a given leg.
///
/// Returns a `DiagonalTensor` if `diagonal` is true, otherwise a `SymmetricTensor`.
///
/// @param leg The space for the identity.
/// @param backend Backend; `nullptr` selects the default.
/// @param labels Optional leg labels; `nullopt` means unlabeled.
/// @param dtype Element dtype.
/// @param device Optional device string; `nullopt` uses the backend default.
/// @param diagonal If true, return a diagonal identity.
[[nodiscard]] TensorPtr eye(Space::Ptr leg,
                            TensorBackend::Ptr backend = nullptr,
                            std::optional<LegLabelList> labels = std::nullopt,
                            Dtype dtype = Dtype::Float64,
                            std::optional<std::string> device = std::nullopt,
                            bool diagonal = true);

/// Convert an existing tensor, checking (co)domain / backend compatibility.
///
/// @param obj Source tensor.
/// @param codomain,domain Codomain and optional domain.
/// @param backend Optional backend override; `nullptr` keeps `obj`'s backend.
/// @param labels Optional leg labels; `nullopt` keeps existing labels.
/// @param dtype Optional dtype conversion; `nullopt` keeps `obj`'s dtype.
/// @param device Optional device; `nullopt` keeps `obj`'s device.
[[nodiscard]] TensorPtr tensor(TensorCPtr obj,
                               TensorProduct::Ptr codomain,
                               TensorProduct::Ptr domain = nullptr,
                               TensorBackend::Ptr backend = nullptr,
                               std::optional<LegLabelList> labels = std::nullopt,
                               std::optional<Dtype> dtype = std::nullopt,
                               std::optional<std::string> device = std::nullopt);

/// Convert a dense block to a `SymmetricTensor`.
///
/// @param obj Dense block data.
/// @param codomain,domain Codomain and optional domain.
/// @param backend Backend; `nullptr` selects the default.
/// @param labels Optional leg labels; `nullopt` means unlabeled.
/// @param dtype Optional dtype; `nullopt` is inferred from `obj`.
/// @param device Optional device; `nullopt` uses the backend default.
/// @param understood_braiding Whether braiding conventions are already applied.
[[nodiscard]] SymmetricTensorPtr tensor(BlockBackend::BlockPtr obj,
                                        TensorProduct::Ptr codomain,
                                        TensorProduct::Ptr domain = nullptr,
                                        TensorBackend::Ptr backend = nullptr,
                                        std::optional<LegLabelList> labels = std::nullopt,
                                        std::optional<Dtype> dtype = std::nullopt,
                                        std::optional<std::string> device = std::nullopt,
                                        bool understood_braiding = false);

/// Add a trivial leg to a tensor.
///
/// A trivial leg is one-dimensional and consists only of the trivial sector of the symmetry.
/// `DiagonalTensor` and `Mask` do not support adding legs and are converted to
/// `SymmetricTensor` first.
///
/// The position of the new leg can be specified in three mutually exclusive ways via
/// `legs_pos`, `codomain_pos`, or `domain_pos`. If `legs_pos` is used,
/// `result.legs[legs_pos]` will be the trivial leg. In most cases that unambiguously
/// assigns it to either the domain or the codomain. If ambiguous
/// (`legs_pos == num_codomain_legs`), it is added to the codomain. Alternatively, it can
/// be added to the codomain at `codomain[codomain_pos]` or to the domain at `domain_pos`.
/// Note the implications for `is_dual`: with `legs_pos`,
/// `result.legs[legs_pos].is_dual == is_dual`, but with `domain_pos`,
/// `result.domain[domain_pos].is_dual == is_dual` (mutually opposite conventions).
/// Per default we use `legs_pos == 0`, i.e. add at `legs[0]` / `codomain[0]`.
///
/// @param tens The tensor to add a leg to.
/// @param legs_pos Position in `legs`; `nullopt` if unused.
/// @param codomain_pos Position in the codomain; `nullopt` if unused.
/// @param domain_pos Position in the domain; `nullopt` if unused.
/// @param label Label for the new leg; `nullopt` for unlabeled.
/// @param is_dual If true, add a dual (bra-like) leg.
[[nodiscard]] TensorPtr add_trivial_leg(TensorCPtr tens,
                                        std::optional<int64> legs_pos = std::nullopt,
                                        std::optional<int64> codomain_pos = std::nullopt,
                                        std::optional<int64> domain_pos = std::nullopt,
                                        std::string label = "",
                                        bool is_dual = false);

/// Return a zero tensor with the same type, dtype, legs, backend and labels.
///
/// @param tensor Template tensor whose metadata is copied.
[[nodiscard]] TensorPtr zero_like(TensorCPtr tensor);

/// Stack a grid of tensors along existing legs.
///
/// Null cells are interpreted as all-zero tensors.
/// The tensors are stacked along the first leg in their codomain and the final leg in their
/// domain. The resulting legs are
/// @f$result.codomain[0] = V = \bigoplus_m V_m@f$ and
/// @f$result.domain[-1] = W = \bigoplus_n W_n@f$, where @f$V_m@f$ is the first codomain leg
/// of all tensors in the @f$m@f$-th row `grid[m]`, and @f$W_n@f$ is the last domain leg of all
/// tensors in the @f$n@f$-th column, i.e. for the tensors `[row[n] for row in grid]`.
///
/// Graphically::
///
///     |                                                      W
///     |                                              │   │ ┏━┷━┓
///     |                                              │   │ ┃p_n┃
///     |                  W                           │   │ ┗━┯━┛
///     |          │   │   │                           │   │   │ W_n
///     |       ┏━━┷━━━┷━━━┷━━┓                     ┏━━┷━━━┷━━━┷━━┓
///     |       ┃     res     ┃    ==   sum_{m,n}   ┃  grid[m][n] ┃
///     |       ┗┯━━━┯━━━┯━━━┯┛                     ┗┯━━━┯━━━┯━━━┯┛
///     |        │   │   │   │                   V_m │   │   │   │
///     |        V                                 ┏━┷━┓ │   │   │
///     |                                          ┃i_m┃ │   │   │
///     |                                          ┗━┯━┛ │   │   │
///     |
///
///
/// where @f$p_n : W = \bigoplus_{n'} W_{n'} \to W_n@f$ is the projection map of the direct sum
/// and @f$i_m : V_m \to \bigoplus_{m'} V_{m'}@f$ the inclusion.
///
/// @param grid Row-major grid of tensors (null cell = zero). All legs except those along
///     which stacking happens must match across the grid; tensors in the same row share the
///     first codomain leg, and tensors in the same column share the last domain leg.
/// @param labels Optional labels for the result; `nullopt` means unlabeled.
/// @param dtype Optional dtype; `nullopt` uses the common dtype of the grid.
/// @param row_labels Optional label per grid row, stored as `DirectSumSpace::summand_labels`
///     on the resulting codomain leg; `nullopt` means unlabeled. If given, must have one
///     entry per row.
/// @param col_labels Optional label per grid column, stored as `DirectSumSpace::summand_labels`
///     on the resulting domain leg; `nullopt` means unlabeled. If given, must have one entry
///     per column.
[[nodiscard]] TensorPtr tensor_from_grid(std::vector<std::vector<TensorPtr>> grid,
                                         std::optional<LegLabelList> labels = std::nullopt,
                                         std::optional<Dtype> dtype = std::nullopt,
                                         std::optional<LegLabelList> row_labels = std::nullopt,
                                         std::optional<LegLabelList> col_labels = std::nullopt);

/// Extract cell ``(row, col)`` from a tensor stacked by `tensor_from_grid`.
///
/// Inverts the direct-sum stacking
/// @f$\mathrm{res} = \sum_{m,n} i_m \circ \mathrm{grid}[m][n] \circ p_n@f$
/// via @f$\mathrm{grid}[m][n] = p_m \circ \mathrm{res} \circ i_n@f$, implemented as
/// `apply_mask` with `DirectSumSpace::projection_onto_summand` on the stacking legs.
///
/// If a stacking leg is a plain `ElementarySpace` (not a `DirectSumSpace`; possible for an
/// isomorphic stacked tensor not produced by `tensor_from_grid`), that side is treated as the
/// identity and only `row`/`col == 0` or `-1` (as an index, not a label) is valid. Recovered
/// cell spaces are isomorphic to the original grid cells' stacking legs (same
/// sectors/multiplicities), but need not be the same objects. Labels of `tensor` are preserved.
///
/// @param tensor Result of `tensor_from_grid` (or an isomorphic stacked tensor).
/// @param row,col Summand index or label into the stacked codomain / domain legs (negative
///     indices OK; labels resolved via `DirectSumSpace::get_summand_idx`).
/// @param row_leg Leg of the row direct sum (default: first codomain leg).
/// @param col_leg Leg of the column direct sum (default: last domain leg, ``-1``).
[[nodiscard]] TensorPtr tensor_grid_cell(TensorCPtr tensor,
                                         DirectSumSpace::SummandRef row,
                                         DirectSumSpace::SummandRef col,
                                         LegRef row_leg = int64{ 0 },
                                         LegRef col_leg = int64{ -1 });

/// Extract a subgrid from a tensor stacked by `tensor_from_grid`.
///
/// Like `tensor_grid_cell`, but for several rows and columns at once. The result is stacked
/// again via `tensor_from_grid` from the cells ``(rows[i], cols[j])``, so its stacking legs are
/// `DirectSumSpace`s with the selected summands in the given order (and their labels).
///
/// To re-combine subgrids, pass them as cells of a grid to `tensor_from_grid` *without*
/// `row_labels` / `col_labels`. The stacking legs of the subgrids are `DirectSumSpace`s, so
/// they are flattened and their `summand_labels` are preserved in the result. Passing explicit
/// labels would instead raise, as one label cannot name several flattened summands.
///
/// @param tensor Result of `tensor_from_grid` (or an isomorphic stacked tensor).
/// @param rows,cols Non-empty lists of summand indices or labels, as in `tensor_grid_cell`.
/// @param row_leg Leg of the row direct sum (default: first codomain leg).
/// @param col_leg Leg of the column direct sum (default: last domain leg, ``-1``).
[[nodiscard]] TensorPtr tensor_subgrid(TensorCPtr tensor,
                                       std::vector<DirectSumSpace::SummandRef> rows,
                                       std::vector<DirectSumSpace::SummandRef> cols,
                                       LegRef row_leg = int64{ 0 },
                                       LegRef col_leg = int64{ -1 });

/// Project selected tensor legs onto summands of their `DirectSumSpace`s.
///
/// Unlike `tensor_grid_cell`, this operation can be applied to any tensor with direct-sum legs,
/// including tensors produced by contracting gridded tensors. Each cell reference is resolved
/// against the corresponding leg's summand labels or indices. If `squeeze` is true, the
/// projected legs are removed; this requires each selected summand to be trivial.
///
/// @param tensor Tensor to project.
/// @param legs Leg labels or indices to project.
/// @param cells Summand labels or indices, parallel to `legs`.
/// @param squeeze Whether to remove the projected trivial legs afterwards.
[[nodiscard]] TensorPtr grid_project(TensorCPtr tensor,
                                     std::vector<LegRef> legs,
                                     std::vector<DirectSumSpace::SummandRef> cells,
                                     bool squeeze = false);

} // namespace lasyten
