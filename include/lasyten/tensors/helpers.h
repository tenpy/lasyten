#pragma once

#include <lasyten/backends/abelian.h>
#include <lasyten/backends/fusion_tree_backend.h>
#include <lasyten/backends/tensor_backend.h>
#include <lasyten/block_backend/dtypes.h>
#include <lasyten/symmetries/spaces.h>
#include <lasyten/tensors/labels.h>
#include <lasyten/tensors/mask.h>
#include <lasyten/tensors/symmetric_tensor.h>
#include <lasyten/tensors/tensor.h>

#include <map>
#include <optional>
#include <string>
#include <tuple>
#include <utility>
#include <variant>
#include <vector>

namespace lasyten {

/// Check if legs are compatible (equal if `expect_equal`, otherwise mutually dual).
///
/// `TensorProduct` (co)domains use the `Space` overload. Single tensor legs
/// (`ElementarySpace` / `LegPipe`) use the `Leg` overload.
///
/// @param legs1,legs2 Legs to compare.
/// @param expect_equal If true, require equality; otherwise require mutual duals.
void _check_compatible_legs(std::vector<Leg::Ptr> const& legs1,
                            std::vector<Leg::Ptr> const& legs2,
                            bool expect_equal = true);
void _check_compatible_legs(std::vector<Space::Ptr> const& legs1,
                            std::vector<Space::Ptr> const& legs2,
                            bool expect_equal = true);

/// Compose `tensor` with a mask, preserving the leg order of `tensor`.
///
/// We expect `tensor.codomain[leg_idx] == mask.domain[0]` if `leg_idx` is in the
/// codomain, or `tensor.domain[co_domain_idx] == mask.codomain[0]` otherwise.
///
/// Graphically::
///
///     |      │   │   │            │   │  ┏┷┓
///     |     ┏┷━━━┷━━━┷┓           │   │  ┃M┃
///     |     ┃ tensor  ┃           │   │  ┗┯┛
///     |     ┗┯━━━┯━━━┯┛   OR     ┏┷━━━┷━━━┷┓
///     |      │  ┏┷┓  │           ┃ tensor  ┃
///     |      │  ┃M┃  │           ┗┯━━━┯━━━┯┛
///     |      │  ┗┯┛  │            │   │   │
///
/// Note that the resulting leg may be smaller than before (for a projection mask
/// in the codomain or an inclusion mask in the domain) or larger (otherwise).
///
/// The result has the same leg order and labels as `tensor`.
///
/// @param tensor Tensor to compose with the mask.
/// @param mask Mask applied on `leg_idx`.
/// @param leg_idx Index of the leg to compose with.
/// @returns Tensor with the mask applied, same leg order/labels as `tensor`.
[[nodiscard]] TensorPtr _compose_with_Mask(TensorCPtr tensor, MaskCPtr mask, int64 leg_idx);

/// Restricted case of `compose` where we assume that both tensors are SymmetricTensor.
///
/// If both tensors have no remaining open legs, returns a scalar (the contraction).
/// Used by both `compose` and `tdot`.
///
/// @param tensor1,tensor2 Symmetric tensors to compose: `tensor1` after `tensor2`.
/// @param relabel1,relabel2 Optional label maps applied before composition.
///     `nullopt` means no relabel.
/// @returns The composed tensor, or a scalar if no open legs remain.
[[nodiscard]] std::variant<SymmetricTensorPtr, BlockBackend::Scalar> _compose_SymmetricTensors(
  SymmetricTensorCPtr tensor1,
  SymmetricTensorCPtr tensor2,
  std::optional<std::map<std::string, std::string>> relabel1 = std::nullopt,
  std::optional<std::map<std::string, std::string>> relabel2 = std::nullopt);

/// Convert tensor from abelian backend to FT backend. Return the data.
///
/// Same idea as `_convert_FT_to_abelian`; see its documentation.
///
/// @param tensor Source tensor on an abelian backend.
/// @param backend Target fusion-tree backend.
/// @param dtype Element dtype for the result data.
/// @param device Device string for the result data.
/// @returns Fusion-tree backend data.
[[nodiscard]] FusionTreeData::Ptr _convert_abelian_to_FT(TensorCPtr tensor,
                                                         FusionTreeBackend::Ptr backend,
                                                         Dtype dtype,
                                                         std::string device);

/// Convert tensor from FT backend to abelian backend. Return the data.
///
/// - For abelian symmetries, a fusion tree is completely determined by its uncoupled sectors
/// - This means that each forest block consists of a single tree block
/// - The blocks of the abelian backend correspond one-to-one to tree blocks in the FT backend,
///   up to reshaping and transposing
/// - All that remains is to make sure we loop over all of them in an efficient manner
/// - It is convenient to do the outer loops over combinations of uncoupled sectors
///   - This way, we have the abelian block_inds by construction
///   - We need to compute the coupled sectors to check for valid fusion channels anyway,
///     which gives us the FT block inds with one additional lookup
///   - While we jump back-and-forth between different coupled sectors, and thus different FT
///     blocks while iterating, we know that we visit the tree-blocks within each FT block *in
///     order*, and we can thus keep track of where we are within each FT block easily
///
/// @param tensor Source tensor on a fusion-tree backend.
/// @param backend Target abelian backend.
/// @param dtype Element dtype for the result data.
/// @param device Device string for the result data.
/// @returns Abelian backend data.
[[nodiscard]] AbelianBackendData::Ptr _convert_FT_to_abelian(TensorCPtr tensor,
                                                             AbelianBackend::Ptr backend,
                                                             Dtype dtype,
                                                             std::string device);

/// Common steps to prepare a SymmetricTensor before a decomposition.
///
/// Returns `(tensor, new_co_domain, combine_codomain, combine_domain)`.
/// `new_co_domain` is a one-factor `TensorProduct` (callers use it as a co-domain).
///
/// @param tensor Tensor to prepare.
/// @param new_leg_dual Whether the new leg introduced by the decomposition is dual.
/// @returns Prepared tensor, new co-domain, and whether codomain/domain were combined.
[[nodiscard]] std::tuple<SymmetricTensorPtr, TensorProduct::Ptr, bool, bool>
_decomposition_prepare(TensorCPtr tensor, bool new_leg_dual);

/// Parse labels for two-leg decompositions (QR, LQ, eigh, …).
///
/// @param new_labels Label sequence for the two new legs.
/// @returns Pair of leg labels for the decomposition factors.
[[nodiscard]] std::pair<std::string, std::string> _decomposition_labels(
  LegLabelList const& new_labels);

/// Parse labels for `svd`. Outer `nullopt` means all-unlabelled (empty strings).
///
/// @param new_labels Optional label sequence; `nullopt` means all unlabeled (`""`).
/// @returns Four leg labels for the SVD factors.
[[nodiscard]] std::tuple<std::string, std::string, std::string, std::string> _svd_new_labels(
  std::optional<LegLabelList> new_labels);

} // namespace lasyten
