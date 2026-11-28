#pragma once

#include <lasyten/lasyten.h>
#include <lasyten/symmetries/spaces.h>
#include <lasyten/tensors/forward_declare.h>
#include <lasyten/tensors/ops_algebra.h>

#include <optional>
#include <string>
#include <utility>
#include <variant>
#include <vector>

namespace lasyten {

/// Single bool (broadcast) or per-leg optional bool (`nullopt` = unspecified).
using BendRight = std::variant<bool, std::vector<std::optional<bool>>>;
/// Single bool (broadcast) or one flag per combined group.
using PipeDualities = std::variant<bool, std::vector<bool>>;

/// Move legs between codomain and domain without changing the order of `tensor.legs`.
///
/// Note that legs are always bent to the right side of the tensor.
/// For more general manipulations involving bends to the left side, use `permute_legs`.
///
/// Graphically::
///
///     |        │   ╭───────────╮
///     |        │   │   ╭───╮   │    ==    bend_legs(T, num_domain_legs=1)
///     |       ┏┷━━━┷━━━┷┓  │   │
///     |       ┃    T    ┃  │   │    ==    bend_legs(T, num_codomain_legs=5)
///     |       ┗┯━━━┯━━━┯┛  │   │
///     |        │   │   │   │   │
///
/// or::
///
///     |        │   │   │   │
///     |       ┏┷━━━┷━━━┷┓  │
///     |       ┃    T    ┃  │        ==    bend_legs(T, num_domain_legs=4)
///     |       ┗┯━━━┯━━━┯┛  │
///     |        │   │   ╰───╯        ==    bend_legs(T, num_codomain_legs=2)
///
/// @param tensor The tensor to modify.
/// @param num_codomain_legs,num_domain_legs Desired number of legs in the (co)domain after
///     bending. Only one is required; `nullopt` means unspecified.
/// @returns The tensor with bent legs.
///
/// @see permute_legs
[[nodiscard]] TensorPtr bend_legs(TensorCPtr tensor,
                                  std::optional<int64> num_codomain_legs = std::nullopt,
                                  std::optional<int64> num_domain_legs = std::nullopt);

/// Check if two tensors have the same legs.
///
/// If there are matching labels in mismatched positions (which indicates that the leg order
/// is mixed up by accident), the error message is amended accordingly on mismatched legs.
/// If the legs still match regardless, a warning is issued.
///
/// @param t1,t2 The tensors to compare.
void check_same_legs(TensorCPtr t1, TensorCPtr t2);

/// Combine (multiple) groups of legs, each to a `LegPipe`.
///
/// If the legs to be combined are contiguous to begin with (and ordered within each group),
/// the combine is just a grouping of the legs::
///
///     |       │   │          ║    │
///     |       │   │   ╭───┬──╨╮   │
///     |      11  10   9   8   7   6
///     |      ┏┷━━━┷━━━┷━━━┷━━━┷━━━┷┓
///     |      ┃          T          ┃    ==   combine_legs(T, [0, 1, 2], [4, 5], [7, 8, 9])
///     |      ┗┯━━━┯━━━┯━━━┯━━━┯━━━┯┛
///     |       0   1   2   3   4   5
///     |       ╰╥──┴───╯   │   ╰╥──╯
///     |        ║          │    ║
///
/// Note that the conventional leg order in the domain goes right to left, such that the first
/// element in the group, `7`, is the *right*-most leg in the product, but we still have
/// `result.domain[2] == LegPipe([T.domain[2], T.domain[3], T.domain[4]])` in left-to-right
/// order.
/// This is needed to make `combine_legs` cooperate seamlessly with `bend_legs`,
/// i.e. you get the same result if you bend legs 6-9 to the codomain first and combine
/// `[7, 8, 9]` there or if you combine them in the domain and then bend leg 6 and the newly
/// combined leg.
/// Another way to see this is that we perform the product of spaces in the `T.legs` first,
/// and then take the dual if we need the combined leg in the domain::
///
///     result.domain[2] == result.legs[4].dual
///                      == LegPipe([T.legs[7], T.legs[8], T.legs[9]]).dual
///                      == LegPipe([T.domain[4].dual, T.domain[3].dual, T.domain[2].dual]).dual
///                      == LegPipe([T.domain[2], T.domain[3], T.domain[4]])
///
/// In the general case, the legs are permuted first, to match that leg order.
/// The combined leg takes the position of the first of its original legs on the tensor.
/// If the symmetry does not have symmetric braids, the `levels` are required to specify the
/// chirality of the braids, like in `permute_legs`. For example::
///
///     |       │           │          ║
///     |       │           │     ╭──┬─╨╮
///     |       │     ╭─────│─────│──╯  │     ╭───╮
///     |      11    10     9     8     7     6   │
///     |      ┏┷━━━━━┷━━━━━┷━━━━━┷━━━━━┷━━━━━┷┓  │
///     |      ┃               T               ┃  │    ==   combine_legs(T, [2, 6, 0], [7, 10, 8])
///     |      ┗┯━━━━━┯━━━━━┯━━━━━┯━━━━━┯━━━━━┯┛  │
///     |       0     1     2     3     4     5   │
///     |       ╰─────│─────│───╮ │     │     │   │
///     |             │     │ ╭─│─│─────│─────│───╯
///     |             │     ╰╥┴─╯ │     │     │
///     |             │      ║    │     │     │
///
/// @param tensor The tensor whose legs should be combined.
/// @param which_legs One or more groups of legs to combine (each by index or label).
/// @param pipe_dualities Optional `LegPipe::is_dual` for each resulting pipe. This is an
///     arbitrary choice per pipe. Pipes are formed such that
///     `result.legs[pipe_idx].is_dual == pipe_dualities[i]`. Defaults to all `false` when
///     a single bool is given; `nullopt` means use the default.
/// @param pipes For each `group = which_legs[i]`, an optional resulting pipe to avoid
///     recomputation. If grouping to the codomain (`group[0] < tensor.num_codomain_legs`),
///     expect `LegPipe` of the corresponding codomain legs; otherwise expect the domain
///     pipe in reverse order. `nullptr` entries are fillers when only some pipes are known.
/// @param levels Ignored if the symmetry has symmetric braids. Otherwise specify braid
///     chirality like in `permute_legs`. `nullopt` means unspecified.
/// @returns A tensor with combined legs, such that up to a `permute_legs`, the original
///     tensor can be recovered with `split_legs`. In both the domain and the codomain, the
///     first leg of each group is replaced by the entire group, in order of appearance in
///     `which_legs`. This may move legs between (co)domain. Non-participating legs keep
///     their relative order. Then each group is replaced by the appropriate product space.
///
/// @see planar_combine_legs
[[nodiscard]] TensorPtr combine_legs(TensorCPtr tensor,
                                     std::vector<std::vector<LegRef>> which_legs,
                                     std::optional<PipeDualities> pipe_dualities = std::nullopt,
                                     std::optional<std::vector<Leg::Ptr>> pipes = std::nullopt,
                                     std::optional<LevelsSpec> levels = std::nullopt);

/// Combine legs of a tensor into two combined `LegPipe`s (matrix form).
///
/// The resulting tensor can be interpreted as a matrix, i.e. has two legs.
///
/// Graphically::
///
///     |                    ║
///     |             ╭─┬─┬──╨────╮
///     |             │ ╰─│─────╮ │
///     |         ╭───│───│─────│─│─╮
///     |         6   5   4     │ │ │
///     |      ┏━━┷━━━┷━━━┷━━┓  │ │ │
///     |      ┃      T      ┃  │ │ │   =    combine_to_matrix(T, [1, 3, -1], [5, 2, 4, 0])
///     |      ┗┯━━━┯━━━┯━━━┯┛  │ │ │
///     |       0   1   2   3   │ │ │
///     |       │   │   ╰───│───╯ │ │
///     |       ╰───│───────│─────╯ │
///     |           ╰──╥────┴───────╯
///     |              ║
///
/// @param tensor The tensor to act on.
/// @param codomain,domain Two groups of legs (by index or label). Together they must
///     comprise all legs of `tensor` without duplicates. Only one of the two is required;
///     the other is determined as the remaining legs. `nullopt` means unspecified.
/// @param levels Ignored if the symmetry has symmetric braids. Otherwise specify braid
///     chirality like in `permute_legs`. `nullopt` means unspecified.
/// @returns The tensor with one combined leg in the codomain and one in the domain.
///
/// @see permute_legs, combine_legs
[[nodiscard]] TensorPtr combine_to_matrix(
  TensorCPtr tensor,
  std::optional<std::vector<LegRef>> codomain = std::nullopt,
  std::optional<std::vector<LegRef>> domain = std::nullopt,
  std::optional<LevelsSpec> levels = std::nullopt);

/// Move one leg of a tensor to a specified position.
///
/// Graphically::
///
///     |        │   ╭───│─╯ │
///     |       ┏┷━━━┷━━━┷━━━┷┓
///     |       ┃      T      ┃       ==    move_leg(T, 6, domain_pos=-2)
///     |       ┗┯━━━┯━━━┯━━━┯┛
///     |        │   │   │   │
///
/// or::
///
///     |        │   │   ╭───│───╮
///     |       ┏┷━━━┷━━━┷━━━┷┓  │
///     |       ┃      T      ┃  │    ==    move_leg(T, 5, codomain_pos=1, bend_right=True)
///     |       ┗┯━━━┯━━━┯━━━┯┛  │
///     |        │ ╭─│───│───│───╯
///
/// @param tensor The tensor to act on.
/// @param which_leg Which leg to move, by index or label.
/// @param codomain_pos If given, move the leg to that position of the resulting codomain.
/// @param domain_pos If given, move the leg to that position of the resulting domain.
/// @param levels Ignored if the symmetry has symmetric braids. Otherwise specify braid
///     chirality like in `permute_legs`. `nullopt` means unspecified.
/// @param bend_right If the moving leg should bend to the right of the tensor (as shown
///     above) or to the left. Ignored if the leg does not bend or if the symmetry has
///     symmetric braids. `nullopt` means unspecified.
/// @returns The tensor with the leg moved.
[[nodiscard]] TensorPtr move_leg(TensorCPtr tensor,
                                 LegRef which_leg,
                                 std::optional<int64> codomain_pos = std::nullopt,
                                 std::optional<int64> domain_pos = std::nullopt,
                                 std::optional<LevelsSpec> levels = std::nullopt,
                                 std::optional<BendRight> bend_right = std::nullopt);

/// Replace a `LegPipe` leg by the equivalent `ElementarySpace`.
///
/// Only the leg changes: the new leg has the same sectors and multiplicities as the pipe
/// (and its basis permutation, if any), and the numerical data is unchanged. This is useful
/// if the internal structure of the pipe is no longer needed, e.g. for MPS bonds obtained by
/// combining several virtual legs. If the leg is not a `LegPipe`, the tensor is returned
/// unchanged. Hidden legs are kept, i.e. a `HiddenLegTensor` gives a `HiddenLegTensor`.
///
/// @param tensor The tensor to act on. Must be a `SymmetricTensor`.
/// @param which_leg The leg to flatten.
/// @returns A tensor with the same data, where the leg `which_leg` is an `ElementarySpace`.
///
/// @see combine_legs, split_legs
[[nodiscard]] TensorPtr flatten_pipe_leg(TensorCPtr tensor, LegRef which_leg);

/// Slide one hidden leg from `A` onto `B` along a contracted public pair of legs.
///
/// The public contraction `tdot(A, B, axis_A, axis_B)` is preserved, including the
/// open hidden leg on the result::
///
///     tdot(A, B, axis_A, axis_B) == tdot(A2, B2, axis_A, axis_B)
///
/// Example (iMPS charge gauging)::
///
///     A[!charge, p, vL, vR], B[p, vL, vR]
///     A2, B2 = move_hidden_leg(A, B, 'vR', 'vL', '!charge', target_codomain_pos=2)
///     # A2[p, vL, vR], B2[p, vL, !charge, vR]
///
/// `A2` is a plain `SymmetricTensor` if no hidden legs remain on `A`, otherwise a
/// `HiddenLegTensor`. `B2` is always a `HiddenLegTensor`.
///
/// The hidden wire is moved by inserting an identity on the hidden space and combining
/// it with the contracted public axes. If the hidden space is one-dimensional (i.e. a single
/// sector with quantum dimension one and multiplicity one, up to rounding of `Space::dim`), the
/// resulting bond pipes are flattened to isomorphic `ElementarySpace`s (a charge-label
/// update). Otherwise the contracted axes remain matching `LegPipe`s that carry the
/// hidden space through the bond.
///
/// Multiple hidden legs on one tensor are allowed (same-plane labels). As for all hidden legs,
/// the moved leg passes behind the public legs (including the bond), and the direction around
/// `A` and `B` is chosen such that it does not cross other hidden legs of `A` or `B`. In
/// particular, the moved leg can be placed next to the existing hidden legs of `B`, e.g.
/// ``target_domain_pos=0`` for the iMPS example above. A target position that changes the
/// cyclic order of the hidden legs requires a genuine braid between them, which raises
/// `BraidChiralityUnspecifiedError` for symmetries with non-symmetric braids. If two
/// tensors both use the same hidden label (e.g. two iMPS sites with ``!charge``), relabel
/// one of them before moving.
///
/// @param A Tensor that currently carries the hidden leg.
/// @param B Tensor that shares a contractible public pair with `A`. Must be a
///     `SymmetricTensor` (including `HiddenLegTensor`). `ChargedTensor`, `DiagonalTensor`,
///     `Identity`, and `Mask` are rejected.
/// @param axis_A,axis_B Public legs to slide along; must be mutual duals (`A.get_leg(axis_A)`
///     equals `B.get_leg(axis_B).dual`).
/// @param hidden_leg_label Hidden label on `A`, including the ``!`` prefix.
/// @param target_codomain_pos If given, place the hidden leg at that position of `B2`'s
///     codomain (same convention as `move_leg`).
/// @param target_domain_pos If given, place the hidden leg at that position of `B2`'s
///     domain. Exactly one of `target_codomain_pos` and `target_domain_pos` is required.
/// @returns `(A2, B2)` as described above.
///
/// @see move_leg, HiddenLegTensor
[[nodiscard]] std::pair<TensorPtr, HiddenLegTensorPtr> move_hidden_leg(
  HiddenLegTensorCPtr A,
  TensorCPtr B,
  LegRef axis_A,
  LegRef axis_B,
  std::string hidden_leg_label,
  std::optional<int64> target_codomain_pos = std::nullopt,
  std::optional<int64> target_domain_pos = std::nullopt);

/// Flip the duality of a bond between two contracted legs (a gauge transformation).
///
/// `A` and `B` share a contractible pair of legs, ``A.get_leg(leg_A) == B.get_leg(leg_B).dual``.
/// The result ``A2, B2`` has the same contraction,
///
///     tdot(A, B, leg_A, leg_B) == tdot(A2, B2, leg_A, leg_B)
///
/// (or the planar version of it, there are no braids involved), but the bond is represented by
/// the *isomorphic* space with opposite ``is_dual``: ``A2.get_leg(leg_A).is_dual`` is
/// ``!A.get_leg(leg_A).is_dual``, and ``B2`` likewise. The sectors of the leg are not changed,
/// i.e. only the convention how the bond space is written down changes. Each leg stays on its
/// side, i.e. in the codomain or in the domain of its tensor, and all labels are preserved.
///
/// One end of the bond gets the isomorphism @f$ Z_a : a^* \to \bar{a} @f$ (see
/// `BaseSymmetry::Z_iso`), the other its inverse. They cancel, so this is a gauge transformation
/// of the bond, comparable to `move_hidden_leg`; `flip_leg_duality` alone does not cancel and
/// changes the tensor. Bending the leg cannot achieve this, since a bent leg is listed as the
/// same space.
///
/// The typical use case is an MPO that was conjugated tensor by tensor: ``dagger`` dualizes all
/// virtual legs, and this function brings them back to the convention of the original MPO, such
/// that the two can be added or compared.
///
/// Internally, each leg is combined into a one-leg pipe with opposite duality and flattened
/// again, as in `move_hidden_leg`. `LegPipe` bond legs are flattened to `ElementarySpace`s; on the
/// `FusionTreeBackend`, nested pipes that are dual raise a `NotImplementedError` (from
/// `flatten_pipe_leg`). A `DirectSumSpace` bond stays a `DirectSumSpace` with the same summands
/// and `summand_labels`, but with each summand flipped (see
/// `ElementarySpace::with_opposite_duality`).
///
/// @note Only the contraction of the pair is guaranteed to be preserved. The public basis order
///     of the flipped leg (its ``basis_perm``) is not, and for non-abelian or anyonic symmetries
///     flipping a single tensor twice differs from the original by the Frobenius-Schur
///     indicators of the bond sectors. Both are irrelevant for virtual bonds.
///
/// @param A,B The tensors, both `SymmetricTensor`s without hidden legs. They may be the same
///     object if `leg_A` and `leg_B` are different legs of it (both ends of the bond on one
///     tensor, e.g. the wrap bond of an infinite MPO with a one-site unit cell). Then both legs
///     are flipped on that tensor, which is returned twice, such that tracing the two legs gives
///     the same result before and after.
/// @param leg_A,leg_B The legs to flip.
/// @returns ``(A2, B2)`` as described above.
///
/// @see flip_leg_duality, move_hidden_leg, flatten_pipe_leg
[[nodiscard]] std::pair<TensorPtr, TensorPtr> flip_bond_duality(TensorCPtr A,
                                                                TensorCPtr B,
                                                                LegRef leg_A,
                                                                LegRef leg_B);

/// Flip the duality (``is_dual``) of a single leg of a tensor.
///
/// Replaces the leg by the isomorphic space with opposite ``is_dual``. Concretely, it applies a
/// specific unitary isomorphism to that leg: the ``Z`` isomorphism @f$ Z_a : a^* \to \bar{a} @f$,
/// block by block on the sectors ``a`` (see `BaseSymmetry::Z_iso`). The sectors are not changed.
/// Labels and the position of the leg (codomain / domain) are preserved.
///
/// @warning This *changes the tensor* by ``Z``, except on a leg with only the trivial sector,
///     where ``Z`` is the identity and only ``is_dual`` changes (e.g. the dummy legs of a finite
///     MPO). Keeping a network consistent is up to the caller: every tensor that shares the leg,
///     explicitly or implicitly (e.g. MPS, MPO and environments on one virtual bond), needs the
///     same flip. For one explicitly contracted pair, `flip_bond_duality` does this as a gauge
///     transformation. Contractions with the conjugate tensor, such as the norm, are unchanged,
///     since ``Z`` is unitary.
///
/// For a leg that is a `LegPipe`, the result is the flattened `ElementarySpace` with opposite
/// duality. A `DirectSumSpace` leg stays a `DirectSumSpace` with the same summands and
/// `summand_labels`, but with each summand flipped.
///
/// @param tensor A `SymmetricTensor` without hidden legs.
/// @param which_leg The leg to flip.
/// @returns The tensor with the flipped leg.
///
/// @see flip_bond_duality, dagger_factorization
[[nodiscard]] TensorPtr flip_leg_duality(TensorCPtr tensor, LegRef which_leg);

/// Permute the legs of a tensor by braiding legs and bending lines.
///
/// Graphically (note that we ignore the `levels` graphically and do not draw braid
/// chiralities)::
///
///     |             │ ╰─│─────╮ │
///     |         ╭───│───│─────│─│─╮
///     |         6   5   4     │ │ │
///     |      ┏━━┷━━━┷━━━┷━━┓  │ │ │
///     |      ┃      T      ┃  │ │ │   =    permute_legs(T, [1, 3, -1], [5, 2, 4, 0])
///     |      ┗┯━━━┯━━━┯━━━┯┛  │ │ │
///     |       0   1   2   3   │ │ │
///     |       │   │   ╰───│───╯ │ │
///     |       ╰───│───────│─────╯
///
///     |        │ ╭─────────│─╯ │
///     |      ╭─│─│─────╮   │   │
///     |      │ │ │     6   5   4
///     |      │ │ │  ┏━━┷━━━┷━━━┷━━┓
///     |      │ │ │  ┃      T      ┃   =   permute_legs(T, [6, 1, 3], [0, 5, 2, 4],
///     bend_right=False) |      │ │ │  ┗┯━━━┯━━━┯━━━┯┛ |      │ │ │   0   1   2   3 |      │ │
///     ╰───│───│───╯   │ |      │ ╰─────╯   │       │
///
/// @note We expect that there are only two cases where you should do explicit leg
///     permutations: firstly, if you need to specify the `levels` explicitly for an anyonic
///     symmetry; secondly, if you are optimizing for performance and know what you are
///     doing. In most other cases, refer to legs by label and let API functions rearrange
///     legs as needed.
///
/// @warning It is inefficient (especially with the fusion-tree backend) to do a series of
///     leg rearrangements as multiple calls. For performance, do them in a single call.
///
/// @param tensor The tensor to permute.
/// @param codomain,domain Which legs of `tensor` (by position in `tensor.legs` or by label)
///     should end up in the (co)domain of the result. Only one of the two is required; the
///     other is determined as the remaining legs, preserving order in `tensor.legs`.
///     Together they must comprise all legs without duplicates. `nullopt` means unspecified.
/// @param levels If the symmetry has symmetric braiding (e.g. group symmetries or fermions),
///     ignored. For non-symmetric braiding, assigns a level/height to each leg: when two
///     legs cross, the higher level goes over the other. Per-leg entries may be `nullopt`
///     when unspecified.
/// @param bend_right For each leg that bends up or down, whether it bends to the right of
///     the tensor (as shown above) or to the left. Ignored for symmetric braids. For anyonic
///     symmetries an explicit choice is required for all bending legs. A single bool
///     broadcasts to all legs; per-leg optional bools allow `nullopt` placeholders for legs
///     that do not bend.
/// @returns The permuted tensor.
[[nodiscard]] TensorPtr permute_legs(TensorCPtr tensor,
                                     std::optional<std::vector<LegRef>> codomain = std::nullopt,
                                     std::optional<std::vector<LegRef>> domain = std::nullopt,
                                     std::optional<LevelsSpec> levels = std::nullopt,
                                     std::optional<BendRight> bend_right = std::nullopt);

/// Split legs that were previously combined using `combine_legs`.
///
/// Graphically::
///
///     |       │  │    │   │   │   │
///     |       ╰──┴───╥╯   │   ╰──╥╯
///     |      ┏━━━━━━━┷━━━━┷━━━━━━┷━┓
///     |      ┃          T          ┃    ==    split_legs(T, [2, 4, 6])
///     |      ┗┯━━━┯━━━━┯━━━━━━━━━━┯┛
///     |       │   │   ╭╨───┬──╮   │
///     |       │   │   │   │   │   │
///
/// This is the inverse of `combine_legs`, up to a possible `permute_legs`.
///
/// @param tensor The tensor to act on.
/// @param legs Which legs to split. If `nullopt`, all legs that are `LegPipe`s are split.
/// @returns The tensor with the selected pipes split.
[[nodiscard]] TensorPtr split_legs(TensorCPtr tensor,
                                   std::optional<std::vector<LegRef>> legs = std::nullopt);

/// Remove trivial legs.
///
/// A leg counts as trivial according to `Space::is_trivial`, i.e. if it consists of a single
/// copy of the trivial sector.
///
/// @param tensor The tensor to act on.
/// @param legs Which legs to squeeze. Squeezed legs must be trivial. If `nullopt`, all
///     trivial legs are squeezed.
/// @returns The tensor with trivial legs removed.
[[nodiscard]] TensorPtr squeeze_legs(TensorCPtr tensor,
                                     std::optional<std::vector<LegRef>> legs = std::nullopt);

/// Contract one multiplicity of one sector on a leg, as a `HiddenLegTensor`.
///
/// The leftover space is an `ElementarySpace` with that sector and multiplicity 1. It becomes
/// a hidden leg (label ``"!slice"``) of the result. This does not require a droppable symmetry.
///
/// Two ways to name the kept copy:
///
/// - `slice_leg(tensor, leg, idx)` — public-basis index. Requires `symmetry.can_be_dropped`.
///   Uses `ElementarySpace::parse_index`, then the reduced multiplicity index of that sector.
/// - `slice_leg(tensor, leg, sector, multiplicity=0)` — always valid. `sector` must appear on
///   that leg; `multiplicity` in `range(leg.sector_multiplicity(sector))`.
///
/// `leg` is an integer index or a label, as for `apply_mask`.
/// Slicing a `ChargedTensor` or `HiddenLegTensor` is not supported.
///
/// After `E, V = eigh(H)`::
///
///     i0 = E.argmin()
///     psi = V.slice_leg('a', i0)
///
/// @param tensor The tensor to slice.
/// @param leg Leg index or label to slice.
/// @param idx Public-basis index into that leg (droppable symmetries only).
/// @returns A `HiddenLegTensor` whose hidden leg is the leftover one-sector space.
[[nodiscard]] HiddenLegTensorPtr slice_leg(TensorCPtr tensor, LegRef leg, int64 idx);

/// Contract one multiplicity of one sector on a leg, as a `HiddenLegTensor`.
///
/// The leftover space is an `ElementarySpace` with that sector and multiplicity 1. It becomes
/// a hidden leg (label ``"!slice"``) of the result. This does not require a droppable symmetry.
///
/// Two ways to name the kept copy:
///
/// - `slice_leg(tensor, leg, idx)` — public-basis index. Requires `symmetry.can_be_dropped`.
///   Uses `ElementarySpace::parse_index`, then the reduced multiplicity index of that sector.
/// - `slice_leg(tensor, leg, sector, multiplicity=0)` — always valid. `sector` must appear on
///   that leg; `multiplicity` in `range(leg.sector_multiplicity(sector))`.
///
/// `leg` is an integer index or a label, as for `apply_mask`.
/// Slicing a `ChargedTensor` or `HiddenLegTensor` is not supported.
///
/// After `E, V = eigh(H)`::
///
///     i0 = E.argmin()
///     psi = V.slice_leg('a', i0)
///
/// @param tensor The tensor to slice.
/// @param leg Leg index or label to slice.
/// @param sector Sector to keep on that leg.
/// @param multiplicity Multiplicity index of `sector` (default `0`).
/// @returns A `HiddenLegTensor` whose hidden leg is the leftover one-sector space.
[[nodiscard]] HiddenLegTensorPtr slice_leg(TensorCPtr tensor,
                                           LegRef leg,
                                           Sector const& sector,
                                           int64 multiplicity = 0);

} // namespace lasyten
