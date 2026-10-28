#pragma once

#include <lasyten/block_backend/block_backend.h>
#include <lasyten/lasyten.h>
#include <lasyten/tensors/decompositions.h>
#include <lasyten/tensors/diagonal_tensor.h>
#include <lasyten/tensors/labels.h>
#include <lasyten/tensors/ops_algebra.h>
#include <lasyten/tensors/ops_legs.h>
#include <lasyten/tensors/sparse.h>
#include <lasyten/tensors/tensor.h>
#include <lasyten/tools/cost_polynomials.h>

#include <map>
#include <memory>
#include <optional>
#include <string>
#include <tuple>
#include <utility>
#include <variant>
#include <vector>

namespace lasyten {

/// One planar-diagram instruction: contraction ``(t1, l1, t2, l2)`` or open leg
/// ``(t1, l1, nullopt, new_label)``.
using DiagramInstruction =
  std::tuple<std::string, std::string, std::optional<std::string>, std::string>;

/// Result of a planar contraction that may fully reduce to a scalar.
using PlanarResult = std::variant<TensorPtr, BlockBackend::Scalar>;

/// Product of zero or more polynomials (empty product is the empty polynomial).
[[nodiscard]] BigOPolynomial product_of(std::vector<BigOPolynomial> const& polys);

/// Strip whitespace and check that `name` is valid as a tensor name or leg label.
[[nodiscard]] std::string _as_valid_name(std::string name);

/// Raise if `actual` is not a cyclic permutation of `expected`.
void _assert_cyclic_labels(std::string const& name,
                           std::vector<std::string> const& expected,
                           std::vector<std::string> const& actual);

/// Split up text that appears as the `tensors` input to `PlanarDiagram`.
[[nodiscard]] std::vector<std::pair<std::string, std::vector<std::string>>> _split_tensor_text(
  std::string const& text);

/// Parse a planar bipartition of legs into two subsets.
[[nodiscard]] std::pair<std::vector<int64>, std::vector<int64>> parse_leg_bipartition(
  std::vector<int64> const& legs,
  int64 num_legs);

/// Placeholder for a tensor used to define `PlanarDiagram` s.
///
/// Attributes:
///
/// labels : list of str
///     The labels of the tensor (up to cyclic permutation). This means that as long as we go
///     clockwise around the shape, any starting point can be chosen for the labels.
/// dims : list of (str | None)
///     For each of the legs, an optional symbol to represent its dimension.
/// cost_to_make : `BigOPolynomial`
///     Algorithmic cost of creating the tensor.
class TensorPlaceholder
{
  public:
    LegLabels labels;
    int64 num_legs = 0;
    std::vector<BigOPolynomial> dims;
    BigOPolynomial cost_to_make;

    explicit TensorPlaceholder(std::vector<std::string> labels,
                               std::vector<BigOPolynomial> dims = {},
                               BigOPolynomial cost_to_make = BigOPolynomial());

    [[nodiscard]] TensorPlaceholder copy(bool deep = true) const;
    [[nodiscard]] std::vector<std::string> string_labels() const;
    [[nodiscard]] std::string __repr__() const;
};

/// Named tensors in a `PlanarDiagram` (insertion order is `tensor_names`).
using TensorPlaceholderMap = std::map<std::string, TensorPlaceholder>;

/// Placeholder labels that `tensor` must match, up to cyclic permutation.
[[nodiscard]] std::vector<std::string> _expected_labels(std::vector<std::string> const& ph_labels,
                                                        TensorCPtr tensor);
[[nodiscard]] std::vector<std::string> _expected_labels(std::vector<std::string> const& ph_labels,
                                                        TensorPlaceholder const& tensor);

/// Node in a `ContractionTree`.
///
/// Represents a single tensor contraction in a contraction tree, where the left and
/// right child (if not `None`) may correspond a single tensor or contractions of
/// multiple tensors. The result of the represented tensor contraction can be part of
/// subsequent contractions represented by the parent (if not `None`).
/// If both children are `None`, the node only represents a tensor.
///
/// A node must not be trivial, that is, it must either represent a tensor contraction
/// (i.e., have a left and right child; value is optional) or have a value different
/// from `None` when representing a tensor.
///
/// Graphically::
///
///     |            parent
///     |              │                   parent━value
///     |            value            ==            ┣━left_child
///     |       ┏━━━━━━┷━━━━━━┓                     ┗━right_child
///     |   left_child   right_child
///
/// The RHS above corresponds to the graphic representation of the node in the full
/// contraction tree, as constructed in `show_whole_tree`.
///
/// @param parent Node representing a subsequent tensor contraction for which the result of the
/// contraction represented by `self` is a left or right child.
/// @param left_child Represents the left tensor to be contracted. May itself be the result of a
/// tensor contraction. May be `None` if `self` represents a single tensor rather than a tensor
/// contraction. In such a case, `right_child` must also be `None`.
/// @param right_child Represents the right tensor to be contracted. May itself be the result of a
/// tensor contraction. May be `None` if `self` represents a single tensor rather than a tensor
/// contraction. In such a case, `left_child` must also be `None`.
/// @param value Value describing the contraction tree node.
class ContractionTreeNode : public std::enable_shared_from_this<ContractionTreeNode>
{
  public:
    using Ptr = std::shared_ptr<ContractionTreeNode>;

    std::weak_ptr<ContractionTreeNode> parent;
    Ptr left_child;
    Ptr right_child;
    std::optional<std::string> value;

    ContractionTreeNode(Ptr parent,
                        Ptr left_child,
                        Ptr right_child,
                        std::optional<std::string> value);

    void test_sanity() const;
    [[nodiscard]] bool is_leaf() const;
    [[nodiscard]] Ptr copy(Ptr parent = nullptr) const;
    [[nodiscard]] std::pair<std::vector<std::string>, int64> get_leaves() const;
    std::pair<std::optional<std::string>, std::optional<std::string>> remove_children();
    std::tuple<std::optional<std::string>, std::string, std::string, std::string>
    pop_contraction();
    [[nodiscard]] std::vector<std::string> _str_lines(std::string const& prefix_0 = "",
                                                      std::string const& prefix = "") const;
    [[nodiscard]] std::string show_whole_tree() const;
};

/// Representation of the contraction order in a `PlanarDiagram` as a tree structure.
///
/// The leaf nodes represent the tensor names in a diagram and the tree structure indicates an
/// order of pairwise contractions.
///
/// The values of non-leaf nodes currently have no meaning and are always set to ``None``,
/// but may cary extra information about leg handling during a pairwise contraction in the future.
///
/// @param root Node representing the root of the contraction tree, i.e., the upper-most node that
/// does not have a parent.
class ContractionTree
{
  public:
    ContractionTreeNode::Ptr root;

    explicit ContractionTree(ContractionTreeNode::Ptr root);

    void test_sanity() const;
    [[nodiscard]] std::vector<std::string> leaves() const;
    [[nodiscard]] int64 num_leaves() const;
    [[nodiscard]] int64 num_nodes() const;
    [[nodiscard]] int64 num_inner_nodes() const;

    [[nodiscard]] static ContractionTree from_contraction_order(
      std::vector<std::pair<std::string, std::string>> const& order);
    [[nodiscard]] static ContractionTree from_single_node(std::string const& node);
    [[nodiscard]] ContractionTree copy() const;
    /// Fuse two trees. In-place on both trees.
    ///
    /// Graphically::
    ///
    ///     |                                        value
    ///     |                                       /     \
    ///     |       a             b                a        b
    ///     |      / \     ,     / \      ->      / \      / \
    ///     |    ... ...       ... ...          ... ...  ... ...
    ///
    /// @param other The contraction tree that will become the right child of the resulting
    /// combined contraction tree; `self` becomes the left child.
    /// @param value The value of the new root node at which `self` and `other` are fused.
    [[nodiscard]] ContractionTree fuse(ContractionTree& other,
                                       std::optional<std::string> value = std::nullopt);
    /// Replace a bottom node (where both children are leaves) with a single leaf, in-place.
    ///
    /// Graphically::
    ///
    ///     |    ...              ...
    ///     |     |                |
    ///     |     X       ->    new_value
    ///     |    / \
    ///     |   a   b
    ///
    /// @returns X : str or None The value at the non-leaf node that is replaced a, b : str or None
    /// The values of the leaf nodes that are removed new_value : str The value of the new leaf,
    /// conventionally ``'a @ b'``.
    std::tuple<std::optional<std::string>, std::string, std::string, std::string>
    pop_contraction();
    [[nodiscard]] std::string str() const;
};

/// Abstract representation for the contraction of multiple tensors without any braids.
///
/// The tensors in a planar diagram are represented using placeholders that have the same leg
/// labels as the actual tensors for which the diagram is to be evaluated. The tensor contractions
/// of the planar diagram, as well as its open (non-contracted) legs, are specified using only the
/// leg labels of the tensors / placeholders. The full diagram must be both planar and connected,
/// meaning that the tensor legs must not braid with each other and that the full diagram is
/// contractible to a single tensor. Disconnected tensors can nevertheless be considered by
/// combining them with another tensor using `outer` before adding them to the diagram.
/// When specifying the leg labels of the tensors, their order must coincide with the conventional
/// (counter-clockwise) leg ordering of tensors. The leg labels may however be cyclically permuted
/// since such permutations are planar. It is further irrelevant how the legs are distributed among
/// the codomain and domain (as long as the order is correct).
///
/// The contractions specified by a planar diagram can be performed for a concrete set of tensors
/// by using `evaluate` or directly calling the planar diagram instance with the
/// corresponding tensors as argument. The result is only specified up to cyclic leg permutations.
///
/// In general, optimizing the contraction order of the tensors is expensive and should be done
/// once during development and then hard-coded. Alternatively, a greedy optimization can be run
/// during the instantiation. The intended use case is to create instances of planar diagrams on
/// module level, such that the instantiation happens at import time.
///
/// It is possible to use a planar diagram for creating a new one by adding or removing a tensor,
/// see `add_tensor` and `remove_tensor`, respectively.
///
/// It is possible to create planar diagrams that contract `ChargedTensor` and
/// `HiddenLegTensor`. Placeholders list **public** leg labels only (the same labels
/// returned by `ChargedTensor.labels` or the non-hidden labels of a `HiddenLegTensor`).
/// A lone `'!'` or any `'!'`-prefixed label is not a valid placeholder label; hidden legs
/// are not part of the planar diagram specification.
///
/// During `evaluate`, `ChargedTensor` inputs are used as-is; contractions via `tdot` /
/// `compose` handle charged states. `HiddenLegTensor` inputs are also kept as-is;
/// dual hidden labels (``!a`` with ``!a*``) are contracted implicitly by `tdot`.
///
/// @param tensors Specifies the tensors in the planar diagram, each with leg labels and a unique
/// name. Syntax for string input: a comma (`,`) separated list of entries, each for one tensor.
/// The entry for a tensor is its name, followed by comma separated leg labels enclosed in
/// brackets. Example: ``'theta[vL, p0, p1, vR], U[p0, p1, p1*, p0*]'``. The same format as the
/// attribute `tensors` (dict) is accepted as well.
/// @param definition Specifies the planar diagram, i.e., how the `tensors` are contracted. Syntax
/// for string input: a comma (`,`) separated list of instructions, each either a contraction or an
/// open leg. Contractions are of the form ``'{tensorA}:{legA} @ {tensorB}:{legB}'``. Open legs are
/// of the form ``'{tensorA}:{legA} -> {new_label}``. The same format as the attribute `definition`
/// (list of tuples) is accepted as well.
/// @param dims Specifies a symbol for the dimension of each leg, used to show or optimize the
/// contraction cost in terms of a `BigOPolynomial`. A dictionary with pairs ``{dim: labels}``
/// indicating that the legs with ``labels`` have a dimension represented by the symbol ``dim``. If
/// given, *all* labels in the diagram should be assigned to a symbol. Legs with the same label
/// must have the same dimension.
/// @param order Specifies the contraction order, or how to determine it. If ``'greedy'`` (default)
/// or ``'optimal'``, it is optimized via `optimize_order`. If ``'definition'``, it is taken from
/// the order of the `definition`, with minimal extra optimizations (always do traces first and
/// when contracting two tensors, contract all shared legs at once). If a single string, expect a
/// comma separated list of instructions ``'{tensorA} @ {tensorB}'`` which indicate the order of
/// pairwise contractions. If nested tuples of strings, interpret those strings as tensor names,
/// and interpret the bracketing as the order of pairwise contractions, contracting innermost
/// tuples first. The same format as the attribute `order` (``ContractionTree``) is accepted as
/// well.
/// @param allow_multiple_charged_tensors Retained for API compatibility. Multiple
/// `ChargedTensor` placeholders with public-only labels are always allowed at verify time.
///
/// Attributes:
///
/// tensors : {str: TensorPlaceholder}
///     The tensors in the planar diagram, as a dictionary from name to its placeholder, which
///     stores leg labels and dims.
/// definition : list of (str, str, str | None, str)
///     Defines the contractions in the planar diagram.
///     An entry ``(t1, l1, t2, l2)`` indicates to contract leg ``l1`` of ``tensors[t1]`` with
///     leg ``l2`` of ``tensors[t2]``.
///     An entry ``(t1, l1, None, new_l)`` indicates that leg ``l1`` of ``tensors[t1]`` is an open
///     leg of the planar diagram and should have label ``new_l`` in the result.
/// order : ContractionTree
///     Specifies the order for the tensor contractions during `evaluate`.
/// open_legs : list of str
///     The open legs of the planar diagram, up to cyclical permutation.
///     This is such that the result of `evaluate` has these leg labels (up to cycl. perm.).
///     Hidden legs are not included.
/// allow_multiple_charged_tensors : bool
///     Retained for API compatibility; no longer restricts diagram construction.
///
/// Examples:
///
/// 1. For a local two-site MPS tensor `theta` with legs ``vL, p0, p1, vR`` and a two-site operator
/// `op` with legs ``p0, p1, p1*, p0*``, the expectation value of `op` can be expressed as the
/// following planar diagram::
///
///     exp_val_diagram = PlanarDiagram(
///         tensors='theta[vL, p0, p1, vR], theta_hc[vR*, p1*, p0*, vL*], op[p0, p1, p1*, p0*]',
///         definition='theta:p0 @ op:p0*, theta:p1 @ op:p1*, '
///         'theta:vL @ theta_hc:vL*, theta:vR @ theta_hc:vR*, '
///         'op:p0 @ theta_hc:p0*, op:p1 @ theta_hc:p1*',
///         dims=dict(chi=['vR', 'vR*', 'vL', 'vL*'], d=['p0', 'p0*', 'p1', 'p1*']),
///     )
///     exp_val = exp_val_diagram.evaluate(dict(theta=theta, theta_hc=theta.hc, op=op))
///
/// 2. For a local two-site MPS tensor `theta` with legs ``vL, p0, p1, vR`` and a two-site unitary
/// operator `U` with legs ``p0, p1, p1*, p0*`` that is applied to `theta` (as done in TEBD), the
/// updated tensor expressed as the following planar diagram::
///
///     TEBD_diagram = PlanarDiagram(
///         tensors='theta[vL, p0, p1, vR], U[p0, p1, p1*, p0*]',
///         definition='theta:p0 @ U:p0*, theta:p1 @ U:p1*, theta:vL -> vL, theta:vR -> vR, U:p0 ->
///         p0, U:p1 -> p1', dims=dict(chi=['vR', 'vL'], d=['p0', 'p0*', 'p1', 'p1*']),
///     )
///     theta_updated = TEBD_diagram.evaluate(dict(theta=theta, U=U))
///
/// 3. The two examples above (`exp_val_diagram` and `TEBD_diagram`) can be related using
/// `add_tensor` and `remove_tensor` (note the correspondence between `op` and `U`)
/// as::
///
///     TEBD_diagram2 = exp_val_diagram.remove_tensor(
///         name='theta_hc',
///         extra_definition='theta:vL -> vL, theta:vR -> vR, '
///         'op:p0 -> p0, op:p1 -> p1',
///     )
///     theta_updated2 = TEBD_diagram2.evaluate(dict(theta=theta, op=U))
///     assert planar_almost_equal(theta_updated, theta_updated2)
///
///     exp_val_diagram2 = TEBD_diagram.add_tensor(
///         tensor='theta_hc[vR*, p1*, p0*, vL*]'
///         extra_definition='theta:vL @ theta_hc:vL*, theta:vR @ theta_hc:vR*, '
///         'U:p0 @ theta_hc:p0*, U:p1 @ theta_hc:p1*',
///         extra_dims='dict(chi=['vR*', 'vL*'], d=['p0*', 'p1*'])'
///     )
///     exp_val2 = exp_val_diagram2.evaluate(dict(theta=theta, theta_hc=theta.hc, U=op))
///     assert np.isclose(exp_val, exp_val2)  # number, not a tensor
class PlanarDiagram
{
  public:
    TensorPlaceholderMap tensors;
    std::vector<std::string> tensor_names_;
    std::vector<DiagramInstruction> definition;
    ContractionTree order;
    std::vector<std::string> open_legs;
    BigOPolynomial contraction_cost;
    bool allow_multiple_charged_tensors = false;

    PlanarDiagram(TensorPlaceholderMap tensors,
                  std::vector<std::string> tensor_names,
                  std::vector<DiagramInstruction> definition,
                  ContractionTree order,
                  bool allow_multiple_charged_tensors = false);

    PlanarDiagram(TensorPlaceholderMap tensors,
                  std::vector<std::string> tensor_names,
                  std::vector<DiagramInstruction> definition,
                  std::string const& order,
                  bool allow_multiple_charged_tensors = false);

    [[nodiscard]] std::vector<std::string> const& tensor_names() const { return tensor_names_; }

    /// Create a new planar diagram with an additional tensor.
    ///
    /// The new planar diagram arises from the old one by adding a single tensor and contracting
    /// (some of) its legs with open legs of the old planar diagram. It is in particular not
    /// possible to change tensor contractions involving two tensors of the old planar diagram.
    ///
    /// TODO should we allow to reference the existing diagram as a whole, instead of its
    ///      individual tensors?
    ///
    /// `extra_dims` is the same as the parameter to `PlanarDiagram`, but applies only to the new
    /// tensor.
    ///
    /// @param extra_tensors Same as the parameter to `PlanarDiagram`, but expect only the
    /// additional tensors to be added to the diagram.
    /// @param extra_definition Same as the parameter to `PlanarDiagram`. Should define for each
    ///     leg of the new tensor whether it is an open leg or contracted with another leg.
    ///     The new `definition` is given by this extra definition together with the old
    ///     definition, except for entries that correspond to legs that were open in the original
    ///     diagram and are now contracted with the new tensor.
    /// @param order Same as the parameter to `PlanarDiagram`, applies to the entire new diagram.
    [[nodiscard]] PlanarDiagram add_tensor(TensorPlaceholderMap extra_tensors,
                                           std::vector<DiagramInstruction> extra_definition,
                                           std::string const& order = "definition") const;

    /// Do the contractions defined by the planar diagram for given concrete `tensors`.
    ///
    /// @param tensors A dict mapping each tensor name (as declared in the `tensors` argument of
    ///     the `PlanarDiagram` constructor) to the concrete `Tensor` that should take its place.
    ///     A positional sequence (e.g. a plain `list`) of tensors is *not* accepted; the mapping
    ///     must be keyed by name.
    [[nodiscard]] PlanarResult evaluate(std::map<std::string, TensorPtr> tensors) const;

    /// Symbolic variant of `evaluate`, taking `TensorPlaceholder`s keyed by tensor name instead
    /// of concrete `Tensor`s.
    [[nodiscard]] TensorPlaceholder evaluate(
      std::map<std::string, TensorPlaceholder> tensors) const;

    /// Find the optimal contraction order for the given planar diagram.
    ///
    /// TODO make it easy to print what you need to hard-code.
    /// TODO allow relations like ``d < w < chi``, or ``d^2 < chi`` to simplify the polynomials.
    /// TODO support cost as polynomials or with concrete numbers
    [[nodiscard]] ContractionTree optimize_order(std::string const& strategy) const;

    [[nodiscard]] static std::vector<DiagramInstruction> parse_definition(
      std::string const& definition);
    [[nodiscard]] static std::vector<DiagramInstruction> parse_definition(
      std::vector<DiagramInstruction> definition);

    [[nodiscard]] ContractionTree parse_order(std::string const& order) const;
    [[nodiscard]] ContractionTree parse_order(ContractionTree const& order) const;

    [[nodiscard]] static TensorPlaceholderMap parse_tensors(
      std::string const& tensors,
      std::optional<std::map<std::string, std::vector<std::string>>> const& dims = std::nullopt,
      std::vector<std::string>* name_order = nullptr);
    [[nodiscard]] static TensorPlaceholderMap parse_tensors(
      TensorPlaceholderMap tensors,
      std::optional<std::map<std::string, std::vector<std::string>>> const& dims = std::nullopt,
      std::vector<std::string>* name_order = nullptr);

    /// Create a new planar diagram by removing one tensor.
    ///
    /// The new planar diagram arises from the old one by removing a single tensor and leaving the
    /// legs that were previously contracted with this tensor open. It is in particular not
    /// possible to change any tensor contractions in the planar diagram.
    ///
    /// @param name The name of the tensor to be removed.
    /// @param extra_definition Extra instructions to be added to the `definition`. Expected to
    /// only contain instructions for the legs that were contracted with `name` in the old planar
    /// diagram and are now open legs. Same format as the `definition` parameter to
    /// `PlanarDiagram`.
    /// @param order Same as the parameter to `PlanarDiagram`, applies to the entire new diagram.
    [[nodiscard]] PlanarDiagram remove_tensor(
      std::string const& name,
      std::vector<DiagramInstruction> extra_definition = {},
      std::string const& order = "greedy") const;

    /// Verify the definition of the planar diagram. Returns the `open_legs`.
    ///
    /// @returns open_legs : list of str The leg labels of a result of `evaluate`. cost :
    /// BigOPolynomial The cost to contract the diagram, as a polynomial in terms of the dims.
    [[nodiscard]] std::pair<std::vector<std::string>, BigOPolynomial> verify_diagram();

    static std::map<std::string, PlanarResult>& _do_contractions(
      std::map<std::string, PlanarResult>& tensors,
      std::vector<std::tuple<std::string, std::string, std::string, std::string>> contractions,
      ContractionTree order);
    static std::map<std::string, TensorPlaceholder>& _do_contractions(
      std::map<std::string, TensorPlaceholder>& tensors,
      std::vector<std::tuple<std::string, std::string, std::string, std::string>> contractions,
      ContractionTree order);

    static void _do_traces(
      std::map<std::string, PlanarResult>& tensors,
      std::vector<std::tuple<std::string, std::string, std::string>> const& traces);
    static void _do_traces(
      std::map<std::string, TensorPlaceholder>& tensors,
      std::vector<std::tuple<std::string, std::string, std::string>> const& traces);

    [[nodiscard]] static PlanarResult _extract_result(
      std::map<std::string, PlanarResult> const& tensors,
      std::vector<std::pair<std::string, std::string>> const& open_legs);
    [[nodiscard]] static TensorPlaceholder _extract_result(
      std::map<std::string, TensorPlaceholder> const& tensors,
      std::vector<std::pair<std::string, std::string>> const& open_legs);

    [[nodiscard]] static DiagramInstruction _parse_contract_instruction(std::string const& i);
    [[nodiscard]] static DiagramInstruction _parse_open_leg_instruction(std::string const& i);
    [[nodiscard]] std::optional<int64> _find_open_leg_definition(std::string const& name,
                                                                 std::string const& leg) const;
};

/// Base class for `LinearOperator`\ s defined in terms of `PlanarDiagram`\ s.
///
/// @warning Instantiating a `PlanarDiagram` may be expensive if the order is optimized.
///     Make sure to either hard-code the order, or make the planar diagram instance as early as
///     possible, e.g., as a *class* variable of the parent class instead of during its
///     construction.
///
/// @param op_diagram The diagram that defines the operator (without acting on a vector).
/// @param matvec_diagram The diagram that defines the action of the operator on a vector. Must
/// have the same tensor names as the `op_diagram` in addition to a single tensor with `vec_name`.
/// @param op_tensors The concrete tensors that define the operator, see `op_diagram`.
/// @param vec_name The name of the "vector", i.e., the tensor that the linear operator acts on in
/// the `matvec_diagram`.
class PlanarLinearOperator : public LinearOperator
{
  public:
    using Ptr = std::shared_ptr<PlanarLinearOperator>;

    PlanarDiagram op_diagram;
    PlanarDiagram matvec_diagram;
    std::map<std::string, TensorPtr> op_tensors;
    std::string vec_name;

    PlanarLinearOperator(PlanarDiagram const& op_diagram,
                         PlanarDiagram const& matvec_diagram,
                         std::map<std::string, TensorPtr> op_tensors,
                         std::string vec_name);

    [[nodiscard]] VectorLike::Ptr matvec(VectorLike::CPtr vec) override;
    [[nodiscard]] TensorPtr to_tensor(TensorBackend::Ptr backend = nullptr) override;
};

/// Factorize a tensor into left and right parts.
///
/// Graphically, here with ``codomain_cut=3, domain_cut=1``::
///
///     |      │   │   │               │           │   │             │   ╭──────╮    │   │
///     |   ┏━━┷━━━┷━━━┷━━┓         ┏━━┷━━━━━━┓   ┏┷━━━┷┓         ┏━━┷━━━┷━━┓   │   ┏┷━━━┷┓
///     |   ┃   tensor    ┃    =    ┃    A    ┠───┨  B  ┃   :=    ┃    A    ┃   │   ┃  B  ┃
///     |   ┗┯━━━┯━━━┯━━━┯┛         ┗┯━━━┯━━━┯┛   ┗━━━━┯┛         ┗┯━━━┯━━━┯┛   │   ┗┯━━━┯┛
///     |    │   │   │   │           │   │   │         │           │   │   │    ╰────╯   │
///
/// @param tensor The tensor to factorize
/// @param codomain_cut The first `codomain_cut` legs from the codomain end up in the codomain of
/// `A`, the rest of the codomain ends up in the codomain of `B`.
/// @param domain_cut The first `domain_cut` legs from the domain end up in the domain of `A`, the
/// rest of the domain ends up in the domain of `B`.
/// @param new_labels The labels for the new legs. Two entries ``[a, b]`` result in ``A.labels[-1 -
/// domain_cut] == a`` and ``B.labels[0] == b`` and a single entry ``a`` is equivalent to ``[a,
/// a*]``.
/// @param cutoff_singular_values If ``None`` (default), we factorize using `qr` without
/// truncation. If given, we use a truncated SVD and truncate by discarding singular values below
/// this threshold.
/// @returns A, B: Tensor A factorization of the `tensor`, such that ``tdot(A, B, -1 - domain_cut,
/// 1)`` reproduces the `tensor`, up to bending and possibly up to truncation if
/// `cutoff_singular_values` is given.
///
/// Notes:
///
/// This is achieved by bending legs such that we can do the factorization as a QR or SVD,
/// then bend back, that is for the example case depicted above::
///
///     |                                             │    │   │   ╭────╮         │   │   │
///     |             │           │   │    ╭──╮       │ ┏━━┷━━━┷━━━┷━━┓ │         │  ┏┷━━━┷┓
///     |             │  ╭────╮   │   │    │  │       │ ┃      B'     ┃ │         │  ┃  B  ┃
///     |             │  │ ┏━━┷━━━┷━━━┷━━┓ │  │       │ ┗━━━━━━┯━━━━━━┛ │         │  ┗┯━━━┯┛
///     |   LHS   =   │  │ ┃   tensor    ┃ │  │   =   │        │        │   =     │   │   │   = RHS
///     |             │  │ ┗┯━━━┯━━━┯━━━┯┛ │  │       │ ┏━━━━━━┷━━━━━━┓ │      ┏━━┷━━━┷━━┓│
///     |             │  │  │   │   │   ╰──╯  │       │ ┃      A'     ┃ │      ┃    A    ┃│
///     |             ╰──╯  │   │   │         │       │ ┗┯━━━┯━━━┯━━━┯┛ │      ┗┯━━━┯━━━┯┛│
///     |                                             ╰──╯   │   │   │  │       │   │   │ │
///
/// Note how we bend some legs to the left, to avoid any braids, such that the operation does not
/// need to specify any braid chiralities.
[[nodiscard]] std::tuple<TensorPtr, TensorPtr> horizontal_factorization(
  TensorCPtr tensor,
  int64 codomain_cut,
  int64 domain_cut,
  std::optional<LegLabelList> new_labels = std::nullopt,
  std::optional<float64> cutoff_singular_values = std::nullopt);

/// Checks if two tensors are equal up to numerical tolerance and planar permutation.
///
/// We first permute the legs of `tensor_1` to the configuration of `tensor_2` and then
/// compare the blocks, i.e., the free parameters of the tensors.
/// The tensors count as almost equal if all block entries, i.e., all their free parameters
/// individually fulfill ``abs(a1 - a2) <= atol + rtol * abs(a1)``.
/// Note that this is a basis-dependent and backend-dependent notion of distance, which does
/// not come from a norm in the strict mathematical sense.
///
/// @param tensor_1, tensor_2 The tensors to compare. The legs of both tensors need to be labelled
/// with the same leg labels in order to find the planar permutation between them.
/// @param atol, rtol Absolute and relative tolerance, see above.
///
/// Notes:
///
/// Unlike `almost_equal`, this function does not have the argument `allow_different_types`
/// since permuting legs may change the tensor type.
///
/// almost_equal
///     Comparison between two tensors without planar permutations.
[[nodiscard]] bool planar_almost_equal(TensorCPtr tensor_1,
                                       TensorCPtr tensor_2,
                                       float64 rtol = 1e-5,
                                       float64 atol = 1e-8);

/// Planar special case of `combine_legs`, without braids.
///
/// The legs to be combined must be contiguous, but they do not need to be ordered within each of
/// the groups. In the general case, the legs are bent up / down before combining. The combined leg
/// is the codomain (domain) if the first leg of the group is in the codomain (domain).
///
/// For example::
///
///     |       ║       ║    │
///     |    ╭──╨╮   ╭──╨╮   │   ╭───╮
///     |    │   9   8   7   6   5   │
///     |    │  ┏┷━━━┷━━━┷━━━┷━━━┷┓  │
///     |    │  ┃        T        ┃  │    ==   planar_combine_legs(T, [-1, 0], [3, 4, 5], [7, 8])
///     |    │  ┗┯━━━┯━━━┯━━━┯━━━┯┛  │
///     |    │   0   1   2   3   4   │
///     |    ╰───╯   │   │   ╰╥──┴───╯
///     |            │   │    ║
///
/// In the above example, choosing the group ``[-1, 0]`` means that the combined leg is in the
/// domain, whereas it would end up in the codomain when specifying ``[0, -1]`` instead.
/// Similarly, the combined leg corresponding to the group ``[3, 4, 5]`` would be in the domain
/// when specifying this group as ``[5, 3, 4]`` or ``[5, 4, 3]``.
///
/// @param T The tensor whose legs should be combined.
/// @param *which_legs One or more groups of legs to combine.
/// @param pipe_dualities Can optionally specify the `is_dual` attribute of each resulting pipe.
/// This is an arbitrary choice for each pipe. The pipes are formed such that
/// ``result.legs.[pipe_idx].is_dual == pipe_dualities[i]``. Defaults to all ``False``.
/// @param pipes For each ``group = which_legs[i]`` of legs, the resulting pipe can be passed to
/// avoid recomputation. If we group to the codomain (``group[0] < tensor.num_codomain_legs``), we
/// expect ``LegPipe([tensor._as_codomain_leg(i) for i in group])``. Otherwise we expect
/// ``LegPipe([tensor._as_domain_leg(i) for i in reversed(group)])``. Note the reverse order in the
/// latter case! In the intended use case, when another tensor with the same legs has already been
/// combined, obtain those pipes simply via `get_leg_co_domain`. It is possible to pass only some
/// of the pipes, use ``None`` as filler. combine_legs
///     Non-planar version that automatically braids legs in order to combine them.
[[nodiscard]] TensorPtr planar_combine_legs(
  TensorCPtr T,
  std::vector<std::vector<LegRef>> which_legs,
  std::optional<PipeDualities> pipe_dualities = std::nullopt,
  std::optional<std::vector<Leg::Ptr>> pipes = std::nullopt);

/// Planar version of `tdot` / pairwise contraction without braids.
[[nodiscard]] PlanarResult planar_contraction(TensorCPtr tensor1,
                                              TensorCPtr tensor2,
                                              std::vector<LegRef> legs1,
                                              std::vector<LegRef> legs2,
                                              std::map<std::string, std::string> relabel1 = {},
                                              std::map<std::string, std::string> relabel2 = {});
[[nodiscard]] TensorPlaceholder planar_contraction(TensorPlaceholder const& tensor1,
                                                   TensorPlaceholder const& tensor2,
                                                   std::vector<LegRef> legs1,
                                                   std::vector<LegRef> legs2);

/// Planar eigen-decomposition of a hermitian tensor.
///
/// A tensor decomposition ``tensor ~ V @ W @ dagger(V)`` with
/// the following properties:
///
/// - ``V`` is unitary.
/// - ``W`` is a `DiagonalTensor` with the real eigenvalues of ``tensor``.
///
/// This planar decomposition differs from `eigh` in the sense that
/// it decomposes a tensor into more general left and right parts rather than into codomain
/// and domain.
///
/// *Assumes* that `tensor` is hermitian with respect to the legs specified by
/// `codomain_cut` and `domain_cut`. If `T` is obtained from `tensor` by bending legs
/// s.t. all legs on the left (right) are in the codomain (domain), or, equivalently,
/// ``T = planar_permute_legs(tensor, domain=[*range(codomain_cut, tensor.num_legs -
/// domain_cut))][::-1])``, then ``dagger(T) ~ T``, which requires in particular that ``T.domain ==
/// T.codomain``.
///
/// Graphically, here with ``codomain_cut=3, domain_cut=1``::
///
///     |                                  │    │   │   │
///     |                                  │   ┏┷━━━┷━━━┷┓
///     |                                  │   ┃dagger(V)┃
///     |        │   │   │   │             │   ┗━┯━━━━━┯━┛
///     |       ┏┷━━━┷━━━┷━━━┷┓            │   ┏━┷━┓   │
///     |       ┃   tensor    ┃    ==      │   ┃ W ┃   │
///     |       ┗┯━━━┯━━━┯━━━┯┛            │   ┗━┯━┛   │
///     |        │   │   │   │           ┏━┷━━━━━┷━┓   │
///     |                                ┃    V    ┃   │
///     |                                ┗┯━━━┯━━━┯┛   │
///     |                                 │   │   │    │
///
/// @param tensor The hermitian tensor to decompose.
/// @param codomain_cut The first `codomain_cut` legs from the codomain end up in the codomain of
/// `V`, the rest of the codomain ends up in the codomain of `dagger(V)`.
/// @param domain_cut The first `domain_cut` legs from the domain end up in the domain of `V`, the
/// rest of the domain ends up in the domain of `dagger(V)`.
/// @param new_labels The labels for the new legs can be specified in the following three ways;
/// Three labels ``[a, b, c]`` result in ``V.labels[-1 - domain_cut] == a`` and ``W.labels == [b,
/// c]``. Two labels ``[a, b]`` are equivalent to ``[a, b, a]``. A single label ``a`` is equivalent
/// to ``[a, a*, a]``. The new legs are unlabelled by default.
/// @param new_leg_dual If the new leg should be a ket space (``False``) or bra space (``True``).
/// @param sort How the eigenvalues should are sorted *within* each charge block. Defaults to
/// ``None``, which is same as '<'. See `argsort` for details.
/// @returns W: `DiagonalTensor` The real eigenvalues. V: `SymmetricTensor` The orthonormal
/// eigenvectors.
///
/// eigh
///     Eigen decomposition with respect to codomain and domain. Corresponds to this
///     function with parameters ``codomain_cut=tensor.num_codomain_legs``,
///     ``domain_cut=0``.
[[nodiscard]] std::tuple<DiagonalTensorPtr, TensorPtr> planar_eigh(
  TensorCPtr tensor,
  int64 codomain_cut,
  int64 domain_cut,
  std::optional<LegLabelList> new_labels = std::nullopt,
  bool new_leg_dual = false,
  std::optional<std::string> sort = std::nullopt);

/// Planar eigen-decomposition of a general (not necessarily hermitian) tensor.
///
/// A tensor decomposition ``tensor ~ V @ W @ pinv(V)``. Unlike `planar_eigh`, `V` is in general
/// not unitary and `W` is generally complex.
///
/// This planar decomposition differs from `eig` in the sense that it decomposes a tensor into
/// more general left and right parts rather than into codomain and domain.
///
/// Requires that the bent tensor `T` obtained by moving legs according to `codomain_cut` and
/// `domain_cut` has ``T.domain == T.codomain``.
///
/// @param tensor The tensor to decompose.
/// @param codomain_cut The first `codomain_cut` legs from the codomain end up in the codomain of
///     `V`.
/// @param domain_cut The first `domain_cut` legs from the domain end up in the domain of `V`.
/// @param new_labels Labels for the new legs, same convention as `eig` / `planar_eigh`.
///     Unlabelled by default.
/// @param new_leg_dual If the new leg should be a ket space (`false`) or bra space (`true`).
/// @param sort How the eigenvalues are sorted *within* each charge block. See `argsort`.
/// @returns `(W, V)`: eigenvalues and right eigenvectors.
[[nodiscard]] std::tuple<DiagonalTensorPtr, TensorPtr> planar_eig(
  TensorCPtr tensor,
  int64 codomain_cut,
  int64 domain_cut,
  std::optional<LegLabelList> new_labels = std::nullopt,
  bool new_leg_dual = false,
  std::optional<std::string> sort = std::nullopt);

/// Planar eigenvalues of a general tensor, without eigenvectors.
///
/// Same as `planar_eig`, but returns only the eigenvalues `W`.
///
/// @param tensor The tensor to decompose.
/// @param codomain_cut, domain_cut Same as `planar_eig`.
/// @param new_labels Labels for `W` only. One label `a` is equivalent to `[a, a*]`. Two labels
///     `[b, c]` set `W.labels == [b, c]`. Unlabelled by default.
/// @param new_leg_dual If the new leg should be a ket space (`false`) or bra space (`true`).
/// @param sort How the eigenvalues are sorted *within* each charge block. See `argsort`.
[[nodiscard]] DiagonalTensorPtr planar_eigvals(
  TensorCPtr tensor,
  int64 codomain_cut,
  int64 domain_cut,
  std::optional<LegLabelList> new_labels = std::nullopt,
  bool new_leg_dual = false,
  std::optional<std::string> sort = std::nullopt);

/// Planar LQ decomposition of a tensor.
///
/// A tensor decomposition ``tensor ~ L @ Q`` with the following
/// properties:
///
/// - ``L`` has a lower triangular structure *in the coupled basis*.
/// - ``Q`` is an isometry.
///
/// This planar decomposition differs from `lq` in the sense that it
/// decomposes a tensor into more general left and right parts rather than into codomain
/// and domain.
///
/// Graphically, here with ``codomain_cut=2, domain_cut=1``::
///
///     |                                  │  │  │  │
///     |                                  │ ┏┷━━┷━━┷┓
///     |        │   │   │   │             │ ┃   Q   ┃
///     |       ┏┷━━━┷━━━┷━━━┷┓            │ ┗━┯━━━┯━┛
///     |       ┃   tensor    ┃    ==      │   │   │
///     |       ┗━━┯━━━┯━━━┯━━┛          ┏━┷━━━┷━┓ │
///     |          │   │   │             ┃   L   ┃ │
///     |                                ┗━┯━━━┯━┛ │
///     |                                  │   │   │
///
/// We always compute the "reduced", a.k.a. "economic" version.
///
/// @param tensor The tensor to decompose.
/// @param codomain_cut The first `codomain_cut` legs from the codomain end up in the codomain of
/// `L`, the rest of the codomain ends up in the codomain of `Q`.
/// @param domain_cut The first `domain_cut` legs from the domain end up in the domain of `L`, the
/// rest of the domain ends up in the domain of `Q`.
/// @param new_labels Labels for the new legs. Either two legs ``[a, b]`` s.t. ``L.labels[-1 -
/// domain_cut] == a`` and ``Q.labels[0] == b``. A single label ``a`` is equivalent to ``[a, a*]``.
/// @param new_leg_dual If the new leg should be a ket space (``False``) or bra space (``True``).
/// lq
///     LQ decomposition with respect to codomain and domain. Corresponds to this
///     function with parameters ``codomain_cut=tensor.num_codomain_legs``,
///     ``domain_cut=0``.
[[nodiscard]] std::tuple<TensorPtr, TensorPtr> planar_lq(
  TensorCPtr tensor,
  int64 codomain_cut,
  int64 domain_cut,
  std::optional<LegLabelList> new_labels = std::nullopt,
  bool new_leg_dual = false);

/// Planar version of `partial_trace`.
///
/// Here, planar means that the trace can be drawn as a diagram in a plane, without any braids.
///
/// For example::
///
///     |    ╭───╮   │   │   ╭───╮
///     |    │   7   6   5   4   │
///     |    │  ┏┷━━━┷━━━┷━━━┷┓  │
///     |    │  ┃      A      ┃  │    ==   planar_partial_trace(A, (0, 1), (2, -1), (3, 4))
///     |    │  ┗┯━━━┯━━━┯━━━┯┛  │
///     |    │   0   1   2   3   │
///     |    │   ╰───╯   │   ╰───╯
///     |    ╰───────────╯
///
/// @param tensor The tensor to act on.
/// @param *pairs A number of pairs, each describing two legs via index or via label. Each pair is
/// connected, realizing a partial trace. By definition, we create loops between the legs in a
/// planar way by connecting them over the left or right side of the tensor. If both a connecting
/// loop over the left and the right side are planar, the result is independent of this choice.
/// Must be compatible ``tensor.get_leg(pair[0]) == tensor.get_leg(pair[1]).dual``.
/// @returns If all legs are traced, a python scalar. If legs are left open, a tensor with the same
/// type as `tensor`.
///
/// partial_trace
///     Non-planar partial trace which may include braiding of legs with specified levels.
[[nodiscard]] PlanarResult planar_partial_trace(TensorCPtr tensor,
                                                std::vector<std::vector<LegRef>> pairs);
[[nodiscard]] TensorPlaceholder planar_partial_trace(TensorPlaceholder const& tensor,
                                                     std::vector<std::vector<LegRef>> pairs);

/// Planar special case of `permute_legs`, without braids.
///
/// It permutes the `legs` only cyclically, and bends them to the proper codomain / domain.
///
/// A planar permutation consists only of leg bends, either to the left or right of the tensor.
/// It leaves the `legs` unchanged up to cyclical permutation.
/// It is fully specified by assigning each leg to either the new codomain or the new domain.
///
/// @param T The tensor whose legs are to be permuted.
/// @param codomain, domain The legs that should be in the new (co)domain, in the correct order.
/// Only one of `codomain`, `domain` is required when the other can be unambiguously inferred. This
/// is the case when the specified `codomain` or `domain` contains at least one leg.
[[nodiscard]] TensorPtr planar_permute_legs(
  TensorCPtr T,
  std::optional<std::vector<LegRef>> codomain = std::nullopt,
  std::optional<std::vector<LegRef>> domain = std::nullopt);

/// Planar QR decomposition of a tensor.
///
/// A tensor decomposition ``tensor ~ Q @ R`` with the following
/// properties:
///
/// - ``Q`` is an isometry.
/// - ``R`` has an upper triangular structure *in the coupled basis*.
///
/// This planar decomposition differs from `qr` in the sense that it
/// decomposes a tensor into more general left and right parts rather than into codomain
/// and domain.
///
/// Graphically, here with ``codomain_cut=2, domain_cut=1``::
///
///     |                                  │  │  │  │
///     |                                  │ ┏┷━━┷━━┷┓
///     |        │   │   │   │             │ ┃   R   ┃
///     |       ┏┷━━━┷━━━┷━━━┷┓            │ ┗━┯━━━┯━┛
///     |       ┃   tensor    ┃    ==      │   │   │
///     |       ┗━━┯━━━┯━━━┯━━┛          ┏━┷━━━┷━┓ │
///     |          │   │   │             ┃   Q   ┃ │
///     |                                ┗━┯━━━┯━┛ │
///     |                                  │   │   │
///
/// We always compute the "reduced", a.k.a. "economic" version.
///
/// @param tensor The tensor to decompose.
/// @param codomain_cut The first `codomain_cut` legs from the codomain end up in the codomain of
/// `Q`, the rest of the codomain ends up in the codomain of `R`.
/// @param domain_cut The first `domain_cut` legs from the domain end up in the domain of `Q`, the
/// rest of the domain ends up in the domain of `R`.
/// @param new_labels Labels for the new legs. Either two legs ``[a, b]`` s.t. ``Q.labels[-1 -
/// domain_cut] == a`` and ``R.labels[0] == b``. A single label ``a`` is equivalent to ``[a, a*]``.
/// @param new_leg_dual If the new leg should be a ket space (``False``) or bra space (``True``).
/// qr
///     QR decomposition with respect to codomain and domain. Corresponds to this
///     function with parameters ``codomain_cut=tensor.num_codomain_legs``,
///     ``domain_cut=0``.
[[nodiscard]] std::tuple<TensorPtr, TensorPtr> planar_qr(
  TensorCPtr tensor,
  int64 codomain_cut,
  int64 domain_cut,
  std::optional<LegLabelList> new_labels = std::nullopt,
  bool new_leg_dual = false);

/// Planar singular value decomposition (SVD) of a tensor.
///
/// A tensor decomposition ``tensor ~ U @ S @ Vh`` with the following
/// properties:
///
/// - ``Vh`` and ``U`` are isometries.
/// - ``S`` is a `DiagonalTensor` with real, non-negative entries.
/// - If `tensor` is a matrix (i.e. if it has exactly one leg each in domain and codomain), it
///   reproduces the usual matrix SVD.
///
/// .. note ::
///     The basis for the newly generated leg is chosen arbitrarily, and in particular, unlike,
///     e.g., `svd`, it is not guaranteed that ``S.diag_numpy`` is sorted.
///
/// This planar decomposition differs from `svd` in the sense that it
/// decomposes a tensor into more general left and right parts rather than into codomain and
/// domain.
///
/// Graphically, here with ``codomain_cut=2, domain_cut=1``::
///
///     |                                  │    │   │   │
///     |                                  │   ┏┷━━━┷━━━┷┓
///     |                                  │   ┃   Vh    ┃
///     |        │   │   │   │             │   ┗━┯━━━━━┯━┛
///     |       ┏┷━━━┷━━━┷━━━┷┓            │   ┏━┷━┓   │
///     |       ┃   tensor    ┃    ==      │   ┃ S ┃   │
///     |       ┗━━┯━━━┯━━━┯━━┛            │   ┗━┯━┛   │
///     |          │   │   │             ┏━┷━━━━━┷━┓   │
///     |                                ┃    U    ┃   │
///     |                                ┗━┯━━━━━┯━┛   │
///     |                                  │     │     │
///
/// We always compute the "reduced", a.k.a. "economic" version of SVD, where the
/// isometries are (in general) not full unitaries.
///
/// @param tensor The tensor to decompose.
/// @param codomain_cut The first `codomain_cut` legs from the codomain end up in the codomain of
/// `U`, the rest of the codomain ends up in the codomain of `Vh`.
/// @param domain_cut The first `domain_cut` legs from the domain end up in the domain of `U`, the
/// rest of the domain ends up in the domain of `Vh`.
/// @param new_labels The labels for the new legs can be specified in the following three ways;
/// Four labels ``[a, b, c, d]`` result in ``U.labels[-1 - domain_cut] == a``, ``S.labels == [b,
/// c]`` and ``Vh.labels[0] == d``. Two labels ``[a, b]`` are equivalent to ``[a, b, a, b]``. A
/// single label ``a`` is equivalent to ``[a, a*, a, a*]``. The new legs are unlabelled by default.
/// @param new_leg_dual If the new leg should be a ket space (``False``) or bra space (``True``).
/// @param algorithm The algorithm (a.k.a. "driver") for the block-wise svd. Choices are
/// backend-specific. See `possible_svd_algorithms`.
/// @returns U: SymmetricTensor S: DiagonalTensor Vh: SymmetricTensor
///
/// svd
///     SVD decomposition with respect to codomain and domain. Corresponds to this
///     function with parameters ``codomain_cut=tensor.num_codomain_legs``,
///     ``domain_cut=0``.
[[nodiscard]] std::tuple<TensorPtr, DiagonalTensorPtr, TensorPtr> planar_svd(
  TensorCPtr tensor,
  int64 codomain_cut,
  int64 domain_cut,
  std::optional<LegLabelList> new_labels = std::nullopt,
  bool new_leg_dual = false,
  std::optional<std::string> algorithm = std::nullopt);

[[nodiscard]] std::tuple<TensorPtr, DiagonalTensorPtr, TensorPtr, float64, float64>
/// Truncated version of `planar_svd`.
planar_truncated_svd(TensorCPtr tensor,
                     int64 codomain_cut,
                     int64 domain_cut,
                     std::optional<LegLabelList> new_labels = std::nullopt,
                     bool new_leg_dual = false,
                     std::optional<std::string> algorithm = std::nullopt,
                     std::optional<float64> normalize_to = std::nullopt,
                     std::optional<int64> chi_max = std::nullopt,
                     int64 chi_min = 1,
                     float64 degeneracy_tol = 0.,
                     float64 trunc_cut = 0.,
                     float64 svd_min = 0.);

[[nodiscard]] std::pair<TensorPtr, std::optional<int64>>
_planar_contraction_helper(TensorCPtr tensor, std::vector<int64> const& contr, bool domain);

} // namespace lasyten
