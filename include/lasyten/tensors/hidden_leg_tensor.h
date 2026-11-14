#pragma once

#include <lasyten/tensors/symmetric_tensor.h>

#include <lasyten/tools/hdf5.h>
#include <memory>
#include <optional>
#include <string>
#include <variant>
#include <vector>

namespace lasyten {

/// A `SymmetricTensor` with one or more legs marked as hidden via `!`-prefixed labels.
///
/// Unlike `ChargedTensor`, this object *is* a symmetric tensor: `data`, `domain`, `codomain`,
/// `labels`, and `num_legs` include the hidden legs. Hidden legs are special only in operations
/// such as `tdot`, `permute_legs`, `combine_legs`, `norm`, etc.:
///
/// - User-facing leg arguments must not name hidden legs.
/// - Contracting two `HiddenLegTensor`s implicitly contracts dual hidden labels
///   (``!a`` with ``!a*``). Equal hidden labels (both starred or neither) raise.
/// - Unmatched hidden legs remain open on the result.
/// - `norm` is equivalent to `inner(T, T, do_dagger=True)` and contracts hidden legs
///   with their duals (same as the Frobenius norm of the underlying `SymmetricTensor`).
/// - Other scalar-returning ops (`trace`, `item`) require that no hidden legs remain on
///   the input (except `inner`, which may pair hidden legs between arguments).
///
/// Construct from an existing tensor by selecting which legs to hide; the constructor prefixes
/// ``!`` to those labels.
class HiddenLegTensor : public SymmetricTensor
{
  public:
    using Ptr = std::shared_ptr<HiddenLegTensor>;
    using CPtr = std::shared_ptr<const HiddenLegTensor>;

    /// Prefix character for hidden-leg labels.
    static constexpr char HIDDEN_PREFIX = '!';

    /// Adopt an existing symmetric tensor whose labels already contain hidden (`!`) legs.
    /// Used by ops that preserve hidden labels on the result.
    explicit HiddenLegTensor(SymmetricTensor::Ptr tensor);

    ~HiddenLegTensor() override = default;

    /// Construct by hiding `which_legs` on an existing tensor (prefixes ``!`` to their labels).
    [[nodiscard]] static Ptr from_tensor(Tensor::Ptr tensor,
                                         std::vector<std::variant<int64, std::string>> which_legs);

    void test_sanity() const override;

    [[nodiscard]] std::string ascii_diagram_type_name() const override;
    [[nodiscard]] std::string class_name() const override;

    /// True for ChargedTensor charge markers and short-lived compose temps (``!``, ``!1``,
    /// ``!A``).
    [[nodiscard]] static bool is_charge_temp_label(std::string const& label);

    /// True if `label` is a user-facing hidden leg label (``!`` prefix, not a charge temp).
    [[nodiscard]] static bool is_hidden_leg_label(std::string const& label);

    /// True if any label starts with ``!``.
    [[nodiscard]] static bool has_hidden_leg_labels(LegLabelList const& labels);

    /// Strip a leading ``!`` from a hidden label. Returns `label` unchanged if not hidden.
    [[nodiscard]] static std::string strip_hidden_prefix(std::string const& label);

    /// Prefix ``!`` to a non-hidden label. Raises if already hidden or empty.
    [[nodiscard]] static std::string add_hidden_prefix(std::string const& label);

    /// Raise if hidden labels contain duplicates or a dual pair (`!a` and `!a*`).
    static void validate_no_dual_hidden_pair(LegLabelList const& labels);

    /// Indices of legs whose labels start with ``!``.
    [[nodiscard]] std::vector<int64> hidden_leg_idcs() const;

    /// Indices of legs that are not hidden.
    [[nodiscard]] std::vector<int64> public_leg_idcs() const;

    /// Return a plain `SymmetricTensor` with ``!`` stripped from hidden labels.
    [[nodiscard]] SymmetricTensorPtr unhide_legs() const;

    /// Same as `unhide_legs` (strip ``!``); always returns a copy when `guarantee_copy`.
    [[nodiscard]] SymmetricTensorPtr as_SymmetricTensor(
      bool guarantee_copy = false,
      std::optional<std::string> warning = std::nullopt) override;

    [[nodiscard]] Tensor::Ptr as_dtype(Dtype dtype) override;

    [[nodiscard]] Tensor::Ptr copy(bool deep = true,
                                   std::optional<std::string> device = std::nullopt,
                                   std::optional<Dtype> dtype = std::nullopt) override;

    [[nodiscard]] Tensor::Ptr dagger() const override;

    [[nodiscard]] Tensor::Ptr to_backend(
      TensorBackend::Ptr backend,
      std::optional<Dtype> dtype = std::nullopt,
      std::optional<std::string> device = std::nullopt) override;

    /// Import from hdf5
    [[nodiscard]] static Ptr from_hdf5(lasyten::hdf5::Loader& loader,
                                       HighFive::Group& h5gr,
                                       std::string const& subpath);

    /// Export to hdf5
    void save_hdf5(lasyten::hdf5::Saver& saver,
                   HighFive::Group& h5gr,
                   std::string const& subpath) const;

    /// Wrap `tensor` as `HiddenLegTensor` if it has hidden labels; otherwise return `tensor`.
    [[nodiscard]] static TensorPtr maybe_wrap(SymmetricTensor::Ptr tensor);
};

/// True if any label contains the character ``!`` (not necessarily as a prefix).
[[nodiscard]] bool label_contains_exclamation(std::string const& label);

/// Raise if any label contains ``!``. Used by normal (non-HiddenLeg) tensors.
void reject_exclamation_in_labels(LegLabelList const& labels, std::string const& context);

} // namespace lasyten
