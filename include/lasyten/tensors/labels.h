#pragma once

#include <lasyten/lasyten.h>
#include <lasyten/tools.h>
#include <lasyten/tools/hdf5.h>

#include <functional>
#include <map>
#include <optional>
#include <string>
#include <variant>
#include <vector>

namespace lasyten {

/// Flat list of leg labels in `legs` order. Empty string means unset.
using LegLabelList = std::vector<std::string>;

/// Reserved character to indicate contractions in `lasyten.planar` diagrams.
inline constexpr char CONTRACT_SYMBOL = '@';
/// Reserved character to select a leg of a tensor in `lasyten.planar` diagrams.
inline constexpr char LEG_SELECT_SYMBOL = ':';
/// Reserved characters to indicate an open leg in `lasyten.planar` diagrams.
inline constexpr char const* OPEN_LEG_SYMBOL = "->";

/// Characters that are forbidden in leg labels.
inline constexpr char const* FORBIDDEN_LEG_LABEL_CHARS[] = {
    " ", "\t", "\n", "@", ":", "-", ">",
};

/// If the given string is a valid leg label (`""` is always valid = unset).
[[nodiscard]] bool is_valid_leg_label(std::string const& label);

/// The label that a combined leg should have.
///
/// @param labels Labels of the legs being combined.
/// @param offset Added to the index used for anonymous `?n` placeholders.
///     Defaults to `0` (needed by `_dual_leg_label` for nested combined labels).
[[nodiscard]] std::string _combine_leg_labels(LegLabelList const& labels, int64 offset = 0);

/// Undo `_combine_leg_labels`, i.e. recover the original labels.
///
/// @param label Combined label, or `""` for an anonymous group.
/// @param num Required length when `label` is empty; otherwise optional.
[[nodiscard]] LegLabelList _split_leg_label(std::string const& label,
                                            std::optional<int64> num = std::nullopt);

/// The label that a leg should have after conjugation.
[[nodiscard]] std::string _dual_leg_label(std::string const& label);

/// Dual labels in reversed order (helper for conjugated combined labels).
[[nodiscard]] LegLabelList _dual_label_list(LegLabelList const& labels);

/// Utility function to combine two lists of labels that should match.
///
/// Per pair of labels:
/// - If one is unset (`""`), use the other.
/// - If they are equal, use that label.
/// - If they are different, emit a DEBUG message to the logger and choose `""`.
///
/// @param labels1,labels2 Label lists that should match pairwise.
/// @returns Combined labels (length of the shorter input).
[[nodiscard]] LegLabelList _get_matching_labels(LegLabelList const& labels1,
                                                LegLabelList const& labels2);

/// Handles labelled legs for tensors (composition, not inheritance).
///
/// Unset labels are empty strings. Behaves list-like for equality, indexing, and
/// assignment from `LegLabelList`, without full iterator support.
class LegLabels
{
  public:
    /// Called before a full or single-label assignment with the proposed full list.
    /// May throw to reject the mutation. Does not own `this`.
    using BeforeHook = std::function<void(LegLabels& self, LegLabelList const& proposed)>;
    /// Called after a successful mutation. Does not own `this`.
    using AfterHook = std::function<void(LegLabels& self)>;

    LegLabels() = default;
    explicit LegLabels(LegLabelList labels);

    LegLabels(LegLabels const& other);
    LegLabels(LegLabels&& other) noexcept;
    LegLabels& operator=(LegLabels const& other);
    LegLabels& operator=(LegLabels&& other) noexcept;

    /// Assign from a label list (same length required if already non-empty).
    LegLabels& operator=(LegLabelList labels);

    [[nodiscard]] bool operator==(LegLabelList const& other) const;
    [[nodiscard]] bool operator!=(LegLabelList const& other) const { return !(*this == other); }
    [[nodiscard]] bool operator==(LegLabels const& other) const;
    [[nodiscard]] bool operator!=(LegLabels const& other) const { return !(*this == other); }

    /// Install mutation hooks (non-owning). Cleared on copy/move of the LegLabels value;
    /// owners must re-bind after copy.
    void set_hooks(BeforeHook before, AfterHook after);
    void clear_hooks();

    void test_sanity() const;

    [[nodiscard]] bool is_fully_labelled() const;

    [[nodiscard]] std::size_t size() const { return labels_.size(); }
    [[nodiscard]] int64 num_legs() const { return static_cast<int64>(labels_.size()); }
    [[nodiscard]] bool empty() const { return labels_.empty(); }

    /// Copy of the internal label list.
    [[nodiscard]] LegLabelList as_list() const { return labels_; }

    [[nodiscard]] std::string const& operator[](int64 idx) const;
    [[nodiscard]] std::string const& at(int64 idx) const { return (*this)[idx]; }

    /// Proxy enabling `labels[i] = ...` and `labels["old"] = "new"`.
    class Ref
    {
      public:
        Ref(LegLabels& self, std::variant<int64, std::string> key)
          : self_(self)
          , key_(std::move(key))
        {
        }

        Ref& operator=(std::string label);
        Ref& operator=(char const* label) { return *this = std::string(label); }

        [[nodiscard]] operator std::string const&() const;

      private:
        LegLabels& self_;
        std::variant<int64, std::string> key_;
    };

    Ref operator[](int64 idx);
    Ref operator[](std::string const& label);

    /// Set all labels in-place (same length if already sized).
    LegLabels& set(LegLabelList labels);

    /// Apply mapping to labels in-place (keys are current labels).
    LegLabels& update(std::map<std::string, std::string> const& mapping);

    /// Normalize an int index or look up a label → single leg index.
    [[nodiscard]] int64 index(int64 idx) const;
    [[nodiscard]] int64 index(std::string const& label) const;

    /// Parse a sequence of indices/labels to leg indices.
    [[nodiscard]] std::vector<int64> indices(
      std::vector<std::variant<int64, std::string>> const& idcs) const;

    [[nodiscard]] bool has(std::string const& label) const;
    [[nodiscard]] bool has(std::vector<std::string> const& labels) const;
    [[nodiscard]] bool contains(std::string const& label) const { return has(label); }

    /// If the given labels and this are the same, up to permutation.
    /// If `planar` is true, only cyclic permutations are accepted.
    [[nodiscard]] bool are(std::vector<std::string> const& labels, bool planar = false) const;

    void save_hdf5(lasyten::hdf5::Saver& saver,
                   HighFive::Group& h5gr,
                   std::string const& subpath) const;

    [[nodiscard]] static LegLabels from_hdf5(lasyten::hdf5::Loader& loader,
                                             HighFive::Group& h5gr,
                                             std::string const& subpath);

  private:
    friend class Ref;

    void assign_impl(LegLabelList labels, bool require_same_size);
    void set_at(int64 pos, std::string label);
    void rename(std::string const& old_label, std::string new_label);
    [[nodiscard]] int64 find_label(std::string const& label) const;

    LegLabelList labels_;
    BeforeHook before_hook_;
    AfterHook after_hook_;
};

} // namespace lasyten
