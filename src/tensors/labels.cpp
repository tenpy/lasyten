#include <lasyten/tensors/labels.h>

#include <lasyten/tools/hdf5_export.h>

#include <cassert>
#include <format>
#include <iostream>
#include <ranges>
#include <set>
#include <sstream>
#include <stdexcept>
#include <unordered_set>

namespace lasyten {

namespace {

/// Duplicate entries in ``seq``, ignoring unset (``""``) labels.
[[nodiscard]] std::unordered_set<std::string>
duplicate_leg_labels(LegLabelList const& seq)
{
    std::unordered_set<std::string> seen;
    std::unordered_set<std::string> dups;
    for (auto const& ele : seq) {
        if (ele.empty()) {
            continue;
        }
        if (!seen.insert(ele).second) {
            dups.insert(ele);
        }
    }
    return dups;
}

[[nodiscard]] std::string
format_leg_labels(LegLabelList const& labels)
{
    std::ostringstream out;
    out << '[';
    bool first = true;
    for (auto const& l : labels) {
        if (!first) {
            out << ", ";
        }
        first = false;
        if (l.empty()) {
            out << "''";
        } else {
            out << l;
        }
    }
    out << ']';
    return out.str();
}

[[nodiscard]] std::string
format_duplicate_error(std::unordered_set<std::string> const& dup)
{
    std::string joined;
    bool first = true;
    for (auto const& d : dup) {
        if (!first) {
            joined += ", ";
        }
        first = false;
        joined += d;
    }
    return std::format("Duplicate leg labels: {}", joined);
}

} // namespace

bool
is_valid_leg_label(std::string const& label)
{
    if (label.empty()) {
        return true;
    }
    // TODO extend: check for valid syntax of combined / conjugated labels?
    for (char const* forbidden : FORBIDDEN_LEG_LABEL_CHARS) {
        if (label.find(forbidden) != std::string::npos) {
            return false;
        }
    }
    return true;
}

std::string
_combine_leg_labels(LegLabelList const& labels, int64 offset)
{
    std::string joined;
    for (std::size_t n = 0; n < labels.size(); ++n) {
        if (n > 0) {
            joined += '.';
        }
        if (labels[n].empty()) {
            joined += std::format("?{}", static_cast<int64>(n) + offset);
        } else {
            joined += labels[n];
        }
    }
    return std::format("({})", joined);
}

LegLabelList
_split_leg_label(std::string const& label, std::optional<int64> num)
{
    if (label.empty()) {
        if (!num.has_value()) {
            throw std::invalid_argument("num is required when splitting an unset combined label");
        }
        return LegLabelList(static_cast<std::size_t>(*num), "");
    }
    if (label.starts_with('(') && label.ends_with(')')) {
        std::string const inner = label.substr(1, label.size() - 2);
        LegLabelList labels;
        // Match Python ``str.split('.')`` (keeps empty segments).
        std::size_t start = 0;
        while (true) {
            std::size_t const dot = inner.find('.', start);
            std::string part =
              (dot == std::string::npos) ? inner.substr(start) : inner.substr(start, dot - start);
            if (part.starts_with('?')) {
                labels.emplace_back("");
            } else {
                labels.emplace_back(std::move(part));
            }
            if (dot == std::string::npos) {
                break;
            }
            start = dot + 1;
        }
        if (num.has_value() && static_cast<int64>(labels.size()) != *num) {
            throw std::invalid_argument(std::format(
              "expected {} components in combined label, got {}", *num, labels.size()));
        }
        return labels;
    }
    throw std::invalid_argument("Invalid format for a combined label");
}

LegLabelList
_dual_label_list(LegLabelList const& labels)
{
    LegLabelList out;
    out.reserve(labels.size());
    for (auto const& l : labels | std::views::reverse) {
        out.push_back(_dual_leg_label(l));
    }
    return out;
}

std::string
_dual_leg_label(std::string const& label)
{
    if (label.empty()) {
        return "";
    }
    if (label.starts_with('(') && label.ends_with(')')) {
        return _combine_leg_labels(_dual_label_list(_split_leg_label(label)));
    }
    if (label.ends_with('*')) {
        return label.substr(0, label.size() - 1);
    }
    return label + '*';
}

LegLabelList
_get_matching_labels(LegLabelList const& labels1, LegLabelList const& labels2)
{
    LegLabelList labels;
    std::vector<int64> conflicts;
    auto const n = std::min(labels1.size(), labels2.size());
    labels.reserve(n);
    for (std::size_t i = 0; i < n; ++i) {
        auto const& l1 = labels1[i];
        auto const& l2 = labels2[i];
        if (l1.empty()) {
            labels.push_back(l2);
        } else if (l2.empty() || l1 == l2) {
            labels.push_back(l1);
        } else {
            conflicts.push_back(static_cast<int64>(i));
            labels.emplace_back("");
        }
    }
    if (!conflicts.empty() && Py_IsInitialized()) {
        std::string positions;
        for (std::size_t i = 0; i < conflicts.size(); ++i) {
            if (i > 0) {
                positions += ", ";
            }
            positions += std::to_string(conflicts[i]);
        }
        auto const msg =
          std::format("Conflicting labels at positions {} are dropped. labels1={}, labels2={}.",
                      positions,
                      format_leg_labels(labels1),
                      format_leg_labels(labels2));
        std::cerr << "lasyten.tensors._tensors DEBUG: " << msg << '\n';
    }
    return labels;
}

LegLabels::LegLabels(LegLabelList labels)
{
    assign_impl(std::move(labels), /*require_same_size=*/false);
}

LegLabels::LegLabels(LegLabels const& other)
  : labels_(other.labels_)
{
    // Hooks are owner-specific; do not copy.
}

LegLabels::LegLabels(LegLabels&& other) noexcept
  : labels_(std::move(other.labels_))
{
    other.before_hook_ = nullptr;
    other.after_hook_ = nullptr;
}

LegLabels&
LegLabels::operator=(LegLabels const& other)
{
    if (this == &other) {
        return *this;
    }
    // Keep our hooks; replace data (with validation).
    return set(other.labels_);
}

LegLabels&
LegLabels::operator=(LegLabels&& other) noexcept
{
    if (this == &other) {
        return *this;
    }
    // Keep our hooks; move data without re-running hooks (move assign of the container
    // is only used when rebuilding owners that re-bind hooks afterward).
    labels_ = std::move(other.labels_);
    other.before_hook_ = nullptr;
    other.after_hook_ = nullptr;
    return *this;
}

LegLabels&
LegLabels::operator=(LegLabelList labels)
{
    return set(std::move(labels));
}

bool
LegLabels::operator==(LegLabelList const& other) const
{
    return labels_ == other;
}

bool
LegLabels::operator==(LegLabels const& other) const
{
    return labels_ == other.labels_;
}

void
LegLabels::set_hooks(BeforeHook before, AfterHook after)
{
    before_hook_ = std::move(before);
    after_hook_ = std::move(after);
}

void
LegLabels::clear_hooks()
{
    before_hook_ = nullptr;
    after_hook_ = nullptr;
}

void
LegLabels::test_sanity() const
{
    assert(std::ranges::all_of(labels_, [](auto const& l) { return is_valid_leg_label(l); }));
    assert(duplicate_leg_labels(labels_).empty());
}

bool
LegLabels::is_fully_labelled() const
{
    return std::ranges::all_of(labels_, [](std::string const& l) { return !l.empty(); });
}

std::string const&
LegLabels::operator[](int64 idx) const
{
    return labels_[static_cast<std::size_t>(index(idx))];
}

LegLabels::Ref
LegLabels::operator[](int64 idx)
{
    return Ref{ *this, idx };
}

LegLabels::Ref
LegLabels::operator[](std::string const& label)
{
    return Ref{ *this, label };
}

LegLabels::Ref&
LegLabels::Ref::operator=(std::string label)
{
    if (std::holds_alternative<int64>(key_)) {
        self_.set_at(std::get<int64>(key_), std::move(label));
    } else {
        self_.rename(std::get<std::string>(key_), std::move(label));
    }
    return *this;
}

LegLabels::Ref::
operator std::string const&() const
{
    if (std::holds_alternative<int64>(key_)) {
        return self_[std::get<int64>(key_)];
    }
    return self_[self_.index(std::get<std::string>(key_))];
}

void
LegLabels::assign_impl(LegLabelList labels, bool require_same_size)
{
    auto const dups = duplicate_leg_labels(labels);
    if (!dups.empty()) {
        throw std::invalid_argument(format_duplicate_error(dups));
    }
    if (require_same_size && !labels_.empty() && labels.size() != labels_.size()) {
        throw std::invalid_argument(
          std::format("expected {} labels, got {}", labels_.size(), labels.size()));
    }
    for (auto const& l : labels) {
        if (!is_valid_leg_label(l)) {
            throw std::invalid_argument(std::format("Invalid leg label: '{}'", l));
        }
    }
    if (before_hook_) {
        before_hook_(*this, labels);
    }
    labels_ = std::move(labels);
    if (after_hook_) {
        after_hook_(*this);
    }
}

LegLabels&
LegLabels::set(LegLabelList labels)
{
    assign_impl(std::move(labels), /*require_same_size=*/!labels_.empty());
    return *this;
}

void
LegLabels::set_at(int64 pos, std::string label)
{
    pos = index(pos);
    if (!is_valid_leg_label(label)) {
        throw std::invalid_argument(std::format("Invalid leg label: '{}'", label));
    }
    if (!label.empty()) {
        for (std::size_t i = 0; i < labels_.size(); ++i) {
            if (static_cast<int64>(i) == pos) {
                continue;
            }
            if (labels_[i] == label) {
                throw std::invalid_argument("Duplicate label");
            }
        }
    }
    LegLabelList proposed = labels_;
    proposed[static_cast<std::size_t>(pos)] = label;
    if (before_hook_) {
        before_hook_(*this, proposed);
    }
    labels_ = std::move(proposed);
    if (after_hook_) {
        after_hook_(*this);
    }
}

void
LegLabels::rename(std::string const& old_label, std::string new_label)
{
    auto const pos = index(old_label);
    set_at(pos, std::move(new_label));
}

LegLabels&
LegLabels::update(std::map<std::string, std::string> const& mapping)
{
    LegLabelList next;
    next.reserve(labels_.size());
    for (auto const& l : labels_) {
        if (!l.empty()) {
            auto it = mapping.find(l);
            next.push_back(it == mapping.end() ? l : it->second);
        } else {
            next.emplace_back("");
        }
    }
    return set(std::move(next));
}

int64
LegLabels::find_label(std::string const& label) const
{
    if (label.empty()) {
        throw std::invalid_argument("Cannot look up an unset (empty) label");
    }
    for (std::size_t i = 0; i < labels_.size(); ++i) {
        if (labels_[i] == label) {
            return static_cast<int64>(i);
        }
    }
    throw std::invalid_argument(
      std::format("No leg with label {}. Labels are {}", label, format_leg_labels(labels_)));
}

int64
LegLabels::index(int64 idx) const
{
    return to_valid_idx(idx, num_legs());
}

int64
LegLabels::index(std::string const& label) const
{
    return find_label(label);
}

std::vector<int64>
LegLabels::indices(std::vector<std::variant<int64, std::string>> const& idcs) const
{
    std::vector<int64> res;
    res.reserve(idcs.size());
    for (auto const& idx : idcs) {
        if (std::holds_alternative<std::string>(idx)) {
            res.push_back(index(std::get<std::string>(idx)));
        } else {
            res.push_back(index(std::get<int64>(idx)));
        }
    }
    return res;
}

bool
LegLabels::has(std::string const& label) const
{
    if (label.empty()) {
        return false;
    }
    return std::ranges::find(labels_, label) != labels_.end();
}

bool
LegLabels::has(std::vector<std::string> const& more) const
{
    return std::ranges::all_of(more, [this](std::string const& l) { return has(l); });
}

bool
LegLabels::are(std::vector<std::string> const& want, bool planar) const
{
    if (!is_fully_labelled()) {
        return false;
    }
    if (static_cast<int64>(want.size()) != num_legs()) {
        return false;
    }
    if (planar) {
        if (want.empty()) {
            return true;
        }
        int64 start = -1;
        try {
            start = find_label(want.front());
        } catch (std::invalid_argument const&) {
            return false;
        }
        for (std::size_t i = 1; i < want.size(); ++i) {
            auto const idx = (static_cast<std::size_t>(start) + i) % want.size();
            if (labels_[idx] != want[i]) {
                return false;
            }
        }
        return true;
    }

    std::set<std::string> a(want.begin(), want.end());
    std::set<std::string> b(labels_.begin(), labels_.end());
    return a == b;
}

void
LegLabels::save_hdf5(lasyten::hdf5::Saver& saver,
                     HighFive::Group& /*h5gr*/,
                     std::string const& subpath) const
{
    hdf5_export::save_leg_labels(saver, subpath + "labels", labels_);
}

LegLabels
LegLabels::from_hdf5(lasyten::hdf5::Loader& loader,
                     HighFive::Group& /*h5gr*/,
                     std::string const& subpath)
{
    return LegLabels(hdf5_export::load_leg_labels(loader, subpath + "labels"));
}

} // namespace lasyten
