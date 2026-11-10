#pragma once

#include <lasyten/lasyten.h>
#include <lasyten/symmetries/trees.h>

#include <cmath>
#include <complex>
#include <cstddef>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace lasyten {

/// Estimated memory footprint in bytes of a mapping key (used by `SparseMapping::approx_bytes`).
[[nodiscard]] inline std::size_t
approx_bytes_of(FusionTree const& tree)
{
    return tree.approx_bytes();
}

[[nodiscard]] inline std::size_t
approx_bytes_of(std::pair<FusionTree, FusionTree> const& pair)
{
    return pair.first.approx_bytes() + pair.second.approx_bytes();
}

/// A sparse matrix, where the labels of basis states are a structured type, not just int.
///
/// Used in `TreePairMapping` and related objects.
///
/// To represent the mapping ``e_j -> \\sum_i A_{ij} e_i``, we store ``self[j][i] = A_{ij}``.
/// I.e. a single entry ``self[j][i] = a`` represents the contribution ``e_j -> a e_i``.
///
/// Unlike the Python ``dict`` subclass, this wraps nested ``std::unordered_map`` and does not
/// inherit from the container type.
template<typename KT, typename Scalar = complex128>
class SparseMapping
{
  public:
    using Key = KT;
    using value_type = Scalar;
    using Inner = std::unordered_map<KT, Scalar>;
    using Outer = std::unordered_map<KT, Inner>;

    Outer data;

    SparseMapping() = default;
    explicit SparseMapping(Outer data_)
      : data(std::move(data_))
    {
    }

    /// The identity mapping ``e_j -> e_j`` on the given keys
    [[nodiscard]] static SparseMapping from_identity(std::vector<KT> const& keys)
    {
        SparseMapping res;
        res.data.reserve(keys.size());
        for (KT const& i : keys) {
            res.data[i] = Inner{ { i, Scalar(1) } };
        }
        return res;
    }

    /// The composite ``res_{ik} = \\sum_j other_{ij} self{jk}``, such that self acts first.
    ///
    /// I.e. ``pre_compose(self, other) : x ↦ other(self(x)) = (other ∘ self)(x)``.
    [[nodiscard]] SparseMapping pre_compose(SparseMapping const& other) const
    {
        // res[k][i] = sum_j other[j][i] * self[k][j]
        SparseMapping res;
        res.data.reserve(data.size());
        for (auto const& [k, self_k] : data) {
            Inner& res_k = res.data[k];
            for (auto const& [j, self_jk] : self_k) {
                auto it_other_j = other.data.find(j);
                if (it_other_j == other.data.end()) {
                    continue;
                }
                for (auto const& [i, other_ij] : it_other_j->second) {
                    res_k[i] += other_ij * self_jk;
                }
            }
        }
        return res;
    }

    /// Estimated memory footprint in bytes (keys, coefficients and hash-table overhead).
    [[nodiscard]] std::size_t approx_bytes() const
    {
        constexpr std::size_t node_overhead = 32;
        std::size_t total = sizeof(*this) + data.bucket_count() * sizeof(void*);
        for (auto const& [j, self_j] : data) {
            total += approx_bytes_of(j) + sizeof(Inner) + node_overhead +
                     self_j.bucket_count() * sizeof(void*);
            for (auto const& [i, value] : self_j) {
                total += approx_bytes_of(i) + sizeof(value) + node_overhead;
            }
        }
        return total;
    }

    /// The idcs ``i`` for which there are entries ``self_{ij} = self[j][i]`` set.
    [[nodiscard]] std::unordered_set<KT> nonzero_rows() const
    {
        std::unordered_set<KT> rows;
        for (auto const& [j, self_j] : data) {
            (void)j;
            for (auto const& [i, a] : self_j) {
                (void)a;
                rows.insert(i);
            }
        }
        return rows;
    }

    /// The idcs ``j`` for which there are entries ``self_{ij} = self[j][i]`` set.
    [[nodiscard]] std::unordered_set<KT> nonzero_cols() const
    {
        std::unordered_set<KT> cols;
        cols.reserve(data.size());
        for (auto const& [j, self_j] : data) {
            (void)self_j;
            cols.insert(j);
        }
        return cols;
    }

    /// Remove small contributions with ``abs(coefficient) <= tol`` in-place.
    ///
    /// Returns nothing (Python returns ``self`` for chaining).
    void prune(float64 tol)
    {
        for (auto& [j, self_j] : data) {
            (void)j;
            Inner kept;
            kept.reserve(self_j.size());
            for (auto const& [i, a] : self_j) {
                if (std::abs(a) > tol) {
                    kept.emplace(i, a);
                }
            }
            self_j = std::move(kept);
        }
    }

    Inner& operator[](KT const& j) { return data[j]; }
    Inner const& at(KT const& j) const { return data.at(j); }

    [[nodiscard]] bool contains(KT const& j) const { return data.find(j) != data.end(); }
    [[nodiscard]] std::size_t size() const { return data.size(); }
    [[nodiscard]] bool empty() const { return data.empty(); }

    auto begin() { return data.begin(); }
    auto end() { return data.end(); }
    auto begin() const { return data.begin(); }
    auto end() const { return data.end(); }
};

/// An identity mapping with same call structure as `SparseMapping`.
template<typename KT, typename Scalar = complex128>
class IdentityMapping
{
  public:
    using Key = KT;
    using value_type = Scalar;

    std::unordered_set<KT> keys;

    /// Estimated memory footprint in bytes.
    [[nodiscard]] std::size_t approx_bytes() const
    {
        std::size_t total = sizeof(*this) + keys.bucket_count() * sizeof(void*);
        for (KT const& k : keys) {
            total += approx_bytes_of(k) + 32;
        }
        return total;
    }

    IdentityMapping() = default;
    explicit IdentityMapping(std::vector<KT> const& keys_)
      : keys(keys_.begin(), keys_.end())
    {
    }
    explicit IdentityMapping(std::unordered_set<KT> keys_)
      : keys(std::move(keys_))
    {
    }

    /// The composite ``res_{ik} = \\sum_j other_{ij} self_{jk}``, such that self acts first.
    [[nodiscard]] SparseMapping<KT, Scalar> pre_compose(
      SparseMapping<KT, Scalar> const& other) const
    {
        // res[k] = other[k] for k in self.keys
        SparseMapping<KT, Scalar> res;
        for (KT const& k : keys) {
            auto it = other.data.find(k);
            if (it == other.data.end()) {
                continue;
            }
            res.data[k] = it->second;
        }
        return res;
    }

    [[nodiscard]] std::unordered_set<KT> nonzero_rows() const { return keys; }
    [[nodiscard]] std::unordered_set<KT> nonzero_cols() const { return keys; }

    /// No-op (identity has no coefficients to prune).
    void prune(float64 /*tol*/) {}
};

/// Sparse mapping with `FusionTree` keys and ``complex128`` coefficients.
using SparseMappingFusionTree = SparseMapping<FusionTree, complex128>;
/// Sparse mapping with ``(FusionTree, FusionTree)`` keys (tree pairs) and ``complex128``
/// coefficients.
using SparseMappingFusionTreePair = SparseMapping<std::pair<FusionTree, FusionTree>, complex128>;
using IdentityMappingFusionTree = IdentityMapping<FusionTree, complex128>;
using IdentityMappingFusionTreePair =
  IdentityMapping<std::pair<FusionTree, FusionTree>, complex128>;

} // namespace lasyten
