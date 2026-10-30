#pragma once

#include <cstddef>
#include <cstdint>
#include <functional>
#include <list>
#include <memory>
#include <mutex>
#include <unordered_map>
#include <utility>
#include <vector>

namespace lasyten {

/// Counters and sizes of a `ByteLRU`; see `ByteLRU::stats`.
struct CacheStats
{
    /// Lookups that found an entry.
    std::uint64_t hits = 0;
    /// Lookups that did not find an entry.
    std::uint64_t misses = 0;
    /// Entries removed to make room (or because the capacity was lowered).
    std::uint64_t evictions = 0;
    /// Insertions that were not cached (cache disabled, or a single entry larger than the cap).
    std::uint64_t rejected = 0;
    /// Number of entries currently stored.
    std::size_t entries = 0;
    /// Estimated total size of the stored entries in bytes.
    std::size_t bytes = 0;
    /// Current capacity in bytes (0 means disabled).
    std::size_t capacity_bytes = 0;
};

/// Thread-safe least-recently-used cache with a byte budget instead of an entry limit.
///
/// The size of an entry is estimated by a user-supplied functor, the capacity is queried from a
/// second functor on every insertion (and on `trim` / `stats`), so a budget stored in the global
/// config can change at runtime: lowering it evicts least-recently-used entries.
/// Values are handed out as ``shared_ptr<const Value>``, so evicting an entry never invalidates a
/// pointer that a caller still holds.
///
/// The lock is *not* held while a missing value is computed (see `get_or_make`), so the compute
/// function may itself use other caches (or even this one).
template<class Key, class Value, class Hash = std::hash<Key>>
class ByteLRU
{
  public:
    using ValuePtr = std::shared_ptr<Value const>;
    using CapacityFn = std::function<std::size_t()>;
    using SizeFn = std::function<std::size_t(Key const&, Value const&)>;

    /// \param capacity_bytes Returns the current budget in bytes; 0 disables the cache.
    /// \param entry_bytes Estimated size of one entry in bytes (including key and node overhead).
    ByteLRU(CapacityFn capacity_bytes, SizeFn entry_bytes)
      : capacity_fn_(std::move(capacity_bytes))
      , size_fn_(std::move(entry_bytes))
    {
    }

    /// Look up `key`; a hit moves the entry to the front. Returns null on a miss.
    ValuePtr find(Key const& key)
    {
        std::size_t const capacity = capacity_fn_();
        std::vector<ValuePtr> dropped; // declared before the lock: destroyed after it is released
        std::lock_guard<std::mutex> lock(mutex_);
        if (capacity < capacity_) { // the budget was lowered since the last insertion
            capacity_ = capacity;
            evict_to_capacity(dropped);
        }
        auto it = map_.find(key);
        if (it == map_.end()) {
            ++misses_;
            return nullptr;
        }
        order_.splice(order_.begin(), order_, it->second.position);
        ++hits_;
        return it->second.value;
    }

    /// Store `value` under `key` (replacing an existing entry) and evict least-recently-used
    /// entries until the budget is respected. Returns the (possibly uncached) value.
    ValuePtr insert(Key key, Value value)
    {
        auto ptr = std::make_shared<Value const>(std::move(value));
        std::size_t const capacity = capacity_fn_();
        std::size_t const size = size_fn_(key, *ptr);
        std::vector<ValuePtr> dropped; // declared before the lock: destroyed after it is released
        std::lock_guard<std::mutex> lock(mutex_);
        capacity_ = capacity;
        if (capacity == 0 || size > capacity) {
            ++rejected_;
            evict_to_capacity(dropped);
            return ptr;
        }
        auto existing = map_.find(key);
        if (existing != map_.end()) {
            bytes_ -= existing->second.bytes;
            order_.erase(existing->second.position);
            dropped.push_back(std::move(existing->second.value));
            map_.erase(existing);
        }
        order_.push_front(key);
        try {
            map_.emplace(order_.front(), Entry{ ptr, size, order_.begin() });
        } catch (...) {
            order_.pop_front(); // keep list and map consistent if the node cannot be created
            throw;
        }
        bytes_ += size;
        evict_to_capacity(dropped);
        return ptr;
    }

    /// Return the cached value for `key`, or compute it with `make()`, cache it and return it.
    template<class Make>
    ValuePtr get_or_make(Key const& key, Make&& make)
    {
        if (auto hit = find(key)) {
            return hit;
        }
        return insert(key, make());
    }

    /// Apply the current capacity (evicting if it was lowered since the last insertion).
    void trim()
    {
        std::size_t const capacity = capacity_fn_();
        std::vector<ValuePtr> dropped; // declared before the lock: destroyed after it is released
        std::lock_guard<std::mutex> lock(mutex_);
        capacity_ = capacity;
        evict_to_capacity(dropped);
    }

    /// Call ``visit(key, value)`` for every stored entry (most recently used first), without
    /// changing the order. The lock is held, so `visit` must not use this cache.
    template<class Visit>
    void for_each(Visit&& visit) const
    {
        std::lock_guard<std::mutex> lock(mutex_);
        for (auto const& key : order_) {
            visit(key, *map_.at(key).value);
        }
    }

    /// Remove every entry for which ``pred(key, value)`` is true; returns how many were removed.
    /// The removed values are destroyed after the lock has been released, because destroying a
    /// value may run arbitrary code (e.g. close files).
    template<class Pred>
    std::size_t erase_if(Pred&& pred)
    {
        std::vector<ValuePtr> removed; // declared before the lock: destroyed after it is released
        std::lock_guard<std::mutex> lock(mutex_);
        for (auto it = map_.begin(); it != map_.end();) {
            if (pred(it->first, *it->second.value)) {
                bytes_ -= it->second.bytes;
                order_.erase(it->second.position);
                removed.push_back(std::move(it->second.value));
                it = map_.erase(it);
            } else {
                ++it;
            }
        }
        evictions_ += removed.size();
        return removed.size();
    }

    /// Remove all entries; the counters are kept (see `reset_counters`).
    void clear()
    {
        // the values are destroyed after the lock is released (destroying one may run arbitrary
        // code)
        decltype(map_) old_map;
        decltype(order_) old_order;
        std::lock_guard<std::mutex> lock(mutex_);
        old_map.swap(map_);
        old_order.swap(order_);
        bytes_ = 0;
    }

    void reset_counters()
    {
        std::lock_guard<std::mutex> lock(mutex_);
        hits_ = misses_ = evictions_ = rejected_ = 0;
    }

    CacheStats stats()
    {
        trim();
        std::lock_guard<std::mutex> lock(mutex_);
        return CacheStats{ hits_, misses_, evictions_, rejected_, map_.size(), bytes_, capacity_ };
    }

  private:
    struct Entry
    {
        ValuePtr value;
        std::size_t bytes;
        typename std::list<Key>::iterator position;
    };

    // caller holds the lock; the evicted values are handed to `dropped`, which the caller destroys
    // after releasing the lock
    void evict_to_capacity(std::vector<ValuePtr>& dropped)
    {
        while (bytes_ > capacity_ && !order_.empty()) {
            auto it = map_.find(order_.back());
            bytes_ -= it->second.bytes;
            dropped.push_back(std::move(it->second.value));
            map_.erase(it);
            order_.pop_back();
            ++evictions_;
        }
    }

    CapacityFn capacity_fn_;
    SizeFn size_fn_;
    mutable std::mutex mutex_;
    std::list<Key> order_; // most recently used first
    std::unordered_map<Key, Entry, Hash> map_;
    std::size_t bytes_ = 0;
    std::size_t capacity_ = 0;
    std::uint64_t hits_ = 0;
    std::uint64_t misses_ = 0;
    std::uint64_t evictions_ = 0;
    std::uint64_t rejected_ = 0;
};

/// Statistics of the process-wide RAM cache for SU(N) F symbols, R symbols and fusion outcomes.
/// The budget is the config option ``su_n_symbol_cache_mb``.
CacheStats symbol_cache_stats();
/// Remove all entries of the SU(N) symbol cache.
void clear_symbol_cache();

/// Memory held by open SU(N) symbol files in the packed layout (see ``SUN``): the index of every
/// file and, for small files, its block data. This memory is *not* part of the symbol cache
/// budget.
struct PackedFileStats
{
    /// Number of open packed F / R files.
    std::uint64_t files = 0;
    /// Bytes of the indices (labels, shapes, offsets) held in RAM.
    std::uint64_t index_bytes = 0;
    /// Bytes of block data held in RAM.
    std::uint64_t data_bytes_in_ram = 0;
};

PackedFileStats packed_file_stats();

/// Statistics of the cache for fusion-tree mappings ("tree plans"). Budget:
/// ``tree_plan_cache_mb``.
CacheStats tree_plan_cache_stats();
/// Remove all entries of the tree-plan cache.
void clear_tree_plan_cache();
/// Diagnostics: for every cached plan and every symmetry it refers to, the strong count of that
/// symmetry and the number of those references that the cached plan itself holds.
std::vector<std::vector<std::pair<long, long>>> tree_plan_cache_debug();

/// Statistics of the ``fusion_trees::size`` memo. Budget: ``tree_size_memo_mb``.
CacheStats tree_size_cache_stats();
/// Remove all entries of the ``fusion_trees::size`` memo.
void clear_tree_size_cache();

} // namespace lasyten
