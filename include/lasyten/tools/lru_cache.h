#pragma once

#include <cstddef>
#include <cstdint>
#include <functional>
#include <list>
#include <memory>
#include <mutex>
#include <unordered_map>
#include <utility>

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
        std::lock_guard<std::mutex> lock(mutex_);
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
        std::lock_guard<std::mutex> lock(mutex_);
        capacity_ = capacity;
        if (capacity == 0 || size > capacity) {
            ++rejected_;
            evict_to_capacity();
            return ptr;
        }
        auto existing = map_.find(key);
        if (existing != map_.end()) {
            bytes_ -= existing->second.bytes;
            order_.erase(existing->second.position);
            map_.erase(existing);
        }
        order_.push_front(key);
        map_.emplace(order_.front(), Entry{ ptr, size, order_.begin() });
        bytes_ += size;
        evict_to_capacity();
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
        std::lock_guard<std::mutex> lock(mutex_);
        capacity_ = capacity;
        evict_to_capacity();
    }

    /// Remove all entries; the counters are kept (see `reset_counters`).
    void clear()
    {
        std::lock_guard<std::mutex> lock(mutex_);
        map_.clear();
        order_.clear();
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

    // caller holds the lock
    void evict_to_capacity()
    {
        while (bytes_ > capacity_ && !order_.empty()) {
            auto it = map_.find(order_.back());
            bytes_ -= it->second.bytes;
            map_.erase(it);
            order_.pop_back();
            ++evictions_;
        }
    }

    CapacityFn capacity_fn_;
    SizeFn size_fn_;
    std::mutex mutex_;
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

/// Test hook: the block data of a packed file is held in RAM if it is not larger than this many
/// bytes (default 256 MB; checked when the file is opened). Returns the previous value.
std::size_t set_packed_data_in_ram_max_bytes(std::size_t max_bytes);

/// Statistics of the cache for fusion-tree mappings ("tree plans"). Budget:
/// ``tree_plan_cache_mb``.
CacheStats tree_plan_cache_stats();
/// Remove all entries of the tree-plan cache.
void clear_tree_plan_cache();

/// Statistics of the ``fusion_trees::size`` memo (one eighth of ``tree_plan_cache_mb``).
CacheStats tree_size_cache_stats();
/// Remove all entries of the ``fusion_trees::size`` memo.
void clear_tree_size_cache();

} // namespace lasyten
