#include <cstddef>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include <lasyten/tools/lru_cache.h>

using namespace lasyten;

// ``assert`` is compiled out in Release builds (NDEBUG), which would make this test a no-op there.
#define CHECK(cond)                                                                               \
    do {                                                                                          \
        if (!(cond)) {                                                                            \
            std::cerr << __FILE__ << ":" << __LINE__ << ": CHECK failed: " #cond << '\n';         \
            return 1;                                                                             \
        }                                                                                         \
    } while (false)

namespace {

/// A value that uses its cache when it is destroyed (needs the cache lock to be free).
struct Reentrant
{
    ByteLRU<std::string, std::shared_ptr<Reentrant>>* cache = nullptr;
    int* destroyed = nullptr;
    ~Reentrant()
    {
        if (destroyed) {
            ++*destroyed;
        }
        if (cache) {
            (void)cache->stats(); // would deadlock if the value were destroyed under the lock
        }
    }
};

/// Throws on the copy after the next `copies_until_throw` copies (to test exception safety of
/// `insert`, which copies the key into the order list first and into the map node second).
struct ThrowingKey
{
    static inline int copies_until_throw = -1; // -1: never throw
    int id = 0;
    explicit ThrowingKey(int i)
      : id(i)
    {
    }
    ThrowingKey(ThrowingKey const& other)
      : id(other.id)
    {
        if (copies_until_throw == 0) {
            copies_until_throw = -1;
            throw std::runtime_error("copy failed");
        }
        if (copies_until_throw > 0) {
            --copies_until_throw;
        }
    }
    bool operator==(ThrowingKey const& o) const { return id == o.id; }
};

} // namespace

template<>
struct std::hash<ThrowingKey>
{
    std::size_t operator()(ThrowingKey const& k) const noexcept { return std::hash<int>()(k.id); }
};

int
test_lru_cache(int /*argc*/, char** /*args*/)
{
    // every entry is accounted as 10 bytes, the budget is changeable at runtime
    std::size_t capacity = 30;
    ByteLRU<std::string, int> cache(
      [&] { return capacity; }, [](std::string const&, int const&) { return std::size_t{ 10 }; });

    // fill up to the budget: three entries fit
    cache.insert("a", 1);
    cache.insert("b", 2);
    cache.insert("c", 3);
    CHECK(cache.stats().entries == 3);
    CHECK(cache.stats().bytes == 30);
    CHECK(cache.stats().evictions == 0);

    // a hit makes "a" the most recently used entry, so inserting "d" evicts "b"
    CHECK(cache.find("a") != nullptr && *cache.find("a") == 1);
    cache.insert("d", 4);
    CHECK(cache.find("b") == nullptr);
    CHECK(cache.find("a") != nullptr);
    CHECK(cache.find("c") != nullptr);
    CHECK(cache.find("d") != nullptr);
    CHECK(cache.stats().evictions == 1);
    CHECK(cache.stats().bytes <= 30);

    // a pointer that was handed out stays valid after the entry is evicted
    auto held = cache.find("a");
    cache.insert("e", 5);
    cache.insert("f", 6);
    cache.insert("g", 7);
    CHECK(cache.find("a") == nullptr);
    CHECK(held && *held == 1);

    // replacing an entry does not change the byte count
    cache.insert("g", 70);
    CHECK(*cache.find("g") == 70);
    CHECK(cache.stats().bytes <= 30);

    // lowering the budget evicts least recently used entries when the stats are read
    capacity = 10;
    auto stats = cache.stats();
    CHECK(stats.entries == 1);
    CHECK(stats.bytes == 10);
    CHECK(stats.capacity_bytes == 10);
    CHECK(cache.find("g") != nullptr);

    // an entry larger than the budget is returned but not stored
    capacity = 5;
    auto big = cache.insert("big", 99);
    CHECK(big && *big == 99);
    CHECK(cache.find("big") == nullptr);
    CHECK(cache.stats().rejected == 1);
    CHECK(cache.stats().entries == 0);

    // a budget of zero disables the cache
    capacity = 0;
    cache.insert("x", 1);
    CHECK(cache.stats().entries == 0);

    // get_or_make computes only on a miss
    capacity = 100;
    int calls = 0;
    auto make = [&] {
        ++calls;
        return 42;
    };
    CHECK(*cache.get_or_make("k", make) == 42);
    CHECK(*cache.get_or_make("k", make) == 42);
    CHECK(calls == 1);

    // the compute function may use the cache itself
    auto nested = cache.get_or_make(
      "outer", [&] { return *cache.get_or_make("inner", [] { return 7; }) + 1; });
    CHECK(*nested == 8);
    CHECK(*cache.find("inner") == 7);

    cache.clear();
    CHECK(cache.stats().entries == 0 && cache.stats().bytes == 0);

    // lowering the budget is also honoured by `find` (it must not keep serving entries beyond it)
    capacity = 100;
    cache.insert("p", 1);
    cache.insert("q", 2);
    cache.insert("r", 3);
    capacity = 10;
    CHECK(cache.find("p") == nullptr); // least recently used: gone
    CHECK(cache.find("q") == nullptr);
    CHECK(cache.find("r") != nullptr); // the most recently used entry fits
    capacity = 100;
    cache.clear();

    // for_each visits every entry (most recently used first) without changing the order
    cache.insert("u", 1);
    cache.insert("v", 2);
    cache.insert("w", 3);
    std::vector<std::string> seen;
    cache.for_each([&](std::string const& key, int const&) { seen.push_back(key); });
    CHECK((seen == std::vector<std::string>{ "w", "v", "u" }));
    seen.clear();
    cache.for_each([&](std::string const& key, int const&) { seen.push_back(key); });
    CHECK((seen == std::vector<std::string>{ "w", "v", "u" }));

    // erase_if removes the matching entries and keeps the byte count and the order consistent
    auto const removed = cache.erase_if([](std::string const&, int const& v) { return v != 2; });
    CHECK(removed == 2);
    CHECK(cache.stats().entries == 1 && cache.stats().bytes == 10);
    CHECK(cache.find("v") != nullptr && cache.find("u") == nullptr && cache.find("w") == nullptr);
    CHECK(cache.erase_if([](std::string const&, int const&) { return false; }) == 0);
    cache.clear();

    // erase_if destroys the removed values only after the lock was released
    {
        std::size_t cap = 100;
        ByteLRU<std::string, std::shared_ptr<Reentrant>> rcache(
          [&] { return cap; },
          [](std::string const&, std::shared_ptr<Reentrant> const&) { return std::size_t{ 10 }; });
        int destroyed = 0;
        for (auto const* key : { "a", "b", "c" }) {
            auto value = std::make_shared<Reentrant>();
            value->cache = &rcache;
            value->destroyed = &destroyed;
            rcache.insert(key, value);
        }
        CHECK(rcache.erase_if([](std::string const& k, std::shared_ptr<Reentrant> const&) {
            return k != "b";
        }) == 2);
        CHECK(destroyed == 2);
        CHECK(rcache.stats().entries == 1);
        rcache.clear();
    }

    // a failing insertion leaves the cache consistent
    {
        std::size_t cap = 100;
        ByteLRU<ThrowingKey, int> kcache(
          [&] { return cap; }, [](ThrowingKey const&, int const&) { return std::size_t{ 10 }; });
        kcache.insert(ThrowingKey(1), 1);
        ThrowingKey::copies_until_throw = 1; // the list node is created, the map node is not
        bool threw = false;
        try {
            kcache.insert(ThrowingKey(2), 2);
        } catch (std::runtime_error const&) {
            threw = true;
        }
        ThrowingKey::copies_until_throw = -1;
        CHECK(threw);
        CHECK(kcache.stats().entries == 1 && kcache.stats().bytes == 10);
        CHECK(kcache.find(ThrowingKey(1)) != nullptr);
        CHECK(kcache.find(ThrowingKey(2)) == nullptr);
        kcache.insert(ThrowingKey(3), 3); // still usable
        CHECK(kcache.stats().entries == 2);
    }

    std::cout << "test_lru_cache passed\n";
    return 0;
}
