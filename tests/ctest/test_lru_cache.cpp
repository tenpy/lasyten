#include <cassert>
#include <cstddef>
#include <iostream>
#include <string>

#include <lasyten/tools/lru_cache.h>

using namespace lasyten;

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
    assert(cache.stats().entries == 3);
    assert(cache.stats().bytes == 30);
    assert(cache.stats().evictions == 0);

    // a hit makes "a" the most recently used entry, so inserting "d" evicts "b"
    assert(cache.find("a") != nullptr && *cache.find("a") == 1);
    cache.insert("d", 4);
    assert(cache.find("b") == nullptr);
    assert(cache.find("a") != nullptr);
    assert(cache.find("c") != nullptr);
    assert(cache.find("d") != nullptr);
    assert(cache.stats().evictions == 1);
    assert(cache.stats().bytes <= 30);

    // a pointer that was handed out stays valid after the entry is evicted
    auto held = cache.find("a");
    cache.insert("e", 5);
    cache.insert("f", 6);
    cache.insert("g", 7);
    assert(cache.find("a") == nullptr);
    assert(held && *held == 1);

    // replacing an entry does not change the byte count
    cache.insert("g", 70);
    assert(*cache.find("g") == 70);
    assert(cache.stats().bytes <= 30);

    // lowering the budget evicts least recently used entries when the stats are read
    capacity = 10;
    auto stats = cache.stats();
    assert(stats.entries == 1);
    assert(stats.bytes == 10);
    assert(stats.capacity_bytes == 10);
    assert(cache.find("g") != nullptr);

    // an entry larger than the budget is returned but not stored
    capacity = 5;
    auto big = cache.insert("big", 99);
    assert(big && *big == 99);
    assert(cache.find("big") == nullptr);
    assert(cache.stats().rejected == 1);
    assert(cache.stats().entries == 0);

    // a budget of zero disables the cache
    capacity = 0;
    cache.insert("x", 1);
    assert(cache.stats().entries == 0);

    // get_or_make computes only on a miss
    capacity = 100;
    int calls = 0;
    auto make = [&] {
        ++calls;
        return 42;
    };
    assert(*cache.get_or_make("k", make) == 42);
    assert(*cache.get_or_make("k", make) == 42);
    assert(calls == 1);

    // the compute function may use the cache itself
    auto nested = cache.get_or_make(
      "outer", [&] { return *cache.get_or_make("inner", [] { return 7; }) + 1; });
    assert(*nested == 8);
    assert(*cache.find("inner") == 7);

    cache.clear();
    assert(cache.stats().entries == 0 && cache.stats().bytes == 0);
    std::cout << "test_lru_cache passed\n";
    return 0;
}
