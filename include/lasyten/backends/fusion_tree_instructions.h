#pragma once

#include <cstdint>
#include <variant>
#include <vector>

namespace lasyten {

/// Instruction to braid two neighboring legs (see docs/cpp/backends/fusion_tree_mapping.rst).
struct BraidInstruction
{
    bool codomain = false;
    int64 idx = 0;
    bool overbraid = false;

    bool operator==(BraidInstruction const&) const = default;
};

struct BendInstruction
{
    bool bend_down = false;

    bool operator==(BendInstruction const&) const = default;
};

struct TwistInstruction
{
    bool codomain = false;
    std::vector<int64> idcs;
    bool overtwist = false;

    bool operator==(TwistInstruction const&) const = default;
};

using Instruction = std::variant<BraidInstruction, BendInstruction, TwistInstruction>;

} // namespace lasyten
