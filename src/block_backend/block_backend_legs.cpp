// BlockBackend functionality that needs the `Leg` classes (symmetries/spaces.cpp), split out of
// block_backend.cpp so that the Eigen block backend does not link against the space classes.

#include <lasyten/block_backend/block_backend.h>
#include <lasyten/symmetries/spaces.h>

#include <stdexcept>
#include <vector>

namespace lasyten {

BlockPtr
BlockBackend::apply_basis_perm(const BlockCPtr& block, const std::vector<LegCPtr>& legs, bool inv)
{
    std::vector<std::vector<int64>> perms;
    perms.reserve(legs.size());
    for (LegCPtr const& leg : legs) {
        if (!leg) {
            throw std::invalid_argument("apply_basis_perm: leg must not be None");
        }
        auto const& perm_vec = inv ? leg.inverse_basis_perm() : leg.basis_perm();
        perms.emplace_back(perm_vec.begin(), perm_vec.end());
    }
    return apply_axis_permutations(block, perms);
}

} // namespace lasyten
