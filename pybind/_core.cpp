#include "py_lasyten_nanobind.h"

using namespace lasyten;

NB_MODULE(_core, m)
{
    m.doc() = "LASyTen python bindings using nanobind"; // optional module docstring

    bind_tools(m);
    bind_cost_polynomials(m);
    bind_config(m);
    bind_block_backend(m);

    // Declare shells for types that appear in earlier binders' signatures (stubgen).
    declare_block_inds(m);
    declare_tensor_backend(m);
    declare_tensors_labels(m);
    declare_tensors_vector_like(m);
    declare_tensors_tensor(m);
    declare_tensors_symmetric_tensor(m);
    declare_tensors_diagonal_tensor(m);
    declare_tensors_mask(m);

    bind_symmetries(m);
    bind_mappings(m); // after FusionTree registration

    bind_tensor_backend(m);
    bind_no_symmetry_backend(m);
    bind_block_inds(m);
    bind_abelian_backend_data(m);
    bind_abelian_backend(m);
    bind_fusion_tree_data(m);
    bind_fusion_tree_backend(m);
    bind_fusion_tree_mapping(m);
    bind_backend_factory(m);

    bind_tensors_labels(m);
    bind_tensors_vector_like(m);
    bind_tensors_tensor(m);
    bind_tensors_direct_sum(m);
    bind_tensors_symmetric_tensor(m);
    bind_tensors_diagonal_tensor(m);
    bind_tensors_mask(m);
    bind_tensors_charged_tensor(m);
    bind_tensors_hidden_leg_tensor(m);
    bind_tensors_helpers(m);
    bind_tensors_constructors(m);
    bind_tensors_ops_elementwise(m);
    bind_tensors_ops_algebra(m);
    bind_tensors_ops_legs(m);
    bind_tensors_decompositions(m);
    bind_tensors_sparse(m);
    bind_tensors_krylov_based(m);
    bind_tensors_planar(m);
    bind_models_degrees_of_freedom(m);
    bind_models_sites(m);
    bind_models_couplings(m);
}
