# #####################################################################
# ## ADD SUB-DIRECTORIES
# #####################################################################

# Sources that do not *own* a Python-only backend implementation.
# Many still compile against py:: APIs via public headers (see LASYTEN_WITH_PYTHON).
target_sources(lasyten
  PRIVATE
  backends/tensor_backend.cpp
  backends/no_symmetry.cpp
  backends/block_inds.cpp
  backends/block_inds_numpy.cpp
  backends/abelian.cpp
  backends/fusion_tree_backend.cpp
  backends/fusion_tree_mapping.cpp
  backends/fusion_tree_permute.cpp
  backends/backend_factory.cpp
  block_backend/block_backend.cpp
  block_backend/dtypes.cpp
  config.cpp
  symmetries/base_symmetry.cpp
  symmetries/symmetry_factor.cpp
  symmetries/symmetry.cpp
  symmetries/sector_numpy.cpp
  symmetries/fusion_symbol.cpp
  symmetries/sector_array.cpp
  symmetries/group.cpp
  symmetries/abelian_group.cpp
  symmetries/factors/no_symmetry.cpp
  symmetries/factors/u1.cpp
  symmetries/factors/zn.cpp
  symmetries/factors/sun.cpp
  symmetries/factors/sun_packed.cpp
  symmetries/factors/fermion_number.cpp
  symmetries/factors/fermion_parity.cpp
  symmetries/factors/zn_anyon_category.cpp
  symmetries/factors/zn_anyon_category2.cpp
  symmetries/factors/quantum_double_zn_anyon_category.cpp
  symmetries/factors/toric_code_category.cpp
  symmetries/factors/fibonacci_anyon_category.cpp
  symmetries/factors/ising_anyon_category.cpp
  symmetries/factors/su2_k_anyon_category.cpp
  symmetries/factors/su3_3_anyon_category.cpp
  symmetries/trees.cpp
  symmetries/spaces.cpp
  tensors/labels.cpp
  tensors/vector_like.cpp
  tensors/tensor.cpp
  tensors/direct_sum.cpp
  tensors/symmetric_tensor.cpp
  tensors/diagonal_tensor.cpp
  tensors/mask.cpp
  tensors/charged_tensor.cpp
  tensors/hidden_leg_tensor.cpp
  tensors/helpers.cpp
  tensors/constructors.cpp
  tensors/ops_elementwise.cpp
  tensors/ops_algebra.cpp
  tensors/ops_legs.cpp
  tensors/decompositions.cpp
  tensors/sparse.cpp
  tensors/krylov_based.cpp
  tensors/planar.cpp
  tools.cpp
  tools/cost_polynomials.cpp
  tools/mappings.cpp
  tools/misc.cpp
  tools/hdf5_export.cpp
)

# Python / pybind11-dependent TUs: NumpyBlockBackend (requires a Python
# interpreter for numpy arrays), Torch/Array-API backends that expose py::array,
# test-only _SU2 (pulls numpy), and models that store py::array / py::object.
if(LASYTEN_WITH_PYTHON)
  target_sources(lasyten
    PRIVATE
    block_backend/array_api.cpp
    block_backend/numpy.cpp
    block_backend/torch.cpp
    symmetries/factors/su2.cpp
    models/degrees_of_freedom.cpp
    models/sites.cpp
    models/couplings.cpp
  )
endif()
