# #####################################################################
# ## ADD SUB-DIRECTORIES
# #####################################################################

# liblasyten sources are split in two lists (convert_nanobind.md, Phase 5):
#
#   LASYTEN_PYTHON_FREE_SOURCES  compile with LASYTEN_HAS_PYTHON=0, i.e. against the opaque
#                                ``py::`` stubs of include/lasyten/detail/python_api.h
#                                (Eigen backend, dtypes, config, symmetries, tools, ...). Always built.
#   LASYTEN_PYTHON_API_SOURCES   still call ``.cast<T>()`` / ``py::array`` / ``py::cast`` /
#                                ``py::isinstance`` in their implementations (tensors, backends,
#                                Sector/FusionSymbol numpy glue). Built only with
#                                LASYTEN_WITH_PYTHON=ON. This list shrinks as Phase 4/5 proceeds;
#                                the root CMakeLists.txt prints it as a status message for OFF.

set(LASYTEN_PYTHON_FREE_SOURCES
  backends/block_inds.cpp
  backends/fusion_tree_permute.cpp
  block_backend/block_backend.cpp
  block_backend/dtypes.cpp
  block_backend/eigen.cpp
  config.cpp
  symmetries/base_symmetry.cpp
  symmetries/symmetry_factor.cpp
  symmetries/symmetry.cpp
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
  tensors/labels.cpp
  tensors/vector_like.cpp
  tensors/direct_sum.cpp
  tensors/hidden_leg_tensor.cpp
  tensors/ops_elementwise.cpp
  tensors/ops_algebra.cpp
  tensors/ops_legs.cpp
  tensors/sparse.cpp
  tensors/planar.cpp
  tools.cpp
  tools/cost_polynomials.cpp
  tools/mappings.cpp
  tools/misc.cpp
  tools/hdf5_export_blocks.cpp
  symmetries/fusion_symbol.cpp
  # compile without Python (numpy-facing parts are under #if LASYTEN_HAS_PYTHON)
  symmetries/sector_numpy.cpp
  block_backend/block_backend_legs.cpp
  tools/hdf5_export.cpp
)

set(LASYTEN_PYTHON_API_SOURCES
  backends/backend_factory.cpp
  backends/tensor_backend.cpp
  backends/no_symmetry.cpp
  backends/block_inds_numpy.cpp
  backends/abelian.cpp
  backends/fusion_tree_backend.cpp
  backends/fusion_tree_mapping.cpp
  symmetries/spaces.cpp
  tensors/tensor.cpp
  tensors/symmetric_tensor.cpp
  tensors/diagonal_tensor.cpp
  tensors/mask.cpp
  tensors/charged_tensor.cpp
  tensors/helpers.cpp
  tensors/constructors.cpp
  tensors/decompositions.cpp
  tensors/krylov_based.cpp
)

target_sources(lasyten PRIVATE ${LASYTEN_PYTHON_FREE_SOURCES})

if(LASYTEN_WITH_PYTHON)
  target_sources(lasyten PRIVATE ${LASYTEN_PYTHON_API_SOURCES})
endif()

# Python-dispatch backends (convert_nanobind.md leftover #5): NumpyBlockBackend (needs a Python
# interpreter for numpy arrays) and ArrayApiBlockBackend. Transitional: EigenBlockBackend is the
# preferred CPU backend. Drop with -DLASYTEN_WITH_NUMPY_BACKEND=OFF (forced OFF without Python).
if(LASYTEN_WITH_NUMPY_BACKEND)
  target_sources(lasyten
    PRIVATE
    block_backend/array_api.cpp
    block_backend/numpy.cpp
  )
endif()

# TorchBlockBackend (GPU / ATen). torch.cpp still exposes py::array, hence Python-gated.
if(LASYTEN_WITH_PYTHON)
  target_sources(lasyten PRIVATE block_backend/torch.cpp)
endif()

# Test-only _SU2 (pulls numpy) and the models, which store py::array / py::object members.
if(LASYTEN_WITH_PYTHON)
  target_sources(lasyten
    PRIVATE
    symmetries/factors/su2.cpp
    models/degrees_of_freedom.cpp
    models/sites.cpp
    models/couplings.cpp
  )
endif()

if(NOT LASYTEN_WITH_PYTHON)
  list(JOIN LASYTEN_PYTHON_API_SOURCES "\n    " _lasyten_skipped_api)
  list(LENGTH LASYTEN_PYTHON_API_SOURCES _lasyten_n_skipped_api)
  list(LENGTH LASYTEN_PYTHON_FREE_SOURCES _lasyten_n_free)
  message(STATUS
    "LASYTEN_WITH_PYTHON=OFF: liblasyten = ${_lasyten_n_free} sources "
    "(LASYTEN_PYTHON_FREE_SOURCES) + eigen backend; ${_lasyten_n_skipped_api} sources still need "
    "Python and are skipped (LASYTEN_PYTHON_API_SOURCES):\n    ${_lasyten_skipped_api}")
endif()
