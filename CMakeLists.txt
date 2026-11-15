message(STATUS "")
message(STATUS "------------------------------------------------------------------------")
message(STATUS " Project lasyten, A Cross-section of Python & C++, Tensor network library ")
message(STATUS "------------------------------------------------------------------------")
message(STATUS "")

# #####################################################################
# ## CMAKE and CXX VERSION
# #####################################################################
cmake_minimum_required(VERSION 3.27)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Prefer the conda-forge compiler (headers + libstdc++) that conda Python
# actually loads. A newer system g++ (e.g. Fedora GCC 16) emits symbols
# such as GLIBCXX_3.4.36 that conda's libstdc++ (GCC 15) does not provide;
# `import lasyten` then fails because Python has already loaded the older
# libstdc++. Install with: conda install -c conda-forge cxx-compiler
if(NOT CMAKE_CXX_COMPILER AND DEFINED ENV{CONDA_PREFIX})
  foreach(_lasyten_cxx
      "$ENV{CONDA_PREFIX}/bin/c++"
      "$ENV{CONDA_PREFIX}/bin/g++")
    if(EXISTS "${_lasyten_cxx}")
      set(CMAKE_CXX_COMPILER "${_lasyten_cxx}")
      break()
    endif()
  endforeach()
  if(NOT CMAKE_CXX_COMPILER)
    file(GLOB _lasyten_conda_gxx "$ENV{CONDA_PREFIX}/bin/*-conda-*-g++")
    if(_lasyten_conda_gxx)
      list(GET _lasyten_conda_gxx 0 CMAKE_CXX_COMPILER)
    endif()
  endif()
endif()

message(STATUS " Generator: ${CMAKE_GENERATOR}")
message(STATUS " Build Target: ${CMAKE_SYSTEM_NAME}-${CMAKE_SYSTEM_PROCESSOR}")
message(STATUS " Installation Prefix: ${CMAKE_INSTALL_PREFIX}")

# #####################################################################
# ## Version
# #####################################################################

# Scikit-build-core sets version
if(NOT SKBUILD_PROJECT_VERSION)
  set(SKBUILD_PROJECT_VERSION 0.0.1)
endif()

# #####################################################################
# ## COMMAND LINE OPTIONS
# #####################################################################
option(BUILD_DOC "Build API documentation" OFF)
option(LINTING "Use clang-tidy for linting" OFF)
option(BUILD_TESTING "Build ctest tests" ON)
option(LASYTEN_WITH_HDF5 "Link hdf5_io_core for native HDF5 I/O" ON)
# When ON (default): build Numpy/Torch/Array-API backends, models, _SU2, and
# the nanobind ``_core`` extension. NumpyBlockBackend requires a live Python
# interpreter (extension or an embedding program) — it is not a pure-C++ backend.
# When OFF: the Python-owned TUs are excluded from liblasyten; ``_core`` is
# not built; Python / nanobind / Torch are not looked for. Public headers
# still mention ``py::`` (see include/lasyten/detail/python_api.h); in this mode ``py::`` is a
# set of opaque stand-in types so declarations compile, but anything that really needs Python
# does not. Builds Eigen + the symmetry stack and two Python-free ctests (no libpython). See
# convert_nanobind.md (Phase 5): the status message printed at configure time lists what is skipped.
option(LASYTEN_WITH_PYTHON
  "Include Python/nanobind-dependent liblasyten sources and build _core" ON)
# nanobind Stable ABI (abi3): ``_core`` and liblasyten are compiled with Py_LIMITED_API (3.12) and
# linked against nanobind-static-abi3, so one ``_core.abi3.so`` works for all Python >= 3.12.
# Only effective with LASYTEN_WITH_PYTHON=ON and a CPython >= 3.12 (otherwise silently OFF).
# pybind11 can not be used with Py_LIMITED_API; liblasyten does not use it anymore.
option(LASYTEN_STABLE_ABI "Build _core against the Python Stable ABI (abi3, Python >= 3.12)" ON)
# The Python-dispatch CPU backends (NumpyBlockBackend, ArrayApiBlockBackend) are transitional
# (convert_nanobind.md leftover #5): EigenBlockBackend ("eigen") is the preferred CPU backend and
# TorchBlockBackend the GPU one. Turning this OFF drops numpy.cpp / array_api.cpp (and their
# bindings); "cpu" then resolves to Eigen. Forced OFF without LASYTEN_WITH_PYTHON.
include(CMakeDependentOption)
cmake_dependent_option(LASYTEN_WITH_NUMPY_BACKEND
  "Build the transitional NumpyBlockBackend / ArrayApiBlockBackend (Python-dispatch)"
  ON "LASYTEN_WITH_PYTHON" OFF)

# Preprocessor switches mirrored into the public headers (include/lasyten/detail/python_api.h).
if(LASYTEN_WITH_PYTHON)
  set(LASYTEN_HAS_PYTHON_DEF 1)
  set(LASYTEN_WITH_TORCH_BACKEND_DEF 1)
else()
  set(LASYTEN_HAS_PYTHON_DEF 0)
  set(LASYTEN_WITH_TORCH_BACKEND_DEF 0)
endif()
if(LASYTEN_WITH_NUMPY_BACKEND)
  set(LASYTEN_WITH_NUMPY_BACKEND_DEF 1)
else()
  set(LASYTEN_WITH_NUMPY_BACKEND_DEF 0)
endif()


# #####################################################################
# ## PROJECT
# #####################################################################
project(
    lasyten
    VERSION ${SKBUILD_PROJECT_VERSION}
    LANGUAGES CXX
)

include(CMakePrintHelpers)
cmake_print_variables(PROJECT_SOURCE_DIR)
cmake_print_variables(CMAKE_CXX_STANDARD)

# #####################################################################
# ## find python / nanobind (only with LASYTEN_WITH_PYTHON)
# #####################################################################
# With LASYTEN_WITH_PYTHON=OFF none of these is required: no Development.Module /
# Development.Embed, no nanobind. (An interpreter is searched for optionally, only
# so that tools like the stub generator could use it; failing to find one is fine.)

if(LASYTEN_WITH_PYTHON)
  set(Python_FIND_VIRTUALENV ONLY)
  set(Python_FIND_STRATEGY LOCATION)

  # Stable ABI needs CPython >= 3.12 (nanobind's linked abi3 mode); decide before asking
  # FindPython for the Development.SABIModule component.
  find_package(Python REQUIRED COMPONENTS Interpreter)
  if(LASYTEN_STABLE_ABI AND (Python_VERSION VERSION_LESS 3.12 OR NOT Python_INTERPRETER_ID STREQUAL "Python"))
    message(STATUS "LASYTEN_STABLE_ABI: needs CPython >= 3.12 (found ${Python_INTERPRETER_ID} "
      "${Python_VERSION}); building a version-specific extension instead")
    set(LASYTEN_STABLE_ABI OFF)
  endif()
  if(LASYTEN_STABLE_ABI)
    find_package(Python REQUIRED COMPONENTS Interpreter Development.Module Development.Embed
      Development.SABIModule)
    set(LASYTEN_NB_STABLE_ABI_ARG STABLE_ABI)
    set(LASYTEN_NB_LIB_NAME nanobind-static-abi3)
  else()
    find_package(Python REQUIRED COMPONENTS Interpreter Development.Module Development.Embed)
    set(LASYTEN_NB_STABLE_ABI_ARG "")
    set(LASYTEN_NB_LIB_NAME nanobind-static)
  endif()
  message(STATUS "LASYTEN_STABLE_ABI: ${LASYTEN_STABLE_ABI}")

  # nanobind builds the ``_core`` extension and provides the Python object API (``nb::``) that
  # liblasyten uses as well (include/lasyten/detail/python_api.h). pybind11 is not used.
  if(NOT nanobind_DIR AND NOT nanobind_ROOT)
    execute_process(
      COMMAND "${Python_EXECUTABLE}" -m nanobind --cmake_dir
      OUTPUT_VARIABLE _lasyten_nanobind_cmake_dir
      OUTPUT_STRIP_TRAILING_WHITESPACE
      RESULT_VARIABLE _lasyten_nanobind_rc
      ERROR_QUIET)
    if(_lasyten_nanobind_rc EQUAL 0 AND _lasyten_nanobind_cmake_dir)
      list(APPEND CMAKE_PREFIX_PATH "${_lasyten_nanobind_cmake_dir}")
    endif()
  endif()
  find_package(nanobind CONFIG REQUIRED)
  # Build libnanobind once and link it into liblasyten: library TUs convert bound
  # C++ types via nanobind, so every executable that links lasyten (ctests, examples) needs
  # the nanobind symbols too. The same target is what nanobind_add_module(_core [STABLE_ABI])
  # links, so the module, the library and the test programs share one libnanobind.
  nanobind_build_library(${LASYTEN_NB_LIB_NAME})
  cmake_print_variables(Python_VERSION)
  cmake_print_variables(Python_INCLUDE_DIRS)
else()
  find_package(Python QUIET COMPONENTS Interpreter)
  message(STATUS "LASYTEN_WITH_PYTHON=OFF: not requiring Python Development.Module/Embed, "
    "nanobind or Torch")
endif()

# #####################################################################
# ## find Eigen3 + LAPACK(E) (native EigenBlockBackend)
# #####################################################################
# EigenBlockBackend (src/block_backend/eigen.cpp) is always built, independent of
# LASYTEN_WITH_PYTHON: Eigen for products / matrix functions, LAPACKE for
# SVD / QR / eigh / eig. conda-forge ships Eigen (header-only) and either OpenBLAS or
# MKL as liblapack/libblas; both provide the LAPACKE symbols.
find_package(Eigen3 REQUIRED NO_MODULE)
find_package(LAPACK REQUIRED)
find_path(LASYTEN_LAPACKE_INCLUDE_DIR lapacke.h
  HINTS "$ENV{CONDA_PREFIX}" "${CMAKE_PREFIX_PATH}"
  PATH_SUFFIXES include include/openblas openblas
  REQUIRED)
cmake_print_variables(Eigen3_VERSION)
cmake_print_variables(LAPACK_LIBRARIES)
cmake_print_variables(LASYTEN_LAPACKE_INCLUDE_DIR)

if(NOT LASYTEN_WITH_PYTHON)
  # NOTE: keep this in sync with src/CMakeLists.txt (LASYTEN_PYTHON_API_SOURCES) and
  # docs/cpp_conversion/convert_nanobind.md (Phase 5).
  message(STATUS "")
  message(STATUS "LASYTEN_WITH_PYTHON=OFF: Python-free liblasyten (no Python, nanobind, numpy, PyTorch)")
  message(STATUS "  Builds: liblasyten with EigenBlockBackend (Eigen + LAPACK), dtypes, config, the")
  message(STATUS "  symmetry stack (Sector, SymmetryFactor / Symmetry, all factors, fusion trees), Space / Leg,")
  message(STATUS "  Tensor classes + NoSymmetryBackend (tdot / compose / svd / qr / lq on Eigen), HDF5")
  message(STATUS "  (de)serialization of those, and the Python-free ctests (test_eigen_block_backend_no_python,")
  message(STATUS "  test_symmetric_blocks_no_python, test_tensor_no_python); none of them links libpython.")
  message(STATUS "  Not built (LASYTEN_PYTHON_API_SOURCES in src/CMakeLists.txt, py::-typed in their")
  message(STATUS "  implementation): Abelian / FusionTree backends, krylov, models,")
  message(STATUS "  Numpy / ArrayApi / Torch block backends, _core and the ctests that embed an interpreter.")
  message(STATUS "  Details: docs/cpp_conversion/convert_nanobind.md (Phase 5 / leftover #1)")
  message(STATUS "")
endif()
if(LASYTEN_WITH_PYTHON AND NOT LASYTEN_WITH_NUMPY_BACKEND)
  message(STATUS "LASYTEN_WITH_NUMPY_BACKEND=OFF: NumpyBlockBackend / ArrayApiBlockBackend are not "
    "built; \"cpu\" resolves to EigenBlockBackend, \"numpy\" is unavailable")
endif()

# Reject a compiler whose libstdc++ is newer than the one conda Python loads.
function(lasyten_glibcxx_max so_file out_var)
  set(_max "")
  if(EXISTS "${so_file}")
    execute_process(
      COMMAND strings -a "${so_file}"
      COMMAND grep "^GLIBCXX_[0-9]"
      COMMAND sort -V
      COMMAND tail -1
      OUTPUT_VARIABLE _sym
      OUTPUT_STRIP_TRAILING_WHITESPACE
      ERROR_QUIET)
    string(REGEX REPLACE "^GLIBCXX_" "" _max "${_sym}")
  endif()
  set(${out_var} "${_max}" PARENT_SCOPE)
endfunction()

set(_lasyten_conda_stdcxx "")
if(DEFINED ENV{CONDA_PREFIX}
   AND EXISTS "$ENV{CONDA_PREFIX}/lib/libstdc++.so.6")
  set(_lasyten_conda_stdcxx "$ENV{CONDA_PREFIX}/lib/libstdc++.so.6")
endif()
if(_lasyten_conda_stdcxx)
  execute_process(
    COMMAND "${CMAKE_CXX_COMPILER}" -print-file-name=libstdc++.so
    OUTPUT_VARIABLE _lasyten_cxx_stdcxx
    OUTPUT_STRIP_TRAILING_WHITESPACE)
  get_filename_component(_lasyten_cxx_stdcxx "${_lasyten_cxx_stdcxx}" REALPATH)
  get_filename_component(_lasyten_conda_stdcxx_real
    "${_lasyten_conda_stdcxx}" REALPATH)
  lasyten_glibcxx_max("${_lasyten_cxx_stdcxx}" _lasyten_cxx_glibcxx)
  lasyten_glibcxx_max("${_lasyten_conda_stdcxx_real}" _lasyten_conda_glibcxx)
  if(_lasyten_cxx_glibcxx AND _lasyten_conda_glibcxx
     AND _lasyten_cxx_glibcxx VERSION_GREATER _lasyten_conda_glibcxx)
    message(FATAL_ERROR
      "C++ compiler ${CMAKE_CXX_COMPILER} uses libstdc++ ${_lasyten_cxx_stdcxx} "
      "(GLIBCXX_${_lasyten_cxx_glibcxx}), but conda Python loads "
      "${_lasyten_conda_stdcxx_real} (GLIBCXX_${_lasyten_conda_glibcxx}). "
      "Install the matching compiler and reconfigure from a clean build dir:\n"
      "  conda install -c conda-forge cxx-compiler\n"
      "  conda activate $CONDA_PREFIX\n"
      "  rm -rf <build-dir>  # or delete CMakeCache.txt\n"
      "The env compiler is selected automatically via CXX / PATH.")
  endif()
endif()


# #####################################################################
# ## find pytorch (same shared libs as `import torch`)
# #####################################################################
# Only needed by torch.cpp, i.e. with LASYTEN_WITH_PYTHON=ON (not looked for otherwise).
if(LASYTEN_WITH_PYTHON)
  #
  # Discover Torch via the active interpreter's installed package so lasyten._core
  # links the same libtorch / libc10 that Python's torch._C loads. That makes
  # `import lasyten` + `import torch` in one process safe (one shared libtorch).
  execute_process(
    COMMAND "${Python_EXECUTABLE}" -c "import torch; print(torch.utils.cmake_prefix_path)"
    OUTPUT_VARIABLE LASYTEN_TORCH_CMAKE_PREFIX
    OUTPUT_STRIP_TRAILING_WHITESPACE
    RESULT_VARIABLE LASYTEN_TORCH_CMAKE_PREFIX_RESULT
  )
  if(NOT LASYTEN_TORCH_CMAKE_PREFIX_RESULT EQUAL 0 OR LASYTEN_TORCH_CMAKE_PREFIX STREQUAL "")
    message(FATAL_ERROR
      "Could not obtain torch.utils.cmake_prefix_path from ${Python_EXECUTABLE}. "
      "Install the matching Python torch package in this environment.")
  endif()
  list(PREPEND CMAKE_PREFIX_PATH "${LASYTEN_TORCH_CMAKE_PREFIX}")

  # Torch's Caffe2/mkl.cmake sets caffe2::mkl INTERFACE_LINK_DIRECTORIES to
  # ${MKL_ROOT}/lib/intel64 (etc.). When MKL_ROOT is unset those become the
  # bogus absolute paths /lib/intel64, and executables that link lasyten (e.g.
  # CommonCppTests) fail to resolve libtorch_cpu's MKL/sleef/omp deps. Conda
  # ships those libs in $CONDA_PREFIX/lib.
  if(NOT MKL_ROOT AND DEFINED ENV{CONDA_PREFIX})
    set(MKL_ROOT "$ENV{CONDA_PREFIX}")
  endif()

  find_package(Torch REQUIRED)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

  # Belt-and-suspenders: overwrite any empty-MKL_ROOT stub dirs Torch still set.
  if(TARGET caffe2::mkl AND MKL_ROOT)
    set_property(TARGET caffe2::mkl PROPERTY INTERFACE_LINK_DIRECTORIES
      "${MKL_ROOT}/lib")
    # Executables that link Torch (e.g. CommonCppTests) must resolve libtorch_cpu's
    # DT_NEEDED deps (MKL, sleef, omp) at link time; -L alone is not enough for
    # that — pass -rpath-link. Shared modules like _core do not need this.
    target_link_options(caffe2::mkl INTERFACE "LINKER:-rpath-link=${MKL_ROOT}/lib")
  endif()

  cmake_print_variables(Torch_DIR)
  cmake_print_variables(TORCH_INSTALL_PREFIX)
  cmake_print_variables(TORCH_LIBRARIES)
endif() # LASYTEN_WITH_PYTHON (Torch)

# #####################################################################
# ## C++ LIBRARY
# #####################################################################

add_library(lasyten STATIC)
set_property(TARGET lasyten PROPERTY C_VISIBILITY_PRESET hidden)
set_property(TARGET lasyten PROPERTY CXX_VISIBILITY_PRESET hidden)
set_property(TARGET lasyten PROPERTY VISIBILITY_INLINES_HIDDEN ON)
# CXX_VISIBILITY_PRESET is ignored for STATIC libraries; nanobind types (nb::object members, ...)
# in bound class layouts need -fvisibility=hidden on every TU that includes them (otherwise GCC
# warns "'SomeClass' declared with greater visibility than the type of its field").
# PUBLIC so dependents (examples, ctests, user programs) get the same flag.
# nanobind_add_module(_core) already adds the flag for the extension module.
target_compile_options(lasyten PRIVATE
  $<$<COMPILE_LANGUAGE:C>:-fvisibility=hidden>
)
target_compile_options(lasyten PUBLIC
  $<$<COMPILE_LANGUAGE:CXX>:-fvisibility=hidden>
  $<$<COMPILE_LANGUAGE:CXX>:-fvisibility-inlines-hidden>
)

target_include_directories(lasyten
  PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/src
  PUBLIC
  $<INSTALL_INTERFACE:include>
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
)
add_subdirectory(src)
target_compile_options(lasyten PUBLIC -fPIC)

# With LASYTEN_WITH_PYTHON=ON, liblasyten uses nanobind's object API (include/lasyten/detail/python_api.h)
# and libtorch backs TorchBlockBackend. NumpyBlockBackend additionally needs a Python interpreter
# at runtime. Linking nanobind-static[-abi3] PUBLIC propagates Py_LIMITED_API (Stable ABI mode) to
# every consumer, so library, module and test programs are compiled consistently.
# With LASYTEN_WITH_PYTHON=OFF none of these is linked (py:: is a set of opaque stubs).
if(LASYTEN_WITH_PYTHON)
  target_link_libraries(lasyten PUBLIC ${LASYTEN_NB_LIB_NAME} "${TORCH_LIBRARIES}")
endif()

# Feature switches seen by the public headers (lasyten/detail/python_api.h and users).
target_compile_definitions(lasyten PUBLIC
  LASYTEN_HAS_PYTHON=${LASYTEN_HAS_PYTHON_DEF}
  LASYTEN_WITH_NUMPY_BACKEND=${LASYTEN_WITH_NUMPY_BACKEND_DEF}
  LASYTEN_WITH_TORCH_BACKEND=${LASYTEN_WITH_TORCH_BACKEND_DEF}
)

# Native EigenBlockBackend: Eigen and LAPACK/BLAS (LAPACKE for SVD/QR/eigh/eig, BLAS for
# Eigen's dense products).
target_link_libraries(lasyten PUBLIC Eigen3::Eigen LAPACK::LAPACK)
target_include_directories(lasyten PRIVATE "${LASYTEN_LAPACKE_INCLUDE_DIR}")
# Route Eigen's dense products through the BLAS that LAPACK::LAPACK links.
set_source_files_properties(
  "${CMAKE_CURRENT_SOURCE_DIR}/src/block_backend/eigen.cpp"
  PROPERTIES COMPILE_DEFINITIONS EIGEN_USE_BLAS)

# #####################################################################
# ## hdf5_io_core (optional native HDF5)
# #####################################################################
if(LASYTEN_WITH_HDF5)
  set(_lasyten_hdf5_io_default "${CMAKE_CURRENT_SOURCE_DIR}/external/hdf5_io")
  # Prefer a sibling checkout that already has the Python-free core (Phase B).
  if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/../tenpy_other/hdf5_io/include/hdf5_io/core_io.h")
    set(_lasyten_hdf5_io_default "${CMAKE_CURRENT_SOURCE_DIR}/../tenpy_other/hdf5_io")
  elseif(NOT EXISTS "${_lasyten_hdf5_io_default}/include/hdf5_io/core_io.h")
    message(FATAL_ERROR
      "LASYTEN_WITH_HDF5=ON but hdf5_io core not found. "
      "Init submodule external/hdf5_io or set LASYTEN_HDF5_IO_DIR to a Phase-B hdf5_io tree.")
  endif()
  set(LASYTEN_HDF5_IO_DIR "${_lasyten_hdf5_io_default}" CACHE PATH
    "Source tree of hdf5_io (must provide hdf5_io_core)")
  set(HDF5_IO_BUILD_PYTHON OFF CACHE BOOL "Build hdf5_io Python module when embedded" FORCE)
  add_subdirectory("${LASYTEN_HDF5_IO_DIR}" "${CMAKE_BINARY_DIR}/_deps/hdf5_io")
  target_link_libraries(lasyten PUBLIC hdf5_io_core)
  target_compile_definitions(lasyten PUBLIC LASYTEN_WITH_HDF5=1)
  message(STATUS "LASYTEN_WITH_HDF5: using ${LASYTEN_HDF5_IO_DIR}")
endif()

# #####################################################################
# ## PYTHON BINDINGS
# #####################################################################
# hdf5_py_bridge.{h,cpp} live under pybind/tools/ and are compiled only into
# ``_core`` (not liblasyten).

if(LASYTEN_WITH_PYTHON)
  nanobind_add_module(_core ${LASYTEN_NB_STABLE_ABI_ARG} pybind/_core.cpp)

  target_include_directories(_core
    PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/pybind
    PUBLIC "${PROJECT_SOURCE_DIR}/include")
  add_subdirectory(pybind)  # adds source files to _core

  target_link_libraries(_core PUBLIC lasyten)
else()
  message(STATUS "LASYTEN_WITH_PYTHON=OFF: skipping nanobind module _core")
endif()

# #####################################################################
# ## Docstrings (Doxygen XML → pybind/docstrings/<rel>, always at build)
# #####################################################################
# PREDEFINED: document the full (Python ON) API even if this build configuration turns parts off.
# Layout: include/lasyten/<rel>  →  pybind/docstrings/<rel> (gitignored; IDE-visible)
# Each pybind TU includes only the docstring headers it needs.
# Requires doxygen (see environment.yml).
#
# Pilot list — extend as more bindings switch to DOC(...):
set(LASYTEN_MKDOC_HEADERS
  tensors/ops_algebra.h
  tensors/constructors.h
  tensors/ops_elementwise.h
  tensors/ops_legs.h
  tensors/decompositions.h
  tensors/helpers.h
  tensors/labels.h
  tensors/vector_like.h
  tensors/direct_sum.h
  tensors/tensor.h
  tensors/symmetric_tensor.h
  tensors/diagonal_tensor.h
  tensors/mask.h
  tensors/charged_tensor.h
  tensors/hidden_leg_tensor.h
  tensors/sparse.h
  tensors/planar.h
  tensors/krylov_based.h
  symmetries/factors/fermion_number.h
  symmetries/factors/fermion_parity.h
  symmetries/factors/fibonacci_anyon_category.h
  symmetries/factors/ising_anyon_category.h
  symmetries/factors/no_symmetry.h
  symmetries/factors/quantum_double_zn_anyon_category.h
  symmetries/factors/su2.h
  symmetries/factors/su2_k_anyon_category.h
  symmetries/factors/su3_3_anyon_category.h
  symmetries/factors/sun.h
  symmetries/factors/toric_code_category.h
  symmetries/factors/u1.h
  symmetries/factors/zn.h
  symmetries/factors/zn_anyon_category.h
  symmetries/factors/zn_anyon_category2.h
  symmetries/abelian_group.h
  symmetries/base_symmetry.h
  symmetries/exceptions.h
  symmetries/group.h
  symmetries/sector.h
  symmetries/spaces.h
  symmetries/styles.h
  symmetries/symmetry.h
  symmetries/symmetry_factor.h
  symmetries/trees.h
  backends/abelian.h
  backends/backend_factory.h
  backends/block_inds.h
  backends/fusion_tree_backend.h
  backends/fusion_tree_mapping.h
  backends/no_symmetry.h
  backends/tensor_backend.h
  block_backend/array_api.h
  block_backend/block_backend.h
  block_backend/dtypes.h
  block_backend/eigen.h
  block_backend/numpy.h
  block_backend/torch.h
  tools/cost_polynomials.h
  tools/mappings.h
  models/couplings.h
  models/degrees_of_freedom.h
  models/sites.h
  tools.h
)

if(LASYTEN_WITH_PYTHON)
find_program(LASYTEN_DOXYGEN_EXECUTABLE NAMES doxygen REQUIRED)
set(LASYTEN_DOCSTRINGS_DIR "${CMAKE_CURRENT_SOURCE_DIR}/pybind/docstrings")
set(LASYTEN_DOCSTRING_SCRIPT
  "${CMAKE_CURRENT_SOURCE_DIR}/scripts/doxygen_xml_to_docstrings.py")
set(LASYTEN_DOCSTRINGS_XML_ROOT
  "${CMAKE_CURRENT_BINARY_DIR}/docstrings_doxygen")

set(_lasyten_doc_outputs "")
foreach(_lasyten_hdr_rel IN LISTS LASYTEN_MKDOC_HEADERS)
  set(_lasyten_src_hdr "${CMAKE_CURRENT_SOURCE_DIR}/include/lasyten/${_lasyten_hdr_rel}")
  set(_lasyten_doc_out "${LASYTEN_DOCSTRINGS_DIR}/${_lasyten_hdr_rel}")
  set(_lasyten_safe_rel "${_lasyten_hdr_rel}")
  string(REPLACE "/" "_" _lasyten_safe_rel "${_lasyten_safe_rel}")
  string(REPLACE "." "_" _lasyten_safe_rel "${_lasyten_safe_rel}")
  set(_lasyten_xml_out "${LASYTEN_DOCSTRINGS_XML_ROOT}/${_lasyten_safe_rel}")
  set(_lasyten_doxyfile "${CMAKE_CURRENT_BINARY_DIR}/Doxyfile_docstrings_${_lasyten_safe_rel}")
  get_filename_component(_lasyten_doc_outdir "${_lasyten_doc_out}" DIRECTORY)

  file(WRITE "${_lasyten_doxyfile}"
"PROJECT_NAME           = lasyten
OUTPUT_DIRECTORY       = ${_lasyten_xml_out}
INPUT                  = ${_lasyten_src_hdr}
RECURSIVE              = NO
GENERATE_HTML          = NO
GENERATE_LATEX         = NO
GENERATE_XML           = YES
EXTRACT_ALL            = YES
QUIET                  = YES
WARNINGS               = NO
MARKDOWN_SUPPORT       = NO
AUTOLINK_SUPPORT       = NO
ENABLE_PREPROCESSING   = YES
MACRO_EXPANSION        = NO
PREDEFINED             = LASYTEN_HAS_PYTHON=1 LASYTEN_WITH_NUMPY_BACKEND=1 LASYTEN_WITH_TORCH_BACKEND=1
")

  # Write into the source tree under pybind/docstrings/ so IDEs can open DOC()
  # headers next to the bindings. Files are gitignored; always regenerated.
  add_custom_command(
    OUTPUT ${_lasyten_doc_out}
    COMMAND ${CMAKE_COMMAND} -E make_directory ${_lasyten_doc_outdir}
    COMMAND ${CMAKE_COMMAND} -E rm -rf ${_lasyten_xml_out}
    COMMAND ${CMAKE_COMMAND} -E make_directory ${_lasyten_xml_out}
    COMMAND ${LASYTEN_DOXYGEN_EXECUTABLE} ${_lasyten_doxyfile}
    COMMAND ${Python_EXECUTABLE} ${LASYTEN_DOCSTRING_SCRIPT}
            --xml-dir ${_lasyten_xml_out}/xml
            --header-rel ${_lasyten_hdr_rel}
            --source-root ${CMAKE_CURRENT_SOURCE_DIR}
            -o ${_lasyten_doc_out}
    DEPENDS ${_lasyten_src_hdr} ${LASYTEN_DOCSTRING_SCRIPT} ${_lasyten_doxyfile}
    COMMENT "Generating pybind/docstrings/${_lasyten_hdr_rel}"
    VERBATIM
  )
  list(APPEND _lasyten_doc_outputs ${_lasyten_doc_out})
endforeach()

add_custom_target(lasyten_generate_docstrings DEPENDS ${_lasyten_doc_outputs})
add_custom_target(lasyten_mkdoc_docstrings DEPENDS lasyten_generate_docstrings)
add_dependencies(_core lasyten_generate_docstrings)
message(STATUS
  "Docstrings: generating ${LASYTEN_MKDOC_HEADERS} under pybind/docstrings/ "
  "(doxygen=${LASYTEN_DOXYGEN_EXECUTABLE})")


# #####################################################################
# ## .pyi stub (nanobind stubgen → lasyten/_core.pyi, for IDE type hints)
# #####################################################################
# Written into the source tree next to _core.so (gitignored; always regenerated)
# so IDEs/LSPs get parameter hints for the compiled extension. Optional: if
# the stub generation fails (e.g. _core cannot be imported at build time), warn and skip
# rather than fail the build.
#
# Do **not** run ``python -m nanobind.stubgen -m _core`` with ``PYTHONPATH=build``:
# when ``LASYTEN_STABLE_ABI`` is on, a leftover ``_core.cpython-*.so`` shadows
# ``_core.abi3.so`` and stubgen emits ``typing.Any`` for ``py::array``. Use
# ``scripts/generate_core_pyi.py`` which loads ``$<TARGET_FILE:_core>`` explicitly.
set(LASYTEN_CORE_PYI "${CMAKE_CURRENT_SOURCE_DIR}/lasyten/_core.pyi")
set(LASYTEN_GENERATE_CORE_PYI
  "${CMAKE_CURRENT_SOURCE_DIR}/scripts/generate_core_pyi.py")
set(LASYTEN_STUBGEN_PATTERNS
  "${CMAKE_CURRENT_SOURCE_DIR}/scripts/stubgen_patterns.txt")

add_custom_command(
  OUTPUT ${LASYTEN_CORE_PYI}
  COMMAND ${CMAKE_COMMAND} -E env LASYTEN_STUBGEN_SOFT=1
          ${Python_EXECUTABLE} ${LASYTEN_GENERATE_CORE_PYI}
          --so $<TARGET_FILE:_core>
          -o ${LASYTEN_CORE_PYI}
          -p ${LASYTEN_STUBGEN_PATTERNS}
  DEPENDS _core ${LASYTEN_GENERATE_CORE_PYI} ${LASYTEN_STUBGEN_PATTERNS}
  COMMENT "Generating lasyten/_core.pyi via nanobind.stubgen"
  VERBATIM
)
add_custom_target(lasyten_generate_pyi_stub ALL DEPENDS ${LASYTEN_CORE_PYI})
add_dependencies(lasyten_generate_pyi_stub _core)


# Resolve libtorch/c10 from the same place as Python torch._C.
# Installed layout: site-packages/lasyten/_core*.so
#   $ORIGIN/../../..  -> env/lib (conda/micromamba shared torch libs)
#   $ORIGIN/../torch/lib -> site-packages/torch/lib (pip wheel layout)
# Do not use INSTALL_RPATH_USE_LINK_PATH: Torch/Caffe2 link dirs may inject
# unrelated paths (e.g. Intel MKL stubs) that we do not want on the module.
get_filename_component(_lasyten_torch_lib_dir "${TORCH_INSTALL_PREFIX}/lib" ABSOLUTE)
set(_lasyten_core_rpath "$ORIGIN/../../..;$ORIGIN/../torch/lib;${_lasyten_torch_lib_dir}")
if(DEFINED ENV{CONDA_PREFIX} AND NOT "$ENV{CONDA_PREFIX}" STREQUAL "")
  list(APPEND _lasyten_core_rpath "$ENV{CONDA_PREFIX}/lib")
endif()
# list(APPEND) on a ;-string splits; rebuild a single RPATH string.
string(JOIN ":" _lasyten_core_rpath_str ${_lasyten_core_rpath})
set_target_properties(_core PROPERTIES
  BUILD_RPATH "${_lasyten_core_rpath_str}"
  INSTALL_RPATH "${_lasyten_core_rpath_str}"
  INSTALL_RPATH_USE_LINK_PATH FALSE
)


# The install directory is the output (wheel) directory
install(TARGETS _core LIBRARY DESTINATION lasyten)
install(FILES "${CMAKE_CURRENT_SOURCE_DIR}/lasyten/py.typed" DESTINATION lasyten)
install(FILES ${LASYTEN_CORE_PYI} DESTINATION lasyten OPTIONAL)

endif() # LASYTEN_WITH_PYTHON


# #####################################################################
# ## DOCUMENTATION
# #####################################################################
if(BUILD_DOC)
  find_package(Doxygen REQUIRED)

  if(DOXYGEN_FOUND)
    configure_file(lasyten_Doxyfile.in lasytenDoxyfile)
    add_custom_target(doc ALL
      ${DOXYGEN_EXECUTABLE} ${PROJECT_SOURCE_DIR}/docs/Doxyfile
      WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}/docs
      COMMENT "Generating API documentation with Doxygen in docs/build_docs/" VERBATIM
    )
  endif(DOXYGEN_FOUND)

  message(STATUS " Build Documentation: YES")
  message(STATUS "  - Doxygen: ${DOXYGEN_EXECUTABLE}")
else()
  message(STATUS " Build Documentation: NO")
endif()

if(DOXYGEN_FOUND)
  install(DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/docs/
    DESTINATION docs
    COMPONENT documentation
  )
endif()


# #####################################################################
# ## clang-tidy for linting
# #####################################################################
if (LINTING)
  find_program(CLANG_TIDY_EXE "clang-tidy")
  if(CLANG_TIDY_EXE)
    message(STATUS "clang-tidy found: ${CLANG_TIDY_EXE}")
    set(CMAKE_CXX_CLANG_TIDY ${CLANG_TIDY_EXE})
  else()
    message(WARNING "clang-tidy not found!")
  endif()
endif()


# #####################################################################
# ## CTest
# #####################################################################

include(CTest)
if(BUILD_TESTING)
  enable_testing()
  add_subdirectory(tests/ctest)
endif()
