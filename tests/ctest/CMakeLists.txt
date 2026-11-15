# #####################################################################
# ## ADD SUB-DIRECTORIES
# #####################################################################

enable_testing()

# C++ tests that exercise NumpyBlockBackend (or other py:: APIs) need
# pybind11::embed so a Python interpreter is available.
#
# With LASYTEN_WITH_PYTHON=OFF only the Python-free tests are built (Phase 5 scaffolding): the
# Eigen block-backend test compiles its numpy round trip out (LASYTEN_HAS_PYTHON=0) and runs
# without an interpreter. The other tests / examples use the full tensor stack, which still needs
# Python.
if(NOT LASYTEN_WITH_PYTHON)
  message(STATUS
    "LASYTEN_WITH_PYTHON=OFF: building only the Python-free ctest (test_eigen_block_backend); "
    "skipping the other tests / examples that need the Python-dependent liblasyten sources.")
  add_executable(EigenNoPythonTests test_eigen_block_backend_main.cpp test_eigen_block_backend.cpp)
  set_source_files_properties(test_eigen_block_backend_main.cpp test_eigen_block_backend.cpp
    PROPERTIES SKIP_LINTING ON)
  target_include_directories(EigenNoPythonTests PRIVATE ${PROJECT_SOURCE_DIR}/include)
  target_link_libraries(EigenNoPythonTests PUBLIC lasyten)
  add_test(NAME test_eigen_block_backend_no_python COMMAND EigenNoPythonTests)
  return()
endif()

# create the testing file and list of tests
set (TestsToRun
  test_sector.cpp
  test_sector_casters.cpp
  test_block_inds.cpp
  test_lru_cache.cpp
  test_eigen_block_backend.cpp
)

create_test_sourcelist(Tests CommonCppTests.cpp ${TestsToRun})


# add the executable
add_executable(CommonCppTests ${Tests})
set_source_files_properties(CommonCppTests.cpp ${TestsToRun} PROPERTIES
  SKIP_LINTING ON
)

target_include_directories(CommonCppTests
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/tests/ctest
  PRIVATE ${PROJECT_SOURCE_DIR}/pybind
  PUBLIC ${PROJECT_SOURCE_DIR}/include
)
# embed: NumpyBlockBackend and related tests need an interpreter; keep even if
# some tests look "pure C++" until headers no longer require py::.
target_link_libraries(CommonCppTests PUBLIC lasyten pybind11::embed)

# Add all the ADD_TEST for each test
foreach (test ${TestsToRun})
  get_filename_component (TName ${test} NAME_WE)
  add_test (NAME ${TName} COMMAND CommonCppTests ${TName} )
endforeach ()

# User-guide example from docs/intro/examples/ (see docs/intro/first_steps.rst).
set(example_heisenberg_src
  ${PROJECT_SOURCE_DIR}/docs/intro/examples/heisenberg_two_site.cpp)
add_executable(example_heisenberg_two_site ${example_heisenberg_src})
set_source_files_properties(${example_heisenberg_src} PROPERTIES SKIP_LINTING ON)
target_include_directories(example_heisenberg_two_site
  PUBLIC ${PROJECT_SOURCE_DIR}/include
)
target_link_libraries(example_heisenberg_two_site PUBLIC lasyten pybind11::embed)
set_target_properties(example_heisenberg_two_site PROPERTIES
  CXX_VISIBILITY_PRESET hidden
  VISIBILITY_INLINES_HIDDEN ON
)
add_test(NAME example_heisenberg_two_site COMMAND example_heisenberg_two_site)
