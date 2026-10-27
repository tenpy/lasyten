# #####################################################################
# ## ADD SUB-DIRECTORIES
# #####################################################################

enable_testing()

# C++ tests that exercise NumpyBlockBackend (or other py:: APIs) need
# pybind11::embed so a Python interpreter is available. Pure C++ unit tests
# that never touch numpy/pybind APIs would not need embed — none of the
# current suite qualifies yet while public headers pull in py:: types.
if(NOT LASYTEN_WITH_PYTHON)
  message(STATUS
    "LASYTEN_WITH_PYTHON=OFF: skipping CommonCppTests / examples that need "
    "numpy backends (excluded from liblasyten).")
  return()
endif()

# create the testing file and list of tests
set (TestsToRun
  test_sector.cpp
  test_sector_casters.cpp
  test_block_inds.cpp
  test_lru_cache.cpp
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
