find_package(Catch2 3 REQUIRED)
include(Catch)

add_executable(pccl_tests
  test_roundtrip.cpp
  test_header.cpp
  test_aabb_edges.cpp
  test_determinism.cpp
  test_corrupted.cpp
  test_nonfinite.cpp
  test_file_io.cpp
)
target_include_directories(pccl_tests PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(pccl_tests PRIVATE
  pccl
  Catch2::Catch2WithMain)
target_compile_options(pccl_tests PRIVATE
  $<$<NOT:$<CXX_COMPILER_ID:MSVC>>:-Wall -Wextra>
  $<$<CXX_COMPILER_ID:MSVC>:/W3>)

catch_discover_tests(pccl_tests)

# Standalone alignment probe (not a Catch2 test; useful as a smoke check).
add_executable(pccl_align_probe test_alignment.cpp)
target_link_libraries(pccl_align_probe PRIVATE pccl)
target_compile_options(pccl_align_probe PRIVATE ${PCCL_RELEASE_FLAGS})
