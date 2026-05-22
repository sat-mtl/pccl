# benchmarks/  -- exploration and timing tools.
#
# Two kinds of tools live here:
#   1. Lightweight, vendored-deps-only:  pccl_roundtrip_bench, pccl_cli.
#      Built whenever PCCL_BUILD_BENCHMARKS is ON. Only need miniply
#      (vendored submodule).
#   2. Heavyweight comparison tool:      pccl_codec_explorer.
#      Built only when PCCL_BUILD_CODEC_EXPLORER is ON. Requires
#      zlib, zstd, lz4, brotli, snappy, lzo as system packages -- it
#      benchmarks pccl's output against those generic byte codecs.

# Vendored miniply (MIT) — PLY loader. Used by every benchmark tool.
set(MINIPLY_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../3rdparty/miniply")
add_library(miniply STATIC ${MINIPLY_DIR}/miniply.cpp)
target_include_directories(miniply PUBLIC ${MINIPLY_DIR})

# roundtrip_bench: load a PLY, encode with each pccl preset, decode, verify
# correctness, print per-stage timings.
add_executable(pccl_roundtrip_bench roundtrip_bench.cpp)
target_include_directories(pccl_roundtrip_bench PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(pccl_roundtrip_bench PRIVATE pccl miniply)
target_compile_options(pccl_roundtrip_bench PRIVATE
  $<$<NOT:$<CXX_COMPILER_ID:MSVC>>:-Wall -Wextra>
  $<$<CXX_COMPILER_ID:MSVC>:/W3>)

# pccl_cli: encode / decode / info subcommands for .pccl files.
add_executable(pccl_cli cli.cpp)
target_include_directories(pccl_cli PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(pccl_cli PRIVATE pccl miniply)
target_compile_options(pccl_cli PRIVATE
  $<$<NOT:$<CXX_COMPILER_ID:MSVC>>:-Wall -Wextra>
  $<$<CXX_COMPILER_ID:MSVC>:/W3>)

# codec_explorer: per-stage exploration across every codec we can link.
# Gated separately because it pulls in 6 extra system packages.
if(PCCL_BUILD_CODEC_EXPLORER)
  find_package(ZLIB REQUIRED)
  find_package(PkgConfig REQUIRED)
  pkg_check_modules(ZSTD REQUIRED IMPORTED_TARGET libzstd)
  pkg_check_modules(LZ4  REQUIRED IMPORTED_TARGET liblz4)

  find_library(SNAPPY_LIB    snappy        REQUIRED)
  find_library(LZO_LIB       lzo2          REQUIRED)
  find_library(BROTLI_ENC    brotlienc     REQUIRED)
  find_library(BROTLI_COM    brotlicommon  REQUIRED)

  add_executable(pccl_codec_explorer codec_explorer.cpp)
  target_include_directories(pccl_codec_explorer PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/../3rdparty/ryg_rans)
  target_link_libraries(pccl_codec_explorer PRIVATE
    miniply
    ZLIB::ZLIB
    PkgConfig::ZSTD
    PkgConfig::LZ4
    ${SNAPPY_LIB}
    ${LZO_LIB}
    ${BROTLI_ENC}
    ${BROTLI_COM}
    streamvbyte
    fse_core
    turbopfor)
  target_compile_options(pccl_codec_explorer PRIVATE
    $<$<NOT:$<CXX_COMPILER_ID:MSVC>>:-Wall -Wextra>
    $<$<CXX_COMPILER_ID:MSVC>:/W3>)
endif()
