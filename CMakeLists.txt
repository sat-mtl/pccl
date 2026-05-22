cmake_minimum_required(VERSION 3.20)
project(pccl
  VERSION 0.1.0
  DESCRIPTION "Point Cloud Compression Library: presets-based encode/decode of point cloud frames"
  LANGUAGES CXX C
)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
  set(CMAKE_INTERPROCEDURAL_OPTIMIZATION ON)
endif()

option(PCCL_BUILD_TESTS           "Build the Catch2 test suite" ON)
option(PCCL_BUILD_BENCHMARKS      "Build the round-trip benchmark + CLI (vendored miniply only)" ON)
option(PCCL_BUILD_CODEC_EXPLORER  "Build pccl_codec_explorer (needs zlib, zstd, lz4, brotli, snappy, lzo)" OFF)
option(PCCL_ENABLE_LTO       "Enable link-time optimisation (IPO)" ON)
option(PCCL_NATIVE_ARCH      "Tune for the host CPU (-march=native / -mcpu=native). Turn OFF for portable builds (CI)." ON)

# Detect the target architecture once so the rest of the file can branch on it.
if(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|AMD64|amd64")
  set(PCCL_ARCH_X86_64 TRUE)
elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64|arm64|ARM64")
  set(PCCL_ARCH_AARCH64 TRUE)
else()
  message(WARNING "pccl: unrecognised CMAKE_SYSTEM_PROCESSOR=${CMAKE_SYSTEM_PROCESSOR}; building with scalar fallbacks only")
endif()

if(NOT MSVC)
  set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS}  -fvisibility=internal -fvisibility-inlines-hidden  -fstrict-aliasing -falign-loops -fno-semantic-interposition ")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}  -fvisibility=internal -fvisibility-inlines-hidden  -fstrict-aliasing -falign-loops  -fno-semantic-interposition ")
  if(PCCL_NATIVE_ARCH)
    if(PCCL_ARCH_X86_64)
      set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS}  -O3 -ffp-contract=fast -march=native -mtune=native")
      set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -ffp-contract=fast -march=native -mtune=native")
    elseif(PCCL_ARCH_AARCH64)
      set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -O3 -ffp-contract=fast -mcpu=native")
      set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -ffp-contract=fast -mcpu=native")
    endif()
  endif()
endif()

set(EXT "${CMAKE_CURRENT_SOURCE_DIR}/3rdparty")

# ------------------------------------------------------------------

add_library(streamvbyte STATIC
  ${EXT}/streamvbyte/src/streamvbyte_encode.c
  ${EXT}/streamvbyte/src/streamvbyte_decode.c
  ${EXT}/streamvbyte/src/streamvbyte_0124_encode.c
  ${EXT}/streamvbyte/src/streamvbyte_0124_decode.c
  ${EXT}/streamvbyte/src/streamvbyte_zigzag.c
  ${EXT}/streamvbyte/src/streamvbytedelta_encode.c
  ${EXT}/streamvbyte/src/streamvbytedelta_decode.c
)
target_include_directories(streamvbyte PUBLIC ${EXT}/streamvbyte/include)
target_compile_options(streamvbyte PRIVATE ${PCCL_RELEASE_FLAGS})

add_library(fse_core STATIC
  ${EXT}/fse/lib/fse_compress.c
  ${EXT}/fse/lib/fse_decompress.c
  ${EXT}/fse/lib/entropy_common.c
  ${EXT}/fse/lib/hist.c
  ${EXT}/fse/lib/debug.c
)
target_include_directories(fse_core PUBLIC ${EXT}/fse/lib)
target_compile_options(fse_core PRIVATE ${PCCL_RELEASE_FLAGS})

set(TURBOPFOR_DIR ${EXT}/turbopfor)

set(TURBOPFOR_BASE_SRCS
  ${TURBOPFOR_DIR}/lib/bic.c
  ${TURBOPFOR_DIR}/lib/bitunpack.c
  ${TURBOPFOR_DIR}/lib/bitpack.c
  ${TURBOPFOR_DIR}/lib/bitutil.c
  ${TURBOPFOR_DIR}/lib/eliasfano.c
  ${TURBOPFOR_DIR}/lib/fp.c
  ${TURBOPFOR_DIR}/lib/transpose.c
  ${TURBOPFOR_DIR}/lib/transpose_.c
  ${TURBOPFOR_DIR}/lib/trlec.c
  ${TURBOPFOR_DIR}/lib/trled.c
  ${TURBOPFOR_DIR}/lib/vp4c.c
  ${TURBOPFOR_DIR}/lib/vp4d.c
  ${TURBOPFOR_DIR}/lib/v8.c
  ${TURBOPFOR_DIR}/lib/v8pack.c
  ${TURBOPFOR_DIR}/lib/vint.c
  ${TURBOPFOR_DIR}/lib/vsimple.c
  ${TURBOPFOR_DIR}/lib/vbit.c
)
if(MSVC)
  set(TURBOPFOR_COMMON_FLAGS /w /wd4133 /wd4244 /wd4267 /TP /FImath.h)
else()
  set(TURBOPFOR_COMMON_FLAGS
    -fstrict-aliasing -fPIC -falign-loops -w
    -Wno-incompatible-pointer-types
    -Wno-error=incompatible-pointer-types
    -Wno-error=int-conversion
  )
endif()
set(TURBOPFOR_INCLUDES
  ${TURBOPFOR_DIR}/include
  ${TURBOPFOR_DIR}/lib/include_
  ${TURBOPFOR_DIR}/lib
  ${TURBOPFOR_DIR}
  ${TURBOPFOR_DIR}/ext
)

add_library(turbopfor_base OBJECT ${TURBOPFOR_BASE_SRCS})
target_include_directories(turbopfor_base PRIVATE ${TURBOPFOR_INCLUDES})
target_compile_definitions(turbopfor_base PRIVATE NDEBUG)
if(MSVC)
  target_compile_options(turbopfor_base PRIVATE /O2 ${TURBOPFOR_COMMON_FLAGS})
elseif(PCCL_ARCH_X86_64)
  target_compile_options(turbopfor_base PRIVATE
    -O3 ${TURBOPFOR_COMMON_FLAGS}
    -march=corei7-avx -mtune=native)
else()
  target_compile_options(turbopfor_base PRIVATE -O3 ${TURBOPFOR_COMMON_FLAGS})
endif()
set_target_properties(turbopfor_base PROPERTIES INTERPROCEDURAL_OPTIMIZATION FALSE)

if(PCCL_ARCH_X86_64)
  set(TURBOPFOR_AVX2_SRCS
    ${TURBOPFOR_DIR}/lib/vp4c.c
    ${TURBOPFOR_DIR}/lib/vp4d.c
    ${TURBOPFOR_DIR}/lib/transpose.c
    ${TURBOPFOR_DIR}/lib/bitpack.c
    ${TURBOPFOR_DIR}/lib/bitunpack.c
    ${TURBOPFOR_DIR}/lib/bitutil.c
  )
  add_library(turbopfor_avx2 OBJECT ${TURBOPFOR_AVX2_SRCS})
  target_compile_definitions(turbopfor_avx2 PRIVATE NDEBUG)
  if(MSVC)
    target_compile_options(turbopfor_avx2 PRIVATE /O2 /arch:AVX2 ${TURBOPFOR_COMMON_FLAGS})
  else()
    target_compile_options(turbopfor_avx2 PRIVATE
      -O3 ${TURBOPFOR_COMMON_FLAGS}
      -march=native -mtune=native)
  endif()
  target_include_directories(turbopfor_avx2 PRIVATE ${TURBOPFOR_INCLUDES})
  set_target_properties(turbopfor_avx2 PROPERTIES INTERPROCEDURAL_OPTIMIZATION FALSE)

  add_library(turbopfor STATIC
    $<TARGET_OBJECTS:turbopfor_base>
    $<TARGET_OBJECTS:turbopfor_avx2>)
else()
  add_library(turbopfor STATIC
    $<TARGET_OBJECTS:turbopfor_base>)
endif()
target_include_directories(turbopfor PUBLIC ${TURBOPFOR_DIR}/include)
set_target_properties(turbopfor PROPERTIES INTERPROCEDURAL_OPTIMIZATION FALSE)

if(PCCL_ARCH_X86_64)
  add_library(rans_static STATIC ${EXT}/rans_static/r8x16_avx2.c)
  target_include_directories(rans_static PUBLIC ${EXT}/rans_static)
  target_compile_definitions(rans_static PRIVATE NDEBUG)
  if(MSVC)
    target_compile_options(rans_static PRIVATE /O2 /w /arch:AVX2)
  else()
    target_compile_options(rans_static PRIVATE
      -O3 -w -mavx2 -mbmi2
      -Wno-error=incompatible-pointer-types
      -Wno-error=int-conversion
      -Wno-error=implicit-function-declaration)
  endif()
else()
  add_library(rans_static STATIC ${EXT}/rans_static/rANS_static4x16.c)
  target_include_directories(rans_static PUBLIC ${EXT}/rans_static)
  target_compile_definitions(rans_static PRIVATE NDEBUG)
  if(MSVC)
    target_compile_options(rans_static PRIVATE /O2 /w)
  else()
    target_compile_options(rans_static PRIVATE
      -O3 -w
      -Wno-error=incompatible-pointer-types
      -Wno-error=int-conversion
      -Wno-error=implicit-function-declaration)
  endif()
endif()

add_library(xsimd INTERFACE)
target_include_directories(xsimd INTERFACE ${EXT}/xsimd/include)

find_package(Boost REQUIRED)

add_library(atomic_queue INTERFACE)
target_include_directories(atomic_queue INTERFACE ${EXT}/atomic_queue/include)

set(THREADS_PREFER_PTHREAD_FLAG TRUE)
find_package(Threads REQUIRED)
add_library(ips4o INTERFACE)
target_include_directories(ips4o INTERFACE ${EXT}/ips4o/include)
if(NOT MSVC)
  target_compile_options(ips4o INTERFACE -pthread)
  target_compile_definitions(ips4o INTERFACE _REENTRANT)
endif()
target_link_libraries(ips4o INTERFACE atomic_queue Threads::Threads)
if(NOT MSVC AND NOT APPLE AND NOT (CMAKE_CXX_COMPILER_ID MATCHES "Clang"))
  target_link_libraries(ips4o INTERFACE atomic)
endif()

# ------------------------------------------------------------------

add_library(pccl STATIC src/pccl.cpp)
if(PCCL_BUILD_BENCHMARKS)
  target_compile_definitions(pccl PRIVATE PCCL_BENCHMARKS)
endif()
target_include_directories(pccl
  PUBLIC  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
          $<INSTALL_INTERFACE:include>
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
target_link_libraries(pccl PUBLIC
  fse_core
  turbopfor
  rans_static
  xsimd
  ips4o
  Boost::headers
)
if(MSVC)
  target_compile_options(pccl PRIVATE /W3)
else()
  target_compile_options(pccl PRIVATE -Wall -Wextra)
endif()
add_library(pccl::pccl ALIAS pccl)

if(PCCL_BUILD_TESTS)
  enable_testing()
  add_subdirectory(tests)
endif()

if(PCCL_BUILD_BENCHMARKS)
  add_subdirectory(benchmarks)
endif()
