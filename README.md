# pccl

A point cloud compression library (positions only) with a tiny public API. 
The caller passes a frame of `float xyz` positions, picks one of four presets, 
and gets back a self-describing byte buffer. 
The receiver calls `decode()` on those bytes and gets the cloud back. 

Ultra-low latency compared to MPEG G-PCC compression (100-300x faster encoding / decoding) at the 
cost of slightly worse compression ratio (between 1.1x and 2.5x the MPEG G-PCC compressed size depending on the cloud). 
Most LIDAR point clouds will encode in a few milliseconds at most ; a 3.2 million point PCL 
encodes to 9 megabytes in ~32 milliseconds and decodes in ~10 milliseconds on a 5950X CPU.
On smaller point clouds (for instance an Orbbec scan) compression within a 60FPS frame budget is possible.

Maximum input precision is 21 bits per axis, way above the actual precision of an actual LIDAR.

## Building

```
cmake -S . -B build -GNinja
cmake --build build
ctest --test-dir build
```

Build options:

| Option | Default | Effect |
|---|---|---|
| `PCCL_BUILD_TESTS` | ON | Build the Catch2 test suite. |
| `PCCL_BUILD_BENCHMARKS` | ON | Build the benchmark and explorer tools (needs zlib, zstd, lz4, brotli, snappy, lzo system packages; miniply is vendored). |
| `PCCL_ENABLE_LTO` | ON | Enable interprocedural optimisation. |
| `PCCL_NATIVE_ARCH` | ON | Tune for the host CPU (`-march=native` on x86_64, `-mcpu=native` on aarch64). Turn OFF for portable CI builds. |

Compile flags applied to library and test code are `-O3 -ffp-contract=fast`
plus the arch-tuning flag selected by `PCCL_NATIVE_ARCH`. The project does
not use `-Ofast` by default
because, in our measurements, `-ffast-math` regressed `std::unique` codegen
under gcc and forced a workaround for an infinity-sentinel pattern in the
AABB reduction. The sentinel was rewritten anyway (it would have been a
latent bug under any `-ffinite-math-only` build), so a build with `-Ofast`
now works correctly; it is not faster in practice for this workload.

## Quick start

Dependencies: C++23, CMake, boost. Multiple compression libraries available for benchmarks (zstd etc.).

The API is two stateful classes that own their scratch buffers. Reuse one
`Encoder` across the whole stream so the steady state stops allocating after the first frame.

```cpp
#include <pccl/pccl.hpp>
#include <span>

pccl::Encoder enc;                       // reuse across frames
pccl::AlignedVec<std::uint8_t> bytes;    // reuse across frames

pccl::EncodeOptions opts;
opts.preset = pccl::Preset::Balanced;    // Fast, Balanced, Small, Smallest
opts.bits   = 16;                        // 1..21 bits per axis

for (const auto& frame : frames)
{
  // xyz is a flat span of floats whose size is a multiple of 3.
  std::span<const float> xyz{frame.xyz, frame.num_points * 3};
  pccl::EncodeStats stats;
  enc.encode(xyz, opts, bytes, &stats);
  send(bytes.data(), bytes.size());
}

// On the receiver:
pccl::Decoder dec;                       // reuse across frames
pccl::AlignedVec<float>  decoded;        // reuse across frames
pccl::FrameMetadata meta;

while (auto packet = recv())
{
  std::span<const std::uint8_t> data{packet.data, packet.size};
  dec.decode(data, decoded, &meta);
  // decoded has 3 * meta.unique_voxels floats
}
```

For streaming, set a shared AABB explicitly so consecutive frames quantise
to the same grid:

```cpp
opts.aabb_min   = {-1.f, -1.f, -1.f};
opts.aabb_max   = { 1.f,  1.f,  1.f};
opts.aabb_valid = true;
```

`pccl::peek_header()` reads the bitstream metadata without touching the
payload, useful for a relay that wants to route or transcode without
decoding.

## Presets

Each preset is a fixed pipeline. All four voxelise the input (collapse
duplicate quantised points). The receiver gets `unique_voxels` floats, not
the original `num_points`.

| Preset | Pipeline | When to pick |
|---|---|---|
| `Fast` | TurboPFor `vbxenc64` (delta + variable byte on u64) | Lowest decode latency. Best ratio per decode millisecond at low/mid bit depth. |
| `Balanced` | TurboPFor `p4ndenc64` on sorted Morton | Densest output at mid/high bit depth. |
| `Small` | delta + LEB128 + FSE entropy coding | Best size/speed compromise when decode budget is tight. |
| `Smallest` | octree occupancy bytes + AVX2 8-way interleaved rANS | Absolute smallest output. Encode/decode are 10× the cost of the others. |

At low bit depths (10-12) Fast becomes attractive: smaller decode-time-per-byte than Balanced. 
At high bit depths (>=20) Balanced wins on size. 
See the per-bit numbers below for the actual tradeoff.

## Wire format

Little-endian throughout.

| Offset | Bytes | Field |
|---:|---:|---|
| 0 | 4 | magic `'P','C','C','1'` |
| 4 | 1 | preset (0..3) |
| 5 | 1 | bits per axis (1..21) |
| 6 | 2 | flags (reserved, must be 0) |
| 8 | 4 | input_points (caller's count, for stats only) |
| 12 | 4 | unique_voxels (count the receiver reconstructs) |
| 16 | 24 | aabb: 6 little-endian float32, min[3] then max[3] |
| 40 | 4 | payload_size |
| 44 | payload_size | preset-specific payload |

`decode()` validates magic, preset range, bits range, and payload bounds; it
throws `std::runtime_error` on any malformed input.

## Benchmarks

Hardware: AMD 5950X (AVX2). Build: gcc 16.1 release, default flags listed above.

Input: `14 Ladybrook Road 10.ply`, 3 202 232 points,
36.65 MiB of raw float xyz (12 B per point). The benchmark loads with
miniply, computes the AABB once, then runs each preset and reports per-stage
timings. Numbers below are medians of three runs.

### Compression ratio and encode time

| bits/axis | preset | bytes | B/point | ratio vs raw | enc (ms) | dec (ms) |
|---:|---|---:|---:|---:|---:|---:|
| 10 | Fast | 3 468 925 | 1.08 | 0.090 | 26 | 4 |
| 10 | Balanced | 2 626 992 | 0.82 | 0.068 | 38 | 18 |
| 10 | Small | 2 239 540 | 0.70 | 0.058 | 31 | 11 |
| 10 | Smallest | 2 214 719 | 0.69 | 0.058 | 143 | 120 |
| 14 | Fast | 9 078 152 | 2.83 | 0.236 | 32 | 10 |
| 14 | Balanced | 7 569 295 | 2.36 | 0.197 | 45 | 25 |
| 14 | Small | 7 897 539 | 2.47 | 0.206 | 53 | 31 |
| 14 | Smallest | 7 533 729 | 2.35 | 0.196 | 434 | 412 |
| 16 | Fast | 12 457 251 | 3.89 | 0.324 | 34 | 12 |
| 16 | Balanced | 9 970 344 | 3.11 | 0.259 | 34 | 14 |
| 16 | Small | 10 513 751 | 3.28 | 0.274 | 48 | 28 |
| 16 | Smallest | 10 068 355 | 3.14 | 0.262 | 563 | 544 |
| 20 | Fast | 18 570 532 | 5.79 | 0.483 | 29 | 8 |
| 20 | Balanced | 14 774 637 | 4.61 | 0.384 | 35 | 14 |
| 20 | Small | 15 618 329 | 4.87 | 0.406 | 54 | 36 |
| 20 | Smallest | 15 100 113 | 4.71 | 0.393 | 844 | 824 |


### Pick-by-use-case

**bits=10 (dense voxel grid, lots of duplicates collapsed. Good enough for tracking):**

| Preset | Output | Encode | Decode | Notes |
|---|---:|---:|---:|---|
| Fast | 3.47 MB | 26 ms | **4 ms** | Best decode-time-per-byte. |
| Small | 2.24 MB | 31 ms | 11 ms | Best size/speed compromise. |
| Balanced | 2.63 MB | 38 ms | 18 ms | Dominated by Small here. Don't pick. |
| Smallest | 2.21 MB | 143 ms | 120 ms | 1% smaller than Small at 11× the cost. |

**bits=14 (live-streaming sweet spot for lidar/photogrammetry):**

| Preset | Output | Encode | Decode | Notes |
|---|---:|---:|---:|---|
| Fast | 9.08 MB | 32 ms | **10 ms** | Vetter than Small at this depth. |
| Balanced | 7.57 MB | 45 ms | 25 ms | Best size when 25 ms decode fits. |
| Small | 7.90 MB | 53 ms | 31 ms | Slower and bigger than Balanced. |
| Smallest | 7.53 MB | 434 ms | 412 ms | 0.5% smaller than Balanced; not worth it. |

**bits=16 (default quality):**

| Preset | Output | Encode | Decode | Notes |
|---|---:|---:|---:|---|
| Fast | 12.5 MB | 34 ms | 12 ms | Fastest decode. |
| Balanced | 9.97 MB | 34 ms | 14 ms | Same encode time as Fast, 20% smaller. |
| Small | 10.5 MB | 48 ms | 28 ms | Dominated by Balanced. |
| Smallest | 10.1 MB | 563 ms | 544 ms | 1% smaller than Balanced at 16× the cost. |

**bits=20 (lossy archival, voxel grid very sparse):**

| Preset | Output | Encode | Decode |
|---|---:|---:|---:|
| Fast | 18.6 MB | 29 ms | 8 ms |
| Balanced | 14.8 MB | 35 ms | 14 ms |
| Small | 15.6 MB | 54 ms | 36 ms |
| Smallest | 15.1 MB | 844 ms | 824 ms |

**Decision summary:**

- **bits ≤ 12**: pick **Fast** for the streaming path, **Small** if you need
  the absolute smallest at sub-30-ms encode.
- **bits 14-16**: **Fast** if decode time matters more than wire bytes;
  **Balanced** otherwise.
- **bits ≥ 18**: **Balanced** wins on size with the smallest size/speed
  penalty; **Fast** still has the cheapest decoder.
- **Smallest**: as useful as the placebo x264 preset.

### Comparison vs MPEG G-PCC

Reference implementation: `mpeg-pcc-tmc13` release-v23.0-rc2 (the official
ISO/IEC MPEG G-PCC encoder). Both codecs built with `-O3 -march=native
-ffp-contract=fast`. tmc13 was given an `inputScale` that matches each
pccl bit-depth. Geometry-only on both sides; medians of three runs.

**Example 1 (3.2M points, architectural):**

| bits | codec | enc (ms) | dec (ms) | output | size vs tmc13 |
|---:|---|---:|---:|---:|---:|
| 10 | tmc13 | 2106 | 803 | 1.46 MB | 1.00× |
| 10 | pccl Fast | **25** | **4** | 3.47 MB | 2.38× |
| 10 | pccl Balanced | 39 | 19 | 2.63 MB | 1.80× |
| 10 | pccl Smallest | 125 | 105 | 2.21 MB | 1.52× |
| 14 | tmc13 | 4616 | 2758 | 6.50 MB | 1.00× |
| 14 | pccl Fast | **33** | **11** | 9.08 MB | 1.39× |
| 14 | pccl Balanced | 45 | 25 | 7.57 MB | 1.16× |
| 14 | pccl Smallest | 383 | 362 | 7.53 MB | 1.15× |
| 16 | tmc13 | 5719 | 3724 | 8.94 MB | 1.00× |
| 16 | pccl Fast | **32** | **12** | 12.5 MB | 1.39× |
| 16 | pccl Balanced | 32 | 13 | 9.97 MB | 1.11× |
| 16 | pccl Smallest | 426 | 404 | 10.07 MB | 1.12× |

**Example 2 (150K points with majority duplicates, LIDAR scan of a group of persons):**

| bits | codec | enc (ms) | dec (ms) | output | size vs tmc13 |
|---:|---|---:|---:|---:|---:|
| 14 | tmc13 | 65 | 27 | 52.8 KB | 1.00× |
| 14 | pccl Balanced | **6** | **0.1** | 60.7 KB | 1.14× |
| 14 | pccl Smallest | 10 | 3 | 61.3 KB | 1.16× |
| 16 | tmc13 | 71 | 33 | 67.1 KB | 1.00× |
| 16 | pccl Balanced | **8** | **0.1** | 74.8 KB | 1.11× |

## Determinism

`encode()` is deterministic at the **decoded-result** level: two encode calls
with the same input produce bitstreams that decode to the same cloud. 
The encoded bytes themselves can differ in a handful of positions when TurboPFor 
(used by the Balanced preset) picks among several legal block encodings, but the decoded result does not.
