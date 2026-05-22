// Point cloud compression benchmark.
//
// Pipeline stages (incremental):
//   stage 0 : raw float xyz                       (12 B / point)
//   stage 1 : quantize to N bits per axis         (3 x u32 packed)
//   stage 2 : Morton interleave to 64-bit code    (8 B / point)
//             + integer-codec column: feed the *sorted* Morton stream
//             directly into streamvbyte_delta / TurboPFor (they do delta
//             + bitpacking / PFor internally).
//   stage 3a: sort + delta as raw u64
//   stage 3b: sort + delta + varint (LEB128)
//   stage 4 : octree occupancy bytes (one byte per internal node)
//
// At each stage we feed the bytes to every generic byte codec and report
// bytes, ratio vs raw, encode time, throughput.

#include "pcc_codec.hpp"
#include "pcc_codec_extra.hpp"
#include "pcc_loader.hpp"
#include "pcc_octree.hpp"
#include "pcc_pipeline.hpp"

#include <cstdio>
#include <functional>
#include <string>
#include <vector>

using namespace pcc;

namespace
{

struct Codec
{
  std::string name;
  std::function<CodecResult(const std::uint8_t*, std::size_t)> fn;
};

std::vector<Codec> byte_codecs()
{
  return {
      {"raw", [](auto p, auto n) { return c_raw(p, n); }},
      {"lz4", [](auto p, auto n) { return c_lz4(p, n); }},
      {"lz4hc-9", [](auto p, auto n) { return c_lz4hc(p, n, 9); }},
      {"snappy", [](auto p, auto n) { return c_snappy(p, n); }},
      {"lzo1x-1", [](auto p, auto n) { return c_lzo(p, n); }},
      {"zlib-6", [](auto p, auto n) { return c_zlib(p, n, 6); }},
      {"zlib-9", [](auto p, auto n) { return c_zlib(p, n, 9); }},
      {"zstd-3", [](auto p, auto n) { return c_zstd(p, n, 3); }},
      {"zstd-9", [](auto p, auto n) { return c_zstd(p, n, 9); }},
      {"zstd-19", [](auto p, auto n) { return c_zstd(p, n, 19); }},
      {"brotli-q5", [](auto p, auto n) { return c_brotli(p, n, 5); }},
      {"brotli-q9", [](auto p, auto n) { return c_brotli(p, n, 9); }},
      {"fse", [](auto p, auto n) { return c_fse(p, n); }},
      {"rans-o0", [](auto p, auto n) { return c_rans_o0(p, n); }},
  };
}

void print_table_header(const char* stage)
{
  std::printf("\n=== %s ===\n", stage);
  std::printf(
      "  %-14s %12s %10s %12s %12s\n", "codec", "bytes", "ratio", "encode_ms", "MB/s");
  std::printf("  %s\n", std::string(64, '-').c_str());
}

void print_row(const CodecResult& r, std::size_t baseline_bytes)
{
  const double r_vs_raw
      = baseline_bytes == 0 ? 0.0 : double(r.output_bytes) / double(baseline_bytes);
  std::printf(
      "  %-14s %12zu %10.4f %12.2f %12.1f\n", r.name.c_str(), r.output_bytes, r_vs_raw,
      r.encode_ms, r.mb_per_s());
}

struct Stage
{
  std::string label;
  const std::uint8_t* data;
  std::size_t size;
};

void run_byte_stage(
    const Stage& s, std::size_t baseline_bytes, const std::vector<Codec>& codecs)
{
  print_table_header(s.label.c_str());
  for(auto& c : codecs)
  {
    try
    {
      auto r = c.fn(s.data, s.size);
      print_row(r, baseline_bytes);
    }
    catch(const std::exception& e)
    {
      std::printf("  %-14s  ERROR: %s\n", c.name.c_str(), e.what());
    }
  }
}

// Apply integer codecs to the *sorted* Morton u64 stream.  They internally do
// delta + PFor / bitpacking, so they replace stages 3a+3b+backend in one call.
void run_morton_int_stage(
    const std::vector<std::uint64_t>& sorted_codes, std::uint32_t bits_per_axis,
    std::size_t baseline_bytes)
{
  const std::size_t n = sorted_codes.size();
  print_table_header("stage 2-int : sorted morton -> integer codec");

  try
  {
    auto r = c_streamvbyte_delta_u64(sorted_codes.data(), n);
    print_row(r, baseline_bytes);
  }
  catch(const std::exception& e)
  {
    std::printf("  svb_delta ERR %s\n", e.what());
  }

  // p4ndenc64 expects sorted, non-strict.  Codes are sorted (and we kept the
  // duplicates so the cloud is fully recoverable).
  try
  {
    auto r = c_turbopfor_p4d_u64(sorted_codes.data(), n);
    print_row(r, baseline_bytes);
  }
  catch(const std::exception& e)
  {
    std::printf("  p4ndenc64 ERR %s\n", e.what());
  }

  // SIMD 32-bit kernels: only meaningful when the morton codes themselves fit
  // in 32 bits (i.e. bits <= 10 -> 30-bit codes).
  if(bits_per_axis <= 10)
  {
    std::vector<std::uint32_t> codes32(n);
    for(std::size_t i = 0; i < n; ++i)
      codes32[i] = (std::uint32_t)sorted_codes[i];

    try
    {
      auto r = c_streamvbyte_delta_u32(codes32.data(), n);
      r.name = "svb_delta_u32";
      print_row(r, baseline_bytes);
    }
    catch(const std::exception& e)
    {
      std::printf("  svb_u32 ERR %s\n", e.what());
    }

    try
    {
      auto r = c_turbopfor_p4d_u32(codes32.data(), n);
      print_row(r, baseline_bytes);
    }
    catch(const std::exception& e)
    {
      std::printf("  p4ndenc128v32 ERR %s\n", e.what());
    }
  }
  else
  {
    // For wider codes use the hi32/lo32 split with the SIMD 32-bit kernels.
    try
    {
      auto r = c_turbopfor_p4d_u64_split32(sorted_codes.data(), n);
      print_row(r, baseline_bytes);
    }
    catch(const std::exception& e)
    {
      std::printf("  p4nd_128v32x2 ERR %s\n", e.what());
    }
  }
}

} // namespace

int main(int argc, char** argv)
{
  if(argc < 2)
  {
    std::fprintf(stderr, "usage: %s <file.ply> [bits1 bits2 ...]\n", argv[0]);
    return 1;
  }

  std::vector<std::uint32_t> bits_list;
  for(int i = 2; i < argc; ++i)
    bits_list.push_back(static_cast<std::uint32_t>(std::stoul(argv[i])));
  if(bits_list.empty())
    bits_list = {10u, 12u, 14u, 16u, 20u};

  std::printf("loading %s ...\n", argv[1]);
  auto t0 = clk::now();
  auto pc = load_ply_positions(argv[1]);
  std::printf("loaded %zu points in %.1f ms\n", pc.num_points(), ms_since(t0));
  std::printf(
      "aabb : [%.3f %.3f %.3f] -> [%.3f %.3f %.3f]\n", pc.aabb_min[0], pc.aabb_min[1],
      pc.aabb_min[2], pc.aabb_max[0], pc.aabb_max[1], pc.aabb_max[2]);

  const std::size_t raw_bytes = pc.bytes();
  std::printf(
      "raw input  : %zu bytes (%.2f MB, %.2f B/point)\n", raw_bytes,
      raw_bytes / (1024.0 * 1024.0), double(raw_bytes) / pc.num_points());

  const auto codecs = byte_codecs();

  // Stage 0 — raw float xyz.
  {
    Stage s{"stage 0 : raw float xyz", byte_ptr(pc.xyz), byte_size(pc.xyz)};
    run_byte_stage(s, raw_bytes, codecs);
  }

  for(std::uint32_t bits : bits_list)
  {
    std::printf("\n#### quantization bits per axis : %u\n", bits);

    auto tq = clk::now();
    auto q = quantize(pc, bits);
    const double q_ms = ms_since(tq);
    std::printf("quantize        : %.1f ms\n", q_ms);

    // Stage 1 — quantized packed as 3 x uint32.
    {
      Stage s{"stage 1 : quantize (3 x u32)", byte_ptr(q.xyz), byte_size(q.xyz)};
      run_byte_stage(s, raw_bytes, codecs);
    }

    // Stage 2 — Morton interleave (u64 per point).
    auto tm = clk::now();
    auto codes = to_morton(q);
    std::sort(codes.begin(), codes.end());
    const double m_ms = ms_since(tm);
    std::printf("morton+sort     : %.1f ms\n", m_ms);
    {
      Stage s{"stage 2 : morton u64 (sorted)", byte_ptr(codes), byte_size(codes)};
      run_byte_stage(s, raw_bytes, codecs);
    }

    // Stage 2-int — feed sorted Morton stream to integer codecs directly.
    run_morton_int_stage(codes, bits, raw_bytes);

    // Stage 3a / 3b — delta + varint backends.
    auto td = clk::now();
    auto codes2 = codes; // sort_delta_raw sorts again (cheap on already-sorted)
    auto raw_deltas = sort_delta_raw(codes2);
    const double dr_ms = ms_since(td);

    auto tv = clk::now();
    auto varint = sort_delta_varint(codes);
    const double dv_ms = ms_since(tv);
    std::printf("sort+delta u64  : %.1f ms\n", dr_ms);
    std::printf(
        "sort+delta vint : %.1f ms  (-> %zu bytes, %.2f B/point)\n", dv_ms,
        varint.size(), double(varint.size()) / pc.num_points());

    {
      Stage s{
          "stage 3a : sort+delta (u64)", byte_ptr(raw_deltas), byte_size(raw_deltas)};
      run_byte_stage(s, raw_bytes, codecs);
    }
    {
      Stage s{"stage 3b : sort+delta+varint", byte_ptr(varint), byte_size(varint)};
      run_byte_stage(s, raw_bytes, codecs);
    }

    // Stage 4 — octree occupancy bytes.
    OctreeStats ostats;
    auto octree_bytes = build_octree_from_quantized(q, ostats);
    std::printf(
        "octree          : %.1f ms  (%zu unique voxels, %zu bytes, %.2f B/point)\n",
        ostats.build_ms, ostats.unique_voxels, ostats.occupancy_bytes,
        double(ostats.occupancy_bytes) / pc.num_points());
    {
      Stage s{
          "stage 4 : octree occupancy bytes", octree_bytes.data(), octree_bytes.size()};
      run_byte_stage(s, raw_bytes, codecs);
    }
  }

  return 0;
}
