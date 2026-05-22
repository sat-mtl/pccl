// Round-trip and stats test for pccl.
//
// Loads a PLY, encodes with every preset at a few bit depths, decodes, and
// verifies that all decoded points match the voxelised input within one
// quantisation step on every axis.

#include "pcc_loader.hpp"

#include <pccl/pccl.hpp>

#include <cmath>

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <set>
#include <string>
#include <vector>

namespace
{

const char* preset_name(pccl::Preset p)
{
  switch(p)
  {
    case pccl::Preset::Fast:
      return "Fast";
    case pccl::Preset::Balanced:
      return "Balanced";
    case pccl::Preset::Small:
      return "Small";
    case pccl::Preset::Smallest:
      return "Smallest";
  }
  return "?";
}

struct V3
{
  float x, y, z;
  bool operator<(const V3& o) const
  {
    if(x != o.x)
      return x < o.x;
    if(y != o.y)
      return y < o.y;
    return z < o.z;
  }
  bool operator==(const V3& o) const { return x == o.x && y == o.y && z == o.z; }
};

// Voxelise the input the same way pccl does, in floats, so we can compare.
pccl::AlignedVec<V3> voxelise_reference(const pcc::PointCloud& pc, std::uint32_t bits)
{
  const std::uint32_t maxv = (1u << bits) - 1u;
  const float fmax = (float)maxv;
  float scale[3], invscale[3];
  for(int k = 0; k < 3; ++k)
  {
    float e = pc.aabb_max[k] - pc.aabb_min[k];
    scale[k] = (e > 0.f) ? (fmax / e) : 0.f;
    invscale[k] = (fmax > 0.f && e > 0.f) ? (e / fmax) : 0.f;
  }
  std::set<std::array<std::uint32_t, 3>> seen;
  const std::size_t n = pc.num_points();
  for(std::size_t i = 0; i < n; ++i)
  {
    const float* p = pc.xyz.data() + 3 * i;
    std::array<std::uint32_t, 3> q;
    for(int k = 0; k < 3; ++k)
    {
      float v = (p[k] - pc.aabb_min[k]) * scale[k];
      if(v < 0.f)
        v = 0.f;
      if(v > fmax)
        v = fmax;
      q[k] = (std::uint32_t)(v + 0.5f);
    }
    seen.insert(q);
  }
  pccl::AlignedVec<V3> out;
  out.reserve(seen.size());
  for(auto& q : seen)
    out.push_back(
        {pc.aabb_min[0] + q[0] * invscale[0], pc.aabb_min[1] + q[1] * invscale[1],
         pc.aabb_min[2] + q[2] * invscale[2]});
  std::sort(out.begin(), out.end());
  return out;
}

bool verify_match(
    const pccl::AlignedVec<V3>& want, const pccl::AlignedVec<float>& decoded,
    const pccl::FrameMetadata& meta)
{
  if(decoded.size() / 3 != want.size())
  {
    std::printf(
        "    FAIL: voxel count %zu (expected %zu)\n", decoded.size() / 3, want.size());
    return false;
  }
  pccl::AlignedVec<V3> have(want.size());
  for(std::size_t i = 0; i < want.size(); ++i)
    have[i] = {decoded[3 * i + 0], decoded[3 * i + 1], decoded[3 * i + 2]};
  std::sort(have.begin(), have.end());

  float tol[3];
  for(int k = 0; k < 3; ++k)
  {
    float e = meta.aabb_max[k] - meta.aabb_min[k];
    float fmax = (float)((1u << meta.bits) - 1u);
    tol[k] = 0.501f * (e / fmax);
  }
  for(std::size_t i = 0; i < want.size(); ++i)
  {
    float dx = std::fabs(have[i].x - want[i].x);
    float dy = std::fabs(have[i].y - want[i].y);
    float dz = std::fabs(have[i].z - want[i].z);
    if(dx > tol[0] || dy > tol[1] || dz > tol[2])
    {
      std::printf(
          "    FAIL voxel %zu: have (%.6g %.6g %.6g) want (%.6g %.6g %.6g)\n", i,
          have[i].x, have[i].y, have[i].z, want[i].x, want[i].y, want[i].z);
      return false;
    }
  }
  return true;
}

void run_one(
    pccl::Encoder& enc, pccl::Decoder& dec, pccl::ByteBuffer& bytes,
    pccl::AlignedVec<float>& decoded, const pcc::PointCloud& pc,
    const pccl::AlignedVec<V3>& ref, pccl::Preset preset, std::uint32_t bits)
{
  pccl::EncodeOptions opts;
  opts.preset = preset;
  opts.bits = (std::uint8_t)bits;
  opts.aabb_valid = false;

  pccl::EncodeStats st;
  enc.encode(
      std::span<const float>(pc.xyz.data(), pc.num_points() * 3), opts, bytes, &st);

  pccl::FrameMetadata meta;
  auto t0 = std::chrono::high_resolution_clock::now();
  dec.decode(std::span<const std::uint8_t>(bytes.data(), bytes.size()), decoded, &meta);
  double dec_ms = std::chrono::duration<double, std::milli>(
                      std::chrono::high_resolution_clock::now() - t0)
                      .count();

  bool ok = verify_match(ref, decoded, meta);
  std::printf(
      "  %-9s bits=%-3u  out=%10zu B  %.3f B/pt  ratio=%.4f  "
      "enc=%6.1f [aabb=%.1f quant=%.1f sort=%.1f dedup=%.1f codec=%.1f] "
      "dec=%6.1f voxels=%zu  %s\n",
      preset_name(preset), bits, st.output_bytes, st.bytes_per_point(), st.ratio(),
      st.encode_ms, st.aabb_ms, st.quant_ms, st.sort_ms, st.dedup_ms, st.codec_ms,
      dec_ms, st.unique_voxels, ok ? "OK" : "MISMATCH");
}

} // namespace

int main(int argc, char** argv)
{
  if(argc < 2)
  {
    std::fprintf(stderr, "usage: %s <file.ply> [bits1 bits2 ...]\n", argv[0]);
    return 1;
  }

  pccl::AlignedVec<std::uint32_t> bits_list;
  for(int i = 2; i < argc; ++i)
    bits_list.push_back((std::uint32_t)std::stoul(argv[i]));
  if(bits_list.empty())
    bits_list = {10u, 14u, 16u};

  std::printf("loading %s ...\n", argv[1]);
  auto pc = pcc::load_ply_positions(argv[1]);
  std::printf(
      "loaded %zu points  aabb [%.3f %.3f %.3f] -> [%.3f %.3f %.3f]\n", pc.num_points(),
      pc.aabb_min[0], pc.aabb_min[1], pc.aabb_min[2], pc.aabb_max[0], pc.aabb_max[1],
      pc.aabb_max[2]);

  pccl::Encoder enc;
  pccl::Decoder dec;
  pccl::ByteBuffer bytes;
  pccl::AlignedVec<float> decoded;

  const auto presets
      = {pccl::Preset::Fast, pccl::Preset::Balanced, pccl::Preset::Small,
         pccl::Preset::Smallest};
  for(auto bits : bits_list)
  {
    auto ref = voxelise_reference(pc, bits);
    std::printf("\n## bits=%u   reference voxel count = %zu\n", bits, ref.size());
    for(auto pr : presets)
      run_one(enc, dec, bytes, decoded, pc, ref, pr, bits);
  }
  return 0;
}
