// Encode -> decode round-trip for every preset, several bit depths, several
// cloud sizes. The decoded voxel set must match a scalar reference
// voxelisation of the input.

#include "test_helpers.hpp"

#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>

using namespace pccl_test;

namespace
{

void run_case(
    std::size_t n_points, std::uint8_t bits, pccl::Preset preset, float radius = 1.f)
{
  const auto xyz = synth_cloud(n_points, radius);

  pccl::EncodeOptions opts;
  opts.preset = preset;
  opts.bits = bits;
  opts.aabb_valid = false;

  pccl::Encoder enc;
  pccl::Decoder dec;
  pccl::ByteBuffer bytes;
  pccl::AlignedVec<float> decoded;

  pccl::EncodeStats st;
  std::size_t enc_size
      = enc.encode(std::span<const float>(xyz.data(), n_points * 3), opts, bytes, &st);
  REQUIRE(enc_size == bytes.size());
  REQUIRE(st.input_points == n_points);

  pccl::FrameMetadata meta;
  std::size_t dec_floats = dec.decode(
      std::span<const std::uint8_t>(bytes.data(), bytes.size()), decoded, &meta);
  REQUIRE(meta.bits == bits);
  REQUIRE(meta.preset == preset);
  REQUIRE(meta.input_points == n_points);
  REQUIRE(dec_floats == decoded.size());
  REQUIRE(decoded.size() == 3 * meta.unique_voxels);

  const auto want = voxelise(xyz, meta.aabb_min, meta.aabb_max, bits);
  const auto have = decoded_voxels_sorted(decoded, meta.unique_voxels);
  REQUIRE(voxels_match(want, have, meta.aabb_min, meta.aabb_max, bits));
}

} // namespace

TEST_CASE("round-trip across presets and bit depths", "[roundtrip]")
{
  const auto preset = GENERATE(
      pccl::Preset::Fast, pccl::Preset::Balanced, pccl::Preset::Small,
      pccl::Preset::Smallest);
  const std::uint8_t bits = GENERATE(8, 10, 14, 16, 20);

  SECTION("small cloud (1024 points)")
  {
    run_case(1024, bits, preset);
  }
  SECTION("medium cloud (50k points)")
  {
    run_case(50'000, bits, preset);
  }
}

TEST_CASE("round-trip at bit-depth extremes", "[roundtrip][extremes]")
{
  run_case(8192, 1, pccl::Preset::Balanced);
  run_case(8192, 21, pccl::Preset::Balanced);
}

TEST_CASE("encode with caller-supplied AABB", "[roundtrip][aabb]")
{
  const auto xyz = synth_cloud(4096, 2.f);
  pccl::EncodeOptions opts;
  opts.preset = pccl::Preset::Balanced;
  opts.bits = 16;
  opts.aabb_min = {-2.f, -2.f, -2.f};
  opts.aabb_max = {2.f, 2.f, 2.f};
  opts.aabb_valid = true;

  pccl::Encoder enc;
  pccl::Decoder dec;
  pccl::ByteBuffer bytes;
  pccl::AlignedVec<float> decoded;
  pccl::EncodeStats st;
  pccl::FrameMetadata meta;

  enc.encode(std::span<const float>(xyz.data(), 4096 * 3), opts, bytes, &st);
  REQUIRE(st.aabb_ms == 0.0); // skipped when caller provided

  dec.decode(std::span<const std::uint8_t>(bytes.data(), bytes.size()), decoded, &meta);
  REQUIRE(meta.aabb_min == opts.aabb_min);
  REQUIRE(meta.aabb_max == opts.aabb_max);
}

TEST_CASE(
    "buffer reuse: capacity preserved across same-size encodes", "[roundtrip][reuse]")
{
  const auto xyz = synth_cloud(8192, 1.f, 17);
  pccl::EncodeOptions opts;
  opts.preset = pccl::Preset::Balanced;
  opts.bits = 16;

  pccl::Encoder enc;
  pccl::ByteBuffer bytes;

  enc.encode(std::span<const float>(xyz.data(), 8192 * 3), opts, bytes);
  const std::size_t cap_after_first = bytes.capacity();
  const auto* data_after_first = bytes.data();

  for(int i = 0; i < 5; ++i)
    enc.encode(std::span<const float>(xyz.data(), 8192 * 3), opts, bytes);

  CHECK(bytes.capacity() == cap_after_first);
  CHECK(bytes.data() == data_after_first); // no reallocation
}
