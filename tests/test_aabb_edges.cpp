// Edge cases for the implicit AABB computation. The encoder must produce a
// usable bitstream for empty, single-point, and degenerate-axis inputs.

#include "test_helpers.hpp"

#include <catch2/catch_test_macros.hpp>

TEST_CASE("empty cloud encodes and decodes", "[aabb][empty]")
{
  pccl::EncodeOptions opts;
  opts.preset = pccl::Preset::Balanced;
  opts.bits = 16;
  opts.aabb_valid = false;

  pccl::Encoder enc;
  pccl::Decoder dec;
  pccl::ByteBuffer bytes;
  pccl::AlignedVec<float> decoded;
  pccl::EncodeStats st;
  pccl::FrameMetadata meta;

  enc.encode(std::span<const float>{}, opts, bytes, &st);
  REQUIRE(st.input_points == 0);
  REQUIRE(st.unique_voxels == 0);

  dec.decode(std::span<const std::uint8_t>(bytes.data(), bytes.size()), decoded, &meta);
  CHECK(meta.unique_voxels == 0);
  CHECK(decoded.empty());
}

TEST_CASE("single point", "[aabb][edge]")
{
  pccl::AlignedVec<float> xyz{1.5f, -2.0f, 7.25f};
  pccl::EncodeOptions opts;
  opts.preset = pccl::Preset::Balanced;
  opts.bits = 16;
  opts.aabb_valid = false;

  pccl::Encoder enc;
  pccl::Decoder dec;
  pccl::ByteBuffer bytes;
  pccl::AlignedVec<float> decoded;
  pccl::FrameMetadata meta;

  enc.encode(std::span<const float>(xyz.data(), 1 * 3), opts, bytes);
  dec.decode(std::span<const std::uint8_t>(bytes.data(), bytes.size()), decoded, &meta);
  REQUIRE(meta.unique_voxels == 1);
  CHECK(decoded[0] == 1.5f);
  CHECK(decoded[1] == -2.0f);
  CHECK(decoded[2] == 7.25f);
}

TEST_CASE("planar cloud (degenerate Z axis)", "[aabb][edge][degenerate]")
{
  pccl::AlignedVec<float> xyz;
  for(int i = 0; i < 256; ++i)
  {
    xyz.push_back((float)i / 16.f);
    xyz.push_back((float)i / 32.f);
    xyz.push_back(0.f);
  }
  pccl::EncodeOptions opts;
  opts.preset = pccl::Preset::Balanced;
  opts.bits = 14;

  pccl::Encoder enc;
  pccl::Decoder dec;
  pccl::ByteBuffer bytes;
  pccl::AlignedVec<float> decoded;
  pccl::FrameMetadata meta;

  enc.encode(std::span<const float>(xyz.data(), 256 * 3), opts, bytes);
  dec.decode(std::span<const std::uint8_t>(bytes.data(), bytes.size()), decoded, &meta);
  REQUIRE(meta.unique_voxels > 0);
  for(std::size_t i = 0; i < meta.unique_voxels; ++i)
    CHECK(std::isfinite(decoded[3 * i + 2]));
}

TEST_CASE("collinear cloud (two degenerate axes)", "[aabb][edge]")
{
  pccl::AlignedVec<float> xyz;
  for(int i = 0; i < 64; ++i)
  {
    xyz.push_back((float)i);
    xyz.push_back(3.f);
    xyz.push_back(-1.f);
  }
  pccl::EncodeOptions opts;
  opts.preset = pccl::Preset::Balanced;
  opts.bits = 14;

  pccl::Encoder enc;
  pccl::Decoder dec;
  pccl::ByteBuffer bytes;
  pccl::AlignedVec<float> decoded;
  pccl::FrameMetadata meta;

  enc.encode(std::span<const float>(xyz.data(), 64 * 3), opts, bytes);
  dec.decode(std::span<const std::uint8_t>(bytes.data(), bytes.size()), decoded, &meta);
  REQUIRE(meta.unique_voxels > 0);
}
