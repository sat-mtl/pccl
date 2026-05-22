// NaN and +/-Inf in input must not crash or produce malformed output.
// They are silently ignored for AABB; they collapse to a valid voxel for
// quantisation. Other points must round-trip normally.

#include "test_helpers.hpp"

#include <catch2/catch_test_macros.hpp>

#include <cmath>

#include <limits>

namespace
{

constexpr float kNaN = std::numeric_limits<float>::quiet_NaN();
constexpr float kInf = std::numeric_limits<float>::infinity();

} // namespace

TEST_CASE("encode survives NaN points scattered in input", "[nonfinite]")
{
  pccl::AlignedVec<float> xyz;
  // 1000 valid points in [-1, 1]^3, with NaN points sprinkled every 17th.
  for(int i = 0; i < 1000; ++i)
  {
    if(i % 17 == 0)
    {
      xyz.insert(xyz.end(), {kNaN, kNaN, kNaN});
    }
    else
    {
      xyz.push_back((float)i * 0.001f);
      xyz.push_back((float)i * 0.002f);
      xyz.push_back((float)i * 0.003f);
    }
  }

  pccl::EncodeOptions opts;
  opts.preset = pccl::Preset::Balanced;
  opts.bits = 14;

  pccl::Encoder enc;
  pccl::Decoder dec;
  pccl::ByteBuffer bytes;
  pccl::AlignedVec<float> decoded;
  pccl::FrameMetadata meta;

  REQUIRE_NOTHROW(enc.encode(std::span<const float>(xyz), opts, bytes));
  REQUIRE_NOTHROW(dec.decode(std::span<const std::uint8_t>(bytes), decoded, &meta));

  // AABB should reflect ONLY the finite points (roughly [0, 1] x [0, 2] x [0, 3]).
  CHECK(std::isfinite(meta.aabb_min[0]));
  CHECK(std::isfinite(meta.aabb_max[0]));
  CHECK(meta.aabb_min[0] >= 0.f);
  CHECK(meta.aabb_max[0] <= 1.f);
  CHECK(meta.unique_voxels > 0);
  // Every decoded coordinate must be finite.
  for(float f : decoded)
    CHECK(std::isfinite(f));
}

TEST_CASE("encode survives +/-Inf points", "[nonfinite]")
{
  pccl::AlignedVec<float> xyz;
  for(int i = 0; i < 500; ++i)
  {
    xyz.push_back((float)i * 0.01f);
    xyz.push_back((float)i * 0.02f);
    xyz.push_back((float)i * 0.03f);
  }
  // Inject +/-Inf at scattered positions.
  xyz[3 * 100 + 0] = kInf;
  xyz[3 * 200 + 1] = -kInf;
  xyz[3 * 300 + 2] = kInf;

  pccl::EncodeOptions opts;
  opts.preset = pccl::Preset::Balanced;
  opts.bits = 14;

  pccl::Encoder enc;
  pccl::Decoder dec;
  pccl::ByteBuffer bytes;
  pccl::AlignedVec<float> decoded;
  pccl::FrameMetadata meta;

  REQUIRE_NOTHROW(enc.encode(std::span<const float>(xyz), opts, bytes));
  REQUIRE_NOTHROW(dec.decode(std::span<const std::uint8_t>(bytes), decoded, &meta));

  // AABB excludes Inf, so it stays bounded.
  for(int k = 0; k < 3; ++k)
  {
    CHECK(std::isfinite(meta.aabb_min[k]));
    CHECK(std::isfinite(meta.aabb_max[k]));
  }
  for(float f : decoded)
    CHECK(std::isfinite(f));
}

TEST_CASE("encode survives all-NaN input", "[nonfinite][edge]")
{
  pccl::AlignedVec<float> xyz(300, kNaN);

  pccl::EncodeOptions opts;
  opts.preset = pccl::Preset::Balanced;
  opts.bits = 12;

  pccl::Encoder enc;
  pccl::Decoder dec;
  pccl::ByteBuffer bytes;
  pccl::AlignedVec<float> decoded;
  pccl::FrameMetadata meta;

  REQUIRE_NOTHROW(enc.encode(std::span<const float>(xyz), opts, bytes));
  REQUIRE_NOTHROW(dec.decode(std::span<const std::uint8_t>(bytes), decoded, &meta));

  // All points collapse to a single voxel (NaN -> lo -> voxel 0).
  CHECK(meta.unique_voxels <= 1);
  for(float f : decoded)
    CHECK(std::isfinite(f));
}
