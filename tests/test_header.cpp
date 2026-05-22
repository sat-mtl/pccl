// peek_header should expose the same metadata the bitstream carries, and
// reject non-pccl input.

#include "test_helpers.hpp"

#include <catch2/catch_test_macros.hpp>

using namespace pccl_test;

TEST_CASE("peek_header round-trips encode metadata", "[header]")
{
  const auto xyz = synth_cloud(2048, 3.f, 7);
  pccl::EncodeOptions opts;
  opts.preset = pccl::Preset::Small;
  opts.bits = 12;
  opts.aabb_min = {-3.f, -3.f, -3.f};
  opts.aabb_max = {3.f, 3.f, 3.f};
  opts.aabb_valid = true;

  pccl::Encoder enc;
  pccl::ByteBuffer bytes;
  enc.encode(std::span<const float>(xyz.data(), 2048 * 3), opts, bytes);

  pccl::FrameHeader h;
  REQUIRE(
      pccl::peek_header(std::span<const std::uint8_t>(bytes.data(), bytes.size()), h));
  CHECK(h.preset == (std::uint8_t)pccl::Preset::Small);
  CHECK(h.bits == 12);
  CHECK(h.input_points == 2048);
  CHECK(h.unique_voxels > 0);
  CHECK(h.payload_size > 0);
  CHECK(h.aabb_min == opts.aabb_min);
  CHECK(h.aabb_max == opts.aabb_max);
  CHECK(bytes.size() == 44 + h.payload_size);
}

TEST_CASE("peek_header rejects too-short input", "[header]")
{
  pccl::FrameHeader h;
  std::array<std::uint8_t, 10> tiny{};
  CHECK_FALSE(
      pccl::peek_header(std::span<const std::uint8_t>(tiny.data(), tiny.size()), h));
}

TEST_CASE("peek_header rejects bad magic", "[header]")
{
  std::array<std::uint8_t, 64> buf{};
  for(int i = 0; i < 4; ++i)
    buf[i] = 'X';
  pccl::FrameHeader h;
  CHECK_FALSE(
      pccl::peek_header(std::span<const std::uint8_t>(buf.data(), buf.size()), h));
}
