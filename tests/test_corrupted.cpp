// decode() must throw on a truncated or corrupted bitstream rather than
// crashing or reading out of bounds.

#include "test_helpers.hpp"

#include <catch2/catch_test_macros.hpp>

#include <stdexcept>

namespace
{

pccl::ByteBuffer good_frame()
{
  const auto xyz = pccl_test::synth_cloud(1024, 1.f, 7);
  pccl::EncodeOptions opts;
  opts.preset = pccl::Preset::Balanced;
  opts.bits = 16;
  pccl::Encoder enc;
  pccl::ByteBuffer out;
  enc.encode(std::span<const float>(xyz.data(), 1024 * 3), opts, out);
  return out;
}

} // namespace

TEST_CASE("decode rejects truncated header", "[corrupted]")
{
  pccl::Decoder dec;
  pccl::AlignedVec<float> out;
  auto bytes = good_frame();
  bytes.resize(10);
  CHECK_THROWS_AS(
      dec.decode(std::span<const std::uint8_t>(bytes.data(), bytes.size()), out),
      std::runtime_error);
}

TEST_CASE("decode rejects truncated payload", "[corrupted]")
{
  pccl::Decoder dec;
  pccl::AlignedVec<float> out;
  auto bytes = good_frame();
  bytes.resize(50);
  CHECK_THROWS_AS(
      dec.decode(std::span<const std::uint8_t>(bytes.data(), bytes.size()), out),
      std::runtime_error);
}

TEST_CASE("decode rejects bad magic", "[corrupted]")
{
  pccl::Decoder dec;
  pccl::AlignedVec<float> out;
  auto bytes = good_frame();
  bytes[0] = 'X';
  CHECK_THROWS_AS(
      dec.decode(std::span<const std::uint8_t>(bytes.data(), bytes.size()), out),
      std::runtime_error);
}

TEST_CASE("decode rejects unknown preset", "[corrupted]")
{
  pccl::Decoder dec;
  pccl::AlignedVec<float> out;
  auto bytes = good_frame();
  bytes[4] = 99;
  CHECK_THROWS_AS(
      dec.decode(std::span<const std::uint8_t>(bytes.data(), bytes.size()), out),
      std::runtime_error);
}

TEST_CASE("decode rejects out-of-range bits", "[corrupted]")
{
  pccl::Decoder dec;
  pccl::AlignedVec<float> out;
  auto bytes = good_frame();
  bytes[5] = 30;
  CHECK_THROWS_AS(
      dec.decode(std::span<const std::uint8_t>(bytes.data(), bytes.size()), out),
      std::runtime_error);
}
