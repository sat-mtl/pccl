// pccl is deterministic at the DECODED level: two encode calls with the
// same input produce bitstreams that decode to the same float buffer. The
// encoded bytes themselves can differ in a few positions when TurboPFor
// (Balanced preset) picks among several legal encodings.

#include "test_helpers.hpp"

#include <catch2/catch_test_macros.hpp>

namespace
{

std::uint64_t fnv1a_64(const std::uint8_t* p, std::size_t n)
{
  std::uint64_t h = 14695981039346656037ull;
  for(std::size_t i = 0; i < n; ++i)
  {
    h ^= p[i];
    h *= 1099511628211ull;
  }
  return h;
}

std::uint64_t hash_decoded(pccl::Decoder& dec, const pccl::ByteBuffer& bytes)
{
  pccl::AlignedVec<float> xyz;
  dec.decode(std::span<const std::uint8_t>(bytes.data(), bytes.size()), xyz);
  return fnv1a_64(
      reinterpret_cast<const std::uint8_t*>(xyz.data()), xyz.size() * sizeof(float));
}

} // namespace

TEST_CASE("decoded result is deterministic for the same input", "[determinism]")
{
  const auto xyz = pccl_test::synth_cloud(10'000, 4.f, 42);

  pccl::EncodeOptions opts;
  opts.preset = pccl::Preset::Balanced;
  opts.bits = 16;
  opts.aabb_valid = false;

  pccl::Encoder enc;
  pccl::Decoder dec;
  pccl::ByteBuffer a, b, c;
  enc.encode(std::span<const float>(xyz.data(), 10'000 * 3), opts, a);
  enc.encode(std::span<const float>(xyz.data(), 10'000 * 3), opts, b);
  enc.encode(std::span<const float>(xyz.data(), 10'000 * 3), opts, c);

  REQUIRE(a.size() == b.size());
  REQUIRE(a.size() == c.size());
  const auto h = hash_decoded(dec, a);
  CHECK(hash_decoded(dec, b) == h);
  CHECK(hash_decoded(dec, c) == h);
}

TEST_CASE("decoded result is deterministic across presets", "[determinism]")
{
  const auto xyz = pccl_test::synth_cloud(4096, 1.f, 99);
  pccl::Encoder enc;
  pccl::Decoder dec;

  for(auto p :
      {pccl::Preset::Fast, pccl::Preset::Balanced, pccl::Preset::Small,
       pccl::Preset::Smallest})
  {
    pccl::EncodeOptions opts;
    opts.preset = p;
    opts.bits = 14;

    pccl::ByteBuffer a, b;
    enc.encode(std::span<const float>(xyz.data(), 4096 * 3), opts, a);
    enc.encode(std::span<const float>(xyz.data(), 4096 * 3), opts, b);
    REQUIRE(a.size() == b.size());
    CHECK(hash_decoded(dec, a) == hash_decoded(dec, b));
  }
}
