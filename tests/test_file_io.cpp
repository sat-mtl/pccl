// Round-trip through disk: encode -> write_pccl_file -> read_pccl_file
// -> decode, and verify the decoded points match the in-memory decode.

#include "test_helpers.hpp"

#include <catch2/catch_test_macros.hpp>

#include <filesystem>
#include <fstream>

namespace fs = std::filesystem;

namespace
{

fs::path unique_tmp(const char* tag)
{
  auto p = fs::temp_directory_path() / (std::string("pccl_test_") + tag + ".pccl");
  fs::remove(p);
  return p;
}

} // namespace

TEST_CASE("file round-trip preserves encoded bytes", "[file_io]")
{
  const auto xyz = pccl_test::synth_cloud(4096, 2.f, 7);

  pccl::EncodeOptions opts;
  opts.preset = pccl::Preset::Balanced;
  opts.bits = 16;

  pccl::Encoder enc;
  pccl::ByteBuffer in_mem;
  enc.encode(std::span<const float>(xyz.data(), 4096 * 3), opts, in_mem);

  const auto path = unique_tmp("roundtrip");
  pccl::write_pccl_file(
      path, std::span<const std::uint8_t>(in_mem.data(), in_mem.size()));

  pccl::ByteBuffer from_disk;
  pccl::read_pccl_file(path, from_disk);

  REQUIRE(from_disk.size() == in_mem.size());
  CHECK(std::memcmp(from_disk.data(), in_mem.data(), in_mem.size()) == 0);

  fs::remove(path);
}

TEST_CASE("file round-trip decodes identical points", "[file_io]")
{
  const auto xyz = pccl_test::synth_cloud(8192, 3.f, 13);

  pccl::EncodeOptions opts;
  opts.preset = pccl::Preset::Smallest;
  opts.bits = 14;

  pccl::Encoder enc;
  pccl::Decoder dec;
  pccl::ByteBuffer bytes;
  enc.encode(std::span<const float>(xyz.data(), 8192 * 3), opts, bytes);

  pccl::AlignedVec<float> mem_decoded;
  pccl::FrameMetadata mem_meta;
  dec.decode(
      std::span<const std::uint8_t>(bytes.data(), bytes.size()), mem_decoded, &mem_meta);

  const auto path = unique_tmp("decode");
  pccl::write_pccl_file(path, std::span<const std::uint8_t>(bytes.data(), bytes.size()));
  pccl::ByteBuffer reloaded;
  pccl::read_pccl_file(path, reloaded);

  pccl::AlignedVec<float> disk_decoded;
  pccl::FrameMetadata disk_meta;
  dec.decode(
      std::span<const std::uint8_t>(reloaded.data(), reloaded.size()), disk_decoded,
      &disk_meta);

  REQUIRE(disk_decoded.size() == mem_decoded.size());
  CHECK(
      std::memcmp(
          disk_decoded.data(), mem_decoded.data(), mem_decoded.size() * sizeof(float))
      == 0);
  CHECK(disk_meta.bits == mem_meta.bits);
  CHECK(disk_meta.unique_voxels == mem_meta.unique_voxels);

  fs::remove(path);
}

TEST_CASE("read_pccl_file rejects non-pccl files", "[file_io]")
{
  const auto path = unique_tmp("badmagic");
  {
    std::ofstream f(path, std::ios::binary);
    const char garbage[64] = "this is not a pccl file at all xxxxxxxxxxxxxx";
    f.write(garbage, sizeof(garbage));
  }

  pccl::ByteBuffer buf;
  REQUIRE_THROWS_AS(pccl::read_pccl_file(path, buf), std::runtime_error);

  fs::remove(path);
}

TEST_CASE("read_pccl_file rejects missing files", "[file_io]")
{
  pccl::ByteBuffer buf;
  REQUIRE_THROWS_AS(
      pccl::read_pccl_file("/nonexistent/path/to/no.pccl", buf), std::runtime_error);
}

TEST_CASE("read_pccl_file rejects truncated files", "[file_io]")
{
  const auto path = unique_tmp("short");
  {
    std::ofstream f(path, std::ios::binary);
    const char tiny[4] = {'P', 'C', 'C', '1'};
    f.write(tiny, sizeof(tiny));
  }

  pccl::ByteBuffer buf;
  REQUIRE_THROWS_AS(pccl::read_pccl_file(path, buf), std::runtime_error);

  fs::remove(path);
}
