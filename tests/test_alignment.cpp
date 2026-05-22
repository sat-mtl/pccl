// Smoke probe printing the alignment of the buffers the caller sees.

#include <pccl/pccl.hpp>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

int main()
{
  pccl::AlignedVec<float> xyz(3 * 8192);
  for(auto& f : xyz)
    f = (float)(std::rand() % 1000) * 0.001f;

  pccl::EncodeOptions opts;
  opts.preset = pccl::Preset::Balanced;
  opts.bits = 16;

  pccl::Encoder enc;
  pccl::Decoder dec;
  pccl::ByteBuffer bytes;
  pccl::AlignedVec<float> decoded;
  pccl::EncodeStats st;
  pccl::FrameMetadata meta;

  enc.encode(std::span<const float>(xyz.data(), 8192 * 3), opts, bytes, &st);
  dec.decode(std::span<const std::uint8_t>(bytes.data(), bytes.size()), decoded, &meta);

  std::printf("input xyz   align mod 64 = %zu\n", (std::uintptr_t)xyz.data() & 63u);
  std::printf("output bytes align mod 64 = %zu\n", (std::uintptr_t)bytes.data() & 63u);
  std::printf("decoded xyz align mod 64 = %zu\n", (std::uintptr_t)decoded.data() & 63u);
  std::printf("encoded bytes = %zu, voxels = %u\n", bytes.size(), meta.unique_voxels);
  return 0;
}
