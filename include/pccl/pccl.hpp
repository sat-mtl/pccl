// Stateful Encoder / Decoder. Reuse one instance + one output buffer
// across frames; steady-state streaming allocates nothing.
//
// Frame format (little-endian):
//   00  4   magic 'P','C','C','1'
//   04  1   preset (0..3)
//   05  1   bits per axis (1..21)
//   06  2   flags (reserved, 0)
//   08  4   input_points
//   12  4   unique_voxels
//   16  24  aabb: 6 float32 (min[3], max[3])
//   40  4   payload_size
//   44  ... payload (preset-specific, opaque)

#pragma once

#include <boost/container/vector.hpp>

#include <pccl/pccl_export.hpp>

#include <array>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <span>

#if defined(_WIN32)
#  include <malloc.h>  // _aligned_malloc / _aligned_free
#else
#  include <cstdlib>   // std::aligned_alloc / std::free
#endif
#include <vector>

namespace pccl
{
constexpr std::size_t kSimdAlign = 64;

template <class T>
struct AlignedAllocator
{
  using value_type = T;
  using propagate_on_container_move_assignment = std::true_type;
  using is_always_equal = std::true_type;

  constexpr AlignedAllocator() noexcept = default;
  template <class U>
  constexpr AlignedAllocator(const AlignedAllocator<U>&) noexcept
  {
  }

  T* allocate(std::size_t n)
  {
    if(n == 0)
      return nullptr;
    std::size_t bytes = n * sizeof(T);
    bytes = (bytes + kSimdAlign - 1) & ~(kSimdAlign - 1);
    void* p =
#if defined(_WIN32)
        ::_aligned_malloc(bytes, kSimdAlign)
#else
        std::aligned_alloc(kSimdAlign, bytes)
#endif
        ;
    return static_cast<T*>(p); // null on OOM (intentional, lets the caller fault)
  }

  void deallocate(T* p, std::size_t) noexcept
  {
#if defined(_WIN32)
    ::_aligned_free(p);
#else
    std::free(p);
#endif
  }

  template <class U>
  void construct(U* p) noexcept(std::is_nothrow_default_constructible_v<U>)
  {
    if constexpr(!std::is_trivially_default_constructible_v<U>)
      ::new(static_cast<void*>(p)) U();
    (void)p;
  }
  template <class U>
  void construct(U* p, boost::container::default_init_t) noexcept(
      std::is_nothrow_default_constructible_v<U>)
  {
    if constexpr(!std::is_trivially_default_constructible_v<U>)
      ::new(static_cast<void*>(p)) U();
    (void)p;
  }
  template <class U, class A0, class... Args>
  void construct(U* p, A0&& a0, Args&&... args)
  {
    ::new(static_cast<void*>(p)) U(std::forward<A0>(a0), std::forward<Args>(args)...);
  }
  template <class U>
  void destroy(U* p) noexcept
  {
    if constexpr(!std::is_trivially_destructible_v<U>)
      p->~U();
    (void)p;
  }
};
template <class T, class U>
constexpr bool
operator==(const AlignedAllocator<T>&, const AlignedAllocator<U>&) noexcept
{
  return true;
}
template <class T, class U>
constexpr bool
operator!=(const AlignedAllocator<T>&, const AlignedAllocator<U>&) noexcept
{
  return false;
}

template <class T>
using AlignedVec = boost::container::vector<T, AlignedAllocator<T>>;

using ByteBuffer = AlignedVec<std::uint8_t>;

enum class Preset : std::uint8_t
{
  Fast = 0,
  Balanced = 1,
  Small = 2,
  Smallest = 3,
};

struct EncodeOptions
{
  Preset preset = Preset::Balanced;
  std::uint8_t bits = 16; // 1..21

  // For streaming, pass a fixed AABB so consecutive frames share the
  // quantisation grid. Without it the encoder computes one per call and
  // decoded coordinates won't line up across frames.
  std::array<float, 3> aabb_min{};
  std::array<float, 3> aabb_max{};
  bool aabb_valid = false;
};

struct EncodeStats
{
  std::size_t input_points = 0;
  std::size_t input_bytes = 0;  // = input_points * 12
  std::size_t output_bytes = 0; // header + payload
  std::size_t unique_voxels = 0;
  double encode_ms = 0.0;

  // Per-stage timing (sums to ~encode_ms).
  double aabb_ms = 0.0;
  double quant_ms = 0.0;
  double sort_ms = 0.0;
  double dedup_ms = 0.0;
  double codec_ms = 0.0;

  double ratio() const noexcept
  {
    return input_bytes == 0 ? 1.0 : double(output_bytes) / double(input_bytes);
  }
  double mb_per_s() const noexcept
  {
    return (encode_ms <= 0.0)
               ? 0.0
               : (double(input_bytes) / (1024.0 * 1024.0)) / (encode_ms / 1000.0);
  }
  double bytes_per_point() const noexcept
  {
    return input_points == 0 ? 0.0 : double(output_bytes) / double(input_points);
  }
};

struct FrameMetadata
{
  Preset preset = Preset::Balanced;
  std::uint8_t bits = 0;
  std::uint32_t input_points = 0;
  std::uint32_t unique_voxels = 0;
  std::array<float, 3> aabb_min{};
  std::array<float, 3> aabb_max{};
};

struct FrameHeader
{
  std::uint8_t preset = 0;
  std::uint8_t bits = 0;
  std::uint32_t input_points = 0;
  std::uint32_t unique_voxels = 0;
  std::array<float, 3> aabb_min{};
  std::array<float, 3> aabb_max{};
  std::uint32_t payload_size = 0;
};

class PCCL_API Encoder
{
public:
  Encoder();
  ~Encoder();

  Encoder(const Encoder&) = delete;
  Encoder& operator=(const Encoder&) = delete;
  Encoder(Encoder&&) noexcept;
  Encoder& operator=(Encoder&&) noexcept;

  // xyz size must be a multiple of 3. Returns out.size()
  std::size_t encode(
      std::span<const float> xyz, const EncodeOptions& opts, ByteBuffer& out,
      EncodeStats* stats = nullptr);

  void reserve(std::size_t expected_points);

  struct Impl;

private:
  std::unique_ptr<Impl> impl_;
};

class PCCL_API Decoder
{
public:
  Decoder();
  ~Decoder();

  Decoder(const Decoder&) = delete;
  Decoder& operator=(const Decoder&) = delete;
  Decoder(Decoder&&) noexcept;
  Decoder& operator=(Decoder&&) noexcept;

  // Returns the number of floats written (= 3 * unique_voxels).
  // Throws std::runtime_error on malformed input.
  std::size_t decode(
      std::span<const std::uint8_t> data, AlignedVec<float>& out_xyz,
      FrameMetadata* out_meta = nullptr);

  void reserve(std::size_t expected_voxels);

  struct Impl;

private:
  std::unique_ptr<Impl> impl_;
};

// Parses the 44-byte header. Does not allocate or touch the payload.
PCCL_API bool peek_header(std::span<const std::uint8_t> data, FrameHeader& out) noexcept;

// .pccl on disk is the raw bitstream
// Both throw std::runtime_error on I/O failure or bad magic.
PCCL_API void
write_pccl_file(const std::filesystem::path& path, std::span<const std::uint8_t> data);
PCCL_API void read_pccl_file(const std::filesystem::path& path, ByteBuffer& out);

}
