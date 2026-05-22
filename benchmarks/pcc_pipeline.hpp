#pragma once

#include "pcc_loader.hpp"

#include <algorithm>
#include <array>
#include <cassert>
#include <cstdint>
#include <cstring>
#include <vector>

namespace pcc
{

// --- Quantization ---------------------------------------------------------
//
// Map floats in [aabb_min, aabb_max] to integers in [0, (1<<bits)-1] on each
// axis. Stored as packed 32-bit triplets (x, y, z) per point.

struct QuantizedCloud
{
  std::vector<std::uint32_t> xyz; // 3 components per point
  std::uint32_t bits = 0;
  std::array<float, 3> aabb_min{};
  std::array<float, 3> aabb_max{};

  std::size_t num_points() const noexcept { return xyz.size() / 3; }
  std::size_t bytes() const noexcept { return xyz.size() * sizeof(std::uint32_t); }
};

inline QuantizedCloud quantize(const PointCloud& pc, std::uint32_t bits)
{
  assert(bits >= 1 && bits <= 21);
  QuantizedCloud q;
  q.bits = bits;
  q.aabb_min = pc.aabb_min;
  q.aabb_max = pc.aabb_max;

  const std::uint32_t maxv = (bits == 32) ? 0xFFFFFFFFu : ((1u << bits) - 1u);
  const float fmax = static_cast<float>(maxv);

  float scale[3];
  for(int k = 0; k < 3; ++k)
  {
    const float e = pc.aabb_max[k] - pc.aabb_min[k];
    scale[k] = (e > 0.f) ? (fmax / e) : 0.f;
  }

  const std::size_t n = pc.num_points();
  q.xyz.resize(n * 3);
  for(std::size_t i = 0; i < n; ++i)
  {
    const float* p = pc.xyz.data() + 3 * i;
    for(int k = 0; k < 3; ++k)
    {
      float v = (p[k] - pc.aabb_min[k]) * scale[k];
      if(v < 0.f)
        v = 0.f;
      if(v > fmax)
        v = fmax;
      q.xyz[3 * i + k] = static_cast<std::uint32_t>(v + 0.5f);
    }
  }
  return q;
}

// --- Morton encoding ------------------------------------------------------
//
// 3-D Morton (Z-order) interleaving. We use up to 21 bits per axis, so the
// resulting code fits in 63 bits and is stored as std::uint64_t.

inline std::uint64_t expand_bits_3(std::uint64_t v)
{
  v &= 0x1fffffull; // 21 bits
  v = (v | (v << 32)) & 0x001f00000000ffffull;
  v = (v | (v << 16)) & 0x001f0000ff0000ffull;
  v = (v | (v << 8)) & 0x100f00f00f00f00full;
  v = (v | (v << 4)) & 0x10c30c30c30c30c3ull;
  v = (v | (v << 2)) & 0x1249249249249249ull;
  return v;
}

inline std::uint64_t morton3(std::uint32_t x, std::uint32_t y, std::uint32_t z)
{
  return expand_bits_3(x) | (expand_bits_3(y) << 1) | (expand_bits_3(z) << 2);
}

inline std::vector<std::uint64_t> to_morton(const QuantizedCloud& q)
{
  const std::size_t n = q.num_points();
  std::vector<std::uint64_t> codes(n);
  for(std::size_t i = 0; i < n; ++i)
    codes[i] = morton3(q.xyz[3 * i], q.xyz[3 * i + 1], q.xyz[3 * i + 2]);
  return codes;
}

// --- Sort + delta + varint -----------------------------------------------
//
// After sorting in ascending Morton order, the consecutive differences are
// small for spatially-coherent clouds (typical step is 1..a few thousand).
// We encode them as unsigned LEB128 (varint) bytes.

inline void varint_encode(std::uint64_t v, std::vector<std::uint8_t>& out)
{
  while(v >= 0x80)
  {
    out.push_back(static_cast<std::uint8_t>(v | 0x80));
    v >>= 7;
  }
  out.push_back(static_cast<std::uint8_t>(v));
}

// Sort morton codes in place and produce varint-encoded deltas.
inline std::vector<std::uint8_t> sort_delta_varint(std::vector<std::uint64_t>& codes)
{
  std::sort(codes.begin(), codes.end());
  std::vector<std::uint8_t> out;
  out.reserve(codes.size() * 2);
  std::uint64_t prev = 0;
  for(std::uint64_t c : codes)
  {
    varint_encode(c - prev, out);
    prev = c;
  }
  return out;
}

// Variant: sort + delta but keep deltas as raw 64-bit ints (no varint), so
// we can also benchmark "what does a generic codec do with raw deltas vs
// raw codes?".
inline std::vector<std::uint64_t> sort_delta_raw(std::vector<std::uint64_t>& codes)
{
  std::sort(codes.begin(), codes.end());
  std::vector<std::uint64_t> d(codes.size());
  std::uint64_t prev = 0;
  for(std::size_t i = 0; i < codes.size(); ++i)
  {
    d[i] = codes[i] - prev;
    prev = codes[i];
  }
  return d;
}

// --- Byte-view helpers ----------------------------------------------------

template <class T>
inline const std::uint8_t* byte_ptr(const std::vector<T>& v)
{
  return reinterpret_cast<const std::uint8_t*>(v.data());
}

template <class T>
inline std::size_t byte_size(const std::vector<T>& v)
{
  return v.size() * sizeof(T);
}

} // namespace pcc
