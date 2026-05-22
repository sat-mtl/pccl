#pragma once

#include <xsimd/xsimd.hpp>

#include <ips4o.hpp>

#if defined(__BMI2__) && (defined(__x86_64__) || defined(_M_X64))
#include <immintrin.h>
#define PCCL_HAS_PDEP 1
#else
#define PCCL_HAS_PDEP 0
#endif

#include <cmath>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>

namespace pccl::simd
{
XSIMD_INLINE bool safe_isfinite(double val) noexcept
{
#if __FINITE_MATH_ONLY__
#if defined(_MSC_VER)
  return safe_isfinite(val);
#elif defined(__APPLE__)
  return __isfinited(val);
#elif defined(__EMSCRIPTEN__)
  return __fpclassifyl(val) != FP_NAN && __fpclassifyl(val) != FP_INFINITE;
#else
  union
  {
    double fp;
    uint64_t bits;
  } num{.fp = val};
  return ((unsigned)(num.bits >> 32) & 0x7fffffff) < 0x7ff00000;
#endif
#else
  return std::isfinite(val);
#endif
}

namespace xs = xsimd;

using ArchF = xs::default_arch;
using BatchF = xs::batch<float, ArchF>;
using BatchI = xs::batch<std::int32_t, ArchF>;
using BatchU = xs::batch<std::uint32_t, ArchF>;
using BatchU64 = xs::batch<std::uint64_t, ArchF>;

static constexpr std::size_t kLanesF = BatchF::size;
static constexpr std::size_t kLanesU64 = BatchU64::size;
static_assert(kLanesF == 2 * kLanesU64);

XSIMD_INLINE BatchU64 ExpandBits3(BatchU64 v)
{
  v = v & BatchU64(0x1fffffull);
  v = (v | (v << 32)) & BatchU64(0x001f00000000ffffull);
  v = (v | (v << 16)) & BatchU64(0x001f0000ff0000ffull);
  v = (v | (v << 8)) & BatchU64(0x100f00f00f00f00full);
  v = (v | (v << 4)) & BatchU64(0x10c30c30c30c30c3ull);
  v = (v | (v << 2)) & BatchU64(0x1249249249249249ull);
  return v;
}

#if PCCL_HAS_PDEP
// Single-instruction 3D Morton interleave for one (x,y,z) triplet with up
// to 21 bits per axis. Mask 0x9249249249249249 = bits 0,3,6,9,... — spreads
// input bit n into output bit 3n.
inline std::uint64_t
morton3_pdep(std::uint32_t x, std::uint32_t y, std::uint32_t z) noexcept
{
  return _pdep_u64(x, 0x9249249249249249ull) | _pdep_u64(y, 0x9249249249249249ull << 1)
         | _pdep_u64(z, 0x9249249249249249ull << 2);
}
#endif

XSIMD_INLINE BatchU64 CompactBits3(BatchU64 v)
{
  v = v & BatchU64(0x1249249249249249ull);
  v = (v ^ (v >> 2)) & BatchU64(0x10c30c30c30c30c3ull);
  v = (v ^ (v >> 4)) & BatchU64(0x100f00f00f00f00full);
  v = (v ^ (v >> 8)) & BatchU64(0x001f0000ff0000ffull);
  v = (v ^ (v >> 16)) & BatchU64(0x001f00000000ffffull);
  v = (v ^ (v >> 32)) & BatchU64(0x00000000001fffffull);
  return v;
}

XSIMD_INLINE void load_xyz(const float* __restrict p, BatchF& vx, BatchF& vy, BatchF& vz)
{
  alignas(64) float xs[kLanesF], ys[kLanesF], zs[kLanesF];
  for(std::size_t i = 0; i < kLanesF; ++i)
  {
    xs[i] = p[3 * i + 0];
    ys[i] = p[3 * i + 1];
    zs[i] = p[3 * i + 2];
  }
  vx = BatchF::load_aligned(xs);
  vy = BatchF::load_aligned(ys);
  vz = BatchF::load_aligned(zs);
}

XSIMD_INLINE void store_xyz(float* __restrict p, BatchF vx, BatchF vy, BatchF vz)
{
  alignas(64) float xs[kLanesF], ys[kLanesF], zs[kLanesF];
  vx.store_aligned(xs);
  vy.store_aligned(ys);
  vz.store_aligned(zs);
  for(std::size_t i = 0; i < kLanesF; ++i)
  {
    p[3 * i + 0] = xs[i];
    p[3 * i + 1] = ys[i];
    p[3 * i + 2] = zs[i];
  }
}

inline void aabb(
    const float* __restrict xyz, std::size_t n, std::array<float, 3>& lo,
    std::array<float, 3>& hi)
{
  if(n == 0)
  {
    lo = {0.f, 0.f, 0.f};
    hi = {1.f, 1.f, 1.f};
    return;
  }

  std::size_t i = 0;
  bool seeded = false;
  for(; i < n; ++i)
  {
    float x = xyz[3 * i + 0];
    float y = xyz[3 * i + 1];
    float z = xyz[3 * i + 2];
    if(safe_isfinite(x) && safe_isfinite(y) && safe_isfinite(z))
    {
      lo = hi = {x, y, z};
      ++i;
      seeded = true;
      break;
    }
  }
  if(!seeded)
  {
    lo = {0.f, 0.f, 0.f};
    hi = {1.f, 1.f, 1.f};
    return;
  }

  if(i + kLanesF <= n)
  {
    BatchF vlo_x(lo[0]), vlo_y(lo[1]), vlo_z(lo[2]);
    BatchF vhi_x(hi[0]), vhi_y(hi[1]), vhi_z(hi[2]);
    const BatchF kPInf(std::numeric_limits<float>::infinity());
    const BatchF kMInf(-std::numeric_limits<float>::infinity());

    for(; i + kLanesF <= n; i += kLanesF)
    {
      BatchF vx, vy, vz;
      load_xyz(xyz + 3 * i, vx, vy, vz);
      auto fin = xs::isfinite(vx) & xs::isfinite(vy) & xs::isfinite(vz);
      vlo_x = xs::min(vlo_x, xs::select(fin, vx, kPInf));
      vlo_y = xs::min(vlo_y, xs::select(fin, vy, kPInf));
      vlo_z = xs::min(vlo_z, xs::select(fin, vz, kPInf));
      vhi_x = xs::max(vhi_x, xs::select(fin, vx, kMInf));
      vhi_y = xs::max(vhi_y, xs::select(fin, vy, kMInf));
      vhi_z = xs::max(vhi_z, xs::select(fin, vz, kMInf));
    }

    lo[0] = xs::reduce_min(vlo_x);
    lo[1] = xs::reduce_min(vlo_y);
    lo[2] = xs::reduce_min(vlo_z);
    hi[0] = xs::reduce_max(vhi_x);
    hi[1] = xs::reduce_max(vhi_y);
    hi[2] = xs::reduce_max(vhi_z);
  }

  for(; i < n; ++i)
  {
    float x = xyz[3 * i + 0];
    float y = xyz[3 * i + 1];
    float z = xyz[3 * i + 2];
    if(safe_isfinite(x))
    {
      if(x < lo[0])
        lo[0] = x;
      if(x > hi[0])
        hi[0] = x;
    }
    if(safe_isfinite(y))
    {
      if(y < lo[1])
        lo[1] = y;
      if(y > hi[1])
        hi[1] = y;
    }
    if(safe_isfinite(z))
    {
      if(z < lo[2])
        lo[2] = z;
      if(z > hi[2])
        hi[2] = z;
    }
  }

  for(int k = 0; k < 3; ++k)
    if(hi[k] <= lo[k])
      hi[k] = std::nextafter(lo[k], std::numeric_limits<float>::infinity());
}

inline void quantize_to_morton(
    const float* __restrict xyz, std::size_t n, const std::array<float, 3>& lo,
    const std::array<float, 3>& hi, std::uint32_t bits, std::uint64_t* __restrict out)
{
  const std::uint32_t maxv = (1u << bits) - 1u;
  const float fmax = (float)maxv;
  float scale[3];
  for(int k = 0; k < 3; ++k)
  {
    float e = hi[k] - lo[k];
    scale[k] = (e > 0.f) ? (fmax / e) : 0.f;
  }

  BatchF vlo_x(lo[0]), vlo_y(lo[1]), vlo_z(lo[2]);
  BatchF vsc_x(scale[0]), vsc_y(scale[1]), vsc_z(scale[2]);
  BatchF vzero(0.f), vfmax(fmax), vhalf(0.5f);

  // NaN -> lo (lands at voxel 0). +/-Inf -> clamp swallows them.
  auto quant_lane = [&](BatchF v, BatchF vlo_, BatchF vsc_) {
    v = xs::select(xs::isnan(v), vlo_, v);
    v = xs::min(xs::max((v - vlo_) * vsc_, vzero), vfmax);
    return xs::batch_cast<std::int32_t>(v + vhalf);
  };

  std::size_t i = 0;
  for(; i + kLanesF <= n; i += kLanesF)
  {
    BatchF vx, vy, vz;
    load_xyz(xyz + 3 * i, vx, vy, vz);

    auto ix = quant_lane(vx, vlo_x, vsc_x);
    auto iy = quant_lane(vy, vlo_y, vsc_y);
    auto iz = quant_lane(vz, vlo_z, vsc_z);

#if PCCL_HAS_PDEP
    alignas(32) std::int32_t qxs[kLanesF], qys[kLanesF], qzs[kLanesF];
    ix.store_aligned(qxs);
    iy.store_aligned(qys);
    iz.store_aligned(qzs);
    for(std::size_t k = 0; k < kLanesF; ++k)
      out[i + k] = morton3_pdep(
          (std::uint32_t)qxs[k], (std::uint32_t)qys[k], (std::uint32_t)qzs[k]);
#else
    // Magic-bits fallback: widen i32 -> 2 x u64, then 5-round bit-spread.
    auto [qx_lo, qx_hi] = xs::widen(xs::bitwise_cast<std::uint32_t>(ix));
    auto [qy_lo, qy_hi] = xs::widen(xs::bitwise_cast<std::uint32_t>(iy));
    auto [qz_lo, qz_hi] = xs::widen(xs::bitwise_cast<std::uint32_t>(iz));

    auto code_lo
        = ExpandBits3(qx_lo) | (ExpandBits3(qy_lo) << 1) | (ExpandBits3(qz_lo) << 2);
    auto code_hi
        = ExpandBits3(qx_hi) | (ExpandBits3(qy_hi) << 1) | (ExpandBits3(qz_hi) << 2);

    code_lo.store_aligned(out + i);
    code_hi.store_aligned(out + i + kLanesU64);
#endif
  }

  // Scalar tail; mirrors the SIMD NaN/Inf handling.
  for(; i < n; ++i)
  {
    const float* p = xyz + 3 * i;
    std::uint32_t q[3];
    for(int k = 0; k < 3; ++k)
    {
      float raw = p[k];
      if(std::isnan(raw))
        raw = lo[k];
      float v = (raw - lo[k]) * scale[k];
      if(v < 0.f)
        v = 0.f;
      if(v > fmax)
        v = fmax;
      q[k] = (std::uint32_t)(v + 0.5f);
    }
    auto expand = [](std::uint64_t v) {
      v &= 0x1fffffull;
      v = (v | (v << 32)) & 0x001f00000000ffffull;
      v = (v | (v << 16)) & 0x001f0000ff0000ffull;
      v = (v | (v << 8)) & 0x100f00f00f00f00full;
      v = (v | (v << 4)) & 0x10c30c30c30c30c3ull;
      v = (v | (v << 2)) & 0x1249249249249249ull;
      return v;
    };
    out[i] = expand(q[0]) | (expand(q[1]) << 1) | (expand(q[2]) << 2);
  }
}

inline void sort_u64(std::uint64_t* __restrict keys, std::size_t n)
{
  ips4o::parallel::sort(keys, keys + n);
}

// Equivalent to std::unique(begin, end) but writes unconditionally
// (no branch on equality) so the compiler vectorises the inner loop.
inline std::size_t unique_u64(std::uint64_t* __restrict a, std::size_t n)
{
  if(n <= 1)
    return n;
  std::size_t w = 1;
  std::uint64_t prev = a[0];
  for(std::size_t r = 1; r < n; ++r)
  {
    std::uint64_t v = a[r];
    a[w] = v;
    w += (v != prev);
    prev = v;
  }
  return w;
}
inline void dequantize_demorton(
    const std::uint64_t* __restrict codes, std::size_t n, const std::array<float, 3>& lo,
    const std::array<float, 3>& hi, std::uint32_t bits, float* __restrict out)
{
  // The public Decoder::decode API guarantees `out` is the data() of an
  // AlignedVec<float>, hence 64-byte aligned. Propagate the contract.
  out = std::assume_aligned<64>(out);
  const std::uint32_t maxv = (1u << bits) - 1u;
  const float fmax = (float)maxv;
  float invscale[3];
  for(int k = 0; k < 3; ++k)
    invscale[k] = (fmax > 0.f) ? ((hi[k] - lo[k]) / fmax) : 0.f;

  alignas(64) std::uint64_t buf_x[kLanesU64];
  alignas(64) std::uint64_t buf_y[kLanesU64];
  alignas(64) std::uint64_t buf_z[kLanesU64];

  std::size_t i = 0;
  for(; i + kLanesU64 <= n; i += kLanesU64)
  {
    auto c = BatchU64::load_aligned(codes + i);
    auto bx = CompactBits3(c);
    auto by = CompactBits3(c >> 1);
    auto bz = CompactBits3(c >> 2);
    bx.store_aligned(buf_x);
    by.store_aligned(buf_y);
    bz.store_aligned(buf_z);
    for(std::size_t j = 0; j < kLanesU64; ++j)
    {
      out[3 * (i + j) + 0] = lo[0] + (float)(std::uint32_t)buf_x[j] * invscale[0];
      out[3 * (i + j) + 1] = lo[1] + (float)(std::uint32_t)buf_y[j] * invscale[1];
      out[3 * (i + j) + 2] = lo[2] + (float)(std::uint32_t)buf_z[j] * invscale[2];
    }
  }

  for(; i < n; ++i)
  {
    auto compact = [](std::uint64_t v) {
      v &= 0x1249249249249249ull;
      v = (v ^ (v >> 2)) & 0x10c30c30c30c30c3ull;
      v = (v ^ (v >> 4)) & 0x100f00f00f00f00full;
      v = (v ^ (v >> 8)) & 0x001f0000ff0000ffull;
      v = (v ^ (v >> 16)) & 0x001f00000000ffffull;
      v = (v ^ (v >> 32)) & 0x00000000001fffffull;
      return (std::uint32_t)v;
    };
    std::uint64_t c = codes[i];
    std::uint32_t qx = compact(c);
    std::uint32_t qy = compact(c >> 1);
    std::uint32_t qz = compact(c >> 2);
    out[3 * i + 0] = lo[0] + qx * invscale[0];
    out[3 * i + 1] = lo[1] + qy * invscale[1];
    out[3 * i + 2] = lo[2] + qz * invscale[2];
  }
}

}
