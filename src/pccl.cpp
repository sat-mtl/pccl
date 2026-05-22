// Encoder / Decoder implementation. All scratch lives in Impl structs
// and is reused across calls; steady-state streaming never allocates.

#include "pccl_simd.hpp"

#include <pccl/pccl.hpp>

extern "C" {
#include <fse.h>
#include <ic.h>
// AVX2 8-way interleaved rANS (r8x16_avx2.c) exports the *_4x16 API.
// The bound() signature here matches the AVX2 file (3-arg), which differs
// from the rANS_static4x16.h header (2-arg) -- declare locally to avoid
// silently passing garbage.
unsigned int rans_compress_bound_4x16(unsigned int size, int order, int* tab);
unsigned char* rans_compress_to_4x16(
    unsigned char* in, unsigned int in_size, unsigned char* out, unsigned int* out_size,
    int order);
unsigned char* rans_uncompress_to_4x16(
    unsigned char* in, unsigned int in_size, unsigned char* out, unsigned int* out_size,
    int order);
}

#include <boost/container/vector.hpp>

#include <cmath>

#include <algorithm>
#include <bit>
#include <cassert>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <new>
#include <stdexcept>
#include <string>
#include <type_traits>

namespace pccl
{
namespace
{

static constexpr auto kDefaultInit = boost::container::default_init;

constexpr std::uint32_t kMagic = 0x31434350u; // 'PCC1' little-endian
constexpr std::size_t kHeaderSize = 44;
constexpr std::size_t kFseBlockBytes = 64 * 1024;
constexpr std::size_t kTurboPad = 256; // p4ndenc64 SIMD overread slack

#ifdef PCCL_BENCHMARKS
class BenchClock
{
  using clk = std::chrono::high_resolution_clock;
  clk::time_point t_;

public:
  BenchClock() noexcept
      : t_(clk::now())
  {
  }
  double tap() noexcept
  {
    auto now = clk::now();
    double ms = std::chrono::duration<double, std::milli>(now - t_).count();
    t_ = now;
    return ms;
  }
  double elapsed() const noexcept
  {
    return std::chrono::duration<double, std::milli>(clk::now() - t_).count();
  }
};
#else
class BenchClock
{
public:
  constexpr BenchClock() noexcept = default;
  constexpr double tap() noexcept { return 0.0; }
  constexpr double elapsed() const noexcept { return 0.0; }
};
#endif

inline void put_u32(std::uint8_t* p, std::uint32_t v)
{
  p[0] = (std::uint8_t)v;
  p[1] = (std::uint8_t)(v >> 8);
  p[2] = (std::uint8_t)(v >> 16);
  p[3] = (std::uint8_t)(v >> 24);
}
inline std::uint32_t get_u32(const std::uint8_t* p)
{
  return (std::uint32_t)p[0] | ((std::uint32_t)p[1] << 8) | ((std::uint32_t)p[2] << 16)
         | ((std::uint32_t)p[3] << 24);
}
inline void put_u16(std::uint8_t* p, std::uint16_t v)
{
  p[0] = (std::uint8_t)v;
  p[1] = (std::uint8_t)(v >> 8);
}
inline std::uint16_t get_u16(const std::uint8_t* p)
{
  return (std::uint16_t)p[0] | ((std::uint16_t)p[1] << 8);
}
inline void put_f32(std::uint8_t* p, float v)
{
  put_u32(p, std::bit_cast<std::uint32_t>(v));
}
inline float get_f32(const std::uint8_t* p)
{
  return std::bit_cast<float>(get_u32(p));
}

inline std::size_t varint_get(const std::uint8_t* p, std::uint64_t& out)
{
  std::uint64_t v = 0;
  int shift = 0;
  std::size_t i = 0;
  while(true)
  {
    std::uint8_t b = p[i++];
    v |= ((std::uint64_t)(b & 0x7F)) << shift;
    if(!(b & 0x80))
      break;
    shift += 7;
    if(shift >= 64)
      throw std::runtime_error("pccl: varint overflow");
  }
  out = v;
  return i;
}

// BFS level-by-level traversal.
void octree_encode(
    const std::uint64_t* sorted_codes, std::size_t n, std::uint32_t depth,
    AlignedVec<std::uint8_t>& out)
{
  out.clear();
  out.reserve(n);
  if(n == 0)
    return;

  struct Range
  {
    std::size_t b, e;
  };
  static thread_local AlignedVec<Range> cur, nxt;
  cur.clear();
  cur.push_back({0, n});

  for(int lvl = (int)depth; lvl > 0; --lvl)
  {
    const int shift = (lvl - 1) * 3;
    nxt.clear();
    nxt.reserve(cur.size() * 8);
    for(const auto& r : cur)
    {
      std::size_t hist[8] = {};
      for(std::size_t i = r.b; i < r.e; ++i)
        ++hist[(sorted_codes[i] >> shift) & 7ull];
      std::uint8_t mask = 0;
      std::size_t pos = r.b;
      for(int c = 0; c < 8; ++c)
        if(hist[c])
        {
          mask |= (std::uint8_t)(1u << c);
          nxt.push_back({pos, pos + hist[c]});
          pos += hist[c];
        }
      out.push_back(mask);
    }
    cur.swap(nxt);
  }
}

void octree_decode(
    const std::uint8_t* occ, std::size_t occ_size, std::uint32_t depth,
    AlignedVec<std::uint64_t>& codes, std::size_t expected_voxels)
{
  codes.clear();
  codes.reserve(expected_voxels);
  if(occ_size == 0)
    return;

  static thread_local AlignedVec<std::uint64_t> cur, nxt;
  cur.clear();
  cur.push_back(0ull);

  std::size_t pos = 0;
  for(int lvl = (int)depth; lvl > 0; --lvl)
  {
    const int shift = (lvl - 1) * 3;
    nxt.clear();
    nxt.reserve(cur.size() * 8);
    for(std::uint64_t prefix : cur)
    {
      if(pos >= occ_size)
        throw std::runtime_error("pccl: octree truncated");
      const std::uint8_t mask = occ[pos++];
      for(int c = 0; c < 8; ++c)
        if(mask & (1u << c))
          nxt.push_back(prefix | ((std::uint64_t)c << shift));
    }
    cur.swap(nxt);
  }
  codes.insert(codes.end(), cur.begin(), cur.end());
  if(pos != occ_size)
    throw std::runtime_error("pccl: octree extra bytes");
}

// 8-way AVX2 interleaved rANS via rans_static (jkbonfield). The frame
// layout is opaque to us; rans_static writes its own size + freq table.
//
// We use order=0. order=1 was tested under both DFS and BFS octree
// traversals and always lost: octree occupancy bytes don't have strong
// byte-to-byte conditional structure (sibling distributions look like
// the marginal), so the per-context table overhead inflates output
// 16-64% across the board.
void rans_encode_o0(
    const std::uint8_t* src, std::size_t n, ByteBuffer& out,
    AlignedVec<std::uint8_t>& scratch)
{
  const unsigned int bound = rans_compress_bound_4x16((unsigned int)n, 0, nullptr);
  if(scratch.size() < bound)
    scratch.resize(bound, kDefaultInit);
  unsigned int csz = bound;
  unsigned char* enc = rans_compress_to_4x16(
      const_cast<unsigned char*>(src), (unsigned int)n, scratch.data(), &csz, 0);
  assert(enc == scratch.data() && "rans_compress_to_4x16 should fill caller buffer");
  (void)enc;
  const std::size_t off = out.size();
  out.resize(off + csz, kDefaultInit);
  std::memcpy(out.data() + off, scratch.data(), csz);
}

void rans_decode_o0(
    const std::uint8_t* src, std::size_t src_size, std::size_t n,
    AlignedVec<std::uint8_t>& out, AlignedVec<std::uint8_t>& /*cum2sym (unused)*/)
{
  out.resize(n, kDefaultInit);
  unsigned int dsz = (unsigned int)n;
  unsigned char* dec = rans_uncompress_to_4x16(
      const_cast<unsigned char*>(src), (unsigned int)src_size, out.data(), &dsz, 0);
  if(!dec || dsz != (unsigned int)n)
    throw std::runtime_error("pccl: rans decode failed");
}

inline void write_header(
    std::uint8_t* out, Preset preset, std::uint8_t bits, std::uint32_t input_points,
    std::uint32_t unique_voxels, const std::array<float, 3>& lo,
    const std::array<float, 3>& hi, std::uint32_t payload_size)
{
  put_u32(out, kMagic);
  out[4] = (std::uint8_t)preset;
  out[5] = bits;
  put_u16(out + 6, 0);
  put_u32(out + 8, input_points);
  put_u32(out + 12, unique_voxels);
  for(int k = 0; k < 3; ++k)
    put_f32(out + 16 + k * 4, lo[k]);
  for(int k = 0; k < 3; ++k)
    put_f32(out + 28 + k * 4, hi[k]);
  put_u32(out + 40, payload_size);
}

} // namespace

struct Encoder::Impl
{
  AlignedVec<std::uint64_t> codes;
  AlignedVec<std::uint64_t> tmp64;       // p4ndenc64 mutable copy + pad
  AlignedVec<std::uint8_t> varint;       // Small
  AlignedVec<std::uint8_t> fse_blkbuf;   // Small FSE scratch
  AlignedVec<std::uint8_t> octree_buf;   // Smallest
  AlignedVec<std::uint8_t> rans_scratch; // Smallest rANS scratch
};

Encoder::Encoder()
    : impl_(std::make_unique<Impl>())
{
}
Encoder::~Encoder() = default;
Encoder::Encoder(Encoder&&) noexcept = default;
Encoder& Encoder::operator=(Encoder&&) noexcept = default;

void Encoder::reserve(std::size_t expected_points)
{
  impl_->codes.reserve(expected_points);
  impl_->tmp64.reserve(expected_points + kTurboPad);
}

namespace
{

void enc_fast(Encoder::Impl& I, std::size_t n, ByteBuffer& out)
{
  // Delta + variable-byte u64 encode via TurboPFor. Single pass over the
  // sorted codes; no hi/lo split, no separate-stream framing.
  const std::size_t off = out.size();
  out.resize(off + vbbound64(n), kDefaultInit);
  std::uint8_t* end = vbxenc64(I.codes.data(), (unsigned)n, out.data() + off, 0ull);
  out.resize((std::size_t)(end - out.data()), kDefaultInit);
}

void enc_balanced(Encoder::Impl& I, std::size_t n, ByteBuffer& out)
{
  // p4ndenc64 mutates its input and overreads up to one block past n;
  // pad+zero the tail so output is deterministic (overread of uninit
  // memory would jitter the encoded bytes).
  I.tmp64.resize(n + kTurboPad, kDefaultInit);
  std::copy(I.codes.begin(), I.codes.begin() + n, I.tmp64.begin());
  std::fill(I.tmp64.begin() + n, I.tmp64.end(), std::uint64_t{0});

  const std::size_t off = out.size();
  out.resize(off + p4nbound64(n), kDefaultInit);
  std::size_t s = p4ndenc64(I.tmp64.data(), n, out.data() + off);
  out.resize(off + s, kDefaultInit);
}

void enc_small(Encoder::Impl& I, std::size_t n, ByteBuffer& out)
{
  auto& v = I.varint;
  // 10 bytes max per varint (9*7 bits + 1 trailing). Pre-size and write
  // through a raw pointer; shrink to actual size at the end.
  v.resize(n * 10 + 16);
  std::uint8_t* __restrict dst = v.data();
  std::uint64_t prev = 0;
  const std::uint64_t* __restrict codes_p = I.codes.data();
  for(std::size_t i = 0; i < n; ++i)
  {
    std::uint64_t d = codes_p[i] - prev;
    prev = codes_p[i];
    while(d >= 0x80)
    {
      *dst++ = (std::uint8_t)(d | 0x80);
      d >>= 7;
    }
    *dst++ = (std::uint8_t)d;
  }
  v.resize((std::size_t)(dst - v.data()));

  const std::size_t blk = kFseBlockBytes;
  const std::size_t total = v.size();
  const std::size_t nb = total == 0 ? 0 : (total + blk - 1) / blk;
  I.fse_blkbuf.resize(FSE_compressBound(blk), kDefaultInit);

  const std::size_t off = out.size();
  out.resize(off + 8, kDefaultInit);
  put_u32(out.data() + off, (std::uint32_t)total);
  put_u32(out.data() + off + 4, (std::uint32_t)nb);

  for(std::size_t o = 0; o < total; o += blk)
  {
    const std::size_t bs = std::min(blk, total - o);
    std::size_t cs
        = FSE_compress(I.fse_blkbuf.data(), I.fse_blkbuf.size(), v.data() + o, bs);
    assert(!FSE_isError(cs) && "FSE_compress with FSE_compressBound sizing cannot fail");
    const std::size_t hdr = out.size();
    if(cs > 1 && cs < bs)
    {
      out.resize(hdr + 8 + cs, kDefaultInit);
      put_u32(out.data() + hdr + 0, (std::uint32_t)cs);
      put_u32(out.data() + hdr + 4, (std::uint32_t)bs);
      std::memcpy(out.data() + hdr + 8, I.fse_blkbuf.data(), cs);
    }
    else
    {
      out.resize(hdr + 8 + bs, kDefaultInit);
      put_u32(out.data() + hdr + 0, 0u);
      put_u32(out.data() + hdr + 4, (std::uint32_t)bs);
      std::memcpy(out.data() + hdr + 8, v.data() + o, bs);
    }
  }
}

void enc_smallest(Encoder::Impl& I, std::size_t n, std::uint32_t bits, ByteBuffer& out)
{
  octree_encode(I.codes.data(), n, bits, I.octree_buf);
  const std::size_t off = out.size();
  out.resize(off + 4, kDefaultInit);
  put_u32(out.data() + off, (std::uint32_t)I.octree_buf.size());
  if(!I.octree_buf.empty())
    rans_encode_o0(I.octree_buf.data(), I.octree_buf.size(), out, I.rans_scratch);
}

} // namespace

std::size_t Encoder::encode(
    std::span<const float> xyz, const EncodeOptions& opts, ByteBuffer& out,
    EncodeStats* stats)
{
  if(opts.bits < 1 || opts.bits > 21) [[unlikely]]
    throw std::runtime_error("pccl: bits must be in [1, 21]");
  if((xyz.size() % 3) != 0) [[unlikely]]
    throw std::runtime_error("pccl: xyz size must be a multiple of 3");

  const std::size_t num_points = xyz.size() / 3;
  const float* __restrict xyz_p = xyz.data();

  auto& I = *impl_;
  BenchClock all;
  BenchClock stage;
  double t_aabb = 0, t_quant = 0, t_sort = 0, t_dedup = 0, t_codec = 0;

  std::array<float, 3> lo, hi;
  if(opts.aabb_valid)
  {
    lo = opts.aabb_min;
    hi = opts.aabb_max;
  }
  else
  {
    simd::aabb(xyz_p, num_points, lo, hi);
    t_aabb = stage.tap();
  }

  std::size_t nv = 0;
  if(num_points > 0)
  {
    stage.tap();
    I.codes.resize(num_points, kDefaultInit);
    simd::quantize_to_morton(xyz_p, num_points, lo, hi, opts.bits, I.codes.data());
    t_quant = stage.tap();

    simd::sort_u64(I.codes.data(), I.codes.size());
    t_sort = stage.tap();

    const std::size_t new_n = simd::unique_u64(I.codes.data(), I.codes.size());
    I.codes.erase(I.codes.begin() + new_n, I.codes.end());
    t_dedup = stage.tap();
    nv = I.codes.size();
  }

  out.resize(kHeaderSize, kDefaultInit);

  if(nv > 0)
  {
    stage.tap();
    switch(opts.preset)
    {
      case Preset::Fast:
        enc_fast(I, nv, out);
        break;
      case Preset::Balanced:
        enc_balanced(I, nv, out);
        break;
      case Preset::Small:
        enc_small(I, nv, out);
        break;
      case Preset::Smallest:
        enc_smallest(I, nv, opts.bits, out);
        break;
    }
    t_codec = stage.tap();
  }

  write_header(
      out.data(), opts.preset, opts.bits, (std::uint32_t)num_points, (std::uint32_t)nv,
      lo, hi, (std::uint32_t)(out.size() - kHeaderSize));

  if(stats)
  {
    stats->input_points = num_points;
    stats->input_bytes = num_points * sizeof(float) * 3;
    stats->unique_voxels = nv;
    stats->output_bytes = out.size();
    stats->encode_ms = all.elapsed();
    stats->aabb_ms = t_aabb;
    stats->quant_ms = t_quant;
    stats->sort_ms = t_sort;
    stats->dedup_ms = t_dedup;
    stats->codec_ms = t_codec;
  }
  return out.size();
}

struct Decoder::Impl
{
  AlignedVec<std::uint64_t> codes;
  AlignedVec<std::uint8_t> varint;       // Small
  AlignedVec<std::uint8_t> occupancy;    // Smallest
  AlignedVec<std::uint8_t> rans_cum2sym; // Smallest rANS lookup (legacy)
};

Decoder::Decoder()
    : impl_(std::make_unique<Impl>())
{
}
Decoder::~Decoder() = default;
Decoder::Decoder(Decoder&&) noexcept = default;
Decoder& Decoder::operator=(Decoder&&) noexcept = default;

void Decoder::reserve(std::size_t expected_voxels)
{
  impl_->codes.reserve(expected_voxels);
}

namespace
{

void dec_fast(
    Decoder::Impl& I, const std::uint8_t* p, std::size_t /*size*/, std::size_t n)
{
  I.codes.resize(n, kDefaultInit);
  (void)vbxdec64(const_cast<std::uint8_t*>(p), (unsigned)n, I.codes.data(), 0ull);
}

void dec_balanced(
    Decoder::Impl& I, const std::uint8_t* p, std::size_t /*size*/, std::size_t n)
{
  I.codes.resize(n, kDefaultInit);
  p4nddec64(const_cast<std::uint8_t*>(p), n, I.codes.data());
}

void dec_small(Decoder::Impl& I, const std::uint8_t* p, std::size_t size, std::size_t n)
{
  if(size < 8)
    throw std::runtime_error("pccl: small payload too small");
  std::size_t total = get_u32(p);
  std::size_t nb = get_u32(p + 4);
  I.varint.resize(total, kDefaultInit);

  std::size_t off_in = 8;
  std::size_t off_out = 0;
  for(std::size_t b = 0; b < nb; ++b)
  {
    if(off_in + 8 > size)
      throw std::runtime_error("pccl: small block hdr truncated");
    std::uint32_t cs = get_u32(p + off_in + 0);
    std::uint32_t bs = get_u32(p + off_in + 4);
    off_in += 8;
    if(off_out + bs > total)
      throw std::runtime_error("pccl: small block overflows total");
    if(cs == 0)
    {
      if(off_in + bs > size)
        throw std::runtime_error("pccl: small raw block truncated");
      std::memcpy(I.varint.data() + off_out, p + off_in, bs);
      off_in += bs;
    }
    else
    {
      if(off_in + cs > size)
        throw std::runtime_error("pccl: small comp block truncated");
      std::size_t r = FSE_decompress(I.varint.data() + off_out, bs, p + off_in, cs);
      if(FSE_isError(r))
        throw std::runtime_error(std::string{"pccl: fse-dec: "} + FSE_getErrorName(r));
      if(r != bs)
        throw std::runtime_error("pccl: small fse size mismatch");
      off_in += cs;
    }
    off_out += bs;
  }

  I.codes.resize(n, kDefaultInit);
  std::uint64_t prev = 0;
  std::size_t i = 0;
  const std::uint8_t* q = I.varint.data();
  const std::uint8_t* qe = q + I.varint.size();
  while(q < qe)
  {
    std::uint64_t d;
    q += varint_get(q, d);
    prev += d;
    I.codes[i++] = prev;
  }
  if(i != n)
    throw std::runtime_error("pccl: small varint count mismatch");
}

void dec_smallest(
    Decoder::Impl& I, const std::uint8_t* p, std::size_t size,
    std::size_t expected_voxels, std::uint32_t bits)
{
  if(size < 4)
    throw std::runtime_error("pccl: smallest payload too small");
  std::size_t occ_size = get_u32(p);
  if(occ_size == 0)
  {
    I.codes.clear();
    return;
  }
  rans_decode_o0(p + 4, size - 4, occ_size, I.occupancy, I.rans_cum2sym);
  octree_decode(I.occupancy.data(), I.occupancy.size(), bits, I.codes, expected_voxels);
}

} // namespace

std::size_t Decoder::decode(
    std::span<const std::uint8_t> data, AlignedVec<float>& out_xyz,
    FrameMetadata* out_meta)
{
  FrameHeader h;
  if(!peek_header(data, h)) [[unlikely]]
    throw std::runtime_error("pccl: bad magic / truncated header");
  if(h.bits < 1 || h.bits > 21) [[unlikely]]
    throw std::runtime_error("pccl: bad bits in header");
  if(h.preset > (std::uint8_t)Preset::Smallest) [[unlikely]]
    throw std::runtime_error("pccl: unknown preset");
  if(kHeaderSize + h.payload_size > data.size()) [[unlikely]]
    throw std::runtime_error("pccl: payload truncated");

  auto& I = *impl_;
  const std::uint8_t* __restrict p = data.data() + kHeaderSize;
  const std::size_t ps = h.payload_size;
  const Preset preset = (Preset)h.preset;
  const std::size_t nv = h.unique_voxels;

  if(nv == 0)
  {
    I.codes.clear();
    out_xyz.clear();
  }
  else
  {
    switch(preset)
    {
      case Preset::Fast:
        dec_fast(I, p, ps, nv);
        break;
      case Preset::Balanced:
        dec_balanced(I, p, ps, nv);
        break;
      case Preset::Small:
        dec_small(I, p, ps, nv);
        break;
      case Preset::Smallest:
        dec_smallest(I, p, ps, nv, h.bits);
        break;
    }
    out_xyz.resize(I.codes.size() * 3, kDefaultInit);
    simd::dequantize_demorton(
        I.codes.data(), I.codes.size(), h.aabb_min, h.aabb_max, h.bits, out_xyz.data());
  }

  if(out_meta)
  {
    out_meta->preset = preset;
    out_meta->bits = h.bits;
    out_meta->input_points = h.input_points;
    out_meta->unique_voxels = (std::uint32_t)I.codes.size();
    out_meta->aabb_min = h.aabb_min;
    out_meta->aabb_max = h.aabb_max;
  }
  return out_xyz.size();
}

bool peek_header(std::span<const std::uint8_t> data, FrameHeader& out) noexcept
{
  if(data.size() < kHeaderSize)
    return false;
  const std::uint8_t* __restrict d = data.data();
  if(get_u32(d) != kMagic)
    return false;
  out.preset = d[4];
  out.bits = d[5];
  out.input_points = get_u32(d + 8);
  out.unique_voxels = get_u32(d + 12);
  for(int k = 0; k < 3; ++k)
    out.aabb_min[k] = get_f32(d + 16 + k * 4);
  for(int k = 0; k < 3; ++k)
    out.aabb_max[k] = get_f32(d + 28 + k * 4);
  out.payload_size = get_u32(d + 40);
  return true;
}

void write_pccl_file(
    const std::filesystem::path& path, std::span<const std::uint8_t> data)
{
  std::ofstream f(path, std::ios::binary | std::ios::trunc);
  if(!f)
    throw std::runtime_error("pccl: cannot open for write: " + path.string());
  f.write(
      reinterpret_cast<const char*>(data.data()),
      static_cast<std::streamsize>(data.size()));
  if(!f)
    throw std::runtime_error("pccl: write failed: " + path.string());
}

void read_pccl_file(const std::filesystem::path& path, ByteBuffer& out)
{
  std::error_code ec;
  const auto sz = std::filesystem::file_size(path, ec);
  if(ec)
    throw std::runtime_error("pccl: cannot stat: " + path.string());
  if(sz < kHeaderSize)
    throw std::runtime_error(
        "pccl: file too small to be a pccl frame: " + path.string());

  std::ifstream f(path, std::ios::binary);
  if(!f)
    throw std::runtime_error("pccl: cannot open for read: " + path.string());

  out.resize(static_cast<std::size_t>(sz), kDefaultInit);
  f.read(reinterpret_cast<char*>(out.data()), static_cast<std::streamsize>(sz));
  if(static_cast<std::uintmax_t>(f.gcount()) != sz)
    throw std::runtime_error("pccl: short read: " + path.string());

  if(get_u32(out.data()) != kMagic)
    throw std::runtime_error("pccl: bad magic in " + path.string());
}

} // namespace pccl
