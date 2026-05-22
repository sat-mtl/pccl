// New codecs added in the second round of benchmarking.
//
// 1) streamvbyte_delta on 32-bit ints (with hi32/lo32 split for u64).
// 2) TurboPFor delta-PFor scalar 64-bit + SIMD 32-bit kernels.
// 3) FSE (Cyan4973/FiniteStateEntropy) on a byte stream, chunked at 64 KiB
//    so we stay under FSE's per-call source-size limit.
// 4) rANS order-0 static (ryg_rans rans_byte.h).
//
// We only need *encode* + sizes + timings for benchmarking; no decoder.

#pragma once

extern "C" {
#include <fse.h>
#include <ic.h>
#include <streamvbyte.h>
#include <streamvbytedelta.h>
}

#include "pcc_codec.hpp"

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

// ryg_rans is a single-header public-domain rANS.  It exposes encoder/decoder
// primitives only; we drive them ourselves with a static order-0 model.
#include "rans_byte.h"

namespace pcc
{

// ---------------------------------------------------------------------------
// streamvbyte_delta — operates on uint32_t arrays.  We support u64 input by
// splitting each value into hi32 + lo32: the hi32 stream is monotonic (delta-
// encoded), the lo32 stream is not (encoded with the plain group-varint).
// ---------------------------------------------------------------------------

inline CodecResult c_streamvbyte_delta_u32(const std::uint32_t* codes, std::size_t n)
{
  CodecResult r;
  r.name = "svb_delta";
  r.input_bytes = n * sizeof(std::uint32_t);

  std::vector<std::uint8_t> dst(streamvbyte_max_compressedbytes((std::uint32_t)n));
  auto t0 = clk::now();
  std::size_t out = streamvbyte_delta_encode(codes, (std::uint32_t)n, dst.data(), 0u);
  r.encode_ms = ms_since(t0);
  r.output_bytes = out;
  return r;
}

inline CodecResult c_streamvbyte_delta_u64(const std::uint64_t* codes, std::size_t n)
{
  CodecResult r;
  r.name = "svb_delta";
  r.input_bytes = n * sizeof(std::uint64_t);

  std::vector<std::uint32_t> hi(n), lo(n);
  std::vector<std::uint8_t> dst_hi(streamvbyte_max_compressedbytes((std::uint32_t)n));
  std::vector<std::uint8_t> dst_lo(streamvbyte_max_compressedbytes((std::uint32_t)n));

  auto t0 = clk::now();
  for(std::size_t i = 0; i < n; ++i)
  {
    hi[i] = (std::uint32_t)(codes[i] >> 32);
    lo[i] = (std::uint32_t)(codes[i] & 0xFFFFFFFFu);
  }
  std::size_t s_hi
      = streamvbyte_delta_encode(hi.data(), (std::uint32_t)n, dst_hi.data(), 0u);
  std::size_t s_lo = streamvbyte_encode(lo.data(), (std::uint32_t)n, dst_lo.data());
  r.encode_ms = ms_since(t0);
  r.output_bytes = s_hi + s_lo;
  return r;
}

// ---------------------------------------------------------------------------
// TurboPFor — `p4ndenc*` and `p4nd1enc*` accept *sorted* integers and do their
// own delta + PFor internally.  `p4ndenc*` allows duplicates; we use that since
// after quantization we may have multiple points per voxel.  The encoder
// destroys its input, so we copy.
// ---------------------------------------------------------------------------

inline CodecResult c_turbopfor_p4d_u64(const std::uint64_t* codes, std::size_t n)
{
  CodecResult r;
  r.name = "p4ndenc64";
  r.input_bytes = n * sizeof(std::uint64_t);

  std::vector<std::uint64_t> tmp(codes, codes + n);
  std::vector<std::uint8_t> dst(p4nbound64(n));
  auto t0 = clk::now();
  std::size_t out = p4ndenc64(tmp.data(), n, dst.data());
  r.encode_ms = ms_since(t0);
  r.output_bytes = out;
  return r;
}

inline CodecResult c_turbopfor_p4d_u32(const std::uint32_t* codes, std::size_t n)
{
  CodecResult r;
  r.name = "p4ndenc128v32";
  r.input_bytes = n * sizeof(std::uint32_t);

  std::vector<std::uint32_t> tmp(codes, codes + n);
  std::vector<std::uint8_t> dst(p4nbound128v32(n));
  auto t0 = clk::now();
  std::size_t out = p4ndenc128v32(tmp.data(), n, dst.data());
  r.encode_ms = ms_since(t0);
  r.output_bytes = out;
  return r;
}

// Combined u64 (split into u32 streams) using TurboPFor SIMD 32-bit kernels.
inline CodecResult c_turbopfor_p4d_u64_split32(const std::uint64_t* codes, std::size_t n)
{
  CodecResult r;
  r.name = "p4nd_128v32x2";
  r.input_bytes = n * sizeof(std::uint64_t);

  std::vector<std::uint32_t> hi(n), lo(n);
  std::vector<std::uint8_t> dst_hi(p4nbound128v32(n));
  std::vector<std::uint8_t> dst_lo(p4nbound128v32(n));

  auto t0 = clk::now();
  for(std::size_t i = 0; i < n; ++i)
  {
    hi[i] = (std::uint32_t)(codes[i] >> 32);
    lo[i] = (std::uint32_t)(codes[i] & 0xFFFFFFFFu);
  }
  // hi is sorted (since codes are sorted) -> delta+PFor
  // lo is *not* sorted -> use the non-delta PFor (n*pack* without the d)
  std::size_t s_hi = p4ndenc128v32(hi.data(), n, dst_hi.data());
  std::size_t s_lo = p4nenc128v32(lo.data(), n, dst_lo.data());
  r.encode_ms = ms_since(t0);
  r.output_bytes = s_hi + s_lo;
  return r;
}

// ---------------------------------------------------------------------------
// FSE on a byte stream.  FSE has a per-call source-size cap so we block at
// 64 KiB and prepend each block with a 4-byte (size, comp_flag) word.
// ---------------------------------------------------------------------------

inline CodecResult c_fse(const std::uint8_t* src, std::size_t n)
{
  CodecResult r;
  r.name = "fse";
  r.input_bytes = n;

  constexpr std::size_t blk = 64 * 1024;
  std::vector<std::uint8_t> blkbuf(FSE_compressBound(blk));
  std::size_t total = 0;
  auto t0 = clk::now();
  for(std::size_t off = 0; off < n; off += blk)
  {
    std::size_t bs = std::min(blk, n - off);
    std::size_t out = FSE_compress(blkbuf.data(), blkbuf.size(), src + off, bs);
    if(FSE_isError(out))
      throw std::runtime_error(std::string{"fse: "} + FSE_getErrorName(out));
    // 4-byte size header per block.  If FSE returns 0 (incompressible) or 1
    // (RLE), the block would be stored raw — count its raw size.
    std::size_t stored = (out > 1) ? out : bs;
    total += 4 + stored;
  }
  r.encode_ms = ms_since(t0);
  r.output_bytes = total;
  return r;
}

// ---------------------------------------------------------------------------
// rANS order-0 static.  Build a 256-bin histogram, normalize to a power-of-two
// total (1<<14), then encode using ryg_rans rans_byte.
// ---------------------------------------------------------------------------

inline CodecResult c_rans_o0(const std::uint8_t* src, std::size_t n)
{
  CodecResult r;
  r.name = "rans-o0";
  r.input_bytes = n;
  if(n == 0)
    return r;

  auto t0 = clk::now();

  // 1) histogram.
  std::uint64_t freqs[256] = {};
  for(std::size_t i = 0; i < n; ++i)
    ++freqs[src[i]];

  // 2) normalize to total = 1<<14 (RANS_BYTE_PROB_BITS-ish).  Anything with
  //    non-zero count must end up with at least 1.
  constexpr std::uint32_t prob_bits = 14;
  constexpr std::uint32_t prob_scale = 1u << prob_bits;
  std::uint32_t norm[256] = {};
  {
    double inv = double(prob_scale) / double(n);
    std::uint32_t sum = 0;
    int max_idx = 0;
    std::uint32_t max_val = 0;
    for(int i = 0; i < 256; ++i)
    {
      if(freqs[i] == 0)
        continue;
      std::uint32_t v = (std::uint32_t)std::max<double>(1.0, double(freqs[i]) * inv);
      norm[i] = v;
      sum += v;
      if(v > max_val)
      {
        max_val = v;
        max_idx = i;
      }
    }
    // adjust the most-frequent symbol so totals match exactly
    if(sum > prob_scale)
      norm[max_idx] -= (sum - prob_scale);
    else if(sum < prob_scale)
      norm[max_idx] += (prob_scale - sum);
  }

  // 3) cumulative frequencies.
  std::uint32_t cum[257];
  cum[0] = 0;
  for(int i = 0; i < 256; ++i)
    cum[i + 1] = cum[i] + norm[i];

  // 4) encode bytes in reverse (rANS is LIFO).
  RansEncSymbol esyms[256];
  for(int i = 0; i < 256; ++i)
    if(norm[i] != 0)
      RansEncSymbolInit(&esyms[i], cum[i], norm[i], prob_bits);

  std::vector<std::uint8_t> outbuf(n + 4096);
  std::uint8_t* rans_begin = nullptr;
  std::uint8_t* ptr = outbuf.data() + outbuf.size();
  RansState rans;
  RansEncInit(&rans);
  for(std::size_t i = n; i--;)
    RansEncPutSymbol(&rans, &ptr, &esyms[src[i]]);
  RansEncFlush(&rans, &ptr);
  rans_begin = ptr;

  r.encode_ms = ms_since(t0);
  // table cost: 256 entries x 4 bytes (worst case) for the normalised frequencies
  const std::size_t table_bytes = 256 * sizeof(std::uint16_t);
  r.output_bytes = (outbuf.data() + outbuf.size() - rans_begin) + table_bytes;
  return r;
}

} // namespace pcc
