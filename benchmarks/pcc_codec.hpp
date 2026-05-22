#pragma once

#include <brotli/encode.h>
#include <lzo/lzo1x.h>

#include <lz4.h>
#include <lz4hc.h>
#include <snappy.h>
#include <zlib.h>
#include <zstd.h>

#include <chrono>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

namespace pcc
{

struct CodecResult
{
  std::string name;
  std::size_t input_bytes = 0;
  std::size_t output_bytes = 0;
  double encode_ms = 0.0;

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
};

using clk = std::chrono::high_resolution_clock;

inline double ms_since(clk::time_point t0)
{
  using namespace std::chrono;
  return duration<double, std::milli>(clk::now() - t0).count();
}

// --- raw (no compression) -----------------------------------------------
inline CodecResult c_raw(const std::uint8_t* src, std::size_t n)
{
  CodecResult r;
  r.name = "raw";
  r.input_bytes = n;
  r.output_bytes = n;
  r.encode_ms = 0.0;
  return r;
}

// --- zlib --------------------------------------------------------------
inline CodecResult c_zlib(const std::uint8_t* src, std::size_t n, int level)
{
  CodecResult r;
  r.name = "zlib-L" + std::to_string(level);
  r.input_bytes = n;

  uLongf cap = compressBound(static_cast<uLong>(n));
  std::vector<std::uint8_t> dst(cap);
  auto t0 = clk::now();
  if(::compress2(dst.data(), &cap, src, static_cast<uLong>(n), level) != Z_OK)
    throw std::runtime_error("zlib failed");
  r.encode_ms = ms_since(t0);
  r.output_bytes = cap;
  return r;
}

// --- zstd --------------------------------------------------------------
inline CodecResult c_zstd(const std::uint8_t* src, std::size_t n, int level)
{
  CodecResult r;
  r.name = "zstd-L" + std::to_string(level);
  r.input_bytes = n;

  std::size_t cap = ZSTD_compressBound(n);
  std::vector<std::uint8_t> dst(cap);
  auto t0 = clk::now();
  std::size_t out = ZSTD_compress(dst.data(), cap, src, n, level);
  if(ZSTD_isError(out))
    throw std::runtime_error(std::string{"zstd: "} + ZSTD_getErrorName(out));
  r.encode_ms = ms_since(t0);
  r.output_bytes = out;
  return r;
}

// --- lz4 ---------------------------------------------------------------
inline CodecResult c_lz4(const std::uint8_t* src, std::size_t n)
{
  CodecResult r;
  r.name = "lz4";
  r.input_bytes = n;

  int cap = LZ4_compressBound(static_cast<int>(n));
  std::vector<std::uint8_t> dst(cap);
  auto t0 = clk::now();
  int out = LZ4_compress_default(
      reinterpret_cast<const char*>(src), reinterpret_cast<char*>(dst.data()),
      static_cast<int>(n), cap);
  if(out <= 0)
    throw std::runtime_error("lz4 failed");
  r.encode_ms = ms_since(t0);
  r.output_bytes = out;
  return r;
}

inline CodecResult c_lz4hc(const std::uint8_t* src, std::size_t n, int level)
{
  CodecResult r;
  r.name = "lz4hc-L" + std::to_string(level);
  r.input_bytes = n;

  int cap = LZ4_compressBound(static_cast<int>(n));
  std::vector<std::uint8_t> dst(cap);
  auto t0 = clk::now();
  int out = LZ4_compress_HC(
      reinterpret_cast<const char*>(src), reinterpret_cast<char*>(dst.data()),
      static_cast<int>(n), cap, level);
  if(out <= 0)
    throw std::runtime_error("lz4hc failed");
  r.encode_ms = ms_since(t0);
  r.output_bytes = out;
  return r;
}

// --- snappy ------------------------------------------------------------
inline CodecResult c_snappy(const std::uint8_t* src, std::size_t n)
{
  CodecResult r;
  r.name = "snappy";
  r.input_bytes = n;

  std::string dst;
  auto t0 = clk::now();
  std::size_t out = snappy::Compress(reinterpret_cast<const char*>(src), n, &dst);
  r.encode_ms = ms_since(t0);
  r.output_bytes = out;
  return r;
}

// --- brotli ------------------------------------------------------------
inline CodecResult c_brotli(const std::uint8_t* src, std::size_t n, int quality)
{
  CodecResult r;
  r.name = "brotli-q" + std::to_string(quality);
  r.input_bytes = n;

  std::size_t cap = BrotliEncoderMaxCompressedSize(n);
  if(cap == 0)
    cap = n + 1024;
  std::vector<std::uint8_t> dst(cap);
  auto t0 = clk::now();
  if(!BrotliEncoderCompress(
         quality, BROTLI_DEFAULT_WINDOW, BROTLI_DEFAULT_MODE, n, src, &cap, dst.data()))
    throw std::runtime_error("brotli failed");
  r.encode_ms = ms_since(t0);
  r.output_bytes = cap;
  return r;
}

// --- lzo ---------------------------------------------------------------
inline CodecResult c_lzo(const std::uint8_t* src, std::size_t n)
{
  static bool inited = false;
  if(!inited)
  {
    if(lzo_init() != LZO_E_OK)
      throw std::runtime_error("lzo init failed");
    inited = true;
  }

  CodecResult r;
  r.name = "lzo1x-1";
  r.input_bytes = n;

  // worst case for lzo1x: n + n/16 + 64 + 3
  std::vector<std::uint8_t> dst(n + n / 16 + 128);
  std::vector<std::uint8_t> wrk(LZO1X_1_MEM_COMPRESS);

  lzo_uint out = 0;
  auto t0 = clk::now();
  int rc = lzo1x_1_compress(src, n, dst.data(), &out, wrk.data());
  if(rc != LZO_E_OK)
    throw std::runtime_error("lzo failed");
  r.encode_ms = ms_since(t0);
  r.output_bytes = out;
  return r;
}

} // namespace pcc
