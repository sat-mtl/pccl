// Octree occupancy encoder.
//
// Given quantized points (3 x uint32, each axis using `bits` bits), we build
// an implicit octree of depth `bits`.  We dedupe points first (one occupied
// voxel = one leaf at the deepest level).  We traverse depth-first; at each
// internal node we emit one byte whose bits indicate which of the 8 children
// contain at least one point.
//
// We never materialise the tree — we operate on the *sorted unique Morton
// codes*.  At each level, the partition of children by the 3 MSBs of the
// remaining Morton bits is a simple linear scan.
//
// The output is a byte stream that can be fed through any of our codecs.

#pragma once

#include "pcc_codec.hpp"
#include "pcc_pipeline.hpp"

#include <algorithm>
#include <cstdint>
#include <vector>

namespace pcc
{

struct OctreeStats
{
  std::size_t unique_voxels = 0;
  std::size_t occupancy_bytes = 0;
  double build_ms = 0.0;
};

inline std::vector<std::uint64_t> sort_dedup(const std::vector<std::uint64_t>& codes_in)
{
  std::vector<std::uint64_t> codes(codes_in);
  std::sort(codes.begin(), codes.end());
  codes.erase(std::unique(codes.begin(), codes.end()), codes.end());
  return codes;
}

// Iterative depth-first encoder.  Each stack frame is the half-open range
// [b, e) of (sorted, deduplicated) Morton codes belonging to the subtree
// rooted at level `lvl` (1-based: lvl == depth at root, lvl == 0 at leaves).
inline std::vector<std::uint8_t> octree_occupancy_encode(
    const std::vector<std::uint64_t>& sorted_codes, std::uint32_t depth_bits_per_axis)
{
  std::vector<std::uint8_t> out;
  out.reserve(sorted_codes.size());

  struct Frame
  {
    std::size_t b, e;
    int lvl;
  };
  std::vector<Frame> stack;
  stack.reserve(depth_bits_per_axis * 8);
  stack.push_back({0, sorted_codes.size(), (int)depth_bits_per_axis});

  while(!stack.empty())
  {
    Frame f = stack.back();
    stack.pop_back();
    if(f.lvl == 0 || f.e == f.b)
      continue;

    const int shift = (f.lvl - 1) * 3;
    const std::uint64_t* p = sorted_codes.data();

    // Partition codes[b..e) into 8 buckets by ((p[i] >> shift) & 7).
    std::size_t bounds[9];
    bounds[0] = f.b;
    std::size_t pos = f.b;
    std::uint8_t mask = 0;
    for(int c = 0; c < 8; ++c)
    {
      const std::size_t start = pos;
      const std::uint64_t want = (std::uint64_t)c;
      while(pos < f.e && ((p[pos] >> shift) & std::uint64_t{7}) == want)
        ++pos;
      if(pos > start)
        mask |= (std::uint8_t)(1u << c);
      bounds[c + 1] = pos;
    }
    out.push_back(mask);

    // Push non-empty children in reverse so processing is in-order (cache-
    // friendlier).
    for(int c = 7; c >= 0; --c)
    {
      if(bounds[c + 1] > bounds[c])
        stack.push_back({bounds[c], bounds[c + 1], f.lvl - 1});
    }
  }
  return out;
}

inline std::vector<std::uint8_t>
build_octree_from_quantized(const QuantizedCloud& q, OctreeStats& stats)
{
  auto t0 = clk::now();
  // Quantized -> Morton.  We already have to_morton() in pcc_pipeline.hpp.
  std::vector<std::uint64_t> codes = to_morton(q);
  std::sort(codes.begin(), codes.end());
  codes.erase(std::unique(codes.begin(), codes.end()), codes.end());
  stats.unique_voxels = codes.size();

  auto bytes = octree_occupancy_encode(codes, q.bits);
  stats.occupancy_bytes = bytes.size();
  stats.build_ms = ms_since(t0);
  return bytes;
}

} // namespace pcc
