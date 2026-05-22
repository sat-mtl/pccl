// Shared helpers for the test suite.

#pragma once

#include <pccl/pccl.hpp>

#include <cmath>

#include <algorithm>
#include <array>
#include <cstdint>
#include <random>
#include <set>
#include <vector>

namespace pccl_test
{

inline pccl::AlignedVec<float>
synth_cloud(std::size_t n, float r = 1.f, std::uint32_t seed = 1234)
{
  pccl::AlignedVec<float> xyz(3 * n);
  std::mt19937 g{seed};
  std::uniform_real_distribution<float> u{-r, r};
  for(auto& f : xyz)
    f = u(g);
  return xyz;
}

inline pccl::AlignedVec<std::array<float, 3>> voxelise(
    const pccl::AlignedVec<float>& xyz, std::array<float, 3> lo, std::array<float, 3> hi,
    std::uint8_t bits)
{
  const std::uint32_t maxv = (1u << bits) - 1u;
  const float fmax = (float)maxv;
  float scale[3], invscale[3];
  for(int k = 0; k < 3; ++k)
  {
    float e = hi[k] - lo[k];
    scale[k] = (e > 0.f) ? (fmax / e) : 0.f;
    invscale[k] = (e > 0.f) ? (e / fmax) : 0.f;
  }
  std::set<std::array<std::uint32_t, 3>> seen;
  const std::size_t n = xyz.size() / 3;
  for(std::size_t i = 0; i < n; ++i)
  {
    std::array<std::uint32_t, 3> q;
    for(int k = 0; k < 3; ++k)
    {
      float v = (xyz[3 * i + k] - lo[k]) * scale[k];
      if(v < 0.f)
        v = 0.f;
      if(v > fmax)
        v = fmax;
      q[k] = (std::uint32_t)(v + 0.5f);
    }
    seen.insert(q);
  }
  pccl::AlignedVec<std::array<float, 3>> out;
  out.reserve(seen.size());
  for(auto& q : seen)
    out.push_back(
        {lo[0] + q[0] * invscale[0], lo[1] + q[1] * invscale[1],
         lo[2] + q[2] * invscale[2]});
  std::sort(out.begin(), out.end());
  return out;
}

inline pccl::AlignedVec<std::array<float, 3>>
decoded_voxels_sorted(const pccl::AlignedVec<float>& xyz, std::size_t unique_voxels)
{
  pccl::AlignedVec<std::array<float, 3>> out(unique_voxels);
  for(std::size_t i = 0; i < unique_voxels; ++i)
    out[i] = {xyz[3 * i + 0], xyz[3 * i + 1], xyz[3 * i + 2]};
  std::sort(out.begin(), out.end());
  return out;
}

inline bool voxels_match(
    const pccl::AlignedVec<std::array<float, 3>>& want,
    const pccl::AlignedVec<std::array<float, 3>>& have, std::array<float, 3> lo,
    std::array<float, 3> hi, std::uint8_t bits)
{
  if(want.size() != have.size())
    return false;
  float tol[3];
  for(int k = 0; k < 3; ++k)
  {
    float fmax = (float)((1u << bits) - 1u);
    tol[k] = 1.5f * (hi[k] - lo[k]) / fmax;
  }
  for(std::size_t i = 0; i < want.size(); ++i)
    for(int k = 0; k < 3; ++k)
      if(std::fabs(want[i][k] - have[i][k]) > tol[k])
        return false;
  return true;
}

} // namespace pccl_test
