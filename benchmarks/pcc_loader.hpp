#pragma once

#include <miniply.h>

#include <array>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace pcc
{

struct PointCloud
{
  std::vector<float> xyz; // interleaved x, y, z
  std::array<float, 3> aabb_min{};
  std::array<float, 3> aabb_max{};

  std::size_t num_points() const noexcept { return xyz.size() / 3; }
  std::size_t bytes() const noexcept { return xyz.size() * sizeof(float); }
};

inline PointCloud load_ply_positions(const std::string& path)
{
  miniply::PLYReader reader(path.c_str());
  if(!reader.valid())
    throw std::runtime_error("miniply: cannot open " + path);

  PointCloud out;
  while(reader.has_element())
  {
    if(reader.element_is(miniply::kPLYVertexElement) && reader.load_element())
    {
      uint32_t pos_idx[3];
      if(!reader.find_pos(pos_idx))
        throw std::runtime_error("ply has no position properties");

      const std::uint32_t n = reader.num_rows();
      out.xyz.resize(std::size_t{n} * 3);
      reader.extract_properties(
          pos_idx, 3, miniply::PLYPropertyType::Float, out.xyz.data());
      break;
    }
    reader.next_element();
  }

  if(out.xyz.empty())
    throw std::runtime_error("ply has no vertex element");

  // Compute AABB.
  float lo[3]
      = {std::numeric_limits<float>::infinity(), std::numeric_limits<float>::infinity(),
         std::numeric_limits<float>::infinity()};
  float hi[3] = {
      -std::numeric_limits<float>::infinity(), -std::numeric_limits<float>::infinity(),
      -std::numeric_limits<float>::infinity()};

  const std::size_t n = out.num_points();
  for(std::size_t i = 0; i < n; ++i)
  {
    const float* p = out.xyz.data() + i * 3;
    for(int k = 0; k < 3; ++k)
    {
      if(p[k] < lo[k])
        lo[k] = p[k];
      if(p[k] > hi[k])
        hi[k] = p[k];
    }
  }
  for(int k = 0; k < 3; ++k)
  {
    out.aabb_min[k] = lo[k];
    out.aabb_max[k] = hi[k];
  }
  return out;
}

} // namespace pcc
