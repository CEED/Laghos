#pragma once

#include <iostream>

#include "linalg/tensor_arrays.hpp"

namespace mfem::future
{

template <typename scalar_t, int ndims, int... tensor_sizes>
void print_tensor_ndarray(
   const tensor_ndarray<scalar_t, ndims, tensor_sizes...>& ta,
   std::ostream& os = std::cout,
   const std::string& name = "tensor_ndarray")
{
   constexpr int tdim = sizeof...(tensor_sizes);

   os << name << "  [shape: ";
   for (int d = 0; d < ndims; ++d)
   {
      if (d > 0) { os << " × "; }
      os << ta.size(d);
   }
   if constexpr (tdim > 0)
   {
      os << "  | tensor: ";
      bool first = true;
      (..., (first ? (first = false,
                      os << tensor_sizes) : (os << "×" << tensor_sizes)));
   }
   os << "]\n";

   const std::size_t total = ta.total_size();
   if (total == 0)
   {
      os << "  (empty)\n\n";
      return;
   }

   for (std::size_t flat = 0; flat < total; ++flat)
   {
      // Compute multi-index from flat index (row-major / last index varies fastest)
      std::array<std::size_t, ndims> idx{};
      std::size_t rem = flat;
      for (int d = ndims - 1; d >= 0; --d)
      {
         idx[d] = rem % ta.size(d);
         rem /= ta.size(d);
      }

      // Print indices nicely
      os << " [";
      for (int d = 0; d < ndims; ++d)
      {
         if (d > 0) { os << ", "; }
         os << idx[d];
      }
      os << "]  (flat " << flat << "): ";

      // Get and print the tensor
      const auto t = ta.get_tensor(idx);
      os << t << "\n";
   }

   os << "\n";
}

} // namespace mfem::future