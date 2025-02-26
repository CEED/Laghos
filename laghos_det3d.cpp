
#include "fem/fespace.hpp"

#include "fem/quadinterpolator.hpp"

#include "laghos_det3d.hpp"

namespace mfem
{
// DIM, SDIM, D1D, Q1D, SMEM

template <>
QuadratureInterpolator::DetKernelType
QuadratureInterpolator::DetKernels::Kernel<3, 3, 2, 2>()
{ return internal::quadrature_interpolator::Det3D<2, 2>; }

template <>
QuadratureInterpolator::DetKernelType
QuadratureInterpolator::DetKernels::Kernel<3, 3, 3, 4>()
{ return internal::quadrature_interpolator::Det3D<3, 4>; }

template <>
QuadratureInterpolator::DetKernelType
QuadratureInterpolator::DetKernels::Kernel<3, 3, 5, 8>()
{ return internal::quadrature_interpolator::Det3D<5, 8>; }

template <>
QuadratureInterpolator::DetKernelType
QuadratureInterpolator::DetKernels::Kernel<3, 3, 9, 16>()
{ return internal::quadrature_interpolator::Det3D<9, 16>; }

} // mfem namespace