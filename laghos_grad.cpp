
#include "fem/fespace.hpp"

#include "fem/quadinterpolator.hpp"

#include "laghos_grad.hpp"

namespace mfem
{

template <>
QuadratureInterpolator::GradKernelType
QuadratureInterpolator::GradKernels::Kernel<3, QVectorLayout::byNODES, false, 3, 2, 2, 0>()
{ return internal::quadrature_interpolator::Derivatives3D<QVectorLayout::byNODES, false, 3, 2, 2>; }

template <>
QuadratureInterpolator::GradKernelType
QuadratureInterpolator::GradKernels::Kernel<3, QVectorLayout::byVDIM, false, 3, 2, 2, 0>()
{ return internal::quadrature_interpolator::Derivatives3D<QVectorLayout::byVDIM, false, 3, 2, 2>; }

template <>
QuadratureInterpolator::GradKernelType
QuadratureInterpolator::GradKernels::Kernel<3, QVectorLayout::byNODES, false, 3, 5, 8, 0>()
{ return internal::quadrature_interpolator::Derivatives3D<QVectorLayout::byNODES, false, 3, 5, 8>; }

template <>
QuadratureInterpolator::GradKernelType
QuadratureInterpolator::GradKernels::Kernel<3, QVectorLayout::byVDIM, false, 3, 9, 16, 0>()
{ return internal::quadrature_interpolator::Derivatives3D<QVectorLayout::byVDIM, false, 3, 9, 16>; }

template <>
QuadratureInterpolator::GradKernelType
QuadratureInterpolator::GradKernels::Kernel<3, QVectorLayout::byNODES, false, 3, 9, 16, 0>()
{ return internal::quadrature_interpolator::Derivatives3D<QVectorLayout::byNODES, false, 3, 9, 16>; }

} // mfem namespace