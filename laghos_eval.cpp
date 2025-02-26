
#include "fem/fespace.hpp"

#include "fem/quadinterpolator.hpp"

#include "laghos_eval.hpp"

namespace mfem
{

template <>
QuadratureInterpolator::TensorEvalKernelType
QuadratureInterpolator::TensorEvalKernels::Kernel<3, QVectorLayout::byNODES, 1, 1, 2, 1>()
{ return internal::quadrature_interpolator::Values3D<QVectorLayout::byNODES, 1, 1, 2>; }

template <>
QuadratureInterpolator::TensorEvalKernelType
QuadratureInterpolator::TensorEvalKernels::Kernel<3, QVectorLayout::byVDIM, 1, 1, 2, 1>()
{ return internal::quadrature_interpolator::Values3D<QVectorLayout::byVDIM, 1, 1, 2>; }

template <>
QuadratureInterpolator::TensorEvalKernelType
QuadratureInterpolator::TensorEvalKernels::Kernel<3, QVectorLayout::byVDIM, 3, 3, 4, 1>()
{ return internal::quadrature_interpolator::Values3D<QVectorLayout::byVDIM, 3, 3, 4>; }

template <>
QuadratureInterpolator::TensorEvalKernelType
QuadratureInterpolator::TensorEvalKernels::Kernel<3, QVectorLayout::byVDIM, 3, 5, 8, 1>()
{ return internal::quadrature_interpolator::Values3D<QVectorLayout::byVDIM, 3, 5, 8>; }

template <>
QuadratureInterpolator::TensorEvalKernelType
QuadratureInterpolator::TensorEvalKernels::Kernel<3, QVectorLayout::byNODES, 1, 8, 16, 1>()
{ return internal::quadrature_interpolator::Values3D<QVectorLayout::byNODES, 1, 8, 16>; }

template <>
QuadratureInterpolator::TensorEvalKernelType
QuadratureInterpolator::TensorEvalKernels::Kernel<3, QVectorLayout::byVDIM, 1, 8, 16, 1>()
{ return internal::quadrature_interpolator::Values3D<QVectorLayout::byVDIM, 1, 8, 16>; }

template <>
QuadratureInterpolator::TensorEvalKernelType
QuadratureInterpolator::TensorEvalKernels::Kernel<3, QVectorLayout::byVDIM, 3, 9, 16, 1>()
{ return internal::quadrature_interpolator::Values3D<QVectorLayout::byVDIM, 3, 9, 16>; }


} // mfem namespace