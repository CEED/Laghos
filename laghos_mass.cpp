
#include "fem/bilininteg.hpp"
#include "fem/integ/bilininteg_mass_kernels.hpp" // for diagonal kernels

namespace mfem
{

template <>
MassIntegrator::ApplyKernelType
MassIntegrator::ApplyPAKernels::Kernel<3,1,2>()
{ return internal::SmemPAMassApply3D<1, 2>; }

template <>
MassIntegrator::ApplyKernelType
MassIntegrator::ApplyPAKernels::Kernel<3,2,4>()
{ return internal::SmemPAMassApply3D<2, 4>; }

template <>
MassIntegrator::ApplyKernelType
MassIntegrator::ApplyPAKernels::Kernel<3,4,8>()
{ return internal::SmemPAMassApply3D<2, 4>; }

template <>
MassIntegrator::ApplyKernelType
MassIntegrator::ApplyPAKernels::Kernel<3,8,16>()
{ return internal::SmemPAMassApply3D<2, 4>; }

template <>
MassIntegrator::ApplyKernelType
MassIntegrator::ApplyPAKernels::Kernel<3,9,16>()
{ return internal::SmemPAMassApply3D<2, 4>; }

///////////////////////////////////////////////////////////////////////////////

template <>
MassIntegrator::DiagonalKernelType
MassIntegrator::DiagonalPAKernels::Kernel<3,1,2>()
{ return internal::SmemPAMassAssembleDiagonal3D<1, 2>; }

template <>
MassIntegrator::DiagonalKernelType
MassIntegrator::DiagonalPAKernels::Kernel<3,2,4>()
{ return internal::SmemPAMassAssembleDiagonal3D<2, 4>; }

template <>
MassIntegrator::DiagonalKernelType
MassIntegrator::DiagonalPAKernels::Kernel<3,4,8>()
{ return internal::SmemPAMassAssembleDiagonal3D<2, 4>; }

template <>
MassIntegrator::DiagonalKernelType
MassIntegrator::DiagonalPAKernels::Kernel<3,8,16>()
{ return internal::SmemPAMassAssembleDiagonal3D<2, 4>; }

template <>
MassIntegrator::DiagonalKernelType
MassIntegrator::DiagonalPAKernels::Kernel<3,9,16>()
{ return internal::SmemPAMassAssembleDiagonal3D<2, 4>; }

}