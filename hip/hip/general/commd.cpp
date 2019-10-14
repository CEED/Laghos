// Copyright (c) 2010, Lawrence Livermore National Security, LLC. Produced at
// the Lawrence Livermore National Laboratory. LLNL-CODE-443211. All Rights
// reserved. See file COPYRIGHT for details.
//
// This file is part of the MFEM library. For more information and source code
// availability see http://mfem.org.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the GNU Lesser General Public License (as published by the Free
// Software Foundation) version 2.1 dated February 1999.
#include "../hip.hpp"

namespace mfem
{

// ***************************************************************************
// * HipCommD
// ***************************************************************************
HipCommD::HipCommD(ParFiniteElementSpace &pfes):
   GroupCommunicator(pfes.GroupComm()),
   d_group_ldof(group_ldof),
   d_group_ltdof(group_ltdof),
   d_group_buf(NULL) {comm_lock=0;}


// ***************************************************************************
// * ~HipCommD
// ***************************************************************************
HipCommD::~HipCommD() { }


// ***************************************************************************
// * kCopyFromTable
// ***************************************************************************
template <class T> static __global__
void k_CopyGroupToBuffer(T *buf,const T *data,const int *dofs)
{
   const int j = blockDim.x * blockIdx.x + threadIdx.x;
   const int idx = dofs[j];
   buf[j]=data[idx];
}

// ***************************************************************************
// ***************************************************************************
template <class T> static
T *d_CopyGroupToBuffer_k(const T *d_ldata,T *d_buf,
                         const HipTable &d_dofs,
                         const int group)
{
   const int ndofs = d_dofs.RowSize(group);
   const int *dofs = d_dofs.GetRow(group);
   hipLaunchKernelGGL((k_CopyGroupToBuffer), dim3(ndofs), dim3(1), 0, 0, d_buf,d_ldata,dofs);
   return d_buf + ndofs;
}

// ***************************************************************************
// * d_CopyGroupToBuffer
// ***************************************************************************
template <class T>
T *HipCommD::d_CopyGroupToBuffer(const T *d_ldata, T *d_buf,
                                  int group, int layout) const
{
   if (layout==2) // master
   {
      return d_CopyGroupToBuffer_k(d_ldata,d_buf,d_group_ltdof,group);
   }
   if (layout==0) // slave
   {
      return d_CopyGroupToBuffer_k(d_ldata,d_buf,d_group_ldof,group);
   }
   assert(false);
   return 0;
}

// ***************************************************************************
// * k_CopyGroupFromBuffer
// ***************************************************************************
template <class T> static __global__
void k_CopyGroupFromBuffer(const T *buf,T *data,const int *dofs)
{
   const int j = blockDim.x * blockIdx.x + threadIdx.x;
   const int idx = dofs[j];
   data[idx]=buf[j];
}

// ***************************************************************************
// * d_CopyGroupFromBuffer
// ***************************************************************************
template <class T>
const T *HipCommD::d_CopyGroupFromBuffer(const T *d_buf, T *d_ldata,
                                          int group, int layout) const
{
   assert(layout==0);
   const int ndofs = d_group_ldof.RowSize(group);
   const int *dofs = d_group_ldof.GetRow(group);
   hipLaunchKernelGGL((k_CopyGroupFromBuffer), dim3(ndofs), dim3(1), 0, 0,
                                              d_buf,d_ldata,dofs);
   return d_buf + ndofs;
}

// ***************************************************************************
// * kAtomicAdd
// ***************************************************************************
template <class T>
static __global__ void kAtomicAdd(T* adrs, const int* dofs,T *value)
{
   const int i = blockDim.x * blockIdx.x + threadIdx.x;
   const int idx = dofs[i];
   adrs[idx] += value[i];
}
template __global__ void kAtomicAdd<int>(int*, const int*, int*);
template __global__ void kAtomicAdd<double>(double*, const int*, double*);

// ***************************************************************************
// * ReduceGroupFromBuffer
// ***************************************************************************
template <class T>
const T *HipCommD::d_ReduceGroupFromBuffer(const T *d_buf, T *d_ldata,
                                            int group, int layout,
                                            void (*Op)(OpData<T>)) const
{
   OpData<T> opd;
   opd.ldata = d_ldata;
   opd.nldofs = group_ldof.RowSize(group);
   opd.nb = 1;
   opd.buf = const_cast<T*>(d_buf);
   opd.ldofs = const_cast<int*>(d_group_ltdof.GetRow(group));
   assert(opd.nb == 1);
   hipLaunchKernelGGL((kAtomicAdd), dim3(opd.nldofs), dim3(1), 0, 0,
                      opd.ldata,opd.ldofs,opd.buf);
   return d_buf + opd.nldofs;
}


// ***************************************************************************
// * d_BcastBegin
// ***************************************************************************
template <class T>
void HipCommD::d_BcastBegin(T *d_ldata, int layout)
{
   MFEM_VERIFY(comm_lock == 0, "object is already in use");
   if (group_buf_size == 0) { return; }

   assert(layout==2);
   // const int rnk = rconfig::Get().Rank();
   int request_counter = 0;
   group_buf.SetSize(group_buf_size*sizeof(T));
   T *buf = (T *)group_buf.GetData();
   if (!d_group_buf)
   {
      d_group_buf = rmalloc<T>::operator new (group_buf_size);
   }
   T *d_buf = (T*)d_group_buf;
   for (int nbr = 1; nbr < nbr_send_groups.Size(); nbr++)
   {
      const int num_send_groups = nbr_send_groups.RowSize(nbr);
      if (num_send_groups > 0)
      {
         T *buf_start = buf;
         T *d_buf_start = d_buf;
         const int *grp_list = nbr_send_groups.GetRow(nbr);
         for (int i = 0; i < num_send_groups; i++)
         {
            T *d_buf_ini = d_buf;
            assert(layout==2);
            d_buf = d_CopyGroupToBuffer(d_ldata, d_buf, grp_list[i], 2);
            buf += d_buf - d_buf_ini;
         }
         if (!rconfig::Get().Aware())
         {
            rmemcpy::rDtoH(buf_start,d_buf_start,(buf-buf_start)*sizeof(T));
         }

         // make sure the device has finished
         if (rconfig::Get().Aware())
         {
            hipStreamSynchronize(0);//*rconfig::Get().Stream());
         }

         if (rconfig::Get().Aware())
            MPI_Isend(d_buf_start,
                      buf - buf_start,
                      MPITypeMap<T>::mpi_type,
                      gtopo.GetNeighborRank(nbr),
                      40822,
                      gtopo.GetComm(),
                      &requests[request_counter]);
         else
            MPI_Isend(buf_start,
                      buf - buf_start,
                      MPITypeMap<T>::mpi_type,
                      gtopo.GetNeighborRank(nbr),
                      40822,
                      gtopo.GetComm(),
                      &requests[request_counter]);
         request_marker[request_counter] = -1; // mark as send request
         request_counter++;
      }

      const int num_recv_groups = nbr_recv_groups.RowSize(nbr);
      if (num_recv_groups > 0)
      {
         const int *grp_list = nbr_recv_groups.GetRow(nbr);
         int recv_size = 0;
         for (int i = 0; i < num_recv_groups; i++)
         {
            recv_size += group_ldof.RowSize(grp_list[i]);
         }
         if (rconfig::Get().Aware())
            MPI_Irecv(d_buf,
                      recv_size,
                      MPITypeMap<T>::mpi_type,
                      gtopo.GetNeighborRank(nbr),
                      40822,
                      gtopo.GetComm(),
                      &requests[request_counter]);
         else
            MPI_Irecv(buf,
                      recv_size,
                      MPITypeMap<T>::mpi_type,
                      gtopo.GetNeighborRank(nbr),
                      40822,
                      gtopo.GetComm(),
                      &requests[request_counter]);
         request_marker[request_counter] = nbr;
         request_counter++;
         buf_offsets[nbr] = buf - (T*)group_buf.GetData();
         buf += recv_size;
         d_buf += recv_size;
      }
   }
   assert(buf - (T*)group_buf.GetData() == group_buf_size);
   comm_lock = 1; // 1 - locked for Bcast
   num_requests = request_counter;
}

// ***************************************************************************
// * d_BcastEnd
// ***************************************************************************
template <class T>
void HipCommD::d_BcastEnd(T *d_ldata, int layout)
{
   if (comm_lock == 0) { return; }
   // const int rnk = rconfig::Get().Rank();
   // The above also handles the case (group_buf_size == 0).
   assert(comm_lock == 1);
   // copy the received data from the buffer to d_ldata, as it arrives
   int idx;
   while (MPI_Waitany(num_requests, requests, &idx, MPI_STATUS_IGNORE),
          idx != MPI_UNDEFINED)
   {
      int nbr = request_marker[idx];
      if (nbr == -1) { continue; } // skip send requests

      const int num_recv_groups = nbr_recv_groups.RowSize(nbr);
      if (num_recv_groups > 0)
      {
         const int *grp_list = nbr_recv_groups.GetRow(nbr);
         int recv_size = 0;
         for (int i = 0; i < num_recv_groups; i++)
         {
            recv_size += group_ldof.RowSize(grp_list[i]);
         }
         const T *buf = (T*)group_buf.GetData() + buf_offsets[nbr];
         const T *d_buf = (T*)d_group_buf + buf_offsets[nbr];
         if (!rconfig::Get().Aware())
         {
            rmemcpy::rHtoD((void*)d_buf,buf,recv_size*sizeof(T));
         }
         for (int i = 0; i < num_recv_groups; i++)
         {
            d_buf = d_CopyGroupFromBuffer(d_buf, d_ldata, grp_list[i], layout);
         }
      }
   }
   comm_lock = 0; // 0 - no lock
   num_requests = 0;
}

// ***************************************************************************
// * d_ReduceBegin
// ***************************************************************************
template <class T>
void HipCommD::d_ReduceBegin(const T *d_ldata)
{
   MFEM_VERIFY(comm_lock == 0, "object is already in use");
   if (group_buf_size == 0) { return; }
   // const int rnk = rconfig::Get().Rank();
   int request_counter = 0;
   group_buf.SetSize(group_buf_size*sizeof(T));
   T *buf = (T *)group_buf.GetData();
   if (!d_group_buf)
   {
      d_group_buf = rmalloc<T>::operator new (group_buf_size);
   }
   T *d_buf = (T*)d_group_buf;
   for (int nbr = 1; nbr < nbr_send_groups.Size(); nbr++)
   {
      const int num_send_groups = nbr_recv_groups.RowSize(nbr);
      if (num_send_groups > 0)
      {
         T *buf_start = buf;
         T *d_buf_start = d_buf;
         const int *grp_list = nbr_recv_groups.GetRow(nbr);
         for (int i = 0; i < num_send_groups; i++)
         {
            T *d_buf_ini = d_buf;
            d_buf = d_CopyGroupToBuffer(d_ldata, d_buf, grp_list[i], 0);
            buf += d_buf - d_buf_ini;
         }
         if (!rconfig::Get().Aware())
         {
            rmemcpy::rDtoH(buf_start,d_buf_start,(buf-buf_start)*sizeof(T));
         }
         // make sure the device has finished
         if (rconfig::Get().Aware())
         {
            hipStreamSynchronize(0);//*rconfig::Get().Stream());
         }
         if (rconfig::Get().Aware())
            MPI_Isend(d_buf_start,
                      buf - buf_start,
                      MPITypeMap<T>::mpi_type,
                      gtopo.GetNeighborRank(nbr),
                      43822,
                      gtopo.GetComm(),
                      &requests[request_counter]);
         else
            MPI_Isend(buf_start,
                      buf - buf_start,
                      MPITypeMap<T>::mpi_type,
                      gtopo.GetNeighborRank(nbr),
                      43822,
                      gtopo.GetComm(),
                      &requests[request_counter]);
         request_marker[request_counter] = -1; // mark as send request
         request_counter++;
      }

      // In Reduce operation: send_groups <--> recv_groups
      const int num_recv_groups = nbr_send_groups.RowSize(nbr);
      if (num_recv_groups > 0)
      {
         const int *grp_list = nbr_send_groups.GetRow(nbr);
         int recv_size = 0;
         for (int i = 0; i < num_recv_groups; i++)
         {
            recv_size += group_ldof.RowSize(grp_list[i]);
         }
         if (rconfig::Get().Aware())
            MPI_Irecv(d_buf,
                      recv_size,
                      MPITypeMap<T>::mpi_type,
                      gtopo.GetNeighborRank(nbr),
                      43822,
                      gtopo.GetComm(),
                      &requests[request_counter]);
         else
            MPI_Irecv(buf,
                      recv_size,
                      MPITypeMap<T>::mpi_type,
                      gtopo.GetNeighborRank(nbr),
                      43822,
                      gtopo.GetComm(),
                      &requests[request_counter]);
         request_marker[request_counter] = nbr;
         request_counter++;
         buf_offsets[nbr] = buf - (T*)group_buf.GetData();
         buf += recv_size;
         d_buf += recv_size;
      }
   }
   assert(buf - (T*)group_buf.GetData() == group_buf_size);
   comm_lock = 2;
   num_requests = request_counter;
}

// ***************************************************************************
// * d_ReduceEnd
// ***************************************************************************
template <class T>
void HipCommD::d_ReduceEnd(T *d_ldata, int layout,
                            void (*Op)(OpData<T>))
{
   if (comm_lock == 0) { return; }
   // const int rnk = rconfig::Get().Rank();
   // The above also handles the case (group_buf_size == 0).
   assert(comm_lock == 2);
   MPI_Waitall(num_requests, requests, MPI_STATUSES_IGNORE);
   for (int nbr = 1; nbr < nbr_send_groups.Size(); nbr++)
   {
      // In Reduce operation: send_groups <--> recv_groups
      const int num_recv_groups = nbr_send_groups.RowSize(nbr);
      if (num_recv_groups > 0)
      {
         const int *grp_list = nbr_send_groups.GetRow(nbr);
         int recv_size = 0;
         for (int i = 0; i < num_recv_groups; i++)
         {
            recv_size += group_ldof.RowSize(grp_list[i]);
         }
         const T *buf = (T*)group_buf.GetData() + buf_offsets[nbr];
         assert(d_group_buf);
         const T *d_buf = (T*)d_group_buf + buf_offsets[nbr];
         if (!rconfig::Get().Aware())
         {
            rmemcpy::rHtoD((void*)d_buf,buf,recv_size*sizeof(T));
         }
         for (int i = 0; i < num_recv_groups; i++)
         {
            d_buf = d_ReduceGroupFromBuffer(d_buf, d_ldata, grp_list[i], layout, Op);
         }
      }
   }
   comm_lock = 0; // 0 - no lock
   num_requests = 0;
}

// ***************************************************************************
// * instantiate HipCommD::Bcast and Reduce for doubles
// ***************************************************************************
template void HipCommD::d_BcastBegin<double>(double*, int);
template void HipCommD::d_BcastEnd<double>(double*, int);
template void HipCommD::d_ReduceBegin<double>(const double *);
template void HipCommD::d_ReduceEnd<double>(double*,int,
                                             void (*)(OpData<double>));

} // namespace mfem
