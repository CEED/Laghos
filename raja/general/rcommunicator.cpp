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
#include "../raja.hpp"

namespace mfem {

  // ***************************************************************************
  // * RajaCommunicator
  // ***************************************************************************
  RajaCommunicator::RajaCommunicator(ParFiniteElementSpace &pfes): GroupCommunicator(pfes.GroupComm()){ }

  
  // ***************************************************************************
  // * ~RajaCommunicator
  // ***************************************************************************
  RajaCommunicator::~RajaCommunicator(){ }

  // ***************************************************************************
  // * d_CopyGroupToBuffer
  // ***************************************************************************
  template <class T>
  T *RajaCommunicator::d_CopyGroupToBuffer(const T *d_ldata,
                                           T *buf, int group,
                                           int layout) const  {
    push(Yellow);
    dbg("\n\t\033[33m[d_CopyGroupToBuffer]\033[m");
    switch (layout) {
    case 1:
      {
        dbg("\n\t\t\033[33m[d_CopyGroupToBuffer] layout 1\033[m");
#ifndef __NVCC__
        pop();
        return std::copy(d_ldata + group_ldof.GetI()[group],
                         d_ldata + group_ldof.GetI()[group+1],
                         buf);
#else
        T* dest = buf;
        const CUdeviceptr src = (CUdeviceptr)(d_ldata + group_ldof.GetI()[group]);
        //assert(group_ldof.GetI()[group+1]>=group_ldof.GetI()[group]);
        const size_t sz = group_ldof.GetI()[group+1]-group_ldof.GetI()[group];
        checkCudaErrors(cuMemcpyDtoHAsync(dest,src,sz*sizeof(T),0));
#endif
      }
    case 2:
      {
        dbg("\n\t\t\033[33m[d_CopyGroupToBuffer] layout 2\033[m");
        const int nltdofs = group_ltdof.RowSize(group);
        const int *ltdofs = group_ltdof.GetRow(group);
        for (int j = 0; j < nltdofs; j++)
        {
#ifndef __NVCC__
          buf[j] = d_ldata[ltdofs[j]];
#else
          T* dest = buf+j;
          const CUdeviceptr src = (CUdeviceptr)(d_ldata+ltdofs[j]);
          const size_t sz = 1;
          checkCudaErrors(cuMemcpyDtoHAsync(dest,src,sz*sizeof(T),0));
#endif
        }
        pop();
        return buf + nltdofs;
      }
    default:
      {
        dbg("\n\t\t\033[33m[d_CopyGroupToBuffer] default\033[m");
        const int nldofs = group_ldof.RowSize(group);
        const int *ldofs = group_ldof.GetRow(group);
        for (int j = 0; j < nldofs; j++)
        {
#ifndef __NVCC__
          buf[j] = d_ldata[ldofs[j]];
#else
          T* dest = buf+j;
          const CUdeviceptr src = (CUdeviceptr)(d_ldata+ldofs[j]);
          const size_t sz = 1;
          checkCudaErrors(cuMemcpyDtoHAsync(dest,src,sz*sizeof(T),0));
#endif
        }
        dbg("\n\t\t\033[33m[d_CopyGroupToBuffer] done\033[m");
        pop();
        return buf + nldofs;
      }
    }
  }

  // ***************************************************************************
  // ***************************************************************************
  template <class T>
  const T *RajaCommunicator::d_CopyGroupFromBuffer(const T *buf, T *d_ldata,
                                                   int group, int layout) const{
    push(Gold);
    dbg("\n\t\033[33m[d_CopyGroupFromBuffer]\033[m");
    const int nldofs = group_ldof.RowSize(group);
    switch (layout)
    {
    case 1:
      {
        dbg("\n\t\t\033[33m[d_CopyGroupFromBuffer] layout 1\033[m");
        std::copy(buf, buf + nldofs, d_ldata + group_ldof.GetI()[group]);
        break;
      }
    case 2:
      {
        dbg("\n\t\t\033[33m[d_CopyGroupFromBuffer] layout 2\033[m");
        const int *ltdofs = group_ltdof.GetRow(group);
        for (int j = 0; j < nldofs; j++)
        {
          d_ldata[ltdofs[j]] = buf[j];
        }
        break;
      }
    default:
      {
        dbg("\n\t\t\033[33m[d_CopyGroupFromBuffer] default\033[m");
        const int *ldofs = group_ldof.GetRow(group);
        for (int j = 0; j < nldofs; j++)
        {
#ifndef __NVCC__
          d_ldata[ldofs[j]] = buf[j];
#else
          CUdeviceptr dest = (CUdeviceptr)(d_ldata+ldofs[j]);
          const T* src = buf+j;
          const size_t sz = 1;
          checkCudaErrors(cuMemcpyHtoDAsync(dest,src,sz*sizeof(T),0));
#endif
        }
        break;
      }
    }
    dbg("\n\t\t\033[33m[d_CopyGroupFromBuffer] done\033[m");
    pop();
    return buf + nldofs;
  }

  // ***************************************************************************
  // * ReduceGroupFromBuffer
  // ***************************************************************************
  template <class T>
  const T *RajaCommunicator::d_ReduceGroupFromBuffer(const T *buf, T *d_ldata,
                                                     int group, int layout,
                                                     void (*Op)(OpData<T>)) const  {
    push(PaleGoldenrod);
    dbg("\n\t\033[33m[d_ReduceGroupFromBuffer]\033[m");
    OpData<T> opd;
    opd.ldata = d_ldata;
    opd.nldofs = group_ldof.RowSize(group);
    opd.nb = 1;
    opd.buf = const_cast<T*>(buf);
    dbg("\n\t\t\033[33m[d_ReduceGroupFromBuffer] layout 2\033[m");
    opd.ldofs = const_cast<int*>(group_ltdof.GetRow(group));
#ifndef __NVCC__
    Op(opd);
#else
    assert(opd.nb == 1);
    // this is the operation to perform:
    // opd.ldata[opd.ldofs[i]] += opd.buf[i];
    // mfem/general/communication.cpp, line 1008
    T h_opdldata_opdldofs_i;
    // Transfer from device the value to +=
    T* dest = &h_opdldata_opdldofs_i;
    for (int i = 0; i < opd.nldofs; i++){
      const CUdeviceptr src = (CUdeviceptr)(opd.ldata+opd.ldofs[i]);
      checkCudaErrors(cuMemcpyDtoH(dest,src,sizeof(T)));
      // Do the +=
      h_opdldata_opdldofs_i += opd.buf[i];
      // Push back the answer          
      checkCudaErrors(cuMemcpyHtoD(src,dest,sizeof(T)));
    }
#endif // __NVCC__
    dbg("\n\t\t\033[33m[d_ReduceGroupFromBuffer] done\033[m");
    pop();
    return buf + opd.nldofs;
  }


  // ***************************************************************************
  // * d_BcastBegin
  // ***************************************************************************
  template <class T>
  void RajaCommunicator::d_BcastBegin(T *d_ldata, int layout) {
    MFEM_VERIFY(comm_lock == 0, "object is already in use");
    if (group_buf_size == 0) { return; }
    
    push(Moccasin);
    const int rnk = rconfig::Get().Rank();
    dbg("\n\033[33;1m[%d-d_BcastBegin]\033[m",rnk);
    int request_counter = 0;
    group_buf.SetSize(group_buf_size*sizeof(T));
    T *buf = (T *)group_buf.GetData();
    for (int nbr = 1; nbr < nbr_send_groups.Size(); nbr++)
    {
      const int num_send_groups = nbr_send_groups.RowSize(nbr);
      if (num_send_groups > 0)
      {
        // Possible optimization:
        //    if (num_send_groups == 1) and (layout == 1) then we do not
        //    need to copy the data in order to send it.
        T *buf_start = buf;
        const int *grp_list = nbr_send_groups.GetRow(nbr);
        for (int i = 0; i < num_send_groups; i++)
        {
          buf = d_CopyGroupToBuffer(d_ldata, buf, grp_list[i], layout);
        }
        push(MPI_Isend,Orange);
        MPI_Isend(buf_start,
                  buf - buf_start,
                  MPITypeMap<T>::mpi_type,
                  gtopo.GetNeighborRank(nbr),
                  40822,
                  gtopo.GetComm(),
                  &requests[request_counter]);
        pop();
        request_marker[request_counter] = -1; // mark as send request
        request_counter++;
      }

      const int num_recv_groups = nbr_recv_groups.RowSize(nbr);
      if (num_recv_groups > 0)
      {
        // Possible optimization (requires interface change):
        //    if (num_recv_groups == 1) and the (output layout == 1) then
        //    we can receive directly in the output buffer; however, at
        //    this point we do not have that information.
        const int *grp_list = nbr_recv_groups.GetRow(nbr);
        int recv_size = 0;
        for (int i = 0; i < num_recv_groups; i++)
        {
          recv_size += group_ldof.RowSize(grp_list[i]);
        }
        push(MPI_Irecv,Orange);
        MPI_Irecv(buf,
                  recv_size,
                  MPITypeMap<T>::mpi_type,
                  gtopo.GetNeighborRank(nbr),
                  40822,
                  gtopo.GetComm(),
                  &requests[request_counter]);
        pop();
        request_marker[request_counter] = nbr;
        request_counter++;
        buf_offsets[nbr] = buf - (T*)group_buf.GetData();
        buf += recv_size;
      }
    }
    MFEM_ASSERT(buf - (T*)group_buf.GetData() == group_buf_size, "");
    comm_lock = 1; // 1 - locked fot Bcast
    num_requests = request_counter;
    dbg("\n\033[33;1m[%d-d_BcastBegin] done\033[m",rnk);
    pop();
  }

  // ***************************************************************************
  // * d_BcastEnd
  // ***************************************************************************
  template <class T>
  void RajaCommunicator::d_BcastEnd(T *d_ldata, int layout) {
    if (comm_lock == 0) { return; }
    push(PeachPuff);
    const int rnk = rconfig::Get().Rank();
    dbg("\n\033[33;1m[%d-d_BcastEnd]\033[m",rnk);
    // The above also handles the case (group_buf_size == 0).
    MFEM_VERIFY(comm_lock == 1, "object is NOT locked for Bcast");
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
        const T *buf = (T*)group_buf.GetData() + buf_offsets[nbr];
        for (int i = 0; i < num_recv_groups; i++)
        {
          buf = d_CopyGroupFromBuffer(buf, d_ldata, grp_list[i], layout);
        }
      }
    }
    comm_lock = 0; // 0 - no lock
    num_requests = 0;
    dbg("\n\033[33;1m[%d-d_BcastEnd] done\033[m",rnk);
    pop();
  }

  // ***************************************************************************
  // * d_ReduceBegin
  // ***************************************************************************
  template <class T>
  void RajaCommunicator::d_ReduceBegin(const T *d_ldata) {
    MFEM_VERIFY(comm_lock == 0, "object is already in use");
    if (group_buf_size == 0) { return; }
    push(PapayaWhip);
    const int rnk = rconfig::Get().Rank();
    dbg("\n\033[33;1m[%d-d_ReduceBegin]\033[m",rnk);

    int request_counter = 0;
    group_buf.SetSize(group_buf_size*sizeof(T));
    T *buf = (T *)group_buf.GetData();
    dbg("\n\033[33;1m[%d-d_ReduceBegin] nbr_send_groups.Size()=%d\033[m",rnk,
           nbr_send_groups.Size());
    for (int nbr = 1; nbr < nbr_send_groups.Size(); nbr++)
    {
      // In Reduce operation: send_groups <--> recv_groups 
      const int num_send_groups = nbr_recv_groups.RowSize(nbr);
      if (num_send_groups > 0)
      {
        T *buf_start = buf;
        const int *grp_list = nbr_recv_groups.GetRow(nbr);
        for (int i = 0; i < num_send_groups; i++)
        {
          const int layout = 0; // d_ldata is an array on all ldofs
          buf = d_CopyGroupToBuffer(d_ldata, buf, grp_list[i], layout);
        }
        dbg("\n\033[33;1m[%d-d_ReduceBegin] MPI_Isend\033[m",rnk);
        push(MPI_Isend,Orange);
        MPI_Isend(buf_start,
                  buf - buf_start,
                  MPITypeMap<T>::mpi_type,
                  gtopo.GetNeighborRank(nbr),
                  43822,
                  gtopo.GetComm(),
                  &requests[request_counter]);
        pop();
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
        dbg("\n\033[33;1m[%d-d_ReduceBegin] MPI_Irecv\033[m",rnk);
        push(MPI_Irecv,Orange);
        MPI_Irecv(buf,
                  recv_size,
                  MPITypeMap<T>::mpi_type,
                  gtopo.GetNeighborRank(nbr),
                  43822,
                  gtopo.GetComm(),
                  &requests[request_counter]);
        pop();
        request_marker[request_counter] = nbr;
        request_counter++;
        buf_offsets[nbr] = buf - (T*)group_buf.GetData();
        buf += recv_size;
      }
    }
    MFEM_ASSERT(buf - (T*)group_buf.GetData() == group_buf_size, "");
    comm_lock = 2;
    num_requests = request_counter;
    dbg("\n\033[33;1m[%d-d_ReduceBegin] done\033[m",rnk);
    pop();
  }

  // ***************************************************************************
  // * d_ReduceEnd
  // ***************************************************************************
  template <class T>
  void RajaCommunicator::d_ReduceEnd(T *d_ldata, int layout,
                                     void (*Op)(OpData<T>)){
    if (comm_lock == 0) { return; }
    push(LavenderBlush);
    const int rnk = rconfig::Get().Rank();
    dbg("\n\033[33;1m[%d-d_ReduceEnd]\033[m",rnk);
    // The above also handles the case (group_buf_size == 0).
    MFEM_VERIFY(comm_lock == 2, "object is NOT locked for Reduce");
    MPI_Waitall(num_requests, requests, MPI_STATUSES_IGNORE);

    for (int nbr = 1; nbr < nbr_send_groups.Size(); nbr++)
    {
      // In Reduce operation: send_groups <--> recv_groups
      const int num_recv_groups = nbr_send_groups.RowSize(nbr);
      if (num_recv_groups > 0)
      {
        const int *grp_list = nbr_send_groups.GetRow(nbr);
        const T *buf = (T*)group_buf.GetData() + buf_offsets[nbr];
        for (int i = 0; i < num_recv_groups; i++)
        {
          buf = d_ReduceGroupFromBuffer(buf, d_ldata, grp_list[i],
                                        layout, Op);
        }
      }
    }
    comm_lock = 0; // 0 - no lock
    num_requests = 0;
    dbg("\n\033[33;1m[%d-d_ReduceEnd] end\033[m",rnk);
    pop();
  }

  // ***************************************************************************
  // * instantiate RajaCommunicator::Bcast and Reduce for int and double
  // ***************************************************************************
  template void RajaCommunicator::d_BcastBegin<int>(int*, int);
  template void RajaCommunicator::d_BcastEnd<int>(int*, int);
  template void RajaCommunicator::d_ReduceBegin<int>(const int*);
  template void RajaCommunicator::d_ReduceEnd<int>(int*,int,void (*)(OpData<int>));

  template void RajaCommunicator::d_BcastBegin<double>(double*, int);
  template void RajaCommunicator::d_BcastEnd<double>(double*, int);
  template void RajaCommunicator::d_ReduceBegin<double>(const double *);
  template void RajaCommunicator::d_ReduceEnd<double>(double*,int,void (*)(OpData<double>));

} // namespace mfem
