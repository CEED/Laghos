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
  RajaCommunicator::RajaCommunicator(GroupTopology &gt): GroupCommunicator(gt){ }

  
  // ***************************************************************************
  // * ~RajaCommunicator
  // ***************************************************************************
  RajaCommunicator::~RajaCommunicator(){ }

  // ***************************************************************************
  // ***************************************************************************
  template <class T>
  T *RajaCommunicator::d_CopyGroupToBuffer(const T *ldata,
                                           T *buf, int group,
                                           int layout) const  {
    switch (layout) {
    case 1:
      {
        return std::copy(ldata + group_ldof.GetI()[group],
                         ldata + group_ldof.GetI()[group+1],
                         buf);
      }
    case 2:
      {
        const int nltdofs = group_ltdof.RowSize(group);
        const int *ltdofs = group_ltdof.GetRow(group);
        for (int j = 0; j < nltdofs; j++)
        {
          buf[j] = ldata[ltdofs[j]];
        }
        return buf + nltdofs;
      }
    default:
      {
        const int nldofs = group_ldof.RowSize(group);
        const int *ldofs = group_ldof.GetRow(group);
        for (int j = 0; j < nldofs; j++)
        {
          buf[j] = ldata[ldofs[j]];
        }
        return buf + nldofs;
      }
    }
  }

  // ***************************************************************************
  // ***************************************************************************
  template <class T>
  const T *RajaCommunicator::d_CopyGroupFromBuffer(const T *buf, T *ldata,
                                                   int group, int layout) const{
    const int nldofs = group_ldof.RowSize(group);
    switch (layout)
    {
    case 1:
      {
        std::copy(buf, buf + nldofs, ldata + group_ldof.GetI()[group]);
        break;
      }
    case 2:
      {
        const int *ltdofs = group_ltdof.GetRow(group);
        for (int j = 0; j < nldofs; j++)
        {
          ldata[ltdofs[j]] = buf[j];
        }
        break;
      }
    default:
      {
        const int *ldofs = group_ldof.GetRow(group);
        for (int j = 0; j < nldofs; j++)
        {
          ldata[ldofs[j]] = buf[j];
        }
        break;
      }
    }
    return buf + nldofs;
  }

  // ***************************************************************************
  // * ReduceGroupFromBuffer
  // ***************************************************************************
  template <class T>
  const T *RajaCommunicator::d_ReduceGroupFromBuffer(const T *buf, T *ldata,
                                                     int group, int layout,
                                                     void (*Op)(OpData<T>)) const  {
    OpData<T> opd;
    opd.ldata = ldata;
    opd.nldofs = group_ldof.RowSize(group);
    opd.nb = 1;
    opd.buf = const_cast<T*>(buf);

    switch (layout)
    {
    case 1:
      {
        MFEM_ABORT("layout 1 is not supported");
        T *dest = ldata + group_ldof.GetI()[group];
        for (int j = 0; j < opd.nldofs; j++)
        {
          dest[j] += buf[j];
        }
        break;
      }
    case 2:
      {
        opd.ldofs = const_cast<int*>(group_ltdof.GetRow(group));
        Op(opd);
        break;
      }
    default:
      {
        opd.ldofs = const_cast<int*>(group_ldof.GetRow(group));
        Op(opd);
        break;
      }
    }
    return buf + opd.nldofs;
  }


  // ***************************************************************************
  // * d_BcastBegin
  // ***************************************************************************
  template <class T>
  void RajaCommunicator::d_BcastBegin(T *ldata, int layout)  {
    MFEM_VERIFY(comm_lock == 0, "object is already in use");

    if (group_buf_size == 0) { return; }
    
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
          buf = d_CopyGroupToBuffer(ldata, buf, grp_list[i], layout);
        }
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
      }
    }
    MFEM_ASSERT(buf - (T*)group_buf.GetData() == group_buf_size, "");
    comm_lock = 1; // 1 - locked fot Bcast
    num_requests = request_counter;
  }

  // ***************************************************************************
  // * d_BcastEnd
  // ***************************************************************************
  template <class T>
  void RajaCommunicator::d_BcastEnd(T *ldata, int layout) {
    if (comm_lock == 0) { return; }
    // The above also handles the case (group_buf_size == 0).
    MFEM_VERIFY(comm_lock == 1, "object is NOT locked for Bcast");
    // copy the received data from the buffer to ldata, as it arrives
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
          buf = d_CopyGroupFromBuffer(buf, ldata, grp_list[i], layout);
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
  void RajaCommunicator::d_ReduceBegin(const T *ldata) {
    MFEM_VERIFY(comm_lock == 0, "object is already in use");

    if (group_buf_size == 0) { return; }

    int request_counter = 0;
    group_buf.SetSize(group_buf_size*sizeof(T));
    T *buf = (T *)group_buf.GetData();
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
          const int layout = 0; // ldata is an array on all ldofs
          buf = d_CopyGroupToBuffer(ldata, buf, grp_list[i], layout);
        }
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
      }
    }
    MFEM_ASSERT(buf - (T*)group_buf.GetData() == group_buf_size, "");
    comm_lock = 2;
    num_requests = request_counter;
  }

  // ***************************************************************************
  // * d_ReduceEnd
  // ***************************************************************************
  template <class T>
  void RajaCommunicator::d_ReduceEnd(T *ldata, int layout, void (*Op)(OpData<T>))
  {
    if (comm_lock == 0) { return; }
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
          buf = d_ReduceGroupFromBuffer(buf, ldata, grp_list[i],
                                        layout, Op);
        }
      }
    }
    comm_lock = 0; // 0 - no lock
    num_requests = 0;
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
