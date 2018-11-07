// Copyright (c) 2017, Lawrence Livermore National Security, LLC. Produced at
// the Lawrence Livermore National Laboratory. LLNL-CODE-734707. All Rights
// reserved. See files LICENSE and NOTICE for details.
//
// This file is part of CEED, a collection of benchmarks, miniapps, software
// libraries and APIs for efficient high-order finite element and spectral
// element discretizations for exascale applications. For more information and
// source code availability see http://github.com/ceed.
//
// The CEED research is supported by the Exascale Computing Project 17-SC-20-SC,
// a collaborative effort of two U.S. Department of Energy organizations (Office
// of Science and the National Nuclear Security Administration) responsible for
// the planning and preparation of a capable exascale ecosystem, including
// software, applications, hardware, advanced system engineering and early
// testbed platforms, in support of the nation's exascale computing imperative.

#include <iostream>
#include <sstream>
#include <mpi.h>

class MpiFlush {
public:
  inline MpiFlush() {}
};

class MpiOstream {
public:
  int rank, procs;
  std::stringstream ss;

  static MpiFlush flush;

  inline MpiOstream() {}

  inline void setup() {
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &procs);
  }

  inline void _flush() {
    std::string message = ss.str();
    ss.str("");
    ss << "[" << rank << "/" << procs << "] ";
    std::string prefix = ss.str();
    ss.str("");
    std::string indent = "\n" + std::string(prefix.length(), ' ');
    for (int i = 0; i < (int) message.size(); ++i) {
      if ((message[i] != '\n') || (i == (message.size() - 1))) {
        ss << message[i];
      } else {
        ss << indent;
      }
    }
    message = ss.str();
    ss.str("");
    for (int i = 0; i < procs; ++i) {
      MPI_Barrier(MPI_COMM_WORLD);
      if (i == rank) {
        std::cout << prefix << message;
      }
    }
    ss.str("");
  }

  template <class TM>
  MpiOstream& operator << (const TM &t) {
    ss << t;
    return *this;
  }
};

template <>
inline MpiOstream& MpiOstream::operator << (const MpiFlush &t) {
  _flush();
  return *this;
}

extern MpiOstream mpiout;
