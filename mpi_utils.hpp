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