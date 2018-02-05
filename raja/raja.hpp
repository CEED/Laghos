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
#ifndef LAGHOS_RAJA
#define LAGHOS_RAJA

// DBG *************************************************************************
//#include "dbg.hpp"

// stdincs *********************************************************************
#undef NDEBUG
#include <stdio.h>
#include <stdarg.h>
#include <assert.h>

// __NVCC__ ********************************************************************
#ifdef __NVCC__
#include <cuda.h>
#define push nvtxRangePush
const uint32_t colors[] = {
  0x000000, 0x800000, 0x008000, 0x808000, 0x000080, 0x800080, 0x008080,
  0xC0C0C0, 0xC0DCC0, 0xA6CAF0, 0x402000, 0x602000, 0x802000, 0xA02000,
  0xC02000, 0xE02000, 0x004000, 0x204000, 0x404000, 0x604000, 0x804000,
  0xA04000, 0xC04000, 0xE04000, 0x006000, 0x206000, 0x406000, 0x606000,
  0x806000, 0xA06000, 0xC06000, 0xE06000, 0x008000, 0x208000, 0x408000,
  0x608000, 0x808000, 0xA08000, 0xC08000, 0xE08000, 0x00A000, 0x20A000,
  0x40A000, 0x60A000, 0x80A000, 0xA0A000, 0xC0A000, 0xE0A000, 0x00C000,
  0x20C000, 0x40C000, 0x60C000, 0x80C000, 0xA0C000, 0xC0C000, 0xE0C000,
  0x00E000, 0x20E000, 0x40E000, 0x60E000, 0x80E000, 0xA0E000, 0xC0E000,
  0xE0E000, 0x000040, 0x200040, 0x400040, 0x600040, 0x800040, 0xA00040,
  0xC00040, 0xE00040, 0x002040, 0x202040, 0x402040, 0x602040, 0x802040,
  0xA02040, 0xC02040, 0xE02040, 0x004040, 0x204040, 0x404040, 0x604040,
  0x804040, 0xA04040, 0xC04040, 0xE04040, 0x006040, 0x206040, 0x406040,
  0x606040, 0x806040, 0xA06040, 0xC06040, 0xE06040, 0x008040, 0x208040,
  0x408040, 0x608040, 0x808040, 0xA08040, 0xC08040, 0xE08040, 0x00A040,
  0x20A040, 0x40A040, 0x60A040, 0x80A040, 0xA0A040, 0xC0A040, 0xE0A040,
  0x00C040, 0x20C040, 0x40C040, 0x60C040, 0x80C040, 0xA0C040, 0xC0C040,
  0xE0C040, 0x00E040, 0x20E040, 0x40E040, 0x60E040, 0x80E040, 0xA0E040,
  0xC0E040, 0xE0E040, 0x000080, 0x200080, 0x400080, 0x600080, 0x800080,
  0xA00080, 0xC00080, 0xE00080, 0x002080, 0x202080, 0x402080, 0x602080,
  0x802080, 0xA02080, 0xC02080, 0xE02080, 0x004080, 0x204080, 0x404080,
  0x604080, 0x804080, 0xA04080, 0xC04080, 0xE04080, 0x006080, 0x206080,
  0x406080, 0x606080, 0x806080, 0xA06080, 0xC06080, 0xE06080, 0x008080,
  0x208080, 0x408080, 0x608080, 0x808080, 0xA08080, 0xC08080, 0xE08080,
  0x00A080, 0x20A080, 0x40A080, 0x60A080, 0x80A080, 0xA0A080, 0xC0A080,
  0xE0A080, 0x00C080, 0x20C080, 0x40C080, 0x60C080, 0x80C080, 0xA0C080,
  0xC0C080, 0xE0C080, 0x00E080, 0x20E080, 0x40E080, 0x60E080, 0x80E080,
  0xA0E080, 0xC0E080, 0xE0E080, 0x0000C0, 0x2000C0, 0x4000C0, 0x6000C0,
  0x8000C0, 0xA000C0, 0xC000C0, 0xE000C0, 0x0020C0, 0x2020C0, 0x4020C0,
  0x6020C0, 0x8020C0, 0xA020C0, 0xC020C0, 0xE020C0, 0x0040C0, 0x2040C0,
  0x4040C0, 0x6040C0, 0x8040C0, 0xA040C0, 0xC040C0, 0xE040C0, 0x0060C0,
  0x2060C0, 0x4060C0, 0x6060C0, 0x8060C0, 0xA060C0, 0xC060C0, 0xE060C0,
  0x0080C0, 0x2080C0, 0x4080C0, 0x6080C0, 0x8080C0, 0xA080C0, 0xC080C0,
  0xE080C0, 0x00A0C0, 0x20A0C0, 0x40A0C0, 0x60A0C0, 0x80A0C0, 0xA0A0C0,
  0xC0A0C0, 0xE0A0C0, 0x00C0C0, 0x20C0C0, 0x40C0C0, 0x60C0C0, 0x80C0C0,
  0xA0C0C0, 0xFFFBF0, 0xA0A0A4, 0x808080, 0xFF0000, 0x00FF00, 0xFFFF00,
  0x0000FF, 0xFF00FF, 0x00FFFF, 0xFFFFFF
};
const int num_colors = sizeof(colors)/sizeof(uint32_t);

#define pushf() nvtxRangePush(__FUNCTION__)
#define pushcn(cid,name){                                              \
    int color_id = cid;                                                \
    color_id = color_id%num_colors;                                   \
    nvtxEventAttributes_t eventAttrib = {0};                          \
    eventAttrib.version = NVTX_VERSION;                               \
    eventAttrib.size = NVTX_EVENT_ATTRIB_STRUCT_SIZE;                 \
    eventAttrib.colorType = NVTX_COLOR_ARGB;                          \
    eventAttrib.color = colors[color_id];                             \
    eventAttrib.messageType = NVTX_MESSAGE_TYPE_ASCII;                \
    eventAttrib.message.ascii = name;                                 \
    nvtxRangePushEx(&eventAttrib);                                    \
  }
#define pushc(cid) pushcn(cid,__FUNCTION__)
#define pop nvtxRangePop
#include <nvToolsExt.h>
#include <cudaProfiler.h>
//cuProfilerStart/cuProfilerStop
#include <helper_cuda.h>
#include <helper_functions.h>
#else
#define nvtxRangePush(...)
#define nvtxRangePop(...)
#define push(...)
#define pushf(...)
#define pushcn(...)
#define pop(...)
#endif

// MFEM/fem  *******************************************************************
#include "fem/gridfunc.hpp"
#include "general/communication.hpp"
#include "fem/pfespace.hpp"


// LAGHOS/raja *****************************************************************
#include "rdbg.hpp"
#include "rconfig.hpp"
#include "rmalloc.hpp"
#include "rarray.hpp"
#include "rvector.hpp"
#include "rtypedef.hpp"

// LAGHOS/raja/kernels *********************************************************
#include "kernels/kernels.hpp"

// LAGHOS/raja *****************************************************************
#include "rfespace.hpp"
#include "rbilinearform.hpp"
#include "rgridfunc.hpp"
#include "rbilininteg.hpp"

#endif // LAGHOS_RAJA

