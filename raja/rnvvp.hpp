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
#ifndef LAGHOS_RAJA_NVVP
#define LAGHOS_RAJA_NVVP

// *****************************************************************************
// https://jonasjacek.github.io/colors
typedef enum {
  Black, Maroon, Green, Olive, Navy, Purple, Teal, Silver,
  Grey, Red, Lime, Yellow, Blue, Fuchsia, Aqua, White
} colors;

// *****************************************************************************
#if defined(__NVCC__) and defined(__NVVP__)

#include <cuda.h>
#include <nvToolsExt.h>
#include <cudaProfiler.h>

static const uint32_t legacy_colors[] = {
0x000000, 0x800000, 0x008000, 0x808000, 0x000080, 0x800080, 0x008080, 0xC0C0C0,
0x808080, 0xFF0000, 0x00FF00, 0xFFFF00, 0x0000FF, 0xFF00FF, 0x00FFFF, 0xFFFFFF,
};
static const int nb_colors = sizeof(legacy_colors)/sizeof(uint32_t);

static char marker[2048];

//__PRETTY_FUNCTION__
#define pop(...) nvtxRangePop()
// *****************************************************************************
#define PUSH2(marker,rgb){                                             \
    const int color_id = rgb%nb_colors;                                \
    nvtxEventAttributes_t eAttrib = {0};                               \
    eAttrib.version = NVTX_VERSION;                                    \
    eAttrib.size = NVTX_EVENT_ATTRIB_STRUCT_SIZE;                      \
    eAttrib.colorType = NVTX_COLOR_ARGB;                               \
    eAttrib.color = legacy_colors[color_id];                           \
    eAttrib.messageType = NVTX_MESSAGE_TYPE_ASCII;                     \
    eAttrib.message.ascii = #marker;                                   \
    nvtxRangePushEx(&eAttrib); }
#define PUSH1(marker) nvtxRangePush(#marker)
#define PUSH0() {                                                     \
    char marker[2048];                                                \
    snprintf(marker,2048,"%s@%s:%d",                                  \
             __PRETTY_FUNCTION__, __FILE__, __LINE__);                \
    nvtxRangePush(marker);                                            \
  }
#define PUSH(a0,a1,a2,a3,a4,a5,a,...) a
// *****************************************************************************
#define LPAREN (
#define COMMA_IF_PARENS(...) ,
#define EXPAND(...) __VA_ARGS__
#define CHOOSE(...) EXPAND(PUSH LPAREN \
      __VA_ARGS__ COMMA_IF_PARENS \
      __VA_ARGS__ COMMA_IF_PARENS __VA_ARGS__ (),  \
      PUSH2, impossible, PUSH2, PUSH1, PUSH0, PUSH1, ))
#define push(...) CHOOSE(__VA_ARGS__)(__VA_ARGS__)

#else // __NVCC__ && _NVVP__

#define pop(...)
#define push(...)
#define cuProfilerStart(...)
#define cuProfilerStop(...)

#endif // defined(__NVCC__) and defined(__NVVP__)

#endif // LAGHOS_RAJA_NVVP

