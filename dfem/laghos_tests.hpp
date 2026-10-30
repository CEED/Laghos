// Copyright (c) 2010-2026, Lawrence Livermore National Security, LLC. Produced
// at the Lawrence Livermore National Laboratory. All Rights reserved. See files
// LICENSE and NOTICE for details. LLNL-CODE-806117.
//
// This file is part of the MFEM library. For more information and source code
// availability visit https://mfem.org.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the BSD-3 license. We welcome feedback and contributions, see file
// CONTRIBUTING.md for details.

#pragma once

// Compact test report lines, after quest/mfem/dft/report.hpp:
//   ▸ name  detail
//     │ tag      label value  label value  ✓

#include <cstdio>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <unistd.h>

namespace report
{

/// True when stdout is a TTY and NO_COLOR is unset
inline bool UseColor()
{
   static const bool color = isatty(fileno(stdout)) && !std::getenv("NO_COLOR");
   return color;
}

/// ANSI escape when colors are on, empty otherwise
inline const char *Ansi(const char *code) { return UseColor() ? code : ""; }

inline const char *Reset() { return Ansi("\x1b[0m"); }
inline const char *Bold() { return Ansi("\x1b[1m"); }
inline const char *Dim() { return Ansi("\x1b[2m"); }
inline const char *Cyan() { return Ansi("\x1b[36m"); }
inline const char *Green() { return Ansi("\x1b[32m"); }
inline const char *Red() { return Ansi("\x1b[31m"); }

/// Display columns of a UTF-8 string, one per code point
inline int Width(const std::string &s)
{
   int w = 0;
   for (unsigned char c : s) { w += (c & 0xC0) != 0x80; }
   return w;
}

/// ▸ name  detail
inline void Header(const std::string &name, const std::string &detail = "")
{
   std::cout << Bold() << Cyan() << "▸ " << name << Reset();
   if (!detail.empty()) { std::cout << "  " << Dim() << detail << Reset(); }
   std::cout << std::endl;
}

/// One aligned row: │ tag  label value  label value ... ✓
class Row
{
   std::ostringstream os;
   int prec;

public:
   explicit Row(int precision = 4) : prec(precision)
   {
      os << "  " << Dim() << "│" << Reset();
   }

   /// Cyan tag padded to width columns
   Row &Tag(const std::string &tag, int width = 8)
   {
      const int pad = width - Width(tag);
      os << ' ' << Cyan() << tag << Reset()
         << std::string(pad > 0 ? pad : 0, ' ');
      return *this;
   }

   /// Dim label then a fixed-width scientific value
   Row &Num(const std::string &label, double value)
   {
      os << "  " << Dim() << label << Reset() << ' '
         << std::scientific << std::setprecision(prec)
         << std::setw(prec + 7) << value;
      return *this;
   }

   /// Same, with a given precision
   Row &Num(const std::string &label, double value, int precision)
   {
      os << "  " << Dim() << label << Reset() << ' '
         << std::scientific << std::setprecision(precision)
         << std::setw(precision + 7) << value;
      return *this;
   }

   /// Dim text
   Row &Text(const std::string &text)
   {
      os << "  " << Dim() << text << Reset();
      return *this;
   }

   /// Red ✗ and tag padded to width columns
   Row &Fail(const std::string &tag, int width = 8)
   {
      const int pad = width - 2 - Width(tag);
      os << ' ' << Red() << "✗ " << tag << Reset()
         << std::string(pad > 0 ? pad : 0, ' ');
      return *this;
   }

   /// Dim label then an integer right-aligned on width columns
   Row &Int(const std::string &label, long long value, int width)
   {
      os << "  " << Dim() << label << Reset() << ' ' << std::setw(width)
         << value;
      return *this;
   }

   /// Dim label then an integer
   Row &Int(const std::string &label, long value)
   {
      os << "  " << Dim() << label << Reset() << ' ' << value;
      return *this;
   }

   /// Green ✓ when ok, red ✗ otherwise
   Row &Mark(bool ok)
   {
      os << "  " << (ok ? Green() : Red()) << (ok ? "✓" : "✗") << Reset();
      return *this;
   }

   ~Row() { std::cout << os.str() << std::endl; }
};

} // namespace report
