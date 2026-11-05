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

#pragma once

#include <cstdio>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <unistd.h>
#include <vector>

#include <mfem.hpp>

// Quantities of a dFEM Laghos run
struct LaghosResults
{
   struct Step { int ti; mfem::real_t t, dt; };
   long long ne = 0, h1_dofs = 0, l2_dofs = 0; // global sizes
   mfem::real_t dt_init = 0.0; // initial time step estimate
   int repeats = 0; // repeated time steps
   std::vector<Step> history; // accepted time steps
   int steps = 0;
   mfem::real_t t = 0.0, dt = 0.0;
   mfem::real_t energy_init = 0.0, energy_final = 0.0;
   mfem::real_t x_norm = 0.0, v_norm = 0.0, e_norm = 0.0;
};

// Results of the next run when not null
inline LaghosResults *laghos_results = nullptr;

// laghos.cpp main, built as laghos() for laghos_tests
int laghos(int argc, char *argv[]);

namespace report
{

// True when stdout is a TTY and NO_COLOR is unset
inline bool UseColor()
{
   static const bool color = isatty(fileno(stdout)) && !std::getenv("NO_COLOR");
   return color;
}

// ANSI escape when colors are on, empty otherwise
inline const char *Ansi(const char *code) { return UseColor() ? code : ""; }

inline const char *Reset() { return Ansi("\x1b[0m"); }
inline const char *Bold() { return Ansi("\x1b[1m"); }
inline const char *Dim() { return Ansi("\x1b[2m"); }
inline const char *Cyan() { return Ansi("\x1b[36m"); }
inline const char *Green() { return Ansi("\x1b[32m"); }
inline const char *Red() { return Ansi("\x1b[31m"); }
inline const char *Yellow() { return Ansi("\x1b[33m"); }
inline const char *Magenta() { return Ansi("\x1b[35m"); }

// Green well under the tolerance, yellow from tolerance/near, red over
inline const char *RelColor(double rel, double tol, double near = 1e3)
{
   if (!(rel <= tol)) { return Red(); } // NaN too
   return rel * near < tol ? Green() : Yellow();
}

// Display columns of a UTF-8 string, one per code point
inline int Width(const std::string &s)
{
   int w = 0;
   for (unsigned char c : s) { w += (c & 0xC0) != 0x80; }
   return w;
}

// ▸ name  detail
inline void Header(const std::string &name, const std::string &detail = "")
{
   std::cout << Bold() << Magenta() << "▸ " << name << Reset();
   if (!detail.empty()) { std::cout << "  " << Dim() << detail << Reset(); }
   std::cout << std::endl;
}

// One aligned row: │ tag  label value  label value ... ✓
class Row
{
   std::ostringstream os;
   int prec;
   const char *color = ""; // tag and values color, cyan tag otherwise

   // Value in the row color
   template <typename T> void Value(const T &value)
   {
      if (*color) { os << color << value << Reset(); }
      else { os << value; }
   }

public:
   explicit Row(int precision = 4) : prec(precision)
   {
      os << "  " << Dim() << "│" << Reset();
   }

   // Color of the tag and of the values, labels stay dim
   Row &Color(const char *c)
   {
      color = c;
      return *this;
   }

   // Tag padded to width columns, cyan or in the row color
   Row &Tag(const std::string &tag, int width = 8)
   {
      const int pad = width - Width(tag);
      os << ' ' << (*color ? color : Cyan()) << tag << Reset()
         << std::string(pad > 0 ? pad : 0, ' ');
      return *this;
   }

   // Dim label then a fixed-width scientific value
   Row &Num(const std::string &label, double value)
   {
      std::ostringstream v;
      v << std::scientific << std::setprecision(prec) << std::setw(prec + 7)
        << value;
      os << "  " << Dim() << label << Reset() << ' ';
      Value(v.str());
      return *this;
   }

   // Same, with a given precision
   Row &Num(const std::string &label, double value, int precision)
   {
      std::ostringstream v;
      v << std::scientific << std::setprecision(precision)
        << std::setw(precision + 7) << value;
      os << "  " << Dim() << label << Reset() << ' ';
      Value(v.str());
      return *this;
   }

   // Dim label then a relative error, colored against the tolerance
   Row &Rel(const std::string &label, double rel, double tol)
   {
      os << "  " << Dim() << label << Reset() << ' ' << RelColor(rel, tol)
         << std::scientific << std::setprecision(1) << std::setw(7) << rel
         << Reset();
      return *this;
   }

   // Dim text
   Row &Text(const std::string &text)
   {
      os << "  " << Dim() << text << Reset();
      return *this;
   }

   // Red ✗ and tag padded to width columns
   Row &Fail(const std::string &tag, int width = 8)
   {
      const int pad = width - 2 - Width(tag);
      os << ' ' << Red() << "✗ " << tag << Reset()
         << std::string(pad > 0 ? pad : 0, ' ');
      return *this;
   }

   // Dim label then an integer right-aligned on width columns
   Row &Int(const std::string &label, long long value, int width)
   {
      std::ostringstream v;
      v << std::setw(width) << value;
      os << "  " << Dim() << label << Reset() << ' ';
      Value(v.str());
      return *this;
   }

   // Dim label then an integer
   Row &Int(const std::string &label, long value)
   {
      os << "  " << Dim() << label << Reset() << ' ';
      Value(value);
      return *this;
   }

   // Green ✓ when ok, red ✗ otherwise
   Row &Mark(bool ok)
   {
      os << "  " << Bold() << (ok ? Green() : Red()) << (ok ? "✓" : "✗")
         << Reset();
      return *this;
   }

   ~Row() { std::cout << os.str() << std::endl; }
};

// ▸ name  passed/total passed, bold green when all passed, bold red otherwise
inline void Summary(const std::string &name, int passed, int total)
{
   const bool ok = passed == total;
   std::cout << Bold() << Cyan() << "▸ " << name << Reset() << "  " << Bold()
             << (ok ? Green() : Red()) << passed << '/' << total << " passed "
             << (ok ? "✓" : "✗") << Reset() << std::endl;
}

} // namespace report
