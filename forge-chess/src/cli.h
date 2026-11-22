// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Vaibhav Agarwal

// forge-chess — cli.h : pretty colors, progress bar, tables (header-only, any OS)
#pragma once
#include <string>
#include <iostream>
#include <iomanip>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cmath>

namespace forge::cli {

inline bool use_color() {
  static int v = -1;
  if (v < 0) {
    v = 1;
    if (std::getenv("NO_COLOR")) { v = 0;
}
    if (std::getenv("TERM") && std::string(std::getenv("TERM")) == "dumb") { v = 0;
}
#ifdef _WIN32
    // Windows 10+ supports ANSI; keep on (modern terminal). Older cmd.exe ignores codes.
#endif
  }
  return v == 1;
}
inline std::string C(const char* code) { return use_color() ? std::string(code) : std::string(); }
inline std::string reset()   { return C("\033[0m"); }
inline std::string bold()    { return C("\033[1m"); }
inline std::string dim()     { return C("\033[2m"); }
inline std::string red()     { return C("\033[31m"); }
inline std::string green()   { return C("\033[32m"); }
inline std::string yellow()  { return C("\033[33m"); }
inline std::string blue()    { return C("\033[34m"); }
inline std::string magenta() { return C("\033[35m"); }
inline std::string cyan()    { return C("\033[36m"); }
inline std::string white()   { return C("\033[37m"); }

inline void header(const std::string& title) {
  std::cout << bold() << cyan()
            << "\n  ███████╗ ██████╗ ██████╗  ██████╗ ███████╗\n"
            << "  ██╔════╝██╔═══██╗██╔══██╗██╔════╝ ██╔════╝\n"
            << "  █████╗  ██║   ██║██████╔╝██║  ███╗█████╗  \n"
            << "  ██╔══╝  ██║   ██║██╔══██╗██║   ██║██╔══╝  \n"
            << "  ██║     ╚██████╔╝██║  ██║╚██████╔╝███████╗\n"
            << "  ╚═╝      ╚═════╝ ╚═╝  ╚═╝ ╚═════╝ ╚══════╝" << reset()
            << "\n  " << bold() << title << reset()
            << dim() << "  — pure C++17  •  AVX2  •  NNUE  •  UCI" << reset() << "\n\n";
  std::cout.flush();
}

struct Bar {
  int width = 34;
  std::chrono::steady_clock::time_point t0 = std::chrono::steady_clock::now();
  std::string label;
  void draw(double frac, const std::string& extra = "") {
    if (frac < 0) { frac = 0; 
}if (frac > 1) { frac = 1;
}
    int fill = (int)std::lround(frac * width);
    auto now = std::chrono::steady_clock::now();
    double el = std::chrono::duration<double>(now - t0).count();
    double eta = (frac > 1e-9) ? el * (1 - frac) / frac : 0;
    std::cout << "\r  " << cyan() << label << reset() << " [";
    for (int i = 0; i < width; i++) {
      if (i < fill) { std::cout << green() << "█" << reset();
      } else if (i == fill) { std::cout << yellow() << "▌" << reset();
      } else { std::cout << dim() << "░" << reset();
}
    }
    std::cout << "] " << bold() << std::setw(5) << std::fixed << std::setprecision(1)
              << (frac * 100.0) << "%" << reset();
    if (!extra.empty()) { std::cout << "  " << extra;
}
    if (eta > 0.5) {
      int e = (int)eta;
      char buf[32]; snprintf(buf, sizeof(buf), "  ETA %d:%02d", e / 60, e % 60);
      std::cout << dim() << buf << reset();
    }
    std::cout.flush();
  }
  void done(const std::string& msg = "") {
    draw(1.0, msg);
    std::cout << "\n";
    std::cout.flush();
  }
};

inline std::string elo_str(double elo) {
  char b[32]; snprintf(b, sizeof(b), "%+.1f", elo);
  return b;
}

inline void kv(const std::string& k, const std::string& v, const std::string& color = "") {
  std::cout << "  " << dim() << std::left << std::setw(18) << k << reset()
            << (color.empty() ? std::string() : color) << v << reset() << "\n";
}

} // namespace forge::cli
