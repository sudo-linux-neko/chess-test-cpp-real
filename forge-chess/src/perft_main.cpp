// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Vaibhav Agarwal

// forge-perft — correctness gate
#include "position.h"
#include "search.h"
#include <iostream>
using namespace forge;
int main() {
  position_init();
  struct T { const char* fen; int depth; uint64_t nodes; };
  T tests[] = {
    {"rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1", 4, 197281},
    {"r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1", 3, 97862},
    {"8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1", 4, 43238},
    {"r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 w kq - 0 1", 3, 9467},
  };
  bool ok = true;
  for (auto& t : tests) {
    Position p; position_from_fen(p, t.fen);
    uint64_t n = perft(p, t.depth);
    std::cout << (n == t.nodes ? "PASS" : "FAIL") << " depth " << t.depth
              << " got " << n << " want " << t.nodes << "\n";
    if (n != t.nodes) { ok = false;
}
  }
  return ok ? 0 : 1;
}
