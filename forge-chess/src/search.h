// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Vaibhav Agarwal

// forge-chess — search.h : TT, eval, negamax search, time mgmt, SMP
#pragma once
#include "core.h"
#include "position.h"
#include "nnue.h"
#include <atomic>
#include <vector>
#include <cstdint>
#include <chrono>
#include <mutex>

namespace forge {

struct Limits {
  int depth = 64;          // max depth if no time limit
  int64_t movetime_ms = 0; // 0 = no fixed time
  int64_t wtime = 0, btime = 0, winc = 0, binc = 0;
  int64_t nodes = 0;       // 0 = unlimited
  bool infinite = false;
  int movestogo = 0;
};

struct SearchInfo {
  Move best = MOVE_NONE;
  Move ponder = MOVE_NONE;
  Score score = 0;
  int depth = 0;
  uint64_t nodes = 0;
  int64_t time_ms = 0;
  std::vector<Move> pv;
};

// ---------- Transposition table (lockless, shared across threads) ----------
enum TTFlag : uint8_t { TT_NONE=0, TT_EXACT=1, TT_LOWER=2, TT_UPPER=3 };

struct TTEntry {
  uint64_t key = 0;
  Move move = MOVE_NONE;
  int16_t score = 0;
  int8_t depth = -1;
  uint8_t flag = TT_NONE;
  uint8_t age = 0;
};

struct TT {
  std::vector<TTEntry> table;
  uint64_t mask = 0;
  std::atomic<uint8_t> age{0};

  void resize(size_t mb);
  void clear();
  [[nodiscard]] size_t size() const { return table.size(); }
  void new_search() { age.fetch_add(1, std::memory_order_relaxed); }

  // Single-thread correctness is what matters; SMP shares via benign races
  // (key verified, so torn reads only cost Elo, never correctness).
  bool probe(U64 hash, TTEntry& out) const;
  void store(U64 hash, Move m, Score s, int depth, TTFlag f);
  [[nodiscard]] int hashfull() const;
};

// ---------- Classical tapered eval (always available) ----------
int evaluate_classical(const Position& p);
// Blended eval: uses NNUE when trained, else classical.
int evaluate(const Position& p, const NNUE& net, NNUEAcc* acc_or_null = nullptr);

// ---------- Searcher ----------
struct Searcher {
  TT* tt = nullptr;
  const NNUE* net = nullptr;

  // ordering tables
  int history[2][64][64]{};
  Move killers[128][2]{};
  Move counter[2][6][64]{}; // [color][piece][to]

  std::atomic<bool>* stop = nullptr;
  std::atomic<uint64_t>* nodes_shared = nullptr;
  uint64_t nodes = 0;
  int seldepth = 0;
  std::chrono::steady_clock::time_point t0;
  int64_t hard_ms = INT64_MAX, soft_ms = INT64_MAX;
  int64_t node_limit = 0;
  int max_depth = 64;
  uint8_t gen = 0;

  // per-ply stacks
  struct Stack { Move move=MOVE_NONE; Move excluded=MOVE_NONE; int stat_score=0; };
  Stack stack[128]{};
  NNUEAcc acc_stack[128]{};

  Searcher() = default;

  SearchInfo search(Position root, const Limits& lim,
                   std::atomic<bool>* stop_flag,
                   std::atomic<uint64_t>* nodes_counter = nullptr);

  Score qsearch(Position& p, Score alpha, Score beta, int ply, NNUEAcc& acc);
  Score negamax(Position& p, Score alpha, Score beta, int depth, int ply,
                bool cut_node, NNUEAcc& acc);

  bool should_stop();
  [[nodiscard]] int64_t elapsed_ms() const;
};

// SMP driver: Lazy SMP lite (helpers warm shared TT). Returns main-thread result.
SearchInfo search_smp(Position root, const Limits& lim, TT& tt, const NNUE& net,
                      int threads, std::atomic<bool>& stop_flag);

// Time allocation helper
int64_t alloc_movetime(const Position& p, const Limits& lim);

// Perft (correctness check)
uint64_t perft(Position& p, int depth);

} // namespace forge
