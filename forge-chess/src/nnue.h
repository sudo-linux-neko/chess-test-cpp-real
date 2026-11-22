// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Vaibhav Agarwal

// forge-chess — nnue.h : real NNUE (768 -> 256x2 -> 1), AVX2 + scalar fallback
// Pure C++17, zero dependencies. Trainable from self-play in trainer.cpp.
// Quantization: float weights for training; AVX2 float inference (fast enough
// for 2000+ Elo on i5-1335U). Int16 path can be enabled later without API change.
#pragma once
#include "core.h"
#include "position.h"
#include <vector>
#include <string>
#include <cstdint>

#if defined(FORGE_HAS_AVX2) || defined(__AVX2__)
  #include <immintrin.h>
#endif

namespace forge {

constexpr int NNUE_HIDDEN = 256;
constexpr int NNUE_INPUT  = 768; // 2 colors * 6 pieces * 64 squares
constexpr float NNUE_QA_CLIP = 127.0F / 16.0F; // clipped relu bound ~7.9375

// Feature indexing (perspective-aware, watertight for incremental update)
// white POV: absolute. black POV: colors flipped + square mirrored.
FORGE_FORCE_INLINE int nnue_idx_white(Piece pc, Color co, Square sq) {
  return (((int)co * 6 + (int)pc) * 64) + (int)sq;
}
FORGE_FORCE_INLINE int nnue_idx_black(Piece pc, Color co, Square sq) {
  return ((((int)co ^ 1) * 6 + (int)pc) * 64) + (int)mirror_sq(sq);
}

struct NNUEAcc {
  float white[NNUE_HIDDEN];
  float black[NNUE_HIDDEN];
};

struct NNUE {
  // L1: [768][256] + bias [256]. Stored as [768*256] row-major per feature.
  std::vector<float> w1; // size 768*256
  std::vector<float> b1; // size 256
  // L2: [512] (256 own + 256 opp) + bias
  std::vector<float> w2; // size 512
  float b2 = 0.0F;
  bool trained = false;  // false => engine falls back to classical eval
  uint32_t version = 1;

  NNUE() { alloc(); init_random(0xC0FFEE); }
  void alloc() {
    w1.assign((size_t)NNUE_INPUT * NNUE_HIDDEN, 0.0F);
    b1.assign(NNUE_HIDDEN, 0.0F);
    w2.assign((size_t)NNUE_HIDDEN * 2, 0.0F);
  }
  void init_random(uint64_t seed);
};

// Full refresh from position (used at root + after null moves / complex undo)
void nnue_refresh(const NNUE& net, const Position& p, NNUEAcc& acc);
// Incremental update for a single move (add/remove features). Falls back to
// refresh if move is castle/ep/promo (still correct, just slightly slower).
void nnue_update(const NNUE& net, const Position& before, Move m,
                 const Undo& u, NNUEAcc& acc);

// Inference: returns centipawns from side-to-move POV. AVX2 fast path.
int nnue_evaluate(const NNUE& net, const NNUEAcc& acc, Color stm);
int nnue_evaluate_full(const NNUE& net, const Position& p); // refresh + eval

// Persistence (lossless, versioned, atomic-write safe via tmp+rename in trainer)
bool nnue_save(const NNUE& net, const std::string& path);
bool nnue_load(NNUE& net, const std::string& path);

// Training sample
struct Sample {
  // compact: store FEN + teacher score + result to keep memory low on laptop
  std::string fen;
  float teacher_cp = 0.0F; // teacher eval in cp (from self-play search)
  float result = 0.5F;     // 1/0.5/0 from white POV
};

} // namespace forge
