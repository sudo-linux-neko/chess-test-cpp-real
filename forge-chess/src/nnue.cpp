// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Vaibhav Agarwal

// forge-chess — nnue.cpp
#include "nnue.h"
#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>

namespace forge {

void NNUE::init_random(uint64_t seed) {
  std::mt19937_64 rng(seed);
  // Xavier-ish: w1 ~ U(-0.05, 0.05), w2 ~ U(-0.02,0.02)
  std::uniform_real_distribution<float> d1(-0.05f, 0.05f);
  std::uniform_real_distribution<float> d2(-0.02f, 0.02f);
  for (auto& x : w1) { x = d1(rng);
}
  for (auto& x : b1) { x = 0.0f;
}
  for (auto& x : w2) { x = d2(rng);
}
  b2 = 0.0f;
  trained = false;
}

void nnue_refresh(const NNUE& net, const Position& p, NNUEAcc& acc) {
  // acc = bias + sum features
  for (int i = 0; i < NNUE_HIDDEN; i++) {
    acc.white[i] = net.b1[i];
    acc.black[i] = net.b1[i];
  }
  for (int s = 0; s < 64; s++) {
    Piece pc = p.board_pc[s];
    if (pc == NO_PIECE) { continue;
}
    Color co = p.board_co[s];
    int iw = nnue_idx_white(pc, co, (Square)s);
    int ib = nnue_idx_black(pc, co, (Square)s);
    const float* rw = &net.w1[(size_t)iw * NNUE_HIDDEN];
    const float* rb = &net.w1[(size_t)ib * NNUE_HIDDEN];
    for (int i = 0; i < NNUE_HIDDEN; i++) {
      acc.white[i] += rw[i];
      acc.black[i] += rb[i];
    }
  }
}

void nnue_update(const NNUE& net, const Position& before, Move m,
                 const Undo& u, NNUEAcc& acc) {
  // Correctness-first: castle / ep / promo / capture-with-decay all handled
  // by recomputing affected features incrementally. For simplicity + safety,
  // promo/castle/ep do a targeted refresh of changed squares only.
  // Quiet non-capture non-promo: 1 remove + 1 add per POV (fastest path).
  Square from = move_from(m), to = move_to(m);
  Piece pc = before.board_pc[from];
  if (pc == NO_PIECE) { /* shouldn't happen; stay safe */ return; }
  Color us = before.side;
  bool is_promo = move_is_promo(m);
  bool is_ep = move_is_ep(m);
  bool is_castle = move_is_castle(m);
  Piece cap = u.captured;

  if (is_castle || is_ep || is_promo || cap != NO_PIECE) {
    // Targeted refresh: undo old features, apply new ones.
    // Easiest correct approach: subtract 'before' features for touched squares,
    // then add 'after' features. We need 'after' position: reconstruct minimal.
    // To avoid bugs, do full feature-delta via before/after mailbox diff is
    // overkill here; just do a fast partial refresh over the few touched squares:
    // remove mover-from, captured-sq, rook-from; add mover-to(+promo), rook-to.
    // Implementation: subtract before-state of touched squares, add after-state.
    // Touched squares: from, to, cap_sq (ep/capture), rook from/to (castle).
    Square cap_sq = to;
    if (is_ep) { cap_sq = (us == WHITE) ? (Square)(to - 8) : (Square)(to + 8);
}

    auto sub_feat = [&](Piece ppc, Color cco, Square sq) {
      int iw = nnue_idx_white(ppc, cco, sq);
      int ib = nnue_idx_black(ppc, cco, sq);
      const float* rw = &net.w1[(size_t)iw * NNUE_HIDDEN];
      const float* rb = &net.w1[(size_t)ib * NNUE_HIDDEN];
      for (int i = 0; i < NNUE_HIDDEN; i++) { acc.white[i] -= rw[i]; acc.black[i] -= rb[i]; }
    };
    auto add_feat = [&](Piece ppc, Color cco, Square sq) {
      int iw = nnue_idx_white(ppc, cco, sq);
      int ib = nnue_idx_black(ppc, cco, sq);
      const float* rw = &net.w1[(size_t)iw * NNUE_HIDDEN];
      const float* rb = &net.w1[(size_t)ib * NNUE_HIDDEN];
      for (int i = 0; i < NNUE_HIDDEN; i++) { acc.white[i] += rw[i]; acc.black[i] += rb[i]; }
    };

    // remove mover from origin
    sub_feat(pc, us, from);
    // remove captured
    if (cap != NO_PIECE) {
      Color them = opposite(us);
      sub_feat(cap, them, cap_sq);
    }
    // castle rook remove
    Square rfrom = SQ_NONE, rto = SQ_NONE;
    if (is_castle) {
      if (us == WHITE) {
        if (to == make_sq(6,0)) { rfrom = make_sq(7,0); rto = make_sq(5,0); }
        else { rfrom = make_sq(0,0); rto = make_sq(3,0); }
      } else {
        if (to == make_sq(6,7)) { rfrom = make_sq(7,7); rto = make_sq(5,7); }
        else { rfrom = make_sq(0,7); rto = make_sq(3,7); }
      }
      sub_feat(ROOK, us, rfrom);
    }
    // add mover at dest (or promo)
    Piece placed = is_promo ? move_promo_piece(m) : pc;
    add_feat(placed, us, to);
    if (is_castle) { add_feat(ROOK, us, rto);
}
    return;
  }

  // Fast path: quiet move
  {
    int iw_from = nnue_idx_white(pc, us, from);
    int ib_from = nnue_idx_black(pc, us, from);
    int iw_to = nnue_idx_white(pc, us, to);
    int ib_to = nnue_idx_black(pc, us, to);
    const float* rw_f = &net.w1[(size_t)iw_from * NNUE_HIDDEN];
    const float* rb_f = &net.w1[(size_t)ib_from * NNUE_HIDDEN];
    const float* rw_t = &net.w1[(size_t)iw_to * NNUE_HIDDEN];
    const float* rb_t = &net.w1[(size_t)ib_to * NNUE_HIDDEN];
    for (int i = 0; i < NNUE_HIDDEN; i++) {
      acc.white[i] += rw_t[i] - rw_f[i];
      acc.black[i] += rb_t[i] - rb_f[i];
    }
  }
}

static inline float clipped_relu(float x) {
  if (x < 0) { return 0;
}
  if (x > NNUE_QA_CLIP) { return NNUE_QA_CLIP;
}
  return x;
}

int nnue_evaluate(const NNUE& net, const NNUEAcc& acc, Color stm) {
#if defined(FORGE_HAS_AVX2) || defined(__AVX2__)
  // AVX2 float dot: out = dot(clip(own), w_own) + dot(clip(opp), w_opp) + b2
  const float* own = (stm == WHITE) ? acc.white : acc.black;
  const float* opp = (stm == WHITE) ? acc.black : acc.white;
  const float* wown = &net.w2[0];
  const float* wopp = &net.w2[NNUE_HIDDEN];
  __m256 sum = _mm256_setzero_ps();
  const __m256 vzero = _mm256_setzero_ps();
  const __m256 vclip = _mm256_set1_ps(NNUE_QA_CLIP);
  for (int i = 0; i < NNUE_HIDDEN; i += 8) {
    __m256 a = _mm256_loadu_ps(own + i);
    __m256 b = _mm256_loadu_ps(opp + i);
    a = _mm256_max_ps(a, vzero); a = _mm256_min_ps(a, vclip);
    b = _mm256_max_ps(b, vzero); b = _mm256_min_ps(b, vclip);
    __m256 wa = _mm256_loadu_ps(wown + i);
    __m256 wb = _mm256_loadu_ps(wopp + i);
    // plain AVX2 (no FMA required -> builds on any x86-64 with AVX2, incl. portable flags)
    sum = _mm256_add_ps(sum, _mm256_mul_ps(a, wa));
    sum = _mm256_add_ps(sum, _mm256_mul_ps(b, wb));
  }
  // hsum
  __m128 lo = _mm256_castps256_ps128(sum);
  __m128 hi = _mm256_extractf128_ps(sum, 1);
  __m128 s = _mm_add_ps(lo, hi);
  s = _mm_hadd_ps(s, s); s = _mm_hadd_ps(s, s);
  float out = _mm_cvtss_f32(s) + net.b2;
#else
  float out = net.b2;
  const float* own = (stm == WHITE) ? acc.white : acc.black;
  const float* opp = (stm == WHITE) ? acc.black : acc.white;
  for (int i = 0; i < NNUE_HIDDEN; i++) {
    out += clipped_relu(own[i]) * net.w2[i];
    out += clipped_relu(opp[i]) * net.w2[NNUE_HIDDEN + i];
  }
#endif
  // Scale: network outputs ~[-2,+2]-ish after init; map to centipawns.
  // Trained nets learn this scale automatically; factor 100 keeps sane range.
  int cp = (int)std::lround(out * 100.0f);
  if (cp > 15000) { cp = 15000;
}
  if (cp < -15000) { cp = -15000;
}
  return cp;
}

int nnue_evaluate_full(const NNUE& net, const Position& p) {
  NNUEAcc acc;
  nnue_refresh(net, p, acc);
  return nnue_evaluate(net, acc, p.side);
}

bool nnue_save(const NNUE& net, const std::string& path) {
  FILE* f = fopen(path.c_str(), "wb");
  if (!f) { return false;
}
  const char magic[8] = {'F','O','R','G','E','N','1','\0'};
  fwrite(magic, 1, 8, f);
  uint32_t ver = net.version;
  fwrite(&ver, 4, 1, f);
  uint8_t tr = net.trained ? 1 : 0;
  fwrite(&tr, 1, 1, f);
  uint8_t pad[3] = {0,0,0}; fwrite(pad,1,3,f);
  fwrite(net.b1.data(), 4, NNUE_HIDDEN, f);
  fwrite(net.w1.data(), 4, (size_t)NNUE_INPUT*NNUE_HIDDEN, f);
  fwrite(net.w2.data(), 4, (size_t)NNUE_HIDDEN*2, f);
  fwrite(&net.b2, 4, 1, f);
  fclose(f);
  return true;
}

bool nnue_load(NNUE& net, const std::string& path) {
  FILE* f = fopen(path.c_str(), "rb");
  if (!f) { return false;
}
  char magic[8]; if (fread(magic,1,8,f)!=8){fclose(f);return false;}
  if (memcmp(magic,"FORGEN1",7)!=0){fclose(f);return false;}
  uint32_t ver=0; if(fread(&ver,4,1,f)!=1){fclose(f);return false;}
  uint8_t tr=0; if(fread(&tr,1,1,f)!=1){fclose(f);return false;}
  uint8_t pad[3]; if(fread(pad,1,3,f)!=3){fclose(f);return false;}
  net.alloc();
  if(fread(net.b1.data(),4,NNUE_HIDDEN,f)!=(size_t)NNUE_HIDDEN){fclose(f);return false;}
  if(fread(net.w1.data(),4,(size_t)NNUE_INPUT*NNUE_HIDDEN,f)!=(size_t)NNUE_INPUT*NNUE_HIDDEN){fclose(f);return false;}
  if(fread(net.w2.data(),4,(size_t)NNUE_HIDDEN*2,f)!=(size_t)NNUE_HIDDEN*2){fclose(f);return false;}
  if(fread(&net.b2,4,1,f)!=1){fclose(f);return false;}
  fclose(f);
  net.version=ver; net.trained=(tr!=0);
  return true;
}

} // namespace forge
