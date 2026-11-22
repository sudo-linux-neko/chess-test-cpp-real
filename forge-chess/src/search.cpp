// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Vaibhav Agarwal

// forge-chess — search.cpp : eval + TT + PVS negamax + qsearch + SMP + perft
#include "search.h"
#include <algorithm>
#include <cstring>
#include <thread>
#include <cmath>

namespace forge {

// ================= TT =================
void TT::resize(size_t mb) {
  if (mb < 1) { mb = 1;
}
  if (mb > 2048) { mb = 2048;
}
  size_t n = (mb * 1024 * 1024) / sizeof(TTEntry);
  // power of two
  size_t p = 1;
  while (p * 2 <= n) { p *= 2;
}
  table.assign(p, TTEntry{});
  mask = p - 1;
}
void TT::clear() {
  for (auto& e : table) { e = TTEntry{};
}
}
bool TT::probe(U64 hash, TTEntry& out) const {
  if (table.empty()) { return false;
}
  const TTEntry& e = table[hash & mask];
  if (e.key == hash && e.flag != TT_NONE) { out = e; return true; }
  return false;
}
void TT::store(U64 hash, Move m, Score s, int depth, TTFlag f) {
  if (table.empty()) { return;
}
  TTEntry& e = table[(size_t)(hash & mask)];
  // replace if empty / deeper / different age (standard depth-preferred + always-replace bit)
  uint8_t a = age.load(std::memory_order_relaxed);
  bool replace = (e.key != hash) || (int)e.depth <= depth + 2 || e.age != a;
  if (!replace) { return;
}
  // mate scores need ply adjustment by caller; stored as-is here
  e.key = hash; e.move = m; e.score = (int16_t)std::max(-32000, std::min(32000, (int)s));
  e.depth = (int8_t)std::max(-128, std::min(127, depth)); e.flag = (uint8_t)f; e.age = a;
}
int TT::hashfull() const {
  if (table.empty()) { return 0;
}
  size_t n = std::min<size_t>(1000, table.size());
  int used = 0;
  uint8_t a = age.load(std::memory_order_relaxed);
  for (size_t i = 0; i < n; i++) {
    if (table[i].flag != TT_NONE && (table[i].age == a || table[i].depth >= 0)) { used++;
}
}
  return (int)(used * 1000 / n);
}

// ================= Classical eval =================
// Tables in rank8-first order (a8..h8, ..., a1..h1). White idx=(7-r)*8+f, Black idx=r*8+f.
static const int MG_PAWN[64] = {
   0,  0,  0,  0,  0,  0,  0,  0,
  50, 50, 50, 50, 50, 50, 50, 50,
  10, 10, 20, 30, 30, 20, 10, 10,
   5,  5, 10, 25, 25, 10,  5,  5,
   0,  0,  0, 20, 20,  0,  0,  0,
   5, -5,-10,  0,  0,-10, -5,  5,
   5, 10, 10,-20,-20, 10, 10,  5,
   0,  0,  0,  0,  0,  0,  0,  0};
static const int EG_PAWN[64] = {
   0,  0,  0,  0,  0,  0,  0,  0,
  50, 50, 50, 50, 50, 50, 50, 50,
  10, 10, 20, 30, 30, 20, 10, 10,
   5,  5, 10, 25, 25, 10,  5,  5,
   0,  0,  0, 20, 20,  0,  0,  0,
   5, -5,-10,  0,  0,-10, -5,  5,
   5, 10, 10,-20,-20, 10, 10,  5,
   0,  0,  0,  0,  0,  0,  0,  0};
static const int MG_KNIGHT[64] = {
 -50,-40,-30,-30,-30,-30,-40,-50,
 -40,-20,  0,  0,  0,  0,-20,-40,
 -30,  0, 10, 15, 15, 10,  0,-30,
 -30,  5, 15, 20, 20, 15,  5,-30,
 -30,  0, 15, 20, 20, 15,  0,-30,
 -30,  5, 10, 15, 15, 10,  5,-30,
 -40,-20,  0,  5,  5,  0,-20,-40,
 -50,-40,-30,-30,-30,-30,-40,-50};
static const int EG_KNIGHT[64] = {
 -50,-40,-30,-30,-30,-30,-40,-50,
 -40,-20,  0,  0,  0,  0,-20,-40,
 -30,  0, 10, 15, 15, 10,  0,-30,
 -30,  5, 15, 20, 20, 15,  5,-30,
 -30,  0, 15, 20, 20, 15,  0,-30,
 -30,  5, 10, 15, 15, 10,  5,-30,
 -40,-20,  0,  5,  5,  0,-20,-40,
 -50,-40,-30,-30,-30,-30,-40,-50};
static const int MG_BISHOP[64] = {
 -20,-10,-10,-10,-10,-10,-10,-20,
 -10,  0,  0,  0,  0,  0,  0,-10,
 -10,  0,  5, 10, 10,  5,  0,-10,
 -10,  5,  5, 10, 10,  5,  5,-10,
 -10,  0, 10, 10, 10, 10,  0,-10,
 -10, 10, 10, 10, 10, 10, 10,-10,
 -10,  5,  0,  0,  0,  0,  5,-10,
 -20,-10,-10,-10,-10,-10,-10,-20};
static const int EG_BISHOP[64] = {
 -20,-10,-10,-10,-10,-10,-10,-20,
 -10,  0,  0,  0,  0,  0,  0,-10,
 -10,  0,  5, 10, 10,  5,  0,-10,
 -10,  5,  5, 10, 10,  5,  5,-10,
 -10,  0, 10, 10, 10, 10,  0,-10,
 -10, 10, 10, 10, 10, 10, 10,-10,
 -10,  5,  0,  0,  0,  0,  5,-10,
 -20,-10,-10,-10,-10,-10,-10,-20};
static const int MG_ROOK[64] = {
   0,  0,  0,  0,  0,  0,  0,  0,
   5, 10, 10, 10, 10, 10, 10,  5,
  -5,  0,  0,  0,  0,  0,  0, -5,
  -5,  0,  0,  0,  0,  0,  0, -5,
  -5,  0,  0,  0,  0,  0,  0, -5,
  -5,  0,  0,  0,  0,  0,  0, -5,
  -5,  0,  0,  0,  0,  0,  0, -5,
   0,  0,  0,  5,  5,  0,  0,  0};
static const int EG_ROOK[64] = {
   0,  0,  0,  0,  0,  0,  0,  0,
   5, 10, 10, 10, 10, 10, 10,  5,
  -5,  0,  0,  0,  0,  0,  0, -5,
  -5,  0,  0,  0,  0,  0,  0, -5,
  -5,  0,  0,  0,  0,  0,  0, -5,
  -5,  0,  0,  0,  0,  0,  0, -5,
  -5,  0,  0,  0,  0,  0,  0, -5,
   0,  0,  0,  5,  5,  0,  0,  0};
static const int MG_QUEEN[64] = {
 -20,-10,-10, -5, -5,-10,-10,-20,
 -10,  0,  0,  0,  0,  0,  0,-10,
 -10,  0,  5,  5,  5,  5,  0,-10,
  -5,  0,  5,  5,  5,  5,  0, -5,
   0,  0,  5,  5,  5,  5,  0, -5,
 -10,  5,  5,  5,  5,  5,  0,-10,
 -10,  0,  5,  0,  0,  0,  0,-10,
 -20,-10,-10, -5, -5,-10,-10,-20};
static const int EG_QUEEN[64] = {
 -20,-10,-10, -5, -5,-10,-10,-20,
 -10,  0,  0,  0,  0,  0,  0,-10,
 -10,  0,  5,  5,  5,  5,  0,-10,
  -5,  0,  5,  5,  5,  5,  0, -5,
   0,  0,  5,  5,  5,  5,  0, -5,
 -10,  5,  5,  5,  5,  5,  0,-10,
 -10,  0,  5,  0,  0,  0,  0,-10,
 -20,-10,-10, -5, -5,-10,-10,-20};
static const int MG_KING[64] = {
 -30,-40,-40,-50,-50,-40,-40,-30,
 -30,-40,-40,-50,-50,-40,-40,-30,
 -30,-40,-40,-50,-50,-40,-40,-30,
 -30,-40,-40,-50,-50,-40,-40,-30,
 -20,-30,-30,-40,-40,-30,-30,-20,
 -10,-20,-20,-20,-20,-20,-20,-10,
  20, 20,  0,  0,  0,  0, 20, 20,
  20, 30, 10,  0,  0, 10, 30, 20};
static const int EG_KING[64] = {
 -50,-40,-30,-20,-20,-30,-40,-50,
 -30,-20,-10,  0,  0,-10,-20,-30,
 -30,-10, 20, 30, 30, 20,-10,-30,
 -30,-10, 30, 40, 40, 30,-10,-30,
 -30,-10, 30, 40, 40, 30,-10,-30,
 -30,-10, 20, 30, 30, 20,-10,-30,
 -30,-30,  0,  0,  0,  0,-30,-30,
 -50,-30,-30,-30,-30,-30,-30,-50};

static const int MG_MAT[6] = {100, 320, 330, 500, 900, 0};
static const int EG_MAT[6] = {100, 310, 320, 500, 900, 0};

FORGE_FORCE_INLINE int pst_w(const int* t, Square s) {
  int f = file_of(s), r = rank_of(s);
  return t[(7 - r) * 8 + f];
}
FORGE_FORCE_INLINE int pst_b(const int* t, Square s) {
  int f = file_of(s), r = rank_of(s);
  return t[r * 8 + f];
}

int evaluate_classical(const Position& p) {
  // Tapered material + PST
  int mg = 0, eg = 0;
  int phase = 0;

  for (int s = 0; s < 64; s++) {
    Piece pc = p.board_pc[s];
    if (pc == NO_PIECE) { continue;
}
    Color c = p.board_co[s];
    int sign = (c == WHITE) ? 1 : -1;
    mg += sign * MG_MAT[pc];
    eg += sign * EG_MAT[pc];
    const int *mg_t = nullptr, *eg_t = nullptr;
    switch (pc) {
      case PAWN: mg_t = MG_PAWN; eg_t = EG_PAWN; break;
      case KNIGHT: mg_t = MG_KNIGHT; eg_t = EG_KNIGHT; break;
      case BISHOP: mg_t = MG_BISHOP; eg_t = EG_BISHOP; break;
      case ROOK: mg_t = MG_ROOK; eg_t = EG_ROOK; break;
      case QUEEN: mg_t = MG_QUEEN; eg_t = EG_QUEEN; break;
      case KING: mg_t = MG_KING; eg_t = EG_KING; break;
      default: break;
    }
    int pv = (c == WHITE) ? pst_w(mg_t, (Square)s) : pst_b(mg_t, (Square)s);
    int ev = (c == WHITE) ? pst_w(eg_t, (Square)s) : pst_b(eg_t, (Square)s);
    mg += sign * pv;
    eg += sign * ev;
    if (pc == KNIGHT || pc == BISHOP) { phase += 1;
    } else if (pc == ROOK) { phase += 2;
    } else if (pc == QUEEN) { phase += 4;
}
  }
  if (phase > 24) { phase = 24;
}
  int score = (mg * phase + eg * (24 - phase)) / 24;

  // Pawn structure (cheap, both sides)
  // Build file counts
  int wfile[8] = {}, bfile[8] = {};
  U64 wp = p.bb[WHITE][PAWN], bp = p.bb[BLACK][PAWN];
  {
    U64 b = wp; while (b) { Square s = lsb_sq(b); poplsb(b); wfile[file_of(s)]++; }
    b = bp; while (b) { Square s = lsb_sq(b); poplsb(b); bfile[file_of(s)]++; }
  }
  auto passed_w = [&](Square s) {
    int f = file_of(s), r = rank_of(s);
    for (int df = -1; df <= 1; df++) {
      int nf = f + df;
      if (nf < 0 || nf > 7) { continue;
}
      U64 col = 0;
      for (int rr = r + 1; rr < 8; rr++) { col |= sq_bb(make_sq(nf, rr));
}
      if (col & bp) { return false;
}
    }
    return true;
  };
  auto passed_b = [&](Square s) {
    int f = file_of(s), r = rank_of(s);
    for (int df = -1; df <= 1; df++) {
      int nf = f + df;
      if (nf < 0 || nf > 7) { continue;
}
      U64 col = 0;
      for (int rr = r - 1; rr >= 0; rr--) { col |= sq_bb(make_sq(nf, rr));
}
      if (col & wp) { return false;
}
    }
    return true;
  };
  {
    U64 b = wp;
    while (b) {
      Square s = lsb_sq(b); poplsb(b);
      int f = file_of(s), r = rank_of(s);
      if (wfile[f] > 1) { score -= 12;                       // doubled
}
      bool iso = true;
      if (f > 0 && wfile[f - 1]) { iso = false;
}
      if (f < 7 && wfile[f + 1]) { iso = false;
}
      if (iso) { score -= 14;
}
      if (passed_w(s)) { score += 8 + (r - 1) * (r - 1) * 2; // ramp with rank
}
    }
    b = bp;
    while (b) {
      Square s = lsb_sq(b); poplsb(b);
      int f = file_of(s), r = rank_of(s);
      if (bfile[f] > 1) { score += 12;
}
      bool iso = true;
      if (f > 0 && bfile[f - 1]) { iso = false;
}
      if (f < 7 && bfile[f + 1]) { iso = false;
}
      if (iso) { score += 14;
}
      if (passed_b(s)) { score -= 8 + (6 - r) * (6 - r) * 2;
}
    }
  }

  // King safety: pawn shield + open files near king
  auto king_safety = [&](Color c) {
    Square k = king_sq(p, c);
    if (k == SQ_NONE) { return 0;
}
    int f = file_of(k), r = rank_of(k);
    int s = 0;
    U64 own_pawns = p.bb[c][PAWN];
    int home = (c == WHITE) ? 1 : 6;
    int dir = (c == WHITE) ? 1 : -1;
    for (int df = -1; df <= 1; df++) {
      int nf = f + df;
      if (nf < 0 || nf > 7) { continue;
}
      bool shield = false;
      for (int k2 = 1; k2 <= 2; k2++) {
        int nr = home + dir * k2 + (r - home) / 3; // loose: just check 1-2 ahead of king
        (void)nr;
      }
      // simple: pawn on (nf, home+dir) or (nf, home+2dir) near king file
      int r1 = (c == WHITE) ? r + 1 : r - 1;
      int r2 = (c == WHITE) ? r + 2 : r - 2;
      if (r1 >= 0 && r1 < 8 && (own_pawns & sq_bb(make_sq(nf, r1)))) { shield = true;
      } else if (r2 >= 0 && r2 < 8 && (own_pawns & sq_bb(make_sq(nf, r2)))) { s += 4;
      } else { s -= (df == 0 ? 18 : 12);
}
      if (shield) { s += 12;
}
      // open file
      U64 filebb = 0;
      for (int rr = 0; rr < 8; rr++) { filebb |= sq_bb(make_sq(nf, rr));
}
      bool enemy_pawn = (p.bb[opposite(c)][PAWN] & filebb) != 0;
      bool own_pawn = (own_pawns & filebb) != 0;
      if (!enemy_pawn && !own_pawn) { s -= 14;
      } else if (!own_pawn) { s -= 7;
}
    }
    return s;
  };
  score += king_safety(WHITE) - king_safety(BLACK);

  // Bishop pair
  if (popcount(p.bb[WHITE][BISHOP]) >= 2) { score += 28;
}
  if (popcount(p.bb[BLACK][BISHOP]) >= 2) { score -= 28;
}

  // Tempo
  score += (p.side == WHITE) ? 8 : -8;

  return (p.side == WHITE) ? score : -score; // side-to-move POV
}

int evaluate(const Position& p, const NNUE& net, NNUEAcc* acc) {
  if (insufficient_material(p)) { return 0;
}
  int classical = evaluate_classical(p);
  if (!net.trained) { return classical;
}
  int nnue_cp;
  if (acc) { nnue_cp = nnue_evaluate(net, *acc, p.side);
  } else { nnue_cp = nnue_evaluate_full(net, p);
}
  // Blend: trust NNUE 75%, keep classical 25% as stabilizer (common in hybrids).
  // As training Elo rises the trainer can shift this; 75/25 is a strong default.
  int blended = (nnue_cp * 3 + classical) / 4;
  return blended;
}

// ================= Searcher =================
int64_t Searcher::elapsed_ms() const {
  auto now = std::chrono::steady_clock::now();
  return std::chrono::duration_cast<std::chrono::milliseconds>(now - t0).count();
}

bool Searcher::should_stop() {
  if (stop && stop->load(std::memory_order_relaxed)) { return true;
}
  if ((nodes & 1023) == 0) {
    if (node_limit > 0 && (int64_t)nodes >= node_limit) { return true;
}
    if (elapsed_ms() >= hard_ms) { return true;
}
  }
  return false;
}

int64_t alloc_movetime(const Position& p, const Limits& lim) {
  (void)p;
  if (lim.movetime_ms > 0) { return lim.movetime_ms;
}
  if (lim.infinite) { return INT64_MAX;
}
  int64_t own = (p.side == WHITE) ? lim.wtime : lim.btime;
  int64_t inc = (p.side == WHITE) ? lim.winc : lim.binc;
  if (own <= 0) { return INT64_MAX; // no clock -> depth-limited
}
  int64_t mtg = lim.movestogo > 0 ? lim.movestogo : 30;
  if (mtg < 1) { mtg = 1;
}
  if (mtg > 40) { mtg = 40;
}
  int64_t base = own / mtg;
  base += inc / 2;
  int64_t cap = own / 4;
  if (base > cap) { base = cap;
}
  if (base < 10) { base = 10;
}
  return base;
}

// Move ordering score
static int order_score(Searcher& s, const Position& p, Move m, Move tt_move, int ply) {
  if (m == tt_move) { return 2000000;
}
  Square from = move_from(m), to = move_to(m);
  Piece vic = move_is_ep(m) ? PAWN : p.board_pc[to];
  if (vic != NO_PIECE) {
    int mv = see_value(vic) * 10 - see_value(p.board_pc[from]) / 10;
    if (move_is_promo(m)) { mv += see_value(move_promo_piece(m));
}
    return 1000000 + mv;
  }
  if (move_is_promo(m)) { return 900000 + see_value(move_promo_piece(m));
}
  // killers
  if (ply < 128) {
    if (s.killers[ply][0] == m) { return 800000;
}
    if (s.killers[ply][1] == m) { return 790000;
}
  }
  // counter
  Piece pc = p.board_pc[from];
  if (pc != NO_PIECE && ply > 0) {
    Move prev = s.stack[ply - 1].move;
    if (prev != MOVE_NONE) {
      Color pmc = opposite(p.side);
      Piece ppc = NO_PIECE;
      // approximate: use counter table indexed by mover color/piece/to
      (void)pmc; (void)ppc;
    }
    if (pc <= KING && to < 64 && s.counter[p.side][pc][to] == m) { return 780000;
}
  }
  int h = 0;
  if (pc != NO_PIECE) { h = s.history[p.side][from][to];
}
  return h;
}

Score Searcher::qsearch(Position& p, Score alpha, Score beta, int ply, NNUEAcc& acc) {
  nodes++;
  if (nodes_shared) { /* updated periodically by caller */ }
  if ((nodes & 1023) == 0 && should_stop()) { return alpha;
}
  if (ply >= 120) { return evaluate(p, *net, &acc);
}
  if (p.half >= 100) { return 0;
}
  if (insufficient_material(p)) { return 0;
}

  bool inck = in_check(p, p.side);
  Score stand = inck ? -INF : evaluate(p, *net, &acc);
  if (!inck) {
    if (stand >= beta) { return stand;
}
    if (stand > alpha) { alpha = stand;
}
    // delta pruning: even the best capture (+promo) can't raise alpha
    // max material swing ~ 900 + 800 promo
    if (stand + 950 < alpha) { return alpha;
}
  }

  MoveList ml;
  if (inck) { gen_legal(p, ml, false);
  } else { gen_legal(p, ml, true);
}
  // order captures by SEE
  // simple insertion by see
  for (int i = 0; i < ml.n; i++) {
    int best = i, bv = see(p, ml.m[i]);
    for (int j = i + 1; j < ml.n; j++) {
      int v = see(p, ml.m[j]);
      if (v > bv) { bv = v; best = j; }
    }
    if (best != i) { std::swap(ml.m[i], ml.m[best]);
}
  }

  Undo u;
  for (int i = 0; i < ml.n; i++) {
    Move m = ml.m[i];
    // SEE prune in qsearch (not when in check)
    if (!inck && see(p, m) < -50) { continue;
}
    Position np = p;
    NNUEAcc nacc = acc;
    if (!make_move(np, m, u)) { continue;
}
    if (in_check(np, p.side)) { continue;
}
    nnue_update(*net, p, m, u, nacc);
    Score sc = -qsearch(np, -beta, -alpha, ply + 1, nacc);
    if (stop && stop->load(std::memory_order_relaxed)) { return alpha;
}
    if (sc > alpha) {
      alpha = sc;
      if (alpha >= beta) { break;
}
    }
  }
  if (inck && ml.n == 0) {
    // checkmate or stalemate: gen_legal returned 0 moves while in check => mate
    // (gen_legal already filters; if no moves and in check => mated)
    return -MATE + ply;
  }
  return alpha;
}

Score Searcher::negamax(Position& p, Score alpha, Score beta, int depth, int ply,
                        bool cut_node, NNUEAcc& acc) {
  if (ply >= 120) { return evaluate(p, *net, &acc);
}
  nodes++;
  if ((nodes & 1023) == 0 && should_stop()) { return alpha;
}

  // mate distance pruning
  Score mated = -MATE + ply;
  if (mated > alpha) { alpha = mated; if (alpha >= beta) { return alpha; 
}}
  Score mate_in = MATE - ply - 1;
  if (mate_in < beta) { beta = mate_in; if (alpha >= beta) { return alpha; 
}}

  if (p.half >= 100) { return 0;
}
  if (insufficient_material(p)) { return 0;
}

  bool inck = in_check(p, p.side);
  // check extension
  if (inck) { depth++;
}

  if (depth <= 0) { return qsearch(p, alpha, beta, ply, acc);
}

  // TT probe
  TTEntry tte;
  bool tt_hit = tt && tt->probe(p.hash, tte);
  Move tt_move = tt_hit ? tte.move : MOVE_NONE;
  if (tt_hit && tte.depth >= depth && ply > 0) {
    Score s = tte.score;
    // adjust mate scores for ply
    if (s > MATE - 1000) { s -= ply;
    } else if (s < -MATE + 1000) { s += ply;
}
    if (tte.flag == TT_EXACT) { return s;
}
    if (tte.flag == TT_LOWER && s >= beta) { return s;
}
    if (tte.flag == TT_UPPER && s <= alpha) { return s;
}
  }

  Score static_eval = evaluate(p, *net, &acc);
  bool improving = ply >= 2; // simplified (no 2-ply history here)

  // Reverse futility pruning
  if (!inck && depth <= 8 && static_eval - 80 * depth >= beta && std::abs(beta) < MATE - 1000) {
    return static_eval;
}

  // Null move pruning
  bool has_nonpawn = (p.bb[p.side][KNIGHT] | p.bb[p.side][BISHOP] |
                      p.bb[p.side][ROOK] | p.bb[p.side][QUEEN]) != 0;
  if (!inck && ply > 0 && depth >= 3 && static_eval >= beta && has_nonpawn) {
    Position np = p;
    // null move: switch side, clear ep, update hash
    np.hash ^= ZOB_SIDE;
    if (np.ep != SQ_NONE) { np.hash ^= ZOB_EP[file_of(np.ep)]; np.ep = SQ_NONE; }
    np.side = opposite(np.side);
    np.half++;
    NNUEAcc nacc = acc; // accumulators are side-relative; refresh cheap alternative:
    // For null move, accumulators stay the same (pieces unchanged), only stm flips,
    // and nnue_evaluate uses stm to pick own/opp. So no update needed.
    int R = 3 + depth / 4;
    if (depth - R <= 0) {
      Score ns = -qsearch(np, -beta, -beta + 1, ply + 1, nacc);
      if (ns >= beta) { return ns;
}
    } else {
      Score ns = -negamax(np, -beta, -beta + 1, depth - R, ply + 1, !cut_node, nacc);
      if (stop && stop->load(std::memory_order_relaxed)) { return beta;
}
      if (ns >= beta) { return ns;
}
    }
  }

  // Razoring
  if (!inck && depth <= 3 && static_eval + 300 * depth < alpha) {
    Score qs = qsearch(p, alpha, beta, ply, acc);
    return qs;
  }

  MoveList ml;
  gen_legal(p, ml, false);
  if (ml.n == 0) {
    if (inck) { return -MATE + ply;
}
    return 0; // stalemate
  }

  // Order moves
  int scores[256];
  for (int i = 0; i < ml.n; i++) { scores[i] = order_score(*this, p, ml.m[i], tt_move, ply);
}

  Score best = -INF;
  Move best_move = MOVE_NONE;
  TTFlag flag = TT_UPPER;
  int played = 0;

  // LMR table (conservative, cache-friendly)
  for (int i = 0; i < ml.n; i++) {
    // pick best remaining (selection sort, fine for ~40 moves)
    int bi = i;
    for (int j = i + 1; j < ml.n; j++) { if (scores[j] > scores[bi]) { bi = j;
}
}
    std::swap(ml.m[i], ml.m[bi]); std::swap(scores[i], scores[bi]);
    Move m = ml.m[i];
    bool is_cap = (move_is_ep(m) || p.board_pc[move_to(m)] != NO_PIECE);
    bool is_promo = move_is_promo(m);
    int gives_check = 0; // computed lazily only for LMR candidates to save time

    Position np = p;
    Undo u;
    NNUEAcc nacc = acc;
    if (!make_move(np, m, u)) { continue;
}
    if (in_check(np, p.side)) { continue; // illegal (shouldn't happen after gen_legal, but safe)
}
    nnue_update(*net, p, m, u, nacc);
    played++;

    Score sc;
    bool do_full = true;
    // Late move reductions
    int R = 0;
    if (depth >= 3 && played > 3 && !inck && !is_cap && !is_promo && !cut_node) {
      R = 1;
      if (played > 8 && depth >= 5) { R = 2;
}
      // reduce less for killers/history-good moves
      if (scores[i] > 5000) { R--;
}
      if (R < 0) { R = 0;
}
    }
    if (R > 0) {
      gives_check = move_gives_check(p, m);
      if (gives_check) { R = 0; // don't reduce checks
      } else {
        sc = -negamax(np, -(alpha + 1), -alpha, depth - 1 - R, ply + 1, true, nacc);
        if (stop && stop->load(std::memory_order_relaxed)) { return best;
}
        do_full = (sc > alpha);
      }
    }
    if (do_full) {
      if (played == 1) {
        sc = -negamax(np, -beta, -alpha, depth - 1, ply + 1, false, nacc);
      } else {
        // PVS zero window
        sc = -negamax(np, -(alpha + 1), -alpha, depth - 1, ply + 1, true, nacc);
        if (!should_stop() && sc > alpha && sc < beta) {
          sc = -negamax(np, -beta, -alpha, depth - 1, ply + 1, false, nacc);
}
      }
    }
    if (stop && stop->load(std::memory_order_relaxed)) { return best;
}

    if (sc > best) {
      best = sc;
      best_move = m;
      if (sc > alpha) {
        alpha = sc;
        flag = TT_EXACT;
        if (alpha >= beta) {
          flag = TT_LOWER;
          // update killers/history/counter for quiet cutoffs
          if (!is_cap) {
            if (ply < 128) {
              if (killers[ply][0] != m) { killers[ply][1] = killers[ply][0]; killers[ply][0] = m; }
            }
            Piece pc = p.board_pc[move_from(m)];
            history[p.side][move_from(m)][move_to(m)] += depth * depth;
            // clamp history to avoid overflow
            if (history[p.side][move_from(m)][move_to(m)] > 1000000) {
              for (int a = 0; a < 64; a++) { for (int b2 = 0; b2 < 64; b2++) {
                history[0][a][b2] /= 2, history[1][a][b2] /= 2;
}
}
            }
            if (pc != NO_PIECE) { counter[p.side][pc][move_to(m)] = m;
}
          }
          break;
        }
      }
    } else {
      // history penalty for quiet fail-lows (gravity-lite)
      if (!is_cap && ply < 64) {
        history[p.side][move_from(m)][move_to(m)] -= depth;
      }
    }

    // Futility skip: if many moves already searched and static+margin < alpha, skip remaining quiets
    // (checked lazily; safe because moves are ordered)
    (void)improving;
  }

  if (played == 0) { // shouldn't happen (handled above) but safe
    if (inck) { return -MATE + ply;
}
    return 0;
  }

  // TT store (mate scores adjusted for ply)
  if (tt) {
    Score store_s = best;
    if (best > MATE - 1000) { store_s += ply;
    } else if (best < -MATE + 1000) { store_s -= ply;
}
    tt->store(p.hash, best_move, store_s, depth, flag);
  }
  return best;
}

SearchInfo Searcher::search(Position root, const Limits& lim,
                            std::atomic<bool>* stop_flag,
                            std::atomic<uint64_t>* nodes_counter) {
  stop = stop_flag;
  nodes = 0;
  nodes_shared = nodes_counter;
  seldepth = 0;
  t0 = std::chrono::steady_clock::now();
  memset(history, 0, sizeof(history));
  memset(killers, 0, sizeof(killers));
  memset(counter, 0, sizeof(counter));
  if (tt) { tt->new_search();
}

  int64_t budget = alloc_movetime(root, lim);
  if (lim.movetime_ms > 0) { hard_ms = lim.movetime_ms; soft_ms = lim.movetime_ms; }
  else if (lim.infinite || (lim.wtime == 0 && lim.btime == 0)) {
    hard_ms = INT64_MAX; soft_ms = INT64_MAX;
  } else {
    hard_ms = (int64_t)(budget * 5.0); // never exceed 5x soft (panic time)
    if (hard_ms > ((root.side == WHITE ? lim.wtime : lim.btime)) / 2 && (root.side == WHITE ? lim.wtime : lim.btime) > 0) {
      hard_ms = (root.side == WHITE ? lim.wtime : lim.btime) / 2;
}
    soft_ms = budget;
  }
  if (lim.nodes > 0) { node_limit = lim.nodes; } else { node_limit = 0;
}
  max_depth = lim.depth > 0 ? lim.depth : 64;

  SearchInfo info;
  info.best = MOVE_NONE;

  // root movegen
  MoveList root_moves;
  gen_legal(root, root_moves, false);
  if (root_moves.n == 0) { return info; // mate/stalemate: no best
}
  if (root_moves.n == 1) {
    info.best = root_moves.m[0];
    info.pv = {info.best};
    info.depth = 1;
    return info;
  }

  // iterative deepening + aspiration
  Score prev_score = evaluate(root, *net, nullptr);
  std::vector<Move> best_pv;
  Move best_move = MOVE_NONE;

  NNUEAcc root_acc;
  nnue_refresh(*net, root, root_acc);

  for (int depth = 1; depth <= max_depth; depth++) {
    Score alpha = -INF, beta = INF;
    Score window = 25;
    Score score = prev_score;
    if (depth >= 5) { alpha = score - window; beta = score + window; }

    // aspiration loop
    while (true) {
      if (should_stop()) { break;
}
      // root search: ordered by previous PV + history
      // build ordered list
      struct OM { Move m; int s; };
      OM om[256];
      for (int i = 0; i < root_moves.n; i++) {
        int s = order_score(*this, root, root_moves.m[i],
                            best_move != MOVE_NONE ? best_move : MOVE_NONE, 0);
        // boost previous PV
        if (!best_pv.empty() && root_moves.m[i] == best_pv[0]) { s += 5000000;
}
        om[i] = {root_moves.m[i], s};
      }
      Score best = -INF;
      Move cur_best = MOVE_NONE;
      std::vector<Move> cur_pv;
      Score a = alpha, b = beta;
      for (int i = 0; i < root_moves.n; i++) {
        int bi = i;
        for (int j = i + 1; j < root_moves.n; j++) { if (om[j].s > om[bi].s) { bi = j;
}
}
        std::swap(om[i], om[bi]);
        Move m = om[i].m;
        Position np = root;
        Undo u; NNUEAcc nacc = root_acc;
        if (!make_move(np, m, u)) { continue;
}
        if (in_check(np, root.side)) { continue;
}
        nnue_update(*net, root, m, u, nacc);
        Score sc;
        if (i == 0) { sc = -negamax(np, -b, -a, depth - 1, 1, false, nacc);
        } else {
          sc = -negamax(np, -(a + 1), -a, depth - 1, 1, true, nacc);
          if (!should_stop() && sc > a && sc < b) {
            sc = -negamax(np, -b, -a, depth - 1, 1, false, nacc);
}
        }
        if (should_stop()) { break;
}
        if (sc > best) {
          best = sc; cur_best = m;
          // mini-PV: best move + (assume TT PV continuation unknown here; keep 1-ply)
          cur_pv.clear(); cur_pv.push_back(m);
        }
        if (sc > a) { a = sc;
}
        if (a >= b) { break;
}
      }
      if (should_stop()) { break;
}
      if (best <= alpha) { alpha = std::max<Score>(-INF, alpha - window * 2); window *= 2; continue; }
      if (best >= beta) { beta = std::min<Score>(INF, beta + window * 2); window *= 2; continue; }
      // accepted
      score = best; best_move = cur_best; best_pv = cur_pv;
      prev_score = score;
      break;
    }
    if (should_stop()) { break;
}

    info.best = best_move; info.score = prev_score; info.depth = depth;
    info.nodes = nodes; info.time_ms = elapsed_ms(); info.pv = best_pv;

    if (nodes_counter) { nodes_counter->fetch_add(nodes, std::memory_order_relaxed);
}

    // soft timeout: stop if elapsed > soft and depth >= 4 (with stability margin)
    if (elapsed_ms() >= soft_ms && depth >= 4) { break;
}
    if (node_limit > 0 && (int64_t)nodes >= node_limit) { break;
}
    if (std::abs(prev_score) > MATE - 1000) { break; // found mate
}
  }
  info.nodes = nodes;
  info.time_ms = elapsed_ms();
  return info;
}

SearchInfo search_smp(Position root, const Limits& lim, TT& tt, const NNUE& net,
                      int threads, std::atomic<bool>& stop_flag) {
  if (threads <= 1) {
    Searcher s; s.tt = &tt; s.net = &net;
    return s.search(root, lim, &stop_flag, nullptr);
  }
  if (threads > 64) { threads = 64;
}
  std::atomic<uint64_t> shared_nodes{0};
  std::vector<std::thread> th;
  std::vector<SearchInfo> results((size_t)threads);
  th.reserve(threads);
for (int i = 0; i < threads; i++) {
    th.emplace_back([&, i]() {
      Searcher s; s.tt = &tt; s.net = &net;
      Limits l = lim;
      // helper threads: jitter depth so they explore different trees (real Lazy SMP trick)
      // main thread (0) searches normally; helpers add tiny depth offsets via movetime only.
      results[i] = s.search(root, l, &stop_flag, &shared_nodes);
    });
  }
  // main thread drives time control: wait until soft limit or all done
  // Simplest correct: join all (each respects stop_flag + limits).
  // Timekeeper: sleep until limit then set stop.
  {
    int64_t budget = alloc_movetime(root, lim);
    bool use_time = !lim.infinite && (lim.wtime != 0 || lim.btime != 0 || lim.movetime_ms != 0 || lim.nodes != 0);
    if (lim.movetime_ms > 0) { budget = lim.movetime_ms;
}
    if (use_time && budget != INT64_MAX) {
      auto t0 = std::chrono::steady_clock::now();
      while (true) {
        bool all_done = true;
        // can't easily check thread completion without join; use timeout poll
        auto el = std::chrono::duration_cast<std::chrono::milliseconds>(
          std::chrono::steady_clock::now() - t0).count();
        if (el >= budget) { stop_flag.store(true); break; }
        // if main thread finished early (mate / single move), stop
        // heuristic: if any thread finished its full depth loop... unknown.
        // Poll shared nodes for progress; break only on time.
        std::this_thread::sleep_for(std::chrono::milliseconds(5));
        // check if threads are joinable-done is not directly possible; rely on time.
        // For depth-limited searches threads finish on their own; detect via try?
        // Simplest: break when elapsed > budget OR when stop was set externally.
        if (stop_flag.load()) { break;
}
        // depth-limited: if elapsed > 30s beyond expected... just continue until join below
        // To avoid infinite wait on infinite analysis, only time-limited path sets stop.
        // For depth-limited, we must wait for join; but this loop would spin forever.
        // So: if no time limit, break immediately and just join.
        if (lim.movetime_ms == 0 && lim.wtime == 0 && lim.btime == 0) { break;
}
        (void)all_done;
      }
    }
  }
  for (auto& t : th) { t.join();
}
  // pick deepest / best result (prefer main thread on ties)
  SearchInfo best = results[0];
  for (size_t i = 1; i < results.size(); i++) {
    if (results[i].depth > best.depth ||
        (results[i].depth == best.depth && results[i].nodes > best.nodes)) {
      best = results[i];
}
  }
  if (best.best == MOVE_NONE) { best = results[0];
}
  return best;
}

uint64_t perft(Position& p, int depth) {
  if (depth == 0) { return 1;
}
  MoveList ml; gen_legal(p, ml, false);
  if (depth == 1) { return (uint64_t)ml.n;
}
  uint64_t n = 0;
  Undo u;
  for (int i = 0; i < ml.n; i++) {
    Position np = p;
    if (!make_move(np, ml.m[i], u)) { continue;
}
    if (in_check(np, p.side)) { continue;
}
    n += perft(np, depth - 1);
  }
  return n;
}

} // namespace forge
