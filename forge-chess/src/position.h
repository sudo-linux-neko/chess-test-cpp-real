// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Vaibhav Agarwal

// forge-chess — position.h : board representation, make/unmake, FEN, SEE helpers
#pragma once
#include "core.h"
#include <string>
#include <vector>

namespace forge {

struct Undo {
  Move move = MOVE_NONE;
  Piece captured = NO_PIECE;
  int castle = 0;
  Square ep = SQ_NONE;
  int half = 0;
  U64 hash = 0;
};

struct Position {
  // 12 bitboards
  U64 bb[2][6]{};
  U64 occ[2]{};
  U64 all = 0;
  // mailbox for fast eval / movegen helpers
  Piece board_pc[64]{};
  Color board_co[64]{};
  Color side = WHITE;
  int castle = 0; // bit0 W-OO bit1 W-OOO bit2 B-OO bit3 B-OOO
  Square ep = SQ_NONE;
  int half = 0, full = 1;
  U64 hash = 0;

  Position() {
    for (int i=0;i<64;i++){ board_pc[i]=NO_PIECE; board_co[i]=NO_COLOR; }
  }
};

void position_init(); // init attacks + zobrist (call once)
bool position_from_fen(Position& p, const std::string& fen);
std::string position_to_fen(const Position& p);
void position_startpos(Position& p);

U64 position_hash(const Position& p);
void position_rebuild(Position& p); // recompute occ/all/mailbox/hash from bb

Square king_sq(const Position& p, Color c);
bool square_attacked(const Position& p, Square s, Color by);
bool in_check(const Position& p, Color c);

bool make_move(Position& p, Move m, Undo& u);
void unmake_move(Position& p, const Undo& u);

// Move list
struct MoveList {
  Move m[256]{};
  int n = 0;
  void push(Move x){ if(n<256) { m[n++]=x; 
}}
  void clear(){ n=0; }
};

void gen_pseudo(const Position& p, MoveList& out, bool captures_only=false);
void gen_legal(const Position& p, MoveList& out, bool captures_only=false);
bool move_is_legal(const Position& p, Move m);
int  move_gives_check(const Position& p, Move m); // 0/1 (cheap approx ok)

// Static exchange evaluation (for ordering / pruning)
int see_value(Piece pc);
int see(const Position& p, Move m);

// Repetition / game-over helpers
bool insufficient_material(const Position& p);

} // namespace forge
