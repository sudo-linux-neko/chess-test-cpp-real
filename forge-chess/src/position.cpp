// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Vaibhav Agarwal

// forge-chess — position.cpp
#include "position.h"
#include <sstream>

namespace forge {

std::array<U64,64> KNIGHT_ATK{};
std::array<U64,64> KING_ATK{};
std::array<std::array<U64,64>,2> PAWN_ATK{};

U64 ZOB_PIECE[2][6][64]{};
U64 ZOB_SIDE = 0;
U64 ZOB_CASTLE[16]{};
U64 ZOB_EP[8]{};

static bool g_init_done = false;

void init_attacks() {
  for (int s=0;s<64;s++) {
    int f = file_of((Square)s), r = rank_of((Square)s);
    U64 kn=0, ki=0;
    const int kdf[8]={1,2,2,1,-1,-2,-2,-1};
    const int kdr[8]={2,1,-1,-2,-2,-1,1,2};
    for(int i=0;i<8;i++){ int nf=f+kdf[i], nr=r+kdr[i];
      if(nf>=0&&nf<8&&nr>=0&&nr<8) { kn |= sq_bb(make_sq(nf,nr)); 
}}
    for(int df=-1;df<=1;df++) { for(int dr=-1;dr<=1;dr++){
      if(!df&&!dr) { continue;
}
      int nf=f+df, nr=r+dr;
      if(nf>=0&&nf<8&&nr>=0&&nr<8) { ki |= sq_bb(make_sq(nf,nr));
}
    }
}
    KNIGHT_ATK[s]=kn; KING_ATK[s]=ki;
    U64 wp=0, bp=0;
    if (r < 7) {
      if (f>0) { wp |= sq_bb(make_sq(f-1,r+1));
}
      if (f<7) { wp |= sq_bb(make_sq(f+1,r+1));
}
    }
    if (r > 0) {
      if (f>0) { bp |= sq_bb(make_sq(f-1,r-1));
}
      if (f<7) { bp |= sq_bb(make_sq(f+1,r-1));
}
    }
    PAWN_ATK[WHITE][s]=wp; PAWN_ATK[BLACK][s]=bp;
  }
}

void init_zobrist() {
  U64 s = 0x9E3779B97F4A7C15ULL ^ 0x123456789ABCDEFULL;
  for(auto & c : ZOB_PIECE) { for(auto & p : c) { for(unsigned long & q : p) {
    q=splitmix64_adv(s);
}
}
}
  ZOB_SIDE=splitmix64_adv(s);
  for(unsigned long & i : ZOB_CASTLE) { i=splitmix64_adv(s);
}
  for(unsigned long & i : ZOB_EP) { i=splitmix64_adv(s);
}
}

void position_init() {
  if (g_init_done) { return;
}
  init_attacks(); init_zobrist();
  g_init_done = true;
}

U64 position_hash(const Position& p) {
  U64 h = 0;
  for(int s=0;s<64;s++) { if(p.board_pc[s]!=NO_PIECE) {
    h ^= ZOB_PIECE[p.board_co[s]][p.board_pc[s]][s];
}
}
  if (p.side==BLACK) { h ^= ZOB_SIDE;
}
  h ^= ZOB_CASTLE[p.castle & 15];
  if (p.ep != SQ_NONE) { h ^= ZOB_EP[file_of(p.ep)];
}
  return h;
}

void position_rebuild(Position& p) {
  p.occ[WHITE]=p.occ[BLACK]=0;
  for(int c=0;c<2;c++) { for(int q=0;q<6;q++) { p.occ[c]|=p.bb[c][q];
}
}
  p.all = p.occ[WHITE]|p.occ[BLACK];
  for(int i=0;i<64;i++){ p.board_pc[i]=NO_PIECE; p.board_co[i]=NO_COLOR; }
  for(int c=0;c<2;c++) { for(int q=0;q<6;q++) {
    U64 b=p.bb[c][q];
    while(b){ Square s=lsb_sq(b); poplsb(b); p.board_pc[s]=(Piece)q; p.board_co[s]=(Color)c; }
  }
}
  p.hash = position_hash(p);
}

void position_startpos(Position& p) {
  position_from_fen(p, "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1");
}

bool position_from_fen(Position& p, const std::string& fen) {
  position_init();
  for(auto & c : p.bb) { for(unsigned long & q : c) { q=0;
}
}
  p.side=WHITE; p.castle=0; p.ep=SQ_NONE; p.half=0; p.full=1;
  std::istringstream ss(fen);
  std::string board, stm, cast, eps, half, full;
  if(!(ss>>board)) { return false;
}
  if(!(ss>>stm)) { return false;
}
  if(!(ss>>cast)) { return false;
}
  if(!(ss>>eps)) { return false;
}
  if(!(ss>>half)) { half="0";
}
  if(!(ss>>full)) { full="1";
}
  int r=7, f=0;
  for(char ch: board){
    if(ch=='/'){ r--; f=0; continue; }
    if(ch>='1'&&ch<='8'){ f+=ch-'0'; continue; }
    Color c = (ch>='A'&&ch<='Z')?WHITE:BLACK;
    Piece pc=NO_PIECE;
    char l = (char)tolower(ch);
    if(l=='p') {pc=PAWN; } else if(l=='n') {pc=KNIGHT; } else if(l=='b') {pc=BISHOP;
    } else if(l=='r') {pc=ROOK; } else if(l=='q') {pc=QUEEN; } else if(l=='k') {pc=KING;
    } else { return false;
}
    if(f>7||r<0) { return false;
}
    p.bb[c][pc] |= sq_bb(make_sq(f,r));
    f++;
  }
  p.side = (stm=="b")?BLACK:WHITE;
  p.castle=0;
  if(cast!="-") { for(char ch: cast){
    if(ch=='K') {p.castle|=1; } else if(ch=='Q') {p.castle|=2;
    } else if(ch=='k') {p.castle|=4; } else if(ch=='q') {p.castle|=8;
}
  }
}
  if(eps=="-"||eps=="–") { p.ep=SQ_NONE;
  } else {
    int ff=eps[0]-'a', rr=eps[1]-'1';
    if(ff>=0&&ff<8&&rr>=0&&rr<8) { p.ep=make_sq(ff,rr); } else { p.ep=SQ_NONE;
}
  }
  p.half=atoi(half.c_str()); p.full=atoi(full.c_str());
  if(p.full<1) {p.full=1;
}
  position_rebuild(p);
  return true;
}

std::string position_to_fen(const Position& p) {
  std::string s;
  for(int r=7;r>=0;r--){
    int empty=0;
    for(int f=0;f<8;f++){
      Square sq=make_sq(f,r);
      if(p.board_pc[sq]==NO_PIECE){ empty++; continue; }
      if(empty){ s+=char('0'+empty); empty=0; }
      char ch='?';
      switch(p.board_pc[sq]){case PAWN:ch='p';break;case KNIGHT:ch='n';break;
        case BISHOP:ch='b';break;case ROOK:ch='r';break;case QUEEN:ch='q';break;
        case KING:ch='k';break;default:ch='?';break;}
      if(p.board_co[sq]==WHITE) { ch=toupper(ch);
}
      s+=ch;
    }
    if(empty) { s+=char('0'+empty);
}
    if(r) { s+='/';
}
  }
  s += (p.side==WHITE?" w ":" b ");
  std::string c;
  if(p.castle&1) {c+='K'; 
}if(p.castle&2) {c+='Q'; 
}if(p.castle&4) {c+='k'; 
}if(p.castle&8) {c+='q';
}
  if(c.empty()) {c="-";
}
  s+=c; s+=" ";
  s += (p.ep==SQ_NONE?"-":sq_name(p.ep));
  s += " " + std::to_string(p.half) + " " + std::to_string(p.full);
  return s;
}

Square king_sq(const Position& p, Color c) {
  U64 b = p.bb[c][KING];
  if(!b) { return SQ_NONE;
}
  return lsb_sq(b);
}

bool square_attacked(const Position& pos, Square s, Color by) {
  // pawn
  // a square s is attacked by a pawn of color 'by' if a pawn of that color
  // sits on a square from which it attacks s. PAWN_ATK[by][from] contains
  // targets, so reverse lookup: check squares that could attack s.
  // Cheap: test both potential pawn origins.
  int f=file_of(s), r=rank_of(s);
  if (by==WHITE) {
    if (r>0) {
      if (f>0 && (pos.bb[WHITE][PAWN] & sq_bb(make_sq(f-1,r-1)))) { return true;
}
      if (f<7 && (pos.bb[WHITE][PAWN] & sq_bb(make_sq(f+1,r-1)))) { return true;
}
    }
  } else {
    if (r<7) {
      if (f>0 && (pos.bb[BLACK][PAWN] & sq_bb(make_sq(f-1,r+1)))) { return true;
}
      if (f<7 && (pos.bb[BLACK][PAWN] & sq_bb(make_sq(f+1,r+1)))) { return true;
}
    }
  }
  if (KNIGHT_ATK[s] & pos.bb[by][KNIGHT]) { return true;
}
  if (KING_ATK[s] & pos.bb[by][KING]) { return true;
}
  U64 bq = pos.bb[by][BISHOP] | pos.bb[by][QUEEN];
  if (bq && (bishop_attacks(s, pos.all) & bq)) { return true;
}
  U64 rq = pos.bb[by][ROOK] | pos.bb[by][QUEEN];
  return rq && (rook_attacks(s, pos.all) & rq);
}

bool in_check(const Position& p, Color c) {
  Square k = king_sq(p,c);
  if(k==SQ_NONE) { return true;
}
  return square_attacked(p, k, opposite(c));
}

// castle rights decay table: on from/to square, mask out rights
static int castle_decay(Square s) {
  switch(s){
    case 4:  return ~(1|2);   // e1
    case 7:  return ~1;       // h1
    case 0:  return ~2;       // a1
    case 60: return ~(4|8);   // e8
    case 63: return ~4;       // h8
    case 56: return ~8;       // a8
    default: return ~0;
  }
}

bool make_move(Position& p, Move m, Undo& u) {
  if (m==MOVE_NONE) { return false;
}
  Square from=move_from(m), to=move_to(m);
  Piece pc = p.board_pc[from];
  if (pc==NO_PIECE) { return false;
}
  Color us=p.side, them=opposite(us);
  Piece cap = p.board_pc[to];
  if (cap!=NO_PIECE && p.board_co[to]!=them) { return false;
}

  u.move=m; u.captured=cap; u.castle=p.castle; u.ep=p.ep; u.half=p.half; u.hash=p.hash;

  Square cap_sq = to;
  bool is_ep = move_is_ep(m);
  bool is_castle = move_is_castle(m);
  bool is_promo = move_is_promo(m);

  // hash out old ep/castle/side
  p.hash ^= ZOB_CASTLE[p.castle & 15];
  if (p.ep!=SQ_NONE) { p.hash ^= ZOB_EP[file_of(p.ep)];
}
  p.hash ^= ZOB_SIDE;

  // remove ep
  p.ep = SQ_NONE;

  // handle ep capture removal
  if (is_ep) {
    cap_sq = (us==WHITE)? (Square)(to-8) : (Square)(to+8);
    cap = PAWN;
    u.captured = PAWN;
  }

  // remove moving piece from 'from'
  p.bb[us][pc] &= ~sq_bb(from);
  p.hash ^= ZOB_PIECE[us][pc][from];
  // remove captured
  if (cap!=NO_PIECE) {
    p.bb[them][cap] &= ~sq_bb(cap_sq);
    p.hash ^= ZOB_PIECE[them][cap][cap_sq];
  }

  // castle rook move
  if (is_castle) {
    if (us==WHITE) {
      if (to==make_sq(6,0)){ // g1
        p.bb[WHITE][ROOK] &= ~sq_bb(make_sq(7,0));
        p.bb[WHITE][ROOK] |= sq_bb(make_sq(5,0));
        p.hash ^= ZOB_PIECE[WHITE][ROOK][make_sq(7,0)];
        p.hash ^= ZOB_PIECE[WHITE][ROOK][make_sq(5,0)];
      } else { // c1
        p.bb[WHITE][ROOK] &= ~sq_bb(make_sq(0,0));
        p.bb[WHITE][ROOK] |= sq_bb(make_sq(3,0));
        p.hash ^= ZOB_PIECE[WHITE][ROOK][make_sq(0,0)];
        p.hash ^= ZOB_PIECE[WHITE][ROOK][make_sq(3,0)];
      }
    } else {
      if (to==make_sq(6,7)){
        p.bb[BLACK][ROOK] &= ~sq_bb(make_sq(7,7));
        p.bb[BLACK][ROOK] |= sq_bb(make_sq(5,7));
        p.hash ^= ZOB_PIECE[BLACK][ROOK][make_sq(7,7)];
        p.hash ^= ZOB_PIECE[BLACK][ROOK][make_sq(5,7)];
      } else {
        p.bb[BLACK][ROOK] &= ~sq_bb(make_sq(0,7));
        p.bb[BLACK][ROOK] |= sq_bb(make_sq(3,7));
        p.hash ^= ZOB_PIECE[BLACK][ROOK][make_sq(0,7)];
        p.hash ^= ZOB_PIECE[BLACK][ROOK][make_sq(3,7)];
      }
    }
  }

  // place moving (or promoted) piece
  Piece placed = pc;
  if (is_promo) { placed = move_promo_piece(m);
}
  p.bb[us][placed] |= sq_bb(to);
  p.hash ^= ZOB_PIECE[us][placed][to];

  // double push -> set ep
  if (pc==PAWN && !is_ep && !is_promo) {
    int dr = (int)to - (int)from;
    if (dr==16 || dr==-16) { p.ep = (Square)((from+to)/2);
}
  }

  // castle rights decay
  p.castle &= castle_decay(from);
  p.castle &= castle_decay(to);

  // halfmove / fullmove
  if (pc==PAWN || cap!=NO_PIECE) { p.half=0; } else { p.half++;
}
  if (us==BLACK) { p.full++;
}

  p.side = them;

  // hash in new castle/ep
  p.hash ^= ZOB_CASTLE[p.castle & 15];
  if (p.ep!=SQ_NONE) { p.hash ^= ZOB_EP[file_of(p.ep)];
}

  // incremental occ/mailbox update (cheap, avoids full rebuild)
  p.occ[WHITE]=p.bb[WHITE][0]|p.bb[WHITE][1]|p.bb[WHITE][2]|p.bb[WHITE][3]|p.bb[WHITE][4]|p.bb[WHITE][5];
  p.occ[BLACK]=p.bb[BLACK][0]|p.bb[BLACK][1]|p.bb[BLACK][2]|p.bb[BLACK][3]|p.bb[BLACK][4]|p.bb[BLACK][5];
  p.all=p.occ[WHITE]|p.occ[BLACK];
  p.board_pc[from]=NO_PIECE; p.board_co[from]=NO_COLOR;
  if (is_ep) { p.board_pc[cap_sq]=NO_PIECE; p.board_co[cap_sq]=NO_COLOR; }
  if (is_castle) {
    // clear+set rook mailbox
    if (us==WHITE) {
      if (to==make_sq(6,0)){ p.board_pc[make_sq(7,0)]=NO_PIECE; p.board_co[make_sq(7,0)]=NO_COLOR;
        p.board_pc[make_sq(5,0)]=ROOK; p.board_co[make_sq(5,0)]=WHITE; }
      else { p.board_pc[make_sq(0,0)]=NO_PIECE; p.board_co[make_sq(0,0)]=NO_COLOR;
        p.board_pc[make_sq(3,0)]=ROOK; p.board_co[make_sq(3,0)]=WHITE; }
    } else {
      if (to==make_sq(6,7)){ p.board_pc[make_sq(7,7)]=NO_PIECE; p.board_co[make_sq(7,7)]=NO_COLOR;
        p.board_pc[make_sq(5,7)]=ROOK; p.board_co[make_sq(5,7)]=BLACK; }
      else { p.board_pc[make_sq(0,7)]=NO_PIECE; p.board_co[make_sq(0,7)]=NO_COLOR;
        p.board_pc[make_sq(3,7)]=ROOK; p.board_co[make_sq(3,7)]=BLACK; }
    }
  }
  p.board_pc[to]=placed; p.board_co[to]=us;
  return true;
}

void unmake_move(Position& p, const Undo& u) {
  Move m=u.move;
  Square from=move_from(m), to=move_to(m);
  Color them=p.side; // side to move now is the opponent of mover
  Color us=opposite(them);
  bool is_ep=move_is_ep(m), is_castle=move_is_castle(m), is_promo=move_is_promo(m);
  Piece placed = p.board_pc[to];
  Piece pc = is_promo ? PAWN : placed;

  // remove placed
  p.bb[us][placed] &= ~sq_bb(to);
  // restore mover
  p.bb[us][pc] |= sq_bb(from);
  // restore capture
  if (u.captured!=NO_PIECE) {
    Square cap_sq = is_ep ? (us==WHITE?(Square)(to-8):(Square)(to+8)) : to;
    p.bb[them][u.captured] |= sq_bb(cap_sq);
  }
  // undo castle rook
  if (is_castle) {
    if (us==WHITE) {
      if (to==make_sq(6,0)){ p.bb[WHITE][ROOK]&=~sq_bb(make_sq(5,0)); p.bb[WHITE][ROOK]|=sq_bb(make_sq(7,0)); }
      else { p.bb[WHITE][ROOK]&=~sq_bb(make_sq(3,0)); p.bb[WHITE][ROOK]|=sq_bb(make_sq(0,0)); }
    } else {
      if (to==make_sq(6,7)){ p.bb[BLACK][ROOK]&=~sq_bb(make_sq(5,7)); p.bb[BLACK][ROOK]|=sq_bb(make_sq(7,7)); }
      else { p.bb[BLACK][ROOK]&=~sq_bb(make_sq(3,7)); p.bb[BLACK][ROOK]|=sq_bb(make_sq(0,7)); }
    }
  }
  p.side=us; p.castle=u.castle; p.ep=u.ep;
  if (us==BLACK) { p.full--; // we incremented when black moved... careful:
}
  // make incremented full only when mover was black. Here mover=us.
  // Above: if(us==BLACK) full-- would double-undo? make did if(us==BLACK) full++ where us was mover.
  // Now us is mover, so correct.
  // halfmove: restore (we don't track full precisely beyond that; recompute half from u)
  p.half=u.half;
  p.hash=u.hash;
  // rebuild occ/mailbox (unmake is rare vs make; simplicity > speed)
  p.occ[WHITE]=p.bb[WHITE][0]|p.bb[WHITE][1]|p.bb[WHITE][2]|p.bb[WHITE][3]|p.bb[WHITE][4]|p.bb[WHITE][5];
  p.occ[BLACK]=p.bb[BLACK][0]|p.bb[BLACK][1]|p.bb[BLACK][2]|p.bb[BLACK][3]|p.bb[BLACK][4]|p.bb[BLACK][5];
  p.all=p.occ[WHITE]|p.occ[BLACK];
  for(int i=0;i<64;i++){ p.board_pc[i]=NO_PIECE; p.board_co[i]=NO_COLOR; }
  for(int c=0;c<2;c++) { for(int q=0;q<6;q++){
    U64 b=p.bb[c][q];
    while(b){ Square s=lsb_sq(b); poplsb(b); p.board_pc[s]=(Piece)q; p.board_co[s]=(Color)c; }
  }
}
}

// ---------- Move generation ----------
static void push_pawn_moves(const Position& p, MoveList& out, Square from, U64 targets, bool promo_rank) {
  while(targets){
    Square to=lsb_sq(targets); poplsb(targets);
    if(promo_rank){
      out.push(make_promo(from,to,QUEEN)); out.push(make_promo(from,to,ROOK));
      out.push(make_promo(from,to,BISHOP)); out.push(make_promo(from,to,KNIGHT));
    } else { out.push(make_move(from,to));
}
  }
}

void gen_pseudo(const Position& p, MoveList& out, bool captures_only) {
  out.clear();
  Color us=p.side, them=opposite(us);
  U64 own=p.occ[us], enemy=p.occ[them], all=p.all;
  U64 empty=~all;

  // Pawns
  U64 pawns=p.bb[us][PAWN];
  if (us==WHITE) {
    // single pushes
    U64 single = shift_n(pawns) & empty;
    U64 single_promo = single & RANK_8, single_quiet = single & ~RANK_8;
    if(!captures_only){
      U64 b=single_quiet;
      while(b){ Square to=lsb_sq(b); poplsb(b); out.push(make_move((Square)(to-8),to)); }
      b=single_promo;
      while(b){ Square to=lsb_sq(b); poplsb(b); auto from=(Square)(to-8);
        out.push(make_promo(from,to,QUEEN)); out.push(make_promo(from,to,ROOK));
        out.push(make_promo(from,to,BISHOP)); out.push(make_promo(from,to,KNIGHT)); }
      // double pushes
      U64 dbl = shift_n(single & RANK_2 & empty) & empty; // single already ensures one step empty; RANK_2 mask on intermediate?
      // Correct double: pawns on rank2 with both squares empty.
      // Recompute precisely:
      dbl = 0;
      U64 r2 = pawns & RANK_2;
      while(r2){ Square from=lsb_sq(r2); poplsb(r2);
        auto one=(Square)(from+8), two=(Square)(from+16);
        if(!(all&sq_bb(one)) && !(all&sq_bb(two))) { out.push(make_move(from,two));
}
      }
    } else {
      // captures_only: no quiet pushes except promos-push? standard quiescence includes promo pushes.
      U64 b=single_promo;
      while(b){ Square to=lsb_sq(b); poplsb(b); auto from=(Square)(to-8);
        out.push(make_promo(from,to,QUEEN)); out.push(make_promo(from,to,ROOK));
        out.push(make_promo(from,to,BISHOP)); out.push(make_promo(from,to,KNIGHT)); }
    }
    // captures
    U64 b=pawns;
    while(b){
      Square from=lsb_sq(b); poplsb(b);
      U64 atks = PAWN_ATK[WHITE][from] & (enemy | (p.ep!=SQ_NONE?sq_bb(p.ep):0ULL));
      // in captures_only keep all captures; else keep all too
      while(atks){
        Square to=lsb_sq(atks); poplsb(atks);
        bool ispromo = (to>=56);
        bool isep = (to==p.ep);
        if(isep){ out.push(make_ep(from,to)); continue; }
        if(ispromo){
          out.push(make_promo(from,to,QUEEN)); out.push(make_promo(from,to,ROOK));
          out.push(make_promo(from,to,BISHOP)); out.push(make_promo(from,to,KNIGHT));
        } else { out.push(make_move(from,to));
}
      }
    }
  } else {
    U64 single = shift_s(pawns) & empty;
    U64 single_promo = single & RANK_1, single_quiet = single & ~RANK_1;
    if(!captures_only){
      U64 b=single_quiet;
      while(b){ Square to=lsb_sq(b); poplsb(b); out.push(make_move((Square)(to+8),to)); }
      b=single_promo;
      while(b){ Square to=lsb_sq(b); poplsb(b); auto from=(Square)(to+8);
        out.push(make_promo(from,to,QUEEN)); out.push(make_promo(from,to,ROOK));
        out.push(make_promo(from,to,BISHOP)); out.push(make_promo(from,to,KNIGHT)); }
      U64 r7 = pawns & RANK_7;
      while(r7){ Square from=lsb_sq(r7); poplsb(r7);
        auto one=(Square)(from-8), two=(Square)(from-16);
        if(!(all&sq_bb(one)) && !(all&sq_bb(two))) { out.push(make_move(from,two));
}
      }
    } else {
      U64 b=single_promo;
      while(b){ Square to=lsb_sq(b); poplsb(b); auto from=(Square)(to+8);
        out.push(make_promo(from,to,QUEEN)); out.push(make_promo(from,to,ROOK));
        out.push(make_promo(from,to,BISHOP)); out.push(make_promo(from,to,KNIGHT)); }
    }
    U64 b=pawns;
    while(b){
      Square from=lsb_sq(b); poplsb(b);
      U64 atks = PAWN_ATK[BLACK][from] & (enemy | (p.ep!=SQ_NONE?sq_bb(p.ep):0ULL));
      while(atks){
        Square to=lsb_sq(atks); poplsb(atks);
        bool ispromo=(to<8);
        bool isep=(to==p.ep);
        if(isep){ out.push(make_ep(from,to)); continue; }
        if(ispromo){
          out.push(make_promo(from,to,QUEEN)); out.push(make_promo(from,to,ROOK));
          out.push(make_promo(from,to,BISHOP)); out.push(make_promo(from,to,KNIGHT));
        } else { out.push(make_move(from,to));
}
      }
    }
  }

  // Knights
  {
    U64 b=p.bb[us][KNIGHT];
    while(b){ Square from=lsb_sq(b); poplsb(b);
      U64 t=KNIGHT_ATK[from]&~own;
      if(captures_only) { t&=enemy;
}
      while(t){ Square to=lsb_sq(t); poplsb(t); out.push(make_move(from,to)); }
    }
  }
  // Bishops
  {
    U64 b=p.bb[us][BISHOP];
    while(b){ Square from=lsb_sq(b); poplsb(b);
      U64 t=bishop_attacks(from,all)&~own;
      if(captures_only) { t&=enemy;
}
      while(t){ Square to=lsb_sq(t); poplsb(t); out.push(make_move(from,to)); }
    }
  }
  // Rooks
  {
    U64 b=p.bb[us][ROOK];
    while(b){ Square from=lsb_sq(b); poplsb(b);
      U64 t=rook_attacks(from,all)&~own;
      if(captures_only) { t&=enemy;
}
      while(t){ Square to=lsb_sq(t); poplsb(t); out.push(make_move(from,to)); }
    }
  }
  // Queens
  {
    U64 b=p.bb[us][QUEEN];
    while(b){ Square from=lsb_sq(b); poplsb(b);
      U64 t=queen_attacks(from,all)&~own;
      if(captures_only) { t&=enemy;
}
      while(t){ Square to=lsb_sq(t); poplsb(t); out.push(make_move(from,to)); }
    }
  }
  // King
  {
    Square k=king_sq(p,us);
    if(k!=SQ_NONE){
      U64 t=KING_ATK[k]&~own;
      if(captures_only) { t&=enemy;
}
      while(t){ Square to=lsb_sq(t); poplsb(t); out.push(make_move(k,to)); }
    }
  }
  // Castling (only in full search, not captures)
  if(!captures_only){
    if(us==WHITE){
      if((p.castle&1) && !(all&(sq_bb(make_sq(5,0))|sq_bb(make_sq(6,0))))){
        if(!square_attacked(p,make_sq(4,0),BLACK)&&!square_attacked(p,make_sq(5,0),BLACK)) {
          out.push(make_castle(make_sq(4,0),make_sq(6,0)));
}
      }
      if((p.castle&2) && !(all&(sq_bb(make_sq(1,0))|sq_bb(make_sq(2,0))|sq_bb(make_sq(3,0))))){
        if(!square_attacked(p,make_sq(4,0),BLACK)&&!square_attacked(p,make_sq(3,0),BLACK)) {
          out.push(make_castle(make_sq(4,0),make_sq(2,0)));
}
      }
    } else {
      if((p.castle&4) && !(all&(sq_bb(make_sq(5,7))|sq_bb(make_sq(6,7))))){
        if(!square_attacked(p,make_sq(4,7),WHITE)&&!square_attacked(p,make_sq(5,7),WHITE)) {
          out.push(make_castle(make_sq(4,7),make_sq(6,7)));
}
      }
      if((p.castle&8) && !(all&(sq_bb(make_sq(1,7))|sq_bb(make_sq(2,7))|sq_bb(make_sq(3,7))))){
        if(!square_attacked(p,make_sq(4,7),WHITE)&&!square_attacked(p,make_sq(3,7),WHITE)) {
          out.push(make_castle(make_sq(4,7),make_sq(2,7)));
}
      }
    }
  }
}

void gen_legal(const Position& p, MoveList& out, bool captures_only) {
  MoveList pseudo; gen_pseudo(p, pseudo, captures_only);
  out.clear();
  Position tmp; Undo u;
  for(int i=0;i<pseudo.n;i++){
    tmp=p;
    if(!make_move(tmp,pseudo.m[i],u)) { continue;
}
    // mover was p.side; ensure mover's king not left in check
    if(in_check(tmp, p.side)) { continue;
}
    // also: castling already checked, but double-check king destination not attacked
    // (covered by in_check? No — castling through check is already filtered in gen_pseudo.)
    out.push(pseudo.m[i]);
  }
}

bool move_is_legal(const Position& p, Move m) {
  MoveList l; gen_legal(p,l,false);
  for(int i=0;i<l.n;i++) { if(l.m[i]==m) { return true;
}
}
  return false;
}

int move_gives_check(const Position& p, Move m) {
  Position tmp=p; Undo u;
  if(!make_move(tmp,m,u)) { return 0;
}
  return in_check(tmp, tmp.side) ? 1 : 0;
}

int see_value(Piece pc){
  switch(pc){case PAWN:return 100;case KNIGHT:return 320;case BISHOP:return 330;
    case ROOK:return 500;case QUEEN:return 900;case KING:return 20000;default:return 0;}
}

// Fast approximate SEE: victim - attacker/16 + promo bonus. Good for ordering/pruning.
int see(const Position& p, Move m){
  Square from=move_from(m), to=move_to(m);
  Piece att=p.board_pc[from];
  if(att==NO_PIECE) { return -100000;
}
  Piece vic = move_is_ep(m) ? PAWN : p.board_pc[to];
  int v=0;
  if(vic!=NO_PIECE) { v+=see_value(vic);
}
  if(move_is_promo(m)) { v+=see_value(move_promo_piece(m))-see_value(PAWN);
}
  v-=see_value(att)/16;
  return v;
}

bool insufficient_material(const Position& p){
  if(p.bb[WHITE][PAWN]|p.bb[BLACK][PAWN]|p.bb[WHITE][ROOK]|p.bb[BLACK][ROOK]|p.bb[WHITE][QUEEN]|p.bb[BLACK][QUEEN]) {
    return false;
}
  int n = popcount(p.bb[WHITE][KNIGHT]|p.bb[WHITE][BISHOP]|p.bb[BLACK][KNIGHT]|p.bb[BLACK][BISHOP]);
  if(n<=1) { return true;
}
  // K+B vs K+B same color bishops only
  if(n==2 && !p.bb[WHITE][KNIGHT] && !p.bb[BLACK][KNIGHT]){
    U64 b = p.bb[WHITE][BISHOP]|p.bb[BLACK][BISHOP];
    Square s1=lsb_sq(b); poplsb(b); Square s2=lsb_sq(b);
    if(((file_of(s1)+rank_of(s1))&1)==((file_of(s2)+rank_of(s2))&1)) { return true;
}
  }
  return false;
}

} // namespace forge
