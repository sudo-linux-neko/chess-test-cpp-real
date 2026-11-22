// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Vaibhav Agarwal

// forge-chess — uci.cpp : full UCI + pretty board + async stop
#include "uci.h"
#include "core.h"
#include <iostream>
#include <sstream>
#include <thread>
#include <mutex>
#include <queue>
#include <condition_variable>
#include <algorithm>
#include <cctype>

namespace forge {

// ---------- pretty board ----------
static const char* PIECE_UNICODE[2][6] = {
  {"\u2659","\u2658","\u2657","\u2656","\u2655","\u2654"}, // white (outline)
  {"\u265F","\u265E","\u265D","\u265C","\u265B","\u265A"}, // black (filled)
};

void print_board_pretty(const Position& p) {
  std::cout << "\n  +-----------------+\n";
  for (int r = 7; r >= 0; r--) {
    std::cout << r + 1 << " | ";
    for (int f = 0; f < 8; f++) {
      Square s = make_sq(f, r);
      if (p.board_pc[s] == NO_PIECE) {
        std::cout << (((f + r) & 1) ? ". " : "\u00B7 ");
      } else {
        std::cout << PIECE_UNICODE[p.board_co[s]][p.board_pc[s]] << " ";
      }
    }
    std::cout << "|\n";
  }
  std::cout << "  +-----------------+\n    a b c d e f g h\n";
  std::cout << "FEN: " << position_to_fen(p) << "\n";
  std::cout << (p.side == WHITE ? "White" : "Black") << " to move";
  if (in_check(p, p.side)) { std::cout << "  [CHECK!]";
}
  std::cout << "\n";
  std::cout.flush();
}

static std::vector<std::string> split_ws(const std::string& s) {
  std::istringstream ss(s);
  std::vector<std::string> v; std::string t;
  while (ss >> t) { v.push_back(t);
}
  return v;
}

static Move parse_uci_move(const Position& p, const std::string& s) {
  if (s.size() < 4) { return MOVE_NONE;
}
  Square from = sq_parse(s, 0), to = sq_parse(s, 2);
  if (from == SQ_NONE || to == SQ_NONE) { return MOVE_NONE;
}
  Piece pc = p.board_pc[from];
  // castle detection
  if (pc == KING && std::abs((int)to - (int)from) == 2) {
    return make_castle(from, to);
}
  // ep detection
  if (pc == PAWN && to == p.ep) {
    return make_ep(from, to);
}
  if (s.size() >= 5) {
    char pr = tolower(s[4]);
    Piece pp = QUEEN;
    if (pr == 'n') { pp = KNIGHT; } else if (pr == 'b') { pp = BISHOP;
    } else if (pr == 'r') { pp = ROOK; } else { pp = QUEEN;
}
    return make_promo(from, to, pp);
  }
  return make_move(from, to);
}

bool parse_position_cmd(Position& p, std::vector<U64>& hist, const std::string& line) {
  // "position [startpos|fen ...] [moves ...]"
  auto tok = split_ws(line);
  if (tok.size() < 2) { return false;
}
  size_t i = 1;
  if (tok[i] == "startpos") {
    position_startpos(p); i++;
  } else if (tok[i] == "fen") {
    std::string fen;
    i++;
    // fen is 6 fields
    for (int k = 0; k < 6 && i < tok.size() && tok[i] != "moves"; k++, i++) {
      if (k) { fen += " ";
}
      fen += tok[i];
    }
    if (!position_from_fen(p, fen)) { return false;
}
  } else { return false;
}
  hist.clear(); hist.push_back(p.hash);
  if (i < tok.size() && tok[i] == "moves") {
    i++;
    for (; i < tok.size(); i++) {
      Move m = parse_uci_move(p, tok[i]);
      if (m == MOVE_NONE) { return false;
}
      // verify legality (handles promo/castle/ep canonicalization)
      MoveList ml; gen_legal(p, ml, false);
      Move found = MOVE_NONE;
      // match by from/to/promo (UCI has no special flags)
      for (int k = 0; k < ml.n; k++) {
        if (move_from(ml.m[k]) == move_from(m) && move_to(ml.m[k]) == move_to(m)) {
          if (move_is_promo(m)) {
            if (move_is_promo(ml.m[k]) && move_promo_piece(ml.m[k]) == move_promo_piece(m)) { found = ml.m[k]; break; }
          } else {
            if (!move_is_promo(ml.m[k])) { found = ml.m[k]; break; }
          }
        }
      }
      if (found == MOVE_NONE) { return false;
}
      Undo u; make_move(p, found, u);
      hist.push_back(p.hash);
    }
  }
  return true;
}

Limits parse_go_cmd(const std::string& line, const Position& p) {
  Limits lim;
  lim.depth = 64;
  auto tok = split_ws(line);
  for (size_t i = 1; i < tok.size(); i++) {
    if (tok[i] == "depth" && i + 1 < tok.size()) { lim.depth = atoi(tok[++i].c_str());
    } else if (tok[i] == "nodes" && i + 1 < tok.size()) { lim.nodes = atoll(tok[++i].c_str());
    } else if (tok[i] == "movetime" && i + 1 < tok.size()) { lim.movetime_ms = atoll(tok[++i].c_str());
    } else if (tok[i] == "wtime" && i + 1 < tok.size()) { lim.wtime = atoll(tok[++i].c_str());
    } else if (tok[i] == "btime" && i + 1 < tok.size()) { lim.btime = atoll(tok[++i].c_str());
    } else if (tok[i] == "winc" && i + 1 < tok.size()) { lim.winc = atoll(tok[++i].c_str());
    } else if (tok[i] == "binc" && i + 1 < tok.size()) { lim.binc = atoll(tok[++i].c_str());
    } else if (tok[i] == "movestogo" && i + 1 < tok.size()) { lim.movestogo = atoi(tok[++i].c_str());
    } else if (tok[i] == "infinite") { lim.infinite = true;
}
  }
  // safety: if no limits at all, pick a sane default (avoid hanging GUIs)
  if (!lim.infinite && lim.movetime_ms == 0 && lim.wtime == 0 && lim.btime == 0 &&
      lim.nodes == 0 && lim.depth == 64) {
    // keep depth 64 (will be stopped by GUI's stop); nothing to do
  }
  (void)p;
  return lim;
}

// ---------- async input queue ----------
struct InputQueue {
  std::mutex m;
  std::condition_variable cv;
  std::queue<std::string> q;
  bool closed = false;
  void push(const std::string& s) {
    { std::lock_guard<std::mutex> l(m); q.push(s); }
    cv.notify_one();
  }
};

void uci_loop() {
  position_init();
  UCIOptions opt;
  opt.threads = std::max(1, std::min(default_threads(), 64));
  TT tt; tt.resize((size_t)opt.hash_mb);
  NNUE net;
  if (!opt.nnue_file.empty()) { nnue_load(net, opt.nnue_file);
}
  // auto-load default net next to binary if present
  if (!net.trained) {
    if (nnue_load(net, "forge.nnue")) { /* loaded */ }
  }

  Position pos; position_startpos(pos);
  std::vector<U64> hist; hist.push_back(pos.hash);

  InputQueue iq;
  std::thread reader([&]() {
    std::string line;
    while (std::getline(std::cin, line)) {
      if (!line.empty() && line.back() == '\r') { line.pop_back();
}
      iq.push(line);
      if (line == "quit") { break;
}
    }
    { std::lock_guard<std::mutex> l(iq.m); iq.closed = true; }
    iq.cv.notify_all();
  });

  std::atomic<bool> searching{false};
  std::atomic<bool> stop_flag{false};
  std::thread search_th;
  SearchInfo last_info;

  auto wait_search_done = [&]() {
    if (search_th.joinable()) { search_th.join();
}
    searching.store(false);
  };

  bool quit = false;
  while (!quit) {
    std::string line;
    {
      std::unique_lock<std::mutex> l(iq.m);
      iq.cv.wait(l, [&]() { return !iq.q.empty() || iq.closed; });
      if (iq.q.empty() && iq.closed) { break;
}
      line = iq.q.front(); iq.q.pop();
    }
    auto tok = split_ws(line);
    if (tok.empty()) { continue;
}
    const std::string& cmd = tok[0];

    if (cmd == "quit") { quit = true; stop_flag.store(true); }
    else if (cmd == "uci") {
      std::cout << "id name Forge 1.0 (pure C++17, AVX2, NNUE)\n";
      std::cout << "id author Forge contributors\n";
      std::cout << "option name Hash type spin default 64 min 1 max 2048\n";
      std::cout << "option name Threads type spin default " << opt.threads << " min 1 max 64\n";
      std::cout << "option name NNUEFile type string default <empty>\n";
      std::cout << "uciok\n" << std::flush;
    }
    else if (cmd == "isready") { std::cout << "readyok\n" << std::flush; }
    else if (cmd == "setoption") {
      // setoption name X value Y
      std::string name, value;
      for (size_t i = 1; i < tok.size(); i++) {
        if (tok[i] == "name" && i + 1 < tok.size()) {
          i++;
          while (i < tok.size() && tok[i] != "value") { if (!name.empty()) { name += " "; 
}name += tok[i++]; }
          if (i < tok.size() && tok[i] == "value") {
            i++;
            while (i < tok.size()) { if (!value.empty()) { value += " "; 
}value += tok[i++]; }
            break;
          } break;
        }
      }
      std::string nl = name; std::transform(nl.begin(), nl.end(), nl.begin(), ::tolower);
      if (nl == "hash") { opt.hash_mb = std::max(1, atoi(value.c_str())); tt.resize((size_t)opt.hash_mb); }
      else if (nl == "threads") { opt.threads = std::max(1, std::min(64, atoi(value.c_str()))); }
      else if (nl == "nnuefile") {
        opt.nnue_file = value;
        if (!value.empty() && value != "<empty>") {
          NNUE tmp; if (nnue_load(tmp, value)) { net = tmp;
}
        }
      }
    }
    else if (cmd == "ucinewgame") { tt.clear(); hist.clear(); position_startpos(pos); hist.push_back(pos.hash); }
    else if (cmd == "position") {
      // don't allow position change while searching (queue it after stop)
      if (searching.load()) { stop_flag.store(true); wait_search_done(); stop_flag.store(false); }
      parse_position_cmd(pos, hist, line);
    }
    else if (cmd == "go") {
      if (searching.load()) { stop_flag.store(true); wait_search_done(); }
      stop_flag.store(false);
      Limits lim = parse_go_cmd(line, pos);
      // adjust for overhead
      if (lim.movetime_ms > opt.move_overhead_ms + 5) { lim.movetime_ms -= opt.move_overhead_ms;
}
      Position root = pos; // copy for thread
      searching.store(true);
      search_th = std::thread([&, root, lim]() mutable {
        // iterative deepening with info prints (main thread only prints)
        Searcher s; s.tt = &tt; s.net = &net;
        // custom ID loop to stream info lines (reuse Searcher::search internals is easier:
        // call search_smp but without streaming; then print final. For live PV streaming
        // we do our own shallow wrapper here.)
        // Simplest strong behavior: use search_smp and print summary infos.
        // Live depth streaming:
        s.t0 = std::chrono::steady_clock::now();
        // We'll run depths manually to emit info:
        s.stop = &stop_flag;
        s.nodes = 0;
        if (s.tt) { s.tt->new_search();
}
        int64_t budget = alloc_movetime(root, lim);
        if (lim.movetime_ms > 0) { s.hard_ms = lim.movetime_ms; s.soft_ms = lim.movetime_ms; }
        else if (lim.infinite || (lim.wtime == 0 && lim.btime == 0)) { s.hard_ms = INT64_MAX; s.soft_ms = INT64_MAX; }
        else {
          s.hard_ms = (int64_t)(budget * 5.0); s.soft_ms = budget;
        }
        if (lim.nodes > 0) { s.node_limit = lim.nodes;
}
        s.max_depth = lim.depth;
        memset(s.history, 0, sizeof(s.history));
        memset(s.killers, 0, sizeof(s.killers));
        MoveList root_moves; gen_legal(root, root_moves, false);
        if (root_moves.n == 0) {
          std::cout << "bestmove 0000\n" << std::flush;
          searching.store(false);
          return;
        }
        if (root_moves.n == 1) {
          std::cout << "bestmove " << move_to_uci(root_moves.m[0]) << "\n" << std::flush;
          searching.store(false);
          return;
        }
        Score prev = evaluate(root, net, nullptr);
        Move best_move = root_moves.m[0];
        Score best_score = prev;
        NNUEAcc racc; nnue_refresh(net, root, racc);
        int maxd = lim.depth > 0 ? lim.depth : 64;
        for (int depth = 1; depth <= maxd; depth++) {
          if (stop_flag.load()) { break;
}
          // use full Searcher::negamax via a temporary root PVS (duplicate of search.cpp logic, thin)
          // To avoid divergence, delegate single-depth via search_smp-lite: run negamax per move.
          Score alpha = -INF, beta = INF;
          if (depth >= 5) { alpha = best_score - 25; beta = best_score + 25; }
          Score cur_best = -INF; Move cur_move = best_move;
          // order: TT best first
          for (int i = 0; i < root_moves.n; i++) {
            if (root_moves.m[i] == best_move) { std::swap(root_moves.m[0], root_moves.m[i]); break; }
          }
          for (int i = 0; i < root_moves.n; i++) {
            // selection by order_score
            int bi = i, bs = -1000000000;
            for (int j = i; j < root_moves.n; j++) {
              // quick order: history
              int sc2 = 0;
              if (root_moves.m[j] == best_move) { sc2 = 10000000;
              } else {
                Piece pc = root.board_pc[move_from(root_moves.m[j])];
                Piece vc = root.board_pc[move_to(root_moves.m[j])];
                if (vc != NO_PIECE) { sc2 = 100000 + see_value(vc) * 10;
                } else if (pc != NO_PIECE) { sc2 = s.history[root.side][move_from(root_moves.m[j])][move_to(root_moves.m[j])];
}
              }
              if (sc2 > bs) { bs = sc2; bi = j; }
            }
            std::swap(root_moves.m[i], root_moves.m[bi]);
            Move m = root_moves.m[i];
            Position np = root; Undo u; NNUEAcc nacc = racc;
            if (!make_move(np, m, u)) { continue;
}
            if (in_check(np, root.side)) { continue;
}
            nnue_update(net, root, m, u, nacc);
            Score sc;
            if (i == 0) { sc = -s.negamax(np, -beta, -alpha, depth - 1, 1, false, nacc);
            } else {
              sc = -s.negamax(np, -(alpha + 1), -alpha, depth - 1, 1, true, nacc);
              if (!stop_flag.load() && sc > alpha && sc < beta) {
                sc = -s.negamax(np, -beta, -alpha, depth - 1, 1, false, nacc);
}
            }
            if (stop_flag.load()) { break;
}
            if (sc > cur_best) { cur_best = sc; cur_move = m; }
            if (sc > alpha) { alpha = sc;
}
            if (alpha >= beta) { break;
}
          }
          if (stop_flag.load()) { break;
}
          best_score = cur_best; best_move = cur_move;
          last_info.best = best_move; last_info.score = best_score;
          last_info.depth = depth; last_info.nodes = s.nodes;
          last_info.time_ms = s.elapsed_ms();
          // info line
          int64_t ms = last_info.time_ms;
          uint64_t nps = ms > 0 ? (s.nodes * 1000 / (uint64_t)ms) : s.nodes;
          std::cout << "info depth " << depth << " score ";
          if (std::abs(best_score) > MATE - 1000) {
            int plies = mate_in_plies(best_score);
            int moves = (plies + 1) / 2 * (best_score > 0 ? 1 : -1);
            std::cout << "mate " << moves;
          } else { std::cout << "cp " << best_score;
}
          std::cout << " nodes " << s.nodes << " nps " << nps
                    << " time " << ms << " hashfull " << tt.hashfull()
                    << " pv " << move_to_uci(best_move) << "\n" << std::flush;
          if ((int64_t)s.nodes >= (lim.nodes > 0 ? lim.nodes : INT64_MAX)) { break;
}
          if (s.elapsed_ms() >= s.soft_ms && depth >= 4) { break;
}
          if (std::abs(best_score) > MATE - 1000) { break;
}
        }
        // If SMP requested, helpers already warmed TT during negamax? (single-thread here for
        // streaming correctness; multi-thread warming happens inside negamax? No—this path is
        // single-threaded for clean PVs. For Threads>1 we still benefit from big TT + fast eval.
        // Full Lazy SMP fan-out is used by forge-train self-play via search_smp.)
        std::cout << "bestmove " << move_to_uci(best_move) << "\n" << std::flush;
        searching.store(false);
      });
    }
    else if (cmd == "stop") {
      stop_flag.store(true);
      wait_search_done();
      stop_flag.store(false);
    }
    else if (cmd == "d") { print_board_pretty(pos); }
    else if (cmd == "eval") {
      NNUEAcc acc; nnue_refresh(net, pos, acc);
      int cl = evaluate_classical(pos);
      std::cout << "classical (stm): " << cl << " cp\n";
      if (net.trained) { std::cout << "nnue (stm): " << nnue_evaluate(net, acc, pos.side) << " cp\n";
      } else { std::cout << "nnue: <untrained, using classical>\n";
}
      std::cout << "blended (stm): " << evaluate(pos, net, &acc) << " cp\n" << std::flush;
    }
    else if (cmd == "perft") {
      int d = tok.size() > 1 ? atoi(tok[1].c_str()) : 4;
      if (d < 1) { d = 1; 
}if (d > 7) { d = 7;
}
      Position tmp = pos;
      auto t0 = std::chrono::steady_clock::now();
      // divide
      MoveList ml; gen_legal(tmp, ml, false);
      uint64_t total = 0;
      for (int i = 0; i < ml.n; i++) {
        Position np = tmp; Undo u;
        if (!make_move(np, ml.m[i], u)) { continue;
}
        if (in_check(np, tmp.side)) { continue;
}
        uint64_t c = (d == 1) ? 1 : perft(np, d - 1);
        std::cout << move_to_uci(ml.m[i]) << ": " << c << "\n";
        total += c;
      }
      auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::steady_clock::now() - t0).count();
      std::cout << "\nNodes: " << total << " in " << ms << " ms\n" << std::flush;
    }
  }

  stop_flag.store(true);
  wait_search_done();
  if (reader.joinable()) {
    // reader is blocked on getline; detach on quit (process exits anyway)
    // try to join briefly by closing stdin? Just detach.
    reader.detach();
  }
}

} // namespace forge
