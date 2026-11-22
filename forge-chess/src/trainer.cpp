// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Vaibhav Agarwal

// forge-chess — trainer.cpp : self-play + Adam NNUE training, progress bar, lossless ckpt
#include "trainer.h"
#include "cli.h"
#include "uci.h"
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <sstream>
#include <iostream>
#include <thread>
#include <mutex>
#include <random>
#include <algorithm>
#include <atomic>
#include <chrono>
#include <csignal>
#include <utility>

namespace forge {

double elo_from_score(double s) {
  if (s <= 0.001) { s = 0.001;
}
  if (s >= 0.999) { s = 0.999;
}
  return -400.0 * std::log10(1.0 / s - 1.0);
}
double score_from_elo(double e) { return 1.0 / (1.0 + std::pow(10.0, -e / 400.0)); }

static Trainer* g_trainer = nullptr;
static void sig_handler(int) { if (g_trainer) { g_trainer->stop_requested.store(true); 
}}

int Trainer::total_params() const {
  return (int)((size_t)NNUE_INPUT * NNUE_HIDDEN + NNUE_HIDDEN + (size_t)NNUE_HIDDEN * 2 + 1);
}
void Trainer::ensure_adam() {
  int n = total_params();
  if ((int)adam_m.size() != n) { adam_m.assign(n, 0.0f); adam_v.assign(n, 0.0f); }
}

// ---------- checkpoint ----------
// Files: <ckpt>.state (text k=v), <ckpt>.adam (binary m+v+t), <ckpt>.net (nnue), <ckpt>.samples (tsv)
bool Trainer::save_checkpoint() {
  std::string s_path = cfg.checkpoint + ".state";
  std::string a_path = cfg.checkpoint + ".adam";
  std::string n_path = cfg.checkpoint + ".net";
  auto write_and_rename = [&](const std::string& path, const std::string& data) {
    std::string tmp = path + ".tmp";
    FILE* f = fopen(tmp.c_str(), "wb");
    if (!f) { return false;
}
    fwrite(data.data(), 1, data.size(), f);
    fclose(f);
    std::rename(tmp.c_str(), path.c_str());
    return true;
  };
  char buf[2048];
  snprintf(buf, sizeof(buf),
    "v=1\ngames_done=%d\nsamples_done=%lld\ncycle=%d\nepoch=%d\nbatch_idx=%d\n"
    "rng_state=%llu\nwins=%d\ndraws=%d\nlosses=%d\ntotal_w=%d\ntotal_d=%d\ntotal_l=%d\n"
    "elo=%.3f\nloss=%.6f\nadam_t=%llu\nlr=%.6f\nlambda=%.4f\ntrained=%d\nbest_trained=%d\n",
    st.games_done, (long long)st.samples_done, st.cycle, st.epoch, st.batch_idx,
    (unsigned long long)st.rng_state, st.wins, st.draws, st.losses,
    st.total_w, st.total_d, st.total_l, st.elo, st.loss,
    (unsigned long long)st.adam_t, cfg.lr, cfg.lambda,
    net.trained ? 1 : 0, best_net.trained ? 1 : 0);
  if (!write_and_rename(s_path, buf)) { return false;
}
  // adam
  {
    std::string tmp = a_path + ".tmp";
    FILE* f = fopen(tmp.c_str(), "wb");
    if (!f) { return false;
}
    uint64_t t = st.adam_t;
    fwrite(&t, 8, 1, f);
    int n = total_params();
    ensure_adam();
    fwrite(adam_m.data(), 4, n, f);
    fwrite(adam_v.data(), 4, n, f);
    fclose(f);
    std::rename(tmp.c_str(), a_path.c_str());
  }
  // nets
  nnue_save(net, n_path + ".tmp");
  std::rename((n_path + ".tmp").c_str(), n_path.c_str());
  nnue_save(best_net, cfg.checkpoint + ".best.tmp");
  std::rename((cfg.checkpoint + ".best.tmp").c_str(), (cfg.checkpoint + ".best").c_str());
  return true;
}

bool Trainer::load_checkpoint() {
  std::string s_path = cfg.checkpoint + ".state";
  FILE* f = fopen(s_path.c_str(), "rb");
  if (!f) { return false;
}
  char data[4096] = {}; size_t n = fread(data, 1, sizeof(data) - 1, f);
  fclose(f);
  if (!n) { return false;
}
  std::string s(data);
  auto get = [&](const char* k, const char* def = "0") {
    std::string key = std::string("\n") + k + "=";
    // also match start
    size_t p = s.find(key);
    if (p == std::string::npos) {
      std::string k2 = std::string(k) + "=";
      if (s.rfind(k2, 0) == 0) { p = (size_t)-1;
      } else { return std::string(def);
}
    }
    size_t v0 = (p == (size_t)-1) ? strlen(k) + 1 : p + key.size();
    size_t v1 = s.find('\n', v0);
    return s.substr(v0, v1 == std::string::npos ? std::string::npos : v1 - v0);
  };
  try {
    st.games_done = std::stoi(get("games_done"));
    st.samples_done = std::stoll(get("samples_done"));
    st.cycle = std::stoi(get("cycle"));
    st.epoch = std::stoi(get("epoch"));
    st.batch_idx = std::stoi(get("batch_idx"));
    st.rng_state = std::stoull(get("rng_state"));
    st.wins = std::stoi(get("wins")); st.draws = std::stoi(get("draws")); st.losses = std::stoi(get("losses"));
    st.total_w = std::stoi(get("total_w")); st.total_d = std::stoi(get("total_d")); st.total_l = std::stoi(get("total_l"));
    st.elo = std::stod(get("elo")); st.loss = std::stod(get("loss"));
    st.adam_t = std::stoull(get("adam_t"));
  } catch (...) { return false; }
  // adam
  {
    std::string a_path = cfg.checkpoint + ".adam";
    FILE* fa = fopen(a_path.c_str(), "rb");
    if (fa) {
      uint64_t t = 0; fread(&t, 8, 1, fa);
      int m = total_params(); ensure_adam();
      size_t r1 = fread(adam_m.data(), 4, m, fa);
      size_t r2 = fread(adam_v.data(), 4, m, fa);
      fclose(fa);
      if ((int)r1 != m || (int)r2 != m) { return false;
}
      st.adam_t = t;
    }
  }
  nnue_load(net, cfg.checkpoint + ".net");
  nnue_load(best_net, cfg.checkpoint + ".best");
  return true;
}

// ---------- data helpers ----------
static bool parse_tsv_line(const std::string& line, Sample& out) {
  // FEN\tcp\tresult  (result white POV)
  size_t t1 = line.rfind('\t');
  if (t1 == std::string::npos) { return false;
}
  size_t t2 = line.rfind('\t', t1 == 0 ? 0 : t1 - 1);
  // FEN itself contains no tabs, so first tab separates FEN
  size_t f = line.find('\t');
  if (f == std::string::npos) { return false;
}
  out.fen = line.substr(0, f);
  std::string rest = line.substr(f + 1);
  // rest: "cp\tresult"
  size_t t = rest.find('\t');
  if (t == std::string::npos) { return false;
}
  try {
    out.teacher_cp = std::stof(rest.substr(0, t));
    out.result = std::stof(rest.substr(t + 1));
  } catch (...) { return false; }
  return true;
}

static std::vector<Sample> load_tsv(const std::string& path, size_t max_n = SIZE_MAX) {
  std::vector<Sample> v;
  std::ifstream in(path);
  if (!in) { return v;
}
  std::string line;
  while (std::getline(in, line)) {
    if (line.empty()) { continue;
}
    Sample s;
    if (!parse_tsv_line(line, s)) { continue;
}
    v.push_back(std::move(s));
    if (v.size() >= max_n) { break;
}
  }
  return v;
}

// ---------- self-play ----------
struct GameResult {
  float result_white = 0.5f; // 1/0.5/0
  std::vector<Sample> samples; // teacher scores stm->white converted at end
  int plies = 0;
};

static uint64_t splitmix(uint64_t& s) {
  uint64_t z = (s += 0x9E3779B97F4A7C15ULL);
  z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
  z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
  return z ^ (z >> 31);
}

static GameResult play_game(const NNUE& net, int depth, int64_t nodes_per_move,
                            int movetime_ms, uint64_t seed, int max_plies = 220) {
  GameResult gr;
  Position p; position_startpos(p);
  TT tt; tt.resize(8);
  std::vector<U64> hashes; hashes.push_back(p.hash);

  // random opening: 4-8 random plies
  uint64_t rng = seed;
  int opening_plies = 4 + (int)(splitmix(rng) % 5);
  for (int i = 0; i < opening_plies; i++) {
    MoveList ml; gen_legal(p, ml, false);
    if (ml.n == 0) { break;
}
    Move m = ml.m[splitmix(rng) % (uint64_t)ml.n];
    Undo u; make_move(p, m, u);
    hashes.push_back(p.hash);
  }

  struct PlyInfo { std::string fen; float teacher_white; };
  std::vector<PlyInfo> infos;

  int consec_big = 0;
  for (int ply = 0; ply < max_plies; ply++) {
    if (insufficient_material(p)) { break;
}
    if (p.half >= 100) { break;
}
    // 3fold
    int reps = 0;
    for (auto h : hashes) { if (h == p.hash) { reps++;
}
}
    if (reps >= 3) { break;
}

    MoveList ml; gen_legal(p, ml, false);
    if (ml.n == 0) { break; // mate/stalemate handled below
}

    Limits lim; lim.depth = depth;
    if (nodes_per_move > 0) { lim.nodes = nodes_per_move; lim.depth = 64; }
    if (movetime_ms > 0) { lim.movetime_ms = movetime_ms; lim.depth = 64; }
    std::atomic<bool> stop{false};
    Searcher s; s.tt = &tt; s.net = &net;
    SearchInfo info = s.search(p, lim, &stop, nullptr);
    Move best = info.best;
    if (best == MOVE_NONE) { best = ml.m[0];
}
    auto teacher_stm = (float)info.score;
    if (teacher_stm > 10000) { teacher_stm = 10000;
}
    if (teacher_stm < -10000) { teacher_stm = -10000;
}
    float teacher_white = (p.side == WHITE) ? teacher_stm : -teacher_stm;

    // resign adjudication
    if (std::abs(teacher_stm) > 900) { consec_big++;
    } else { consec_big = 0;
}
    if (consec_big >= 6 && ply > 20) { break; // losing side resigns; result set below
}

    // record sample every 2 plies after opening
    if (ply >= 4 && (ply % 2 == 0)) {
      infos.push_back({position_to_fen(p), teacher_white});
    }

    Undo u; make_move(p, best, u);
    hashes.push_back(p.hash);
  }

  // result
  MoveList ml; gen_legal(p, ml, false);
  float res = 0.5f;
  if (ml.n == 0) {
    if (in_check(p, p.side)) { res = (p.side == WHITE) ? 0.0f : 1.0f;
    } else { res = 0.5f;
}
  } else if (consec_big >= 6) {
    // last teacher decides: recompute quick eval sign
    int e = evaluate(p, net, nullptr);
    // e is stm POV; white POV = stm==WHITE? e : -e
    float w = (p.side == WHITE) ? (float)e : -(float)e;
    res = (w > 0) ? 0.0f : 1.0f; // side to move is losing -> opponent wins
    // NOTE: if white to move and losing (w<0) => black wins => res=0. Correct above? w>0 means stm winning.
    // stm==side to move; if stm winning, stm gets win.
    if (w > 0) { res = (p.side == WHITE) ? 1.0f : 0.0f;
    } else { res = (p.side == WHITE) ? 0.0f : 1.0f;
}
  } else {
    // adjudicate by eval if game hit move limit / fifty
    int e = evaluate(p, net, nullptr);
    float w = (p.side == WHITE) ? (float)e : -(float)e;
    if (w > 500) { res = 1.0f;
    } else if (w < -500) { res = 0.0f;
    } else { res = 0.5f;
}
  }
  gr.result_white = res;
  gr.plies = (int)hashes.size();
  gr.samples.reserve(infos.size());
  for (auto& in : infos) {
    Sample s; s.fen = in.fen; s.teacher_cp = in.teacher_white; s.result = res;
    gr.samples.push_back(std::move(s));
  }
  return gr;
}

// ---------- training math ----------
static inline float sigmoidf(float x) { return 1.0f / (1.0f + std::exp(-x)); }

// Parse FEN board into active feature lists (white-acc indices + black-acc indices)
static void fen_features(const std::string& fen, std::vector<int>& fw, std::vector<int>& fb) {
  fw.clear(); fb.clear();
  // board part only
  size_t sp = fen.find(' ');
  std::string board = (sp == std::string::npos) ? fen : fen.substr(0, sp);
  int r = 7, f = 0;
  for (char ch : board) {
    if (ch == '/') { r--; f = 0; continue; }
    if (ch >= '1' && ch <= '8') { f += ch - '0'; continue; }
    Color c = (ch >= 'A' && ch <= 'Z') ? WHITE : BLACK;
    Piece pc = NO_PIECE;
    char l = (char)tolower(ch);
    if (l == 'p') { pc = PAWN; } else if (l == 'n') { pc = KNIGHT; } else if (l == 'b') { pc = BISHOP;
    } else if (l == 'r') { pc = ROOK; } else if (l == 'q') { pc = QUEEN; } else if (l == 'k') { pc = KING;
}
    if (pc == NO_PIECE || f > 7 || r < 0) { continue;
}
    Square sq = make_sq(f, r);
    fw.push_back(nnue_idx_white(pc, c, sq));
    fb.push_back(nnue_idx_black(pc, c, sq));
    f++;
  }
}
static Color fen_stm(const std::string& fen) {
  // second field
  size_t a = fen.find(' ');
  if (a == std::string::npos) { return WHITE;
}
  size_t b = fen.find(' ', a + 1);
  std::string stm = fen.substr(a + 1, b == std::string::npos ? 1 : b - a - 1);
  return (stm == "b" || stm == "B") ? BLACK : WHITE;
}

// One Adam training run over samples (resumable via st.epoch/batch_idx + deterministic shuffle)
static double train_epochs(Trainer& tr, std::vector<Sample>& data) {
  using namespace cli;
  if (data.empty()) { return 0.0;
}
  tr.ensure_adam();
  const int H = NNUE_HIDDEN;
  const size_t W1N = (size_t)NNUE_INPUT * H;
  const int B = tr.cfg.batch;
  const float lr = tr.cfg.lr;
  const float lambda = tr.cfg.lambda;
  const float b1 = 0.9f, b2 = 0.999f, eps = 1e-8f;

  int n = tr.total_params();
  std::vector<float> grad(n);

  std::vector<int> fw; fw.reserve(32);
  std::vector<int> fb; fb.reserve(32);
  float acc_w[256], acc_b[256];

  int start_cycle_epoch = tr.st.epoch;
  for (int ep = start_cycle_epoch; ep < tr.cfg.epochs_per_cycle; ep++) {
    tr.st.epoch = ep;
    // deterministic shuffle for lossless resume
    std::vector<int> order(data.size());
    for (size_t i = 0; i < order.size(); i++) { order[i] = (int)i;
}
    std::mt19937_64 rng((uint64_t)tr.cfg.seed + (uint64_t)tr.st.cycle * 100003ULL + (uint64_t)ep * 1009ULL);
    std::shuffle(order.begin(), order.end(), rng);

    int nbatches = (int)((data.size() + B - 1) / B);
    int start_b = (ep == start_cycle_epoch) ? tr.st.batch_idx : 0;
    Bar bar; bar.label = "epoch " + std::to_string(ep + 1) + "/" + std::to_string(tr.cfg.epochs_per_cycle);
    double run_loss = 0; int run_n = 0;
    for (int bi = start_b; bi < nbatches; bi++) {
      if (tr.stop_requested.load()) { tr.st.batch_idx = bi; tr.save_checkpoint(); return run_n ? run_loss / run_n : 0; }
      tr.st.batch_idx = bi;
      int b0 = bi * B, b1e = std::min<int>((int)data.size(), b0 + B);
      std::fill(grad.begin(), grad.end(), 0.0f);
      double batch_loss = 0;
      for (int k = b0; k < b1e; k++) {
        const Sample& s = data[(size_t)order[k]];
        fen_features(s.fen, fw, fb);
        Color stm = fen_stm(s.fen);
        // forward acc
        for (int i = 0; i < H; i++) { acc_w[i] = tr.net.b1[i]; acc_b[i] = tr.net.b1[i]; }
        for (int fidx : fw) {
          const float* r = &tr.net.w1[(size_t)fidx * H];
          for (int i = 0; i < H; i++) { acc_w[i] += r[i];
}
        }
        for (int fidx : fb) {
          const float* r = &tr.net.w1[(size_t)fidx * H];
          for (int i = 0; i < H; i++) { acc_b[i] += r[i];
}
        }
        float cw[256], cb[256];
        for (int i = 0; i < H; i++) {
          float a = acc_w[i]; cw[i] = a < 0 ? 0 : (a > NNUE_QA_CLIP ? NNUE_QA_CLIP : a);
          float bb = acc_b[i]; cb[i] = bb < 0 ? 0 : (bb > NNUE_QA_CLIP ? NNUE_QA_CLIP : bb);
        }
        const float* own = (stm == WHITE) ? cw : cb;
        const float* opp = (stm == WHITE) ? cb : cw;
        float pred = tr.net.b2;
        for (int i = 0; i < H; i++) { pred += own[i] * tr.net.w2[i] + opp[i] * tr.net.w2[H + i];
}
        float p = sigmoidf(pred / 400.0f);
        float yt = sigmoidf(s.teacher_cp / 400.0f);
        float y = lambda * yt + (1 - lambda) * s.result;
        // white-POV result vs stm? samples store white POV result + white POV teacher.
        // Convert to stm POV:
        if (stm == BLACK) { y = 1.0f - y; }
        float err = p - y;
        batch_loss += (double)(err * err);
        float dLdpred = 2.0f * err * p * (1 - p) / 400.0f;
        // grads w2/b2
        float* gw2 = grad.data() + W1N + H;
        for (int i = 0; i < H; i++) { gw2[i] += dLdpred * own[i]; gw2[H + i] += dLdpred * opp[i]; }
        grad[n - 1] += dLdpred;
        // backprop to acc pre-activations
        // dL/da_w[i] etc.
        // own corresponds to (stm==W? acc_w : acc_b), opp the other.
        for (int i = 0; i < H; i++) {
          float d_own = dLdpred * tr.net.w2[i];
          float d_opp = dLdpred * tr.net.w2[H + i];
          float d_aw, d_ab;
          if (stm == WHITE) { d_aw = d_own; d_ab = d_opp; }
          else { d_aw = d_opp; d_ab = d_own; }
          // clip derivative
          if (acc_w[i] <= 0 || acc_w[i] >= NNUE_QA_CLIP) { d_aw = 0;
}
          if (acc_b[i] <= 0 || acc_b[i] >= NNUE_QA_CLIP) { d_ab = 0;
}
          if (d_aw == 0 && d_ab == 0) { continue;
}
          // b1 grad (shared)
          grad[W1N + i] += d_aw + d_ab;
          // w1 grads
          float* gw1 = grad.data();
          if (d_aw) { for (int fidx : fw) { gw1[(size_t)fidx * H + i] += d_aw;
}
}
          if (d_ab) { for (int fidx : fb) { gw1[(size_t)fidx * H + i] += d_ab;
}
}
        }
      }
      batch_loss /= (b1e - b0);
      run_loss += batch_loss * (b1e - b0); run_n += (b1e - b0);
      // average grads
      float inv = 1.0f / (b1e - b0);
      // Adam update
      tr.st.adam_t++;
      uint64_t t = tr.st.adam_t;
      float bc1 = 1.0f - std::pow(b1, (float)t);
      float bc2 = 1.0f - std::pow(b2, (float)t);
      float step = lr * std::sqrt(bc2) / bc1;
      float* params = nullptr; // update via net vectors directly
      // flatten view: [w1][b1][w2][b2]
      // update w1
      for (size_t i = 0; i < W1N; i++) {
        float g = grad[i] * inv;
        tr.adam_m[i] = b1 * tr.adam_m[i] + (1 - b1) * g;
        tr.adam_v[i] = b2 * tr.adam_v[i] + (1 - b2) * g * g;
        tr.net.w1[i] -= step * tr.adam_m[i] / (std::sqrt(tr.adam_v[i]) + eps);
      }
      for (int i = 0; i < H; i++) {
        float g = grad[W1N + i] * inv;
        int idx = (int)W1N + i;
        tr.adam_m[idx] = b1 * tr.adam_m[idx] + (1 - b1) * g;
        tr.adam_v[idx] = b2 * tr.adam_v[idx] + (1 - b2) * g * g;
        tr.net.b1[i] -= step * tr.adam_m[idx] / (std::sqrt(tr.adam_v[idx]) + eps);
      }
      for (int i = 0; i < 2 * H; i++) {
        float g = grad[W1N + H + i] * inv;
        int idx = (int)W1N + H + i;
        tr.adam_m[idx] = b1 * tr.adam_m[idx] + (1 - b1) * g;
        tr.adam_v[idx] = b2 * tr.adam_v[idx] + (1 - b2) * g * g;
        tr.net.w2[i] -= step * tr.adam_m[idx] / (std::sqrt(tr.adam_v[idx]) + eps);
      }
      {
        float g = grad[n - 1] * inv;
        tr.adam_m[n - 1] = b1 * tr.adam_m[n - 1] + (1 - b1) * g;
        tr.adam_v[n - 1] = b2 * tr.adam_v[n - 1] + (1 - b2) * g * g;
        tr.net.b2 -= step * tr.adam_m[n - 1] / (std::sqrt(tr.adam_v[n - 1]) + eps);
      }
      (void)params;
      char extra[160];
      snprintf(extra, sizeof(extra), "loss %.5f  lr %.4f", run_loss / run_n, lr);
      bar.draw((double)(bi + 1) / nbatches, extra);
      if ((bi + 1) % tr.cfg.ckpt_every_batches == 0) { tr.save_checkpoint();
}
    }
    bar.done("loss " + std::to_string(run_loss / run_n));
    tr.st.loss = run_n ? run_loss / run_n : 0;
    tr.st.batch_idx = 0; // next epoch starts at 0
    tr.save_checkpoint();
    if (tr.stop_requested.load()) { return tr.st.loss;
}
  }
  tr.st.epoch = 0; tr.st.batch_idx = 0;
  return tr.st.loss;
}

// ---------- validation: new vs old ----------
static void validation_match(const NNUE& a, const NNUE& b, int games, int depth,
                             int64_t npm, int movetime, int threads, uint64_t seed,
                             int& w, int& d, int& l, std::atomic<bool>* stop) {
  w = d = l = 0;
  std::mutex m;
  std::atomic<int> next{0};
  auto worker = [&]() {
    while (true) {
      int g = next.fetch_add(1);
      if (g >= games) { break;
}
      if (stop && stop->load()) { break;
}
      // alternate colors
      bool a_white = (g % 2 == 0);
      const NNUE& wnet = a_white ? a : b;
      const NNUE& bnet = a_white ? b : a;
      Position p; position_startpos(p);
      TT ttw; ttw.resize(8); TT ttb; ttb.resize(8);
      std::vector<U64> hh; hh.push_back(p.hash);
      uint64_t rng = seed + (uint64_t)g * 7919ULL;
      int ops = 4 + (int)(splitmix(rng) % 5);
      for (int i = 0; i < ops; i++) {
        MoveList ml; gen_legal(p, ml, false);
        if (!ml.n) { break;
}
        Move m = ml.m[splitmix(rng) % ml.n];
        Undo u; make_move(p, m, u); hh.push_back(p.hash);
      }
      int consec = 0;
      float res = 0.5f;
      bool decided = false;
      for (int ply = 0; ply < 200 && !decided; ply++) {
        if (insufficient_material(p) || p.half >= 100) { break;
}
        int reps = 0; for (auto h : hh) { if (h == p.hash) { reps++;
}
}
        if (reps >= 3) { break;
}
        MoveList ml; gen_legal(p, ml, false);
        if (ml.n == 0) {
          if (in_check(p, p.side)) { res = (p.side == WHITE) ? 0.0f : 1.0f;
          } else { res = 0.5f;
}
          decided = true; break;
        }
        const NNUE& side_net = (p.side == WHITE) ? wnet : bnet;
        TT& side_tt = (p.side == WHITE) ? ttw : ttb;
        Limits lim; lim.depth = depth;
        if (npm > 0) { lim.nodes = npm; lim.depth = 64; }
        if (movetime > 0) { lim.movetime_ms = movetime; lim.depth = 64; }
        std::atomic<bool> s{false};
        Searcher sr; sr.tt = &side_tt; sr.net = &side_net;
        SearchInfo info = sr.search(p, lim, &s, nullptr);
        Move bm = info.best == MOVE_NONE ? ml.m[0] : info.best;
        if (std::abs(info.score) > 900) { consec++; } else { consec = 0;
}
        if (consec >= 6 && ply > 20) {
          int e = info.score; // stm POV
          bool stm_winning = e > 0;
          bool white_stm = (p.side == WHITE);
          bool white_wins = (stm_winning == white_stm);
          res = white_wins ? 1.0f : 0.0f;
          decided = true; break;
        }
        Undo u; make_move(p, bm, u); hh.push_back(p.hash);
      }
      if (!decided) {
        int e = evaluate(p, (p.side == WHITE ? wnet : bnet), nullptr);
        float wv = (p.side == WHITE) ? (float)e : -(float)e;
        if (wv > 400) { res = 1.0f; } else if (wv < -400) { res = 0.0f; } else { res = 0.5f;
}
      }
      bool a_won = (a_white && res == 1.0f) || (!a_white && res == 0.0f);
      bool drawn = (res == 0.5f);
      std::lock_guard<std::mutex> lk(m);
      if (drawn) { d++; } else if (a_won) { w++; } else { l++;
}
    }
  };
  int nthreads = threads > 0 ? threads : 1;
  std::vector<std::thread> th;
  th.reserve(nthreads);
for (int i = 0; i < nthreads; i++) { th.emplace_back(worker);
}
  for (auto& t : th) { t.join();
}
}

int Trainer::run() {
  using namespace cli;
  position_init();
  g_trainer = this;
  std::signal(SIGINT, sig_handler);
  std::signal(SIGTERM, sig_handler);

  header("FORGE TRAINER — self-play + NNUE");
  bool resumed = load_checkpoint();
  if (resumed) {
    kv("resumed checkpoint", cfg.checkpoint, green());
    kv("games done", std::to_string(st.games_done));
    kv("cycle / epoch / batch",
       std::to_string(st.cycle) + " / " + std::to_string(st.epoch) + " / " + std::to_string(st.batch_idx));
    kv("elo so far", elo_str(st.elo), yellow());
  } else {
    kv("starting fresh", cfg.out_net, green());
    net.init_random((uint64_t)cfg.seed);
    best_net = net;
    st.rng_state = (uint64_t)cfg.seed ^ 0x9E3779B97F4A7C15ULL;
    ensure_adam();
  }

  int threads = cfg.threads > 0 ? cfg.threads : default_threads();
  kv("threads", std::to_string(threads));
  kv("games target", std::to_string(cfg.games));
  kv("cycles", std::to_string(cfg.cycles));
  kv("search depth", std::to_string(cfg.search_depth));
  kv("batch / lr / lambda", std::to_string(cfg.batch) + " / " + std::to_string(cfg.lr) +
                             " / " + std::to_string(cfg.lambda));
  kv("output net", cfg.out_net);
  kv("checkpoint", cfg.checkpoint + ".{state,adam,net,best,samples}");
  std::cout << "\n  " << dim() << "Press Ctrl+C once to pause losslessly — resume with the same command."
            << reset() << "\n\n";

  int games_per_cycle = std::max(1, cfg.games / std::max(1, cfg.cycles));
  int start_cycle = st.cycle;

  for (int cyc = start_cycle; cyc < cfg.cycles; cyc++) {
    if (stop_requested.load()) { save_checkpoint(); std::cout << "\n  Paused. Resume to continue.\n"; return 130; }
    st.cycle = cyc;
    std::cout << bold() << cyan() << "\n  ── cycle " << (cyc + 1) << "/" << cfg.cycles
              << reset() << "\n";

    // ---- self-play ----
    int need = games_per_cycle;
    // if resuming mid-cycle with games already partially done for this cycle, adjust:
    // st.games_done counts lifetime; cycle progress = games_done - cyc*games_per_cycle
    int cycle_done = st.games_done - cyc * games_per_cycle;
    if (cycle_done < 0) { cycle_done = 0;
}
    int todo = need - cycle_done;
    std::vector<Sample> cycle_samples;
    // reload pending samples if resuming (lossless)
    {
      std::string sp = cfg.checkpoint + ".samples";
      std::ifstream in(sp);
      if (todo != need) {
        auto v = load_tsv(sp);
        cycle_samples = std::move(v);
      }
    }

    if (todo > 0) {
      Bar bar; bar.label = "self-play";
      std::mutex m;
      std::atomic<int> next{0};
      std::atomic<int> done{0};
      std::atomic<int64_t> samples_count{(int64_t)cycle_samples.size()};
      // preload existing count for bar
      int workers = std::min(threads, todo);
      std::vector<std::thread> th;
      std::vector<std::vector<Sample>> per_thread(workers);
      auto t_start = std::chrono::steady_clock::now();
      th.reserve(workers);
for (int wi = 0; wi < workers; wi++) {
        th.emplace_back([&, wi]() {
          TT tt; tt.resize(8);
          while (true) {
            int g = next.fetch_add(1);
            if (g >= todo) { break;
}
            if (stop_requested.load()) { break;
}
            uint64_t seed = st.rng_state + (uint64_t)(st.games_done + g) * 6364136223846793005ULL + 1442695040888963407ULL;
            GameResult gr = play_game(best_net.trained ? best_net : net,
                                      cfg.search_depth, cfg.nodes_per_move,
                                      cfg.movetime_ms, seed);
            {
              std::lock_guard<std::mutex> lk(m);
              for (auto& s : gr.samples) { per_thread[wi].push_back(std::move(s));
}
            }
            int d = done.fetch_add(1) + 1;
            samples_count.fetch_add((int64_t)gr.samples.size());
            double el = std::chrono::duration<double>(
              std::chrono::steady_clock::now() - t_start).count();
            double gps = d / std::max(0.1, el);
            char extra[192];
            snprintf(extra, sizeof(extra), "%d/%d games  %.1f g/s  %lld samples",
                     d, todo, gps, (long long)samples_count.load());
            bar.draw((double)d / todo, extra);
          }
        });
      }
      for (auto& t : th) { t.join();
}
      bar.done("");
      if (stop_requested.load()) {
        // merge what we have, persist samples + checkpoint, exit
        for (auto& v : per_thread) { for (auto& s : v) { cycle_samples.push_back(std::move(s));
}
}
        // atomic samples persist
        {
          std::string sp = cfg.checkpoint + ".samples";
          std::string tmp = sp + ".tmp";
          FILE* f = fopen(tmp.c_str(), "ab");
          if (!f) { f = fopen(tmp.c_str(), "wb");
}
          if (f) {
            // rewrite fully to avoid dup on repeated pauses: write merged set
            fclose(f);
            f = fopen(tmp.c_str(), "wb");
            for (auto& s : cycle_samples) { fprintf(f, "%s\t%.1f\t%.2f\n", s.fen.c_str(), s.teacher_cp, s.result);
}
            fclose(f);
            std::rename(tmp.c_str(), sp.c_str());
          }
        }
        st.games_done += done.load();
        save_checkpoint();
        std::cout << "\n  " << yellow() << "Paused during self-play — progress saved losslessly."
                  << reset() << "\n  Resume with the same command.\n";
        return 130;
      }
      for (auto& v : per_thread) { for (auto& s : v) { cycle_samples.push_back(std::move(s));
}
}
      st.games_done += todo;
      st.samples_done += (int64_t)cycle_samples.size();
      // persist cycle samples (for lossless resume of training phase)
      {
        std::string sp = cfg.checkpoint + ".samples";
        std::string tmp = sp + ".tmp";
        FILE* f = fopen(tmp.c_str(), "wb");
        if (f) {
          for (auto& s : cycle_samples) { fprintf(f, "%s\t%.1f\t%.2f\n", s.fen.c_str(), s.teacher_cp, s.result);
}
          fclose(f);
          std::rename(tmp.c_str(), sp.c_str());
        }
      }
      // append to lifetime data log
      if (cfg.append_data && !cycle_samples.empty()) {
        FILE* f = fopen(cfg.data_file.c_str(), "a");
        if (f) {
          for (auto& s : cycle_samples) { fprintf(f, "%s\t%.1f\t%.2f\n", s.fen.c_str(), s.teacher_cp, s.result);
}
          fclose(f);
        }
      }
      kv("games played", std::to_string(st.games_done), green());
      kv("cycle samples", std::to_string(cycle_samples.size()));
    } else {
      // resuming directly into training phase; load samples
      cycle_samples = load_tsv(cfg.checkpoint + ".samples");
      kv("reloaded samples", std::to_string(cycle_samples.size()));
    }

    if (stop_requested.load()) { save_checkpoint(); return 130; }

    // ---- add supervised data (optional) ----
    if (!cfg.supervised_file.empty()) {
      auto sup = load_tsv(cfg.supervised_file);
      kv("supervised samples", std::to_string(sup.size()));
      cycle_samples.insert(cycle_samples.end(), sup.begin(), sup.end());
    }
    if (cycle_samples.empty()) {
      std::cout << "  " << red() << "No samples — skipping training this cycle." << reset() << "\n";
      continue;
    }

    // ---- train ----
    std::cout << "  " << bold() << "training NNUE on " << cycle_samples.size()
              << " samples..." << reset() << "\n";
    NNUE before = net;
    double loss = train_epochs(*this, cycle_samples);
    net.trained = true;
    kv("train loss", std::to_string(loss));
    save_checkpoint();
    if (stop_requested.load()) {
      std::cout << "\n  Paused during training — progress saved.\n";
      return 130;
    }

    // ---- validate new vs best ----
    std::cout << "  " << bold() << "validating (new vs best, " << cfg.validation_games
              << " games)..." << reset() << "\n";
    int w = 0, d = 0, l = 0;
    {
      Bar bar; bar.label = "validation";
      bar.draw(0.0, "playing...");
      validation_match(net, best_net.trained ? best_net : before,
                       cfg.validation_games, std::min(6, cfg.search_depth),
                       cfg.nodes_per_move, cfg.movetime_ms,
                       std::min(threads, 8), st.rng_state + (uint64_t)cyc * 1000003ULL,
                       w, d, l, &stop_requested);
      bar.done("");
    }
    double score = (w + d * 0.5) / std::max(1, w + d + l);
    double delo = elo_from_score(score);
    // error bar ~ 400/sqrt(N)
    double err = 400.0 / std::sqrt((double)std::max(1, w + d + l));
    std::cout << "  score " << w << "W " << d << "D " << l << "L  ("
              << std::fixed << std::setprecision(1) << (score * 100) << "%)  "
              << "Elo " << elo_str(delo) << " ± " << std::setprecision(0) << err << "\n";
    st.wins = w; st.draws = d; st.losses = l;
    st.total_w += w; st.total_d += d; st.total_l += l;
    if (score > 0.5) {
      best_net = net;
      st.elo += delo;
      kv("ACCEPTED — new best", elo_str(st.elo) + " cumulative", green());
    } else {
      // keep best, but still carry net forward (common practice: keep training from new weights)
      best_net = best_net.trained ? best_net : net;
      kv("kept previous best", "new net scored " + std::to_string((int)(score * 100)) + "%", yellow());
      // continue training from new net anyway (it saw more data)
    }
    // reset per-cycle epoch counters
    st.epoch = 0; st.batch_idx = 0;
    // clear cycle samples file (training for this cycle is committed)
    std::remove((cfg.checkpoint + ".samples").c_str());
    // save + export best
    save_checkpoint();
    nnue_save(best_net, cfg.out_net + ".tmp");
    std::rename((cfg.out_net + ".tmp").c_str(), cfg.out_net.c_str());
    kv("exported best net", cfg.out_net, green());
    kv("cumulative elo", elo_str(st.elo), yellow());
    // advance RNG
    st.rng_state = st.rng_state * 6364136223846793005ULL + 1442695040888963407ULL;
    save_checkpoint();
  }

  std::cout << bold() << green() << "\n  Training complete." << reset()
            << "  Best net: " << cfg.out_net
            << "  Elo: " << elo_str(st.elo) << "\n";
  std::cout << "  Run the engine with: ./forge  (then: setoption name NNUEFile value "
            << cfg.out_net << ")\n\n";
  g_trainer = nullptr;
  return 0;
}

int trainer_run(TrainerConfig cfg) {
  Trainer tr(std::move(cfg));
  return tr.run();
}

} // namespace forge
