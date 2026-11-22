// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Vaibhav Agarwal

// forge-chess — trainer.h : self-play + NNUE training with lossless pause/resume
#pragma once
#include "position.h"
#include "nnue.h"
#include "search.h"
#include <string>
#include <utility>
#include <vector>
#include <atomic>
#include <cstdint>

namespace forge {

struct TrainerConfig {
  std::string out_net = "forge.nnue";       // final net
  std::string checkpoint = "forge.ckpt";    // checkpoint prefix (files: .state .adam .net)
  std::string data_file = "forge_data.tsv"; // appended self-play samples (FEN\tcp\tresult)
  std::string supervised_file;              // optional extra supervised data (same TSV)

  int games = 2000;            // total self-play games this run
  int cycles = 4;              // split games into cycles, train after each
  int threads = 0;             // 0 = auto (all cores)
  int search_depth = 6;        // self-play search depth (4-8 sane on laptop)
  int64_t nodes_per_move = 0;  // 0 = use depth; else cap nodes
  int movetime_ms = 0;         // 0 = depth/nodes; else fixed ms per move

  int epochs_per_cycle = 3;
  int batch = 1024;
  float lr = 0.001f;
  float lambda = 0.6f;         // weight on teacher vs result (0..1)
  int validation_games = 60;   // games new-vs-old per cycle for Elo
  int seed = 12345;

  int ckpt_every_batches = 200; // lossless resume granularity
  bool append_data = true;
};

struct TrainerState {
  int games_done = 0;
  int64_t samples_done = 0;
  int cycle = 0, epoch = 0, batch_idx = 0;
  uint64_t rng_state = 0;
  int wins = 0, draws = 0, losses = 0;       // validation tally (current cycle)
  int total_w = 0, total_d = 0, total_l = 0; // lifetime validation
  double elo = 0.0;                          // cumulative Elo estimate
  double loss = 0.0;
  uint64_t adam_t = 0;
};

double elo_from_score(double score); // score in [0,1] -> elo diff
double score_from_elo(double elo);

struct Trainer {
  TrainerConfig cfg;
  NNUE net;            // training net
  NNUE best_net;       // best validated net
  TrainerState st;
  std::vector<float> adam_m, adam_v; // sized to total params
  std::atomic<bool> stop_requested{false};

  Trainer(TrainerConfig c) : cfg(std::move(c)) {}

  // main entry: resumes if checkpoint exists, else fresh
  int run();

  // checkpoint (atomic tmp+rename, lossless incl. Adam + RNG + Elo)
  bool save_checkpoint();
  bool load_checkpoint();

  // helpers
  void ensure_adam();
  [[nodiscard]] int total_params() const;
  float* param_ptr(int idx); // 0=w1 1=b1 2=w2 3=b2 (flat helpers via offsets)
};

int trainer_run(TrainerConfig cfg);

} // namespace forge
