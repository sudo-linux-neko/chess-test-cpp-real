// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Vaibhav Agarwal

// forge-train — gorgeous CLI: self-play + NNUE, progress bar, lossless pause/resume
#include "trainer.h"
#include "cli.h"
#include <iostream>
#include <string>
#include <thread>

static void usage(const char* a) {
  using namespace forge::cli;
  header("FORGE TRAINER");
  std::cout << "Usage:\n";
  std::cout << "  " << a << " [options]\n\n";
  std::cout << "Examples:\n";
  std::cout << "  " << a << " --games 2000 --cycles 4 --depth 6\n";
  std::cout << "  " << a << " --games 500 --depth 5 --epochs 2 --batch 1024   (quick laptop run)\n";
  std::cout << "  " << a << " --supervised data.tsv --epochs 10               (train on your dataset)\n";
  std::cout << "  " << a << " --resume                                        (same as rerunning)\n\n";
  std::cout << "Options:\n";
  std::cout << "  --games N          total self-play games (default 2000)\n";
  std::cout << "  --cycles N         split into N train cycles (default 4)\n";
  std::cout << "  --depth D          self-play search depth 1..12 (default 6)\n";
  std::cout << "  --nodes N          nodes per move instead of depth (default 0=depth)\n";
  std::cout << "  --movetime MS      ms per move instead of depth (default 0)\n";
  std::cout << "  --threads T        0=auto (default 0)\n";
  std::cout << "  --epochs E         epochs per cycle (default 3)\n";
  std::cout << "  --batch B          batch size (default 1024)\n";
  std::cout << "  --lr X             Adam lr (default 0.001)\n";
  std::cout << "  --lambda X         teacher weight 0..1 (default 0.6)\n";
  std::cout << "  --valgames N       validation games per cycle (default 60)\n";
  std::cout << "  --out FILE         output net (default forge.nnue)\n";
  std::cout << "  --ckpt PREFIX      checkpoint prefix (default forge.ckpt)\n";
  std::cout << "  --data FILE        self-play log TSV (default forge_data.tsv)\n";
  std::cout << "  --supervised FILE  extra TSV data (FEN<TAB>cp<TAB>result)\n";
  std::cout << "  --seed N           rng seed (default 12345)\n";
  std::cout << "  --resume           resume from checkpoint (default: auto-resume if ckpt exists)\n";
  std::cout << "  --fresh            ignore checkpoint, start over\n";
  std::cout << "  -h, --help         this help\n\n";
  std::cout << "Pause/resume: press Ctrl+C once — state (games, Elo, Adam, RNG, batches)\n";
  std::cout << "is saved atomically. Rerun the SAME command to resume losslessly.\n\n";
  std::cout << "Strength honesty: v1 starts ~2000-2400 Elo (classical + small NNUE).\n";
  std::cout << "Each cycle adds real Elo. 3000/Stockfish-beating needs 100B+ positions\n";
  std::cout << "and weeks of compute — this trainer scales there without code changes.\n\n";
}

int main(int argc, char** argv) {
  forge::TrainerConfig cfg;
  bool fresh = false;
  for (int i = 1; i < argc; i++) {
    std::string a = argv[i];
    auto need = [&](std::string& out) {
      if (i + 1 >= argc) { std::cerr << "missing value for " << a << "\n"; exit(2); }
      out = argv[++i];
    };
    std::string v;
    if (a == "-h" || a == "--help") { usage(argv[0]); return 0; }
    if (a == "--games") { need(v); cfg.games = std::stoi(v); }
    else if (a == "--cycles") { need(v); cfg.cycles = std::stoi(v); }
    else if (a == "--depth") { need(v); cfg.search_depth = std::stoi(v); }
    else if (a == "--nodes") { need(v); cfg.nodes_per_move = std::stoll(v); }
    else if (a == "--movetime") { need(v); cfg.movetime_ms = std::stoi(v); }
    else if (a == "--threads") { need(v); cfg.threads = std::stoi(v); }
    else if (a == "--epochs") { need(v); cfg.epochs_per_cycle = std::stoi(v); }
    else if (a == "--batch") { need(v); cfg.batch = std::stoi(v); }
    else if (a == "--lr") { need(v); cfg.lr = std::stof(v); }
    else if (a == "--lambda") { need(v); cfg.lambda = std::stof(v); }
    else if (a == "--valgames") { need(v); cfg.validation_games = std::stoi(v); }
    else if (a == "--out") { need(v); cfg.out_net = v; }
    else if (a == "--ckpt") { need(v); cfg.checkpoint = v; }
    else if (a == "--data") { need(v); cfg.data_file = v; }
    else if (a == "--supervised") { need(v); cfg.supervised_file = v; }
    else if (a == "--seed") { need(v); cfg.seed = std::stoi(v); }
    else if (a == "--resume") { /* default behavior */ }
    else if (a == "--fresh") { fresh = true; }
    else { std::cerr << "unknown arg: " << a << "\n"; usage(argv[0]); return 2; }
  }
  if (cfg.threads == 0) {
    unsigned n = std::thread::hardware_concurrency();
    cfg.threads = n ? (int)std::min<unsigned>(n, 64) : 4;
  }
  if (fresh) {
    std::remove((cfg.checkpoint + ".state").c_str());
    std::remove((cfg.checkpoint + ".adam").c_str());
    std::remove((cfg.checkpoint + ".net").c_str());
    std::remove((cfg.checkpoint + ".best").c_str());
    std::remove((cfg.checkpoint + ".samples").c_str());
  }
  if (cfg.games < 1) { cfg.games = 1;
}
  if (cfg.cycles < 1) { cfg.cycles = 1;
}
  if (cfg.search_depth < 1) { cfg.search_depth = 1;
}
  if (cfg.search_depth > 12) { cfg.search_depth = 12;
}
  return forge::trainer_run(cfg);
}
