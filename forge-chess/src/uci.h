// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Vaibhav Agarwal

// forge-chess — uci.h
#pragma once
#include "position.h"
#include "search.h"
#include <string>
#include <vector>
#include <atomic>

namespace forge {

struct UCIOptions {
  int hash_mb = 64;
  int threads = 1;
  std::string nnue_file;
  int move_overhead_ms = 50;
};

void uci_loop();
void print_board_pretty(const Position& p);

// Parse helpers exposed for trainer/perft
bool parse_position_cmd(Position& p, std::vector<U64>& hist, const std::string& line);
Limits parse_go_cmd(const std::string& line, const Position& p);

} // namespace forge
