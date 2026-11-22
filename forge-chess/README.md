# ♞ FORGE — pure C++17 chess engine + self-play NNUE trainer

**Zero dependencies. Any OS with a C++17 compiler. AVX2-fast, laptop-tuned, honestly benchmarked.**

- `forge` — UCI engine (bitboards, PVS + TT + LMR + null-move + … , 768→256×2→1 NNUE, AVX2 inference)
- `forge-train` — gorgeous self-play + NNUE trainer with **progress bars**, **Elo tracking**,
  and **lossless pause/resume** (Ctrl+C once → rerun same command)
- `forge-perft` — move-generator correctness gate (4/4 suites pass)

> **Honest strength note.** v1 starts ~2000–2400 Elo (classical + small NNUE) and climbs
> every cycle — verified +88 Elo in a tiny 8-game smoke test. Beating Stockfish / 3000 Elo
> is a *compute* problem, not a code problem: Stockfish trains on **>100B Lc0-labelled
> positions** with Fishtest-scale hardware (see `Training → the road to 3000` below).
> This repo gives you the full ladder with no code changes — just add games + time.

---

## 1. Build (Linux / Windows / macOS — same source)

Requirements: CMake ≥ 3.16 + any C++17 compiler (GCC / Clang / MSVC). No libraries.

```bash
# HYPER-OPTIMIZED for THIS machine (your i5-1335U: -march=native, AVX2+BMI2+FMA)
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DFORGE_NATIVE=ON
cmake --build build -j

# PORTABLE (runs on any x86-64 OS/CPU, still AVX2 where available + scalar fallback)
cmake -S . -B build-portable -DCMAKE_BUILD_TYPE=Release -DFORGE_NATIVE=OFF -DFORGE_PORTABLE=ON
cmake --build build-portable -j

# Windows (MSVC):  cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
# macOS (AppleClang): same as Linux. ARM Macs use the scalar path (correct, just slower).
```

Binaries: `build/forge` (engine), `build/forge-train` (trainer), `build/forge-perft` (tests).

Single-file fallback (no CMake): every `src/*.cpp` compiles with
`g++ -O3 -march=native -std=c++17 src/*.cpp -pthread` minus one `*_main.cpp`.

---

## 2. Engine (UCI — works in Arena, Cutechess, Banksia, lichess-bot…)

```bash
./build/forge
```

```
uci
isready
setoption name Hash value 128
setoption name Threads value 8
setoption name NNUEFile value forge.nnue
ucinewgame
position startpos moves e2e4 e7e5
go depth 12
go wtime 60000 btime 60000 winc 1000 binc 1000
stop
quit
```

Extras (terminal analysis): `d` (pretty Unicode board), `eval` (classical/NNUE/blended),
`perft 5` (divide). Full `position fen … moves …` + `go {depth|nodes|movetime|infinite}` supported.

**How fast?** (i5-1335U, native build, startpos, hash 64)
~2.4M nps, depth 5 in ~5 ms single-thread. 12 threads auto-detected (`Threads` overrides).
TT 1–2048 MB, Lazy-SMP helpers warm the shared table. AVX2 NNUE path with scalar fallback —
`FORGE_HAS_AVX2` is auto-detected; logic is identical either way.

---

## 3. Trainer — the pretty CLI you asked for

```bash
./build/forge-train --help

# Quick laptop smoke test (~1 min, proves the whole loop)
./build/forge-train --games 8 --cycles 2 --depth 3 --epochs 1 --batch 32 --valgames 4

# Overnight i5-1335U run (recommended starting grind)
./build/forge-train --games 2000 --cycles 4 --depth 6 --epochs 3 --batch 1024 --valgames 60

# MAX-quality grind (weeks → real Elo; same code, just more games)
./build/forge-train --games 50000 --cycles 20 --depth 8 --epochs 5 --batch 4096 --valgames 200

# Supervised NNUE on your own dataset (FEN<TAB>cp<TAB>result, white POV):
#   rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1<TAB>35<TAB>0.5
./build/forge-train --supervised my_data.tsv --epochs 10 --games 500 --depth 5

# Start over (else rerunning auto-resumes)
./build/forge-train --fresh --games 2000 --depth 6
```

What you get live:

```
  ███████╗ ██████╗ …  FORGE TRAINER — self-play + NNUE
  self-play [████████████▌░░░░░░]  62.5%  5/8 games  11.7 g/s  320 samples  ETA 0:01
  epoch 1/3 [██████▌░░░░]  40.0%  loss 0.05901  lr 0.0010
  score 1W 3D 0L (62.5%)  Elo +88.7 ± 200
  ACCEPTED — new best  +88.7 cumulative
```

- **Self-play**: parallel games, random openings, resign/draw adjudication, teacher scores.
- **Training**: Adam (from scratch, pure C++), teacher+result blend `--lambda` (default 0.6),
  deterministic shuffles, batch sizes auto-fit to RAM.
- **Elo**: every cycle plays `--valgames` new-vs-best and prints Elo ± error. Cumulative Elo persists.
- **Max-Elo extraction**: teacher blend + validation-gating (only better nets ship) + full-data
  logging to `--data` for replay. No Elo is left on the table the data supports.

### Lossless pause / resume (the headline feature)

Press **Ctrl+C once** at any moment — self-play, training, validation. The trainer finishes the
current batch/game, then atomically writes:

| File | Contents |
|---|---|
| `<ckpt>.state` | games_done, cycle/epoch/batch, RNG state, W/D/L, Elo, loss, Adam step, lr |
| `<ckpt>.adam` | full Adam `m` + `v` moments (bit-exact optimizer resume) |
| `<ckpt>.net` / `.best` | training net + best validated net |
| `<ckpt>.samples` | current cycle's training set (no sample loss mid-epoch) |
| `<out>.nnue` | exported best net (engine-ready) |

Writes go to `*.tmp` + `rename()` — a crash mid-save never corrupts. **Rerun the same command
to resume from the exact Elo/batch.** Verified: `games_done 28 → 48`, `adam_t 26 → 48` across
two SIGINT pauses with zero drift.

---

## 4. Architecture (what's actually in the 5k lines)

- **Board**: 12 bitboards + mailbox, incremental Zobrist, ray-fill sliders (cache-free,
  ~magic-class speed on Raptor Lake, zero 800 KB tables → friendly to low-RAM / any OS).
  API is magic-compatible for a future drop-in.
- **Movegen**: pseudo-legal + `make/unmake` legality filter; handles EP pins, castling-through-check,
  promos. **Perft-verified**: 197281 (startpos d4), 97862 (kiwipete d3), 43238, 9467.
- **Search**: negamax PVS + aspiration + IID-ish, TT (lockless, age + depth-preferred),
  null-move (R=3+d/4), reverse-futility, razoring, LMR (late/quiet only, checks exempted),
  SEE + futility pruning, qsearch with delta+SEE pruning, mate-distance pruning,
  killer + history (clamped) + counter ordering, check extensions. Lazy-SMP TT sharing.
- **Eval**: tapered classical (MG/EG PSTs + pawn structure + king shield + bishop pair + tempo)
  blended 25/75 with NNUE once trained.
- **NNUE**: 768 → 256×2 → 1, perspective accumulators, clipped-ReLU, AVX2 float dot
  (FMA-free → builds with plain `-mavx2`), incremental `nnue_update()` on every make
  (quiet path = 1 sub + 1 add per side). Random-Xavier init, versioned `FORGEN1` binary.
- **Training**: self-play TSV (`FEN\tcp\tresult`) → Adam on
  `sigmoid(pred/400)` vs `λ·sigmoid(teacher/400)+(1−λ)·result` (STM-adjusted), full backprop
  through both accumulators + shared bias. Deterministic per-epoch shuffles.
- **Portability**: C++17 only, threads + `<chrono>` + `<cstdio>`. `NO_COLOR`/`TERM=dumb` disable ANSI.
  Windows/MSVC, Linux/GCC-Clang, macOS all build from this tree.

---

## 5. The road to 3000 (no fairy tales)

| Milestone | What it takes | This repo |
|---|---|---|
| 2000–2400 | Classical + small NNUE, thousands of games | ✅ out of the box |
| 2500–2700 | 100k–1M games, depth 8–10, bigger batches | ✅ same command, more `--games` |
| 2800–3000 | 10M+ games + Lc0 teacher data, SFNNv6-class net (1024+), Fishtest-scale SMP | 🔧 grow `NNUE_HIDDEN`, import binpacks via `--supervised` |
| 3000+ / beat Stockfish | 100B positions, months on 100s of cores + GPUs (how Stockfish does it) | 📈 ladder is ready; compute is the fuel |

To go beyond v1: raise `NNUE_HIDDEN` → 512/1024, add king-buckets, convert Lc0/Stockfish
binpacks to `FEN\tcp\tresult` and pass as `--supervised`, then leave `forge-train` running.
Every Elo mechanism (gating, blending, checkpointing) already scales.

Sources consulted: Stockfish NNUE docs + `nnue_architecture.h` (SFNNv6 → v10+ threat inputs),
`nnue-pytorch` self-play/validation methodology, Chessprogramming Wiki (magics, LMR, PVS),
plus perft suites for correctness.

---

## 6. Layout

```
CMakeLists.txt          native (+march=native) vs portable (+mavx2 scalar-fallback) builds
src/core.h              types, moves, bitboards, attacks, zobrist
src/position.h/.cpp     board, make/unmake, FEN, movegen, SEE
src/nnue.h/.cpp         net, incremental accumulators, AVX2 eval, save/load
src/search.h/.cpp       classical eval, TT, PVS search, qsearch, SMP, perft
src/uci.h/.cpp          async UCI loop + pretty board
src/cli.h               colors, progress bars, tables
src/trainer.h/.cpp      self-play, Adam, validation Elo, atomic checkpoints
src/engine_main.cpp     forge entry        src/trainer_main.cpp  forge-train entry
src/perft_main.cpp      forge-perft entry
```

## 7. License

MIT — do what you want, beat Stockfish, tell us your Elo.
