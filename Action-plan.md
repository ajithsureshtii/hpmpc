# Action Plan: Multi-GPU Support for H100 / A100 Distributed Execution

## Context

**Goal:** Enable P0 to run on an H100 machine and P1 to run on an A100 machine, each with 4 GPUs (80 GB each). Measure and compare GPU vs CPU execution using the configs in `measurements/configs/artifacts/triad/2pc/GPU/`.

**This repo** (`hpmpc`, branch `aby2-merge`) is the H100 machine's copy.

**Reference:** `../../benchmarking/hpmpc` is an older fork where multi-GPU work was developed and tested on a single H100 machine (8 GPUs, both parties on one box). That fork is **not** to be copied verbatim — it targets a single-machine scenario. We understand the intent and apply only what is necessary for a true two-machine distributed setup.

**Already done in this repo:** Docker files (`Dockerfile`, `docker/Dockerfile`, `docker-run.sh`) are present and already match the benchmarking fork exactly.

---

## Full Audit of Benchmarking Fork Changes

### 1. Makefile — GPU build infrastructure
- **CUDA lib path:** `/opt/cuda/targets/x86_64-linux/lib` → `/usr/local/cuda/lib64` (Docker image layout).
- **CHEETAH_GPU_ARCH auto-detect:** `$(shell nvidia-smi --query-gpu=compute_cap ...)` instead of hardcoded `90` — same Makefile works on H100 (sm_90) and A100 (sm_80).
- **NVCC_LINKFLAGS:** `-Xlinker -rpath,...` so the troy/HE shared libs are found at runtime (NVCC does not accept `-Wl,` syntax).
- **CHEETAH_GPU_REVERSE flag:** Swap which party encrypts (P1 encrypts weights instead of inputs). Needed if the A100 should be the encrypting side.
- **CHEETAH_FC_GPU flag:** GPU-accelerated FC triple generation via troy-nova `MatmulHelper`. Default=1.
- **CHEETAH_NUM_GPUS flag:** Routes forked process `i` to `gpu_id = process_offset % CHEETAH_NUM_GPUS`. Default=1.
- **`convtriple_check` target:** Auto-rebuild ConvTriple when any of `CHEETAH_GPU / CHEETAH_GPU_ARCH / CHEETAH_GPU_REVERSE / CHEETAH_FC_GPU` changes. Stamp file tracks last-built config; mismatch triggers full deps+build.
- **CONFIG_OPTIONS:** `CHEETAH_NUM_GPUS` and `CHEETAH_FC_GPU` added so they are baked into the compiled executable.

### 2. config.h
- `#ifndef CHEETAH_NUM_GPUS / #define CHEETAH_NUM_GPUS 1` default guard.

### 3. core/generate_beaver_tiples.hpp
- Computes `cheetah_gpu_id = process_offset % CHEETAH_NUM_GPUS`.
- Passes it to both `generateConvTriplesCheetahWrapper(...)` call sites and both `generateFCTriplesCheetah(...)` call sites.

### 4. core/cuda/gemm_cutlass_int.cu (online GEMM)
- `cudaSetDevice(process_offset % CHEETAH_NUM_GPUS)` before each CUTLASS GEMM.
- Commented out `uint16_t` template instantiation (unsupported on sm_90/sm_80).

### 5. core/cuda/conv_cutlass_int.cu
- Commented out `uint16_t` template instantiation.

### 6. nn/ConvTriple submodule
- `conv2d_gpu.cuh/.cu`: `device_id` param added to `setup()`, `conv2d()`, and all four `conv2d_ab*` variants; `cudaSetDevice(device_id)` in `setup()`.
- `fc_gpu.cu/.cuh`: New GPU-accelerated FC triple generation using troy-nova `MatmulHelper`. Supports `fc_ab2`, `fc_ab2_reverse`, `fc_ab`, `fc_ab_reverse`, `fc()` dispatcher.
- `hpmpc_interface.hpp/.cpp`: `gpu_id` param on both `generateConvTriplesCheetahWrapper` and `generateFCTriplesCheetah`.
- Bug fixes: CUDA context init before GlobalPool, expand_seed before NTT, disconnect after triples, `pack_lwe=false` in all MatmulHelper constructors.

### 7. measurements/run_config.py — three critical fixes
- **Sorted config file ordering** (`sorted(os.listdir(...))`): When running on two separate machines, P0 and P1 must iterate configs in the **exact same order** or they will deadlock — one compiles/runs config A while the other compiled config B. Without this fix, `os.listdir()` returns filesystem-dependent order which can differ between machines.
- **Streaming output** (`stdout=subprocess.STDOUT`, `readline` loop): Long GPU builds and runs don't buffer until completion; you see progress as it happens.
- **Port-freeing cleanup after each run** (`pkill -f run.sh; pkill -f run-P; pgrep -f python | grep -v <own_pid> | xargs kill`): Leftover processes from a run hold ports open and cause the next run to fail. The self-PID exclusion is critical — an earlier version of the cleanup killed the python script itself.
- **`-G player:device` flag:** For per-party GPU restriction on single-machine runs only (not needed for distributed but useful for future local tests).

### 8. Config file structure in measurements/configs/artifacts/triad/2pc/GPU/
The benchmarking fork did a full restructure that peregrine is missing:

| Benchmarking file | Peregrine equivalent | Delta |
|---|---|---|
| `a_2PC_single_batch_all_opt_A2bits.conf` | `2PC_single_batch_all_opt_A2bits.conf` | Missing: `CHEETAH_FC_GPU=1`, `CHEETAH_DISCONNECT=1`. No GPU arch coverage for USE_CUDA_GEMM (not present, correct for single batch). |
| `a_2PC_single_batch_all_opt_reshared.conf` | `2PC_single_batch_all_opt_reshared.conf` | Missing: `CHEETAH_FC_GPU=1`, `CHEETAH_DISCONNECT=1`. |
| `b_2PC_multi_batch_all_opt_A2bits.conf` | `2PC_multi_batch_all_opt_A2bits.conf` | Missing: `CHEETAH_NUM_GPUS=8→4`, `CHEETAH_FC_GPU=1`, `CHEETAH_DISCONNECT=1`. |
| `b_2PC_multi_batch_all_opt_reshared.conf` | `2PC_multi_batch_all_opt_reshared.conf` | Missing: `CHEETAH_NUM_GPUS=8→4`, `CHEETAH_FC_GPU=1`, `CHEETAH_DISCONNECT=1`. |
| `c_CPU_2PC_single_batch_all_opt_A2bits.conf` | **does not exist** | CPU baseline for single-batch A2bits comparison. |
| `c_CPU_2PC_single_batch_all_opt_reshared.conf` | **does not exist** | CPU baseline for single-batch reshared comparison. |
| `d_CPU_2PC_multi_batch_all_opt_A2bits.conf` | **does not exist** | CPU baseline for multi-batch A2bits comparison. |
| `d_CPU_2PC_multi_batch_all_opt_reshared.conf` | **does not exist** | CPU baseline for multi-batch reshared comparison. |

The `a_`/`b_`/`c_`/`d_` **prefix naming** is not cosmetic — it controls sort order. When `run_config.py` is pointed at the folder, it processes files alphabetically: `a_` (single-batch GPU) before `b_` (multi-batch GPU), and a separate CPU folder pass. Without the prefix, the files `2PC_multi_*` and `2PC_single_*` sort in a different and inconsistent order.

**The CPU config files (`c_CPU_*`, `d_CPU_*`) are how you do the CPU vs GPU comparison**: instead of always using `--override CHEETAH_GPU=0`, you have explicit CPU configs alongside GPU configs in the same folder — both parties simply run `run_config.py` over the folder and get both GPU and CPU results in sequence.

### 9. scripts/run.sh and scripts/run_locally.sh — -G flag
- Added `-G player:device` for per-party `CUDA_VISIBLE_DEVICES` restriction.
- **For distributed two-machine runs this flag is not needed** — each Docker container already sees only its own GPUs.
- Useful for future single-machine smoke tests and sanity checks.

---

## What Is Already Done in This Repo

| Item | Status |
|------|--------|
| `docker/Dockerfile` | ✅ Identical to benchmarking fork (includes iperf3, ping) |
| `Dockerfile` (root) | ✅ Identical |
| `docker-run.sh` | ✅ Identical |
| ConvTriple submodule commit (`c67aa63`) | ✅ Latest fixed version — all multi-GPU and bug-fix commits included |
| Some GPU configs exist | ⚠️ Present but incomplete (missing fields, missing CPU counterparts, missing prefix naming) |

---

## What Is NOT in Scope

- The benchmarking fork's `run_cases.sh` (a quick 4-case sanity script targeting a different single-machine workflow).
- `test-details/H100.md` (single-machine benchmark results from the other fork).
- `CHEETAH_GPU_REVERSE` in configs — this swaps encryption roles and should only be set after measurement shows the A100 side is the bottleneck as encryptor. Leave as default (0) for now.

---

## Proposed Changes — Commit Plan

### Commit 1: `fix: update ConvTriple submodule URL to fork and set branch`
**File:** `.gitmodules`

Change:
```
url = https://github.com/chart21/ConvTriple
```
To:
```
url = https://github.com/ajithsureshtii/ConvTriple
branch = aby2-merge
```

**Why:** The submodule pointer already points to the correct commit (`c67aa63`) but the URL still points to upstream. A fresh `git clone --recurse-submodules` would fail since the upstream does not contain the GPU commits. This must be the first commit so everything that follows can be cloned and reproduced cleanly.

---

### Commit 2: `fix: Makefile — CUDA path and NVCC linker flags for Docker environment`
**File:** `Makefile`

- `HE_PATHS`: `/opt/cuda/targets/x86_64-linux/lib` → `/usr/local/cuda/lib64`
- Add `NVCC_LINKFLAGS` variable with `-Xlinker -rpath,${CHEETAH}/build/lib:${CHEETAH}/deps/lib -L... $(HE_LIBS) -lssl -lcrypto`
- Use `$(NVCC_LINKFLAGS)` in all three nvcc link lines (USE_CUDA_GEMM=1/2/4 cases)

**Why:** The Docker image places CUDA at `/usr/local/cuda`. The missing rpath caused runtime `libHE.so` / `libtroy.so` not found errors when running GPU-linked executables outside of the build directory.

---

### Commit 3: `feat: Makefile — auto-detect GPU arch, GPU flags, convtriple_check target`
**File:** `Makefile`

- `CHEETAH_GPU_ARCH`: hardcoded `90` → `$(shell nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -1 | tr -d '.')`
- Add: `CHEETAH_GPU_REVERSE ?= 0`, `CHEETAH_FC_GPU ?= 1`, `CHEETAH_NUM_GPUS ?= 1`
- Update stamp content: `gpu=$(CHEETAH_GPU) arch=$(CHEETAH_GPU_ARCH) reverse=$(CHEETAH_GPU_REVERSE) fc_gpu=$(CHEETAH_FC_GPU)`
- Add full `convtriple_check` target with stamp mismatch detection, GPU/CPU branch, and executable clean-on-rebuild
- Wire `compile_pch` to depend on `convtriple_check`; `compile_executables` to depend on `compile_pch`
- Add `CHEETAH_NUM_GPUS` and `CHEETAH_FC_GPU` to `CONFIG_OPTIONS`
- Add `convtriple_check` to `.PHONY`

**Why:** Without this, switching between GPU and CPU builds requires manually rebuilding ConvTriple. On the A100 machine the arch would be sm_80, not sm_90 — auto-detect ensures the same Makefile works on both machines without modification.

---

### Commit 4: `feat: config.h — add CHEETAH_NUM_GPUS default define`
**File:** `config.h`

Add after existing GPU-related defines:
```cpp
#ifndef CHEETAH_NUM_GPUS
#define CHEETAH_NUM_GPUS 1
#endif
```

**Why:** `gemm_cutlass_int.cu` references this macro directly. Without the guard, direct nvcc compilation (outside of Make) fails with undeclared identifier. Also serves as documentation of the flag's default.

---

### Commit 5: `feat: dispatch preprocessing to per-GPU in generate_beaver_tiples.hpp`
**File:** `core/generate_beaver_tiples.hpp`

- Add `const int cheetah_gpu_id = process_offset % CHEETAH_NUM_GPUS;` in the forked child section
- Pass `cheetah_gpu_id` as the last argument to all four call sites:
  - `generateConvTriplesCheetahWrapper(...)` (two call sites)
  - `generateFCTriplesCheetah(...)` (two call sites)

**Why:** This is the core of multi-GPU preprocessing. Without it, all `PROCESS_NUM=24` forked children route their HE computations to GPU 0, completely negating the benefit of having 4 GPUs. With `CHEETAH_NUM_GPUS=4`, process 0→GPU0, process 1→GPU1, process 2→GPU2, process 3→GPU3, process 4→GPU0, etc.

---

### Commit 6: `feat: online GEMM multi-GPU — cudaSetDevice per process in gemm_cutlass_int.cu`
**File:** `core/cuda/gemm_cutlass_int.cu`

- Add `#include <cuda_runtime.h>`
- Add `extern int process_offset;` and `#ifndef CHEETAH_NUM_GPUS / #define CHEETAH_NUM_GPUS 1` guard
- Call `cudaSetDevice(process_offset % CHEETAH_NUM_GPUS)` at the top of `gemm_cutlass()`
- Comment out the `uint16_t` template explicit instantiation (compile error on sm_90/sm_80)

**Why:** `USE_CUDA_GEMM=2` is set in the multi-batch GPU configs. Without `cudaSetDevice`, all online GEMM calls land on GPU 0. The `uint16_t` CUTLASS template fails to compile on H100/A100 (CUTLASS INT8/INT16 not universally supported).

---

### Commit 7: `fix: conv_cutlass_int.cu — comment out uint16_t template instantiation`
**File:** `core/cuda/conv_cutlass_int.cu`

Comment out the `uint16_t` explicit template instantiation.

**Why:** Same CUTLASS architecture restriction as above. This code path is not used in the test configs (which use `DATTYPE=32` or `DATTYPE=256`).

---

### Commit 8: `fix: run_config.py — sorted ordering, streaming output, port-freeing cleanup`
**File:** `measurements/run_config.py`

Three changes, all independent but bundled since they're all in one small file:

1. **Sorted config ordering:**
   ```python
   # before:
   config_files = [os.path.join(args.config, f) for f in os.listdir(args.config) if ...]
   # after:
   config_files = sorted([os.path.join(args.config, f) for f in os.listdir(args.config) if ...])
   ```

2. **Streaming output:**
   ```python
   # before: subprocess.Popen + communicate() (buffers until end)
   # after: readline() loop with stdout=subprocess.STDOUT, flush=True
   ```

3. **Port-freeing cleanup after each run (with self-PID guard):**
   ```python
   my_pid = os.getpid()
   cleanup_command = (
       f"pkill -9 -f run.sh; pkill -9 -f run-P; "
       f"pgrep -f python | grep -v {my_pid} | xargs -r kill -9; "
       f"sleep 1; clear"
   )
   subprocess.run(cleanup_command, shell=True)
   ```

**Why sorted ordering is critical:** When `run_config.py` is invoked on both machines simultaneously, `os.listdir()` may return files in different orders on different Linux filesystems. If P0 compiles and runs config A while P1 has compiled config B, they will never connect — both timeout waiting for the other. This is a plausible root cause of the "sync issue" in the old fork when multiple configs were run in sequence.

**Why streaming output matters:** GPU builds and runs take minutes each. Without streaming, you see nothing until the full make + run completes — you can't distinguish a hanging run from a slow run.

**Why the self-PID guard:** The original cleanup in benchmarking (commit `28d3f88`) killed all python processes including the measurement script itself, discovered and fixed in `80c8bca`. We apply the fix directly.

---

### Commit 9: `feat: rename GPU configs with a/b prefix, add CPU counterpart configs`
**Directory:** `measurements/configs/artifacts/triad/2pc/GPU/`

Renames (preserving content, updating missing fields):
- `2PC_single_batch_all_opt_A2bits.conf` → `a_2PC_single_batch_all_opt_A2bits.conf` + add `CHEETAH_FC_GPU=1`, `CHEETAH_DISCONNECT=1`
- `2PC_single_batch_all_opt_reshared.conf` → `a_2PC_single_batch_all_opt_reshared.conf` + add `CHEETAH_FC_GPU=1`, `CHEETAH_DISCONNECT=1`
- `2PC_multi_batch_all_opt_A2bits.conf` → `b_2PC_multi_batch_all_opt_A2bits.conf` + add `CHEETAH_NUM_GPUS=4`, `CHEETAH_FC_GPU=1`, `CHEETAH_DISCONNECT=1`
- `2PC_multi_batch_all_opt_reshared.conf` → `b_2PC_multi_batch_all_opt_reshared.conf` + add `CHEETAH_NUM_GPUS=4`, `CHEETAH_FC_GPU=1`, `CHEETAH_DISCONNECT=1`

New CPU baseline files (no GPU flags, explicit CPU settings):
- `c_CPU_2PC_single_batch_all_opt_A2bits.conf` — same as `a_*_A2bits` but `CHEETAH_GPU=0`, `USE_CUDA_GEMM=0`, `ADDITIONAL_GEMM_THREADS=24`
- `c_CPU_2PC_single_batch_all_opt_reshared.conf`
- `d_CPU_2PC_multi_batch_all_opt_A2bits.conf` — same as `b_*_A2bits` but `CHEETAH_GPU=0`, `USE_CUDA_GEMM=0`, `ADDITIONAL_GEMM_THREADS=1`
- `d_CPU_2PC_multi_batch_all_opt_reshared.conf`

**Why `CHEETAH_NUM_GPUS=4` not 8:** Benchmarking used 8 GPUs total on one machine, splitting them 7+1 between the two parties. We have 4 GPUs per machine, each party gets all 4. With `PROCESS_NUM=24` and `CHEETAH_NUM_GPUS=4`, each GPU handles 6 forked processes.

**Why prefix naming controls sort order:** With the `a_`/`b_`/`c_`/`d_` scheme, `sorted()` guarantees:
1. `a_*` single-batch GPU configs run first (fast, good sanity check)
2. `b_*` multi-batch GPU configs run second
3. `c_*` single-batch CPU configs run third
4. `d_*` multi-batch CPU configs run last

Both machines will always process configs in this identical order, eliminating the sync-ordering bug.

---

## Probably Reviewed (Needs Discussion Before Proceeding)

| Item | Question |
|------|----------|
| `CHEETAH_GPU_REVERSE` | Leave at 0 (default) for initial measurements. Only add if A100 side proves to be the encryption bottleneck. Enabling it requires a ConvTriple rebuild with `-gpu-reverse`. |
| `CHEETAH_THREADS=32` in single-batch vs `=1` in multi-batch | Single-batch uses 32 threads within one process; multi-batch uses 24 processes with 1 thread each. This is intentional — just confirming you want both measured. |
| `FUNCTION_IDENTIFIER=87,187,287` in single-batch configs | This runs ResNet50-Cheetah (87), AlexNet (187), and VGG16 (287) in sequence. Requires pretrained models and datasets from `nn/Pygeon`. Confirm this is desired for initial testing or whether to start with just 87. |
| `CHEETAH_NUM_GPUS=4` for A100 vs H100 | Both machines use the same config files. The A100 also has 4 GPUs, so `CHEETAH_NUM_GPUS=4` is correct for both. Confirm. |
| `USE_CUDA_GEMM=2` in multi-batch GPU configs | The benchmarking fork observed CUTLASS GEMM is faster for large batch sizes (PROCESS_NUM=24, DATTYPE=256). Leave enabled. |

---

## Changelog

### Before this work
- Docker: ✅ already correct
- ConvTriple submodule commit: ✅ correct commit, ❌ wrong URL in .gitmodules (won't clone from fresh)
- Makefile: ❌ wrong CUDA path, ❌ no rpath, ❌ no convtriple_check, ❌ missing GPU flags, ❌ GPU arch hardcoded to 90
- config.h: ❌ missing CHEETAH_NUM_GPUS guard
- generate_beaver_tiples.hpp: ❌ all processes use GPU 0
- gemm_cutlass_int.cu: ❌ no cudaSetDevice, ❌ uint16_t compile error on H100/A100
- conv_cutlass_int.cu: ❌ uint16_t compile error on H100/A100
- run_config.py: ❌ non-deterministic ordering (critical for distributed), ❌ buffered output, ❌ no port cleanup
- GPU configs: ⚠️ partial (missing CHEETAH_FC_GPU, CHEETAH_DISCONNECT, CHEETAH_NUM_GPUS), no CPU counterparts, no sort-stable naming
### After all 9 commits
- `git clone --recurse-submodules` works on a fresh machine ✅
- `./docker-run.sh --build --gpus all` launches the correct environment ✅
- `make -j CHEETAH_GPU=1 ...` auto-detects GPU arch and rebuilds ConvTriple if needed ✅
- `PROCESS_NUM=24 CHEETAH_NUM_GPUS=4` distributes preprocessing across all 4 GPUs ✅
- `run_config.py` processes configs in identical sorted order on both machines ✅
- Both `a_*`/`b_*` GPU and `c_*`/`d_*` CPU configs present — full comparison in one folder pass ✅
- Run directly: `python3 measurements/run_config.py measurements/configs/artifacts/triad/2pc/GPU/ -p 0 -a <H100_IP> -b <A100_IP>` ✅

---

## The Old Fork's Sync Issue — Root Cause Analysis

The benchmarking fork was developed single-machine. The "sync issue" when moving to distributed likely combined several factors, all now addressed:

1. **Non-deterministic config ordering** — P0 and P1 ran different configs simultaneously and deadlocked. Fixed by `sorted()` in run_config.py (Commit 8).
2. **Ports not freed between runs** — previous run's process held a port open, next run couldn't bind. Fixed by cleanup in run_config.py (Commit 8).
3. **Socket not released between layers** — ConvTriple sockets stayed open across multiple triple generation calls. Fixed by `CHEETAH_DISCONNECT=1` (Commit 9).
4. **All processes piling onto GPU 0** — CUDA OOM or contention. Fixed by `CHEETAH_NUM_GPUS` dispatch (Commit 5).
5. **Troy GlobalPool CUDA context crash** — Fixed upstream in ConvTriple submodule (already at `c67aa63`).
6. **Expand_seed race across CUDA forks** — Fixed in ConvTriple submodule.

The principle going forward: keep `convtriple_check` in the Makefile so the stamp always reflects what was actually compiled, and never manually rebuild ConvTriple without going through `make`.
