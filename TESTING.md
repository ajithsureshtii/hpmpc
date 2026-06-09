# Multi-Machine GPU Benchmark Testing Guide

Two-machine distributed setup: **P0 (H100, 4× 80 GB)** and **P1 (A100, 4× 80 GB)**.
Protocol: ABY2 / Cheetah (PROTOCOL=4). Framework: HPMPC + ConvTriple.

---

## Prerequisites

- Both machines cloned from `https://github.com/ajithsureshtii/hpmpc`, branch `aby2-merge`
- Docker installed with NVIDIA Container Toolkit (`nvidia-docker2`)
- SSH key added to GitHub on both machines (needed to clone submodules over SSH)
- Machines can reach each other over the network (test with `ping`)
- Know the IP addresses:
  - `H100_IP` — IP of the H100 machine (P0)
  - `A100_IP` — IP of the A100 machine (P1)

---

## Part 1 — Initial Setup (both machines)

### 1.1 Clone the repo

```bash
git clone --recurse-submodules -b aby2-merge \
    git@github.com:ajithsureshtii/hpmpc.git
cd hpmpc
```

If you forgot `--recurse-submodules`:
```bash
git submodule update --init --recursive
```

### 1.2 Build the Docker image

```bash
./docker-run.sh --build --gpus all
```

This builds from `docker/Dockerfile` (CUDA 12.9, Ubuntu 24.04) and tags it `hpmpc-gpu`.

### 1.3 Start the container

```bash
./docker-run.sh --gpus all
```

The repo directory is mounted at `/hpmpc` inside the container. All subsequent commands
run **inside the container** unless noted otherwise.

---

## Part 2 — Build ConvTriple (GPU HE preprocessing library)

ConvTriple must be built once per machine. It patches and compiles troy-nova (GPU HE lib).
The GPU architecture is detected automatically from the GPU in the container.

```bash
cd /hpmpc/nn/ConvTriple

# Build deps (SEAL + troy-nova, applies GlobalPool and matmul seed-expansion patches)
GPU_ARCHITECTURE=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1 | tr -d '.')
GPU_ARCHITECTURE=$GPU_ARCHITECTURE ./deps.sh -gpu

# Build the ConvTriple library
./build.sh -gpu
```

> **Note:** `deps.sh -gpu` takes 15–30 minutes on first run. It compiles SEAL and troy-nova.
> Subsequent builds reuse the `deps/` directory unless you run `rm -rf deps build` first.

### Verify: smoke-test GPU triple generation

After the build, optionally run the standalone test (both machines, simultaneously):

```bash
# H100 (P0 = party 0):
cd /hpmpc/nn/ConvTriple/build/bin
./conv_triple_gpu_test 0 <A100_IP>

# A100 (P1 = party 1), at the same time:
cd /hpmpc/nn/ConvTriple/build/bin
./conv_triple_gpu_test 1 <H100_IP>
```

You should see per-process GPU assignment and timing printed for each child process.

---

## Part 3 — Build HPMPC

The hpmpc Makefile auto-detects GPU arch and rebuilds ConvTriple if GPU flags change.
Build flags are baked in at compile time.

### 3.1 GPU build (runs GPU preprocessing + GPU online GEMM)

Run on **both machines** simultaneously (they must compile the same target):

```bash
cd /hpmpc
make -j CHEETAH_GPU=1 CHEETAH_FC_GPU=1 CHEETAH_NUM_GPUS=4 \
        USE_CUDA_GEMM=2 PROTOCOL=4 DATTYPE=256 PROCESS_NUM=24 \
        BITLENGTH=32 PARTY=all
```

> The first GPU build is slow — it recompiles ConvTriple against the detected arch (sm_90
> on H100, sm_80 on A100). Subsequent builds with the same flags are fast (stamp check).

### 3.2 CPU build (for baseline comparison)

```bash
cd /hpmpc
make -j CHEETAH_GPU=0 USE_CUDA_GEMM=0 PROTOCOL=4 DATTYPE=32 \
        BITLENGTH=32 PARTY=all
```

---

## Part 4 — Run Benchmarks via `run_config.py`

All configs are in `measurements/configs/artifacts/triad/2pc/GPU/`.
Run on **both machines at the same time** using the correct party ID.

### Config naming convention

| Prefix | Batch mode | GPU/CPU |
|---|---|---|
| `a_*` | Single batch (DATTYPE=32, PROCESS_NUM=1) | GPU |
| `b_*` | Multi batch (DATTYPE=256, PROCESS_NUM=24) | GPU |
| `c_*` | Single batch | CPU baseline |
| `d_*` | Multi batch | CPU baseline |

Files are processed in alphabetical order — `a_*` first through `d_*` last. Both machines
**must** process configs in the same order (guaranteed by `sorted()` in `run_config.py`).

---

### 4.1 Run all configs (GPU + CPU, full comparison)

This runs all 8 configs in sequence: single-batch GPU → multi-batch GPU → single-batch CPU →
multi-batch CPU. Both machines must start this simultaneously.

**H100 machine (P0):**
```bash
cd /hpmpc
python3 measurements/run_config.py \
    measurements/configs/artifacts/triad/2pc/GPU/ \
    -p 0 -a <H100_IP> -b <A100_IP>
```

**A100 machine (P1), at the same time:**
```bash
cd /hpmpc
python3 measurements/run_config.py \
    measurements/configs/artifacts/triad/2pc/GPU/ \
    -p 1 -a <H100_IP> -b <A100_IP>
```

Logs are saved to `measurements/logs/` with timestamps.

---

### 4.2 GPU only (skip CPU configs)

Pass individual GPU config files instead of the folder:

```bash
python3 measurements/run_config.py \
    measurements/configs/artifacts/triad/2pc/GPU/a_2PC_single_batch_all_opt_A2bits.conf \
    measurements/configs/artifacts/triad/2pc/GPU/a_2PC_single_batch_all_opt_reshared.conf \
    measurements/configs/artifacts/triad/2pc/GPU/b_2PC_multi_batch_all_opt_A2bits.conf \
    measurements/configs/artifacts/triad/2pc/GPU/b_2PC_multi_batch_all_opt_reshared.conf \
    -p 0 -a <H100_IP> -b <A100_IP>
```

---

### 4.3 1 GPU vs 4 GPU comparison

Override `CHEETAH_NUM_GPUS` at runtime using `--override`:

**1 GPU run:**
```bash
# Both machines simultaneously
python3 measurements/run_config.py \
    measurements/configs/artifacts/triad/2pc/GPU/ \
    -p 0 -a <H100_IP> -b <A100_IP> \
    --override CHEETAH_NUM_GPUS=1 CHEETAH_GPU=1 CHEETAH_FC_GPU=1
```

**4 GPU run** (default, already in configs — no override needed):
```bash
python3 measurements/run_config.py \
    measurements/configs/artifacts/triad/2pc/GPU/ \
    -p 0 -a <H100_IP> -b <A100_IP>
```

> Run them back-to-back (one after the other on both machines) to compare.
> Keep the `b_*` multi-batch GPU configs in scope — those use `PROCESS_NUM=24`
> which is where multi-GPU parallelism has the most impact.

---

### 4.4 CPU baseline only

```bash
python3 measurements/run_config.py \
    measurements/configs/artifacts/triad/2pc/GPU/c_2PC_single_batch_all_opt_A2bits_CPU.conf \
    measurements/configs/artifacts/triad/2pc/GPU/c_2PC_single_batch_all_opt_reshared_CPU.conf \
    measurements/configs/artifacts/triad/2pc/GPU/d_2PC_multi_batch_all_opt_A2bits_CPU.conf \
    measurements/configs/artifacts/triad/2pc/GPU/d_2PC_multi_batch_all_opt_reshared_CPU.conf \
    -p 0 -a <H100_IP> -b <A100_IP>
```

---

## Part 5 — Understanding the Results

### What each config measures

| Config | Network | CHEETAH_GPU | USE_CUDA_GEMM | PROCESS_NUM | Key observation |
|---|---|---|---|---|---|
| `a_*_A2bits` | ResNet50/AlexNet/VGG (single batch) | 1 | 0 | 1 | GPU HE preprocessing benefit |
| `a_*_reshared` | Same + reshare optimisation | 1 | 2 | 1 | GPU preprocessing + GPU online GEMM |
| `b_*_A2bits` | ResNet50 (24 parallel) | 1 | 2 | 24 | Full 4-GPU load distribution |
| `b_*_reshared` | Same + reshare | 1 | 2 | 24 | Full pipeline GPU acceleration |
| `c_*` | Same as `a_*` | 0 | 0 | 1 | CPU HE preprocessing baseline |
| `d_*` | Same as `b_*` | 0 | 0 | 24 | CPU baseline for batched workload |

### Key metrics to compare

- **Preprocessing time (PRE)**: GPU vs CPU HE triple generation — expect 5–10× speedup
  with `CHEETAH_GPU=1` vs `CHEETAH_GPU=0`
- **1 GPU vs 4 GPU**: With `PROCESS_NUM=24`, expect ~4× preprocessing speedup going from
  `CHEETAH_NUM_GPUS=1` to `CHEETAH_NUM_GPUS=4` (24 processes spread across 4 GPUs vs 1)
- **Online time**: `USE_CUDA_GEMM=2` accelerates the online matrix multiply on GPU

### Reading the logs

Logs land in `measurements/logs/`. Each run produces two files:
- `<config_name>_<timestamp>.log` — full stdout including make output and timing
- `<config_name>_<timestamp>.log-config` — the exact parameters used

Search for timing:
```bash
grep -E "PRE|ONLINE|triple time" measurements/logs/<logfile>.log
```

---

## Part 6 — Network Connectivity Check

Before running distributed benchmarks, verify connectivity and measure bandwidth:

```bash
# From H100, test TCP throughput to A100:
iperf3 -s                            # on A100
iperf3 -c <A100_IP> -t 10            # on H100

# Latency:
ping <A100_IP> -c 10
```

HPMPC uses multiple TCP ports starting from the base port. Ensure the firewall allows
outbound connections from both machines on the relevant port range.

---

## Part 7 — Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| Both machines hang forever | Config ordering mismatch or one side crashed | Check logs; both sides must start `run_config.py` close together |
| `libHE.so not found` at runtime | Missing rpath or deps not built | Rebuild ConvTriple (`./build.sh -gpu`); check `nn/ConvTriple/build/lib/` |
| `[Array::to_device_inplace] Memory pool handle required` | Troy GlobalPool not patched | `rm -rf nn/ConvTriple/deps nn/ConvTriple/build` and redo `deps.sh -gpu` |
| `Argument contains seed` on FC triples | Troy matmul patch not applied | Same as above — redo deps |
| `Column size mismatch` | pack_lwe mismatch (old build) | Full clean rebuild of ConvTriple |
| Build fails with `USE_CUDA_GEMM=2`, uint16_t error | Old CUTLASS INT16 issue | Already fixed in `conv_cutlass_int.cu` and `gemm_cutlass_int.cu` |
| `convtriple_check` rebuilds ConvTriple unexpectedly | GPU flags changed | Expected behaviour — stamp file tracks flags. Let it complete. |
| Process count mismatch | `PROCESS_NUM` differs between make and config | Config value wins at runtime; ensure make and config agree |

---

## Quick Reference

```bash
# Full benchmark run — both machines simultaneously
# H100 (P0):
python3 measurements/run_config.py measurements/configs/artifacts/triad/2pc/GPU/ \
    -p 0 -a <H100_IP> -b <A100_IP>

# A100 (P1):
python3 measurements/run_config.py measurements/configs/artifacts/triad/2pc/GPU/ \
    -p 1 -a <H100_IP> -b <A100_IP>

# 1-GPU override (for comparison):
python3 measurements/run_config.py measurements/configs/artifacts/triad/2pc/GPU/ \
    -p 0 -a <H100_IP> -b <A100_IP> --override CHEETAH_NUM_GPUS=1

# Rebuild ConvTriple from scratch (after troy patch changes or first run on new machine):
cd /hpmpc/nn/ConvTriple
rm -rf deps build
GPU_ARCHITECTURE=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1 | tr -d '.')
GPU_ARCHITECTURE=$GPU_ARCHITECTURE ./deps.sh -gpu && ./build.sh -gpu
```
