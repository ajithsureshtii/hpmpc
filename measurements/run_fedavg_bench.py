"""Micro-benchmark for Flotilla's secure_mpc hpmpc backends (see
flotilla/docs/secure_aggregation/overhead_report.md). For each protocol,
runs the REAL production code path -- HpmpcBackend.run_aggregation_round,
imported directly from flotilla/src, not reimplemented -- with all of that
protocol's party processes on loopback within ONE container. This isolates
the protocol's own timing/communication overhead from cross-container
Docker-network jitter (still measured, at a coarser level, by Phase E's real
multi-container end-to-end runs).

Payload size: 61,706 elements, the real LeNet5 model's exact parameter count
(models/LeNet5/model.py) -- computed via
`sum(p.numel() for p in state_dict.values())`, not guessed; see this
script's own LENET5_PARAM_COUNT constant and the overhead report's method
section for that computation.

Client count is fixed at 3 and NOT varied as a benchmark dimension:
backend_hpmpc.py sums every selected client's share fields itself, in
Python, BEFORE ever invoking the compiled binary (see that module's
docstring) -- the binary only ever sees ONE already-summed vector per round,
regardless of how many clients contributed to it. So this benchmark
constructs one already-representative vector's shares directly; simulating
N separate clients and summing them in Python would exercise the exact same
hpmpc subprocess call with an identical input, at the cost of N times the
Python-side sharing work for zero additional signal about the MPC
protocol's own cost.

Communication/timing numbers come from hpmpc's own stdout, which
protocol_executer.hpp's shared driver code prints unconditionally (no extra
build flags) regardless of which FUNCTION is compiled in -- see
"Sending to other players: ... MB" (core/utils/print.hpp's
print_communication(), called once per round from the init-phase
finalize()) and "Time measured to perform ... clock: ...s" (this file's own
per-round instrumentation). This script captures each round's raw stdout,
assembles a synthetic multi-run log file in hpmpc's own run_config.py
convention, and reuses parse_logs.py's existing regex-based parser on it
rather than reimplementing that parsing.
"""

import argparse
import asyncio
import json
import os
import statistics
import sys
import time
from pathlib import Path
from unittest.mock import patch

import numpy as np

HPMPC_ROOT = Path(__file__).resolve().parents[1]
FLOTILLA_SRC = HPMPC_ROOT.parent / "flotilla" / "src"
sys.path.insert(0, str(FLOTILLA_SRC))
sys.path.insert(0, str(Path(__file__).resolve().parent))

# FedLogger (server/secure_agg/backends/backend_hpmpc.py's self._logger)
# reads config/logger.conf relative to the process's CWD, exactly like every
# other Flotilla entrypoint (flo_server.py, flo_secure_agg_party.py) that's
# always launched from src/ -- match that convention here too, since this
# benchmark constructs the real HpmpcBackend class unmodified.
os.chdir(FLOTILLA_SRC)

from server.secure_agg.backends.backend_hpmpc import HpmpcBackend  # noqa: E402
from server.secure_agg.backends.base import PartyEndpoint, TensorSpec  # noqa: E402
from server.secure_agg.fixed_point_codec import FixedPointCodec  # noqa: E402
from server.secure_agg.sharing_schemes.replicated3pc import Replicated3PCScheme  # noqa: E402
from server.secure_agg.sharing_schemes.tetrad4pc import Tetrad4PCScheme  # noqa: E402

import parse_logs  # noqa: E402

# See models/LeNet5/model.py -- verified via
# `sum(p.numel() for p in LeNet5_class(...).state_dict().values())` = 61706
# (conv1: 150+6, conv2: 2400+16, fc1: 48000+120, fc2: 10080+84, fc3: 840+10).
LENET5_PARAM_COUNT = 61706

BITLENGTH = 64
FRAC_BITS = 13
ROUND_TIMEOUT_S = 30.0

PROTOCOLS = {
    "replicated": {"protocol": 2, "num_parties": 3, "scheme": "replicated3pc"},
    "trio": {"protocol": 5, "num_parties": 3, "scheme": "replicated3pc"},
    "tetrad": {"protocol": 8, "num_parties": 4, "scheme": "tetrad4pc"},
}


def _make_scheme(name):
    if name == "replicated3pc":
        return Replicated3PCScheme(bitlength=BITLENGTH)
    if name == "tetrad4pc":
        return Tetrad4PCScheme(bitlength=BITLENGTH)
    raise ValueError(f"unknown sharing scheme: {name}")


async def _run_one_round(protocol_name, executable_root, tmp_root, round_index, captured_stdout):
    cfg = PROTOCOLS[protocol_name]
    codec = FixedPointCodec(bitlength=BITLENGTH, frac_bits=FRAC_BITS)
    scheme = _make_scheme(cfg["scheme"])
    rng = np.random.default_rng(round_index)

    plaintext = rng.uniform(-1.0, 1.0, size=LENET5_PARAM_COUNT).astype(np.float64)
    fixedpoint = codec.encode(plaintext)
    party_shares = {s.party_index: s for s in scheme.share(fixedpoint, rng)}

    executable_dir = str(Path(executable_root) / protocol_name)
    tensor_specs = {"w": TensorSpec(layer_name="w", shape=(LENET5_PARAM_COUNT,), dtype="float32")}

    backends = []
    for party_index in range(cfg["num_parties"]):
        backend = HpmpcBackend(
            party_index=party_index,
            num_parties=cfg["num_parties"],
            codec=codec,
            protocol=cfg["protocol"],
            executable_dir=executable_dir,
            tmp_dir=str(Path(tmp_root) / protocol_name / str(party_index) / str(round_index)),
            log_stdout=True,
        )
        peers = [
            PartyEndpoint(party_index=p, host="127.0.0.1", port=0)
            for p in range(cfg["num_parties"])
            if p != party_index
        ]
        await backend.start(peers)
        backends.append(backend)

    round_id = f"bench-{protocol_name}:{round_index}"

    def _make_capturing_info(party_index):
        def _info(event_name, message):
            if event_name == "fedparty.hpmpc.round_stdout":
                captured_stdout.append((party_index, message))

        return _info

    # Patch each backend's own logger.info to capture hpmpc's raw stdout
    # instead of relying on FedLogger's real sink -- same technique
    # tests/unit/test_backend_hpmpc.py's
    # test_run_aggregation_round_logs_stdout_when_log_stdout_enabled already
    # uses to verify this call happens; here it's the payload this whole
    # benchmark actually wants.
    patches = [
        patch.object(backend._logger, "info", side_effect=_make_capturing_info(i))
        for i, backend in enumerate(backends)
    ]
    for p in patches:
        p.start()
    wall_start = time.perf_counter()
    try:
        await asyncio.gather(
            *(
                backend.run_aggregation_round(
                    round_id=round_id,
                    shares={"client1": {"w": party_shares[i]}},
                    tensor_specs=tensor_specs,
                    timeout_s=ROUND_TIMEOUT_S,
                )
                for i, backend in enumerate(backends)
            )
        )
    finally:
        wall_elapsed_s = time.perf_counter() - wall_start
        for p in patches:
            p.stop()
    return wall_elapsed_s


def _write_synthetic_log(protocol_name, all_rounds_stdout, log_path):
    """Assembles a log file in run_config.py's own convention (`====== Run
    i/N ======` delimiters + a `Running: KEY=VAL` line) purely so
    parse_logs.py's existing, unmodified parser can be reused on it exactly
    as it parses hpmpc's own real benchmark logs."""
    cfg = PROTOCOLS[protocol_name]
    lines = []
    for round_index, per_party_stdout in enumerate(all_rounds_stdout, start=1):
        lines.append(f"====== Run {round_index}/{len(all_rounds_stdout)} ======")
        lines.append(
            f"Running: PROTOCOL={cfg['protocol']} BITLENGTH={BITLENGTH} "
            f"DATTYPE=64 NUM_INPUTS={LENET5_PARAM_COUNT} FUNCTION_IDENTIFIER=90"
        )
        for _party_index, message in per_party_stdout:
            lines.append(message)
    log_path.write_text("\n".join(lines) + "\n")


async def _bench_protocol(protocol_name, iterations, executable_root, tmp_root, log_dir):
    all_rounds_stdout = []
    wall_times = []
    for round_index in range(iterations):
        captured = []
        wall_elapsed_s = await _run_one_round(
            protocol_name, executable_root, tmp_root, round_index, captured
        )
        wall_times.append(wall_elapsed_s)
        all_rounds_stdout.append(captured)
        print(f"  [{protocol_name}] round {round_index + 1}/{iterations}: {wall_elapsed_s:.4f}s wall")

    log_path = Path(log_dir) / f"{protocol_name}_bench.log"
    _write_synthetic_log(protocol_name, all_rounds_stdout, log_path)
    parsed_runs = parse_logs.parse_log_file(str(log_path))

    return {
        "protocol_name": protocol_name,
        "iterations": iterations,
        "wall_time_s": {
            "mean": statistics.mean(wall_times),
            "median": statistics.median(wall_times),
            "p95": sorted(wall_times)[max(0, int(len(wall_times) * 0.95) - 1)],
            "min": min(wall_times),
            "max": max(wall_times),
        },
        "hpmpc_reported": parsed_runs,
        "log_file": str(log_path),
    }


async def main_async(args):
    results = {}
    for protocol_name in args.protocols:
        print(f"Benchmarking {protocol_name} ({args.iterations} rounds, "
              f"{LENET5_PARAM_COUNT} elements)...")
        results[protocol_name] = await _bench_protocol(
            protocol_name, args.iterations, args.executable_root, args.tmp_root, args.log_dir
        )
    Path(args.output).write_text(json.dumps(results, indent=2, default=str))
    print(f"Wrote {args.output}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--protocols", nargs="+", default=list(PROTOCOLS), choices=list(PROTOCOLS),
        help="Which protocols to benchmark (default: all)",
    )
    parser.add_argument("--iterations", type=int, default=30, help="Rounds per protocol (default: 30)")
    parser.add_argument(
        "--executable-root", default=str(HPMPC_ROOT / "executables"),
        help="Directory containing executables/<name>/ subdirs (default: hpmpc/executables)",
    )
    parser.add_argument("--tmp-root", default="/tmp/fedavg_bench", help="Scratch dir for input/output files")
    parser.add_argument("--log-dir", default=str(HPMPC_ROOT / "measurements" / "logs"), help="Where to write synthetic log files")
    parser.add_argument("--output", default=str(HPMPC_ROOT / "measurements" / "fedavg_bench_results.json"))
    args = parser.parse_args()

    Path(args.log_dir).mkdir(parents=True, exist_ok=True)
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
