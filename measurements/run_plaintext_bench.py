"""Plaintext-baseline micro-benchmark, paired with run_fedavg_bench.py --
see flotilla/docs/secure_aggregation/overhead_report.md. Measures what
`fedavg`'s aggregator actually does with the same-shape real LeNet5
state_dict (see run_fedavg_bench.py's LENET5_PARAM_COUNT docstring for the
element-count derivation): a plain weighted-sum loop over 3 synthetic
clients' updates (mirrors aggregator_fedavg.py's own arithmetic), timed over
many iterations, plus the real one-client-to-one-server wire payload size
(`len(pickle.dumps(state_dict))`, matching client_grpc_manager.py's actual
serialization) as the "communication" figure for the plaintext path -- there
is no party-to-party tier at all in fedavg, which the report must label
explicitly `n/a`, not blank or zero, so it isn't misread as "no overhead".

Pure Python + numpy/torch -- no hpmpc build needed, so unlike
run_fedavg_bench.py this can run directly on the host, no container
required.
"""

import argparse
import json
import pickle
import statistics
import sys
import time
from collections import OrderedDict
from pathlib import Path

import numpy as np
import torch

FLOTILLA_ROOT = Path(__file__).resolve().parents[1].parent / "flotilla"
sys.path.insert(0, str(FLOTILLA_ROOT / "src"))
sys.path.insert(0, str(FLOTILLA_ROOT))

from models.LeNet5.model import LeNet5_class  # noqa: E402

NUM_CLIENTS = 3


def _real_lenet5_state_dict():
    model = LeNet5_class(args={"num_classes": 10})
    return model.state_dict()


def _weighted_sum(client_state_dicts, dataset_sizes):
    """Mirrors aggregator_fedavg.py's own weighted-sum arithmetic: for each
    layer, sum(client_tensor * dataset_size) across clients, then divide by
    the total dataset size."""
    total_weight = sum(dataset_sizes)
    layer_names = client_state_dicts[0].keys()
    result = OrderedDict()
    for layer_name in layer_names:
        acc = torch.zeros_like(client_state_dicts[0][layer_name])
        for state_dict, weight in zip(client_state_dicts, dataset_sizes):
            acc = acc + state_dict[layer_name] * weight
        result[layer_name] = acc / total_weight
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--iterations", type=int, default=1000)
    parser.add_argument(
        "--output",
        default=str(Path(__file__).resolve().parent / "plaintext_bench_results.json"),
    )
    args = parser.parse_args()

    rng = np.random.default_rng(0)
    base_state_dict = _real_lenet5_state_dict()
    param_count = sum(t.numel() for t in base_state_dict.values())

    client_state_dicts = []
    dataset_sizes = []
    for client_index in range(NUM_CLIENTS):
        state_dict = OrderedDict(
            (name, tensor.clone() + float(rng.normal(scale=0.01)))
            for name, tensor in base_state_dict.items()
        )
        client_state_dicts.append(state_dict)
        dataset_sizes.append(100 + client_index * 25)

    wall_times = []
    for _ in range(args.iterations):
        start = time.perf_counter()
        _weighted_sum(client_state_dicts, dataset_sizes)
        wall_times.append(time.perf_counter() - start)

    # Real wire payload: one client's one state_dict update to the server --
    # the literal serialization client_grpc_manager.py performs today.
    wire_payload_bytes = len(pickle.dumps(client_state_dicts[0]))

    results = {
        "param_count": param_count,
        "num_clients": NUM_CLIENTS,
        "iterations": args.iterations,
        "wall_time_s": {
            "mean": statistics.mean(wall_times),
            "median": statistics.median(wall_times),
            "p95": sorted(wall_times)[max(0, int(len(wall_times) * 0.95) - 1)],
            "min": min(wall_times),
            "max": max(wall_times),
        },
        "client_to_server_wire_bytes": wire_payload_bytes,
        "party_to_party_wire_bytes": "n/a -- fedavg has no party-to-party communication tier at all",
    }
    Path(args.output).write_text(json.dumps(results, indent=2))
    print(json.dumps(results, indent=2))
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
