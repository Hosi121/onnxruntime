"""Repeat selected measurements with new runtime processes for each pair."""

import argparse
import json
import os
from pathlib import Path
import statistics
import tempfile

from benchmark import cases, digest, run_pair


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--runner", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--pairs", type=int, default=5)
    parser.add_argument("--rounds", type=int, default=4)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument(
        "--model", type=Path, default=Path(__file__).parent / "ppocrv6_tiny_det.onnx"
    )
    parser.add_argument(
        "--image",
        type=Path,
        default=Path(".tools/esp-dl-upstream/examples/pp_ocr_v6/main/pp_ocr_v6.jpg"),
    )
    parser.add_argument(
        "--control",
        type=Path,
        default=Path(
            ".tools/esp-dl-upstream/examples/tutorial/how_to_quantize_model/quantize_mobilenetv2/models/torch/mobilenet_v2.onnx"
        ),
    )
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    args.cpus = sorted(os.sched_getaffinity(0))[:4]
    args.suite = "all"
    selected = {
        "convtranspose_16_16_184_184",
        "fallback_dilation",
        "ppocrv6_tiny_det_736",
        "control_mobilenetv2",
    }
    fixtures = [
        (name, model, sample) for name, model, sample in cases(args) if name in selected
    ]
    baseline, candidate = args.baseline, args.candidate
    report = {
        "libraries": {str(p): digest(p) for p in (baseline, candidate)},
        "script_sha256": digest(__file__),
        "benchmark_sha256": digest(Path(__file__).with_name("benchmark.py")),
        "fresh_process_pairs": args.pairs,
        "rounds_per_pair": args.rounds,
        "cases": [],
    }
    with tempfile.TemporaryDirectory(prefix="churin-ort-repeat-") as temporary:
        for pair in range(args.pairs):
            for name, model, sample in fixtures if pair % 2 == 0 else fixtures[::-1]:
                for threads in (1, 4):
                    args.baseline, args.candidate = baseline, candidate
                    row = run_pair(args, Path(temporary), name, model, sample, threads)
                    row.update(pair=pair, comparison="baseline_candidate")
                    report["cases"].append(row)
                    print(
                        f"Pair {pair + 1}: {name} t={threads}, {row['reduction_percent']:.2f}% less",
                        flush=True,
                    )
                    args.output.write_text(json.dumps(report, indent=2) + "\n")
            # Use identical libraries to measure variation between processes.
            for name, model, sample in fixtures:
                if name not in ("fallback_dilation", "ppocrv6_tiny_det_736"):
                    continue
                for threads in (1, 4):
                    args.baseline = args.candidate = baseline
                    row = run_pair(args, Path(temporary), name, model, sample, threads)
                    row.update(pair=pair, comparison="baseline_baseline")
                    report["cases"].append(row)
                    args.output.write_text(json.dumps(report, indent=2) + "\n")
        report["summary"] = []
        keys = sorted(
            {(r["name"], r["threads"], r["comparison"]) for r in report["cases"]}
        )
        for name, threads, comparison in keys:
            rows = [
                r
                for r in report["cases"]
                if (r["name"], r["threads"], r["comparison"])
                == (name, threads, comparison)
            ]
            reductions = [r["reduction_percent"] for r in rows]
            row = {
                "name": name,
                "threads": threads,
                "comparison": comparison,
                "median_reduction_percent": statistics.median(reductions),
                "pair_reduction_percent": reductions,
                "baseline_ms": statistics.median(r["baseline_median_ms"] for r in rows),
                "candidate_ms": statistics.median(
                    r["candidate_median_ms"] for r in rows
                ),
            }
            report["summary"].append(row)
            print(row, flush=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
