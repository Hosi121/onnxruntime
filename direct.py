"""Call the actual Linux Col2im symbols with shared input and output buffers.

This check excludes GEMM and the session. The timed loop includes ctypes calls.
The tested Set helper does not use its CPUMathUtil pointer.
"""

import argparse
import ctypes
import hashlib
import json
import os
from pathlib import Path
import statistics
import subprocess
import time

import numpy as np


def load(path):
    symbols = subprocess.check_output(["nm", "-S", str(path)], text=True).splitlines()
    api = [s.split() for s in symbols if s.split()[-1].split("@")[0] == "OrtGetApiBase"]
    col = [s.split() for s in symbols if "4math6Col2imIfNS_11CPUMathUtilELi2EEE" in s]
    assert len(api) == len(col) == 1
    library = ctypes.CDLL(str(path.resolve()))
    base = ctypes.cast(library.OrtGetApiBase, ctypes.c_void_p).value - int(
        api[0][0], 16
    )
    pointer = ctypes.POINTER(ctypes.c_float)
    signature = ctypes.CFUNCTYPE(
        None, pointer, *([ctypes.c_int64] * 13), pointer, ctypes.c_void_p
    )
    function = signature(base + int(col[0][0], 16))
    return (
        library,
        function,
        {
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "symbol": col[0][-1],
            "symbol_bytes": int(col[0][1], 16),
        },
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    cpu = min(os.sched_getaffinity(0))
    os.sched_setaffinity(0, [cpu])
    loaded = [load(path) for path in (args.baseline, args.candidate)]
    report = {
        "cpu": cpu,
        "libraries": [r[2] for r in loaded],
        "rounds": 12,
        "iterations": 250,
        "scope": "Actual Col2im symbols, shared buffers, no GEMM, no session. ctypes calls are timed.",
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "cases": [],
    }
    rng = np.random.default_rng(20)
    for name, geometry in [
        ("fast", (16, 368, 368, 2, 2, 1, 1, 0, 0, 0, 0, 2, 2)),
        ("dilation", (16, 129, 129, 2, 2, 2, 2, 0, 0, 0, 0, 2, 2)),
        ("overlap", (16, 129, 129, 3, 3, 1, 1, 0, 0, 0, 0, 2, 2)),
        ("padding", (16, 126, 126, 2, 2, 1, 1, 1, 1, 1, 1, 2, 2)),
        ("output_padding", (16, 129, 129, 2, 2, 1, 1, 0, 0, 0, 0, 2, 2)),
        ("padding_stride1", (16, 64, 64, 3, 3, 1, 1, 1, 1, 1, 1, 1, 1)),
        ("dilation_stride1", (16, 66, 66, 2, 2, 2, 2, 0, 0, 0, 0, 1, 1)),
        ("padding_kernel4", (16, 128, 128, 4, 4, 1, 1, 1, 1, 1, 1, 2, 2)),
    ]:
        c, h, w, kh, kw, dh, dw, pt, pl, pb, pr, sh, sw = geometry
        oh = (h + pt + pb - (dh * (kh - 1) + 1)) // sh + 1
        ow = (w + pl + pr - (dw * (kw - 1) + 1)) // sw + 1
        source = rng.uniform(-1, 1, c * kh * kw * oh * ow).astype(np.float32)
        output = np.full(c * h * w + 2, np.float32(12345))
        pointer = ctypes.POINTER(ctypes.c_float)
        arguments = (
            source.ctypes.data_as(pointer),
            *geometry,
            output[1:-1].ctypes.data_as(pointer),
            None,
        )
        loaded[0][1](*arguments)
        reference = output.copy()
        for entry in loaded:
            for _ in range(10):
                entry[1](*arguments)
        timings = [[], []]
        for turn in range(report["rounds"]):
            for index in (0, 1) if turn % 2 == 0 else (1, 0):
                function = loaded[index][1]
                start = time.perf_counter_ns()
                for _ in range(report["iterations"]):
                    function(*arguments)
                timings[index].append(
                    (time.perf_counter_ns() - start) / 1e6 / report["iterations"]
                )
                assert output.tobytes() == reference.tobytes()
                assert output[0] == output[-1] == 12345
        base, candidate = map(statistics.median, timings)
        row = {
            "name": name,
            "geometry": geometry,
            "round_ms": timings,
            "baseline_ms": base,
            "candidate_ms": candidate,
            "reduction_percent": 100 * (1 - candidate / base),
            "bitwise_equal": True,
        }
        report["cases"].append(row)
        args.output.write_text(json.dumps(report, indent=2) + "\n")
        print(
            name,
            f"{base:.5f} -> {candidate:.5f} ms ({row['reduction_percent']:.2f}% less)",
            flush=True,
        )


if __name__ == "__main__":
    main()
