"""Compare two builds with the same CPU session settings and input data."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import statistics
import subprocess
import tempfile

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper
from PIL import Image


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class Runner:
    def __init__(
        self, executable, library, model, data, shape, threads, cpus, profile=None
    ):
        command = [
            str(executable),
            str(library),
            str(model),
            str(data),
            ",".join(map(str, shape)),
            str(threads),
        ]
        if profile:
            command.append(str(profile))
        self.process = subprocess.Popen(
            command,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            text=True,
            preexec_fn=lambda: os.sched_setaffinity(0, cpus),
        )
        self.info = self.read()
        assert self.info["ready"]

    def read(self):
        line = self.process.stdout.readline()
        if not line:
            raise RuntimeError(f"Runner stopped: {self.process.poll()}")
        return json.loads(line)

    def command(self, command):
        self.process.stdin.write(command + "\n")
        self.process.stdin.flush()
        return self.read()

    def close(self):
        if self.process.poll() is None:
            self.process.stdin.write("quit\n")
            self.process.stdin.flush()
        self.process.wait(timeout=30)
        assert self.process.returncode == 0


def operator_model(
    ci,
    co,
    height,
    width,
    rng,
    kernel=2,
    stride=2,
    pads=None,
    dilation=1,
    extra=None,
    group=1,
):
    pads = pads or [0, 0, 0, 0]
    extra = extra or [0, 0]
    weights = (
        rng.standard_normal((ci, co // group, kernel, kernel)) / np.sqrt(ci)
    ).astype(np.float32)
    bias = rng.standard_normal(co).astype(np.float32)
    shape = [1, ci, height, width]
    spatial = [
        (d - 1) * stride
        - pads[i]
        - pads[i + 2]
        + dilation * (kernel - 1)
        + 1
        + extra[i]
        for i, d in enumerate((height, width))
    ]
    node = helper.make_node(
        "ConvTranspose",
        ["X", "W", "B"],
        ["Y"],
        kernel_shape=[kernel, kernel],
        strides=[stride, stride],
        pads=pads,
        dilations=[dilation, dilation],
        output_padding=extra,
        group=group,
    )
    graph = helper.make_graph(
        [node],
        "col2im_cpu",
        [helper.make_tensor_value_info("X", TensorProto.FLOAT, shape)],
        [helper.make_tensor_value_info("Y", TensorProto.FLOAT, [1, co, *spatial])],
        [numpy_helper.from_array(weights, "W"), numpy_helper.from_array(bias, "B")],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 10
    onnx.checker.check_model(model)
    return model, rng.standard_normal(shape).astype(np.float32)


def cases(args):
    rng = np.random.default_rng(20260920)
    if args.suite in ("all", "operators"):
        for ci, co, h, w in [
            (16, 1, 368, 368),
            (16, 16, 184, 184),
            (64, 32, 64, 64),
            (128, 64, 32, 32),
            (1, 1, 1, 1),
            (3, 5, 1, 7),
        ]:
            model, sample = operator_model(ci, co, h, w, rng)
            yield f"convtranspose_{ci}_{co}_{h}_{w}", model, sample
        for name, options in [
            ("overlap", {"kernel": 3}),
            ("padding", {"pads": [1, 1, 1, 1]}),
            ("dilation", {"dilation": 2}),
            ("output_padding", {"extra": [1, 1]}),
        ]:
            model, sample = operator_model(16, 16, 64, 64, rng, **options)
            yield f"fallback_{name}", model, sample
    if args.suite in ("all", "models"):
        original = Image.open(args.image).convert("RGB")
        for size in (256, 736, 1024):
            image = np.asarray(
                original.resize((size, size), Image.Resampling.BILINEAR),
                dtype=np.float32,
            )
            image = image[:, :, ::-1] / np.float32(255)
            image = (
                image - np.array([0.485, 0.456, 0.406], dtype=np.float32)
            ) / np.array([0.229, 0.224, 0.225], dtype=np.float32)
            sample = np.ascontiguousarray(image.transpose(2, 0, 1)[None])
            yield f"ppocrv6_tiny_det_{size}", args.model, sample
        image = np.asarray(
            original.resize((224, 224), Image.Resampling.BILINEAR), dtype=np.float32
        ) / np.float32(255)
        image = (image - np.array([0.485, 0.456, 0.406], dtype=np.float32)) / np.array(
            [0.229, 0.224, 0.225], dtype=np.float32
        )
        yield (
            "control_mobilenetv2",
            args.control,
            np.ascontiguousarray(image.transpose(2, 0, 1)[None]),
        )


def run_pair(args, directory, name, model, sample, threads):
    model_path = model
    if isinstance(model, onnx.ModelProto):
        model_path = directory / "case.onnx"
        onnx.save(model, model_path)
    input_path = directory / "input.bin"
    sample.tofile(input_path)
    cpus = args.cpus[:threads]
    runners = []
    row = {
        "name": name,
        "shape": list(sample.shape),
        "threads": threads,
        "cpus": cpus,
        "model_sha256": digest(model_path),
        "input_sha256": digest(input_path),
        "round_ms": [[], []],
        "checks": [],
    }
    try:
        for library in (args.baseline, args.candidate):
            runners.append(
                Runner(
                    args.runner,
                    library,
                    model_path,
                    input_path,
                    sample.shape,
                    threads,
                    cpus,
                )
            )
        row["builds"] = [runner.info for runner in runners]
        warmup = [r.command("warmup 10") for r in runners]
        # Each timed block lasts at least about 100 ms on the slower build.
        iterations = max(
            args.iterations, min(100000, int(100 / max(r["ms"] for r in warmup)))
        )
        row["iterations_per_round"] = iterations
        for turn in range(args.rounds):
            for index in (0, 1) if turn % 2 == 0 else (1, 0):
                result = runners[index].command(f"time {iterations}")
                row["round_ms"][index].append(result["ms"])
            paths = [directory / f"output_{i}.bin" for i in range(2)]
            info = [
                runner.command(f"dump {path}")
                for runner, path in zip(runners, paths, strict=True)
            ]
            outputs = [path.read_bytes() for path in paths]
            if outputs[0] != outputs[1]:
                arrays = [np.frombuffer(output, dtype=np.float32) for output in outputs]
                raise AssertionError(
                    f"Output mismatch in {name}: {np.max(np.abs(arrays[0] - arrays[1]))}"
                )
            row["checks"].append(
                {
                    "bitwise_equal": True,
                    "elements": info[0]["elements"],
                    "sha256": hashlib.sha256(outputs[0]).hexdigest(),
                }
            )
    finally:
        for runner in runners:
            runner.close()
    base, changed = [statistics.median(times) for times in row["round_ms"]]
    row.update(
        baseline_median_ms=base,
        candidate_median_ms=changed,
        reduction_percent=100 * (1 - changed / base),
        speedup=base / changed,
        paired_reduction_percent=[
            100 * (1 - b / a) for a, b in zip(*row["round_ms"], strict=True)
        ],
    )
    return row


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--runner", type=Path, required=True)
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
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--suite", choices=("all", "operators", "models"), default="all"
    )
    parser.add_argument("--rounds", type=int, default=8)
    parser.add_argument("--iterations", type=int, default=20)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    args.cpus = sorted(os.sched_getaffinity(0))[:4]
    report = {
        "platform": platform.platform(),
        "optimization": "ORT_ENABLE_ALL",
        "provider": "CPUExecutionProvider",
        "intra_op_allow_spinning": False,
        "rounds": args.rounds,
        "input_note": "BGR, square resize, ImageNet normalization.",
        "libraries": {str(p): digest(p) for p in (args.baseline, args.candidate)},
        "script_sha256": digest(__file__),
        "cases": [],
    }
    with tempfile.TemporaryDirectory(prefix="churin-ort-integration-") as temporary:
        for name, model, sample in cases(args):
            for threads in (1, 4):
                row = run_pair(args, Path(temporary), name, model, sample, threads)
                report["cases"].append(row)
                args.output.write_text(json.dumps(report, indent=2) + "\n")
                print(
                    f"{name} threads={threads}: {row['baseline_median_ms']:.4f} -> "
                    f"{row['candidate_median_ms']:.4f} ms ({row['reduction_percent']:.1f}% less)",
                    flush=True,
                )


if __name__ == "__main__":
    main()
