"""Record the CPU ConvTranspose nodes that run in the OCR detector."""

import argparse
import json
import os
from pathlib import Path
import tempfile

from benchmark import Runner, cases, digest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--library", type=Path, required=True)
    parser.add_argument("--runner", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    args.suite = "models"
    args.control = None
    name, model, sample = next(
        row for row in cases(args) if row[0] == "ppocrv6_tiny_det_736"
    )
    with tempfile.TemporaryDirectory(prefix="ort-profile-") as temp:
        directory = Path(temp)
        data = directory / "input.bin"
        sample.tofile(data)
        runner = Runner(
            args.runner, args.library, model, data, sample.shape,
            1, [min(os.sched_getaffinity(0))], directory / "profile"
        )
        try:
            runner.command("warmup 5")
        finally:
            runner.close()
        events = json.loads(next(directory.glob("profile*.json")).read_text())
        nodes = {}
        for event in events:
            info = event.get("args", {})
            if info.get("op_name") == "ConvTranspose":
                entry = nodes.setdefault(event["name"], {"calls": 0, "args": info})
                entry["calls"] += 1
    report = {
        "model": name,
        "model_sha256": digest(model),
        "library_sha256": digest(args.library),
        "nodes": nodes,
        "scope": "Node names, shapes, and CPU placement only. These are not benchmark times.",
    }
    assert len(nodes) == 2
    assert all(row["calls"] == 5 for row in nodes.values())
    assert all(row["args"]["provider"] == "CPUExecutionProvider" for row in nodes.values())
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
