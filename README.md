# CPU Col2im in ONNX Runtime

This test compares two source builds of ONNX Runtime. The base commit is
`09dfa6ad06ed8072b2fbe57687d4f71c7914b025`. The change is on
[`perf/cpu-col2im`](https://github.com/Hosi121/onnxruntime/tree/perf/cpu-col2im).

The change has two parts in the FP32 NCHW Col2im function:

1. For a 2x2 kernel with stride 2, zero padding, dilation 1, and even output
   dimensions, write the four adjacent output values together. Each output has
   one source value. Keep the addition to positive zero to preserve signed zero.
2. For padding or dilation, calculate the valid source columns before the row
   loop. Remove the bounds check from the inner loop. This also lets the compiler
   use SIMD for stride 1. The sum order for each output does not change.

Both changes keep the GEMM, model, weights, thread settings, and API unchanged.
The first path remains O(Q) for Q output values. Its source-level memory access
falls from about 16Q to 8Q bytes for FP32. These figures exclude cache effects,
write allocation, and GEMM. The full GEMM result buffer remains allocated. This
change does not reduce allocated RAM.

## Measurement

The host is an Intel Core Ultra 7 255H, Linux x86-64 under WSL2. Both builds use
GCC 13.3, CMake 4.2.1, Ninja, Release `-O3`, and the CPU execution provider.
They do not use `-march=native` or fast math. See `build_source.json` for the
base build command and compiler flags. Its candidate fields describe the first
trial, not the final change. See `final.json` for the final source and library hashes.

`runner.cc` loads one library in each process through the C API. It times
`Session::Run` in C++, including output ownership. It uses full graph optimization,
one inter-op thread, and either one or four intra-op threads. Worker spinning is
disabled so the other process stays idle. The processes use CPU 0 or CPUs 0-3.

`benchmark.py` tests 13 ConvTranspose configurations, three OCR input sizes, and
a MobileNetV2 control. It runs 10 warmups and eight alternating A/B or B/A rounds.
Each block has at least 20 calls and targets at least 100 ms on the slower build.
It compares every output bit after every round.

`replicate.py` repeats the selected cases with five new process pairs. Each pair
has four alternating rounds. It also compares two copies of the base library
on selected cases. This measures variation between processes on this host.
The reports retain every round, process pair, input hash, and output hash.

The four-thread measurements have more variation than the one-thread results.
MobileNetV2 has no ConvTranspose node. Changes in its time are control results,
not an expected benefit from this code. There are no ARM or GPU performance
measurements in this report.

## Results

The table shows the median of the five paired time reductions. The range contains
all five pairs. A positive value means less time. This statistic is not the ratio
of the separately calculated median times. The JSON also contains those times.

| Case | Threads | Median time reduction | Range |
| --- | ---: | ---: | ---: |
| `control_mobilenetv2` | 1 | 0.07% | -7.91% to 3.01% |
| `control_mobilenetv2` | 4 | -1.28% | -17.50% to 24.19% |
| `convtranspose_16_16_184_184` | 1 | 38.69% | 37.00% to 39.53% |
| `convtranspose_16_16_184_184` | 4 | 34.11% | 29.46% to 36.44% |
| `fallback_dilation` | 1 | 1.61% | -1.62% to 5.88% |
| `fallback_dilation` | 4 | 1.11% | -6.26% to 6.01% |
| `fallback_dilation_stride1` | 1 | 35.33% | 27.10% to 38.66% |
| `fallback_dilation_stride1` | 4 | 34.15% | 24.49% to 36.27% |
| `fallback_padding_stride1` | 1 | 36.92% | 31.38% to 39.47% |
| `fallback_padding_stride1` | 4 | 46.99% | 35.30% to 54.49% |
| `ppocrv6_tiny_det_1024` | 1 | 3.78% | 2.66% to 9.78% |
| `ppocrv6_tiny_det_1024` | 4 | 6.98% | -2.94% to 7.80% |
| `ppocrv6_tiny_det_736` | 1 | 6.06% | 3.41% to 9.09% |
| `ppocrv6_tiny_det_736` | 4 | 7.20% | 1.88% to 9.13% |

`fallback_` is a case name in the scripts. The padding and dilation cases now use
the valid-column calculation. The name does not mean that their code is unchanged.

For the OCR 736 A/A control, the median apparent reduction was 2.13% with one
thread and 1.05% with four threads. Its individual results ranged from -3.68% to
11.48%, and from -8.08% to 2.82%, respectively. Small gains and losses require
care on this host. The 2x2 and stride-1 operator gains are larger and repeat in
every fresh process pair.

The following table contains the full initial sweep for the final code. Times
are medians of eight rounds in one process pair. Use the repeated measurements
above to assess variation.

| Case | Threads | Base (ms) | Changed (ms) | Time reduction |
| --- | ---: | ---: | ---: | ---: |
| `convtranspose_16_1_368_368` | 1 | 0.706274 | 0.540385 | 23.49% |
| `convtranspose_16_1_368_368` | 4 | 0.539322 | 0.373211 | 30.80% |
| `convtranspose_16_16_184_184` | 1 | 2.763427 | 1.606432 | 41.87% |
| `convtranspose_16_16_184_184` | 4 | 2.587302 | 1.561462 | 39.65% |
| `convtranspose_64_32_64_64` | 1 | 0.745180 | 0.650849 | 12.66% |
| `convtranspose_64_32_64_64` | 4 | 0.625773 | 0.480352 | 23.24% |
| `convtranspose_128_64_32_32` | 1 | 0.553746 | 0.518482 | 6.37% |
| `convtranspose_128_64_32_32` | 4 | 0.325236 | 0.275178 | 15.39% |
| `convtranspose_1_1_1_1` | 1 | 0.001073 | 0.001150 | -7.11% |
| `convtranspose_1_1_1_1` | 4 | 0.001076 | 0.001109 | -3.07% |
| `convtranspose_3_5_1_7` | 1 | 0.001169 | 0.001175 | -0.52% |
| `convtranspose_3_5_1_7` | 4 | 0.001178 | 0.001172 | 0.52% |
| `fallback_overlap` | 1 | 0.355819 | 0.357002 | -0.33% |
| `fallback_overlap` | 4 | 0.416092 | 0.392677 | 5.63% |
| `fallback_padding` | 1 | 0.173092 | 0.165371 | 4.46% |
| `fallback_padding` | 4 | 0.148143 | 0.148856 | -0.48% |
| `fallback_dilation` | 1 | 0.195096 | 0.193734 | 0.70% |
| `fallback_dilation` | 4 | 0.170509 | 0.184098 | -7.97% |
| `fallback_output_padding` | 1 | 0.195105 | 0.199186 | -2.09% |
| `fallback_output_padding` | 4 | 0.185080 | 0.177953 | 3.85% |
| `fallback_padding_stride1` | 1 | 0.346295 | 0.213171 | 38.44% |
| `fallback_padding_stride1` | 4 | 0.414301 | 0.186728 | 54.93% |
| `fallback_dilation_stride1` | 1 | 0.160666 | 0.102912 | 35.95% |
| `fallback_dilation_stride1` | 4 | 0.147929 | 0.092138 | 37.71% |
| `fallback_padding_kernel4` | 1 | 0.617971 | 0.622854 | -0.79% |
| `fallback_padding_kernel4` | 4 | 0.828780 | 0.817485 | 1.36% |
| `ppocrv6_tiny_det_256` | 1 | 5.516763 | 5.508105 | 0.16% |
| `ppocrv6_tiny_det_256` | 4 | 3.850623 | 3.737954 | 2.93% |
| `ppocrv6_tiny_det_736` | 1 | 51.022883 | 48.684610 | 4.58% |
| `ppocrv6_tiny_det_736` | 4 | 23.914386 | 22.356208 | 6.52% |
| `ppocrv6_tiny_det_1024` | 1 | 107.520018 | 98.021054 | 8.83% |
| `ppocrv6_tiny_det_1024` | 4 | 41.877709 | 38.622982 | 7.77% |
| `control_mobilenetv2` | 1 | 5.547123 | 5.433184 | 2.05% |
| `control_mobilenetv2` | 4 | 3.213148 | 3.425463 | -6.61% |

The one-cell case takes about 1 microsecond. The sweep measured an increase of
less than 0.1 microsecond. The control and other shapes also show positive and
negative changes. This report does not claim that every shape is faster.

The final code passed 10 math tests and 81 ConvTranspose/Col2Im tests. Three tests
for unavailable CUDA or DirectML configurations were skipped. New tests cover
22 layouts, output guards, signed zero, infinity, NaN, subnormal values, groups,
bias, and batches. The extracted-body ASan/UBSan check passed 1,198 conditions
and compared 4,066,230 output values. The two session reports contain 552 full
output comparisons against the changed library. All were bitwise equal.

The actual-library diagnostic with shared buffers measured 58.25% less time in
the 2x2 Col2im path, 68.25% less time for padding with stride 1, and 64.40% less
time for dilation with stride 1. These times exclude GEMM and the session. They
must not be presented as full-model improvements.

The runtime profile confirms that both detector ConvTranspose nodes execute on
the CPU. See `range_profile.json`.

## Inputs

The OCR model is the official
[PaddlePaddle PP-OCRv6 tiny detector](https://huggingface.co/PaddlePaddle/PP-OCRv6_tiny_det_onnx).
It has two 2x2, stride-2 ConvTranspose nodes with weights `[16,16,2,2]` and
`[16,1,2,2]`. The model is FP32. This test measures the detector session, not the
text recognizer or the preprocessing and postprocessing stages.

`model_source.json` and `input_sources.json` pin the model, sample image, and
MobileNetV2 control by URL, revision, and SHA-256. The benchmark resizes the same
image to 256, 736, and 1024 pixels square. It uses BGR and the detector's mean and
standard deviation. The control uses RGB at 224 pixels square. The synthetic
ConvTranspose cases use a fixed seed, nonconstant weights, and bias.

Output equality checks test numerical preservation. They are not a new OCR
accuracy score on a labeled dataset.

## Reproduce on Linux x86-64

Use Python 3.12, GCC 13, CMake 4.2.1, and Ninja. The source build requires several
GB of temporary space. Keep one build directory and rebuild it after the source
change.

```bash
git clone --depth 1 --single-branch --branch bench/cpu-col2im \
  https://github.com/Hosi121/onnxruntime.git measurements
GIT_LFS_SKIP_SMUDGE=1 git clone --filter=blob:none --single-branch \
  --branch perf/cpu-col2im https://github.com/Hosi121/onnxruntime.git ort
git -C ort checkout 09dfa6ad06ed8072b2fbe57687d4f71c7914b025
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install numpy==2.5.3 onnx==1.17.0 Pillow==12.2.0 \
  cmake==4.2.1 ninja==1.13.2
python ort/tools/ci_build/build.py --build_dir "$PWD/ort-build" \
  --config Release --update --build --build_shared_lib --parallel 10 \
  --cmake_generator Ninja --skip_submodule_sync --skip_tests \
  --targets onnxruntime --cmake_extra_defines \
  onnxruntime_BUILD_UNIT_TESTS=ON CMAKE_EXPORT_COMPILE_COMMANDS=ON
cp ort-build/Release/libonnxruntime.so ort-build/baseline.so
git -C ort checkout perf/cpu-col2im
python ort/tools/ci_build/build.py --build_dir "$PWD/ort-build" \
  --config Release --build --parallel 10 \
  --targets onnxruntime onnxruntime_test_all onnxruntime_provider_test
g++ -O2 -std=c++17 -I ort/include/onnxruntime/core/session \
  measurements/runner.cc -ldl -o ort-build/runner
```

Download the pinned public inputs and check their hashes:

```bash
python - <<'PY'
import hashlib
import json
from pathlib import Path
from urllib.request import urlopen

root = Path("measurements")
sources = json.loads((root / "input_sources.json").read_text())
sources["detector"] = json.loads((root / "model_source.json").read_text())
names = {"image": "sample.jpg", "control": "mobilenet_v2.onnx",
         "detector": "ppocrv6_tiny_det.onnx"}
for key, source in sources.items():
    data = urlopen(source["url"]).read()
    assert hashlib.sha256(data).hexdigest() == source["sha256"]
    (root / names[key]).write_bytes(data)
PY
python -B measurements/benchmark.py --baseline ort-build/baseline.so \
  --candidate ort-build/Release/libonnxruntime.so --runner ort-build/runner \
  --model measurements/ppocrv6_tiny_det.onnx --image measurements/sample.jpg \
  --control measurements/mobilenet_v2.onnx --suite all --output suite.json
python -B measurements/replicate.py --baseline ort-build/baseline.so \
  --candidate ort-build/Release/libonnxruntime.so --runner ort-build/runner \
  --model measurements/ppocrv6_tiny_det.onnx --image measurements/sample.jpg \
  --control measurements/mobilenet_v2.onnx --output repeat.json
```

The scripts reject an existing output path. Use a new path for a new run.

```bash
cd ort-build/Release
./onnxruntime_test_all --gtest_filter='MathTest.*:*MathGemmTest*'
./onnxruntime_provider_test --gtest_filter='*ConvTranspose*:*Col2Im*'
cd ../..
python -B measurements/sanitize.py --repository ort --output sanitizer.json
python -B measurements/profile_model.py --library ort-build/Release/libonnxruntime.so \
  --runner ort-build/runner --model measurements/ppocrv6_tiny_det.onnx \
  --image measurements/sample.jpg --output profile.json
```

The sanitizer check extracts the exact old and new Col2im function bodies. It
replaces the zero-fill and narrowing helpers for its small inputs. It uses ASan
and UBSan. It is not a full-runtime sanitizer build.

`direct.py` can also call the actual Col2im symbols in the two Linux libraries
with the same input and output buffers. It excludes GEMM and the session. Its
timer includes the ctypes calls. It requires the local symbols from unstripped
libraries. It is a diagnostic tool, not a runtime API example.

## Trial records

The first trial added only the 2x2 path. The `outlined`, `dispatch`, and `nested`
trials checked whether the location of that path affected other shapes. Some
trials made other shapes slower. They are not the final implementation.
The final `range` trial also moves the valid-column calculation out of the inner
loop. Files with `range_` in their names describe that version.

The first script versions and their raw reports remain in this branch's parent
commit. The benchmark branch contains no model weights, images, or build outputs.
