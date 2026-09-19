"""Check the two Col2im function bodies with ASan and UBSan.

The harness replaces the ORT zero fill and narrowing helpers for small inputs.
It does not build the full runtime with sanitizers.
"""

import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess
import tempfile


def extract(source, name):
    pattern = r"template <>\s*void Col2im<float, CPUMathUtil, StorageOrder::NCHW>\(.*?(?=\ntemplate <>)"
    match = re.search(pattern, source, re.S)
    body = match.group()
    helper_start = source.find("static EIGEN_DONT_INLINE void Col2im2x2Stride2(")
    helper = ""
    if helper_start >= 0:
        helper = source[helper_start : match.start()].replace(
            "EIGEN_DONT_INLINE", "__attribute__((noinline))"
        )
        helper = helper.replace("Col2im2x2Stride2", name + "_2x2")
        body = body.replace("Col2im2x2Stride2", name + "_2x2")
    return helper + re.sub(
        r"template <>\s*void Col2im<float, CPUMathUtil, StorageOrder::NCHW>",
        "void " + name,
        body,
    )


PREFIX = r"""
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <limits>
#include <random>
#include <vector>
struct CPUMathUtil {};
template<class T, class P> void Set(ptrdiff_t n, T v, T* p, P*) { std::fill(p, p+n, v); }
template<class T> T narrow(int64_t x) { return static_cast<T>(x); }
bool is_a_ge_zero_and_a_lt_b(int64_t a, int64_t b) { return a >= 0 && a < b; }
"""

MAIN = r"""
int main() {
  std::mt19937 rng(20260920);
  int checks=0;
  uint64_t values=0;
  for (int t=0; t<600; ++t) {
    int64_t ch=1+rng()%4, h=2+2*(rng()%18), w=2+2*(rng()%67);
    int64_t kh=2, kw=2, sh=2, sw=2, dh=1, dw=1, pt=0, pl=0, pb=0, pr=0;
    if (t >= 100) {
      kh=1+rng()%3; kw=1+rng()%3; sh=1+rng()%3; sw=1+rng()%3;
      dh=1+rng()%2; dw=1+rng()%2;
      pt=rng()%3; pl=rng()%3; pb=rng()%3; pr=rng()%3;
      h+=rng()%2; w+=rng()%2;
    }
    if (h+pt+pb < dh*(kh-1)+1 || w+pl+pr < dw*(kw-1)+1) continue;
    int64_t oh=(h+pt+pb-(dh*(kh-1)+1))/sh+1, ow=(w+pl+pr-(dw*(kw-1)+1))/sw+1;
    size_t n=ch*kh*kw*oh*ow, m=ch*h*w;
    for (int special=0; special<2; ++special) {
      std::vector<float> source(n+2,12345.0f), a(m+2,23456.0f), b(m+2,23456.0f);
      const uint32_t bits[]={0,0x80000000,0x7f800000,0xff800000,0x7fc12345,0xffc23456,1,0x80000001};
      for (size_t i=0; i<n; ++i) {
        if (special) std::memcpy(&source[i+1], &bits[i%8], 4);
        else source[i+1]=static_cast<float>(static_cast<int>(rng()%2049)-1024)*0.03125f;
      }
      baseline(source.data()+1,ch,h,w,kh,kw,dh,dw,pt,pl,pb,pr,sh,sw,a.data()+1,nullptr);
      candidate(source.data()+1,ch,h,w,kh,kw,dh,dw,pt,pl,pb,pr,sh,sw,b.data()+1,nullptr);
      if (std::memcmp(a.data(),b.data(),(m+2)*4) || a.front()!=23456.0f || a.back()!=23456.0f ||
          source.front()!=12345.0f || source.back()!=12345.0f) {
        std::cerr << "Mismatch at case " << t << ", pattern " << special << '\n'; return 1;
      }
      ++checks; values+=m;
    }
  }
  std::cout << "{\"checks\":" << checks << ",\"values\":" << values
            << ",\"bitwise_equal\":true,\"guards_unchanged\":true}\n";
}
"""


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", type=Path, required=True)
    parser.add_argument("--base", default="09dfa6ad06ed8072b2fbe57687d4f71c7914b025")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    path = "onnxruntime/core/util/math_cpu.cc"
    old = subprocess.check_output(
        ["git", "show", f"{args.base}:{path}"], cwd=args.repository, text=True
    )
    new = (args.repository / path).read_text()
    code = PREFIX + extract(old, "baseline") + extract(new, "candidate") + MAIN
    flags = [
        "-std=c++20",
        "-O2",
        "-g",
        "-fno-omit-frame-pointer",
        "-fsanitize=address,undefined",
        "-fno-sanitize-recover=all",
        "-fno-fast-math",
        "-ffp-contract=off",
    ]
    with tempfile.TemporaryDirectory(prefix="churin-ort-sanitize-") as temp:
        source = Path(temp) / "test.cc"
        source.write_text(code)
        binary = Path(temp) / "test"
        subprocess.run(["g++", *flags, str(source), "-o", str(binary)], check=True)
        result = subprocess.run(
            [str(binary)], check=True, capture_output=True, text=True
        )
    report = json.loads(result.stdout)
    report.update(
        base=args.base,
        candidate_sha256=hashlib.sha256(new.encode()).hexdigest(),
        harness_sha256=hashlib.sha256(code.encode()).hexdigest(),
        flags=flags,
        stderr=result.stderr,
        scope="Extracted Col2im bodies with small-input helper replacements. Not a full ASan runtime build.",
    )
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
