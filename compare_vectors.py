#!/usr/bin/env python3
"""批量测试向量对拍入口。"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from xor_recovery.triton_compare import compare_all_vectors
from xor_recovery.vectors import build_test_vectors


def configure_utf8_console() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="用恢复出的公式和二进制做多向量对拍。")
    parser.add_argument(
        "--binary-dir",
        default="build",
        help="包含 encrypt_demo.exe 和 encrypt_demo.protected.exe 的目录",
    )
    return parser


def main() -> int:
    configure_utf8_console()
    args = build_parser().parse_args()
    binary_dir = Path(args.binary_dir)

    report = compare_all_vectors(binary_dir, build_test_vectors())
    print(f"对拍通过: {len(report.cases)} 个向量全部一致")
    for case in report.cases:
        print(
            f"  {case.vector.name}: "
            f"plain={case.vector.plaintext:#010x} "
            f"key={case.vector.key:#010x} "
            f"trace={case.verification.trace_result:#010x}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
