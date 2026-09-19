"""一键复现入口：python main.py [--bootstrap 10000]"""
from __future__ import annotations

import argparse
import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from src.pipeline import run


def main() -> None:
    parser = argparse.ArgumentParser(description="Cookie Cats A/B 实验分析流水线")
    parser.add_argument("--bootstrap", type=int, default=10_000,
                        help="bootstrap 次数（默认 10000）")
    args = parser.parse_args()
    run(n_bootstrap=args.bootstrap)


if __name__ == "__main__":
    main()
