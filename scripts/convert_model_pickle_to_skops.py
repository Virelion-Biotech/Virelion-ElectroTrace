#!/usr/bin/env python3
"""Convert a trusted legacy ElectroTrace pickle to the safe .skops format."""
from __future__ import annotations

import argparse
from pathlib import Path

from electrotrace.candidate_suppressor import CandidateSuppressor


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument(
        "--allow-pickle",
        action="store_true",
        required=True,
        help="required because conversion loads executable legacy serialization",
    )
    args = parser.parse_args()
    model = CandidateSuppressor.load(args.input, allow_pickle=True)
    model.save(args.output)
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
