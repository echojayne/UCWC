#!/usr/bin/env python
"""Download or refresh RIDAS semantic decoder weights."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ucwc.semantic import ensure_ridas_weights


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ridas-decoder", default="data/semantic_catalogs/weights/ridas/decoder_cifar100.pth")
    parser.add_argument("--ridas-classifier", default="data/semantic_catalogs/weights/ridas/classifier.pth")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    paths = ensure_ridas_weights(
        decoder_path=args.ridas_decoder,
        classifier_path=args.ridas_classifier,
        overwrite=args.overwrite,
    )
    print(paths)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
