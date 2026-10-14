"""Explain Carlsen-Firouzja (2021) with saved continuations and Opus 5 motifs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from binary_properties.inference import infer
from binary_properties.io import position_from_dict
from binary_properties.libraries import load_builtin_library


def comparison() -> dict:
    """Saved continuation positions from https://explaining-chess.vercel.app."""
    # Website example: carlsen-firouzja-2021-p58 (February 22, 2021).
    return {
        # 30. Rf4 hxg6 31. hxg6 a5 32. Rxd4 Rg3
        # 33. b4 axb4 34. Rxb4 Rxg6 35. c4 Rc8
        "positive": [
            {
                "fen": "2r3k1/1p4p1/3n2r1/3P1p2/1RP5/3B4/1P6/1K1R4 w - - 1 36",
                "color": "white",
            }
        ],
        # 30. Rdf1 hxg6 31. hxg6 Re5 32. Rg1 Rf6
        # 33. Rh2 Rxd5 34. Re2 Kf8
        "negative": [
            {
                "fen": "5k2/1p4p1/p2n1rP1/3r1p2/3p4/1P1B4/1PP1R3/1K4R1 w - - 2 35",
                "color": "white",
            }
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write-comparison", type=Path)
    args = parser.parse_args()
    groups = comparison()
    if args.write_comparison:
        with args.write_comparison.open("x", encoding="utf-8") as handle:
            json.dump(groups, handle, indent=2)
            handle.write("\n")
        print(f"Saved comparison to {args.write_comparison}")
    library = load_builtin_library("chess/claude-opus-5")
    positive = [position_from_dict(row, "chess") for row in groups["positive"]]
    negative = [position_from_dict(row, "chess") for row in groups["negative"]]
    result = infer(library, positive, negative)
    print("Library: chess/claude-opus-5")
    print("Game: Magnus Carlsen-Alireza Firouzja, February 22, 2021")
    print("Precomputed continuations: 30.Rf4 (Stockfish) vs. 30.Rdf1 (played)")
    for motif in result["selected"]:
        print(f"\n{motif['name']}\n  {motif['description']}")
        print(
            f"  Support fit: {motif['support_fit']:.0%}; discovery delta: {motif['delta']:.4f}"
        )
    if result["errors"] or not result["selected"] or result["used_fallback"]:
        raise RuntimeError(f"Example could not explain the game comparison: {result}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
