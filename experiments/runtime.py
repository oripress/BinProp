"""Evaluate Direct functions without requiring discovery's color symmetry.

Run generated code with external isolation; this is not a security sandbox.
"""

import ast
import json
import re
import subprocess
import sys
from pathlib import Path

from binary_properties.games.common import execute_source, parse_source
from binary_properties.workers import run_worker


def helper_names(source, game):
    tree = ast.parse(source)
    names = [n.name for n in tree.body if isinstance(n, ast.FunctionDef)]
    if game == "chess":
        return sorted(
            {n for n in names if re.fullmatch(r"explain_\d+", n)},
            key=lambda n: int(n.split("_")[1]),
        )[:3]
    requested = [n for n in names if n == "explain" or re.fullmatch(r"explain_?\d+", n)]
    return (requested or [n for n in names if n in {"motif", "basis"}])[:3]


def helper_source(source, name, game):
    tree = ast.parse(source)
    if game == "chess":
        tree.body = [
            n
            for n in tree.body
            if isinstance(n, (ast.Import, ast.ImportFrom, ast.Assign, ast.AnnAssign))
            or (
                isinstance(n, ast.FunctionDef)
                and (n.name == name or not re.fullmatch(r"explain(?:_\d+)?", n.name))
            )
        ]
    fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name)
    if any(
        isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == name
        for n in ast.walk(tree)
    ):
        raise ValueError("Recursive Direct helpers are not allowed")
    fn.name = "motif"
    tree.body += [
        ast.Assign(
            targets=[ast.Name(id="MOTIF_NAME", ctx=ast.Store())],
            value=ast.Constant(name),
        ),
        ast.Assign(
            targets=[ast.Name(id="MOTIF_DESCRIPTION", ctx=ast.Store())],
            value=ast.Constant(name),
        ),
    ]
    return ast.unparse(ast.fix_missing_locations(tree))


def evaluate_direct(game, source, name, positions, *, timeout=60):
    transformed = helper_source(source, name, game)
    payload = {
        "game": game,
        "source": transformed,
        "positions": [{"board": p.board, "color": p.color} for p in positions],
    }
    try:
        result = run_worker(
            "experiments.runtime",
            json.dumps(payload),
            timeout=timeout,
            extra_path=Path(__file__).resolve().parents[1],
        )
    except subprocess.TimeoutExpired as error:
        raise TimeoutError("Direct helper exceeded its execution deadline") from error
    if result.returncode:
        raise ValueError("Direct worker failed")
    decoded = json.loads(result.stdout)
    if decoded.get("error"):
        raise ValueError(decoded["error"])
    values = decoded["values"]
    if len(values) != len(positions) or any(
        v is not None and type(v) is not bool for v in values
    ):
        raise ValueError("Direct worker returned invalid outputs")
    return values


def worker(payload):
    game = payload["game"]
    if game not in ("chess", "go"):
        raise ValueError("Expected chess or go")
    allowed = {"chess", "math"} if game == "chess" else {"math", "sgfmill"}
    tree, fn = parse_source(payload["source"], allowed)
    extra = {}
    if game == "chess":
        import chess

        for index, letter in enumerate("ABCDEFGH"):
            setattr(chess, "FILE_" + letter, index)
        extra["chess"] = chess
    _, _, function = execute_source(tree, fn, allowed, **extra)
    argc = len(fn.args.posonlyargs) + len(fn.args.args)
    if argc not in (1, 2):
        raise ValueError("Direct helper requires one or two positional arguments")
    if game == "go":
        empty = ["." * 19 for _ in range(19)]
        for color in ("B", "W"):
            value = function(empty) if argc == 1 else function(empty, color)
            if type(value) is not bool:
                raise ValueError("Go helper failed its empty-board smoke check")
    values = []
    for p in payload["positions"]:
        try:
            # Chess Direct historically calls only with FEN, even for two-arg definitions.
            value = (
                function(p["board"])
                if game == "chess" or argc == 1
                else function(p["board"], p["color"])
            )
            values.append(value if type(value) is bool else None)
        except Exception:
            values.append(None)
    return {"values": values}


if __name__ == "__main__":
    try:
        output = worker(json.load(sys.stdin))
    except Exception as error:
        output = {"error": f"{type(error).__name__}: {error}"}
    print(json.dumps(output))
