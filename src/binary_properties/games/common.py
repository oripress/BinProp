"""Check generated Python code. These checks are not a security sandbox."""

from __future__ import annotations

import ast
from dataclasses import dataclass
from typing import Any, Callable


class MotifValidationError(ValueError):
    """Invalid candidate source, result, or supplied position."""


MOTIF_FIELD_LIMITS = {"name": 180, "description": 2000, "source": 12000}


def validate_motif_text(value: object, field: str) -> str:
    """Use the same text contract for generated and serialized motifs."""
    limit = MOTIF_FIELD_LIMITS[field]
    if not isinstance(value, str) or not value.strip() or len(value) > limit:
        label = "source" if field == "source" else f"MOTIF_{field.upper()}"
        raise MotifValidationError(
            f"Invalid motif {field} ({label}): "
            f"expected 1–{limit} characters, not only whitespace"
        )
    return value


_BLOCKED_CALLS = {
    "compile",
    "delattr",
    "eval",
    "exec",
    "getattr",
    "globals",
    "input",
    "locals",
    "open",
    "setattr",
    "vars",
    "__import__",
}
_SAFE_BUILTINS = {
    "abs": abs,
    "all": all,
    "any": any,
    "bin": bin,
    "bool": bool,
    "dict": dict,
    "enumerate": enumerate,
    "Exception": Exception,
    "float": float,
    "int": int,
    "len": len,
    "list": list,
    "max": max,
    "min": min,
    "range": range,
    "round": round,
    "set": set,
    "sorted": sorted,
    "str": str,
    "sum": sum,
    "tuple": tuple,
    "ValueError": ValueError,
    "zip": zip,
}


@dataclass(frozen=True)
class CompiledMotif:
    name: str
    description: str
    evaluate: Callable[[Any, str], bool]


def parse_source(
    source: str, allowed_imports: set[str]
) -> tuple[ast.Module, ast.FunctionDef]:
    validate_motif_text(source, "source")
    try:
        tree = ast.parse(source)
    except SyntaxError as exc:
        raise MotifValidationError(f"syntax error: {exc}") from exc
    functions = {
        node.name: node for node in tree.body if isinstance(node, ast.FunctionDef)
    }
    fn = functions.get("motif") or functions.get("basis")
    if fn is None:
        raise MotifValidationError("source must define motif (or compatible basis)")
    if fn.args.vararg or fn.args.kwarg or fn.args.kwonlyargs:
        raise MotifValidationError(
            "motif must use only its required positional arguments"
        )
    for node in ast.walk(tree):
        if isinstance(
            node,
            (
                ast.AsyncFunctionDef,
                ast.ClassDef,
                ast.Lambda,
                ast.With,
                ast.AsyncWith,
                ast.Raise,
                ast.Delete,
                ast.Global,
                ast.Nonlocal,
            ),
        ):
            raise MotifValidationError(f"{type(node).__name__} is not allowed")
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.split(".", 1)[0] not in allowed_imports:
                    raise MotifValidationError(f"import {alias.name!r} is not allowed")
        if isinstance(node, ast.ImportFrom):
            if (
                node.level
                or not node.module
                or node.module.split(".", 1)[0] not in allowed_imports
            ):
                raise MotifValidationError(
                    f"from {node.module!r} import is not allowed"
                )
        if isinstance(node, ast.Attribute) and node.attr.startswith("__"):
            raise MotifValidationError("dunder attribute access is not allowed")
        if isinstance(node, ast.Name) and node.id.startswith("__"):
            raise MotifValidationError("dunder names are not allowed")
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            if node.func.id in _BLOCKED_CALLS or node.func.id == fn.name:
                raise MotifValidationError(f"call to {node.func.id!r} is not allowed")
    return tree, fn


def argument_names(fn: ast.FunctionDef) -> list[str]:
    return [arg.arg for arg in [*fn.args.posonlyargs, *fn.args.args]]


def require_color_usage(fn: ast.FunctionDef) -> None:
    args = argument_names(fn)
    if len(args) != 2 or not any(
        isinstance(node, ast.Name)
        and isinstance(node.ctx, ast.Load)
        and node.id == args[1]
        for node in ast.walk(fn)
    ):
        raise MotifValidationError("two-argument motif must use its color argument")


def execute_source(
    tree: ast.Module, fn: ast.FunctionDef, allowed_imports: set[str], **extra: Any
):
    # Run only in the worker subprocess; AST checks do not provide isolation.
    def allowed_import(name, globals_=None, locals_=None, fromlist=(), level=0):
        if level or name.split(".", 1)[0] not in allowed_imports:
            raise ImportError(f"import {name!r} is not allowed")
        return __import__(name, globals_, locals_, fromlist, level)

    env = {"__builtins__": _SAFE_BUILTINS | {"__import__": allowed_import}, **extra}
    exec(compile(tree, "<motif>", "exec"), env, env)
    name, description = env.get("MOTIF_NAME"), env.get("MOTIF_DESCRIPTION")
    validate_motif_text(name, "name")
    validate_motif_text(description, "description")
    func = env.get(fn.name)
    if not callable(func):
        raise MotifValidationError(f"{fn.name} must be callable")
    return name, description, func


def require_bool(value: Any) -> bool:
    if not isinstance(value, bool):
        raise MotifValidationError(
            f"motif returned {type(value).__name__}, expected bool"
        )
    return value
