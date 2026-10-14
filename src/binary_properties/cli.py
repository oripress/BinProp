from __future__ import annotations

import argparse
import json
import os
import sys
from contextlib import nullcontext
from dataclasses import asdict
from hashlib import sha256
from pathlib import Path

from .discovery import DiscoveryConfig, discover
from .inference import infer
from .io import load_groups, load_positions, write_json
from .libraries import list_libraries, resolve_library
from .provider import (
    DEFAULT_API_KEY_ENV,
    DEFAULT_BASE_URL,
    ChatClient,
    ProviderError,
)
from .runs import run_lock, save_checkpoint
from .types import Library


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(
        description="BinProp: discover and use binary chess and Go concepts."
    )
    commands = result.add_subparsers(dest="command", required=True)
    libraries = commands.add_parser(
        "libraries", help="List the bundled paper motif libraries"
    )
    libraries.add_argument(
        "--json", action="store_true", help="Print the catalog as JSON"
    )
    discovery = commands.add_parser(
        "discover", help="Discover motifs from your labeled positions"
    )
    discovery.add_argument("--game", choices=("chess", "go"))
    discovery.add_argument("--positions", type=Path, help="Your labeled JSONL file")
    discovery.add_argument(
        "--output",
        type=Path,
        help="New directory for checkpoints and logs",
    )
    discovery.add_argument(
        "--initial",
        help="Library JSON path or bundled ID to seed a new run",
    )
    discovery.add_argument(
        "--resume", type=Path, help="Resume an existing run directory"
    )
    discovery.add_argument("--model", help="Model ID (or MOTIF_MODEL)")
    discovery.add_argument(
        "--base-url",
        help="API base URL (or MOTIF_BASE_URL; default: Nebius Token Factory)",
    )
    discovery.add_argument(
        "--api-key-env",
        help=f"API-key environment variable (default: {DEFAULT_API_KEY_ENV})",
    )
    discovery.add_argument("--iterations", type=int)
    discovery.add_argument("--min-delta", type=float)
    discovery.add_argument("--max-containment", type=float)
    discovery.add_argument("--max-motifs", type=int, help="0 means uncapped")
    discovery.add_argument(
        "--timeout", type=float, help="Seconds per candidate evaluation"
    )
    discovery.add_argument(
        "--max-p95-ms", type=float, help="Per-position runtime limit"
    )
    discovery.add_argument("--seed", type=int)
    discovery.add_argument("--max-tokens", type=int)
    discovery.add_argument("--request-timeout", type=float)
    discovery.add_argument(
        "--request-extra",
        type=Path,
        help="Optional JSON API options, e.g. reasoning settings",
    )
    inference = commands.add_parser(
        "infer",
        help="Select explanations for positive versus negative positions",
    )
    inference.add_argument(
        "--library",
        required=True,
        help="Library JSON path or bundled ID, e.g. chess/kimi-k3",
    )
    inference.add_argument(
        "--positions",
        type=Path,
        required=True,
        help="JSON object with positive and negative lists",
    )
    inference.add_argument(
        "--output", type=Path, help="JSON result path; stdout by default"
    )
    inference.add_argument("--top-k", type=int, default=3)
    inference.add_argument("--min-fit", type=float, default=0.5)
    inference.add_argument("--timeout", type=float, default=60)
    return result


def _resume_run(args, directory: Path):
    """Restore settings without using the current environment defaults."""
    forbidden = (
        "game",
        "output",
        "initial",
        "model",
        "base_url",
        "api_key_env",
        "min_delta",
        "max_containment",
        "max_motifs",
        "timeout",
        "max_p95_ms",
        "seed",
        "max_tokens",
        "request_timeout",
        "request_extra",
    )
    if any(getattr(args, name) is not None for name in forbidden):
        raise ValueError(
            "Resume only accepts --iterations and an optional relocated --positions"
        )
    state = json.loads((directory / "checkpoint.json").read_text(encoding="utf-8"))
    if not isinstance(state, dict) or not isinstance(state.get("run"), dict):
        raise ValueError("Invalid run checkpoint; older runs can use --initial instead")
    run = dict(state["run"])
    required = (
        "positions_path",
        "positions_sha256",
        "model",
        "base_url",
        "api_key_env",
        "max_tokens",
        "request_timeout",
        "request_extra",
        "request_extra_sha256",
    )
    if not all(name in run for name in required):
        raise ValueError("Incomplete run checkpoint")
    game = state.get("game")
    if game not in ("chess", "go"):
        raise ValueError("Invalid checkpoint game")
    saved_config = state.get("config")
    cursor = state.get("next_iteration")
    if not isinstance(saved_config, dict) or type(cursor) is not int or cursor < 0:
        raise ValueError("Invalid checkpoint progress")
    target = saved_config.get("iterations")
    if type(target) is not int or target < cursor:
        raise ValueError("Invalid checkpoint target")
    iterations = args.iterations if args.iterations is not None else target - cursor
    config = DiscoveryConfig(**{**saved_config, "iterations": iterations})
    positions_path = (args.positions or Path(run["positions_path"])).resolve()
    if sha256(positions_path.read_bytes()).hexdigest() != run["positions_sha256"]:
        raise ValueError("Resume positions have changed; use --initial for a new run")
    run["positions_path"] = str(positions_path)
    return state, config, run


def _run_discovery(args, directory: Path) -> int:
    state = None
    if args.resume:
        state, config, run = _resume_run(args, directory)
        game = state["game"]
        positions_path = Path(run["positions_path"])
        initial = None
    else:
        if not args.game or not args.positions:
            raise ValueError("New discovery requires --game and --positions")
        game = args.game
        model = args.model or os.environ.get("MOTIF_MODEL")
        if not model:
            raise ValueError("Set --model (or MOTIF_MODEL)")
        config_fields = (
            "iterations",
            "min_delta",
            "max_containment",
            "max_motifs",
            "timeout",
            "max_p95_ms",
            "seed",
        )
        config = DiscoveryConfig(
            **{
                name: getattr(args, name)
                for name in config_fields
                if getattr(args, name) is not None
            }
        )
        positions_path = args.positions.resolve()
        initial = resolve_library(args.initial) if args.initial else Library(game)
        if initial.game != game:
            raise ValueError("Initial library game does not match --game")
        extra = json.loads(args.request_extra.read_text()) if args.request_extra else {}
        run = {
            "game": game,
            "model": model,
            "base_url": args.base_url
            or os.environ.get("MOTIF_BASE_URL")
            or DEFAULT_BASE_URL,
            "api_key_env": args.api_key_env or DEFAULT_API_KEY_ENV,
            "max_tokens": args.max_tokens if args.max_tokens is not None else 4096,
            "request_timeout": args.request_timeout
            if args.request_timeout is not None
            else 120,
            "positions_path": str(positions_path),
            "positions_sha256": sha256(positions_path.read_bytes()).hexdigest(),
            "request_extra": extra,
            "request_extra_sha256": sha256(
                json.dumps(extra, sort_keys=True, allow_nan=False).encode()
            ).hexdigest(),
        }
        if args.initial:
            reference = str(args.initial)
            run["initial_library_reference"] = (
                Path(reference).name if Path(reference).is_file() else reference
            )

    extra = run["request_extra"]
    if (
        sha256(json.dumps(extra, sort_keys=True, allow_nan=False).encode()).hexdigest()
        != run["request_extra_sha256"]
    ):
        raise ValueError("Checkpoint provider options do not match their fingerprint")
    positions = load_positions(positions_path, game)
    client = ChatClient(
        run["model"],
        run["base_url"],
        api_key_env=run["api_key_env"],
        max_tokens=run["max_tokens"],
        timeout=run["request_timeout"],
        extra_body=extra,
    )
    if config.iterations and not os.environ.get(run["api_key_env"]):
        raise ProviderError(f"Set the {run['api_key_env']} environment variable")
    run.update(
        config=asdict(config),
        position_count=len(positions),
        winning_count=sum(position.label == 1 for position in positions),
        losing_count=sum(position.label == 0 for position in positions),
    )

    def checkpoint(current):
        save_checkpoint(directory, current, run)

    def progress(event, _library):
        print(
            f"Iteration {event['iteration']}: {event['reason']}",
            file=sys.stderr,
        )

    with run_lock(directory, create=True) if not args.resume else nullcontext():
        discover(
            game,
            positions,
            client,
            config=config,
            initial_library=initial,
            resume_state=state,
            on_iteration=progress,
            on_checkpoint=checkpoint,
        )
        print(str(directory / "basis.json"))
        return 0


def _discover(args) -> int:
    if args.resume:
        directory = args.resume.resolve()
        if not (directory / "checkpoint.json").is_file():
            raise ValueError(
                "No resumable checkpoint; use --initial with a new output directory"
            )
    else:
        if not args.model and not os.environ.get("MOTIF_MODEL"):
            raise ValueError("Set --model (or MOTIF_MODEL)")
        if not args.output:
            raise ValueError("New discovery requires --output")
        directory = args.output.resolve()
        if directory.exists():
            raise ValueError("Output already exists; choose a new directory")
        return _run_discovery(args, directory)
    with run_lock(directory):
        return _run_discovery(args, directory)


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    try:
        if args.command == "libraries":
            catalog = list_libraries()
            if args.json:
                print(json.dumps(catalog, ensure_ascii=False, indent=2))
            else:
                print(f"{'Library':<29} {'Motifs':>6}  Model")
                for entry in catalog:
                    print(
                        f"{entry['id']:<29} {entry['motif_count']:>6}  {entry['model']}"
                    )
            return 0
        if args.command == "infer":
            library = resolve_library(args.library)
            positive, negative = load_groups(args.positions, library.game)
            result = infer(
                library,
                positive,
                negative,
                top_k=args.top_k,
                min_fit=args.min_fit,
                timeout=args.timeout,
            )
            if args.output:
                if args.output.exists():
                    raise ValueError("Output already exists; choose a new result path")
                write_json(args.output, result)
            else:
                print(json.dumps(result, ensure_ascii=False, allow_nan=False, indent=2))
            if result.get("errors") and not result.get("ranked"):
                print(
                    "All eligible concepts failed evaluation; see errors in the result.",
                    file=sys.stderr,
                )
                return 2
            return 0
        return _discover(args)
    except KeyboardInterrupt:
        print(
            "Interrupted. Completed checkpoints remain in the output directory.",
            file=sys.stderr,
        )
        return 130
    except (OSError, ValueError, RuntimeError, TimeoutError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
