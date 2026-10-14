"""Run Direct generation or evaluate a BinProp library on held-out episodes."""

import argparse
import json
import math
import time
from dataclasses import asdict
from hashlib import sha256
from pathlib import Path
from statistics import fmean

from binary_properties.io import write_json
from binary_properties.libraries import resolve_library
from binary_properties.provider import (
    DEFAULT_API_KEY_ENV,
    DEFAULT_BASE_URL,
    ChatClient,
    ProviderError,
)
from binary_properties.runs import run_lock

from .data import load_episodes
from .prompts import direct_messages
from .scoring import direct_candidates, extract_code, score_binprop, score_direct


def deliver(client, messages, *, attempts=5, sleep=time.sleep):
    """Provider failures are not delivered code attempts; authorization errors stop."""
    for attempt in range(attempts):
        try:
            return client.complete(messages)
        except ProviderError as error:
            if not error.retryable or attempt + 1 == attempts:
                raise
            sleep(min(2**attempt, 8))
    raise AssertionError("Unreachable")


def generate_direct(
    episode,
    client,
    *,
    no_tools=False,
    timeout=60,
    record=lambda _: None,
    previous_attempts=(),
):
    original = direct_messages(episode, no_tools=no_tools)
    messages, attempts = original, list(previous_attempts)
    if len(attempts) > 3 or any(
        row["attempt"] != i + 1 for i, row in enumerate(attempts)
    ):
        raise ValueError("Invalid saved Direct attempts")
    for attempt in range(3):
        if attempt < len(attempts):
            response = attempts[attempt]["response"]
        else:
            response = deliver(client, messages)
        source = extract_code(response)
        candidates, errors = (
            direct_candidates(episode, source, timeout=timeout)
            if source
            else ([], ["no_code"])
        )
        if attempt == len(attempts):
            attempts.append(
                {
                    "attempt": attempt + 1,
                    "messages": messages,
                    "response": response,
                    "source": source,
                    "errors": errors,
                }
            )
            record(attempts)
        if candidates:
            return source, attempts
        feedback = "; ".join(errors)[:2000] or "No requested helper compiled."
        # Only syntax/compilation/support-execution feedback enters repairs.
        repair = (
            original[1]["content"]
            + "\n\nYour previous answer failed validation.\n"
            + feedback
        )
        if source:
            repair += "\n\nPrevious Python candidate:\n" + source[-6000:]
        repair += (
            "\n\nReturn one Python code block with the three requested Boolean helpers."
        )
        messages = [original[0], {"role": "user", "content": repair}]
    return None, attempts


def _direct_episode(episode, client, attempts_path, *, no_tools=False, timeout=60):
    saved = (
        json.loads(attempts_path.read_text())
        if attempts_path.exists()
        else {"id": episode.id, "attempts": []}
    )
    if saved["id"] != episode.id:
        raise ValueError("Saved Direct attempts belong to another episode")

    def record(attempts):
        write_json(attempts_path, {"id": episode.id, "attempts": attempts})

    source, _ = generate_direct(
        episode,
        client,
        no_tools=no_tools,
        timeout=timeout,
        record=record,
        previous_attempts=saved["attempts"],
    )
    if source:
        return score_direct(episode, source, timeout=timeout)
    return {
        "id": episode.id,
        "method": "direct",
        "score": 0.0,
        "motif_count": 0,
        "motifs": [],
        "errors": ["no_valid_delivered_output"],
        "used_fallback": False,
    }


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    commands = result.add_subparsers(dest="method", required=True)
    for method in ("direct", "binprop"):
        command = commands.add_parser(method)
        command.add_argument("--game", choices=("chess", "go"), required=True)
        command.add_argument("--episodes", type=Path, required=True)
        command.add_argument(
            "--output", type=Path, required=True, help="A new run directory"
        )
        command.add_argument(
            "--resume",
            action="store_true",
            help="Continue this same saved configuration",
        )
        command.add_argument(
            "--timeout",
            type=float,
            default=60,
            help="Predicate execution deadline in seconds",
        )
        if method == "binprop":
            command.add_argument("--library", required=True)
        else:
            command.add_argument("--model", required=True)
            command.add_argument("--base-url", default=DEFAULT_BASE_URL)
            command.add_argument("--api-key-env", default=DEFAULT_API_KEY_ENV)
            command.add_argument(
                "--request-extra",
                type=Path,
                help="JSON provider options; never put credentials here",
            )
            command.add_argument("--max-tokens", type=int, default=65536)
            command.add_argument("--request-timeout", type=float, default=3000)
            command.add_argument(
                "--no-tools",
                action="store_true",
                help="Use the proprietary-model chess prompt variant",
            )
    return result


def run(args):
    if not math.isfinite(args.timeout) or args.timeout <= 0:
        raise ValueError("Execution timeout must be positive")
    episodes = load_episodes(args.episodes, args.game)
    config = {
        "method": args.method,
        "game": args.game,
        "episodes_sha256": sha256(args.episodes.read_bytes()).hexdigest(),
        "episode_count": len(episodes),
        "timeout": args.timeout,
        "support_per_side": 4,
        "hidden_per_group": 8,
        "top_k": 3,
        "min_support_fit": 0.5,
    }
    library, client = None, None
    if args.method == "binprop":
        library = resolve_library(args.library)
        if library.game != args.game:
            raise ValueError("Library and episode games must match")
        config["library"] = asdict(library)
    else:
        extra = {"temperature": 1.0, "top_p": 1.0}
        if args.request_extra:
            options = json.loads(args.request_extra.read_text())
            if not isinstance(options, dict):
                raise ValueError("Request options must be a JSON object")
            extra.update(options)
        client = ChatClient(
            model=args.model,
            base_url=args.base_url,
            api_key_env=args.api_key_env,
            max_tokens=args.max_tokens,
            timeout=args.request_timeout,
            attempts=1,
            extra_body=extra,
        )
        config.update(
            model=client.model,
            base_url=client.base_url,
            api_key_env=client.api_key_env,
            max_tokens=client.max_tokens,
            request_timeout=client.timeout,
            request_extra=extra,
            no_tools=args.no_tools,
            delivered_attempts=3,
            provider_attempts=5,
        )
    with run_lock(args.output, create=not args.resume):
        config_path = args.output / "run.json"
        if args.resume:
            if json.loads(config_path.read_text()) != config:
                raise ValueError("Resume inputs or settings differ from the saved run")
        else:
            write_json(config_path, config)
        cases = args.output / "cases"
        cases.mkdir(exist_ok=True)
        results = []
        for index, episode in enumerate(episodes):
            path = cases / f"{index:04d}.json"
            if path.exists():
                row = json.loads(path.read_text())
                if row["id"] != episode.id:
                    raise ValueError("Saved episode id differs from the input")
            else:
                row = (
                    score_binprop(episode, library, timeout=args.timeout)
                    if args.method == "binprop"
                    else _direct_episode(
                        episode,
                        client,
                        cases / f"{index:04d}-attempts.json",
                        no_tools=args.no_tools,
                        timeout=args.timeout,
                    )
                )
                write_json(path, row)
            results.append(row)
            print(
                f"{index + 1}/{len(episodes)} {episode.id}: {row['score']:.6f}",
                flush=True,
            )
        summary = {
            "method": args.method,
            "game": args.game,
            "episode_count": len(results),
            "mean_score": fmean(row["score"] for row in results),
        }
        write_json(args.output / "summary.json", summary)
        print(json.dumps(summary, indent=2))
    return 0


def main(argv=None):
    try:
        return run(parser().parse_args(argv))
    except (ValueError, KeyError, OSError, ProviderError) as error:
        print(f"Run stopped: {error}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
