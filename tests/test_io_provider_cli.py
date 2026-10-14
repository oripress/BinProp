"""Offline release integration tests; synthetic boards and fake credentials only."""

import http.client
import io
import json
import os
import tempfile
import unittest
import urllib.error
import urllib.request
from contextlib import redirect_stderr, redirect_stdout
from email.message import Message
from pathlib import Path
from unittest.mock import patch

from helpers import CHESS_SOURCE, GO_SOURCE, chess_row, go_row

from binary_properties.cli import main
from binary_properties.io import (
    load_library,
    load_positions,
    position_from_dict,
    save_library,
)
from binary_properties.provider import ChatClient, ProviderError, _NoRedirect
from binary_properties.types import Library, Motif


class InputTests(unittest.TestCase):
    def test_strict_labels_and_go_color(self):
        for invalid in (True, "draw", 2, 0.0):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                position_from_dict(
                    {**go_row("B"), "label": invalid}, "go", require_label=True
                )
        row = go_row("B")
        del row["color"]
        with self.assertRaises(ValueError):
            position_from_dict(row, "go")
        self.assertEqual(position_from_dict(go_row("W", "losing"), "go").label, 0)

    def test_invalid_boards_and_blank_files(self):
        with self.assertRaises(ValueError):
            position_from_dict({"fen": "not-a-fen"}, "chess")
        with self.assertRaises(ValueError):
            position_from_dict({"board_rows": ["." * 9] * 9, "color": "B"}, "go")
        with tempfile.TemporaryDirectory() as directory:
            p = Path(directory) / "positions.jsonl"
            p.write_text("\n")
            with self.assertRaises(ValueError):
                load_positions(p, "go")
            p.write_text(json.dumps(go_row("B", 1)) + "\n")
            with self.assertRaises(ValueError):
                load_positions(p, "go")

    def test_chess_fen_requires_exactly_six_fields(self):
        complete = chess_row(True)["fen"]
        fields = complete.split()
        for fen in [" ".join(fields[:count]) for count in range(1, 6)] + [
            complete + " extra"
        ]:
            with (
                self.subTest(fen=fen),
                self.assertRaisesRegex(ValueError, "six-field FEN"),
            ):
                position_from_dict({"fen": fen, "color": "white"}, "chess")
        self.assertEqual(position_from_dict({"fen": complete}, "chess").board, complete)

    def test_library_roundtrip_and_source_hash(self):
        motif = Motif(
            "More stones", "The evaluated player has more stones.", GO_SOURCE, 1.0
        )
        with tempfile.TemporaryDirectory() as directory:
            p = Path(directory) / "basis.json"
            save_library(p, Library("go", [motif]))
            self.assertEqual(load_library(p).motifs[0], motif)
            raw = json.loads(p.read_text())
            raw["motifs"][0]["source"] += "\n"
            p.write_text(json.dumps(raw))
            with self.assertRaises(ValueError):
                load_library(p)

    def test_library_rejects_nonfinite_delta(self):
        with tempfile.TemporaryDirectory() as directory:
            p = Path(directory) / "basis.json"
            motif = Motif("More stones", "Description", GO_SOURCE, float("nan"))
            with self.assertRaises(ValueError):
                save_library(p, Library("go", [motif]))
            self.assertFalse(p.exists())


class ProviderTests(unittest.TestCase):
    def test_token_factory_default_request_and_trailing_slash(self):
        for base_url in (None, "https://api.tokenfactory.nebius.com/v1/"):
            with self.subTest(base_url=base_url):
                client = (
                    ChatClient("unit-model")
                    if base_url is None
                    else ChatClient("unit-model", base_url)
                )
                response = io.BytesIO(
                    json.dumps({"choices": [{"message": {"content": "code"}}]}).encode()
                )
                with (
                    patch.dict(
                        os.environ,
                        {
                            "NEBIUS_API_KEY": "nebius-test-key",
                            "OPENAI_API_KEY": "wrong-key",
                        },
                        clear=True,
                    ),
                    patch(
                        "binary_properties.provider._open", return_value=response
                    ) as opened,
                ):
                    self.assertEqual(
                        client.complete([{"role": "user", "content": "test"}]), "code"
                    )
                sent = opened.call_args.args[0]
                self.assertEqual(
                    sent.full_url,
                    "https://api.tokenfactory.nebius.com/v1/chat/completions",
                )
                self.assertEqual(
                    sent.get_header("Authorization"), "Bearer nebius-test-key"
                )
                self.assertEqual(json.loads(sent.data)["model"], "unit-model")

    def test_token_factory_does_not_fall_back_to_openai_credentials(self):
        with (
            patch.dict(
                os.environ, {"OPENAI_API_KEY": "wrong-provider-key"}, clear=True
            ),
            patch("binary_properties.provider._open") as opened,
            self.assertRaisesRegex(ProviderError, "NEBIUS_API_KEY"),
        ):
            ChatClient("unit-model").complete([])
        opened.assert_not_called()

    def test_explicit_key_request_and_response(self):
        response = io.BytesIO(
            json.dumps(
                {"choices": [{"message": {"content": "code"}, "finish_reason": "stop"}]}
            ).encode()
        )
        client = ChatClient(
            "unit-model",
            "https://provider.example/custom/v1/",
            api_key_env="CUSTOM_API_KEY",
            extra_body={"temperature": 0},
        )
        with (
            patch.dict(
                os.environ,
                {
                    "CUSTOM_API_KEY": "unit-only-key",
                    "NEBIUS_API_KEY": "wrong-provider-key",
                },
                clear=True,
            ),
            patch("binary_properties.provider._open", return_value=response) as request,
        ):
            self.assertEqual(
                client.complete([{"role": "user", "content": "test"}]), "code"
            )
        sent = request.call_args.args[0]
        self.assertEqual(
            sent.full_url, "https://provider.example/custom/v1/chat/completions"
        )
        self.assertEqual(sent.get_header("Authorization"), "Bearer unit-only-key")
        self.assertEqual(json.loads(sent.data)["temperature"], 0)

    def test_redirect_never_forwards_authorization(self):
        handler = _NoRedirect()
        opener = urllib.request.build_opener(handler)
        request = urllib.request.Request(
            "https://provider.example/v1/chat/completions",
            data=b"{}",
            headers={"Authorization": "Bearer unit-only-key"},
        )
        headers = Message()
        headers["Location"] = "https://different.example/collect"
        with patch.object(
            opener, "open", side_effect=AssertionError("No redirected request")
        ) as follow:
            with self.assertRaises(urllib.error.HTTPError) as error:
                opener.error("http", request, io.BytesIO(), 302, "Found", headers)
        self.assertEqual(error.exception.code, 302)
        follow.assert_not_called()

    def test_http_error_is_redacted_and_not_retried_for_authentication(self):
        client = ChatClient("unit-model", "https://provider.example/v1")
        failure = urllib.error.HTTPError(
            "https://provider.example",
            401,
            "secret-in-error",
            {},
            io.BytesIO(b"PRIVATE_RESPONSE"),
        )
        with (
            patch.dict(os.environ, {"NEBIUS_API_KEY": "unit-only-key"}, clear=True),
            patch("binary_properties.provider._open", side_effect=failure) as opened,
        ):
            with self.assertRaises(ProviderError) as error:
                client.complete([])
        self.assertEqual(str(error.exception), "Model endpoint returned HTTP 401")
        self.assertEqual(opened.call_count, 1)

    def test_transient_failure_has_bounded_retry(self):
        client = ChatClient("unit-model", "https://provider.example/v1", attempts=2)
        response = io.BytesIO(
            json.dumps({"choices": [{"message": {"content": "code"}}]}).encode()
        )
        failure = urllib.error.HTTPError(
            "https://provider.example", 503, "Unavailable", {}, io.BytesIO()
        )
        with (
            patch.dict(os.environ, {"NEBIUS_API_KEY": "unit-only-key"}, clear=True),
            patch(
                "binary_properties.provider._open", side_effect=[failure, response]
            ) as opened,
            patch("binary_properties.provider.time.sleep"),
        ):
            self.assertEqual(client.complete([]), "code")
            self.assertEqual(opened.call_count, 2)

    def test_interrupted_http_body_retries_without_consuming_answer(self):
        class ResponseSocket:
            def makefile(self, *args, **kwargs):
                return io.BytesIO(
                    b"HTTP/1.1 200 OK\r\nContent-Length: 100\r\n"
                    b"Content-Type: application/json\r\n\r\nPRIVATE_BODY"
                )

        interrupted = http.client.HTTPResponse(ResponseSocket())
        interrupted.begin()
        response = io.BytesIO(
            json.dumps({"choices": [{"message": {"content": "recovered"}}]}).encode()
        )
        with (
            patch.dict(os.environ, {"NEBIUS_API_KEY": "unit-only-key"}, clear=True),
            patch(
                "binary_properties.provider._open",
                side_effect=[interrupted, response],
            ) as opened,
            patch("binary_properties.provider.time.sleep") as sleep,
        ):
            self.assertEqual(
                ChatClient("unit-model", attempts=2).complete([]), "recovered"
            )
        self.assertEqual(opened.call_count, 2)
        sleep.assert_called_once_with(1)

    def test_http_and_connection_read_failures_have_bounded_redacted_retry(self):
        for failure in (
            http.client.IncompleteRead(b"PRIVATE_BODY", 100),
            http.client.RemoteDisconnected("PRIVATE_REASON"),
            ConnectionResetError("PRIVATE_REASON"),
        ):
            with self.subTest(failure=type(failure).__name__):

                class InterruptedResponse(io.BytesIO):
                    def read(self, *args, **kwargs):
                        raise failure

                with (
                    patch.dict(
                        os.environ, {"NEBIUS_API_KEY": "unit-only-key"}, clear=True
                    ),
                    patch(
                        "binary_properties.provider._open",
                        side_effect=[InterruptedResponse(), InterruptedResponse()],
                    ) as opened,
                    patch("binary_properties.provider.time.sleep") as sleep,
                    self.assertRaises(ProviderError) as error,
                ):
                    ChatClient("unit-model", attempts=2).complete([])
                self.assertTrue(error.exception.retryable)
                self.assertIsNone(error.exception.status_code)
                self.assertNotIn("PRIVATE", str(error.exception))
                self.assertEqual(opened.call_count, 2)
                sleep.assert_called_once_with(1)

    def test_missing_key_and_reserved_options(self):
        with self.assertRaises(ValueError):
            ChatClient(
                "unit-model", "https://provider.example/v1", extra_body={"messages": []}
            )
        with (
            patch.dict(os.environ, {}, clear=True),
            self.assertRaises(ProviderError) as error,
        ):
            ChatClient("unit-model", "https://provider.example/v1").complete([])
        self.assertFalse(error.exception.retryable)
        self.assertIsNone(error.exception.status_code)

    def test_http_errors_expose_status_and_retryability(self):
        client = ChatClient("unit-model", "https://provider.example/v1", attempts=1)
        for status in (302, 400, 401, 403, 404, 429, 500, 502, 503, 504):
            failure = urllib.error.HTTPError(
                "https://provider.example",
                status,
                "PRIVATE_REASON",
                {},
                io.BytesIO(b"PRIVATE_RESPONSE"),
            )
            with (
                self.subTest(status=status),
                patch.dict(os.environ, {"NEBIUS_API_KEY": "unit-only-key"}, clear=True),
                patch(
                    "binary_properties.provider._open", side_effect=failure
                ) as opened,
                self.assertRaises(ProviderError) as error,
            ):
                client.complete([])
            self.assertEqual(error.exception.status_code, status)
            self.assertEqual(
                error.exception.retryable, status in (429, 500, 502, 503, 504)
            )
            self.assertNotIn("PRIVATE", str(error.exception))
            self.assertEqual(opened.call_count, 1)

    def test_invalid_or_incomplete_response_is_retryable(self):
        client = ChatClient("unit-model", "https://provider.example/v1", attempts=1)
        responses = (
            b"not JSON",
            b"{}",
            b'{"choices": []}',
            b'{"choices": [{"message": {"content": null}}]}',
            b'{"choices": [{"message": {"content": " "}}]}',
            b'{"choices": [{"message": {"content": "code"}, "finish_reason": "length"}]}',
        )
        for response in responses:
            with (
                self.subTest(response=response),
                patch.dict(os.environ, {"NEBIUS_API_KEY": "unit-only-key"}, clear=True),
                patch(
                    "binary_properties.provider._open",
                    return_value=io.BytesIO(response),
                ),
                self.assertRaises(ProviderError) as error,
            ):
                client.complete([])
            self.assertTrue(error.exception.retryable)
            self.assertIsNone(error.exception.status_code)

    def test_connection_failures_are_retryable_and_redacted(self):
        client = ChatClient("unit-model", "https://provider.example/v1", attempts=1)
        for failure in (
            urllib.error.URLError("PRIVATE_REASON"),
            TimeoutError("PRIVATE_REASON"),
        ):
            with (
                self.subTest(failure=type(failure)),
                patch.dict(os.environ, {"NEBIUS_API_KEY": "unit-only-key"}, clear=True),
                patch("binary_properties.provider._open", side_effect=failure),
                self.assertRaises(ProviderError) as error,
            ):
                client.complete([])
            self.assertTrue(error.exception.retryable)
            self.assertIsNone(error.exception.status_code)
            self.assertNotIn("PRIVATE", str(error.exception))


class CliTests(unittest.TestCase):
    def test_discover_then_infer_both_games_without_network(self):
        for game, source, make in (
            ("chess", CHESS_SOURCE, chess_row),
            ("go", GO_SOURCE, go_row),
        ):
            with self.subTest(game=game), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                positive = make(True, 1) if game == "chess" else make("B", 1)
                negative = make(False, 0) if game == "chess" else make("W", 0)
                training = root / "positions.jsonl"
                training.write_text(
                    json.dumps(positive) + "\n" + json.dumps(negative) + "\n"
                )
                extra = root / "request.json"
                extra.write_text(
                    json.dumps({"metadata": {"private": "PRIVATE_OPTION_MARKER"}})
                )
                output = root / "run"
                completion = "```python\n" + source + "```"
                stdout, stderr = io.StringIO(), io.StringIO()
                with (
                    patch.dict(
                        os.environ, {"NEBIUS_API_KEY": "unit-only-key"}, clear=True
                    ),
                    patch(
                        "binary_properties.provider._open",
                        return_value=io.BytesIO(
                            json.dumps(
                                {"choices": [{"message": {"content": completion}}]}
                            ).encode()
                        ),
                    ) as request,
                    patch(
                        "urllib.request.urlopen",
                        side_effect=AssertionError("Network forbidden"),
                    ),
                    redirect_stdout(stdout),
                    redirect_stderr(stderr),
                ):
                    code = main(
                        [
                            "discover",
                            "--game",
                            game,
                            "--positions",
                            str(training),
                            "--output",
                            str(output),
                            "--model",
                            "unit-model",
                            "--iterations",
                            "1",
                            "--max-p95-ms",
                            "100",
                            "--request-extra",
                            str(extra),
                        ]
                    )
                self.assertEqual(code, 0, stderr.getvalue())
                self.assertEqual(
                    request.call_args.args[0].full_url,
                    "https://api.tokenfactory.nebius.com/v1/chat/completions",
                )
                self.assertEqual(
                    request.call_args.args[0].get_header("Authorization"),
                    "Bearer unit-only-key",
                )
                library = load_library(output / "basis.json")
                self.assertEqual(len(library.motifs), 1)
                self.assertEqual(library.motifs[0].delta, 1)
                for artifact in (output / "run.json", output / "basis.json"):
                    self.assertNotIn("unit-only-key", artifact.read_text())
                    self.assertNotIn("PRIVATE_OPTION_MARKER", artifact.read_text())
                comparison = root / "comparison.json"
                comparison.write_text(
                    json.dumps({"positive": [positive], "negative": [negative]})
                )
                explanation = root / "explanation.json"
                with (
                    patch(
                        "binary_properties.provider.ChatClient.complete",
                        side_effect=AssertionError("Inference must be local"),
                    ),
                    redirect_stdout(io.StringIO()),
                    redirect_stderr(stderr),
                ):
                    code = main(
                        [
                            "infer",
                            "--library",
                            str(output / "basis.json"),
                            "--positions",
                            str(comparison),
                            "--output",
                            str(explanation),
                        ]
                    )
                self.assertEqual(code, 0, stderr.getvalue())
                result = json.loads(explanation.read_text())
                self.assertEqual(result["selected"][0]["support_fit"], 1)
                self.assertEqual(result["selected"][0]["source"], source)
                self.assertEqual(result["errors"], [])

    def test_discover_provider_environment_and_flag_overrides(self):
        cases = (
            (
                [],
                "env-model",
                "https://environment.example/v1",
                "NEBIUS_API_KEY",
            ),
            (
                [
                    "--model",
                    "flag-model",
                    "--base-url",
                    "https://flag.example/custom/v1",
                    "--api-key-env",
                    "CUSTOM_API_KEY",
                ],
                "flag-model",
                "https://flag.example/custom/v1",
                "CUSTOM_API_KEY",
            ),
        )
        for flags, expected_model, expected_url, expected_key in cases:
            with self.subTest(flags=flags), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                training = root / "positions.jsonl"
                training.write_text(
                    json.dumps(go_row("B", 1))
                    + "\n"
                    + json.dumps(go_row("W", 0))
                    + "\n"
                )
                stderr = io.StringIO()
                with (
                    patch.dict(
                        os.environ,
                        {
                            "MOTIF_MODEL": "env-model",
                            "MOTIF_BASE_URL": "https://environment.example/v1",
                            expected_key: "unit-only-key",
                        },
                        clear=True,
                    ),
                    patch("binary_properties.cli.ChatClient") as client,
                    patch(
                        "binary_properties.cli.discover", return_value=Library("go")
                    ) as discovery,
                    redirect_stdout(io.StringIO()),
                    redirect_stderr(stderr),
                ):
                    code = main(
                        [
                            "discover",
                            "--game",
                            "go",
                            "--positions",
                            str(training),
                            "--output",
                            str(root / "run"),
                            *flags,
                        ]
                    )
                self.assertEqual(code, 0, stderr.getvalue())
                self.assertEqual(client.call_args.args, (expected_model, expected_url))
                self.assertEqual(client.call_args.kwargs["api_key_env"], expected_key)
                self.assertIs(discovery.call_args.args[2], client.return_value)

    def test_discover_still_requires_model(self):
        with (
            patch.dict(os.environ, {"NEBIUS_API_KEY": "unit-only-key"}, clear=True),
            patch("binary_properties.cli.ChatClient") as client,
            patch("binary_properties.cli.load_positions") as positions,
            redirect_stdout(io.StringIO()),
            redirect_stderr(io.StringIO()) as stderr,
        ):
            code = main(
                [
                    "discover",
                    "--game",
                    "chess",
                    "--positions",
                    "unused.jsonl",
                    "--output",
                    "unused-run",
                ]
            )
        self.assertEqual(code, 1)
        self.assertIn("--model", stderr.getvalue())
        self.assertIn("MOTIF_MODEL", stderr.getvalue())
        client.assert_not_called()
        positions.assert_not_called()

    def test_abbreviated_fen_is_rejected_before_provider_initialization(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            training = root / "positions.jsonl"
            shorthand = {**chess_row(True, 1), "fen": chess_row(True)["fen"].split()[0]}
            training.write_text(
                json.dumps(shorthand) + "\n" + json.dumps(chess_row(False, 0)) + "\n"
            )
            output = root / "run"
            stderr = io.StringIO()
            with (
                patch("binary_properties.cli.ChatClient") as client,
                patch("binary_properties.cli.discover") as discovery,
                redirect_stdout(io.StringIO()),
                redirect_stderr(stderr),
            ):
                code = main(
                    [
                        "discover",
                        "--game",
                        "chess",
                        "--positions",
                        str(training),
                        "--output",
                        str(output),
                        "--model",
                        "unit-model",
                        "--base-url",
                        "https://provider.example/v1",
                        "--iterations",
                        "1",
                    ]
                )
            self.assertEqual(code, 1)
            self.assertIn("six-field FEN", stderr.getvalue())
            client.assert_not_called()
            discovery.assert_not_called()
            self.assertFalse(output.exists())

    def test_inference_failure_is_nonzero_and_preserves_report(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            basis = root / "basis.json"
            save_library(
                basis,
                Library("go", [Motif("Broken", "Broken code", "invalid code", 0.5)]),
            )
            groups = root / "groups.json"
            groups.write_text(
                json.dumps({"positive": [go_row("B")], "negative": [go_row("W")]})
            )
            target = root / "result.json"
            with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
                code = main(
                    [
                        "infer",
                        "--library",
                        str(basis),
                        "--positions",
                        str(groups),
                        "--output",
                        str(target),
                    ]
                )
            self.assertEqual(code, 2)
            self.assertTrue(json.loads(target.read_text())["errors"])


if __name__ == "__main__":
    unittest.main()
