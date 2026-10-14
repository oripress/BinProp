"""Paper Direct prompts; only the eight visible boards are interpolated."""

from pathlib import Path

CHESS_SYSTEM = "You produce concise chess motif predicates and one final python-chess predicate. Return exactly one Python code block defining def explain(fen: str) -> bool, with helper predicates when requested."
GO_SYSTEM = "You write concise Python functions for Go board motifs. Return code only."


def direct_messages(episode, *, no_tools=False):
    name = episode.game + "-direct"
    if episode.game == "chess" and no_tools:
        name += "-no-tools"
    template = (
        (Path(__file__).parent / "prompts" / (name + ".txt")).read_text().rstrip("\n")
    )

    def visible(positions, label):
        if episode.game == "chess":
            return "\n".join(f"{i}. {p.board}" for i, p in enumerate(positions, 1))
        return (
            "\n\n".join(
                f"{label} {i}:\n" + "\n".join(p.board)
                for i, p in enumerate(positions, 1)
            )
            + "\n"
        )

    user = template.replace(
        "{{positive_support}}", visible(episode.positive, "Positive")
    )
    user = user.replace("{{negative_support}}", visible(episode.negative, "Negative"))
    user = user.replace("{{color}}", episode.positive[0].color)
    return [
        {
            "role": "system",
            "content": CHESS_SYSTEM if episode.game == "chess" else GO_SYSTEM,
        },
        {"role": "user", "content": user},
    ]
