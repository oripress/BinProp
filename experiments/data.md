# Data access

No processed datasets, outputs, or collection script are included. Follow the
paper's data-collection procedures using the raw games below; rebuilding may
yield different positions.

## Chess

- **Discovery:** January-November 2025 PGNs from the
  [Lichess Elite Database](https://database.nikonoel.fr/), evaluated with
  Stockfish 18 at depth 12. Keep 100,000 winning positions (+125 to +250
  centipawns) and 100,000 losing positions (-250 to -125), balanced by side
  to move. Scores are for the player to move; exclude mates.
- **Validation and test:** master-game PGNs from
  [PGN Mentor](https://www.pgnmentor.com/files.html). Follow *Chess Data Collection*
  in the paper for game-disjoint splits, 100 starting positions per split,
  quiet continuations, and depth-18 rescoring.
- Engine: [Stockfish](https://stockfishchess.org/download/).

## Go

- Games: [Joe's Go Database](https://pjreddie.com/projects/jgdb/), restricted
  to 19-by-19, no-handicap games with both players at least 7 dan.
- Engine: [KataGo](https://github.com/lightvector/KataGo), with
  `kata1-b18c384nbt-s9996604416-d4316597426`. Keep 100,000 winning and
  100,000 losing positions, balanced by side to move, with evaluations in
  +7.5 to +15.0 and -15.0 to -7.5 points.
- Follow *Go / Data Collection* in the paper for validation and test games,
  rescoring, continuations, and nearby negatives. Each split has 100 episodes,
  with 12 positives, 12 negatives, and 8 nearby negatives per episode before
  choosing support positions.

## Runner inputs

Write discovery positions as [labeled JSONL](../docs/formats.md#discovery-input-jsonl)
and evaluation positions as [episodes](../docs/formats.md#evaluation-episodes-jsonl).
Engine scores determine labels but are not shown to Direct.
