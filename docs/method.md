# Method

## Discovery

Each iteration asks the model for a Boolean function and description, then
runs it on winning and losing positions:

`delta = P(True | winning) - P(True | losing)`

Reject invalid or slow functions and those below the delta threshold.
To limit redundancy, compare the winning positions where each motif holds.
Containment is the larger of the two intersection fractions. A candidate can
replace overlapping motifs only if its delta is higher than each of theirs.
At the library size cap, it must also beat the weakest motif it replaces.
Decisions and example boards become feedback for the next proposal.

Defaults: delta at least 0.01 for chess and greater than 0.01 for Go,
containment cap 0.70, no size cap. Chess also rejects duplicate names,
identical or complementary outputs, and activation rates outside 0.005-0.95.

This runner implements the delta/containment loop, without the research
harness's proxy, precheck and extra feedback steps. Use the bundled libraries
to evaluate the paper's fixed motif sets.

## Inference

Compute the fraction of positives where the motif holds and negatives where it
does not. Support fit is the smaller fraction.
Rank positive-delta motifs by fit times delta, breaking ties by fit, delta,
then name. Return up to three with fit at least 0.5; if none qualifies,
return the highest-ranked valid motif.

Inference returns the original descriptions and functions without model calls.
Execution failures and source/metadata mismatches are listed in `errors` and
excluded from selection.

## Held-out scoring

Direct requests three functions and selects those with support fit at least 0.5.
If none qualifies, select the valid function with the highest fit. BinProp uses
the same threshold, but ranks by fit times delta.

Score each selected motif by the lowest of three rates: true on positives,
false on negatives, and false on nearby negatives. Average within each episode,
then across episodes. This also applies when only one or two motifs are selected;
no valid output scores zero.

Chess Direct excludes functions with invalid held-out outputs before selection;
Go counts invalid outputs as incorrect. This preserves the paper's evaluators.
Held-out scores never enter prompts or repair feedback.

## Direct retries

Allow one answer and two code repairs. Invalid code counts as an attempt;
syntax, compilation and support-execution errors are used as repair feedback.
Empty answers, truncations and retryable provider failures do not consume
code attempts. Stop after five consecutive request failures, or immediately
on authentication and other non-retryable HTTP errors. Only answer text is used,
not the reasoning channel.

## Execution and run records

Functions run in a subprocess with a deadline covering compilation and execution.
Chess functions must pass color-mirror checks; Go functions may use fixed komi
assumptions. These checks are not a sandbox: isolate generated code from secrets
and network access.

Runs save settings and an input hash. [Checkpoints](iteration.md#resume-the-same-run)
also save feedback for resuming. A fixed sampling seed does not make model
responses deterministic.
