# Method: what the numbers mean

## Design

One experiment = the same harness, the same ordered tasks, one **arm** at a time.

- **baseline**: the main LLM makes every decision by prompting (or your `legacy=` code) and writes the output.
- **menu**: a decision model makes every decision by single-token menu readout. The main LLM still writes the output.
- **hybrid**: menu readout first; decisions with confidence below τ (or an explicit abstain) are handed to the main LLM, exactly as the baseline would answer them.

Arms run sequentially so a local GPU serves one configuration at a time, after one warm-up call per endpoint (no cold-start in the first task). With "Clean" mode tasks run one at a time, so measured latency is not a queueing artefact.

## Task accuracy

Each row is scored by your `score()` (or the built-in comparison) as 0/1 or a float. For arm *a* against the baseline *b*, the paired difference per task is `d_i = score_a,i − score_b,i`.

- **Δ accuracy** = mean(d). **95% CI** = paired bootstrap (2000 resamples, seeded) of that mean.
- **Verdict** is a non-inferiority test at your margin *m* (default 5 points): **safe** if the CI's lower bound ≥ −m; **hurts** if the point estimate is below −m *and* the whole CI is below zero (a loss that is both large and detectable); otherwise **not proven** (the data neither shows nor rules out a loss that size; the report says whether the estimate leans worse, and how many tasks would settle it).
- **Sign test**: exact two-sided binomial p on the discordant tasks (a McNemar test for 0/1 scores). It answers “do the two arms differ?”, which is a different question from “is the difference small enough?”; the verdict uses the latter.

## Speed, tokens, cost

- **End-to-end latency** per task = wall time of `run()` plus the recorded original duration of any *replayed* tool call (a cache hit costs ~0 wall time but the tool really took that long). Reported as mean/p50/p95; the headline speed-up is ratio of paired means with a bootstrap interval.
- **Main-LLM calls / tokens** count calls to the main-LLM endpoint (generation *and* baseline decisions). Decision-model calls and time are reported separately (`Decider ms`), so an offload never hides its own cost.
- **Cost** uses the $/M-token prices you enter per endpoint (0 for local models: the tool then shows tokens and latency instead of dollars).

## Per-decision analysis

Decisions are matched between an arm and the baseline by a fingerprint of `(site, key, state, instructions, options)` within the same task. Only *matched* decisions (both arms saw identical input) enter the per-site table and the threshold sweep. Decisions that follow a different earlier decision have different state and appear only in end-to-end numbers.

- With ground truth (`task["truth"]`): decision accuracy of the decision model vs of the baseline LLM, per site.
- Without: agreement between them (which is *not* accuracy).
- **Threshold sweep**: for τ on a grid, the share of matched decisions the model would keep (confidence ≥ τ) and, with truth, the accuracy of the hybrid that keeps those and uses the baseline LLM's answers for the rest. The **suggested τ** is the lowest one whose hybrid accuracy is within 1 point of the LLM's (or, without truth, ≥97% agreement on kept decisions) and that keeps at least 5 decisions.
- The sweep is a **screening estimate**. It assumes the LLM's answers on the escalated decisions are what the hybrid would get, which is exact per decision but ignores downstream effects. **Verify** re-runs the harness end to end at that τ; only that result is a claim.

The curve is built from the data, one point per distinct confidence value (not a fixed grid), so it is exact for models whose useful thresholds all sit between 0.99 and 1.0, which is where over-confident models live. The x axis is coverage (the share of decisions the decision model answers, most confident first), so the effect of every threshold is readable without knowing the confidence scale.

## Site recommendations and the policy

“Move” if the model's accuracy over the site's matched decisions is within 3 points of the LLM's (truth) or agreement ≥ 95% (no truth). “Move with confidence ≥ x” if only a thresholded hybrid meets the bar. “Keep on the LLM” otherwise.

The **recommended policy** is the set of those per-site choices: a confidence floor per site (`tau_by_site`; a floor above 1 means “always ask the LLM”). It is still an estimate, built from per-site curves. **Verify this policy end to end** runs the baseline and exactly this routing as a new experiment, which is the only result that counts as a claim about the combination.

## How many tasks are enough

The interval on an accuracy difference is set by how often the two arms *disagree* on a task, not by accuracy itself. With paired differences of variance σ², a 95% half-width of *m* needs about 3.84·σ²/m² tasks. When a verdict is “not proven” the report shows that number for your observed disagreement rate: a candidate that flips 1 task in 10 needs roughly 150 tasks to be shown non-inferior at a 5-point margin. Small runs are for finding out *where* to look; the verify step on a larger set is for proving it.

## Imported (replay) harnesses

A harness built from a call log is scored on **fidelity**, not accuracy: the fraction of moved decisions that match
the logged answer. The baseline arm replays the user's own logged prompt, so it should score near 1.0, and the
paired comparison then answers "does moving these decisions change what the pipeline decides?".

Two limits apply on top of everything below. The logged prompts are fixed, so a changed decision cannot change what
a later step sees — no downstream effect is measurable. And the log is the reference, not the truth: if the original
model was wrong, reproducing it faithfully scores 1.0. For a real accuracy claim, add ground truth, which means
writing the harness (`docs/HARNESS.md`) or labelling a sample. See [TRACES.md](TRACES.md).

## Threats to validity (read before quoting a number)

1. **Baseline prompt quality.** A weak baseline prompt flatters any alternative. Use `legacy=` with your real prompt.
2. **Task set size and representativeness.** 30 tasks give wide intervals; the tool says “not proven” rather than hide it. Tasks should look like your traffic.
3. **Ground truth.** Truth you wrote yourself may share your blind spots; use a held-out labelled sample if you can.
4. **Shared hardware.** Other jobs on the same GPU inflate latency; the tool cannot separate them out. Repeat runs when in doubt.
5. **Zero-shot general models** are often over-confident, so thresholding buys little. That is a real result about that model, not a tool limitation.
6. **Replay caveat.** Tool results are shared across arms; a decision that changes the *arguments* of a tool produces a live call for that arm. Latency includes it; accuracy reflects it.
7. **An imported log's token and cost figures are only as good as the log.** Missing token counts are estimated
   from text length; the report says when it estimated, and your own measured averages override it.
8. **Nothing here is a claim about the decision model in general.** It measures one harness on one task set.
