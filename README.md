# Marketing Preference Probe (v0.1)

Compare **real human headline A/B test winners** against model predictions from:

- `typesafe/jev-1.14` (decisions)
- `openai/gpt-6-luna-decisions` (decisions)
- `openai/gpt-6-luna` (generative chat)

No API calls occur by default. Requires Python 3.11+, `uv`, and an OpenRouter key for live tests. Model IDs and API response schemas should be validated with smoke tests before a full run.

## 1. Install

```bash
git clone https://github.com/pabloloyola/marketing-preference-probe.git
cd marketing-preference-probe
uv sync
# Make sure OPENROUTER_API_KEY is set in your local environment (do not commit it)
uv run marketing-probe smoke                 # dry run, free
uv run marketing-probe smoke --live          # three real API requests
```

Use `read -s OPENROUTER_API_KEY` and `export OPENROUTER_API_KEY` to enter a key interactively, if needed.

## 2. Download the actual human dataset

Get **Upworthy exploratory packages CSV** from the original archive: https://osf.io/jd64p/ .
The original archive is explained at https://upworthy.natematias.com/about-the-archive .
Place the downloaded file under `data/`, for example `data/upworthy-archive-exploratory-packages-03.12.2020.csv`.

The dataset is **not** included in GitHub. No synthetic human outcomes are used.

```bash
uv run marketing-probe prepare --csv data/upworthy-archive-exploratory-packages-03.12.2020.csv --n 100
```

Filtering: same test, same nonempty image ID, same lede, at least 1,000 impressions per arm, no flagged randomization problems; one pair per test; randomized A/B order reproducibly. If there aren't enough eligible comparisons, reduce `--n`.

## 3. Run safely

```bash
uv run marketing-probe run --pairs data/pairs.jsonl --limit 3    # DRY RUN; $0
uv run marketing-probe run --pairs data/pairs.jsonl --limit 3 --reps 1 --live --budget 0.10 --max-calls 9
uv run marketing-probe analyze --pairs data/pairs.jsonl --predictions runs/predictions.jsonl
```

After inspecting the output:

```bash
uv run marketing-probe run --limit 100 --reps 1 --budget 1.00 --max-calls 300 --live
uv run marketing-probe analyze
```

The budget guardrail uses reported `usage.cost`, or reserves $0.003 per request when cost is missing. It is **best-effort, not a provider-enforced hard limit**. Keep `--max-calls` small during testing. Runs append JSONL and skip previously completed (pair, model, repetition) records.

## Interpretation

- `p_A` predicts **whether A has higher CTR**, not the fraction of humans who click A.
- Human winner labels are noisy, particularly for nearly tied CTRs. A future version should account for binomial uncertainty.
- This is text-only. Matching image IDs isolate the headline comparison; multimodal banner evaluation is separate.
- Public historical data may have appeared in pretraining. This is a feasibility probe, not a definitive behavioral benchmark.
- Generative JSON probabilities and decision-model response distributions are not necessarily calibrated to the same target.
- Grounding in real prior preference examples is planned for a second phase.

## Data attribution

Matias, J. N., Munger, K., Le Quere, M. A., & Ebersole, C. (2021). *The Upworthy Research Archive, a time series of 32,487 experiments in U.S. media.* Scientific Data 8, 195. https://doi.org/10.1038/s41597-021-00934-7

## Files

- `src/marketing_probe/api.py`: OpenRouter chat and decisions adapters
- `src/marketing_probe/dataset.py`: human data selection
- `src/marketing_probe/cli.py`: CLI, budget guardrails, evaluation
- `tests/`: local sanity tests

OpenRouter docs: https://openrouter.ai/api/alpha/decisions and https://openrouter.ai/docs

## Troubleshooting: generative Luna returns no content

Some reasoning calls can finish without visible text if the output budget is exhausted. The generative adapter now uses `max_tokens=1024` (reasoning + final text), checks for empty/null content, and reports an actionable `APIError` instead of crashing with an `AttributeError`. This can increase cost for difficult prompts. The runner does not silently retry paid requests. To resume an interrupted run, simply rerun the same command: successfully recorded (pair, model, repetition) calls are skipped.

## Analysis v2: offline diagnostic report

After the 100-pair run finishes:

```bash
uv run marketing-probe report
open runs/analysis_v2/report.md
```

Produces `report.md`, `summary.json`, `accuracy.svg`, `brier.svg`,
`confidence_bins.csv`, `ctr_margin_bins.csv`, and `confident_errors.json`.
No API key, network access, or paid calls needed. Paired bootstrap intervals compare
models on shared pair IDs. Confidence bins describe **winner correctness**, not human
click-share calibration. Near-tied CTR results are noisy; this report is exploratory.

## Experiment 2: Grounding predictions in observed A/B outcomes

Uses separate original-archive experiments for training examples (80), probability calibration (40), and an untouched test (80). **The original 100-pair exploratory test IDs are excluded.** The test labels never enter model prompts.

```bash
git pull origin main
uv run marketing-probe prepare2 --csv data/upworthy-archive-exploratory-packages-03.12.2020.csv
uv run marketing-probe run2 --limit 2                   # dry-run, $0
uv run marketing-probe run2 --limit 2 --budget .10 --max-calls 24 --live
uv run marketing-probe analyze2
```

After the smoke test, finish all comparisons (`80 cal + 80 test` per model/condition, 960 requests maximum):

```bash
uv run marketing-probe run2 --budget 1.00 --max-calls 960 --live
uv run marketing-probe analyze2
```

The runner resumes from completed records. The `--max-calls` and `--budget` controls apply to that output file including earlier successful requests; this is a **best-effort guardrail**, not a provider-enforced hard spending limit. Run the complete test only when the dry-run and the initial paid calls work.

**Conditions:** zero-shot and a fixed 5-example prompt containing real human A/B winners from train. **Calibration:** fit a shrinkage coefficient on model probabilities from `cal` only, and apply it to `test`. **Statistical reference:** regularized logistic regression using simple headline features and only `train`.

Review `runs/experiment2_report.json`. Do not compare partial test coverage as if it were a complete benchmark. This evaluates headline A/B winner forecasts—not individual consumer choice or click shares. Public archive contamination remains possible.

## Experiment 3 — Retrieval-grounded choice prediction

Uses the **existing Experiment 2 train/cal/test split**. For each new pair, retrieve
the five most similar A/B tests from the 80 training examples using lexical TF-IDF,
align alternatives A and B (including reversing the historical winner when needed),
and include those outcomes in the decision prompt. The target's result is never used
to retrieve or construct its examples.

```bash
git pull origin main
uv run python -m unittest discover -s tests -v
uv run marketing-probe preview3 --k 5
uv run marketing-probe run3 --limit 2                         # dry run, no charge
uv run marketing-probe run3 --limit 2 --budget .10 --max-calls 12 --live
uv run marketing-probe analyze3
```

For the full run (40 calibration + 80 test pairs, 3 models; 360 calls):

```bash
uv run marketing-probe run3 --budget .50 --max-calls 360 --live
uv run marketing-probe analyze3
```

The report in `runs/experiment3_report.json` includes **retrieved vs random
few-shot vs zero-shot** results using the saved Experiment 2 predictions, plus
a retrieval-only baseline (no model API calls). Calibrated probabilities use
only the existing separate calibration set. Intermediate/partial reports show
coverage; do not interpret them as full benchmark results.

**Notes:** TF-IDF is inexpensive, but 80 training examples may not include
genuinely relevant A/B tests. Retried runs skip completed calls. The spending
guardrails are best-effort, not a provider-enforced hard cap. The evaluation
is historical headline winner prediction, not customer-level simulation.


## Retrieval v2: larger pool, semantic embeddings, choice-structure features

The previous 80-example TF-IDF Experiment 3 is preserved. This upgraded experiment
builds a separate 2,000-comparison retrieval corpus, excluding the original exploratory
100 and *all* Experiment 2 train/cal/test **test IDs and exact headline pairs**.

**1. Prepare pool (offline, no OpenRouter calls):**
\`\`\`bash
uv run marketing-probe prepare-retrieval --csv data/upworthy-archive-exploratory-packages-03.12.2020.csv --size 2000
\`\`\`

**2. Inspect lightweight choice-aware and lexical retrieval first:**
\`\`\`bash
uv run marketing-probe preview-retrieval --method structure
uv run marketing-probe preview-retrieval --method tfidf
\`\`\`

**3. Install local embedding dependencies (larger download, but no API tokens):**
\`\`\`bash
uv sync --extra retrieval
uv run --extra retrieval marketing-probe preview-retrieval --method semantic
uv run --extra retrieval marketing-probe preview-retrieval --method hybrid
\`\`\`

Embedding model: \`Qwen/Qwen3-Embedding-0.6B\`, through
\`sentence-transformers\`. The first execution downloads the model and computes
embeddings for the historical corpus + held-out queries. It caches normalized
vectors in \`data/retrieval_v2/embeddings.npz\`; subsequent semantic/hybrid
previews reuse the cache. This may consume substantial disk/memory and time on a
Mac; CPU fallback is possible.

**4. Validate dry run, then try a small paid experiment:**
\`\`\`bash
uv run marketing-probe run-retrieval --method structure --limit 2
uv run marketing-probe run-retrieval --method structure --limit 2 --budget .10 --max-calls 12 --live
uv run marketing-probe analyze-retrieval --method structure
\`\`\`

When ready to compare Qwen-embedding retrieval:
\`\`\`bash
uv run --extra retrieval marketing-probe run-retrieval --method hybrid --limit 2 --budget .10 --max-calls 12 --live
uv run --extra retrieval marketing-probe analyze-retrieval --method hybrid
\`\`\`

Full experiments omit \`--limit\` (360 requests/method). Different methods write
separate JSONL files. The analysis reuses Experiment 2 zero-shot and random
few-shot predictions and adds a retrieval-only baseline. Calibration is fitted
solely on the 40 calibration outcomes.

**Method definitions:** \`tfidf\` = A/B-aligned lexical cosine;
\`structure\` = cosine on measurable differences (headline length, punctuation,
numbers, simple emotional/curiosity dictionaries, pronouns);
\`semantic\` = A/B-aligned Qwen embeddings; \`hybrid\` = equal-weight semantic
and structure similarity. Structure features are rough proxies, **not validated
psychological or marketing labels**. Greater semantic similarity alone is not
proof of better behavioral relevance. Compare held-out performance, not
subjective retrieval quality alone.

No live requests without \`--live\`. The budget guards are best-effort and
cannot enforce a strict provider billing ceiling. Do not use test results to
pick the best method and present them as unbiased confirmatory evidence.


## Final offline diagnostic: untouched Upworthy experiments

After inspecting Experiments 1–3, treat all earlier test sets as exploratory.
This diagnostic sets aside a NEW held-out set that excludes:
- the original 100 exploratory pair test IDs,
- all 80/40/80 Experiment 2 test IDs,
- all 2,000 retrieval-pool test IDs,
- duplicate exact pairs regardless of A/B order.

It trains a **regularized logistic regression** on signed A–B headline-style
features from the same 2,000-example retrieval pool used by the hybrid
retrieval-only predictor. Logistic regression has **no intercept** so
reversing the choices reverses the probability. Neither model requires
OpenRouter inference. Both use a fresh calibration set for optional
shrinkage, and a further untouched test set for evaluation.

\`\`\`bash
git pull origin main
uv run python -m unittest discover -s tests -v

# Fresh 40 calibration + 120 test comparisons. Does not change earlier files.
uv run marketing-probe prepare-final \
  --csv data/upworthy-archive-exploratory-packages-03.12.2020.csv \
  --cal 40 --test 120

# Uses locally cached/downloaded Qwen3 embeddings, no OpenRouter API calls.
uv run --extra retrieval marketing-probe analyze-final

cat runs/final_diagnostic/report.md
\`\`\`

Outputs:
- \`data/final_holdout/cal.jsonl\`, \`test.jsonl\`, \`manifest.json\` (never committed)
- \`runs/final_diagnostic/report.md\`
- \`runs/final_diagnostic/summary.json\`

The manifest records hashes; analysis refuses changed test/cal files or
overlap with prior experimental cases. Diagnostic comparisons include a
constant 50/50 model, logistic regression, and hybrid retrieval-only model,
each with raw and calibration-shrunk probabilities, paired bootstrap
accuracy/Brier intervals, and accuracy broken down by observed CTR
difference-to-standard-error ratio (<1, 1–2, >=2).

**The reliability bands are descriptive:** they use observed CTR differences
to group tests, not model inputs, and the z-score is an approximate measure
that may be noisy. The statistical model uses handcrafted textual features;
this is not a learned customer representation. This is a *one-time* final
diagnostic: choosing the best method from its test scores and further tuning
on this same set would invalidate confirmatory claims.

**Dataset-size correction (2026-11-06):** With the original selection requirements
(at least 1,000 impressions per arm, same image/lede, one pair per
randomized experiment), this archive contains only 2,482 eligible distinct
tests. The existing 2,000-case retrieval pool and the 300 previously used
tests leave at most 182 for completely fresh evaluation, so the original
100+300 holdout request was impossible. Defaults are now **40 calibration
and 120 test cases**. The candidate enumeration uses all eligible tests,
then excludes every protected experiment ID and duplicate pair before
splitting. If fewer than 160 survive, the command reports the actual
number remaining rather than silently changing the inclusion requirements.

## Calibration Experiment 1 — observation uncertainty versus model confidence

This is a retrospective OFFLINE diagnostic using Experiment 2's 40 calibration and 80 test cases, their click counts, and saved model predictions. No OpenRouter requests or Qwen model inference are necessary.

Commands:

    git pull origin main
    uv run python -m unittest discover -s tests -v
    uv run marketing-probe calibration1
    cat runs/calibration_1/report.md

Outputs under runs/calibration_1/: report.md, summary.json (per-model diagnostics and reliability bins), and observation_uncertainty.csv (posterior probabilities and credible intervals).

The observation model uses independent Beta(a,a) CTR priors with a=0.5 and a=2.0, 4,000 posterior draws, and a normal approximation to the repeat-test winner probability for 1,000 *future* impressions per arm. It reports both latent CTR superiority and repeat-test outcome probabilities; these are different events.

**Caution:** Posterior results are conditioned on the historical clicks and are not independent ground truth or valid future-outcome labels. Previously inspected cases remain exploratory. The current confidence transformation is restricted shrinkage toward 0.5, not a full calibration fit. Brier scores and reliability bins use the recorded *hard winner* labels.

For Experiment 2 predictions only, if local retrieval prediction files are absent:

    uv run marketing-probe calibration1 --predictions runs/experiment2_predictions.jsonl

To increase simulation draws:

    uv run marketing-probe calibration1 --draws 8000 --repeat-n 2000
