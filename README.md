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
