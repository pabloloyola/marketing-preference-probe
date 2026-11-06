# Research memorandum: marketing choice prediction and calibration

**Date:** 2026-11-06  
**Project:** Marketing Preference Probe  
**Status:** Exploratory experiments completed; fresh offline diagnostic completed; calibration workstream proposed  
**Evidence:** Project source code and user-reported terminal outputs. Raw local datasets and prediction files were not available to rerun here.

## Executive summary

We studied whether decision-oriented inference can predict which headline earns the higher click-through rate (CTR) in historical randomized Upworthy A/B experiments, at lower serving cost than generative inference. We compared Jev decisions, Luna decisions, and Luna generative; tested zero-shot inference, five fixed observed examples, lexical retrieval, semantic retrieval using local Qwen3 embeddings, and a hybrid of semantic similarity and headline-choice structure. We tested shrinkage calibration and statistical/retrieval-only baselines.

The principal conclusion is **operational efficiency without demonstrated behavioral probability skill**. The decisions endpoints were faster and cheaper on our first 100 requests, but none of the tested approaches reliably beat simple probability baselines. Adding observed historical choices generally failed to improve and often worsened winner prediction. Qualitatively better semantic retrieval did not imply better downstream prediction. A hybrid retrieval-only baseline matched the best LLM accuracy on an exploratory 80-case test, but its apparent performance did not persist on a fresh 120-case test.

The current calibration procedure is not a general solution: it shrinks predicted probabilities toward 0.5 while preserving the selected winner. Large improvements in Brier score can therefore occur without any improvement in discrimination. We recommend a new workstream dedicated to **defining the forecast target and separating uncertainty in the underlying CTRs from model uncertainty** before further model or retrieval iteration.

### Decisions

- **Close** headline retrieval-prompt optimization as an exploratory phase.
- **Keep** these experiments as a reproducible feasibility and negative-results case study.
- **Prioritize** calibration, probability semantics, click-count uncertainty, and out-of-sample decision value.
- **Do not claim** synthetic customer preference or utility recovery from these headline-level experiments.

---

## 1. Motivation and scientific questions

The motivating engineering premise was that a decisions endpoint could replace repeated generative sampling in marketing simulations, potentially providing cheaper choice distributions. This is useful only if the reported distributions relate to target human behavior. Serving cost, winner ranking, and probability calibration are separate dimensions.

Research questions:

1. Can decision inference be delivered faster and more cheaply than generative inference?
2. Can it predict which headline wins on real randomized click outcomes?
3. Do observed historical choices improve predictions when supplied through examples or retrieval?
4. Are reported probabilities meaningful estimates of an identified event?
5. Do apparent improvements generalize to previously unused experiments?

The present study measures an **aggregate historical headline comparison**, not an individual-level preference, a purchase decision, or a utility parameter.

## 2. Data and experimental design

### 2.1 Upworthy archive and selection

The pipeline uses the Upworthy Research Archive exploratory-packages CSV. Eligible pairs come from a single randomized test and have distinct headlines, the same nonempty image ID and lede, a minimum of 1,000 impressions per arm, valid click counts, and no flagged experiment problem. The loader selects at most one pair per experiment and reproducibly randomizes A/B order. This reduces some variation within tests; it does not make all cross-test environments comparable.

Under these filters, exhaustive preparation on the user's CSV found **2,482 distinct eligible tests**. This number is specific to our filters and is not the number of tests in the entire archive.

For each arm, empirical CTR is the number of clicks divided by impressions. The code labels A the winner if its observed CTR is strictly higher, and B otherwise. Thus an exact CTR tie is assigned B, while a model probability of exactly 0.5 is classified as A when scoring. We should correct and explicitly document this asymmetric tie policy in future work.

### 2.2 What does the probability represent?

The model output p_A is prompted as the **probability headline A has higher CTR**. It is not the click probability of an exposed reader, the CTR of headline A, or the probability that a specific person prefers A.

Several target events must be distinguished:

- **Latent superiority:** probability the underlying population CTR of A exceeds that of B.
- **Repeat-test winner:** probability A obtains a higher *realized* CTR than B in a new finite experiment of specified sample sizes.
- **Magnitude of treatment effect:** expected difference between the underlying CTRs, or expected incremental clicks.
- **Individual preference:** probability a specified person or segment chooses or clicks A.

The reported benchmark treats the **realized historical winner** as a hard binary label and scores p_A against it. That is not automatically a valid reference for latent superiority: sampling variation may reverse a real but small CTR advantage.

The API also presents a cross-model comparability issue. The generative endpoint emits a number in JSON after being asked for a subjective probability. The decisions endpoints provide a distribution over answer alternatives. We have not established that these numerical distributions have identical probabilistic meaning.

### 2.3 Evaluation partitions

| Phase | Cases | Purpose |
|---|---:|---|
| Experiment 1 | 100 tests | Initial zero-shot, latency, and cost probe |
| Experiment 2 | 80 train, 40 calibration, 80 test | Fixed five-shot versus zero-shot |
| Retrieval v1 | 80 training pairs | Preliminary TF-IDF retrieval |
| Retrieval v2 | 2,000 separate historical pairs | Semantic and hybrid retrieval; reused 40/80 evaluation |
| Final offline diagnostic | 2,000 existing training/retrieval pairs plus 40 new calibration and 120 new tests | Independent check of retrieval-only versus logistic regression |

Experiment 2 excluded Experiment 1 test IDs. The expanded retrieval corpus excluded prior selected experiments and duplicate headline pairs. The final offline diagnostic excluded the prior selected cases and the 2,000-case pool by experiment ID and exact pair signature; it also checked file hashes.

Because we repeatedly inspected and compared systems on the original 80-case test, it should be considered **exploratory**. The later 120-case holdout was fresh relative to that iteration but is now exposed; tuning on it and calling the same set confirmatory again would be invalid.

### 2.4 Methods evaluated

**Zero-shot:** Predict A or B using headline wording and a common historical publisher framing.

**Fixed few-shot:** Include five historical A/B outcomes sampled from the small training split, without tailoring the examples to the target.

**TF-IDF retrieval:** Use cosine similarity on headline-level TF-IDF vectors. Pair alignment is the maximum of matched A-to-A/B-to-B and swapped A-to-B/B-to-A similarity; reversed examples have their winner label flipped.

**Semantic retrieval:** Embed both headlines with local Qwen3-Embedding-0.6B, rank matched/swapped headline pair similarities, and present the top five historical comparisons from a 2,000-test pool.

**Choice-structure / hybrid retrieval:** Represent a comparison as signed A-minus-B differences in observable headline features (length, punctuation, numbers, emotion/curiosity keyword fractions, pronouns), and combine its structural cosine with semantic similarity. These are handcrafted proxies, not validated behavioral determinants.

**Retrieval-only predictor:** Smooth a similarity-weighted average of the winner labels of retrieved historical comparisons; no LLM is called.

**Feature logistic regression:** Train a supervised logistic classifier using signed differences of headline features. The final version used 2,000 training examples and excluded an intercept to impose A/B reversal symmetry.

**Post-hoc shrinkage calibration:** Fit alpha between 0 and 1 on calibration data and return calibrated_p = 0.5 + alpha times (raw_p - 0.5). This reduces confidence but cannot improve ranking or fix a wrong decision boundary.

---

## 3. Experimental results

### 3.1 Initial zero-shot feasibility: 100 tests

| Configuration | Accuracy | Raw Brier | Mean latency | Reported inference cost for 100 calls |
|---|---:|---:|---:|---:|
| Jev decisions | 57% | 0.2855 | 0.524 s | $0.001862 |
| Luna decisions | 51% | 0.3430 | 0.607 s | $0.002314 |
| Luna generative | 51% | 0.2891 | 3.076 s | $0.007166 |

A paired-bootstrap interval for Jev minus generative Luna accuracy was approximately **-3 to +15 percentage points** around an observed +6 points; the corresponding interval versus Luna decisions was **-4 to +16 points**. These differences are not established as statistically reliable.

The decision endpoints were much faster and cheaper for the tested provider, request format, and model versions. The recorded costs exclude local retrieval and engineering. Yet all three uncalibrated probability scores were worse than the Brier score of **0.25** from always forecasting 0.5.

### 3.2 Fixed historical examples: 80 test cases

| Model | Zero-shot accuracy | Fixed five-shot accuracy | Zero-shot raw Brier | Five-shot raw Brier |
|---|---:|---:|---:|---:|
| Jev | 51.25% | 47.50% | 0.3068 | 0.2912 |
| Luna generative | **56.25%** | 52.50% | 0.2707 | 0.2852 |
| Luna decisions | 48.75% | 46.25% | 0.3482 | 0.3441 |

All conditions had 80/80 test coverage and 40/40 calibration coverage. Five fixed examples did not improve accuracy for any model. A simple feature logistic regression trained on just 80 training comparisons achieved 46.25% accuracy and raw Brier 0.2564 on the same evaluation split.

Shrinkage often improved Brier by collapsing probabilities toward neutrality. Jev zero-shot had alpha about **0.104**, producing calibrated Brier **0.249984**; such a tiny difference from 0.25 is not evidence of practically meaningful prediction skill.

### 3.3 Retrieval: improving the examples did not improve outcomes

The initial 80-example TF-IDF retrieval returned mostly superficial lexical matches. Expansion to 2,000 tests and Qwen embeddings yielded more plausible topic-level and narrative matches. Hybrid retrieval additionally emphasized stylistic differences between headlines. This improved **qualitative relevance**, not necessarily behavioral relevance.

On the 80-case evaluation:

| Model | Zero-shot | Fixed examples | Semantic five-shot | Hybrid five-shot |
|---|---:|---:|---:|---:|
| Jev | 51.25% | 47.50% | 43.75% | 50.00% |
| Luna generative | **56.25%** | 52.50% | 43.75% | 50.00% |
| Luna decisions | 48.75% | 46.25% | 35.00% | 51.25% |

Semantic examples lowered accuracy relative to zero-shot for all models; hybrid recovered some accuracy but did not consistently improve on zero-shot.

The **semantic retrieval-only** baseline reached 42.50% accuracy and raw Brier 0.2945. The **hybrid retrieval-only** baseline reached **56.25%** accuracy, matching zero-shot generative Luna on these 80 tests, but with raw Brier **0.2631** and calibrated Brier **0.2585**, both worse than constant 0.25.

This was a reason to test generalization, not proof that hybrid retrieval predicts human behavior.

### 3.4 Fresh offline diagnostic: 120 tests

The final check compared a regularized logistic classifier and the hybrid retrieval-only baseline on a fresh 120-test sample, with 40 independent calibration cases. Both used the same 2,000 historical comparisons as their source of observed behavior. No new API inference was needed.

| System | Accuracy | Raw Brier | Raw log loss | Calibrated Brier |
|---|---:|---:|---:|---:|
| Constant p=0.5 | 44.17%* | **0.2500** | **0.6931** | **0.2500** |
| Choice-feature logistic | **55.00%** | 0.2517 | 0.6968 | 0.250024 |
| Hybrid retrieval-only | 48.33% | 0.2801 | 0.7637 | 0.250920 |

*The constant predictor's accuracy is mechanically the frequency of A winners because the scorer resolves exact 0.5 predictions to A. Treat it as a probability baseline, not a meaningful classification policy.*

Logistic regression predicted **66 of 120** winners correctly; hybrid retrieval predicted **58 of 120**. Paired-bootstrap 95% intervals were wide:

- Logistic minus constant tie-to-A: **+10.8 points**, interval **[-1.7, +23.3]**.
- Logistic minus hybrid retrieval: **+6.7 points**, interval **[-5.8, +20.0]**.
- Constant minus hybrid retrieval: **-4.2 points**, interval **[-16.7, +8.3]**.

All include zero. Logistic regression achieved modestly higher accuracy but **did not beat the trivial 0.5 probabilistic forecast** on Brier or log loss.

The calibration shrinkage factor on this fresh 40-case calibration set was **0.2341** for logistic regression and **0.1525** for hybrid retrieval, indicating that their unshrunk probability magnitudes were not well supported under the current scoring target.

### 3.5 Outcome-reliability stratification

We grouped the 120 realized A/B tests by the absolute empirical CTR difference divided by an estimated standard error:

| Empirical margin reliability | n | Logistic accuracy | Hybrid retrieval accuracy |
|---|---:|---:|---:|
| z below 1 | 57 | 49.1% | 47.4% |
| z from 1 to below 2 | 34 | 58.8% | 52.9% |
| z at least 2 | 29 | 62.1% | 44.8% |

The logistic pattern suggests greater discrimination when winner outcomes are less ambiguous, but the clearer group contains only 29 cases and was examined post hoc. A CTR z-score is a descriptive approximation, not a proof that the latent human winner is known.

---

## 4. What the evidence supports

**Supported within the tested settings:** Decision-oriented APIs can reduce request latency and provider-reported cost; naive probability magnitudes are poor; simple shrinkage substantially reduces overconfidence-like losses under hard-winner scoring; stronger retrieval does not automatically yield better behavioral predictions.

**Suggestive but uncertain:** Some interpretable headline-choice structure may help classify clearer CTR outcomes. The final logistic model's 55% accuracy and the z-stratified trend warrant methodological interest but do not establish a deployable predictor.

**Not supported:** Jev is inherently more behaviorally faithful than a generative model; retrieved examples reliably improve forecasting; the hybrid retrieval-only accuracy is stable; any system supplies reliable calibrated winner probabilities or CTR uplift estimates.

**Not tested:** Individual or segment-specific preferences, utility recovery, personalized counterfactual decisions, distribution shift to Japanese advertising, and creative/banner evaluation.

This is a **limited negative-result/feasibility report**, not a general result that AI decision models are incapable of preference prediction.

## 5. Why the existing calibration evidence is incomplete

### 5.1 Hard winner labels conflate statistical and behavioral uncertainty

For arm X in {A,B}, observed CTR is clicks_X divided by impressions_X. Finite impressions produce sampling noise, sometimes reversing which arm appears better. A model forecasting that A has a higher underlying CTR can be correctly uncertain even when the observed binary label is B, especially with small effect sizes.

We should distinguish:

1. Probability of **latent population superiority** (the true CTR of A exceeds B).
2. Probability of **winning a new finite A/B experiment**, conditional on new arm sample sizes.
3. Predicted **CTR difference or incremental clicks**.
4. An **individual or segment's response probability**.

The present score is for a historical realized-winner label. Switching targets requires a new definition and appropriate evaluation data.

### 5.2 Decision outputs need semantic auditing

The decisions API's probabilities may be preferences over answer choices rather than epistemic probabilities of a future experimental event. The generative endpoint's subjective number may differ systematically in scale and interpretation. Even an algorithm that ranks correctly can have badly calibrated numbers.

Calibration should be examined *conditional on a documented forecast question*, not performed on anonymous scores of unspecified meaning.

### 5.3 Our transform is only one-parameter shrinkage

Current mapping:

calibrated_p = 0.5 + alpha * (p_raw - 0.5), for alpha in [0,1].

It cannot correct an overall A/B bias, improve discrimination, or learn nonlinear reliability. It can make poor predictions look nearly optimal in Brier terms by pushing them close to 0.5. Its success is diagnostic of excessive confidence **for the chosen label**, not evidence that we recovered the correct probabilistic behavior.

### 5.4 The external information is weak

Our input has two headlines but not the original readers' characteristics, platform context, content timing, or a user history. Topic-related historical examples do not guarantee transfer of treatment effects. This is exactly why results should not be described as persona simulation or utility inference.

---

## 6. Calibration research agenda

### 6.1 Formalize the target before fitting a calibrator

**Primary practical estimand:** Probability A will beat B in a *future, specified-size* A/B trial, not merely a distribution over output labels. Future trial arm sizes must be part of the target.

**Secondary estimand:** Posterior probability that the latent CTR of A exceeds B. This depends on an explicit observation model and prior and is **not a ground-truth label**.

**Business estimand:** Expected CTR uplift or expected additional clicks from choosing A, with an uncertainty interval. A higher probability of being the winner does not imply higher expected value.

For a simplified observation model:

- clicks_A ~ Binomial(impressions_A, true_CTR_A).
- clicks_B ~ Binomial(impressions_B, true_CTR_B).

With explicit beta priors, posterior simulation can estimate uncertainty in each CTR, the difference between CTRs, and a winner event. Varying prior strength and assuming common versus heterogeneous experiment effects will be essential sensitivity checks.

**Critical guardrail:** A posterior computed from held-out click counts is an uncertainty-aware *description of observed data*. It must not be treated as an independent prediction target validating a forecasting model that did not see those counts.

### 6.2 Candidate calibrators

Compare a small pre-specified family fit only on calibration data:

- Identity mapping (no calibration).
- Current shrinkage toward 0.5.
- Regularized logistic/Platt mapping: calibrated_p = sigmoid(intercept + slope * logit(p_raw)), with clipping at 0/1.
- Reversal-symmetric temperature mapping with zero intercept and nonnegative slope.
- Monotone isotonic mapping only when calibration sample size supports it; 40-case or 100-case fits may be too unstable.

Also compare to p=0.5, a **training-only majority/prevalence probability**, and a supervised statistical choice model. Selection must use inner cross-validation or a dedicated calibration split, never the final test.

### 6.3 Metrics

Report proper probabilities and practical choice metrics separately:

- Brier and log loss with paired uncertainty intervals.
- Calibration-in-the-large, slope, and intercept; counts in reliability bins and interval uncertainty.
- Discrimination (accuracy with explicit tie/abstention policy, AUC when appropriate).
- Sharpness, meaning whether probabilities move away from 0.5 *when justified*.
- A/B order complementarity: p(A,B) should approximately equal 1 - p(B,A).
- Stability to paraphrase, example order, and repeated requests.
- Expected policy value/regret using realistic traffic, treatment costs, and CTR uplift when available.
- Stratification by impression count and expected outcome ambiguity, without interpreting tiny groups as proof.

Expected calibration error (ECE) can be supplemental but is highly sensitive to bins and sample size; it must not replace proper scores or uncertainty intervals.

### 6.4 Immediate offline experiment: calibration and observation uncertainty

**Phase 1 — Audit existing data and code:** Verify click-count arithmetic, model-prompt semantics, tie behavior, split integrity, and A/B reversal invariance. Document the distinct outcome events being modeled.

**Phase 2 — Model observation noise:** Use stored clicks and impressions to estimate beta-binomial posterior distributions for the arm CTRs, along with probabilities of latent superiority and repeat-test winner under specified sample sizes. Include prior sensitivity and synthetic positive-control experiments with known ground truth.

**Phase 3 — Diagnose present predictions:** For saved Jev/Luna outputs, quantify confidence distributions, reliability curves, calibration slopes, Brier decomposition where applicable, and discrimination versus fixed baselines. Compare raw and shrunk forecasts, including when the shrinkage factor is nearly zero.

**Phase 4 — Fit calibrators with strict data separation:** Compare identity, shrinkage, Platt/temperature, and a statistical baseline on calibration-only data; freeze a protocol. The old 80 and 120 test sets have been inspected and are useful for diagnostics, **not new confirmatory claims**.

**Phase 5 — Acquire independent test evidence:** The present eligible Upworthy pool is nearly exhausted by the protected splits. Meaningful confirmatory evaluation requires a distinct dataset, a new collection, or prospective A/B tests. Do not silently reclaim previously viewed tests.

**Stopping rule:** A method must demonstrate out-of-sample proper-score improvement *and* usable decision value over base-rate and supervised baselines, not merely make probabilities flatter. Confidence intervals, target semantics, and context dependence must be explicit.

### 6.5 Questions worth distinguishing experimentally

1. **Overconfidence versus label noise:** Are model probabilities genuinely too extreme for the event they predict, or are they being scored against an especially noisy finite-sample winner?
2. **Ranking versus probability information:** Does a model discriminate between likely winners while failing calibration, or is there no stable ranking information to calibrate?
3. **Format dependence:** Does an API's choice distribution change under A/B swapping or equivalent phrasing despite an identical underlying economic comparison?
4. **Transfer versus personalization:** Can an aggregate headline model generalize at all, and what additional audience-level repeated choices are necessary to identify individual utility?

The third question is particularly relevant to decision-model architecture; the fourth would require new data and a separate research design.

---

## 7. Engineering and reporting recommendations

- Preserve hashes and local archived copies of the archive CSV, processed splits, 2,000-example pool, prediction JSONL, provider responses where permissible, and final reports. Run files and datasets are excluded from Git, so this memo alone is not a full reproducibility package.
- Store model identifier, served model/provider, prompt version, calibration target, code SHA, seed, timing, reported cost, and experiment identifiers with every run.
- Correct the exact-0.5 and exact-CTR-tie policies before comparing classifier accuracy with trivial baselines.
- Treat latency and provider cost as one dimension; include local retrieval computation and embedding-model storage for end-to-end deployment economics.
- Use counts and CTR differences, not only winner labels, in calibration diagnostics; account for binomial uncertainty.
- Preserve the exploratory/final distinction. Any further tuning based on the final 120 cases makes them exploratory.
- Distinguish probability quality, economic decision quality, and causal utility recovery. None implies the others.

## 8. Recommended scientific framing and decision

The more consequential question may be:

> Under what conditions can a model's distribution over choices be interpreted and calibrated as a probability of human outcomes, given incomplete context, finite behavioral evidence, and uncertainty about latent preferences?

This distinguishes three objects that are often conflated: (i) probability assigned to a model action or answer, (ii) uncertainty about the human decision process, and (iii) noise in observed experimental outcomes.

The present findings motivate this question but **do not establish a publishable contribution, novelty, or utility recovery**. A literature review and decisive experiments with known or repeated behavioral ground truth would be needed.

**Decision:** Stop developing headline-retrieval prompting variants. Start a standalone **Calibration and Observation Uncertainty** phase in this repository, initially offline. Only after establishing a valid probability target and measurable out-of-sample skill should we revisit segment-specific or person-specific AI decision simulation.

---

## Evidence map and provenance

**Reported numeric results:** The first 100-case report discussed during Experiment 1; the complete user-run Experiment 2 80-case report; the two user-run semantic/hybrid retrieval reports covering the same 80 cases; and the user-run final 120-case diagnostic output and local report. All tabulated numbers above are transcribed or rounded from those reports, not independently recomputed from the local prediction files in this session.

**Code map:**

- src/marketing_probe/dataset.py — data eligibility, empirical arm CTRs, sample construction.
- src/marketing_probe/api.py — decisions and generative API adapters; p_A extraction; timing/cost.
- src/marketing_probe/experiment2.py — five-shot protocol, statistical baseline, Brier/log loss and shrinkage.
- src/marketing_probe/experiment3.py — initial TF-IDF retrieval.
- src/marketing_probe/retrieval_v2.py — enlarged pool, embeddings, choice-structure similarity, hybrid retrieval.
- src/marketing_probe/final_diagnostic.py — fresh holdout, logistic model, paired bootstrap, empirical z-score bands.

**Dataset reference:** Matias, J. N., Munger, K., Le Quere, M. A., & Ebersole, C. (2021). *The Upworthy Research Archive, a time series of 32,487 experiments in U.S. media.* Scientific Data 8, 195. DOI: 10.1038/s41597-021-00934-7.

**One-sentence conclusion:** Serving efficiency was demonstrated for our configurations; calibrated behavioral forecasting was not. The next study must define the probability's meaning before calibrating it.
