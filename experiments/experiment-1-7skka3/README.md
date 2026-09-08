# How often Qwen3-8B's chain of thought verbalizes the verified cause of its behavior

## What this is
For 100 `Qwen3-8B` thinking-mode behaviors from the paper "Would this change your answer?" (arXiv 2608.16747, released chive run `qwen3_8b_wildchat_thinking_eval`) whose cause was pinned down by a verified counterfactual (|effect| ≥ 50 pp, verification ≥ 9, clean edit), we resample the original prompt 30 times with thinking on, grade behavior with the paper's grader, and ask an LLM judge whether the `<think>` trace attributes the behavior to the causal factor. The same judgment is made for an inert factor from the same investigation (87/100) and for the visible answer.

Result (judge v3, primary): reasoning attributes the behavior to the causal factor in 42.0% of behavior-exhibiting completions (95% cluster-bootstrap CI 34.2–50.0, n = 2163); inert factor 32.6% (24.5–40.9); paired difference +10.7 pp (2.4–19.2). Cross-family judge agreement on Q2: kappa 0.468. The as-planned judge (v2, factor shown as edit text) gave 39.8% but failed the pre-registered kappa ≥ 0.4 check (0.370); its defect (reading the factor as the edited text) was fixed by rewriting each factor into an outcome-blind description of the original conversation content. Both versions are in `results/`.

## Layout
- `src/select_investigations.py`: selection rule (seed 42) → `results/manifest.jsonl`, `results/prompts.jsonl`, `results/selection_summary.json`.
- `src/resample.py`, `src/resample_job.sh`: SGLang server + two-stage token-level thinking cap (1024 think / 2048 total), 30 samples per prompt; writes `resample/completions.jsonl` + `run_meta.json` to `$SILICO_EXPERIMENT_ARTIFACTS_DIR`.
- `src/patch_boundary.py`: fixes the one pre-fix boundary artifact in the production output → `resample/completions_patched.jsonl` (used downstream).
- `src/grader.py`: paper's grader prompt/tool (vendored) with each investigation's classifier question → `results/behavior_grades.jsonl`.
- `src/factor_rewrite.py`: outcome-blind rewrite of each factor's edit text into a description of the original content → `results/factor_descriptions.jsonl` (judge v3 input).
- `src/judge.py`: Q1 mention / Q2 attribution / Q3 alternative judge; `--factor-descriptions` selects the v3 prompt → `results/judge_verdicts_v3.jsonl`, `results/judge_verdicts_answer_v3.jsonl`, validation files `results/judge_validation_{gpt,temp1}_v3.jsonl`; v2 outputs are the same names without `_v3`.
- `src/analyze.py`: cluster bootstrap, per-investigation table, slices, validation kappas, 20 random examples → `results/summary.json`, `results/per_investigation.csv`, `results/examples.json`, `results/plot_data.json` (`--version v2` writes the `_v2` variants).
- `figures/prepare_data.py` + `figures/<name>/plot.py`: figure bundles (`per_investigation_attribution_hist`, `pooled_rates`, `paired_difference`, `slices`, `fresh_vs_original_behavior_rate`).
- `vendor/chive/`: byte-for-byte vendored chive modules (commit 2423952f), see `vendor/VENDORED.md`.
- `report_summary.json`, `claims_manifest.json`: report contract.

## How to reproduce
```
apy src/select_investigations.py                      # needs the chive data under $SILICO_EXPERIMENT_ARTIFACTS_DIR/chive_data
# GPU job (job-core, 1xH100, ~18 min): bash experiments/experiment-1-7skka3/src/resample_job.sh --concurrency 24
apy src/patch_boundary.py
uv run --no-sync python src/grader.py --completions $SILICO_EXPERIMENT_ARTIFACTS_DIR/resample/completions_patched.jsonl
uv run --no-sync python src/factor_rewrite.py
C=$SILICO_EXPERIMENT_ARTIFACTS_DIR/resample/completions_patched.jsonl
uv run --no-sync python src/judge.py --completions $C --factor-descriptions results/factor_descriptions.jsonl --tag v3 --out results/judge_verdicts_v3.jsonl
uv run --no-sync python src/judge.py --completions $C --factor-descriptions results/factor_descriptions.jsonl --tag v3 --segment answer --factors causal --only-behavior --out results/judge_verdicts_answer_v3.jsonl
uv run --no-sync python src/judge.py --completions $C --factor-descriptions results/factor_descriptions.jsonl --model openai/gpt-5.6-terra --sample 300 --tag v3_gpt --out results/judge_validation_gpt_v3.jsonl
uv run --no-sync python src/judge.py --completions $C --factor-descriptions results/factor_descriptions.jsonl --temperature 1 --sample 300 --tag v3_temp1 --out results/judge_validation_temp1_v3.jsonl
apy src/analyze.py --completions $C --version v3 && apy src/analyze.py --completions $C --version v2
uv run --no-sync python figures/prepare_data.py
```
Grader/judge calls go through OpenRouter (`OPENROUTER_API_KEY`): grader `anthropic/claude-sonnet-4.6` (100 calls), judge `anthropic/claude-sonnet-5` (v2: 7773 calls, v3: 7773 calls, 187 rewrite calls, 2×300 temp-1 validation), second-family judge `openai/gpt-5.6-terra` (2×300 calls).

## Outputs
- Fresh completions: `artifact://truthful-ai-7a2189/experiments/exp_01m1z3kydjek0arxfb1w7skka3/resample/completions.jsonl` (31 MB, 100 prompts × 30, raw job output, job 940569256367, 2026-09-08) and pod copy `/srv/silico-state/users/u-ec27714b7bf8/artifacts/silico/experiments/_flat/exp_01m1z3kydjek0arxfb1w7skka3/resample/{completions.jsonl,completions_patched.jsonl}`. Model `Qwen/Qwen3-8B` revision `b968826d9c46dd6066d109eabc6255188de91218`, SGLang fork 0.0.0.dev1+gecadc97c0, temperature 1.0, top_p 0.95, top_k 20, think cap 1024, total 2048.
- Source data (downloaded 2026-09-07 from `adamkarvonen/chive-data`, run `qwen3_8b_wildchat_thinking_eval`): `/srv/silico-state/users/u-ec27714b7bf8/artifacts/silico/experiments/_flat/exp_01m1z3kydjek0arxfb1w7skka3/chive_data/datasets/qwen3_8b_wildchat_thinking_eval/`.
- Results: `results/summary.json` (primary, judge v3), `results/summary_v2.json`, `results/per_investigation.csv`, `results/examples.json`, per-item verdicts in `results/judge_verdicts*.jsonl`.
