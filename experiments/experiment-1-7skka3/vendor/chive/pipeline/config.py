"""Pipeline configuration: DEFAULTS + per-run config.json loading.

DEFAULTS is the SINGLE source of truth for every pipeline hyperparameter:

- `run_pipeline.py` merges a run's config.json over it (unknown keys fail
  loudly — a misspelled knob must never silently fall back to its default).
- The standalone stage CLIs (generate_completions / screen_completions /
  investigate / verify_investigations / generate_synthetic_data) take their
  argparse defaults FROM this dict. Never hardcode a second copy of a default
  in a CLI — divergent defaults in different entry points are how a sweep
  silently runs with different settings than the pipeline it's compared to.

Values must stay pure literals (no imports from stage modules) so every stage
module can import this one without circular-import risk.
"""

import json
from pathlib import Path

from chive import paths

# ---------------------------------------------------------------------------
# Pipeline defaults — all hyperparameters in one place.
#
# Per-run config.json overrides any of these. Keys in config.json that match
# a key here take precedence; anything not in config.json uses the default.
# Keys starting with "_" (e.g. "_comment") are ignored.
# ---------------------------------------------------------------------------

DEFAULTS = {
    # --- Global ---
    "run_name": None,                  # REQUIRED: human-readable run label
    # ONE concurrency number for every stage. It is the per-endpoint in-flight
    # cap enforced at the leaf (LLMClient.call), so it means "max simultaneous
    # requests to the model endpoint" — fan-out (e.g. stage-3 run_prompt sampling
    # 10 completions) is throttled automatically and never multiplies past this.
    # Each stage also uses it as a loose bound on how many tasks it opens at once.
    "concurrency": 100,
    "batch_chunk_size": 10000,          # requests per batch chunk (batch mode)
    # NOTE: the investigated model's completion sampling length (500 tok) and
    # temperature (1.0) are FIXED in generate_completions.py
    # (MAX_RESPONSE_TOKENS / COMPLETION_TEMPERATURE), not config;
    # top_p/top_k/min_p come from the Qwen profile in llm_client.

    # Completions per prompt — ONE knob shared by stage 1 and every stage-3
    # run_prompt experiment, so the two cannot drift. 10 for training-data runs
    # (cheap); 30 for held-out test runs (statistical power: at 30-vs-30, ~20+
    # percentage-point rate differences are distinguishable from sampling
    # noise; at 10-vs-10 only ~40+). The investigator is always shown at most
    # the first 10; the grader counts all of them.
    "n_completions": 10,
    # Behavior grader: applies the stage-2 classifier_question to every
    # completion of every stage-3 experiment (per-completion verdicts, counted
    # in code). Always on.
    "grader_llm": "anthropic:claude-sonnet-4-6",

    # Investigated-model (target) behavior-sampling knobs — shared by Stage 1 AND
    # Stage 3's run_prompt so the original behavior and its counterfactuals are
    # drawn from the SAME generative distribution. Defaults reproduce the paper's
    # no-think runs byte-for-byte (budget None -> chat_template enable_thinking:False,
    # 500-token answers). To investigate a THINKING model: set a budget
    # (e.g. 1024) and a larger target_max_tokens (e.g. 2048 = ~1024 answer headroom).
    # The vLLM server must be launched with --reasoning-config '{}' (the pool's
    # submit_qwen3_8b.sh already is). Temperature stays the fixed COMPLETION_TEMPERATURE.
    "target_thinking_budget": None,
    "target_max_tokens": 500,          # keep equal to generate_completions.MAX_RESPONSE_TOKENS unless investigating a thinking model

    # --- Stage 0 + 1: Prompt slice + completion generation ---
    # Stage 0 slices the first N rows of the canonical pool into the run dir;
    # Stage 1 generates completions via an llm_client backend (vLLM server /
    # OpenRouter) — same spec form as stages 2-4. No WildChat streaming, no
    # offline-vLLM/Slurm GPU jobs, no tokenization (the pool is messages-only).
    "prompt_pool": str(paths.RUNS_DIR / "wildchat_plus_agentic" / "prompts.jsonl"),
    "stage1_llm": None,                # REQUIRED backend spec, e.g. "vllm:Qwen/Qwen3-32B-FP8@http://node:8000" or "openrouter:Qwen/Qwen3.5-397B-A17B"
    "stage1_n_prompts": 100,           # prompts to take from the pool
    "stage1_offset": 0,                # skip the first N pool rows (for disjoint runs)
    "exclude_files": None,             # completions/prompts files to dedup the slice against
    "vllm_job_name": "vllm_server",    # Slurm job name used to resolve a bare "vllm:MODEL" (no @url) stage1_llm spec
    "completions_file": None,          # optional pre-built completions .jsonl (one PromptCompletions record per line); skips stages 0-1

    # --- Stages 2 + 4: Screening & verification (LLM judge w/ tool call) ---
    # ONE knob for the judge model: a backend spec, same form as stage1_llm.
    # The judge model is experiment-critical, so it is required and explicit —
    # no bare-model-name default that could silently decide who judges. Batch
    # mode is Anthropic-only and parses the model out of this spec.
    "judge_llm": None,                 # REQUIRED, e.g. "anthropic:claude-opus-4-6" or "vllm:Qwen/...@http://node:8000"
    "judge_max_tokens": 8192,          # max response tokens
    "judge_thinking_budget": 4096,     # extended thinking budget (async AND batch; 0 disables)
    "stage2_n": 100,                   # prompts to screen (None = all)
    "stage2_mode": "async",            # "async" or "batch"
    "skip_stage5_synthetic": False,    # if true, skip Stage 5 entirely
    "stage4_mode": "async",            # "async" or "batch"
    "stage4_batch_chunk_size": 500,    # verification prompts are large (~330KB each)
    "stage4_n_verifications": 1,       # independent verifier samples per investigation (score = ensemble mean)

    # --- Stage 3: Investigation (Opus agent with tool use) ---
    "stage3_min_score": 3,             # min screening score to investigate
    "stage3_n": None,                  # optional cap after score filtering
    "stage3_llm": None,                # REQUIRED backend spec for stage 3 agent (same form as stage1_llm)
    "target_llm": None,                # REQUIRED backend spec for the investigated model (run_prompt counterfactuals)
    "stage3_max_tokens": 16384,        # max response tokens for investigator
    "stage3_thinking_budget": 4096,    # extended thinking budget for investigator
    "stage3_max_agent_turns": 10,      # max agent turns per investigation (was 30; capped at 10 to stay under context-window limit — each turn cycle adds ~10k tokens of conversation history)
    "stage3_thorough": False,          # methodology mode: False=efficient (training data, 5-8 experiments, throughput); True=thorough (held-out test data, rule out competing hypotheses, ~10-15 experiments — bump stage3_max_agent_turns accordingly)
    "stage3_n_completions_shown": 10,  # full completions the investigator sees per experiment (grader counts ALL n). Lower (e.g. 5) to bound agent context with long thinking completions.
    "stage3_max_skip_rate": 1.0,       # hard cap on stage-3 skip rate (1.0 = warn only; pass e.g. 0.001 for canonical runs)

    # --- Stage 5: Synthetic data generation ---
    "stage5_mode": "async",                               # "async" or "batch" (batch is Anthropic-only)
    "stage5_llm": None,                                   # REQUIRED backend spec for synthesis (same form as stage1_llm)
    "stage5_min_interest_score": 3,
    "stage5_min_verification_score": 7,
    "stage5_n_counterfactual_experiments": 3,
    "stage5_seed": 42,
    # Stage 5 thinking is part of the data-generating process — for runs where
    # the trained model will be deployed without reasoning, set this to 0 so
    # the synth data doesn't bake in `<think>` tokens the trained model can't
    # produce. Applies to async AND batch mode (the two must not diverge).
    "stage5_thinking_budget": 2048,
}


def load_config(config_path: Path) -> dict:
    """Load per-run config, merged with defaults.

    Fails loudly on unknown keys: a misspelled knob (e.g. "stage3_max_token")
    would otherwise be silently ignored and the default silently used. Keys
    starting with "_" (e.g. "_comment") are allowed and passed through.
    """
    user_config = json.loads(config_path.read_text())
    unknown = sorted(
        k for k in user_config if k not in DEFAULTS and not k.startswith("_")
    )
    assert not unknown, (
        f"Unknown config key(s) in {config_path}: {unknown}. "
        f"Every key must match a key in chive/pipeline/config.py DEFAULTS "
        f"(prefix comments with '_'). Known keys: {sorted(DEFAULTS)}"
    )
    merged = dict(DEFAULTS)
    merged.update(user_config)
    assert merged["run_name"], f"config['run_name'] is required in {config_path}"
    return merged
