"""Single source of truth for data locations.

Every script that reads run dirs or experiment results resolves paths through this
module — never hardcode a data path elsewhere. The data tree is materialized by
``scripts/download_data.py`` (from the HuggingFace dataset repo) or by pointing
``CHIVE_DATA`` at an existing tree.

Layout under DATA_ROOT:
    runs/<run_name>/...            investigation run dirs (same internal layout as always:
                                   investigations.json, verification.json, transcripts/,
                                   binary_counterfactual_questions.jsonl, eval_set_*.jsonl,
                                   binary_evals/, selfexpl_cache/, selfexpl_evals/)
    experiments/<name>/...         per-experiment results/ + eval_sets/
"""

import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

DATA_ROOT = Path(os.environ.get("CHIVE_DATA", REPO_ROOT / "data"))
RUNS_DIR = DATA_ROOT / "runs"
EXPERIMENTS_DATA_DIR = DATA_ROOT / "experiments"

# The paper's canonical investigation runs.
CANONICAL_RUNS = {
    "gemma3-27b-it": "opus_investigates_gemma3_27b_it_5000_thorough_n30",
    "qwen3-8b": "opus_investigates_qwen3_8b_5000_thorough_n30",
    "qwen3-8b-petri": "opus_investigates_qwen3_8b_petri_thorough_n30",
    "qwen3.5-397b": "opus_investigates_qwen3p5_397b_5000_thorough_n30",
    "qwen3.5-397b-petri": "opus_investigates_qwen3p5_397b_petri_thorough_n30",
}


def run_dir(key_or_name: str) -> Path:
    """Resolve a run by canonical key (e.g. 'qwen3-8b') or literal dir name."""
    name = CANONICAL_RUNS.get(key_or_name, key_or_name)
    path = RUNS_DIR / name
    if not path.is_dir():
        raise FileNotFoundError(
            f"Run dir not found: {path}. Run scripts/download_data.py or set CHIVE_DATA."
        )
    return path


def experiment_data_dir(name: str) -> Path:
    """Data dir for one experiment (results/, eval_sets/)."""
    return EXPERIMENTS_DATA_DIR / name
