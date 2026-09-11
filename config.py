"""Paths, model identifiers and shared constants.

Override any of these with environment variables of the same name, e.g.

    EN_TRP_XLSX=/data/en-trp.xlsx python run_submission.py
"""

import os

# --- Team / submission -------------------------------------------------------
TEAM_NAME = os.environ.get("TEAM_NAME", "SCE-KIIT")

# --- Models ------------------------------------------------------------------
MODEL_ID = os.environ.get("MODEL_ID", "agnivamaiti/KokLLaMA-3.2-3B-Instruct")
BASE_ID = os.environ.get("BASE_ID", "meta-llama/Llama-3.2-3B-Instruct")

# --- Data --------------------------------------------------------------------
# Official WMT 2026 blind test files (source column only, no references).
EN_TRP_XLSX = os.environ.get("EN_TRP_XLSX", "data/en-trp Test.xlsx")
TRP_EN_XLSX = os.environ.get("TRP_EN_XLSX", "data/trp-en Test.xlsx")

# Official WMT 2026 English-Kokborok parallel training corpus (2,266 pairs).
# Used ONLY to build the internal validation split and few-shot prompts --
# never for fine-tuning. See the paper, Section 3.2.
TRAIN_XLSX = os.environ.get(
    "TRAIN_XLSX", "data/English-Kokborok Training Data 2026.xlsx"
)

OUT_DIR = os.environ.get("OUT_DIR", "outputs")

# --- Reproducibility ---------------------------------------------------------
# Seed for the deterministic train/dev/eval split and for qualitative sampling.
# Decoding itself is deterministic (do_sample=False), so no generation seed is
# required to reproduce the reported numbers.
SEED = int(os.environ.get("SEED", 42))

# --- Decoding (paper Section 5.3) --------------------------------------------
PRIMARY_NUM_BEAMS = 4
CONTRASTIVE_NUM_BEAMS = 1
MAX_NEW_TOKENS = 128
REPETITION_PENALTY = 1.3
BATCH_SIZE = 16
CONTRASTIVE_BATCH_SIZE = 32
MAX_INPUT_LENGTH = 512

# --- Data cleaning (paper Section 3.2) ---------------------------------------
MIN_WORDS = 3
MAX_WORDS = 200
MAX_LENGTH_RATIO = 5.0
