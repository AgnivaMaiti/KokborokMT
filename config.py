"""Paths, model identifiers and shared constants.

Override any of these with environment variables of the same name, e.g.

    EN_TRP_XLSX=/data/en-trp.xlsx python run_submission.py
"""

import os

# --- Team / submission -------------------------------------------------------
# Affects output filenames only. The source notebook emitted "SCE-KIIT-NLP";
# the files were renamed to "SCE-KIIT" before submission, which is the name the
# organisers scored and published. We default to the submitted name.
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

# Kokborok conversational instruction corpus used for SFT (11,428 pairs).
# NOT redistributed here -- see the README, "Corpus provenance". The public
# release `agnivamaiti/kokborok-qa` is an earlier, smaller snapshot (4,943 rows,
# 2,514 distinct pairs) and will NOT reproduce the reported numbers.
SFT_CORPUS = os.environ.get("SFT_CORPUS", "data/kokborok_train_corpus_11428.jsonl")

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

# --- QLoRA fine-tuning (paper Table 1) ---------------------------------------
# Reproduced from the model card and adapter_config.json. See train.py for the
# important caveat about what this configuration is and is not.
LORA_R = 64
LORA_ALPHA = 128
LORA_DROPOUT = 0.05
LORA_TARGET_MODULES = [
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
]
NUM_EPOCHS = 3
PER_DEVICE_TRAIN_BATCH_SIZE = 8
GRAD_ACCUM_STEPS = 2          # effective batch 16
LEARNING_RATE = 2e-4
OPTIM = "paged_adamw_32bit"
MAX_SEQ_LENGTH = 1024
SFT_EVAL_FRACTION = 0.05      # 95/5 train/eval split -> 10,856 / 572
ADAPTER_OUT_DIR = os.environ.get("ADAPTER_OUT_DIR", "outputs/kokllama-adapter")
