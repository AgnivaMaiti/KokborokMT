"""Loading the WMT 2026 test files and building the internal evaluation split.

The official test files contain a source column only (1,000 sentences per
direction) and no references, so they cannot be scored locally. All internally
reported numbers come from a deterministic 10% slice of the official *training*
corpus -- see `build_splits` and Section 3.2 of the paper.
"""

import re
import unicodedata

import pandas as pd
from sklearn.model_selection import train_test_split

import config


def normalize(text):
    """NFC-normalize and collapse runs of whitespace."""
    text = unicodedata.normalize("NFC", str(text))
    return re.sub(r"\s+", " ", text).strip()


def is_valid(en, trp, mn=None, mx=None, ratio=None):
    """Length-outlier and alignment-ratio filter (paper Section 3.2)."""
    mn = config.MIN_WORDS if mn is None else mn
    mx = config.MAX_WORDS if mx is None else mx
    ratio = config.MAX_LENGTH_RATIO if ratio is None else ratio
    ew, tw = len(en.split()), len(trp.split())
    if tw == 0:
        return False
    return mn <= ew <= mx and mn <= tw <= mx and 1 / ratio <= ew / tw <= ratio


def load_test_sentences(en_trp_xlsx=None, trp_en_xlsx=None):
    """Return (en_sentences, trp_sentences) from the official blind test files."""
    en_trp_xlsx = en_trp_xlsx or config.EN_TRP_XLSX
    trp_en_xlsx = trp_en_xlsx or config.TRP_EN_XLSX

    en_df = pd.read_excel(en_trp_xlsx, sheet_name="Final Data")
    trp_df = pd.read_excel(trp_en_xlsx, sheet_name="Final Data")

    en_sentences = en_df["English Sentences"].dropna().tolist()
    trp_sentences = trp_df["Target Sentence"].dropna().tolist()

    print(f"WMT test - EN->TRP: {len(en_sentences)} | TRP->EN: {len(trp_sentences)}")
    return en_sentences, trp_sentences


def build_splits(train_xlsx=None, seed=None):
    """Deterministically split the official parallel corpus 80/10/10.

    Returns (train_pairs, dev_pairs, eval_en, eval_trp). With the shipped corpus
    and seed=42 this yields train=1812, dev=227, eval=227.
    """
    train_xlsx = train_xlsx or config.TRAIN_XLSX
    seed = config.SEED if seed is None else seed

    train_df = pd.read_excel(train_xlsx).dropna(subset=["English", "Kokborok"])
    pairs = [
        (normalize(r["English"]), normalize(r["Kokborok"]))
        for _, r in train_df.iterrows()
    ]
    valid_pairs = [p for p in pairs if is_valid(p[0], p[1])]

    td, eval_test = train_test_split(valid_pairs, test_size=0.1, random_state=seed)
    train_pairs, dev_pairs = train_test_split(td, test_size=0.111, random_state=seed)

    eval_en = [p[0] for p in eval_test]
    eval_trp = [p[1] for p in eval_test]

    print(f"Training data: {len(valid_pairs)} valid pairs")
    print(
        f"Split: train={len(train_pairs)}, dev={len(dev_pairs)}, eval={len(eval_test)}"
    )
    return train_pairs, dev_pairs, eval_en, eval_trp
