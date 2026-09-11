"""Produce the WMT 2026 submission files.

Primary run     : beam search (num_beams=4) + post-processing
Contrastive run : greedy decoding (num_beams=1) + post-processing

Usage:
    python run_submission.py            # both runs
    python run_submission.py --primary
    python run_submission.py --contrastive
"""

import argparse
import os
import time

import config
from data import load_test_sentences
from generate import generate_translations
from metrics import copy_rate
from model import load_model
from prompts import make_en_to_trp, make_trp_to_en


def _write(path, lines):
    with open(path, "w", encoding="utf-8") as f:
        for line in lines:
            f.write(line + "\n")
    print(f"  wrote {len(lines)} lines -> {path}")


def run(tag, num_beams, batch_size, en_sentences, trp_sentences, tok, mdl):
    print(f"\n=== {tag} run (num_beams={num_beams}) ===")

    t0 = time.time()
    en_to_trp = generate_translations(
        en_sentences,
        make_en_to_trp,
        tok=tok,
        mdl=mdl,
        batch_size=batch_size,
        num_beams=num_beams,
        direction="en_to_trp",
        apply_postprocess=True,
    )
    print(f"EN->TRP done in {(time.time() - t0) / 60:.1f} min")

    t0 = time.time()
    trp_to_en = generate_translations(
        trp_sentences,
        make_trp_to_en,
        tok=tok,
        mdl=mdl,
        batch_size=batch_size,
        num_beams=num_beams,
        direction="trp_to_en",
        apply_postprocess=True,
    )
    print(f"TRP->EN done in {(time.time() - t0) / 60:.1f} min")

    os.makedirs(config.OUT_DIR, exist_ok=True)
    _write(
        os.path.join(config.OUT_DIR, f"{config.TEAM_NAME}_{tag}_en_to_trp.txt"),
        en_to_trp,
    )
    _write(
        os.path.join(config.OUT_DIR, f"{config.TEAM_NAME}_{tag}_trp_to_en.txt"),
        trp_to_en,
    )

    assert len(en_to_trp) == len(en_sentences), "EN->TRP line count mismatch"
    assert len(trp_to_en) == len(trp_sentences), "TRP->EN line count mismatch"

    cr = copy_rate(trp_sentences, trp_to_en)
    print(f"TRP->EN copy rate: {cr * 100:.0f}%")
    if cr > 0.2:
        print("WARNING: copy rate >20% -- check the make_trp_to_en prompt.")

    return en_to_trp, trp_to_en


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--primary", action="store_true")
    ap.add_argument("--contrastive", action="store_true")
    args = ap.parse_args()

    both = not (args.primary or args.contrastive)

    en_sentences, trp_sentences = load_test_sentences()
    tok, mdl = load_model()

    if args.primary or both:
        run(
            "primary",
            config.PRIMARY_NUM_BEAMS,
            config.BATCH_SIZE,
            en_sentences,
            trp_sentences,
            tok,
            mdl,
        )
    if args.contrastive or both:
        run(
            "contrastive",
            config.CONTRASTIVE_NUM_BEAMS,
            config.CONTRASTIVE_BATCH_SIZE,
            en_sentences,
            trp_sentences,
            tok,
            mdl,
        )


if __name__ == "__main__":
    main()
