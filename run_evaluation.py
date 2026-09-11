"""Internal evaluation on the held-out split (n=227).

Reproduces the paper's Table 2 (KokLLaMA vs. zero-shot baseline) and Table 5
(post-processing ablation, Appendix A).

IMPORTANT: this split is a 10% slice of the official *training* corpus, whose
content is predominantly biblical. The official WMT 2026 test set is news-domain
and is distributed without references. As reported in the paper, these internal
figures substantially under-estimate official test performance (0.37 vs. 5.11
BLEU for EN->TRP). Treat them as in-domain validation only.

Usage:
    python run_evaluation.py
"""

import gc

import torch

import config
from data import build_splits
from generate import generate_translations
from metrics import compute_metrics, copy_rate
from model import load_model
from postprocess import postprocess
from prompts import (
    make_en_to_trp,
    make_en_to_trp_base,
    make_trp_to_en,
    make_trp_to_en_base,
)


def _row(name, direction, scores):
    bleu, chrf, ter = scores
    print(f"{name:<40} {direction:<8} {bleu:>6.2f} {chrf:>7.2f} {ter:>7.2f}")


def main():
    _, _, eval_en, eval_trp = build_splits()
    refs_trp = [[r] for r in eval_trp]
    refs_en = [[r] for r in eval_en]

    tok, mdl = load_model(config.MODEL_ID)

    # --- Row A: raw output, no post-processing ------------------------------
    print(f"\nGenerating eval outputs (no post-proc), n={len(eval_en)} ...")
    et_raw = generate_translations(
        eval_en, make_en_to_trp, tok=tok, mdl=mdl,
        direction="en_to_trp", apply_postprocess=False,
    )
    te_raw = generate_translations(
        eval_trp, make_trp_to_en, tok=tok, mdl=mdl,
        direction="trp_to_en", apply_postprocess=False,
    )
    a_et = compute_metrics(et_raw, refs_trp)
    a_te = compute_metrics(te_raw, refs_en)

    print(f"Copy rate - EN->TRP: {copy_rate(eval_en, et_raw) * 100:.0f}%  "
          f"| TRP->EN: {copy_rate(eval_trp, te_raw) * 100:.0f}%")

    # --- Row B: same outputs, post-processed --------------------------------
    et_clean = [postprocess(t, "en_to_trp") for t in et_raw]
    te_clean = [postprocess(t, "trp_to_en") for t in te_raw]
    b_et = compute_metrics(et_clean, refs_trp)
    b_te = compute_metrics(te_clean, refs_en)

    # --- Zero-shot baseline --------------------------------------------------
    del mdl
    torch.cuda.empty_cache()
    gc.collect()

    base_tok, base_mdl = load_model(config.BASE_ID)
    base_et = generate_translations(
        eval_en, make_en_to_trp_base, tok=base_tok, mdl=base_mdl,
        direction="en_to_trp", apply_postprocess=True,
    )
    base_te = generate_translations(
        eval_trp, make_trp_to_en_base, tok=base_tok, mdl=base_mdl,
        direction="trp_to_en", apply_postprocess=True,
    )
    z_et = compute_metrics(base_et, refs_trp)
    z_te = compute_metrics(base_te, refs_en)

    # --- Tables --------------------------------------------------------------
    bar = "=" * 72
    print(f"\n{bar}\nTABLE 2: KokLLaMA vs. Zero-Shot Baseline (n={len(eval_en)})\n{bar}")
    print(f"{'System':<40} {'Dir':<8} {'BLEU':>6} {'chrF++':>7} {'TER':>7}")
    print("-" * 72)
    _row("Llama-3.2-3B-Instruct (zero-shot)", "EN->TRP", z_et)
    _row("Llama-3.2-3B-Instruct (zero-shot)", "TRP->EN", z_te)
    _row("KokLLaMA (prompt + post-proc) [P]", "EN->TRP", b_et)
    _row("KokLLaMA (prompt + post-proc) [P]", "TRP->EN", b_te)
    print(bar)
    print("[P] = primary submission configuration")

    print(f"\n{bar}\nTABLE 5: Ablation - effect of post-processing\n{bar}")
    print(f"{'System / Config':<40} {'Dir':<8} {'BLEU':>6} {'chrF++':>7} {'TER':>7}")
    print("-" * 72)
    _row("KokLLaMA (no post-proc)", "EN->TRP", a_et)
    _row("KokLLaMA (no post-proc)", "TRP->EN", a_te)
    _row("KokLLaMA (with post-proc)", "EN->TRP", b_et)
    _row("KokLLaMA (with post-proc)", "TRP->EN", b_te)
    print(bar)
    print("Note: COMET omitted -- see metrics.py.")


if __name__ == "__main__":
    main()
