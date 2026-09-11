"""Evaluation metrics (paper Section 5.1).

SacreBLEU signature: nrefs:1|case:mixed|eff:no|tok:13a|smooth:exp
chrF++ is chrF with word_order=2. TER is normalized and case-insensitive.

COMET is not computed here: it was unavailable in our evaluation environment
because of an unresolved dependency conflict. The COMET figures reported in the
paper come from the shared task organisers.
"""

import evaluate as hf_evaluate

_bleu = None
_chrf = None
_ter = None


def _load():
    global _bleu, _chrf, _ter
    if _bleu is None:
        _bleu = hf_evaluate.load("sacrebleu")
        _chrf = hf_evaluate.load("chrf")
        _ter = hf_evaluate.load("ter")
    return _bleu, _chrf, _ter


def compute_metrics(preds, refs_list):
    """`refs_list` must be [[ref], [ref], ...]. Returns (bleu, chrf++, ter)."""
    bleu, chrf, ter = _load()
    b = bleu.compute(predictions=preds, references=refs_list)
    c = chrf.compute(predictions=preds, references=refs_list, word_order=2)
    t = ter.compute(
        predictions=preds,
        references=refs_list,
        normalized=True,
        case_sensitive=False,
    )
    return b["score"], c["score"], t["score"]


def copy_rate(sources, preds, threshold=0.6, sample=50):
    """Fraction of predictions that are near-copies of their source.

    A sanity check for the TRP->EN direction, where the model is prone to
    echoing the Kokborok input instead of translating it.
    """
    n = min(sample, len(sources))
    copies = 0
    for s, p in zip(sources[:n], preds[:n]):
        s_toks = set(s.lower().split())
        p_toks = set(p.lower().split())
        if len(s_toks & p_toks) / max(len(s_toks), 1) > threshold:
            copies += 1
    return copies / n if n else 0.0
