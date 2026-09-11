"""Fast, model-free reproducibility check.

Verifies the parts of the paper that do not need a GPU or the fine-tuned model:

  * the official parallel corpus filters down to 2,266 usable pairs;
  * the deterministic split yields train=1,812 / dev=227 / eval=227
    (paper Section 3.2);
  * the instruction corpus, if present, splits 95/5 into 10,856 / 572
    (paper Section 3.1 and Table 1);
  * `postprocess()` behaves as documented in Section 4.3.

Run this first. It takes a couple of seconds and confirms that your data files
are the same ones used in the paper.

    python verify_split.py
"""

import os
import sys

import config
from data import build_splits, load_instruction_corpus, split_instruction_corpus
from postprocess import postprocess

EXPECTED_VALID_PAIRS = 2266
EXPECTED_SPLIT = (1812, 227, 227)
EXPECTED_CORPUS = 11428
EXPECTED_SFT_SPLIT = (10856, 572)

_failures = []


def check(label, got, want):
    ok = got == want
    print("%-52s %-22s %s" % (label, got, "OK" if ok else "MISMATCH (want %s)" % (want,)))
    if not ok:
        _failures.append(label)
    return ok


def main():
    print("=" * 86)
    print("Parallel corpus (paper Section 3.2)")
    print("=" * 86)

    if not os.path.exists(config.TRAIN_XLSX):
        print("SKIP: %s not found." % config.TRAIN_XLSX)
        print("      Place the official WMT 2026 training corpus there and re-run.")
        return 1

    train_pairs, dev_pairs, eval_en, eval_trp = build_splits()
    check("usable pairs after filtering", len(train_pairs) + len(dev_pairs) + len(eval_en),
          EXPECTED_VALID_PAIRS)
    check("split (train, dev, eval)",
          (len(train_pairs), len(dev_pairs), len(eval_en)), EXPECTED_SPLIT)

    print()
    print("=" * 86)
    print("Instruction corpus (paper Section 3.1, Table 1)")
    print("=" * 86)

    if os.path.exists(config.SFT_CORPUS):
        records = load_instruction_corpus()
        check("instruction pairs loaded", len(records), EXPECTED_CORPUS)
        sft_train, sft_eval = split_instruction_corpus(records)
        check("95/5 split (train, eval)", (len(sft_train), len(sft_eval)),
              EXPECTED_SFT_SPLIT)
    else:
        print("SKIP: %s not found." % config.SFT_CORPUS)
        print("      The instruction corpus is not redistributed with this repository;")
        print("      see the README section 'Corpus provenance'.")

    print()
    print("=" * 86)
    print("Post-processing (paper Section 4.3)")
    print("=" * 86)

    loop = ("Chwrai-rok nwngni simi phaiwi nini nok. "
            "Kaisa kaisa jorao bohrokni sakani. Kaisa kaisa jorao bohrokni sakani.")
    check("loop truncated to first sentence",
          postprocess(loop, "en_to_trp"),
          "Chwrai-rok nwngni simi phaiwi nini nok.")
    check("primer stripped",
          postprocess("Kokborok: Nini apha tei bwsa-no borom rudi.", "en_to_trp"),
          "Nini apha tei bwsa-no borom rudi.")
    check("trailing note removed",
          postprocess("Bini kok wngkha (Note: Kaitor means Lord)", "en_to_trp"),
          "Bini kok wngkha")
    check("empty generation flagged", postprocess("", "en_to_trp"), "<EMPTY>")

    print()
    if _failures:
        print("FAILED: %d check(s) did not match the paper: %s"
              % (len(_failures), ", ".join(_failures)))
        return 1
    print("All checks match the values reported in the paper.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
