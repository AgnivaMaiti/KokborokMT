# KokborokMT

Inference and evaluation code for **KokLLaMA-3.2-3B-Instruct**, the SCE-KIIT
submission to the WMT 2026 Shared Task on Low-Resource Indic Language
Translation (Category 2: English ↔ Kokborok).

> **KokLLaMA: Cross-Task Adaptation for Low-Resource English–Kokborok Translation**
> Agniva Maiti, Aarsh Muley, Sovan Kumar Sahoo
> School of Computer Science and Engineering, KIIT Deemed to be University

- Model: https://huggingface.co/agnivamaiti/KokLLaMA-3.2-3B-Instruct
- Corpus: https://huggingface.co/datasets/agnivamaiti/kokborok-qa

## Scope of this repository

This repository contains the full **inference and evaluation** pipeline, plus a
**reconstructed** fine-tuning driver.

The script that actually produced the released adapter was not preserved.
`train.py` is a reconstruction, written from the hyperparameters recorded on the
model card and in `adapter_config.json` (Table 1 of the paper) and from the
corpus split documented in Section 3.1. It reproduces the training
*configuration* faithfully; it is not a bit-for-bit replay, and the warmup
schedule and hardware were never logged. `train.py` says so at the top, and
prints which of its settings are recovered facts and which are defaults chosen
after the fact. If you want the exact adapter, download it rather than
retraining it.

## Official WMT 2026 results

| Direction | Run | BLEU | METEOR | chrF++ | TER | COMET |
|---|---|---|---|---|---|---|
| EN→TRP | Primary (beam 4) | **5.11** | 21.04 | 27.09 | 98.67 | 54.35 |
| EN→TRP | Contrastive (greedy) | 1.98 | 11.94 | 20.32 | 108.65 | 49.89 |
| TRP→EN | Primary (beam 4) | **3.58** | 23.59 | 26.14 | 200.86 | 53.49 |
| TRP→EN | Contrastive (greedy) | 1.45 | 16.72 | 22.13 | 137.37 | 53.31 |

Primary runs placed 3rd of 4 (EN→TRP) and 3rd of 5 (TRP→EN) submitted primary
systems.

## What this code reproduces

Every generation call here matches the submitted notebook parameter-for-parameter
(verified against the original notebook: batch size, beams, `max_new_tokens`,
`repetition_penalty`, post-processing flag, per call site). `postprocess()` is
verified to reproduce the notebook's own test cases byte-for-byte.

| Paper item | Reproducible here? | Entry point |
|---|---|---|
| Data splits (2,266 → 1,812/227/227; 11,428 → 10,856/572) | Yes, in seconds, no GPU | `verify_split.py` |
| Submission files (primary + contrastive) | Yes | `run_submission.py` |
| Table 3 — KokLLaMA vs. zero-shot baseline | Yes | `run_evaluation.py` |
| Table 4 — qualitative examples | Yes | `run_evaluation.py --qualitative` |
| Table 5 — post-processing ablation | Yes | `run_evaluation.py` |
| Table 1 — training hyperparameters | Configuration only, see above | `train.py` |
| Table 6 — conversational QA eval (n=50) | **No** | evaluation set not preserved |

Run `verify_split.py` first. It needs no GPU and no model download, and it
confirms that your copies of the data files are the same ones the paper used.

One gap is unrecoverable. The 50-pair conversational evaluation set behind
Table 6 was defined in a notebook cell that was deleted before the notebook was
saved, and no copy survives. That table is reported in the paper as a
development observation rather than a reproducible result.
`make_en_to_trp_few_shot()` in `prompts.py` is the few-shot prompt that
experiment used; it is included for reference but nothing here calls it.

## Layout

| File | Purpose |
|---|---|
| `config.py` | Paths, model IDs, seed, decoding and training parameters |
| `data.py` | Loads the test files and the instruction corpus; builds both splits |
| `verify_split.py` | Fast, model-free check that the data matches the paper |
| `train.py` | Reconstructed QLoRA driver — read its header first |
| `prompts.py` | Translation prompts (KokLLaMA, baseline, few-shot) |
| `postprocess.py` | Rule-based output cleanup |
| `model.py` | 4-bit NF4 model loading |
| `generate.py` | Batched deterministic generation |
| `metrics.py` | SacreBLEU, chrF++, TER |
| `run_submission.py` | Produces the primary and contrastive submission files |
| `run_evaluation.py` | Internal evaluation, ablation and baseline tables |

## Usage

```bash
pip install -r requirements.txt
```

Place the shared task files under `data/`:

```
data/en-trp Test.xlsx
data/trp-en Test.xlsx
data/English-Kokborok Training Data 2026.xlsx
```

Then:

```bash
python verify_split.py                   # first: confirms your data matches the paper
python run_submission.py                 # writes the four submission files to outputs/
python run_evaluation.py                 # Tables 3 and 5
python run_evaluation.py --qualitative   # also prints the Table 4 examples
python train.py --dry-run                # prints the training plan, trains nothing
```

`run_evaluation.py` additionally writes `outputs/eval_outputs.json` containing
the raw and post-processed generations for both directions, so the ablation can
be re-scored without re-running the model.

A GPU with roughly 8 GB of VRAM is sufficient for 4-bit inference.

## Reproducibility notes

**Seeds.** `config.SEED = 42` controls the train/dev/eval split
(`sklearn.train_test_split(random_state=42)`) and qualitative sampling. Decoding
is deterministic (`do_sample=False`), so temperature and top-p are inert and no
generation seed is required.

**Decoding.** Primary: `num_beams=4`, `max_new_tokens=128`,
`repetition_penalty=1.3`, `early_stopping=True`, batch 16, inputs truncated to
512 tokens, left padding. Contrastive: identical but `num_beams=1`, batch 32.
`no_repeat_ngram_size` is deliberately **not** set, matching the submitted
system — see `generate.py` for why this matters.

**Training hyperparameters** (from the model card and `adapter_config.json`;
implemented in `train.py`, which is a reconstruction — see "Scope" above):

| | |
|---|---|
| Base model | `meta-llama/Llama-3.2-3B-Instruct` |
| LoRA rank / α / dropout | 64 / 128 / 0.05 |
| Target modules | `q,k,v,o,gate,up,down_proj` |
| Quantization | NF4, double quant, bf16 compute |
| Epochs | 3 |
| Batch size | 8 per device, grad-accum 2 (effective 16) |
| Learning rate | 2e-4 |
| Optimizer | `paged_adamw_32bit` |
| Max sequence length | 1024 |
| Train / eval split | 95 / 5 |

The warmup schedule, hardware configuration and wall-clock training time were
not logged during the original run and are therefore not reported.

**Corpus provenance.** This matters, and the numbers are easy to confuse.

| | pairs |
|---|---|
| Instruction corpus used for fine-tuning | 11,428 |
| → training split (95%) | 10,856 |
| → held-out split (5%) | 572 |
| Public release `agnivamaiti/kokborok-qa` | 4,943 rows / **2,514 distinct** |

The public release is an **earlier and smaller snapshot**, not a deduplicated
copy and not a strict subset: it contains roughly 49% duplicate rows, and 81.7%
of its distinct pairs also occur in the 11,428-pair corpus. Fine-tuning on the
release alone will **not** reproduce the reported model. The 11,428-pair corpus
is not redistributed here; `config.SFT_CORPUS` points at where `train.py` and
`verify_split.py` expect to find it.

The `10,856` figure that appears in the paper is the *training split*, not the
corpus size — `verify_split.py` checks this derivation explicitly.

**Evaluation caveat.** The internal split is a 10% slice of the official
*training* corpus and is predominantly biblical in domain, whereas the official
test set is news. Internal figures substantially under-estimate official
performance (0.37 vs. 5.11 BLEU for EN→TRP).

**Not included.** The conversational-QA evaluation reported in the paper (n=50,
Table 6) used an evaluation set that was not preserved in the source notebook,
so that experiment is not reproducible from this repository.

## Citation

```bibtex
@inproceedings{maiti-etal-2026-kokllama,
  title     = {{SCE-KIIT}: {KokLLaMA}: Cross-Task Adaptation for Low-Resource
               {E}nglish--{K}okborok Translation},
  author    = {Maiti, Agniva and Muley, Aarsh and Sahoo, Sovan Kumar},
  booktitle = {Proceedings of the Eleventh Conference on Machine Translation (WMT)},
  year      = {2026}
}
```

## License

Code released under the MIT License. The fine-tuned adapter inherits the
Llama 3.2 Community License; the instruction corpus is CC BY-NC 4.0.
