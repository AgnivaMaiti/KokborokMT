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

This repository contains the **inference and evaluation** pipeline only.

The QLoRA fine-tuning script is **not** included — it was not preserved from the
original training run. The training hyperparameters are documented below and on
the model card, and `adapter_config.json` in the model repository records the
LoRA configuration exactly, but the training driver itself cannot be republished
faithfully and we would rather omit it than reconstruct something that was not
what actually ran.

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
| Submission files (primary + contrastive) | Yes | `run_submission.py` |
| Table 2 — KokLLaMA vs. zero-shot baseline | Yes | `run_evaluation.py` |
| Table 5 — post-processing ablation | Yes | `run_evaluation.py` |
| Table 4 — qualitative examples | Yes | `run_evaluation.py --qualitative` |
| Table 1 — training hyperparameters | Documented, not runnable | training script not preserved |
| Table 3 — conversational QA eval (n=50) | **No** | evaluation set not preserved |

The two gaps are honest ones. The QLoRA fine-tuning driver was lost, and the
50-pair conversational evaluation set used for Table 3 was defined in a notebook
cell that was deleted before the notebook was saved, so neither can be
republished faithfully. `make_en_to_trp_few_shot()` in `prompts.py` is the
few-shot prompt that experiment used; it is included for reference but nothing
in this repository calls it.

## Layout

| File | Purpose |
|---|---|
| `config.py` | Paths, model IDs, seed, decoding parameters |
| `data.py` | Loads the test files; builds the deterministic 80/10/10 split |
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
python run_submission.py                 # writes the four submission files to outputs/
python run_evaluation.py                 # Tables 2 and 5
python run_evaluation.py --qualitative   # also prints the Table 4 examples
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
the training script itself is not in this repository):

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

**Corpus version.** The model was fine-tuned on 10,856 instruction pairs; the
publicly released `agnivamaiti/kokborok-qa` contains 4,943 deduplicated pairs.
Re-running with only the released split may not reproduce the reported numbers
exactly.

**Evaluation caveat.** The internal split is a 10% slice of the official
*training* corpus and is predominantly biblical in domain, whereas the official
test set is news. Internal figures substantially under-estimate official
performance (0.37 vs. 5.11 BLEU for EN→TRP).

**Not included.** The conversational-QA evaluation reported in the paper (n=50)
used an evaluation set that was not preserved in the source notebook, so that
experiment is not reproducible from this repository.

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
