"""QLoRA fine-tuning driver -- RECONSTRUCTED, not the original script.

=============================================================================
READ THIS BEFORE USING OR CITING THIS FILE
=============================================================================

The script that actually produced `agnivamaiti/KokLLaMA-3.2-3B-Instruct` was
not preserved. This file is a reconstruction, written after the fact from two
sources that *were* preserved:

  * the hyperparameters recorded on the model card and in the adapter's
    `adapter_config.json`, reproduced in Table 1 of the paper;
  * the corpus and 95/5 split documented in Section 3.1.

It therefore reproduces the training *configuration* faithfully, and it is the
best available description of how the released adapter was produced. It is NOT
a bit-for-bit replay of the original run. Two things in particular are unknown
and are not reconstructed here, because they were never logged:

  * the warmup schedule;
  * the hardware and wall-clock time.

Anything this script does that is not pinned by Table 1 -- warmup ratio,
logging cadence, checkpointing, scheduler shape -- is a reasonable default
chosen here, not a recovered fact. Re-running it will produce an adapter
trained the same way, but not numerically identical weights.

If you want the exact released adapter, download it rather than retraining:

    agnivamaiti/KokLLaMA-3.2-3B-Instruct

=============================================================================

The instruction corpus is not redistributed with this repository. See the
README section "Corpus provenance" for what the public release does and does
not contain.

Usage:
    python train.py
    python train.py --dry-run     # build everything, print the plan, train nothing
"""

import argparse
import os
import sys

import config
from data import format_sft_example, load_instruction_corpus, split_instruction_corpus

# Defaults chosen here, NOT recovered from the original run.
WARMUP_RATIO = 0.03
LR_SCHEDULER = "cosine"
LOGGING_STEPS = 25
SAVE_STRATEGY = "epoch"


def build_datasets():
    records = load_instruction_corpus()
    train_ds, eval_ds = split_instruction_corpus(records)
    train_ds = train_ds.map(lambda r: {"text": format_sft_example(r)})
    eval_ds = eval_ds.map(lambda r: {"text": format_sft_example(r)})
    return train_ds, eval_ds


def print_plan(train_ds, eval_ds):
    eff = config.PER_DEVICE_TRAIN_BATCH_SIZE * config.GRAD_ACCUM_STEPS
    steps = (len(train_ds) // eff) * config.NUM_EPOCHS
    print()
    print("=" * 72)
    print("Training plan (paper Table 1)")
    print("=" * 72)
    rows = [
        ("base model", config.BASE_ID),
        ("train / eval pairs", "%d / %d" % (len(train_ds), len(eval_ds))),
        ("LoRA r / alpha / dropout",
         "%d / %d / %s" % (config.LORA_R, config.LORA_ALPHA, config.LORA_DROPOUT)),
        ("target modules", ", ".join(config.LORA_TARGET_MODULES)),
        ("quantization", "NF4, double quant, bf16 compute"),
        ("epochs", config.NUM_EPOCHS),
        ("batch / grad-accum / effective",
         "%d / %d / %d" % (config.PER_DEVICE_TRAIN_BATCH_SIZE,
                           config.GRAD_ACCUM_STEPS, eff)),
        ("learning rate", config.LEARNING_RATE),
        ("optimizer", config.OPTIM),
        ("max sequence length", config.MAX_SEQ_LENGTH),
        ("approx. optimizer steps", steps),
        ("warmup ratio", "%s   (NOT recovered -- default)" % WARMUP_RATIO),
        ("lr scheduler", "%s   (NOT recovered -- default)" % LR_SCHEDULER),
    ]
    for k, v in rows:
        print("  %-32s %s" % (k, v))
    print("=" * 72)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true",
                    help="build the datasets and print the plan without training")
    args = ap.parse_args()

    if not os.path.exists(config.SFT_CORPUS):
        print("ERROR: instruction corpus not found at %s" % config.SFT_CORPUS)
        print("       It is not redistributed here; see the README.")
        return 1

    train_ds, eval_ds = build_datasets()
    print_plan(train_ds, eval_ds)

    if args.dry_run:
        print("\n--dry-run: stopping before training.")
        return 0

    import torch
    from peft import LoraConfig
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from trl import SFTConfig, SFTTrainer

    from model import bnb_config

    tokenizer = AutoTokenizer.from_pretrained(config.BASE_ID)
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        config.BASE_ID,
        quantization_config=bnb_config(),
        device_map="auto",
        torch_dtype=torch.bfloat16,
    )
    model.config.use_cache = False

    peft_config = LoraConfig(
        r=config.LORA_R,
        lora_alpha=config.LORA_ALPHA,
        lora_dropout=config.LORA_DROPOUT,
        target_modules=config.LORA_TARGET_MODULES,
        bias="none",
        task_type="CAUSAL_LM",
    )

    sft_config = SFTConfig(
        output_dir=config.ADAPTER_OUT_DIR,
        num_train_epochs=config.NUM_EPOCHS,
        per_device_train_batch_size=config.PER_DEVICE_TRAIN_BATCH_SIZE,
        gradient_accumulation_steps=config.GRAD_ACCUM_STEPS,
        learning_rate=config.LEARNING_RATE,
        optim=config.OPTIM,
        max_seq_length=config.MAX_SEQ_LENGTH,
        bf16=True,
        seed=config.SEED,
        # --- defaults chosen here, not recovered ---
        warmup_ratio=WARMUP_RATIO,
        lr_scheduler_type=LR_SCHEDULER,
        logging_steps=LOGGING_STEPS,
        save_strategy=SAVE_STRATEGY,
        dataset_text_field="text",
        report_to="none",
    )

    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        peft_config=peft_config,
        processing_class=tokenizer,
    )

    trainer.train()
    trainer.save_model(config.ADAPTER_OUT_DIR)
    tokenizer.save_pretrained(config.ADAPTER_OUT_DIR)
    print("\nAdapter written to %s" % config.ADAPTER_OUT_DIR)
    print("NOTE: this is a re-training, not the released adapter.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
