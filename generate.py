"""Batched translation generation (paper Section 5.3).

Decoding is fully deterministic: sampling is disabled, so temperature and top-p
are inert and no generation seed is needed.

Note that `no_repeat_ngram_size` is deliberately NOT set here, matching the
submitted system. In hindsight this is the most likely proximate cause of the
repetition loops discussed in the paper: `repetition_penalty` rescales logits
but does not hard-block repeated n-grams.
"""

import torch
from tqdm.auto import tqdm

import config
from postprocess import EMPTY, postprocess


def generate_translations(
    sentences,
    prompt_fn,
    tok,
    mdl,
    batch_size=None,
    max_new_tokens=None,
    num_beams=None,
    direction="en_to_trp",
    apply_postprocess=False,
):
    """Translate `sentences` with `prompt_fn`, returning a list of strings."""
    batch_size = config.BATCH_SIZE if batch_size is None else batch_size
    max_new_tokens = config.MAX_NEW_TOKENS if max_new_tokens is None else max_new_tokens
    num_beams = config.PRIMARY_NUM_BEAMS if num_beams is None else num_beams

    translations = []
    prompts = [prompt_fn(s) for s in sentences]

    for i in tqdm(range(0, len(prompts), batch_size), desc="Translating"):
        batch = prompts[i : i + batch_size]
        inputs = tok(
            batch,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=config.MAX_INPUT_LENGTH,
            add_special_tokens=True,
        ).to(mdl.device)

        with torch.no_grad():
            output_ids = mdl.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                num_beams=num_beams,
                do_sample=False,
                temperature=1.0,
                top_p=1.0,
                pad_token_id=tok.eos_token_id,
                eos_token_id=tok.eos_token_id,
                early_stopping=True,
                repetition_penalty=config.REPETITION_PENALTY,
            )

        input_length = inputs["input_ids"].shape[1]
        for out in output_ids:
            new_tokens = out[input_length:]
            if len(new_tokens) == 0:
                translations.append(EMPTY)
                continue
            decoded = tok.decode(new_tokens, skip_special_tokens=True).strip()
            # First-newline cut -- applied before post-processing.
            decoded = decoded.split("\n")[0].strip()
            if apply_postprocess:
                decoded = postprocess(decoded, direction)
            translations.append(decoded if decoded else EMPTY)

    return translations
