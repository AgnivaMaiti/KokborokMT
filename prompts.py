"""Prompt templates (paper Section 4.2).

Both translation prompts terminate in a *completion primer* rather than an
instruction ("Kokborok:" / "In English, this means:"). This is the strongest
mechanism we found for suppressing source copying without retraining, since it
forces the model to continue in the target language rather than echo the input.

The baseline variants are identical except that they gloss Kokborok explicitly,
because the unmodified base model has no reliable association for the bare
language name.
"""


# --- KokLLaMA (fine-tuned) ---------------------------------------------------

def make_en_to_trp(sentence):
    """English -> Kokborok, primed so the model continues in Kokborok."""
    return (
        f"User: Translate this English sentence into Kokborok (Roman script).\n"
        f"Rules: Output ONLY the Kokborok translation. "
        f"No English words. No explanations. No notes. No parentheses. No quotes.\n"
        f"English: {sentence}\n"
        f"Assistant: Kokborok:"
    )


def make_trp_to_en(sentence):
    """Kokborok -> English, primed with an anti-copy completion cue."""
    return (
        f"User: The sentence below is written in Kokborok, a Tibeto-Burman language "
        f"spoken in Tripura, India. What does it mean in English?\n"
        f"Rules: Write only a fluent English sentence. "
        f"Do not copy Kokborok words. No explanations. No notes.\n"
        f"Kokborok: {sentence}\n"
        f"Assistant: In English, this means:"
    )


# --- Zero-shot baseline (unmodified Llama-3.2-3B-Instruct) -------------------

def make_en_to_trp_base(sentence):
    return (
        f"User: Translate this English sentence into Kokborok "
        f"(a Tibeto-Burman language of Tripura, India, written in Roman script).\n"
        f"Rules: Output ONLY the Kokborok translation. No English. No explanations.\n"
        f"English: {sentence}\n"
        f"Assistant: Kokborok:"
    )


def make_trp_to_en_base(sentence):
    return (
        f"User: The sentence below is in Kokborok, a Tibeto-Burman language "
        f"spoken in Tripura, India. What does it mean in English?\n"
        f"Rules: Write only a fluent English sentence. No Kokborok words. "
        f"No explanations.\n"
        f"Kokborok: {sentence}\n"
        f"Assistant: In English, this means:"
    )


# --- Few-shot variant --------------------------------------------------------

def make_en_to_trp_few_shot(sentence, examples):
    """Prime output style with parallel examples drawn from the training split.

    `examples` is a list of (english, kokborok) tuples. In the reported
    experiments this is `train_pairs[:3]`.
    """
    prompt = (
        "User: You are an expert English to Kokborok translator. "
        "Translate the English sentences into Kokborok (Roman script).\n\n"
    )
    for en, trp in examples:
        prompt += f"English: {en}\nKokborok: {trp}\n\n"
    prompt += f"English: {sentence}\nAssistant: Kokborok:"
    return prompt
