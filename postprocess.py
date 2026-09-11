"""Rule-based output cleanup (paper Section 4.3).

Note: there is deliberately NO explicit repetition or duplicate-phrase detector
here. Loop truncation is an emergent consequence of step 6 (keep only the first
sentence) combined with the first-newline cut applied in `generate.py`: because
degenerate repetitions almost always begin after the first complete sentence,
retaining only that sentence removes them.

This also explains the two known failure modes:
  * inline parenthetical glosses -- e.g. "(face)", "(dust)" -- survive, because
    they occur mid-sentence rather than as trailing note blocks;
  * legitimate multi-sentence translations are silently truncated.
"""

import re

EMPTY = "<EMPTY>"

_EXPLANATORY_PREFIXES = (
    r"^(Aboni ortho wngkha|Kokborok wngkha|English translation wngkha|"
    r"English meaning|Translation|Answer)\s*:\s*"
)

_QUOTE_CHARS = "'‘’\"“”"


def postprocess(text, direction="en_to_trp"):
    """Clean residual model artefacts. `direction` is 'en_to_trp' or 'trp_to_en'."""
    # 1. Strip the completion primer if the model echoes it back.
    if direction == "en_to_trp":
        text = re.sub(r"^Kokborok\s*:", "", text, flags=re.IGNORECASE).strip()
    else:
        text = re.sub(
            r"^In English,\s*(this means\s*)?:?", "", text, flags=re.IGNORECASE
        ).strip()

    # 2. Strip explanatory prefixes (an artefact of the conversational QA SFT).
    text = re.sub(_EXPLANATORY_PREFIXES, "", text, flags=re.IGNORECASE).strip()

    # 3. If the model wrapped the output in quotes, take the first quoted span.
    qm = re.search(rf"[{_QUOTE_CHARS}](.*?)[{_QUOTE_CHARS}]", text)
    if qm and len(qm.group(1)) > 3:
        text = qm.group(1).strip()

    # 4. Delete trailing "(Note: ...)" blocks and "(ABC = ...)" expansions.
    text = re.sub(r"\(Note:.*$", "", text, flags=re.IGNORECASE).strip()
    text = re.sub(r"\([A-Z][^)]*=[^)]*\)", "", text).strip()

    # 5. Strip residual leading/trailing quotation characters.
    text = text.strip(_QUOTE_CHARS).strip()

    # 6. Keep only the first sentence (this is what truncates repetition loops).
    parts = re.split(r"(?<=[.?!])\s+(?=[A-Z])", text)
    text = parts[0].strip() if parts else text

    return text if text.strip() else EMPTY
