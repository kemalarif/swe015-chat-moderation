"""Post-processing: mask a flagged message before other players see it."""
from .lexicon import find_matches

HIDDEN = "🔒 *mesaj moderasyon tarafından gizlendi*"


def mask(text: str, star: str = "\\*") -> str:
    """Mask the words that match the offensive lexicon (s****r). If the model flagged the message but
    no word matches the lexicon (implicit insult), hide the whole message."""
    out, hit = [], False
    for tok in text.split(" "):
        if tok and find_matches(tok):
            hit = True
            out.append(tok[0] + star * (len(tok) - 1))
        else:
            out.append(tok)
    return " ".join(out) if hit else HIDDEN
