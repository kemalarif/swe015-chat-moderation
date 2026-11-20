"""Regex normalization of noisy Turkish chat text (Area B).

Two levels:
  normalize_light(text) -> for BERTurk (keeps case, Turkish letters and punctuation)
  normalize_full(text)  -> for lexicon / tf-idf / word2vec (lower-case, ASCII-folded, tokenizable)

Handled noise:
  * user mentions / URLs              "@USER", "https://..."    -> "@user", "url"
  * repeated letters                  "siktirrrr", "çoooook"    -> "siktir", "çok"
  * spaced-out letters                "B O K", "L Ü T F E N"   -> "bok", "lütfen"
  * leetspeak inside words            "s1kt1r", "@pt@l", "g3r1"  -> "siktir", "aptal", "geri"
  * Turkish casing                    "IŞIK", "İSTANBUL"        -> "ışık", "istanbul"
  * Turkish-character folding (full)  "şerefsiz", "gerızekalı"  -> "serefsiz", "gerizekali"
"""
import re

_URL = re.compile(r"(https?://\S+|www\.\S+|\bURL\b)")
_MENTION = re.compile(r"(?<!\S)@\w+(?=[\s.,!?:;)]|$)")
_REPEAT = re.compile(r"(\w)\1{2,}")                       # 3+ same chars -> 1
_SPACED = re.compile(r"\b(?:\w ){2,}\w\b")                # b o k  /  L Ü T F E N
_LEET_TOKEN = re.compile(r"\S*[A-Za-zÇĞİÖŞÜçğıöşü]\S*")  # tokens that contain letters
_LEET_MAP = str.maketrans({"0": "o", "1": "i", "3": "e", "4": "a", "5": "s",
                           "7": "t", "8": "b", "@": "a", "$": "s", "€": "e"})
_TR_FOLD = str.maketrans({"ç": "c", "ğ": "g", "ı": "i", "ö": "o", "ş": "s", "ü": "u",
                          "â": "a", "î": "i", "û": "u"})
_NON_WORD = re.compile(r"[^\w\s]")
_WS = re.compile(r"\s+")


def tr_lower(text: str) -> str:
    """Turkish-aware lower-casing (I -> ı, İ -> i)."""
    return text.replace("I", "ı").replace("İ", "i").lower()


def _join_spaced(m: re.Match) -> str:
    return m.group(0).replace(" ", "")


def _deleet(m: re.Match) -> str:
    tok = m.group(0)
    if tok.startswith("@") and tok[1:].isalpha():           # @user mention
        return tok
    letters = sum(c.isalpha() for c in tok)
    leet = sum(c in "0134578@$€" for c in tok)
    # only rewrite mixed tokens like s1kt1r / @pt@l, not "2014" or "5'te"
    if leet and letters >= 2 and not tok[0].isdigit():
        return tok.translate(_LEET_MAP)
    return tok


def normalize_light(text: str) -> str:
    text = _URL.sub("url", text)
    text = _MENTION.sub("@user", text)
    text = _SPACED.sub(_join_spaced, text)
    text = _LEET_TOKEN.sub(_deleet, text)
    text = _REPEAT.sub(r"\1", text)
    return _WS.sub(" ", text).strip()


def normalize_full(text: str) -> str:
    text = normalize_light(text)
    text = tr_lower(text).translate(_TR_FOLD)
    text = _NON_WORD.sub(" ", text)
    return _WS.sub(" ", text).strip()


def tokenize(text: str) -> list[str]:
    """Whitespace tokenization of fully normalized text."""
    return normalize_full(text).split()


if __name__ == "__main__":
    for s in ["@USER Sen tam bir GERİZEKALISIN!!! https://t.co/x",
              "s1kt1r git b u r a d a n",
              "çoooook güzelll olmuşşş 👏",
              "L Ü T F E N  B O K",
              "2014'te 5 gol attı @pt@l"]:
        print(f"{s!r}\n  light: {normalize_light(s)!r}\n  full : {normalize_full(s)!r}")
