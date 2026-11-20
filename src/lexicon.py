"""Rule baseline: edit-distance lexicon matcher (Area B).

A message is OFFENSIVE if any of its normalized tokens matches an entry of a
hand-written lexicon of Turkish offensive roots.

Matching rules (all strings are ASCII-folded, see normalize.normalize_full):
  * short entries (<= 3 chars, e.g. "amk", "pic")  -> exact token match only
  * 4-5 char entries (e.g. "salak", "aptal")         -> exact prefix match (allows Turkish suffixes:
                                                         "salaklar", "aptalsin")
  * >= 6 char entries (e.g. "orospu", "serefsiz")   -> minimum edit distance between the entry and
                                                         the token prefix <= 1 (<= 2 for >= 9 chars),
                                                         first letter must agree  ("orspu", "serefsz")
"""
from functools import lru_cache

from .normalize import normalize_full

# Hand-written lexicon of offensive roots (ASCII-folded, lower-case).
SHORT_EXACT = {
    "amk", "aq", "amq", "mk", "sg", "oc", "pic", "got", "bok", "sik", "it", "ipne", "ibne",
    "pust", "yarak", "amcik", "mal",
}
ROOTS = [
    # profanity
    "siktir", "sikerim", "sikeyim", "siktigim", "sikik", "sikim", "hassiktir", "hasiktir",
    "orospu", "orosbu", "kahpe", "yavsak", "pezevenk", "yarrak", "yarram", "dalyarak", "amina",
    "aminakoyim", "anani", "ananin amina", "bacini", "ebeni", "gotveren", "gotunu", "gotune",
    "gotlek", "gavat", "kaltak", "fahise", "surtuk", "kevase", "kerhane", "sokayim", "sokarim",
    "piclik", "ibneler", "godos", "puseet",
    # insults
    "serefsiz", "haysiyetsiz", "namussuz", "gerizekali", "gerzek", "salak", "aptal", "ahmak",
    "dangalak", "hiyar", "angut", "ezik", "pislik", "kancik", "embesil", "beyinsiz", "mankafa",
    "denyo", "hoduk", "lavuk", "zibidi", "dallama", "keriz", "soysuz", "alcak", "terbiyesiz",
    "edepsiz", "ahlaksiz", "rezil", "yobaz", "sapik", "malsin", "salaksin", "aptalsin",
    "it oglu", "kopek oglu", "puseet",
]
# multi-word entries are matched on the joined bigram
ROOTS = sorted(set(r.replace(" ", "") for r in ROOTS))


def min_edit_distance(a: str, b: str) -> int:
    """Levenshtein distance (insert / delete / substitute cost 1), dynamic programming."""
    if a == b:
        return 0
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i] + [0] * len(b)
        for j, cb in enumerate(b, 1):
            cur[j] = min(prev[j] + 1,             # deletion
                         cur[j - 1] + 1,          # insertion
                         prev[j - 1] + (ca != cb))  # substitution
        prev = cur
    return prev[-1]


def _max_dist(entry: str) -> int:
    if len(entry) >= 9:
        return 2
    if len(entry) >= 6:
        return 1
    return 0


@lru_cache(maxsize=200_000)
def match_token(tok: str):
    """Return (entry, distance) of the best lexicon match for a token, or None."""
    if tok in SHORT_EXACT:
        return tok, 0
    best = None
    for e in ROOTS:
        k = _max_dist(e)
        if k == 0:
            if tok.startswith(e):
                return e, 0
            continue
        if tok[0] != e[0] or len(tok) < len(e) - k:
            continue
        # compare entry with token prefixes of length len(e)-k .. len(e)+k (suffix tolerant)
        d = min(min_edit_distance(tok[:L], e) for L in range(len(e) - k, len(e) + k + 1))
        if d <= k and (best is None or d < best[1]):
            best = (e, d)
            if d == 0:
                break
    return best


def find_matches(text: str):
    """All lexicon hits in a message: list of (token, entry, distance)."""
    toks = normalize_full(text).split()
    cands = toks + [a + b for a, b in zip(toks, toks[1:])]   # unigrams + joined bigrams
    hits = []
    for t in cands:
        m = match_token(t)
        if m:
            hits.append((t, m[0], m[1]))
    return hits


class LexiconClassifier:
    name = "Lexicon + edit distance (rules)"

    def predict_proba_one(self, text: str) -> float:
        hits = find_matches(text)
        if not hits:
            return 0.0
        d = min(h[2] for h in hits)
        return 1.0 if d == 0 else 0.9 if d == 1 else 0.8

    def predict_proba(self, texts):
        return [self.predict_proba_one(t) for t in texts]


if __name__ == "__main__":
    for s in ["sen tam bir ger1zekalısın", "orspu çocuğu", "bugün çok sıkıldım",
              "malzemeleri götürdüm", "Ş E R E F S İ Z", "kalsın böyle", "aptallar"]:
        print(f"{s!r:35s} -> {find_matches(s)}")
