"""Social-media text normalisation shared by all models."""
import re

_URL = re.compile(r"https?://\S+|www\.\S+")
_USER = re.compile(r"@\w+")
_HASHTAG = re.compile(r"#(\w+)")
_REPEAT = re.compile(r"(.)\1{3,}")


def clean(text: str) -> str:
    t = text.lower()
    t = _URL.sub(" URL ", t)
    t = _USER.sub(" @USER ", t)
    t = _HASHTAG.sub(r" \1 ", t)       # keep hashtag word as a token
    t = _REPEAT.sub(r"\1\1\1", t)      # sooooo -> sooo
    return re.sub(r"\s+", " ", t).strip()


def words(text: str) -> list[str]:
    return text.split()
