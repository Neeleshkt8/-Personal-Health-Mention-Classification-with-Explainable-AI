"""Data: loader for real PHM datasets + a synthetic social-media generator for offline demos.

Task definition
---------------
Main task (3 classes):
  0 LITERAL_PHM    - the author (or someone they describe) literally has the condition
                     "woke up with a migraine again, can't see straight"
  1 LITERAL_OTHER  - literal mention that is not a personal health report (news, info, awareness)
                     "new study links poor sleep to migraine"
  2 FIGURATIVE     - health word used figuratively / hyperbolically
                     "this group project is giving me a migraine"
Auxiliary task (binary): does the post contain figurative language at all?
  Always 1 for FIGURATIVE, but also 1 for literal posts that use a metaphor/simile
  ("my migraine feels like a jackhammer") - so it is a related but distinct signal.
"""
from __future__ import annotations

import random
import re
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

LABELS = ["LITERAL_PHM", "LITERAL_OTHER", "FIGURATIVE"]
LABEL2ID = {l: i for i, l in enumerate(LABELS)}

# --------------------------------------------------------------------------- loader
_ALIASES = {  # map common dataset label names onto the 3-way scheme
    "phm": 0, "personal": 0, "personal health mention": 0, "self": 0, "literal_phm": 0,
    "non-phm": 1, "non_personal": 1, "non-personal": 1, "other": 1, "literal_other": 1, "mention": 1,
    "figurative": 2, "figurative_health_mention": 2, "hyperbolic": 2, "fig": 2,
}

_FIG_MARKERS = re.compile(r"\b(?:like a|like an|as if|as though|feels like|literally dying|killing me)\b", re.I)


def load_csv(path: str | Path, text_col="text", label_col="label", fig_col: str | None = "figurative") -> pd.DataFrame:
    """Load a real dataset (e.g. a Reddit/Twitter health-mention corpus) from CSV.
    `label` may be int ids 0-2 or names (see _ALIASES). If no figurative column exists,
    the auxiliary label is derived: FIGURATIVE class OR an explicit simile/metaphor marker."""
    df = pd.read_csv(path)
    out = pd.DataFrame({"text": df[text_col].astype(str)})
    lab = df[label_col]
    if pd.api.types.is_numeric_dtype(lab):
        out["label"] = lab.astype(int)
    else:
        out["label"] = lab.astype(str).str.strip().str.lower().map(lambda s: LABEL2ID.get(s.upper(), _ALIASES.get(s)))
    bad = out["label"].isna() | ~out["label"].isin([0, 1, 2])
    if bad.any():
        raise ValueError(f"{bad.sum()} rows have unmappable labels, e.g. {df.loc[bad, label_col].unique()[:5]}")
    out["label"] = out["label"].astype(int)
    if fig_col and fig_col in df.columns:
        out["figurative"] = df[fig_col].astype(int)
    else:
        out["figurative"] = ((out["label"] == 2) | out["text"].str.contains(_FIG_MARKERS)).astype(int)
    return out


# --------------------------------------------------------------------------- synthetic
CONDITIONS = {
    # condition: (literal personal templates-friendly noun, figurative templates)
    "migraine": ["{x} is giving me a migraine", "{x} = instant migraine", "reading {x} gave me a migraine"],
    "headache": ["{x} is such a headache", "dealing with {x} is a total headache", "{x} is a headache and a half"],
    "heart attack": ["{x} almost gave me a heart attack", "nearly had a heart attack when i saw {x}",
                     "{x} is going to give me a heart attack"],
    "fever": ["{x} fever is real this week", "{x} was a total fever dream", "got a bad case of {x} fever"],
    "cancer": ["{x} is pure cancer", "the {x} comment section is cancer", "{x} is a cancer on this app"],
    "allergy": ["i have a serious allergy to {x}", "my allergy to {x} is acting up again lol"],
    "depression": ["{x} has me in a deep depression lmao", "post-{x} depression hitting hard"],
    "stroke": ["{x} was a stroke of genius", "{x} must be having a stroke", "pure stroke of luck with {x}"],
    "cough": ["*cough* {x} *cough* overrated", "{x} is *cough* not that great"],
    "anxiety": ["{x} is giving me anxiety lol", "the anxiety of watching {x} live"],
    "flu": ["{x} fever spreading like the flu", "{x} hype is catching like the flu"],
    "asthma": ["climbing {x} stairs gave me asthma lol", "{x} took my breath away like an asthma attack"],
}
THINGS = ["this group project", "monday traffic", "my landlord", "the new update", "this exam", "the finale",
          "my inbox", "that plot twist", "the season finale", "this spreadsheet", "the stock market",
          "the ref", "my team's defense", "this playlist", "tax season", "the wifi", "black friday",
          "that last episode", "the group chat", "this meeting"]
TIMES = ["since yesterday", "all week", "for three days now", "again today", "since last night",
         "this morning", "on and off for a month", "since monday"]
METAPHORS = ["a jackhammer in my skull", "an elephant sitting on my chest", "a truck hit me",
             "my head is in a vise", "i'm on fire", "breathing through a straw", "a dark cloud on me"]
PEOPLE = ["my mom", "my dad", "my brother", "my sister", "my husband", "my wife", "my son", "my roommate"]
PLACES = ["the uk", "texas", "india", "rural areas", "the us", "europe", "canada"]

LIT_PHM = [
    "i've had a {c} {t}", "woke up with a {c} {t}", "doctor confirmed i have {c}", "my {c} is so bad {t}",
    "can't sleep because of my {c}", "stuck in bed with {c} {t}", "finally seeing a doctor about my {c}",
    "{p} was diagnosed with {c} {t}", "{p} has had {c} {t}, worried", "my {c} meds aren't working",
    "nearly had a {c} last year and spent a week in hospital", "had to leave work early, {c} kicked in",
    "anyone else get {c} after the vaccine? had it {t}", "third {c} episode {t}, calling my gp",
    # hard: figurative-looking context, literal meaning
    "got a {c} during {x} and had to go home", "{x} was cancelled for me, {c} {t}",
    "missed {x} because of my {c}", "my {c} started right in the middle of {x}, meds are not helping",
    "{x} is giving me a {c}, seriously, been to urgent care twice",
]
# hard negatives for PHM: first person, but not a personal health report
LIT_OTHER_HARD = [
    "my {c} test came back negative, phew", "i'm writing an essay on {c} for class",
    "i donated to {c} research today", "my doctor said it's definitely not {c}",
    "i used to think {c} was rare, it isn't", "i'm running a 5k for {c} awareness during {x}",
]
# hard positives for FIGURATIVE: personal framing, still figurative
FIG_HARD = [
    "i swear i get a {c} every time i open {x}", "my wallet had a {c} after {x}",
    "i'm having a full {c} over {x}", "i caught {x} {c} and there is no cure lol",
]
LIT_OTHER = [
    "new study links poor sleep to {c}", "{c} rates are rising in {pl}", "is {c} contagious?",
    "5 early warning signs of {c} you shouldn't ignore", "this month is {c} awareness month",
    "researchers found a new treatment for {c}", "how is {c} diagnosed? asking for a school project",
    "{c} costs {pl} billions every year", "free {c} screening at the community center saturday",
    "what's the difference between {c} and a normal headache?", "the {c} ward at the hospital is expanding",
]
NOISE_PRE = ["", "", "", "ugh ", "lol ", "honestly ", "ok so ", "@user ", "update: "]
NOISE_POST = ["", "", "", " 😩", " 😂", " #mondays", " smh", " http://url", " 🙃", " fml", " !!"]


def _corrupt(s: str, rng: random.Random) -> str:
    """Real posts are messy: truncated mid-thought, typos, dropped words.
    Truncation can delete the disambiguating cue, creating genuinely ambiguous posts."""
    w = s.split()
    if rng.random() < 0.20 and len(w) > 5:  # truncated post / character-limit preview
        w = w[: max(3, int(len(w) * rng.uniform(0.45, 0.75)))]
    if rng.random() < 0.25:  # drop a random word
        w.pop(rng.randrange(len(w)))
    out = []
    for tok in w:
        if len(tok) > 3 and rng.random() < 0.06:  # typo: swap two adjacent characters
            i = rng.randrange(len(tok) - 1)
            tok = tok[:i] + tok[i + 1] + tok[i] + tok[i + 2:]
        out.append(tok)
    return " ".join(out)


def make_synthetic(n: int = 3000, seed: int = 42, label_noise: float = 0.03) -> pd.DataFrame:
    rng = random.Random(seed)
    rows = []
    conds = list(CONDITIONS)
    weights = [0.38, 0.27, 0.35]
    for _ in range(n):
        y = rng.choices([0, 1, 2], weights)[0]
        c = rng.choice(conds)
        fig = 0
        if y == 0:
            s = rng.choice(LIT_PHM).format(c=c, t=rng.choice(TIMES), p=rng.choice(PEOPLE), x=rng.choice(THINGS))
            if rng.random() < 0.25:  # literal mention that also uses figurative language
                s += rng.choice([", feels like ", ", it's like ", " - like "]) + rng.choice(METAPHORS)
                fig = 1
        elif y == 1:
            tpl = rng.choice(LIT_OTHER_HARD if rng.random() < 0.3 else LIT_OTHER)
            s = tpl.format(c=c, pl=rng.choice(PLACES), x=rng.choice(THINGS))
        else:
            tpl = rng.choice(FIG_HARD) if rng.random() < 0.3 else rng.choice(CONDITIONS[c])
            s = tpl.format(x=rng.choice(THINGS), c=c)
            fig = 1
        s = _corrupt(s, rng)
        s = rng.choice(NOISE_PRE) + s + rng.choice(NOISE_POST)
        if rng.random() < 0.3:
            s = s[0].upper() + s[1:]
        rows.append((s, y, fig))
    df = pd.DataFrame(rows, columns=["text", "label", "figurative"])
    df["noisy"] = 0  # ground truth for the label-noise detector (synthetic data only)
    # annotation noise, like real crowd-sourced labels
    flip = df.sample(frac=label_noise, random_state=seed).index
    df.loc[flip, "label"] = [rng.choice([l for l in (0, 1, 2) if l != y]) for y in df.loc[flip, "label"]]
    df.loc[flip, "figurative"] = (df.loc[flip, "label"] == 2).astype(int) | df.loc[flip, "figurative"]
    df.loc[flip, "noisy"] = 1
    return df.drop_duplicates("text").reset_index(drop=True)


def split(df: pd.DataFrame, seed: int = 42, val=0.15, test=0.15):
    train, rest = train_test_split(df, test_size=val + test, stratify=df["label"], random_state=seed)
    v, t = train_test_split(rest, test_size=test / (val + test), stratify=rest["label"], random_state=seed)
    return train.reset_index(drop=True), v.reset_index(drop=True), t.reset_index(drop=True)
