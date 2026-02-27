import json
import re
from typing import Optional

LABEL_ALIASES_DISAPERE = {
    "arg_structuring": "arg_structuring",
    "arg_evaluative": "arg_evaluative",
    "arg_request": "arg_request",
    "arg_social": "arg_social",
    "arg_fact": "arg_fact",
    "none": "none",
    "arg_other": "arg_other",
}

LABEL_ALIASES_PRAGTAG = {
    "Strength": "Strength",
    "Weakness": "Weakness",
    "Todo": "Todo",
    "Structure": "Structure",
    "Recap": "Recap",
    "Other": "Other",
    # tolerate lowercase / variants
    "strength": "Strength",
    "weakness": "Weakness",
    "todo": "Todo",
    "structure": "Structure",
    "recap": "Recap",
    "other": "Other",
}

_CORRECT_RE = re.compile(r"^\s*correct\s*$", re.IGNORECASE)
_LABEL_RE = re.compile(r"Correct\s*Label\s*:\s*([A-Za-z_]+)", re.IGNORECASE)

def parse_llm_check_item(text: str, label_aliases: dict[str,str]) -> Optional[str]:
    """Return None if LLM says correct; else normalized label name."""
    if text is None:
        return None
    s = str(text).strip()
    if _CORRECT_RE.match(s):
        return None

    m = _LABEL_RE.search(s)
    cand = None
    if m:
        cand = m.group(1).strip()
    else:
        # fallback: try keyword containment
        for k in label_aliases.keys():
            if k in s:
                cand = k
                break
    if cand is None:
        return None
    return label_aliases.get(cand, cand)

def load_llm_check_list(path: str) -> list[str]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)
