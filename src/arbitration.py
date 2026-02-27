from typing import Optional, Tuple

def decide_label(
    y_orig: str,
    y_llm: Optional[str],
    y_small: str,
    y_final: Optional[str] = None,
) -> Tuple[str, str]:
    """Return (final_label, reason)."""
    if y_llm is None or y_llm == y_orig:
        return y_orig, "llm_correct_or_agree"

    # conflict
    if y_small == y_orig:
        return y_orig, "small_support_orig"
    if y_small == y_llm:
        return y_llm, "small_support_llm"

    if y_final is not None:
        return y_final, "final_recheck"

    return y_llm, "fallback_llm"
