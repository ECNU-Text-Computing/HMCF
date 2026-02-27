import json
from typing import Dict, Optional

def build_recheck_index(jsonl_path: str, text_key: str = "text", label_key: str = "Final-Recheck") -> Dict[str, str]:
    idx: Dict[str, str] = {}
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            t = obj.get(text_key, "")
            lab = obj.get(label_key, None)
            if t and lab is not None:
                idx[t] = lab
    return idx

def get_final_recheck(index: Dict[str, str], target_text: str) -> Optional[str]:
    # exact match (recommended)
    return index.get(target_text)
