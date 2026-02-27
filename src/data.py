import json
import os
from dataclasses import dataclass
from typing import List, Dict, Iterator, Tuple

import torch
from torch.utils.data import Dataset
from transformers import RobertaTokenizer

# ---------------------------
# Label sets
# ---------------------------
DISAPERE_LABELS = ['arg_structuring','arg_evaluative','arg_request','arg_social','arg_fact','none','arg_other']
DISAPERE_LABEL2ID = {k:i for i,k in enumerate(DISAPERE_LABELS)}
DISAPERE_ID2LABEL = {i:k for k,i in DISAPERE_LABEL2ID.items()}

PRAGTAG_LABELS = ["Strength", "Weakness", "Todo", "Structure", "Recap", "Other"]
PRAGTAG_LABEL2ID = {k:i for i,k in enumerate(PRAGTAG_LABELS)}
PRAGTAG_ID2LABEL = {i:k for k,i in PRAGTAG_LABEL2ID.items()}

# ---------------------------
# Examples
# ---------------------------
@dataclass
class Example:
    text: str
    label_id: int

class GoldenJsonDataset(Dataset):
    """Dataset built from reconciled JSON: list[{text,label}]"""
    def __init__(self, json_path: str, label2id: Dict[str,int]):
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        self.items: List[Example] = []
        for obj in data:
            t = obj["text"]
            lab = obj["label"]
            self.items.append(Example(t, int(label2id[lab])))

    def __len__(self): return len(self.items)
    def __getitem__(self, idx):
        ex = self.items[idx]
        return ex.text, ex.label_id

# ---------------------------
# Raw readers for reconciliation
# ---------------------------
class DisapereRawReader:
    """Iterate DISAPERE raw directory and yield (file, sent_idx, text, orig_label_name)."""
    def __init__(self, raw_dir: str):
        self.files = [f for f in os.listdir(raw_dir) if f.endswith(".json")]
        self.raw_dir = raw_dir

    def iter_sentences(self) -> Iterator[Tuple[str,int,str,str]]:
        for fn in self.files:
            path = os.path.join(self.raw_dir, fn)
            with open(path, "r", encoding="utf-8") as f:
                obj = json.load(f)
            sents = obj.get("review_sentences", [])
            for i, x in enumerate(sents):
                text = x.get("text", "")
                y_orig = x.get("review_action", "none")
                yield fn, i, text, y_orig

class PragTagRawReader:
    """Iterate PragTag2023 raw json list and yield (id, sent_idx, text, orig_label_name)."""
    def __init__(self, raw_json: str):
        with open(raw_json, "r", encoding="utf-8") as f:
            self.data = json.load(f)

    def iter_sentences(self) -> Iterator[Tuple[str,int,str,str]]:
        for item in self.data:
            item_id = str(item.get("id"))
            sentences = item.get("sentences", [])
            labels = item.get("labels", [])
            n = min(len(sentences), len(labels))
            for i in range(n):
                yield item_id, i, sentences[i], labels[i]

# ---------------------------
# Collate
# ---------------------------
def collate_batch(batch, tokenizer: RobertaTokenizer, max_len: int, device: torch.device):
    texts, labels = zip(*batch)
    enc = tokenizer(list(texts), padding="max_length", truncation=True, max_length=max_len, return_tensors="pt")
    input_ids = enc["input_ids"].to(device=device, dtype=torch.long)
    attention_mask = enc["attention_mask"].to(device=device, dtype=torch.long)
    y = torch.tensor(labels, dtype=torch.long, device=device)
    return input_ids, attention_mask, y
