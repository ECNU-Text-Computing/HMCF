#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
import os
from pathlib import Path

import torch
from transformers import RobertaTokenizer

from src.data import (
    DisapereRawReader, PragTagRawReader,
    DISAPERE_LABELS, PRAGTAG_LABELS
)
from src.llm_check import (
    load_llm_check_list, parse_llm_check_item,
    LABEL_ALIASES_DISAPERE, LABEL_ALIASES_PRAGTAG
)
from src.recheck import build_recheck_index, get_final_recheck
from src.arbitration import decide_label
from src.change_table import build_change_table
from src.utils import set_seed, get_device

def load_small_model(path: str, device: torch.device):
    m = torch.load(path, map_location=device, weights_only=False)
    m.eval()
    m.to(device)
    return m

def predict_label(text: str, tokenizer: RobertaTokenizer, model, id2label: list[str], max_len: int, device: torch.device) -> str:
    enc = tokenizer(text, padding="max_length", truncation=True, max_length=max_len, return_tensors="pt")
    input_ids = enc["input_ids"].to(device=device, dtype=torch.long)
    attn = enc["attention_mask"].to(device=device, dtype=torch.long)
    with torch.no_grad():
        logits = model(input_ids, attn)
        pred = int(torch.argmax(logits, dim=-1).item())
    return id2label[pred]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["disapere", "pragtag"], required=True)

    # DISAPERE inputs
    ap.add_argument("--raw-dir", default=None, help="DISAPERE raw json directory")

    # PragTag inputs
    ap.add_argument("--raw-json", default=None, help="PragTag raw json list file")

    ap.add_argument("--llm-check-dir", required=True, help="LLM-check directory")
    ap.add_argument("--final-recheck", required=False, default=None, help="JSONL file for final recheck")
    ap.add_argument("--small-model", required=True, help="Path to small model (.pth/.pt)")
    ap.add_argument("--hf-model", required=True, help="Path or name of HF tokenizer/model used by small model")
    ap.add_argument("--max-len", type=int, default=256)
    ap.add_argument("--out-json", required=True)
    ap.add_argument("--out-changes", required=True)
    ap.add_argument("--device", default=None)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    set_seed(args.seed)
    device = get_device(args.device)

    tokenizer = RobertaTokenizer.from_pretrained(args.hf_model)
    small_model = load_small_model(args.small_model, device=device)

    recheck_index = None
    if args.final_recheck:
        recheck_index = build_recheck_index(args.final_recheck)

    if args.dataset == "disapere":
        if not args.raw_dir:
            raise SystemExit("--raw-dir is required for disapere")
        reader = DisapereRawReader(args.raw_dir)
        id2label = DISAPERE_LABELS
        aliases = LABEL_ALIASES_DISAPERE
        # llm-check files match raw filenames
        def llm_path(key): return os.path.join(args.llm_check_dir, key)
    else:
        if not args.raw_json:
            raise SystemExit("--raw-json is required for pragtag")
        reader = PragTagRawReader(args.raw_json)
        id2label = PRAGTAG_LABELS
        aliases = LABEL_ALIASES_PRAGTAG
        # llm-check files are {id}.json
        def llm_path(key): return os.path.join(args.llm_check_dir, f"{key}.json")

    golden = []
    orig_labels, final_labels = [], []

    cache_llm = {}

    for key, idx, text, y_orig in reader.iter_sentences():
        # load llm list lazily per key
        if key not in cache_llm:
            cache_llm[key] = load_llm_check_list(llm_path(key))
        llm_list = cache_llm[key]
        llm_item = llm_list[idx] if idx < len(llm_list) else None
        y_llm = parse_llm_check_item(llm_item, aliases)

        y_small = predict_label(text, tokenizer, small_model, id2label, args.max_len, device)

        y_final = None
        if recheck_index is not None:
            y_final = get_final_recheck(recheck_index, text)

        y_out, reason = decide_label(y_orig=y_orig, y_llm=y_llm, y_small=y_small, y_final=y_final)

        golden.append({"text": text, "label": y_out, "meta": {"key": key, "sent_idx": idx, "reason": reason}})
        orig_labels.append(y_orig)
        final_labels.append(y_out)

    Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump(golden, f, ensure_ascii=False, indent=2)

    ct = build_change_table(orig_labels, final_labels)
    Path(args.out_changes).parent.mkdir(parents=True, exist_ok=True)
    ct.to_csv(args.out_changes, index=False, encoding="utf-8")

    print(f"[OK] golden json: {args.out_json}")
    print(f"[OK] changes csv: {args.out_changes}")
    print(ct.head(20).to_string(index=False))

if __name__ == "__main__":
    main()
