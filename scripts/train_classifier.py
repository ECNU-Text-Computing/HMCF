#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import RobertaTokenizer, get_cosine_schedule_with_warmup
from torch.optim import AdamW

from src.data import (
    GoldenJsonDataset,
    DISAPERE_LABEL2ID, DISAPERE_ID2LABEL,
    PRAGTAG_LABEL2ID, PRAGTAG_ID2LABEL,
    collate_batch
)
from src.model import RobertaClassifier
from src.train_loop import train_one_epoch, evaluate
from src.metrics import save_confusion_matrix, save_metrics_json
from src.utils import set_seed, get_device

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--label-set", choices=["disapere", "pragtag"], required=True)
    ap.add_argument("--train-json", required=True)
    ap.add_argument("--valid-json", required=True)
    ap.add_argument("--hf-model", required=True)
    ap.add_argument("--epochs", type=int, default=9)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--weight-decay", type=float, default=0.001)
    ap.add_argument("--max-len", type=int, default=256)
    ap.add_argument("--device", default=None)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()

    set_seed(args.seed)
    device = get_device(args.device)

    tokenizer = RobertaTokenizer.from_pretrained(args.hf_model)

    if args.label_set == "disapere":
        label2id, id2label = DISAPERE_LABEL2ID, DISAPERE_ID2LABEL
    else:
        label2id, id2label = PRAGTAG_LABEL2ID, PRAGTAG_ID2LABEL

    train_ds = GoldenJsonDataset(args.train_json, label2id)
    valid_ds = GoldenJsonDataset(args.valid_json, label2id)

    train_dl = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        collate_fn=lambda b: collate_batch(b, tokenizer, args.max_len, device)
    )
    valid_dl = DataLoader(
        valid_ds, batch_size=args.batch_size, shuffle=False,
        collate_fn=lambda b: collate_batch(b, tokenizer, args.max_len, device)
    )

    model = RobertaClassifier(args.hf_model, num_classes=len(label2id)).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=len(train_dl),
        num_training_steps=args.epochs * len(train_dl)
    )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    best_macro = -1.0
    best_path = out_dir / "best.pt"

    for epoch in range(1, args.epochs + 1):
        print(f"===== Epoch {epoch}/{args.epochs} =====")
        train_one_epoch(model, train_dl, criterion, optimizer, scheduler, device)
        scores, y_true, y_pred = evaluate(model, valid_dl, criterion)
        print(f"[valid] loss={scores['loss']:.4f} micro_f1={scores['micro_f1']:.4f} macro_f1={scores['macro_f1']:.4f} weighted_f1={scores['weighted_f1']:.4f}")

        if scores["macro_f1"] > best_macro:
            best_macro = scores["macro_f1"]
            torch.save(model.state_dict(), best_path)
            save_confusion_matrix(y_true, y_pred, id2label, str(out_dir / "confusion_matrix.png"), title=f"{args.label_set} Valid Confusion Matrix")
            save_metrics_json(scores, str(out_dir / "metrics.json"))
            print(f"[OK] saved best to: {best_path}")

    print(f"[DONE] best macro_f1={best_macro:.4f} at {best_path}")

if __name__ == "__main__":
    main()
