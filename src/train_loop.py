from typing import Dict, Tuple, List
import torch
from torch.utils.data import DataLoader
from transformers import get_cosine_schedule_with_warmup
from sklearn.metrics import f1_score

from .metrics import compute_scores

def train_one_epoch(model, dataloader: DataLoader, criterion, optimizer, scheduler, device: torch.device, log_interval: int = 50):
    model.train()
    all_pred, all_true = [], []
    for step, (input_ids, attn, y) in enumerate(dataloader):
        logits = model(input_ids, attn)
        loss = criterion(logits, y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

        pred = torch.argmax(logits.detach(), dim=-1).cpu().tolist()
        true = y.detach().cpu().tolist()
        all_pred.extend(pred)
        all_true.extend(true)

        if log_interval and step % log_interval == 0 and step > 0:
            scores = compute_scores(all_true, all_pred)
            print(f"[train] step {step}/{len(dataloader)} loss={loss.item():.4f} micro_f1={scores['micro_f1']:.4f} macro_f1={scores['macro_f1']:.4f}")

def evaluate(model, dataloader: DataLoader, criterion):
    model.eval()
    all_pred, all_true = [], []
    total_loss = 0.0
    with torch.no_grad():
        for (input_ids, attn, y) in dataloader:
            logits = model(input_ids, attn)
            loss = criterion(logits, y)
            total_loss += float(loss.item())
            pred = torch.argmax(logits, dim=-1).cpu().tolist()
            true = y.cpu().tolist()
            all_pred.extend(pred)
            all_true.extend(true)
    scores = compute_scores(all_true, all_pred)
    scores["loss"] = total_loss / max(1, len(dataloader))
    return scores, all_true, all_pred
