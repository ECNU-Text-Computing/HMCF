import json
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, f1_score

def compute_scores(y_true: List[int], y_pred: List[int]) -> Dict[str, float]:
    macro_f1 = f1_score(y_true, y_pred, average="macro")
    micro_f1 = f1_score(y_true, y_pred, average="micro")
    weighted_f1 = f1_score(y_true, y_pred, average="weighted")
    return {"macro_f1": float(macro_f1), "micro_f1": float(micro_f1), "weighted_f1": float(weighted_f1)}

def save_confusion_matrix(
    y_true: List[int],
    y_pred: List[int],
    id2label: Dict[int,str],
    out_png: str,
    title: str = "Confusion Matrix"
) -> None:
    labels = [id2label[i] for i in sorted(id2label.keys())]
    cm = confusion_matrix(
        [id2label[i] for i in y_true],
        [id2label[i] for i in y_pred],
        labels=labels
    )
    df = pd.DataFrame(cm, index=labels, columns=labels)
    plt.figure(figsize=(10, 8))
    plt.imshow(df.values)
    plt.xticks(range(len(labels)), labels, rotation=45, ha="right")
    plt.yticks(range(len(labels)), labels)
    plt.title(title)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    # annotate
    for i in range(len(labels)):
        for j in range(len(labels)):
            plt.text(j, i, str(df.values[i, j]), ha="center", va="center")
    plt.tight_layout()
    Path(out_png).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close()

def save_metrics_json(metrics: Dict[str,float], out_path: str) -> None:
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
