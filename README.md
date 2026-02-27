# HMCF


1) **Label reconciliation** (Human label + LLM-check + small model + optional final recheck)  
2) **Train/evaluate** a RoBERTa sentence classifier on the reconciled dataset

## Installation

```bash
pip install -r requirements.txt
```

## Quick Start

### A) Reconcile labels 
#### DISAPERE

```bash
python scripts/reconcile_labels.py \
  --dataset disapere \
  --raw-dir /path/to/DISAPERE/train \
  --llm-check-dir /path/to/DISAPERE-Check/QwenTrain \
  --final-recheck /path/to/trainAnnotation2.json \
  --small-model /path/to/GoldenLabel_Best.pth \
  --hf-model /path/to/RoBERTa \
  --out-json /path/to/train_golden.json \
  --out-changes /path/to/changes.csv \
  --device cuda:0
```

#### PragTag2023

```bash
python scripts/reconcile_labels.py \
  --dataset pragtag \
  --raw-json /path/to/PragTag2023/train_inputs_full.json \
  --llm-check-dir /path/to/PragTag2023_QwenCheck \
  --final-recheck /path/to/PragTag2023Annotation2.jsonl \
  --small-model /path/to/PragTag2023_Goldenlabel.pth \
  --hf-model /path/to/RoBERTa \
  --out-json /path/to/pragtag_train_golden.json \
  --out-changes /path/to/pragtag_changes.csv \
  --device cuda:0
```


### B) Train + evaluate classifier

```bash
python scripts/train_classifier.py \
  --label-set disapere \
  --train-json /path/to/train_golden.json \
  --valid-json /path/to/valid_golden.json \
  --hf-model /path/to/RoBERTa \
  --epochs 8 \
  --batch-size 16 \
  --lr 1e-5 \
  --device cuda:0 \
  --out-dir runs/exp1
```

For PragTag:
```bash
python scripts/train_classifier.py \
  --label-set pragtag \
  --train-json /path/to/pragtag_train_golden.json \
  --valid-json /path/to/pragtag_valid_golden.json \
  --hf-model /path/to/RoBERTa \
  --epochs 8 \
  --batch-size 16 \
  --lr 1e-5 \
  --device cuda:0 \
  --out-dir runs/pragtag_exp1
```


