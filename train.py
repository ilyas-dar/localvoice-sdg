"""
LocalVoice-SDG — XLM-RoBERTa Fine-tuning
------------------------------------------
Multi-label SDG classification (17 classes) on full_dataset.csv.
Run this on Google Colab with a T4 GPU (~15 minutes).

Colab setup (run these in a cell first):
  !pip install transformers torch scikit-learn pandas numpy -q
  # upload full_dataset.csv via Files panel on the left
"""

import pandas as pd
import numpy as np
import json
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    get_linear_schedule_with_warmup,
)
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
import os

# ─── CONFIG ───────────────────────────────────────────────────────────────────
MODEL_NAME  = "xlm-roberta-base"
NUM_LABELS  = 17
MAX_LENGTH  = 256
BATCH_SIZE  = 16
EPOCHS      = 10
LR          = 2e-5
OUTPUT_DIR  = "./localvoice_sdg_model"
CSV_PATH    = "/content/full_dataset.csv"
# ──────────────────────────────────────────────────────────────────────────────

SDG_NAMES = {
    1:"No Poverty", 2:"Zero Hunger", 3:"Good Health", 4:"Education",
    5:"Gender Equality", 6:"Clean Water", 7:"Clean Energy", 8:"Decent Work",
    9:"Infrastructure", 10:"Inequalities", 11:"Sustainable Cities",
    12:"Consumption", 13:"Climate Action", 14:"Life Below Water",
    15:"Life on Land", 16:"Peace & Justice", 17:"Partnerships",
}


def parse_labels(val):
    if pd.isna(val) or str(val).strip() in ("SKIP", "", "nan"):
        return []
    try:
        labels = json.loads(str(val).replace("'", '"'))
        return [int(l) for l in labels if 1 <= int(l) <= 17]
    except Exception:
        return []


def to_vector(labels):
    vec = np.zeros(NUM_LABELS, dtype=np.float32)
    for l in labels:
        vec[l - 1] = 1.0
    return vec


def load_data(path):
    df = pd.read_csv(path)
    df["parsed"] = df["sdg_labels"].apply(parse_labels)
    df = df[df["parsed"].apply(len) > 0].reset_index(drop=True)
    df["vec"] = df["parsed"].apply(to_vector)
    print(f"  loaded {len(df)} labeled samples")
    return df["text"].tolist(), np.stack(df["vec"].values)


class SDGDataset(Dataset):
    def __init__(self, texts, labels, tokenizer):
        self.texts    = texts
        self.labels   = labels
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        enc = self.tokenizer(
            self.texts[idx],
            max_length=MAX_LENGTH,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        return {
            "input_ids":      enc["input_ids"].squeeze(),
            "attention_mask": enc["attention_mask"].squeeze(),
            "labels":         torch.tensor(self.labels[idx], dtype=torch.float),
        }


def evaluate(model, loader, device):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch in loader:
            out = model(
                input_ids=batch["input_ids"].to(device),
                attention_mask=batch["attention_mask"].to(device),
            )
            preds = (torch.sigmoid(out.logits) > 0.3).cpu().numpy()
            all_preds.append(preds)
            all_labels.append(batch["labels"].numpy())
    return np.vstack(all_preds), np.vstack(all_labels)


def compute_pos_weight(labels_matrix):
    pos = labels_matrix.sum(axis=0)
    neg = len(labels_matrix) - pos
    weight = np.where(pos > 0, neg / np.maximum(pos, 1), 1.0)
    return torch.tensor(weight, dtype=torch.float)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n{'='*55}")
    print(f"  LocalVoice-SDG Training")
    print(f"{'='*55}")
    print(f"  device : {device}")
    print(f"  model  : {MODEL_NAME}")
    print(f"  epochs : {EPOCHS}  |  batch : {BATCH_SIZE}  |  lr : {LR}")

    texts, labels = load_data(CSV_PATH)

    X_tr, X_tmp, y_tr, y_tmp = train_test_split(texts, labels, test_size=0.2, random_state=42)
    X_val, X_te, y_val, y_te = train_test_split(X_tmp, y_tmp, test_size=0.5, random_state=42)
    print(f"  split  : train={len(X_tr)}  val={len(X_val)}  test={len(X_te)}")

    tokenizer  = AutoTokenizer.from_pretrained(MODEL_NAME)
    train_dl   = DataLoader(SDGDataset(X_tr,  y_tr,  tokenizer), batch_size=BATCH_SIZE, shuffle=True)
    val_dl     = DataLoader(SDGDataset(X_val, y_val, tokenizer), batch_size=BATCH_SIZE)
    test_dl    = DataLoader(SDGDataset(X_te,  y_te,  tokenizer), batch_size=BATCH_SIZE)

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, num_labels=NUM_LABELS, problem_type="multi_label_classification"
    ).to(device)

    # weighted loss: rare SDGs get higher penalty when missed
    pos_weight = compute_pos_weight(y_tr).to(device)
    loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
    total_steps = len(train_dl) * EPOCHS
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=total_steps // 10, num_training_steps=total_steps
    )

    best_f1 = 0.0
    print()

    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss = 0
        for batch in train_dl:
            logits = model(
                input_ids=batch["input_ids"].to(device),
                attention_mask=batch["attention_mask"].to(device),
            ).logits
            loss = loss_fn(logits, batch["labels"].to(device))
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            total_loss += loss.item()

        preds, true = evaluate(model, val_dl, device)
        val_f1 = f1_score(true, preds, average="macro", zero_division=0)
        avg_loss = total_loss / len(train_dl)
        marker = ""
        if val_f1 > best_f1:
            best_f1 = val_f1
            os.makedirs(OUTPUT_DIR, exist_ok=True)
            model.save_pretrained(OUTPUT_DIR)
            tokenizer.save_pretrained(OUTPUT_DIR)
            marker = "  ← saved"
        print(f"  epoch {epoch}/{EPOCHS}  loss={avg_loss:.4f}  val_macro_f1={val_f1:.4f}{marker}")

    # ── test evaluation ────────────────────────────────────────────────────────
    print(f"\n  loading best model for test evaluation...")
    model = AutoModelForSequenceClassification.from_pretrained(OUTPUT_DIR).to(device)
    preds, true = evaluate(model, test_dl, device)

    macro_f1 = f1_score(true, preds, average="macro", zero_division=0)
    micro_f1 = f1_score(true, preds, average="micro", zero_division=0)

    print(f"\n{'='*55}")
    print(f"  TEST RESULTS")
    print(f"{'='*55}")
    print(f"  Macro F1 : {macro_f1:.4f}")
    print(f"  Micro F1 : {micro_f1:.4f}")
    print(f"\n  Per-SDG breakdown:")
    for i in range(NUM_LABELS):
        sdg = i + 1
        support = int(true[:, i].sum())
        if support == 0:
            continue
        f1 = f1_score(true[:, i], preds[:, i], zero_division=0)
        bar = "█" * int(f1 * 20)
        print(f"    SDG {sdg:>2} {SDG_NAMES[sdg]:<22} F1={f1:.3f}  {bar}  (n={support})")

    print(f"\n  model saved to: {OUTPUT_DIR}/")
    print(f"  download the '{OUTPUT_DIR}' folder from Colab Files panel")
    print(f"{'='*55}\n")


if __name__ == "__main__":
    main()
