# LOCAL-VOICE-SDG TRAINING SCRIPT
# IN THIS TRAINING WE HAVE INCLUDED THE FOLLOWING:
"""
1. ASL asymmetric loss: helps in multi-label classification THAT IS THE MAIN DELAINGS ON OUR PROJECT.
2. K-fold cross-validation: a special one which is STRATIFIED K-FOLD CROSS-VALIDATION, which is used 
    to ensure that each fold has the same proportion of classes as the original dataset.
"""


import pandas as pd
import json
import torch

import torch.nn as nn
import numpy as np

from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
# MY CONFIGURATION OF MODEL
MODEL_NAME="xlm-roberta-base"
NO_OF_LABELS=17 #sdgs
THE_BATCH_SIZE=16
E_POX=10
LRATE=2e-5
MY_FOLDS=5
TEST_PARTS=0.15
SEED42=42
OUTPUT_DIR="models"
CSV_PATH="data/processed/full_dataset.csv"


SDG_NAMES = {
    1: "No Poverty",
    2: "Zero Hunger",
    3: "Good Health and Well-being",
    4: "Quality Education",
    5: "Gender Equality",
    6: "Clean Water and Sanitation",
    7: "Affordable and Clean Energy",
    8: "Decent Work and Economic Growth",
    9: "Industry, Innovation and Infrastructure",
    10: "Reduced Inequalities",
    11: "Sustainable Cities and Communities",
    12: "Responsible Consumption and Production",
    13: "Climate Action",
    14: "Life Below Water",
    15: "Life On Land",
    16: "Peace, Justice and Strong Institutions",
    17: "Partnerships for the Goals"
}




def parse_labels(label_str):
    if label_str == "SKIP":
        return []
    labels=json.loads(label_str)
    return labels
    

def load_data(csvPath):
    df=pd.read_csv(csvPath)
    texts=df['text'].tolist()
    df["parsed_labels"]=df["sdg_labels"].apply(parse_labels)
    df=df[df["parsed_labels"].apply(len)>0]

    texts=df['text'].tolist()

    labels=[]
    for lbl in df["parsed_labels"]:
        vec=[0]*NO_OF_LABELS
        for l in lbl:
            vec[l-1]=1
        labels.append(vec)
    return texts, labels


from torch.utils.data import Dataset
class SDGDataset(Dataset):
    def __init__(self, texts, labels, tokenizer):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
    def __len__(self):
        return len(self.texts)
    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        encoding = self.tokenizer(text, truncation=True, padding='max_length', max_length=256, return_tensors='pt')

        input_ids = encoding['input_ids'].squeeze(0)
        attention_mask = encoding['attention_mask'].squeeze(0)
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': torch.tensor(label, dtype=torch.float)
        }
        


texts, labels = load_data(CSV_PATH)


from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

dataset = SDGDataset(texts, labels, tokenizer)
print(f"Dataset size: {len(dataset)} samples")
sample = dataset[0]
print("Sample input_ids shape:", sample['input_ids'].shape)
print("Sample attention_mask shape:", sample['attention_mask'].shape)
print("Sample labels:", sample['labels'])

class AsymmetricLoss(torch.nn.Module):
    def __init__(self, gamma_neg=4, gamma_pos=1, clip=0.05):
        super(AsymmetricLoss, self).__init__()
        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip = clip
    def forward(self, logits, targets):
        probs_pos = torch.sigmoid(logits)
        probs_neg=(1.0-probs_pos+self.clip).clamp(max=1.0)
        loss_pos=targets * torch.log(probs_pos.clamp(min=1e-8)) * (1-probs_pos) ** self.gamma_pos
        loss_neg=(1-targets) * torch.log(probs_neg.clamp(min=1e-8)) * (probs_neg) ** self.gamma_neg
        loss=loss_pos+loss_neg
        return -loss.mean()    

fake_logits = torch.randn(4, 17)
fake_targets = torch.randint(0, 2, (4, 17)).float()

criterion = AsymmetricLoss()
loss_value = criterion(fake_logits, fake_targets)
print("Loss:", loss_value)

all_idxs=np.arange(len(texts))
rand_num_generator=np.random.RandomState(seed=SEED42)
shuf_idxs=rand_num_generator.permutation(all_idxs)

n_test=int(len(texts)*TEST_PARTS)
test_idxs=shuf_idxs[:n_test]
trainval_idxs=shuf_idxs[n_test:]

print(type(labels))

test_texts=[texts[i] for i in test_idxs]
test_labels=[labels[i] for i in test_idxs]

trainval_texts=[texts[i] for i in trainval_idxs]
trainval_labels=[labels[i] for i in trainval_idxs]

print(f"test set {len(test_texts)} samples")
print(f"trainval set {len(trainval_texts)} samples")
print(f"total: {len(test_texts) + len(trainval_texts)}(should equal {len(texts)})")


trainval_labels_arr = np.array(trainval_labels)
print(trainval_labels_arr.shape)


mskf = MultilabelStratifiedKFold(n_splits=MY_FOLDS, shuffle=True, random_state=SEED42)
fold_splits = list(mskf.split(trainval_texts, trainval_labels_arr))
print(len(fold_splits))

first_fold_train, first_fold_val = fold_splits[0]
print("train size:", len(first_fold_train))
print("val size:", len(first_fold_val))