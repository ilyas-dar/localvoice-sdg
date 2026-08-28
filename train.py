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