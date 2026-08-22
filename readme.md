# LocalVoice-SDG

A multilingual NLP framework that maps real community voices from rural India (farmers, panchayats, Punjab) to UN Sustainable Development Goals (SDGs) and surfaces actionable government schemes.

---

## Overview

Rural communities in India articulate issues in Hindi, Punjabi, or code-mixed language — but government schemes are documented in English. LocalVoice-SDG bridges this gap by:

1. Accepting text in English, Hindi, or Punjabi
2. Translating non-English input to English (NLLB-200)
3. Classifying the text across all 17 UN SDGs (XLM-RoBERTa)
4. Returning matched SDGs with confidence scores

---

## Pipeline

```
Input text (English / Hindi / Punjabi)
        ↓
Language detection (langdetect + script detection)
        ↓
Translation to English (NLLB-200-distilled-600M)  ← if not English
        ↓
Multi-label SDG classification (XLM-RoBERTa-base fine-tuned)
        ↓
Top SDGs + confidence scores
```

---

## Dataset

We curated a novel multilingual SDG-classification corpus grounded in Indian rural community discourse.

| Split | Source | Size |
|-------|--------|------|
| Annotated | 10 Indian govt PDFs (PM-KISAN, MGNREGA, Ayushman Bharat, JJM, BBBP, PM Ujjwala, eNAM, PMFBY, PMGSY, NAFCC) | 752 chunks |
| Synthetic | LLM-generated (AWS Bedrock Claude Haiku) for SDGs 10–17 | 486 chunks |
| **Total** | | **1,238 labeled chunks** |

**SDG coverage (all 17 SDGs):**

| SDG | Name | Samples |
|-----|------|---------|
| 1 | No Poverty | 201 |
| 2 | Zero Hunger | 215 |
| 3 | Good Health | 138 |
| 4 | Education | 76 |
| 5 | Gender Equality | 76 |
| 6 | Clean Water | 72 |
| 7 | Clean Energy | 78 |
| 8 | Decent Work | 118 |
| 9 | Infrastructure | 124 |
| 10 | Inequalities | 79 |
| 11 | Sustainable Cities | 84 |
| 12 | Responsible Consumption | 79 |
| 13 | Climate Action | 81 |
| 14 | Life Below Water | 73 |
| 15 | Life on Land | 137 |
| 16 | Peace & Justice | 80 |
| 17 | Partnerships | 76 |

Each chunk is 80–120 words. Labels are multi-label (1–2 SDGs per chunk).

---

## Model

- **Base:** `xlm-roberta-base` (HuggingFace)
- **Task:** Multi-label classification (17 classes)
- **Training:** 5 epochs → 10 epochs with weighted BCE loss (pos_weight per SDG)
- **Hardware:** Google Colab T4 GPU (~15 minutes)
- **Threshold:** 0.30 sigmoid confidence

---

## Evaluation Results

Evaluated on 124 held-out test samples (10% of dataset, random_state=42).

### Overall Metrics

| Metric | Score |
|--------|-------|
| **Macro F1** | **0.8115** |
| **Micro F1** | **0.8213** |
| Weighted F1 | 0.8377 |
| Macro Precision | 0.7133 |
| Macro Recall | 0.9706 |
| Hamming Loss | 0.0365 |
| Subset Accuracy | 0.5565 |

### Per-SDG Breakdown

| SDG | Name | Precision | Recall | F1 | Support |
|-----|------|-----------|--------|----|---------|
| 1 | No Poverty | 0.594 | 1.000 | 0.745 | 19 |
| 2 | Zero Hunger | 0.667 | 1.000 | 0.800 | 18 |
| 3 | Good Health | 0.833 | 1.000 | 0.909 | 15 |
| 4 | Education | 0.846 | 1.000 | 0.917 | 11 |
| 5 | Gender Equality | 0.846 | 1.000 | 0.917 | 11 |
| 6 | Clean Water | 0.727 | 1.000 | 0.842 | 8 |
| 7 | Clean Energy | 0.750 | 1.000 | 0.857 | 9 |
| 8 | Decent Work | 0.143 | 0.500 | 0.222 | 2 |
| 9 | Infrastructure | 0.800 | 1.000 | 0.889 | 12 |
| 10 | Inequalities | 0.571 | 1.000 | 0.727 | 4 |
| 11 | Sustainable Cities | 0.667 | 1.000 | 0.800 | 6 |
| 12 | Responsible Consumption | 0.542 | 1.000 | 0.703 | 13 |
| 13 | Climate Action | 0.500 | 1.000 | 0.667 | 9 |
| 14 | Life Below Water | 1.000 | 1.000 | **1.000** | 9 |
| 15 | Life on Land | 0.783 | 1.000 | 0.878 | 18 |
| 16 | Peace & Justice | 1.000 | 1.000 | **1.000** | 8 |
| 17 | Partnerships | 0.857 | 1.000 | 0.923 | 6 |



---

## Repository Structure

```
localvoice-sdg/
├── data/
│   ├── raw/                    # 810 balanced chunks from 10 PDFs
│   ├── processed/              # annotated_chunks.csv, full_dataset.csv
│   └── synthetic/              # 486 synthetic chunks (SDGs 10-17)
├── localvoice_sdg_model/       # fine-tuned XLM-RoBERTa weights
├── pdfs/                       # 10 source government PDFs
├── pdf_extractor.py            # PDF → chunks pipeline
├── auto_annotate.py            # AWS Bedrock auto-annotation
├── synthetic_gen.py            # synthetic data generation
├── train.py                    # XLM-RoBERTa fine-tuning (run on Colab)
├── translate.py                # NLLB-200 translation pipeline
├── predict.py                  # inference — text → SDG predictions
└── requirements.txt
```

---

## Usage

### Install

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install transformers sentencepiece scikit-learn pandas
```

### Predict

```bash
# English
python predict.py "Farmers in Punjab are facing severe water shortage"

# Punjabi
python predict.py "ਸਾਡੇ ਪਿੰਡ ਵਿੱਚ ਪਾਣੀ ਦੀ ਬਹੁਤ ਕਮੀ ਹੈ"

# Hindi
python predict.py "किसानों को फसल बीमा नहीं मिल रहा"

# Interactive mode
python predict.py
```

### Train (Google Colab T4 GPU)

```bash
# Upload full_dataset.csv and train.py to Colab, then:
!pip install transformers torch scikit-learn pandas -q
!python train.py
```

---

## Example Predictions

| Input | Language | Predicted SDGs |
|-------|----------|----------------|
| "Sarpanch is taking bribes and blocking MGNREGA wages" | English | SDG 16 Peace & Justice (0.963) |
| "Sutlej river polluted by industrial discharge" | English | SDG 14 Life Below Water (0.930) |
| "Stubble burning causing air pollution and soil degradation" | English | SDG 15 Life on Land (0.944), SDG 13 Climate Action (0.575) |
| "ਸਾਡੇ ਪਿੰਡ ਵਿੱਚ ਪਾਣੀ ਦੀ ਬਹੁਤ ਕਮੀ ਹੈ" | Punjabi | SDG 11 (0.865), SDG 9 (0.674) |
| "किसानों को फसल बीमा नहीं मिल रहा" | Hindi | SDG 2 Zero Hunger (0.529), SDG 1 No Poverty (0.306) |

---

## Tech Stack

| Component | Technology |
|-----------|-----------|
| Classification model | XLM-RoBERTa-base (fine-tuned) |
| Translation | NLLB-200-distilled-600M (Meta) |
| Language detection | langdetect + Unicode script detection |
| Data annotation | AWS Bedrock (Claude Haiku) |
| Training | PyTorch + HuggingFace Transformers |
| PDF extraction | pdfplumber + pdfminer |

---

## Author

Ilyas — Research project on multilingual SDG classification for rural India.
