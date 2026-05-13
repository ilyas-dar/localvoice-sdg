"""
LocalVoice-SDG — Inference Script
-----------------------------------
Takes any text (English/Hindi/Punjabi) and predicts SDG labels.

Usage:
  python predict.py "Farmers in Punjab have no water for irrigation"
  python predict.py  # interactive mode
"""

import sys
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from translate import translate_to_english

MODEL_PATH  = "./localvoice_sdg_model"
THRESHOLD   = 0.30
MAX_LENGTH  = 256

SDG_NAMES = {
    1:"No Poverty", 2:"Zero Hunger", 3:"Good Health", 4:"Education",
    5:"Gender Equality", 6:"Clean Water", 7:"Clean Energy", 8:"Decent Work",
    9:"Infrastructure", 10:"Inequalities", 11:"Sustainable Cities",
    12:"Consumption", 13:"Climate Action", 14:"Life Below Water",
    15:"Life on Land", 16:"Peace & Justice", 17:"Partnerships",
}

SDG_EMOJI = {
    1:"🟥", 2:"🟨", 3:"🟩", 4:"🟥", 5:"🟧", 6:"🟦", 7:"🟨",
    8:"🟫", 9:"🟧", 10:"🟪", 11:"🟧", 12:"🟨", 13:"🟩",
    14:"🟦", 15:"🟩", 16:"🟦", 17:"🟦",
}


def load_model():
    print("  loading model...", end="", flush=True)
    tok   = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
    model.eval()
    print(" ready\n")
    return tok, model


def predict(text, tok, model, threshold=THRESHOLD):
    inputs = tok(text, return_tensors="pt", truncation=True,
                 max_length=MAX_LENGTH, padding=True)
    with torch.no_grad():
        probs = torch.sigmoid(model(**inputs).logits)[0].tolist()

    results = []
    for i, p in enumerate(probs):
        if p >= threshold:
            sdg = i + 1
            results.append({"sdg": sdg, "name": SDG_NAMES[sdg], "confidence": round(p, 3)})

    results.sort(key=lambda x: -x["confidence"])
    return results


def print_result(text, results, translated=None, lang="en"):
    print(f"  Input : {text[:100]}{'...' if len(text) > 100 else ''}")
    if lang != "en" and translated:
        print(f"  ({lang.upper()}) → {translated[:100]}")
    print(f"  {'─'*50}")
    if not results:
        print("  No SDG matched (try lowering threshold)")
    else:
        for r in results:
            bar = "█" * int(r["confidence"] * 20)
            emoji = SDG_EMOJI.get(r["sdg"], "⬜")
            print(f"  {emoji} SDG {r['sdg']:>2} {r['name']:<22}  {r['confidence']:.3f}  {bar}")
    print()


def main():
    print("\n" + "="*55)
    print("  LocalVoice-SDG — SDG Classifier")
    print("="*55 + "\n")

    tok, model = load_model()

    # single text from command line
    if len(sys.argv) > 1:
        text = " ".join(sys.argv[1:])
        translated, lang = translate_to_english(text)
        results = predict(translated, tok, model)
        print_result(text, results, translated, lang)
        return

    # interactive mode
    print("  Enter text to classify in English/Hindi/Punjabi (or 'quit' to exit)\n")
    while True:
        try:
            text = input("  >> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n  bye!")
            break
        if not text:
            continue
        if text.lower() in ("quit", "exit", "q"):
            print("  bye!")
            break
        translated, lang = translate_to_english(text)
        results = predict(translated, tok, model)
        print_result(text, results, translated, lang)


if __name__ == "__main__":
    main()
