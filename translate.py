"""
LocalVoice-SDG — Translation Pipeline
---------------------------------------
Detects language (Hindi/Punjabi/English) and translates to English
using NLLB-200-distilled-600M before SDG classification.

First run downloads the model (~2.4GB) to ~/.cache/huggingface/
"""

from langdetect import detect, LangDetectException
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# NLLB language codes
LANG_MAP = {
    "hi": "hin_Deva",   # Hindi
    "pa": "pan_Guru",   # Punjabi (Gurmukhi)
    "en": "eng_Latn",   # English
}

NLLB_MODEL = "facebook/nllb-200-distilled-600M"

_tok   = None
_model = None


def load_translator():
    global _tok, _model
    if _tok is None:
        print("  loading translation model (first run downloads ~2.4GB)...", flush=True)
        _tok   = AutoTokenizer.from_pretrained(NLLB_MODEL)
        _model = AutoModelForSeq2SeqLM.from_pretrained(NLLB_MODEL)
        _model.eval()
        print("  translation model ready")
    return _tok, _model


def detect_language(text):
    # check for Devanagari (Hindi) or Gurmukhi (Punjabi) script first
    for ch in text:
        cp = ord(ch)
        if 0x0900 <= cp <= 0x097F:
            return "hi"   # Devanagari
        if 0x0A00 <= cp <= 0x0A7F:
            return "pa"   # Gurmukhi
    try:
        lang = detect(text)
        return lang if lang in LANG_MAP else "en"
    except LangDetectException:
        return "en"


def translate_to_english(text):
    """
    Returns (translated_text, source_lang).
    If already English, returns (text, 'en') without calling the model.
    """
    lang = detect_language(text)

    if lang == "en":
        return text, "en"

    src_code = LANG_MAP.get(lang, "hin_Deva")
    tok, model = load_translator()

    inputs = tok(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
    tok.src_lang = src_code
    inputs = tok(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
    forced_bos = tok.convert_tokens_to_ids("eng_Latn")

    import torch
    with torch.no_grad():
        output = model.generate(
            **inputs,
            forced_bos_token_id=forced_bos,
            max_length=512,
        )
    translated = tok.decode(output[0], skip_special_tokens=True)
    return translated, lang


if __name__ == "__main__":
    import sys

    tests = [
        "Hamare gaon mein paani ki bahut kami hai, fasal sukh rahi hai",
        "ਸਾਡੇ ਪਿੰਡ ਵਿੱਚ ਪਾਣੀ ਦੀ ਬਹੁਤ ਕਮੀ ਹੈ",
        "PM-KISAN gives income support to farmers",
    ]

    if len(sys.argv) > 1:
        tests = [" ".join(sys.argv[1:])]

    print("\n  Loading...\n")
    for text in tests:
        translated, lang = translate_to_english(text)
        print(f"  [{lang}] {text}")
        if lang != "en":
            print(f"  → {translated}")
        print()
