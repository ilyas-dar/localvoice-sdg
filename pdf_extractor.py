"""
LocalVoice-SDG — PDF Extraction Pipeline v2
---------------------------------------------
improvements over v1:
  - skips table of contents / index sections
  - skips table rows and number-heavy lines
  - skips header/footer boilerplate
  - finds minimum chunk count across all sources
  - caps all sources at that minimum (equal distribution)
  - saves both full and balanced CSVs

Install:
  pip install pdfplumber langdetect pandas tqdm

Usage:
  python pdf_extractor.py
  → saves full_chunks.csv (uncapped, all chunks)
  → prints per-source counts + threshold
  → saves balanced_chunks.csv (capped at min source count)
"""

import os
import re
import uuid
import pdfplumber
import pandas as pd
from tqdm import tqdm
from langdetect import detect, DetectorFactory
from langdetect.lang_detect_exception import LangDetectException

DetectorFactory.seed = 42

# ─── CONFIG ───────────────────────────────────────────────────────────────────
PDF_FOLDER       = "./pdfs"
FULL_CSV         = "./data/raw/full_chunks.csv"
BALANCED_CSV     = "./data/raw/balanced_chunks.csv"
MIN_WORDS        = 60
MAX_WORDS        = 130
KEEP_LANGS       = {"en", "hi", "pa"}

# TOC detection — pages matching these patterns get skipped entirely
TOC_PATTERNS = [
    r"\.{4,}",
    r"^(contents|index|table of contents|list of tables|list of figures|abbreviations)$",
]

# line-level boilerplate to drop before chunking
SKIP_LINE_PATTERNS = [
    r"^\s*page\s*\d+\s*$",
    r"^\s*\d+\s*$",
    r"^\s*(ministry|government of india)\s*$",
    r"^(sl\.?\s*no\.?|s\.no|sr\.?\s*no)",
    r"^\s*fig(ure)?\s*\d+",
    r"^\s*table\s*\d+",
    r"©|all rights reserved|printed at",
    r"^\s*\*{2,}\s*$",
]

# chunks with more than this ratio of number tokens = table noise, skip
NUMBER_RATIO_THRESHOLD = 0.35
# ──────────────────────────────────────────────────────────────────────────────


def is_toc_page(text: str) -> bool:
    """
    returns True if page looks like a table of contents or index.
    checks for dot leaders, standalone page numbers, TOC headers.
    """
    lines = text.strip().split("\n")
    if not lines:
        return False

    first_line = lines[0].strip().lower()
    for pattern in TOC_PATTERNS:
        if re.match(pattern, first_line, re.IGNORECASE):
            return True

    # more than 40% of lines end with a page number = TOC
    lines_ending_number = sum(1 for l in lines if re.search(r"\s+\d+\s*$", l))
    if len(lines) > 3 and lines_ending_number / len(lines) > 0.4:
        return True

    # more than 40% of lines have dot leaders = TOC
    lines_with_dots = sum(1 for l in lines if re.search(r"\.{4,}", l))
    if len(lines) > 3 and lines_with_dots / len(lines) > 0.4:
        return True

    return False


def is_skip_line(line: str) -> bool:
    """returns True if line is boilerplate noise that should be dropped"""
    line = line.strip()
    if not line:
        return True
    for pattern in SKIP_LINE_PATTERNS:
        if re.search(pattern, line, re.IGNORECASE):
            return True
    return False


def is_table_heavy(text: str) -> bool:
    """
    returns True if chunk is mostly numbers — likely a table extraction.
    these are useless for NLP training.
    """
    tokens = text.split()
    if not tokens:
        return True
    number_tokens = sum(1 for t in tokens if re.match(r"^[\d,\.%₹$\-/]+$", t))
    return number_tokens / len(tokens) > NUMBER_RATIO_THRESHOLD


def clean_text(text: str) -> str:
    """cleans page text line by line, drops boilerplate"""
    if not text:
        return ""

    lines = text.split("\n")
    cleaned_lines = []

    for line in lines:
        if is_skip_line(line):
            continue
        line = re.sub(r"http\S+|www\.\S+", "", line)
        line = re.sub(r"[-_]{3,}", " ", line)
        line = re.sub(r"\s+", " ", line).strip()
        if len(line) > 20:
            cleaned_lines.append(line)

    return " ".join(cleaned_lines).strip()


def extract_text_from_pdf(pdf_path: str) -> str:
    """
    extracts text from PDF page by page.
    skips scanned pages, TOC pages, cleans each page.
    """
    full_text = []
    skipped_toc = 0

    try:
        with pdfplumber.open(pdf_path) as pdf:
            for page_num, page in enumerate(pdf.pages):
                page_text = page.extract_text()

                if not page_text:
                    continue

                if is_toc_page(page_text):
                    skipped_toc += 1
                    continue

                cleaned = clean_text(page_text)
                if len(cleaned) > 80:
                    full_text.append(cleaned)

    except Exception as e:
        print(f"  error reading {pdf_path}: {e}")
        return ""

    if skipped_toc:
        print(f"  skipped {skipped_toc} TOC/index pages")

    return " ".join(full_text)


def chunk_text(text: str) -> list[str]:
    """
    splits text into 60-130 word chunks at sentence boundaries.
    drops chunks that are table-heavy (number ratio too high).
    """
    sentences = re.split(r"(?<=[.!?])\s+", text)
    chunks = []
    current_chunk = []
    current_word_count = 0

    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence or len(sentence) < 20:
            continue

        word_count = len(sentence.split())

        if current_word_count + word_count <= MAX_WORDS:
            current_chunk.append(sentence)
            current_word_count += word_count
        else:
            if current_word_count >= MIN_WORDS:
                chunk_str = " ".join(current_chunk)
                if not is_table_heavy(chunk_str):
                    chunks.append(chunk_str)
            current_chunk = [sentence]
            current_word_count = word_count

    # last chunk
    if current_chunk and current_word_count >= MIN_WORDS:
        chunk_str = " ".join(current_chunk)
        if not is_table_heavy(chunk_str):
            chunks.append(chunk_str)

    return chunks


def detect_language(text: str) -> str:
    """detects language — en/hi/pa or other"""
    try:
        lang = detect(text)
        return lang if lang in KEEP_LANGS else "other"
    except LangDetectException:
        return "unknown"


def process_all_pdfs(pdf_folder: str) -> pd.DataFrame:
    """
    runs full extraction pipeline on all PDFs.
    returns DataFrame of all extracted chunks.
    """
    pdf_files = [
        os.path.join(pdf_folder, f)
        for f in os.listdir(pdf_folder)
        if f.lower().endswith(".pdf")
    ]

    if not pdf_files:
        print(f"no PDFs found in {pdf_folder}")
        return pd.DataFrame()

    print(f"found {len(pdf_files)} PDFs\n")
    all_chunks = []

    for pdf_path in tqdm(pdf_files, desc="extracting"):
        source_name = os.path.splitext(os.path.basename(pdf_path))[0]
        print(f"\nprocessing: {source_name}")

        raw_text = extract_text_from_pdf(pdf_path)
        if not raw_text:
            print(f"  skipping — no text extracted")
            continue

        print(f"  extracted {len(raw_text.split())} words")
        chunks = chunk_text(raw_text)
        print(f"  {len(chunks)} chunks after TOC/table filtering")

        kept = 0
        for chunk in chunks:
            lang = detect_language(chunk)
            if lang == "other":
                continue

            all_chunks.append({
                "id":               str(uuid.uuid4())[:8],
                "text":             chunk,
                "word_count":       len(chunk.split()),
                "language":         lang,
                "source":           source_name,
                "source_type":      "government_document",
                "collection_date":  pd.Timestamp.today().strftime("%Y-%m-%d"),
                "sdg_labels":       "",
                "primary_sdg":      "",
                "sdg_targets":      "",
                "confidence":       "",
                "annotation_notes": "",
            })
            kept += 1

        print(f"  kept {kept} after language filter")

    return pd.DataFrame(all_chunks)


def apply_threshold(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """
    finds source with minimum chunks, uses that as the cap.
    all sources get truncated to that count for equal distribution.
    returns balanced DataFrame and threshold value.
    """
    source_counts = df["source"].value_counts()

    print(f"\n{'='*50}")
    print("per-source counts before balancing:")
    print(source_counts.to_string())

    min_source = source_counts.idxmin()
    threshold = int(source_counts.min())

    print(f"\nminimum: '{min_source}' = {threshold} chunks")
    print(f"threshold set to {threshold} — all sources capped here")

    frames = []
    for source, group in df.groupby("source"):
        frames.append(group.head(threshold))
    balanced = pd.concat(frames).reset_index(drop=True)

    return balanced, threshold


def save_csv(df: pd.DataFrame, path: str, label: str):
    """saves DataFrame to CSV with summary stats"""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8-sig")

    print(f"\n{label}")
    print(f"total chunks: {len(df)}")
    print(f"language distribution:\n{df['language'].value_counts().to_string()}")
    print(f"avg words per chunk: {df['word_count'].mean():.0f}")
    print(f"source breakdown:\n{df['source'].value_counts().to_string()}")
    print(f"saved to: {path}")


# ─── MAIN ────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("LocalVoice-SDG — PDF Extraction Pipeline v2")
    print("=" * 50)

    # step 1: extract everything into a DataFrame
    df_full = process_all_pdfs(PDF_FOLDER)

    if df_full.empty:
        print("no chunks extracted. check your PDF folder.")
        exit()

    # step 2: save full unbalanced dataset
    save_csv(df_full, FULL_CSV, "FULL DATASET (unbalanced):")

    # step 3: balance and save
    df_balanced, threshold = apply_threshold(df_full)
    save_csv(df_balanced, BALANCED_CSV, f"BALANCED DATASET (threshold={threshold}):")

    print(f"\n{'='*50}")
    print(f"done. {len(df_balanced)} balanced chunks ready for annotation.")
    print(f"open data/raw/balanced_chunks.csv and fill in sdg_labels column.")