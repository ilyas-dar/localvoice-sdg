"""
LocalVoice-SDG — Auto Annotation using Groq API
-------------------------------------------------
automatically labels all 810 chunks with SDG tags,
reasoning, and confidence using Groq (llama-3.1-8b).

completely free, no daily quota limits.
saves after every 10 chunks — safe to stop and resume.

Install:
  pip install groq

Usage:
  export GROQ_API_KEY=your_key_here
  python auto_annotate.py
"""

import os
import csv
import time
import json
from groq import Groq

# ─── CONFIG ───────────────────────────────────────────────────────────────────
INPUT_CSV  = "./data/raw/balanced_chunks.csv"
OUTPUT_CSV = "./data/processed/annotated_chunks.csv"
MODEL      = "llama-3.1-8b-instant"
SLEEP_SEC  = 0.5   # groq is fast, 0.5s is fine
# ──────────────────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are an expert annotator for a multilingual NLP research project called LocalVoice-SDG.
Your job is to read a text chunk from an Indian government document and assign the correct UN Sustainable Development Goal (SDG) labels.

SDG reference:
1  = No Poverty (income support, financial assistance, welfare, cash transfer, BPL)
2  = Zero Hunger (agriculture, crops, farmers, food security, yield, seeds, fertilizer, kisan)
3  = Good Health (health, hospital, medicine, insurance, nutrition, disease, Ayushman)
4  = Quality Education (school, literacy, dropout, scholarship, education)
5  = Gender Equality (women, girl child, SHG, mahila, empowerment, Beti Bachao)
6  = Clean Water & Sanitation (drinking water, groundwater, sanitation, toilet, irrigation, JJM)
7  = Affordable Clean Energy (LPG, solar, electricity, clean cooking, Ujjwala, biogas)
8  = Decent Work & Economy (employment, wages, MGNREGA, job, livelihood, rural employment)
9  = Industry & Infrastructure (road, bridge, connectivity, PMGSY, rural roads, construction)
10 = Reduced Inequalities (SC, ST, OBC, marginalized, tribal, inclusion, backward class)
11 = Sustainable Cities (housing, urban, slum, shelter, PMAY, village development)
12 = Responsible Consumption (food waste, pesticide, organic farming, stubble burning)
13 = Climate Action (flood, drought, climate, disaster, extreme weather, crop loss, NAFCC)
14 = Life Below Water (river, lake, water pollution, fisheries, aquatic)
15 = Life on Land (forest, soil, biodiversity, land degradation, deforestation)
16 = Peace & Justice (governance, corruption, panchayat, legal rights, transparency)
17 = Partnerships (collaboration, funding, international, inter-agency)

Rules:
- assign 1 to 3 SDG labels maximum
- pick only the most relevant SDGs
- if chunk has no meaningful content (only names, numbers, tables, addresses) set skip=true
- confidence: high=clearly matches, medium=partially matches, low=loosely matches
- reasoning: 1 short sentence only

IMPORTANT: Respond ONLY with valid JSON. No explanation. No markdown. No code fences."""

PROMPT_TEMPLATE = """Annotate this text chunk from Indian government document.
Source: {source}

TEXT:
{text}

Respond with ONLY this JSON, nothing else:
{{"skip": false, "sdg_labels": [2, 8], "primary_sdg": 2, "sdg_targets": "", "confidence": "high", "reasoning": "text discusses crop yield and farm labour costs"}}

If chunk should be skipped (only names/numbers/tables):
{{"skip": true, "sdg_labels": [], "primary_sdg": null, "sdg_targets": "", "confidence": "low", "reasoning": "chunk contains only names or numbers"}}"""


def load_csv(path):
    """loads CSV as list of dicts"""
    with open(path, encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def save_csv(rows, path):
    """saves list of dicts to CSV"""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def is_annotated(row):
    """returns True if row already has sdg_labels filled"""
    val = row.get("sdg_labels", "").strip()
    return val != "" and val != "nan"


def clean_json(raw):
    """strips markdown fences if model wraps response"""
    raw = raw.strip()
    if "```" in raw:
        parts = raw.split("```")
        for part in parts:
            part = part.strip()
            if part.startswith("json"):
                part = part[4:]
            part = part.strip()
            if part.startswith("{"):
                return part
    # find first { to last }
    start = raw.find("{")
    end = raw.rfind("}") + 1
    if start != -1 and end > start:
        return raw[start:end]
    return raw


def annotate_with_groq(client, text, source):
    """
    calls Groq API to annotate a single chunk.
    returns parsed result dict or None if failed.
    """
    prompt = PROMPT_TEMPLATE.format(
        source=source,
        text=text[:1200]
    )

    try:
        response = client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": prompt}
            ],
            temperature=0.1,   # low temp for consistent structured output
            max_tokens=200,
        )

        raw = response.choices[0].message.content
        raw = clean_json(raw)
        result = json.loads(raw)
        return result

    except json.JSONDecodeError as e:
        print(f"\n  JSON error: {e} | raw: {raw[:100]}")
        return None
    except Exception as e:
        print(f"\n  API error: {e}")
        return None


def apply_result(row, result):
    """applies annotation result to row dict"""
    if result.get("skip", False):
        row["sdg_labels"]       = "SKIP"
        row["primary_sdg"]      = ""
        row["sdg_targets"]      = ""
        row["confidence"]       = "low"
        row["annotation_notes"] = result.get("reasoning", "skipped")
    else:
        labels  = result.get("sdg_labels", [])
        primary = result.get("primary_sdg", "")
        row["sdg_labels"]       = str(labels)
        row["primary_sdg"]      = str(primary) if primary else ""
        row["sdg_targets"]      = str(result.get("sdg_targets", ""))
        row["confidence"]       = result.get("confidence", "medium")
        row["annotation_notes"] = result.get("reasoning", "")
    return row


def print_summary(rows):
    """prints final summary with SDG distribution"""
    total   = len(rows)
    labeled = [r for r in rows if is_annotated(r) and r.get("sdg_labels") != "SKIP"]
    skipped = [r for r in rows if r.get("sdg_labels") == "SKIP"]
    failed  = [r for r in rows if not is_annotated(r)]

    print(f"\n{'='*60}")
    print(f"  AUTO-ANNOTATION COMPLETE")
    print(f"{'='*60}")
    print(f"  total:    {total}")
    print(f"  labeled:  {len(labeled)}")
    print(f"  skipped:  {len(skipped)}  (no-content chunks)")
    print(f"  failed:   {len(failed)}   (rerun to retry)")

    from collections import Counter
    all_labels = []
    for r in labeled:
        try:
            labels = json.loads(r["sdg_labels"].replace("'", '"'))
            all_labels.extend(labels)
        except:
            pass

    sdg_names = {
        1:"No Poverty", 2:"Zero Hunger", 3:"Good Health",
        4:"Education", 5:"Gender Equality", 6:"Clean Water",
        7:"Clean Energy", 8:"Decent Work", 9:"Infrastructure",
        10:"Inequalities", 11:"Sustainable Cities", 12:"Consumption",
        13:"Climate Action", 14:"Life Below Water", 15:"Life on Land",
        16:"Peace & Justice", 17:"Partnerships"
    }

    if all_labels:
        counts = Counter(all_labels)
        print(f"\n  SDG distribution:")
        for sdg in sorted(counts.keys()):
            bar = "█" * min(counts[sdg] // 2, 30)
            print(f"  SDG {sdg:2d} {sdg_names.get(sdg,''):22s} {counts[sdg]:4d}  {bar}")

    print(f"\n  saved to: {OUTPUT_CSV}")
    print(f"{'='*60}")
    print(f"\n  next: git add data/processed/annotated_chunks.csv")
    print(f"        git commit -m 'data: auto-annotate {len(labeled)} chunks via Groq'")


# ─── MAIN ────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("\n" + "="*60)
    print("  LocalVoice-SDG — Auto Annotation via Groq API")
    print("="*60)

    # check API key
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        print("\n  ERROR: GROQ_API_KEY not set")
        print("  run: export GROQ_API_KEY=your_key_here")
        exit(1)

    client = Groq(api_key=api_key)
    print(f"  model: {MODEL}")

    # load chunks
    rows  = load_csv(INPUT_CSV)
    total = len(rows)
    print(f"  loaded {total} chunks")

    # resume if output exists
    if os.path.exists(OUTPUT_CSV):
        print(f"  resuming from existing output...")
        done_rows = load_csv(OUTPUT_CSV)
        done_map  = {r["id"]: r for r in done_rows}
        for row in rows:
            if row["id"] in done_map:
                saved = done_map[row["id"]]
                if is_annotated(saved):
                    row["sdg_labels"]       = saved.get("sdg_labels", "")
                    row["primary_sdg"]      = saved.get("primary_sdg", "")
                    row["sdg_targets"]      = saved.get("sdg_targets", "")
                    row["confidence"]       = saved.get("confidence", "")
                    row["annotation_notes"] = saved.get("annotation_notes", "")
        already = sum(1 for r in rows if is_annotated(r))
        print(f"  already done: {already} chunks")
    else:
        already = 0

    remaining = total - already
    est_mins  = (remaining * SLEEP_SEC) / 60
    print(f"  remaining: {remaining} chunks")
    print(f"  estimated time: ~{est_mins:.1f} minutes")
    print(f"\n  starting in 3 seconds... Ctrl+C to stop anytime")
    time.sleep(3)

    # main loop
    errors = 0
    for i, row in enumerate(rows):
        if is_annotated(row):
            continue

        text   = row.get("text", "")
        source = row.get("source", "")

        result = annotate_with_groq(client, text, source)

        if result:
            rows[i] = apply_result(row, result)
            errors  = 0
        else:
            errors += 1
            rows[i]["annotation_notes"] = "auto-annotation failed"
            if errors >= 5:
                print(f"\n  5 consecutive errors — saving and stopping")
                print(f"  check API key or rate limits then rerun")
                break

        # save every 10 chunks
        if i % 10 == 0:
            save_csv(rows, OUTPUT_CSV)
            done = sum(1 for r in rows if is_annotated(r))
            print(f"  progress: {done}/{total} ({done/total*100:.1f}%)    ", end="\r")

        time.sleep(SLEEP_SEC)

    # final save
    save_csv(rows, OUTPUT_CSV)
    print_summary(rows)