"""
LocalVoice-SDG — Synthetic Data Generation
-------------------------------------------
Generates synthetic English text chunks for SDGs with zero or near-zero coverage.
Uses AWS Bedrock (same setup as auto_annotate.py).
Outputs to data/synthetic/synthetic_chunks.csv in the same schema as annotated_chunks.csv.

Usage:
  export AWS_REGION=ap-south-1
  export AWS_BEARER_TOKEN_BEDROCK=your_token_here
  python synthetic_gen.py
"""

import os
import csv
import time
import json
import re
import uuid
import boto3
from datetime import date

OUTPUT_CSV  = "./data/synthetic/synthetic_chunks.csv"
MODEL_ID    = "anthropic.claude-3-haiku-20240307-v1:0"
SLEEP_SEC   = 0.8
CHUNKS_PER_SDG = 80  # generates this many per SDG target

# top-up mode: generate only this many chunks for specific SDGs (overrides CHUNKS_PER_SDG)
TOPUP = {
    11: 20,  # annotated has 64, need ~20 more synthetic to reach ~80
    12: 21,  # synthetic has 59, need 21 more to reach ~80
}

SDG_TARGETS = {
    11: {
        "name": "Sustainable Cities and Communities",
        "description": "Rural housing, sanitation, urban-rural migration, Pradhan Mantri Awas Yojana, village infrastructure, panchayat planning",
        "context": "Punjab/rural India: PMAY housing scheme, open defecation free villages, swachh bharat in panchayats, rural-to-urban migration of farm youth, village road connectivity, slum conditions in mandi towns",
        "keywords": "PMAY, housing, sanitation, ODF, village, panchayat planning, rural infrastructure, migration, swachh bharat, community spaces"
    },
    10: {
        "name": "Reduced Inequalities",
        "description": "Inequality between SC/ST/OBC communities and upper castes, marginalized groups, wage gaps, exclusion from schemes",
        "context": "Punjab/rural India: Dalit farmers denied access to mandis, SC community excluded from water schemes, tribal communities without land rights, wage discrimination in agriculture",
        "keywords": "SC/ST, Dalit, marginalized, inequality, discrimination, exclusion, OBC, tribal, caste, landless labourers"
    },
    12: {
        "name": "Responsible Consumption and Production",
        "description": "Sustainable farming, reduction of pesticide/fertilizer overuse, food waste, organic farming, stubble burning as pollution",
        "context": "Punjab: excessive pesticide use causing cancer villages, stubble burning pollution, paddy-wheat monoculture depleting soil, chemical farming vs natural farming, food loss post-harvest",
        "keywords": "pesticide, stubble burning, organic farming, sustainable agriculture, food waste, chemical farming, soil health, overuse of fertilizers"
    },
    14: {
        "name": "Life Below Water",
        "description": "River and lake pollution, water bodies, fisheries, aquatic biodiversity, industrial discharge into rivers",
        "context": "Punjab: Sutlej/Beas/Ghaggar river pollution from industrial units, dyeing factories discharging into rivers, fishermen losing livelihood, village ponds drying up or being polluted",
        "keywords": "river pollution, Sutlej, Beas, industrial discharge, fishermen, aquatic life, water bodies, fish, pond, wetland"
    },
    16: {
        "name": "Peace Justice and Strong Institutions",
        "description": "Governance, corruption in scheme implementation, panchayat accountability, grievance redressal, RTI, legal rights of farmers",
        "context": "Punjab: corruption in MGNREGA wage payments, sarpanch nepotism, farmers unable to access scheme benefits due to middlemen, demand for transparent governance, women sarpanch empowerment",
        "keywords": "corruption, panchayat, sarpanch, governance, RTI, grievance, transparency, accountability, legal rights, middlemen"
    },
    17: {
        "name": "Partnerships for the Goals",
        "description": "Collaboration between government, NGOs, international bodies, funding partnerships, data sharing for rural development",
        "context": "Punjab: NGO-government collaboration on farmer training, NABARD partnership with SHGs, international funding for climate adaptation, KVK-farmer partnerships, digital governance initiatives",
        "keywords": "partnership, collaboration, NGO, funding, NABARD, KVK, international, cooperation, joint scheme, multi-stakeholder"
    },
    15: {
        "name": "Life on Land",
        "description": "Forest cover, soil degradation, biodiversity loss, land use change, desertification, stubble burning effects on air and soil",
        "context": "Punjab: soil health declining from chemical farming, waterlogging due to excess irrigation, saline soil, forest cover reduction, biodiversity loss in agriculture zones",
        "keywords": "soil degradation, saline land, waterlogging, biodiversity, forest, land use, desertification, crop residue, soil health"
    },
}

SYSTEM_PROMPT = """You are generating training data for LocalVoice-SDG, an NLP research project that classifies
community issues from rural Punjab and India to UN Sustainable Development Goals (SDGs).

You generate realistic short text chunks (80-120 words) that represent:
- Government scheme descriptions or reports
- Farmer/villager complaints or observations
- NGO field reports
- News article excerpts about rural issues
- Parliamentary question text

Style rules:
- Ground text in Punjab/rural India context — mention districts, crops (wheat/paddy), rivers (Sutlej/Beas),
  government programs, local place names where natural
- Keep language direct and realistic — not overly formal UN language
- Vary perspective: sometimes first-person farmer, sometimes third-person reporter, sometimes official report
- Each chunk should be self-contained and meaningful

Respond ONLY with a JSON array of chunk objects. No markdown. No explanation."""

PROMPT_TEMPLATE = """Generate {n} short text chunks (80-120 words each) related to SDG {sdg_num}: {sdg_name}.

Topic focus: {description}

Punjab/India context: {context}

Key themes to include (vary across chunks): {keywords}

Return a JSON array:
[
  {{"text": "chunk text here...", "sdg_labels": [{sdg_num}], "primary_sdg": {sdg_num}, "confidence": "high", "reasoning": "one line why this matches SDG {sdg_num}"}},
  ...
]

Generate exactly {n} chunks. Each must be 80-120 words. Vary the style and specific topic across chunks."""


def load_done_ids():
    if not os.path.exists(OUTPUT_CSV):
        return set()
    with open(OUTPUT_CSV, encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    return set(r.get("synthetic_sdg", "") for r in rows)


def save_chunks(new_chunks, append=True):
    os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
    fieldnames = [
        "id", "text", "word_count", "language", "source", "source_type",
        "collection_date", "sdg_labels", "primary_sdg", "sdg_targets",
        "confidence", "annotation_notes", "synthetic_sdg"
    ]
    file_exists = os.path.exists(OUTPUT_CSV)
    mode = "a" if append and file_exists else "w"
    with open(OUTPUT_CSV, mode, newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists or mode == "w":
            writer.writeheader()
        writer.writerows(new_chunks)


def generate_for_sdg(client, sdg_num, sdg_info, n=CHUNKS_PER_SDG):
    prompt = PROMPT_TEMPLATE.format(
        n=n,
        sdg_num=sdg_num,
        sdg_name=sdg_info["name"],
        description=sdg_info["description"],
        context=sdg_info["context"],
        keywords=sdg_info["keywords"],
    )

    body = json.dumps({
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": 4000,
        "system": SYSTEM_PROMPT,
        "messages": [{"role": "user", "content": prompt}]
    })

    try:
        response = client.invoke_model(
            modelId=MODEL_ID,
            body=body,
            contentType="application/json",
            accept="application/json",
        )
        raw = json.loads(response["body"].read())["content"][0]["text"]

        # strip markdown fences
        raw = raw.strip()
        if "```" in raw:
            for part in raw.split("```"):
                part = part.strip()
                if part.startswith("json"):
                    part = part[4:].strip()
                if part.startswith("["):
                    raw = part
                    break

        start = raw.find("[")
        end = raw.rfind("]") + 1
        if start != -1 and end > start:
            raw = raw[start:end]

        # try direct parse first
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            pass

        # fallback: extract each object individually with regex
        chunks = []
        # find all {...} blocks
        for m in re.finditer(r'\{[^{}]*\}', raw, re.DOTALL):
            obj_str = m.group(0)
            # repair newlines inside string values
            obj_str = re.sub(r'(?<=: ")(.*?)(?=")', lambda x: x.group(0).replace('\n', ' ').replace('\r', ''), obj_str, flags=re.DOTALL)
            try:
                chunks.append(json.loads(obj_str))
            except json.JSONDecodeError:
                continue
        if chunks:
            return chunks

        print(f"\n  could not recover JSON for SDG {sdg_num}")
        return []

    except Exception as e:
        print(f"\n  API error for SDG {sdg_num}: {e}")
        return []


def chunks_to_rows(chunks, sdg_num):
    today = date.today().isoformat()
    rows = []
    for c in chunks:
        text = c.get("text", "").strip()
        if not text or len(text.split()) < 30:
            continue
        rows.append({
            "id":               uuid.uuid4().hex[:8],
            "text":             text,
            "word_count":       len(text.split()),
            "language":         "en",
            "source":           f"synthetic_sdg{sdg_num}",
            "source_type":      "synthetic",
            "collection_date":  today,
            "sdg_labels":       str(c.get("sdg_labels", [sdg_num])),
            "primary_sdg":      str(c.get("primary_sdg", sdg_num)),
            "sdg_targets":      "",
            "confidence":       c.get("confidence", "high"),
            "annotation_notes": c.get("reasoning", ""),
            "synthetic_sdg":    str(sdg_num),
        })
    return rows


# ─── MAIN ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("\n" + "="*60)
    print("  LocalVoice-SDG — Synthetic Data Generation")
    print("="*60)

    region = os.environ.get("AWS_REGION", "ap-south-1")
    token  = os.environ.get("AWS_BEARER_TOKEN_BEDROCK", "")

    if not token:
        print("\n  ERROR: AWS_BEARER_TOKEN_BEDROCK not set")
        print("  run: export AWS_BEARER_TOKEN_BEDROCK=your_token")
        exit(1)

    client = boto3.client(
        service_name="bedrock-runtime",
        region_name=region,
        aws_access_key_id="bedrock",
        aws_secret_access_key="bedrock",
        aws_session_token=token,
    )

    print(f"  model: {MODEL_ID}")
    print(f"  target: {CHUNKS_PER_SDG} chunks per SDG")
    print(f"  SDGs to generate: {list(SDG_TARGETS.keys())}")

    done_sdgs = load_done_ids()
    total_generated = 0

    for sdg_num, sdg_info in SDG_TARGETS.items():
        target_n = TOPUP.get(sdg_num, CHUNKS_PER_SDG)

        # skip SDGs not in TOPUP if we only want to top up
        if TOPUP and sdg_num not in TOPUP:
            continue

        if str(sdg_num) in done_sdgs and sdg_num not in TOPUP:
            print(f"\n  SDG {sdg_num} already done — skipping")
            continue

        print(f"\n  generating SDG {sdg_num}: {sdg_info['name']} (target +{target_n}) ...", end="", flush=True)

        # split into batches of 20
        n_batches = max(1, round(target_n / 20))
        per_batch = round(target_n / n_batches)
        all_chunks = []
        for batch in range(n_batches):
            result = generate_for_sdg(client, sdg_num, sdg_info, n=per_batch)
            all_chunks.extend(result)
            time.sleep(SLEEP_SEC)

        rows = chunks_to_rows(all_chunks, sdg_num)
        if rows:
            save_chunks(rows)
            total_generated += len(rows)
            print(f" done — {len(rows)} chunks saved")
        else:
            print(f" FAILED — no chunks returned, will retry on rerun")

        time.sleep(SLEEP_SEC * 2)

    print(f"\n{'='*60}")
    print(f"  SYNTHETIC GENERATION COMPLETE")
    print(f"  total new chunks: {total_generated}")
    print(f"  saved to: {OUTPUT_CSV}")
    print(f"{'='*60}")
    print(f"\n  next: review a sample, then merge with annotated_chunks.csv")
    print(f"  merge command:")
    print(f"    python3 -c \"")
    print(f"      import pandas as pd")
    print(f"      a = pd.read_csv('./data/processed/annotated_chunks.csv')")
    print(f"      s = pd.read_csv('./data/synthetic/synthetic_chunks.csv')")
    print(f"      merged = pd.concat([a, s], ignore_index=True)")
    print(f"      merged.to_csv('./data/processed/full_dataset.csv', index=False)\"")
