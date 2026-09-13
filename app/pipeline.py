"""
Core inference pipeline: ticket text -> model predictions -> KB retrieval -> LLM guidance.

This is the notebook logic from LLM_Banking_Implementation.ipynb, converted from an
interactive/Colab script into reusable functions with no input() calls, no
drive.mount(), and no getpass(). Everything is loaded once at import time
(module-level globals) instead of per-request, so the API stays fast.
"""

import ast
import gc
import json
import os
import pickle
from datetime import datetime

import pandas as pd
import torch
import torch.nn.functional as F
from openai import OpenAI
from transformers import AutoTokenizer

from app.model import MultiTaskModel

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
# The 4 model artifacts are NOT stored in this code repo (they're too large
# for a normal git push without LFS setup). Instead they live in a Hugging
# Face model repo (separate from Spaces — this part of HF is still free and
# has no compute restrictions), and this app downloads them once at startup
# and caches them locally inside the container.
ARTIFACTS_DIR = os.environ.get("ARTIFACTS_DIR", "artifacts")
HF_REPO_ID = os.environ.get("HF_REPO_ID")  # e.g. "yourusername/banking-ai-artifacts"
HF_TOKEN = os.environ.get("HF_TOKEN")  # required if the HF repo above is Private

CKPT_PATH = os.environ.get("CKPT_PATH", f"{ARTIFACTS_DIR}/multitask_distilbert_clean.pt")
ENCODER_PATH = os.environ.get("ENCODER_PATH", f"{ARTIFACTS_DIR}/label_encoders.pkl")
KB_PATH = os.environ.get("KB_PATH", f"{ARTIFACTS_DIR}/kb_policies_rag.csv")
KB_EMB_PATH = os.environ.get("KB_EMB_PATH", f"{ARTIFACTS_DIR}/kb_embeddings.pt")

ARTIFACT_FILENAMES = [
    "multitask_distilbert_clean.pt",
    "label_encoders.pkl",
    "kb_policies_rag.csv",
    "kb_embeddings.pt",
]


def download_artifacts_if_needed():
    """Download the 4 artifact files from the Hugging Face model repo on first
    startup, if they aren't already present locally. Safe to call every
    startup — it skips files that already exist."""
    os.makedirs(ARTIFACTS_DIR, exist_ok=True)

    all_present = all(
        os.path.exists(os.path.join(ARTIFACTS_DIR, fname)) for fname in ARTIFACT_FILENAMES
    )
    if all_present:
        print("All artifacts already present locally, skipping download.")
        return

    if not HF_REPO_ID:
        raise RuntimeError(
            "Artifacts are missing locally and HF_REPO_ID is not set. "
            "Set the HF_REPO_ID environment variable to your Hugging Face "
            "model repo (e.g. 'yourusername/banking-ai-artifacts')."
        )

    from huggingface_hub import hf_hub_download

    print(f"Downloading artifacts from Hugging Face repo: {HF_REPO_ID}")
    for fname in ARTIFACT_FILENAMES:
        local_path = os.path.join(ARTIFACTS_DIR, fname)
        if os.path.exists(local_path):
            continue
        downloaded_path = hf_hub_download(repo_id=HF_REPO_ID, filename=fname, token=HF_TOKEN)
        # hf_hub_download caches elsewhere; copy/symlink it to our artifacts dir
        if not os.path.exists(local_path):
            os.symlink(downloaded_path, local_path)
        print(f"  - {fname} downloaded")

OPENAI_MODEL = os.environ.get("OPENAI_MODEL", "gpt-5-mini-2025-08-07")
CONF_THRESHOLD = float(os.environ.get("CONF_THRESHOLD", "0.40"))

TASKS = ["intent", "issue_type", "product", "urgency", "sentiment", "routing_queue"]

REQUIRED_KEYS = [
    "interaction_id", "timestamp", "mode", "summary",
    "actions", "clarifications", "risk_notes", "kb_policies_used",
]
ALLOWED_MODES = {"HIGH_CONFIDENCE", "REVIEW_REQUIRED"}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------------------------------------------------------------------------
# Globals populated by load_artifacts() — called once at app startup.
# ---------------------------------------------------------------------------
model = None
backbone = None
tokenizer = None
encoders = None
kb_df = None
kb_embeddings_norm = None
client = None


def load_artifacts():
    """Load model, encoders, KB, KB embeddings, and OpenAI client. Call once at startup."""
    global model, backbone, tokenizer, encoders, kb_df, kb_embeddings_norm, client

    download_artifacts_if_needed()

    # mmap=True avoids reading the whole checkpoint file into RAM up front;
    # tensors are paged in from disk as needed instead.
    checkpoint = torch.load(CKPT_PATH, map_location=device, mmap=True)
    model_name = checkpoint["model_name"]
    num_labels_dict = checkpoint["num_labels_dict"]

    # pretrained_backbone=False: build the architecture from config only
    # (no download of generic pretrained weights) since load_state_dict()
    # below immediately overwrites every weight with our fine-tuned ones.
    # This avoids briefly holding two full copies of the model in memory.
    model = MultiTaskModel(model_name, num_labels_dict, pretrained_backbone=False).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    # Free the checkpoint dict now that its tensors have been copied into
    # the model — this is the single biggest memory saving at startup.
    del checkpoint
    gc.collect()

    backbone = model.backbone
    backbone.eval()

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    with open(ENCODER_PATH, "rb") as f:
        encoders = pickle.load(f)

    kb_df = pd.read_csv(KB_PATH)
    for col in ["product_tags", "issue_type_tags", "intent_tags"]:
        kb_df[col] = kb_df[col].apply(ast.literal_eval)

    kb_embeddings = torch.load(KB_EMB_PATH, map_location=device, mmap=True)
    kb_embeddings_norm = F.normalize(kb_embeddings, p=2, dim=1).clone()
    del kb_embeddings
    gc.collect()

    # Requires OPENAI_API_KEY to already be set as an env var / platform secret.
    client = OpenAI()

    print("Artifacts loaded. Model:", model_name, "| KB rows:", len(kb_df))


# ---------------------------------------------------------------------------
# Feature building — identical logic to build_input_text() in the ML notebook.
# ---------------------------------------------------------------------------
def build_input_text(channel, customer_segment, subject, timestamp, ticket_text):
    if timestamp:
        t = pd.to_datetime(timestamp, errors="coerce")
        if pd.isna(t):
            t = pd.Timestamp.now()
    else:
        t = pd.Timestamp.now()

    dow = t.day_name()
    h = t.hour
    if 5 <= h < 12:
        tb = "morning"
    elif 12 <= h < 17:
        tb = "afternoon"
    elif 17 <= h < 22:
        tb = "evening"
    else:
        tb = "night"
    bh = "yes" if 8 <= h < 18 else "no"
    we = "yes" if t.dayofweek >= 5 else "no"

    parts = [
        f"[CHANNEL={channel}]",
        f"[SEGMENT={customer_segment}]",
        f"[DOW={dow}]",
        f"[TIME_BUCKET={tb}]",
        f"[BUSINESS_HOURS={bh}]",
        f"[WEEKEND={we}]",
    ]
    if subject:
        parts.append(f"[SUBJECT={subject}]")
    parts.append(ticket_text)
    return " ".join(parts)


# ---------------------------------------------------------------------------
# Model inference
# ---------------------------------------------------------------------------
@torch.no_grad()
def run_model(input_text):
    enc = tokenizer(
        input_text, padding=True, truncation=True, max_length=256, return_tensors="pt"
    ).to(device)
    outputs = model(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"])

    pred_labels, pred_tags, confidences = {}, {}, {}
    for t in TASKS:
        probs = torch.softmax(outputs[t], dim=-1)[0]
        conf, idx = torch.max(probs, dim=0)
        conf = float(conf.item())
        label = encoders[t].inverse_transform([int(idx.item())])[0]

        pred_labels[t] = label
        confidences[t] = conf
        pred_tags[t] = "OK" if conf >= CONF_THRESHOLD else "Review Required"

    return pred_labels, pred_tags, confidences


# ---------------------------------------------------------------------------
# KB retrieval + rerank
# ---------------------------------------------------------------------------
@torch.no_grad()
def embed_query(text, max_length=256):
    enc = tokenizer(
        text, return_tensors="pt", padding=True, truncation=True, max_length=max_length
    ).to(device)
    out = backbone(**enc).last_hidden_state
    emb = out[:, 0, :]
    return F.normalize(emb, p=2, dim=1)


@torch.no_grad()
def retrieve_kb(query_text, top_k=5):
    query_emb = embed_query(query_text)
    sims = torch.matmul(kb_embeddings_norm.to(device), query_emb[0])
    top_vals, top_idx = torch.topk(sims, k=min(top_k, len(kb_df)))
    hits = kb_df.iloc[top_idx.cpu().numpy()].copy()
    hits["similarity"] = top_vals.cpu().numpy()
    return hits


def rerank_kb_hits(kb_hits, pred_labels):
    def score_row(row):
        score = row["similarity"]
        if pred_labels["product"] in row["product_tags"]:
            score += 0.10
        if pred_labels["issue_type"] in row["issue_type_tags"]:
            score += 0.10
        if pred_labels["intent"] in row["intent_tags"]:
            score += 0.05
        return score

    kb_hits = kb_hits.copy()
    kb_hits["final_score"] = kb_hits.apply(score_row, axis=1)
    return kb_hits.sort_values("final_score", ascending=False)


# ---------------------------------------------------------------------------
# Prompt building — identical to build_agent_guidance_prompt() in the notebook.
# ---------------------------------------------------------------------------
def build_agent_guidance_prompt(input_text, pred_labels, pred_tags, confidences, kb_hits, max_kb=2):
    now = datetime.now()
    interaction_id = now.strftime("%d%m%y%H%M%S%f")[:17]
    timestamp_str = now.strftime("%d/%m/%y %H:%M:%S")
    all_ok = all(pred_tags[t] == "OK" for t in pred_tags)
    mode = "HIGH_CONFIDENCE" if all_ok else "REVIEW_REQUIRED"

    lines = []
    lines.append("=== LOG DATA ===")
    lines.append(f"INTERACTION_ID: {interaction_id}")
    lines.append(f"CURRENT_DATETIME: {timestamp_str}")
    lines.append(f"MODE: {mode}\n")

    lines.append("=== LLM CONTEXT ===")
    lines.append("You are an internal AI copilot for a banking support agent. Never talk to the customer.")
    lines.append("Use ONLY the MODEL PREDICTIONS and RETRIEVED KB SNIPPETS for all actions, clarifications and risk_notes.")
    lines.append("Do NOT invent new policies, products, fees, legal terms, timeframes or guarantees.")
    lines.append("If something is not clearly covered in the KB, output clarifying questions instead of guessing.")
    lines.append("Reply with a single valid JSON object only (no markdown, no extra text).")
    lines.append("Use double-quoted keys/strings, and in kb_policies_used reference KB items by their IDs and titles.\n")

    lines.append("=== TICKET DATA ===")
    lines.append(f"{input_text}\n")

    lines.append("=== MODEL PREDICTIONS ===")
    for t in TASKS:
        lines.append(f"- {t}: {pred_labels[t]} (conf={confidences[t]:.2f}, tag={pred_tags[t]})")

    lines.append("\n=== RETRIEVED KB SNIPPETS ===")
    kb_refs = []
    for i, (_, row) in enumerate(kb_hits.head(max_kb).iterrows(), start=1):
        body = row["body_text"]
        if len(body) > 600:
            body = body[:600] + " ..."
        kb_label = f"KB{i}"
        kb_refs.append(f"{kb_label} - {row['title']}")
        lines.append(f"[{kb_label}] ID={row['kb_id']}  Title={row['title']}")
        lines.append(body)
        lines.append("")

    if kb_refs:
        lines.append("KB_LABELS_SUMMARY:")
        for ref in kb_refs:
            lines.append(f"- {ref}")
        lines.append("")

    lines.append("=== OUTPUT SCHEMA ===")
    lines.append(
        "Return ONLY a JSON object of this exact form (no extra text):\n"
        "{\n"
        '  "interaction_id": "string - copy INTERACTION_ID exactly",\n'
        '  "timestamp": "string - copy CURRENT_DATETIME exactly",\n'
        '  "mode": "HIGH_CONFIDENCE or REVIEW_REQUIRED",\n'
        '  "summary": "one-sentence internal summary of the case",\n'
        '  "actions": ["2 to 4 short action strings for the agent"],\n'
        '  "clarifications": ["0 to 4 short questions to ask the customer"],\n'
        '  "risk_notes": ["0 to 3 very short risk/compliance notes"],\n'
        '  "kb_policies_used": [\n'
        '    {"kb_id": "string", "title": "string"},\n'
        '    {"kb_id": "string", "title": "string"}\n'
        "  ]\n"
        "}"
    )

    return "\n".join(lines), interaction_id, timestamp_str, mode


def validate_agent_json(raw_text):
    try:
        data = json.loads(raw_text)
    except json.JSONDecodeError as e:
        return False, f"JSON decode error: {e}"

    if not isinstance(data, dict):
        return False, "Top-level JSON is not an object."

    missing = [k for k in REQUIRED_KEYS if k not in data]
    if missing:
        return False, f"Missing required keys: {missing}"

    if data["mode"] not in ALLOWED_MODES:
        return False, f"mode must be one of {sorted(ALLOWED_MODES)}."

    for key in ["actions", "clarifications", "risk_notes"]:
        val = data[key]
        if not isinstance(val, list) or any(not isinstance(x, str) for x in val):
