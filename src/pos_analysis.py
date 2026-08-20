"""POS saliency distribution: human TFD vs model explanations.

Bar chart comparing the POS distribution of human eye-gaze (TFD) with
the best and worst model-method combinations, following Ikhwantri et al. Fig. 2.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
os.chdir(os.path.dirname(__file__))

import matplotlib  # noqa E402
import numpy as np  # noqa E402
import pandas as pd  # noqa E402
import spacy  # noqa E402

matplotlib.use("Agg")
from collections import defaultdict  # noqa E402

import matplotlib.pyplot as plt  # noqa E402

from utils.tobii import get_tfds, read_all_data  # noqa E402

DATA = "/home/pol/Documents/eye_tracker/eye_tracker_sexism/data"
TOBII = DATA + "/tobii/"
OUT = "/home/pol/Documents/eye_tracker/eye_tracker_sexism/latex/figures"
EXPL_DIR = os.path.join(os.path.dirname(__file__), "explanations")

nlp = spacy.load("es_core_news_sm")

mused = pd.read_csv(DATA + "/mused_chosen_data.csv")
mused["num_id"] = mused.id.str.split("_").str[1].astype(int)
text_ids_str = mused.id.tolist()
num_ids = mused.num_id.tolist()
texts = dict(zip(num_ids, mused.text_clean))
text_ids = num_ids

aoi_hit, calibration_dfs, all_participants, dfs = read_all_data(TOBII + "/all_parquets")
participants = list(dict.fromkeys([x[0] for x in all_participants]))
text_dfs, aoi_cols_dict, tfds = get_tfds(dfs, aoi_hit, participants, text_ids)

id_map = dict(zip(num_ids, text_ids_str))

POS_TAGS_ORDER = [
    "NOM",
    "VERB",
    "ADJ",
    "ADV",
    "NOMP",
    "AUX",
    "PREP",
    "DET",
    "CONJ",
    "PRON",
    "NUM",
]

# spaCy English → Catalan
SPACY_TO_CA = {
    "NOUN": "NOM",
    "VERB": "VERB",
    "ADJ": "ADJ",
    "ADV": "ADV",
    "PROPN": "NOMP",
    "AUX": "AUX",
    "ADP": "PREP",
    "DET": "DET",
    "CCONJ": "CONJ",
    "SCONJ": "CONJ",
    "PRON": "PRON",
    "NUM": "NUM",
}


def align_and_tag(raw_tokens, raw_doc):
    """Align raw split tokens to spaCy POS tags."""
    raw_spacy_tokens = [t.text for t in raw_doc]
    raw_to_spacy_idx = []
    spacy_used = [False] * len(raw_spacy_tokens)
    for raw_tok in raw_tokens:
        best_idx = -1
        for j, st in enumerate(raw_spacy_tokens):
            if not spacy_used[j] and raw_tok.lower() == st.lower():
                best_idx = j
                spacy_used[j] = True
                break
        raw_to_spacy_idx.append(best_idx)
    pos_tags = []
    for idx in raw_to_spacy_idx:
        if idx >= 0:
            spacy_pos = raw_doc[idx].pos_
            pos_tags.append(SPACY_TO_CA.get(spacy_pos, spacy_pos))
        else:
            pos_tags.append("X")
    return pos_tags


# ── Human TFD per POS ────────────────────────────────────────────────────────
print("Computing human TFD POS distribution...")
human_pos_sums = defaultdict(float)
human_total = 0.0
n_texts_human = 0

for tid in text_ids:
    raw_tokens = texts[tid].split()
    raw_doc = nlp(texts[tid])

    tfd_values = []
    for participant, participant_tfd in tfds.items():
        if tid in participant_tfd:
            tfd_values.append(participant_tfd[tid])
    if not tfd_values:
        continue
    mean_tfd = np.mean(tfd_values, axis=0)

    if len(mean_tfd) != len(raw_tokens):
        continue

    pos_tags = align_and_tag(raw_tokens, raw_doc)
    for pos, tfd_val in zip(pos_tags, mean_tfd):
        human_pos_sums[pos] += max(tfd_val, 0)
        human_total += max(tfd_val, 0)
    n_texts_human += 1

print(f"  {n_texts_human} texts, total TFD={human_total:.4f}")

# ── Model saliency per POS ──────────────────────────────────────────────────
model_configs = [
    ("beto_filtered.csv", "lrp", "BETO LRP"),
    ("mrbert_filtered.csv", "ig", "MrBERT IG"),
]

model_pos_results = {}

for fname, method, label in model_configs:
    expl_path = os.path.join(EXPL_DIR, fname)
    expl_df = pd.read_csv(expl_path)
    expl_df = expl_df[expl_df["method"] == method]

    pos_sums = defaultdict(float)
    total = 0.0
    n_texts_used = 0

    for tid in text_ids:
        str_id = id_map[tid]
        raw_tokens = texts[tid].split()
        raw_doc = nlp(texts[tid])

        text_expl = expl_df[expl_df["text_id"] == str_id]
        if text_expl.empty:
            continue

        saliency_dict = dict(zip(text_expl["word"], text_expl["salience"]))
        saliency = np.array([saliency_dict.get(w, 0.0) for w in raw_tokens])

        if np.max(saliency) <= 0:
            continue

        pos_tags = align_and_tag(raw_tokens, raw_doc)
        for pos, sal in zip(pos_tags, saliency):
            pos_sums[pos] += max(sal, 0)
            total += max(sal, 0)
        n_texts_used += 1

    model_pos_results[label] = (pos_sums, total, n_texts_used)
    print(f"  {label}: {n_texts_used} texts, total saliency={total:.2f}")

# ── Normalize and build DataFrame ────────────────────────────────────────────
keep_pos = [
    p for p in POS_TAGS_ORDER if human_pos_sums.get(p, 0) / max(human_total, 1) > 0.005
]

rows = []
for pos in keep_pos:
    rows.append(
        {
            "Source": "Humà (TFD)",
            "POS": pos,
            "Weight": human_pos_sums.get(pos, 0) / max(human_total, 1),
        }
    )

for label, (pos_sums, total, _) in model_pos_results.items():
    for pos in keep_pos:
        rows.append(
            {"Source": label, "POS": pos, "Weight": pos_sums.get(pos, 0) / max(total, 1)}
        )

df = pd.DataFrame(rows)
print(f"\nDataFrame shape: {df.shape}")
print(f"Sources: {df['Source'].unique()}")
print(f"POS tags: {keep_pos}")

# ── Plot ─────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 5))

sources = df["Source"].unique()
x = np.arange(len(keep_pos))
width = 0.25
colors = ["#2196F3", "#4CAF50", "#F44336"]

for i, source in enumerate(sources):
    subset = df[df["Source"] == source]
    vals = []
    for pos in keep_pos:
        row = subset[subset["POS"] == pos]
        vals.append(row["Weight"].values[0] if len(row) > 0 else 0)
    ax.bar(x + i * width, vals, width, label=source, color=colors[i], alpha=0.85)

ax.set_xlabel("Categoria gramatical")
ax.set_ylabel("Proporció de prominència")
ax.set_xticks(x + width)
ax.set_xticklabels(keep_pos, rotation=45, ha="right")
ax.legend(loc="upper right")
ax.set_ylim(0, max(df["Weight"]) * 1.15)
ax.grid(axis="y", alpha=0.3)

plt.tight_layout()
os.makedirs(OUT, exist_ok=True)
outpath = os.path.join(OUT, "pos_saliency_distribution.pdf")
plt.savefig(outpath, dpi=150, bbox_inches="tight")
print(f"\nSaved: {outpath}")

csv_path = os.path.join(OUT, "pos_saliency_distribution.csv")
df_pivot = df.pivot(index="POS", columns="Source", values="Weight").fillna(0)
df_pivot.to_csv(csv_path)
print(f"Saved: {csv_path}")

print("\n--- POS Distribution ---")
for pos in keep_pos:
    h = human_pos_sums.get(pos, 0) / max(human_total, 1) * 100
    m_vals = []
    for label, (pos_sums, total, _) in model_pos_results.items():
        m_vals.append(f"{pos_sums.get(pos, 0) / max(total, 1) * 100:.1f}%")
    print(
        f"  {pos:6s}: Humà={h:5.1f}%  "
        + "  ".join(f"{l}={v}" for l, v in zip([c[2] for c in model_configs], m_vals))
    )
