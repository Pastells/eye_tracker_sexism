"""Stopword ablation: JS for all three distribution pairs.

Computes Jensen-Shannon divergence between:
1. Human TFD vs segment annotations
2. Model saliency vs segment annotations
3. Model saliency vs human TFD

Each computed with all tokens vs. stopwords filtered.
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
os.chdir(os.path.dirname(__file__))

import numpy as np
import pandas as pd
import spacy
from utils.tobii import get_tfds, read_all_data
from utils.metrics import jensen_shannon_divergence

DATA = "/home/pol/Documents/eye_tracker/eye_tracker_sexism/data"
TOBII = DATA + "/tobii/"
EXPL_DIR = os.path.join(os.path.dirname(__file__), "explanations")
CHOSEN_PATH = os.path.join(DATA, "chosen_data_full.csv")

nlp = spacy.load("es_core_news_sm")
es_stopwords = set(nlp.Defaults.stop_words)

# Load texts and spans
chosen = pd.read_csv(CHOSEN_PATH)
text_ids_str = chosen.id.tolist()
text_ids = [int(v.replace("video_", "")) for v in text_ids_str]
texts = dict(zip(text_ids, chosen.text_clean))
id_map = dict(zip(text_ids, text_ids_str))
# Spans are character-level; convert to word mask
spans_dict = {}
for _, row in chosen.iterrows():
    tid = int(row["id"].replace("video_", ""))
    label_raw = row.get("label_clean", "")
    if pd.isna(label_raw):
        spans_dict[tid] = []
    else:
        try:
            spans_dict[tid] = eval(label_raw)
        except:
            spans_dict[tid] = []

# Load Tobii
aoi_hit, calibration_dfs, all_participants, dfs = read_all_data(TOBII + "/all_parquets")
participants = list(dict.fromkeys([x[0] for x in all_participants]))
text_dfs, aoi_cols_dict, tfds = get_tfds(dfs, aoi_hit, participants, text_ids)


def spans_to_word_mask(spans, text):
    words = text.split()
    mask = np.zeros(len(words), dtype=bool)
    char_pos = 0
    word_starts = []
    for w in words:
        word_starts.append(char_pos)
        char_pos += len(w) + 1
    for span in spans:
        start = span["start"]
        end = span["end"]
        for i, ws in enumerate(word_starts):
            we = ws + len(words[i])
            if ws >= start and we <= end:
                mask[i] = True
            elif ws < end and we > start:
                mask[i] = True
    return mask


def normalize(d):
    s = d.sum()
    return d / s if s > 0 else d


# Model configs
model_configs = [
    ("beto_filtered.csv", "lrp", "BETO LRP"),
    ("mrbert_filtered.csv", "ig", "MrBERT IG"),
]

results = []

for tid in text_ids:
    raw_tokens = texts[tid].split()
    n_tokens = len(raw_tokens)
    spans = spans_dict.get(tid, [])
    span_mask = spans_to_word_mask(spans, texts[tid]).astype(float)

    # Human TFD
    tfd_values = []
    for participant, participant_tfd in tfds.items():
        if tid in participant_tfd:
            tfd_values.append(participant_tfd[tid])
    if not tfd_values:
        continue
    human_tfd = np.mean(tfd_values, axis=0)
    if len(human_tfd) != n_tokens:
        continue

    # Stopword mask
    keep = np.array([0.0 if t.lower() in es_stopwords else 1.0 for t in raw_tokens])

    human_all = normalize(human_tfd)
    span_all = normalize(span_mask)
    human_filt = normalize(human_tfd * keep)
    span_filt = normalize(span_mask * keep)

    js_human_span_all = jensen_shannon_divergence(span_all, human_all)
    js_human_span_filt = jensen_shannon_divergence(span_filt, human_filt)

    row_base = {
        "text_id": tid,
        "js_human_span_all": js_human_span_all,
        "js_human_span_filt": js_human_span_filt,
    }

    for fname, method, label in model_configs:
        expl_path = os.path.join(EXPL_DIR, fname)
        expl_df = pd.read_csv(expl_path)
        expl_df = expl_df[expl_df["method"] == method]
        text_expl = expl_df[expl_df["text_id"] == id_map[tid]]
        if text_expl.empty:
            continue
        saliency_dict = dict(zip(text_expl["word"], text_expl["salience"]))
        saliency = np.array([saliency_dict.get(w, 0.0) for w in raw_tokens])
        if saliency.sum() <= 0:
            continue

        model_all = normalize(saliency)
        model_filt = normalize(saliency * keep)

        r = dict(row_base)
        r["model"] = label
        r["js_model_span_all"] = jensen_shannon_divergence(span_all, model_all)
        r["js_model_span_filt"] = jensen_shannon_divergence(span_filt, model_filt)
        r["js_model_human_all"] = jensen_shannon_divergence(human_all, model_all)
        r["js_model_human_filt"] = jensen_shannon_divergence(human_filt, model_filt)
        results.append(r)

df = pd.DataFrame(results)
print(f"Texts: {df['text_id'].nunique()}, rows: {len(df)}\n")

# Print results
print(f"{'Parella':<35s} {'Tots':>12s} {'Filtrat':>12s} {'Δ':>8s}")
print("-" * 70)

# Human-segments (same for all models)
sub = df.drop_duplicates(subset=["text_id"])
mean_all = sub["js_human_span_all"].mean()
mean_filt = sub["js_human_span_filt"].mean()
pct = (mean_filt - mean_all) / mean_all * 100
print(f"{'Humà–Segments (TFD)':<35s} {mean_all:>12.3f} {mean_filt:>12.3f} {pct:>+7.1f}%")

for label in ["BETO LRP", "MrBERT IG"]:
    sub = df[df["model"] == label]
    for pair_name, col_all, col_filt in [
        (f"Model–Segments ({label})", "js_model_span_all", "js_model_span_filt"),
        (f"Model–Humà ({label})", "js_model_human_all", "js_model_human_filt"),
    ]:
        ma = sub[col_all].mean()
        mf = sub[col_filt].mean()
        pct = (mf - ma) / ma * 100
        print(f"{pair_name:<35s} {ma:>12.3f} {mf:>12.3f} {pct:>+7.1f}%")
