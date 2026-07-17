"""
compare_results.py

Compare model explanations with span annotations and eye-tracking data.

Produces:
  - Per-text, per-model, per-method comparison metrics
  - Aggregated tables (Spearman correlations, overlap percentages)
  - Cross-entropy, JS-divergence (tripartite: model vs spans, model vs human)
  - Per-typology breakdown
  - LaTeX-ready tables for results.tex

Usage:
    uv run python compare_results.py                          # all checkpoints
    uv run python compare_results.py --checkpoint mrbert_filtered  # one checkpoint
    uv run python compare_results.py --output results_model_comparison.tex
"""

import argparse
import os
import sys
import warnings

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["NCCL_P2P_DISABLE"] = "1"
os.environ["NCCL_IB_DISABLE"] = "1"

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

sys.path.insert(0, os.path.dirname(__file__))
from utils.metrics import jensen_shannon_divergence, safe_cross_entropy, safe_kl_divergence

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
CHOSEN_PATH = os.path.join(DATA_DIR, "chosen_data_full.csv")
EXPLANATIONS_DIR = os.path.join(os.path.dirname(__file__), "explanations")
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "latex", "tables")

# Span label types
SPAN_LABELS = [
    "CRITIQUE",
    "IDEOLOGICAL AND INEQUALITY",
    "INEQUALITY/DISCRIMINATION",
    "IMPLICIT SEXISM",
    "IRONY",
    "JOKE",
    "OBJECTIFICATION",
    "REPORTED SEXISM",
    "STEREOTYPE",
]


def load_chosen_texts():
    """Load chosen texts with span annotations."""
    df = pd.read_csv(CHOSEN_PATH)
    return df


def parse_spans(label_clean_str):
    """Parse label_clean column into list of span dicts."""
    if pd.isna(label_clean_str):
        return []
    try:
        spans = eval(label_clean_str)
        if isinstance(spans, list):
            return spans
    except:
        pass
    return []


def text_to_words(text):
    """Whitespace tokenize text into words (matching eye-tracking AOI convention)."""
    return text.split()


def spans_to_word_mask(spans, text):
    """Convert character-level spans to a binary mask over whitespace tokens."""
    words = text_to_words(text)
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
            if ws < end and we > start:
                mask[i] = True

    return mask


def spans_to_label_masks(spans, text):
    """Convert spans to a dict of label → binary mask."""
    words = text_to_words(text)
    masks = {}

    char_pos = 0
    word_starts = []
    for w in words:
        word_starts.append(char_pos)
        char_pos += len(w) + 1

    for span in spans:
        span_labels = span.get("labels", [])
        if not span_labels:
            span_labels = [span.get("label", "UNKNOWN")]
        for label in span_labels:
            if label not in masks:
                masks[label] = np.zeros(len(words), dtype=bool)
            start = span["start"]
            end = span["end"]
            for i, ws in enumerate(word_starts):
                we = ws + len(words[i])
                if ws < end and we > start:
                    masks[label][i] = True

    return masks


def normalize_distribution(arr):
    """Normalize array to probability distribution (sum to 1)."""
    arr = np.abs(arr)
    total = arr.sum()
    if total > 0:
        return arr / total
    return np.ones_like(arr) / len(arr)


def load_eye_tracking_data():
    """Load eye-tracking metrics per token from the existing analysis.

    Returns DataFrame with: text_id, word, tfd, ffd, fc (averaged across participants)
    """
    from utils.tobii import (
        compute_all_token_metrics,
        get_tfds,
        read_all_data,
    )

    parquet_dir = os.path.join(DATA_DIR, "tobii", "all_parquets")
    if not os.path.exists(parquet_dir):
        print(f"WARNING: Parquet directory not found: {parquet_dir}")
        return pd.DataFrame()

    print("Loading eye-tracking data...")
    aoi_hit, calibration_dfs, all_participants, dfs = read_all_data(parquet_dir)

    seen = set()
    participants = []
    for p in all_participants:
        name = p[0] if isinstance(p, list) else p
        if name not in seen:
            seen.add(name)
            participants.append(name)

    chosen = pd.read_csv(CHOSEN_PATH)
    text_ids = [int(v.replace("video_", "")) for v in chosen.id.tolist()]

    text_dfs, aoi_cols_dict, tfds = get_tfds(dfs, aoi_hit, participants, text_ids)

    general_path = os.path.join(DATA_DIR, "tobii", "general.tsv")
    general = pd.read_csv(general_path, sep="\t")
    general = general.drop_duplicates(subset=["Participant name"])
    general = general.sort_values("Participant name").reset_index(drop=True)
    gender_map = general.set_index("Participant name")["sexe"].to_dict()
    male_count, female_count = 0, 0
    p_map = {}
    for p in sorted(gender_map.keys()):
        sex = gender_map[p]
        if sex == "male":
            male_count += 1
            p_map[p] = f"M{male_count}"
        else:
            female_count += 1
            p_map[p] = f"F{female_count}"

    text_dfs_anon = {}
    for p, texts in text_dfs.items():
        new_p = p_map.get(p, p)
        text_dfs_anon[new_p] = texts
    text_dfs = text_dfs_anon

    print(f"Computing eye-tracking metrics for {len(text_dfs)} participants...")
    metrics_df = compute_all_token_metrics(text_dfs, text_ids)

    if metrics_df.empty:
        print("WARNING: No eye-tracking metrics computed")
        return pd.DataFrame()

    metrics_df["word"] = metrics_df["AOI"].str.split(".", n=1).str[1]

    avg_metrics = (
        metrics_df.groupby(["text_id", "word"])
        .agg({"tfd": "mean", "ffd": "mean", "fc": "mean"})
        .reset_index()
    )

    avg_metrics["text_id"] = avg_metrics["text_id"].apply(lambda x: f"video_{x}")

    print(f"Loaded eye-tracking for {avg_metrics['text_id'].nunique()} texts")
    return avg_metrics


def load_explanations(checkpoint_name):
    """Load model explanations from CSV."""
    csv_path = os.path.join(EXPLANATIONS_DIR, f"{checkpoint_name}.csv")
    if not os.path.exists(csv_path):
        print(f"WARNING: Explanations not found: {csv_path}")
        return pd.DataFrame()
    return pd.read_csv(csv_path)


def safe_spearman(a, b):
    """Spearman correlation that handles constant arrays."""
    if np.std(a) == 0 or np.std(b) == 0:
        return np.nan, np.nan
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        rho, p = spearmanr(a, b)
    return rho, p


def compute_comparison_for_text(text_id, text, spans, eye_df, expl_df, method):
    """Compute comparison metrics for a single text × method.

    Returns dict with:
      - spearman_tfd, spearman_ffd, spearman_fc: correlation with eye-tracking
      - overlap_span: % of high-saliency words inside annotated spans
      - ce_model_span, kl_model_span, js_model_span: model vs spans distribution
      - ce_model_human, kl_model_human, js_model_human: model vs human attention
      - n_words, n_span_words
    """
    words = text_to_words(text)
    n_words = len(words)

    span_mask = spans_to_word_mask(spans, text)
    n_span_words = span_mask.sum()

    # Model saliency
    text_expl = expl_df[(expl_df["text_id"] == text_id) & (expl_df["method"] == method)]
    if text_expl.empty:
        return None

    saliency_dict = dict(zip(text_expl["word"], text_expl["salience"]))
    saliency = np.array([saliency_dict.get(w, 0.0) for w in words])

    # Span annotation as distribution (binary → normalized)
    span_dist = normalize_distribution(span_mask.astype(float))

    # Model saliency as distribution
    model_dist = normalize_distribution(saliency)

    # CE, KL, JS: model vs spans
    ce_model_span = safe_cross_entropy(span_dist, model_dist)
    kl_model_span = safe_kl_divergence(span_dist, model_dist)
    js_model_span = jensen_shannon_divergence(span_dist, model_dist)

    # Eye-tracking metrics
    text_eye = eye_df[eye_df["text_id"] == text_id]
    if text_eye.empty:
        return {
            "spearman_tfd": np.nan,
            "spearman_ffd": np.nan,
            "spearman_fc": np.nan,
            "overlap_span": np.nan,
            "ce_model_span": ce_model_span,
            "kl_model_span": kl_model_span,
            "js_model_span": js_model_span,
            "ce_model_human": np.nan,
            "kl_model_human": np.nan,
            "js_model_human": np.nan,
            "n_words": n_words,
            "n_span_words": int(n_span_words),
        }

    eye_dict = dict(zip(text_eye["word"], text_eye["tfd"]))
    eye_tfd = np.array([eye_dict.get(w, 0.0) for w in words])
    eye_dict_ffd = dict(zip(text_eye["word"], text_eye["ffd"]))
    eye_ffd = np.array([eye_dict_ffd.get(w, 0.0) for w in words])
    eye_dict_fc = dict(zip(text_eye["word"], text_eye["fc"]))
    eye_fc = np.array([eye_dict_fc.get(w, 0.0) for w in words])

    # Human attention as distribution (TFD)
    human_dist = normalize_distribution(eye_tfd)

    # CE, KL, JS: model vs human
    ce_model_human = safe_cross_entropy(human_dist, model_dist)
    kl_model_human = safe_kl_divergence(human_dist, model_dist)
    js_model_human = jensen_shannon_divergence(human_dist, model_dist)

    rho_tfd, _ = safe_spearman(saliency, eye_tfd)
    rho_ffd, _ = safe_spearman(saliency, eye_ffd)
    rho_fc, _ = safe_spearman(saliency, eye_fc)

    # Overlap with spans
    if n_span_words > 0 and n_words > 0:
        threshold = np.percentile(saliency, 80) if np.max(saliency) > 0 else 0
        top_mask = saliency >= threshold
        overlap = np.sum(top_mask & span_mask) / max(np.sum(top_mask), 1)
    else:
        overlap = np.nan

    return {
        "spearman_tfd": rho_tfd,
        "spearman_ffd": rho_ffd,
        "spearman_fc": rho_fc,
        "overlap_span": overlap,
        "ce_model_span": ce_model_span,
        "kl_model_span": kl_model_span,
        "js_model_span": js_model_span,
        "ce_model_human": ce_model_human,
        "kl_model_human": kl_model_human,
        "js_model_human": js_model_human,
        "n_words": n_words,
        "n_span_words": int(n_span_words),
    }


def compute_typology_for_text(text_id, text, spans, eye_df, expl_df, method):
    """Compute per-typology metrics for a single text × method.

    Returns list of dicts, one per span label present in this text.
    """
    words = text_to_words(text)
    n_words = len(words)

    label_masks = spans_to_label_masks(spans, text)
    if not label_masks:
        return []

    # Model saliency
    text_expl = expl_df[(expl_df["text_id"] == text_id) & (expl_df["method"] == method)]
    if text_expl.empty:
        return []

    saliency_dict = dict(zip(text_expl["word"], text_expl["salience"]))
    saliency = np.array([saliency_dict.get(w, 0.0) for w in words])

    # Eye-tracking
    text_eye = eye_df[eye_df["text_id"] == text_id]
    has_eye = not text_eye.empty
    if has_eye:
        eye_dict_tfd = dict(zip(text_eye["word"], text_eye["tfd"]))
        eye_tfd = np.array([eye_dict_tfd.get(w, 0.0) for w in words])
        eye_dict_fc = dict(zip(text_eye["word"], text_eye["fc"]))
        eye_fc = np.array([eye_dict_fc.get(w, 0.0) for w in words])

    results = []
    for label, mask in label_masks.items():
        n_in_span = mask.sum()
        if n_in_span == 0:
            continue

        sal_in_span = saliency[mask]
        sal_out_span = saliency[~mask]

        # Mean saliency inside vs outside
        mean_sal_in = float(np.mean(sal_in_span)) if len(sal_in_span) > 0 else 0.0
        mean_sal_out = float(np.mean(sal_out_span)) if len(sal_out_span) > 0 else 0.0

        # Spearman: saliency vs TFD inside this span type
        rho_tfd_in = np.nan
        rho_fc_in = np.nan
        if has_eye:
            eye_tfd_in = eye_tfd[mask]
            eye_fc_in = eye_fc[mask]
            if len(sal_in_span) >= 3:
                r, _ = safe_spearman(sal_in_span, eye_tfd_in)
                rho_tfd_in = r
                r, _ = safe_spearman(sal_in_span, eye_fc_in)
                rho_fc_in = r

        # JS divergence: model saliency inside span vs outside span (same-size arrays)
        in_vals = saliency.copy()
        in_vals[~mask] = 0.0
        out_vals = saliency.copy()
        out_vals[mask] = 0.0
        sal_span_dist = normalize_distribution(in_vals)
        sal_out_dist = normalize_distribution(out_vals)
        js_in_out = jensen_shannon_divergence(sal_span_dist, sal_out_dist)

        results.append(
            {
                "text_id": text_id,
                "label": label,
                "n_words_in_span": int(n_in_span),
                "mean_saliency_in": mean_sal_in,
                "mean_saliency_out": mean_sal_out,
                "spearman_tfd_in_span": rho_tfd_in,
                "spearman_fc_in_span": rho_fc_in,
                "js_model_span_type": js_in_out,
            }
        )

    return results


def generate_latex_table(results_df, caption, label, columns, col_names):
    """Generate a LaTeX table from results DataFrame."""
    n_cols = len(columns)
    col_spec = "l" + "c" * (n_cols - 1)

    lines = [
        "\\begin{table}[h]",
        "\\centering",
        f"\\begin{{tabular}}{{{col_spec}}}",
        "\\toprule",
        " & ".join(col_names) + " \\\\",
        "\\midrule",
    ]

    for _, row in results_df.iterrows():
        vals = []
        for col in columns:
            v = row[col]
            if pd.isna(v):
                vals.append("---")
            elif isinstance(v, float):
                vals.append(f"{v:.3f}")
            else:
                vals.append(str(v))
        lines.append(" & ".join(vals) + " \\\\")

    lines.extend(
        [
            "\\bottomrule",
            "\\end{tabular}",
            f"\\caption{{{caption}}}",
            f"\\label{{{label}}}",
            "\\end{table}",
        ]
    )

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(
        description="Compare model explanations with spans and eye-tracking"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Specific checkpoint to compare (default: all found)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output LaTeX file (default: auto-generated)",
    )
    parser.add_argument(
        "--no_eye_tracking",
        action="store_true",
        help="Skip eye-tracking comparison (faster)",
    )
    args = parser.parse_args()

    print("Loading chosen texts...")
    chosen_df = load_chosen_texts()

    eye_df = pd.DataFrame()
    if not args.no_eye_tracking:
        print("Loading eye-tracking data...")
        eye_df = load_eye_tracking_data()
        if not eye_df.empty:
            print(f"  Loaded eye-tracking for {eye_df.text_id.nunique()} texts")

    if args.checkpoint:
        ckpt_names = [args.checkpoint]
    else:
        ckpt_names = (
            [
                f.replace(".csv", "")
                for f in os.listdir(EXPLANATIONS_DIR)
                if f.endswith(".csv")
            ]
            if os.path.exists(EXPLANATIONS_DIR)
            else []
        )

    if not ckpt_names:
        print("No explanation files found. Run explain_models.py first.")
        sys.exit(1)

    print(f"Found {len(ckpt_names)} checkpoints: {ckpt_names}")

    all_results = []
    all_typology_records = []

    for ckpt_name in ckpt_names:
        print(f"\n--- Processing {ckpt_name} ---")
        expl_df = load_explanations(ckpt_name)
        if expl_df.empty:
            continue

        methods = expl_df["method"].unique().tolist()
        print(f"  Methods: {methods}")

        for method in methods:
            print(f"  Method: {method}")
            method_results = []
            method_typology = []

            for _, row in chosen_df.iterrows():
                text_id = row["id"]
                text = str(row["text_clean"]).replace("\n", " ")
                spans = parse_spans(row.get("label_clean", ""))

                result = compute_comparison_for_text(
                    text_id, text, spans, eye_df, expl_df, method
                )
                if result is None:
                    continue

                result["text_id"] = text_id
                result["checkpoint"] = ckpt_name
                result["method"] = method
                method_results.append(result)

                # Per-typology
                typo_records = compute_typology_for_text(
                    text_id, text, spans, eye_df, expl_df, method
                )
                for tr in typo_records:
                    tr["checkpoint"] = ckpt_name
                    tr["method"] = method
                method_typology.extend(typo_records)

            if method_results:
                df = pd.DataFrame(method_results)
                agg = {
                    "checkpoint": ckpt_name,
                    "method": method,
                    "spearman_tfd_mean": df["spearman_tfd"].mean(),
                    "spearman_ffd_mean": df["spearman_ffd"].mean(),
                    "spearman_fc_mean": df["spearman_fc"].mean(),
                    "overlap_span_mean": df["overlap_span"].mean(),
                    "ce_model_span_mean": df["ce_model_span"].mean(),
                    "kl_model_span_mean": df["kl_model_span"].mean(),
                    "js_model_span_mean": df["js_model_span"].mean(),
                    "ce_model_human_mean": df["ce_model_human"].mean(),
                    "kl_model_human_mean": df["kl_model_human"].mean(),
                    "js_model_human_mean": df["js_model_human"].mean(),
                    "n_texts": len(df),
                }
                print(f"    Spearman TFD: {agg['spearman_tfd_mean']:.3f}")
                print(f"    Spearman FC:  {agg['spearman_fc_mean']:.3f}")
                print(f"    JS (model vs span):  {agg['js_model_span_mean']:.3f}")
                print(f"    JS (model vs human): {agg['js_model_human_mean']:.3f}")
                print(f"    Overlap span: {agg['overlap_span_mean']:.3f}")
                all_results.append(agg)

            all_typology_records.extend(method_typology)

    if not all_results:
        print("No results to aggregate.")
        return

    results_df = pd.DataFrame(all_results)

    # Save main CSV
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    csv_path = os.path.join(OUTPUT_DIR, "model_comparison.csv")
    results_df.to_csv(csv_path, index=False)
    print(f"\nSaved CSV: {csv_path}")

    # Save typology CSV
    if all_typology_records:
        typo_df = pd.DataFrame(all_typology_records)
        typo_csv_path = os.path.join(OUTPUT_DIR, "model_typology.csv")
        typo_df.to_csv(typo_csv_path, index=False)
        print(f"Saved typology CSV: {typo_csv_path}")

    # Generate LaTeX tables
    if args.output:
        latex_path = args.output
    else:
        latex_path = os.path.join(OUTPUT_DIR, "model_comparison.tex")

    latex_parts = []
    latex_parts.append("% Auto-generated by compare_results.py\n")

    # Table 1: Spearman correlations
    spearman_table = results_df[
        [
            "checkpoint",
            "method",
            "spearman_tfd_mean",
            "spearman_ffd_mean",
            "spearman_fc_mean",
        ]
    ]
    latex_parts.append(
        generate_latex_table(
            spearman_table,
            caption="Correlació de Spearman: saliència del model vs. mètriques d'eye-tracking (mitjana sobre texts).",
            label="tab:model_vs_eye_tracking",
            columns=[
                "checkpoint",
                "method",
                "spearman_tfd_mean",
                "spearman_ffd_mean",
                "spearman_fc_mean",
            ],
            col_names=["Checkpoint", "Mètode", "$\\rho$ TFD", "$\\rho$ FFD", "$\\rho$ FC"],
        )
    )
    latex_parts.append("")

    # Table 2: Overlap with spans
    overlap_table = results_df[["checkpoint", "method", "overlap_span_mean"]]
    latex_parts.append(
        generate_latex_table(
            overlap_table,
            caption="Solapament: paraules amb saliència més alta vs. paraules dins dels spans anotats.",
            label="tab:model_vs_spans",
            columns=["checkpoint", "method", "overlap_span_mean"],
            col_names=["Checkpoint", "Mètode", "Solapament"],
        )
    )
    latex_parts.append("")

    # Table 3: Cross-entropy, KL, JS (model vs spans)
    dist_span_table = results_df[
        [
            "checkpoint",
            "method",
            "ce_model_span_mean",
            "kl_model_span_mean",
            "js_model_span_mean",
        ]
    ]
    latex_parts.append(
        generate_latex_table(
            dist_span_table,
            caption="Mètriques de distribució: saliència del model vs. anotacions span (mitjana sobre texts).",
            label="tab:model_vs_span_dist",
            columns=[
                "checkpoint",
                "method",
                "ce_model_span_mean",
                "kl_model_span_mean",
                "js_model_span_mean",
            ],
            col_names=["Checkpoint", "Mètode", "Cross-Entropy", "KL", "JS"],
        )
    )
    latex_parts.append("")

    # Table 4: Cross-entropy, KL, JS (model vs human attention)
    dist_human_table = results_df[
        [
            "checkpoint",
            "method",
            "ce_model_human_mean",
            "kl_model_human_mean",
            "js_model_human_mean",
        ]
    ]
    latex_parts.append(
        generate_latex_table(
            dist_human_table,
            caption="Mètriques de distribució: saliència del model vs. atenció humana (TFD, mitjana sobre texts).",
            label="tab:model_vs_human_dist",
            columns=[
                "checkpoint",
                "method",
                "ce_model_human_mean",
                "kl_model_human_mean",
                "js_model_human_mean",
            ],
            col_names=["Checkpoint", "Mètode", "Cross-Entropy", "KL", "JS"],
        )
    )
    latex_parts.append("")

    # Table 5: Per-typology aggregated
    if all_typology_records:
        typo_df = pd.DataFrame(all_typology_records)
        typo_agg = (
            typo_df.groupby(["label"])
            .agg(
                {
                    "mean_saliency_in": "mean",
                    "mean_saliency_out": "mean",
                    "spearman_tfd_in_span": "mean",
                    "spearman_fc_in_span": "mean",
                    "js_model_span_type": "mean",
                    "n_words_in_span": "mean",
                }
            )
            .reset_index()
        )
        typo_agg = typo_agg.sort_values("n_words_in_span", ascending=False)

        latex_parts.append(
            generate_latex_table(
                typo_agg,
                caption="Mètriques per tipus d'etiqueta span: saliència del model dins de cada tipologia (mitjana sobre texts i checkpoints).",
                label="tab:model_typology",
                columns=[
                    "label",
                    "mean_saliency_in",
                    "mean_saliency_out",
                    "spearman_tfd_in_span",
                    "spearman_fc_in_span",
                    "js_model_span_type",
                ],
                col_names=[
                    "Etiqueta",
                    "Sal. mitjana (dins)",
                    "Sal. mitjana (fora)",
                    "$\\rho$ TFD (dins)",
                    "$\\rho$ FC (dins)",
                    "JS (dins vs tot)",
                ],
            )
        )
        latex_parts.append("")

    with open(latex_path, "w") as f:
        f.write("\n".join(latex_parts))
        f.write("\n")

    print(f"Saved LaTeX: {latex_path}")


if __name__ == "__main__":
    main()
