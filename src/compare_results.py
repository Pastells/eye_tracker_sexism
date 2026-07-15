"""
compare_results.py

Compare model explanations with span annotations and eye-tracking data.

Produces:
  - Per-text, per-model, per-method comparison metrics
  - Aggregated tables (Spearman correlations, overlap percentages)
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

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
CHOSEN_PATH = os.path.join(DATA_DIR, "chosen_data_full.csv")
EXPLANATIONS_DIR = os.path.join(os.path.dirname(__file__), "explanations")
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "latex", "tables")


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

    # Build char offset → word index mapping
    char_pos = 0
    word_starts = []
    for w in words:
        word_starts.append(char_pos)
        char_pos += len(w) + 1  # +1 for space

    for span in spans:
        start = span["start"]
        end = span["end"]
        for i, ws in enumerate(word_starts):
            we = ws + len(words[i])
            # Word overlaps with span if intervals intersect
            if ws < end and we > start:
                mask[i] = True

    return mask


def load_eye_tracking_data():
    """Load eye-tracking metrics per token from the existing analysis.

    Returns DataFrame with: text_id, word, tfd, ffd, fc (averaged across participants)
    """
    # We need to compute eye-tracking metrics from raw data
    # Import the necessary functions
    sys.path.insert(0, os.path.dirname(__file__))

    from utils.tobii import (
        compute_all_token_metrics,
        extract_fixations,
        read_data,
    )

    # Find all parquet files
    parquet_dir = os.path.join(DATA_DIR, "tobii", "all_parquets")
    if not os.path.exists(parquet_dir):
        print(f"WARNING: Parquet directory not found: {parquet_dir}")
        print("Eye-tracking comparison will be skipped.")
        return pd.DataFrame()

    parquet_files = sorted(
        [
            os.path.join(parquet_dir, f)
            for f in os.listdir(parquet_dir)
            if f.endswith(".parquet")
        ]
    )

    if not parquet_files:
        print("WARNING: No parquet files found")
        return pd.DataFrame()

    # Load participant metadata
    general_path = os.path.join(DATA_DIR, "tobii", "general.tsv")
    general = pd.read_csv(general_path, sep="\t")
    general = general.drop_duplicates(subset=["Participant name"])
    gender_map = general.set_index("Participant name")["sexe"].to_dict()

    # Load all data
    from pathlib import Path

    text_dfs = {}
    for pf in parquet_files:
        fname = Path(pf).stem
        # Parse participant and text from filename: e.g., "gabriel video_5"
        parts = fname.split(" ", 1)
        if len(parts) < 2:
            continue
        participant = parts[0].strip()
        text_part = parts[1].strip()

        # Normalize participant name for lookup
        if participant not in gender_map:
            # Try matching with underscores
            for key in gender_map:
                if key.lower().replace("_", " ") == participant.lower():
                    participant = key
                    break

        try:
            df = read_data(pf)
            if participant not in text_dfs:
                text_dfs[participant] = {}
            text_dfs[participant][text_part] = df
        except Exception as e:
            print(f"  Warning: Could not load {pf}: {e}")

    if not text_dfs:
        print("WARNING: No eye-tracking data loaded")
        return pd.DataFrame()

    # Get text IDs from chosen texts
    chosen = pd.read_csv(CHOSEN_PATH)
    text_ids = chosen.id.tolist()

    # Compute per-token metrics
    print("Computing eye-tracking metrics per token...")
    metrics_df = compute_all_token_metrics(text_dfs, text_ids)

    if metrics_df.empty:
        print("WARNING: No eye-tracking metrics computed")
        return pd.DataFrame()

    # Extract word from AOI (format: "idx.word")
    metrics_df["word"] = metrics_df["AOI"].str.split(".", n=1).str[1]

    # Average across participants per text_id × word
    avg_metrics = (
        metrics_df.groupby(["text_id", "word"])
        .agg({"tfd": "mean", "ffd": "mean", "fc": "mean"})
        .reset_index()
    )

    return avg_metrics


def load_explanations(checkpoint_name):
    """Load model explanations from CSV."""
    csv_path = os.path.join(EXPLANATIONS_DIR, f"{checkpoint_name}.csv")
    if not os.path.exists(csv_path):
        print(f"WARNING: Explanations not found: {csv_path}")
        return pd.DataFrame()
    return pd.read_csv(csv_path)


def compute_comparison_for_text(text_id, text, spans, eye_df, expl_df, method):
    """Compute comparison metrics for a single text × method.

    Returns dict with:
      - spearman_tfd, spearman_ffd, spearman_fc: correlation with eye-tracking
      - overlap_span: % of high-saliency words inside annotated spans
      - n_words: number of words in text
      - n_spans: number of annotated spans
    """
    words = text_to_words(text)
    n_words = len(words)

    # Span mask
    span_mask = spans_to_word_mask(spans, text)
    n_span_words = span_mask.sum()

    # Model saliency for this text and method
    text_expl = expl_df[(expl_df["text_id"] == text_id) & (expl_df["method"] == method)]
    if text_expl.empty:
        return None

    # Align saliency with words
    saliency_dict = dict(zip(text_expl["word"], text_expl["salience"]))
    saliency = np.array([saliency_dict.get(w, 0.0) for w in words])

    # Eye-tracking metrics
    text_eye = eye_df[eye_df["text_id"] == text_id]
    if text_eye.empty:
        # No eye-tracking data for this text
        return {
            "spearman_tfd": np.nan,
            "spearman_ffd": np.nan,
            "spearman_fc": np.nan,
            "overlap_span": np.nan,
            "n_words": n_words,
            "n_span_words": int(n_span_words),
        }

    eye_dict = dict(zip(text_eye["word"], text_eye["tfd"]))
    eye_tfd = np.array([eye_dict.get(w, 0.0) for w in words])
    eye_dict_ffd = dict(zip(text_eye["word"], text_eye["ffd"]))
    eye_ffd = np.array([eye_dict_ffd.get(w, 0.0) for w in words])
    eye_dict_fc = dict(zip(text_eye["word"], text_eye["fc"]))
    eye_fc = np.array([eye_dict_fc.get(w, 0.0) for w in words])

    # Spearman correlations (handle constant arrays)
    def safe_spearman(a, b):
        if np.std(a) == 0 or np.std(b) == 0:
            return np.nan
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            rho, p = spearmanr(a, b)
        return rho

    rho_tfd = safe_spearman(saliency, eye_tfd)
    rho_ffd = safe_spearman(saliency, eye_ffd)
    rho_fc = safe_spearman(saliency, eye_fc)

    # Overlap with spans: top-20% salient words vs span words
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
        "n_words": n_words,
        "n_span_words": int(n_span_words),
    }


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

    # Load chosen texts
    print("Loading chosen texts...")
    chosen_df = load_chosen_texts()

    # Load eye-tracking data
    eye_df = pd.DataFrame()
    if not args.no_eye_tracking:
        print("Loading eye-tracking data...")
        eye_df = load_eye_tracking_data()
        if not eye_df.empty:
            print(f"  Loaded eye-tracking for {eye_df.text_id.nunique()} texts")

    # Find explanation files
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

            if method_results:
                df = pd.DataFrame(method_results)
                # Aggregate across texts
                agg = {
                    "checkpoint": ckpt_name,
                    "method": method,
                    "spearman_tfd_mean": df["spearman_tfd"].mean(),
                    "spearman_ffd_mean": df["spearman_ffd"].mean(),
                    "spearman_fc_mean": df["spearman_fc"].mean(),
                    "overlap_span_mean": df["overlap_span"].mean(),
                    "n_texts": len(df),
                }
                print(f"    Spearman TFD: {agg['spearman_tfd_mean']:.3f}")
                print(f"    Spearman FFD: {agg['spearman_ffd_mean']:.3f}")
                print(f"    Spearman FC:  {agg['spearman_fc_mean']:.3f}")
                print(f"    Overlap span: {agg['overlap_span_mean']:.3f}")
                all_results.append(agg)

    if not all_results:
        print("No results to aggregate.")
        return

    results_df = pd.DataFrame(all_results)

    # Save CSV
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    csv_path = os.path.join(OUTPUT_DIR, "model_comparison.csv")
    results_df.to_csv(csv_path, index=False)
    print(f"\nSaved CSV: {csv_path}")

    # Generate LaTeX tables
    if args.output:
        latex_path = args.output
    else:
        latex_path = os.path.join(OUTPUT_DIR, "model_comparison.tex")

    # Table 1: Spearman correlations (model vs eye-tracking)
    spearman_table = results_df[
        [
            "checkpoint",
            "method",
            "spearman_tfd_mean",
            "spearman_ffd_mean",
            "spearman_fc_mean",
        ]
    ]
    spearman_latex = generate_latex_table(
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
        col_names=["Checkpoint", "Mètode", "ρ TFD", "ρ FFD", "ρ FC"],
    )

    # Table 2: Overlap with spans
    overlap_table = results_df[["checkpoint", "method", "overlap_span_mean"]]
    overlap_latex = generate_latex_table(
        overlap_table,
        caption="Solapament: paraules amb saliència més alta vs. paraules dins dels spans anotats.",
        label="tab:model_vs_spans",
        columns=["checkpoint", "method", "overlap_span_mean"],
        col_names=["Checkpoint", "Mètode", "Solapament"],
    )

    with open(latex_path, "w") as f:
        f.write("% Auto-generated by compare_results.py\n\n")
        f.write(spearman_latex)
        f.write("\n\n")
        f.write(overlap_latex)
        f.write("\n")

    print(f"Saved LaTeX: {latex_path}")


if __name__ == "__main__":
    main()
