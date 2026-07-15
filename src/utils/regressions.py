import time

import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import fisher_exact, spearmanr

from utils.tobii import (
    cluster_salient,
    compute_regression_metrics,
    detect_regressions,
    extract_fixations,
)


def bh_fdr(p_values):
    """Benjamini-Hochberg FDR correction (enforces monotonicity from largest to smallest)."""
    n = len(p_values)
    ranked = np.argsort(p_values)
    adjusted = np.zeros(n)
    for i, idx in enumerate(ranked):
        adjusted[idx] = min(p_values[idx] * n / (i + 1), 1.0)
    cummin = 1.0
    for i in range(n - 1, -1, -1):
        idx = ranked[i]
        cummin = min(cummin, adjusted[idx])
        adjusted[idx] = cummin
    return adjusted


def compute_all_regressions(text_dfs, text_ids):
    """Compute regressions for all participant × text pairs.

    Returns:
        all_regressions_df: DataFrame with one row per regression
        all_reg_metrics_df: DataFrame with summary metrics per (participant, text)
    """
    all_regressions = []
    all_reg_metrics = []

    for participant in list(text_dfs.keys()):
        for text_id in text_ids:
            if text_id not in text_dfs.get(participant, {}):
                continue
            df = text_dfs[participant][text_id]
            fixations = extract_fixations(df)
            regressions = detect_regressions(fixations)
            metrics = compute_regression_metrics(fixations, regressions)
            if len(regressions) > 0:
                regressions = regressions.copy()
                regressions["participant"] = participant
                regressions["text_id"] = text_id
                all_regressions.append(regressions)
            metrics["participant"] = participant
            metrics["text_id"] = text_id
            all_reg_metrics.append(metrics)

    all_regressions_df = (
        pd.concat(all_regressions, ignore_index=True) if all_regressions else pd.DataFrame()
    )
    all_reg_metrics_df = pd.DataFrame(all_reg_metrics)

    all_regressions_df["target_idx"] = (
        all_regressions_df["to_AOI"].str.split(".", n=1).str[0].astype(int)
    )
    all_regressions_df["source_idx"] = (
        all_regressions_df["from_AOI"].str.split(".", n=1).str[0].astype(int)
    )

    return all_regressions_df, all_reg_metrics_df


def compute_participant_baselines(
    text_dfs, text_ids, texts, all_reg_metrics_df, all_regressions_df, n_bins=10
):
    """Compute per-participant regression baselines including reading speed.

    Returns:
        participant_baselines: dict {participant: baseline_dict}
        mean_fd_all: global mean fixation duration
    """
    all_mean_fds = {}
    for participant in list(text_dfs.keys()):
        durations = []
        for text_id in text_ids:
            if text_id not in text_dfs.get(participant, {}):
                continue
            df = text_dfs[participant][text_id]
            fix = df[(df["event_type"] == "Fixation") & df["AOI"].notna()]
            durations.extend(fix["duration"].dropna().tolist())
        all_mean_fds[participant] = np.mean(durations) if durations else 0

    mean_fd_all = np.mean(list(all_mean_fds.values()))

    participant_baselines = {}
    for participant in list(text_dfs.keys()):
        p_mets = all_reg_metrics_df[all_reg_metrics_df["participant"] == participant]
        p_regs = all_regressions_df[all_regressions_df["participant"] == participant]

        total_saccades = p_mets["total_saccades"].sum()
        total_regressions = p_mets["total_regressions"].sum()
        reg_rate = total_regressions / total_saccades if total_saccades > 0 else 0

        target_positions = []
        source_positions = []
        for _, row in p_regs.iterrows():
            n_tok = len(texts[row["text_id"]].split())
            target_positions.append(row["target_idx"] / max(n_tok - 1, 1))
            source_positions.append(row["source_idx"] / max(n_tok - 1, 1))

        target_hist, _ = np.histogram(target_positions, bins=n_bins, range=(0, 1))
        target_probs = target_hist / max(target_hist.sum(), 1)

        source_hist, _ = np.histogram(source_positions, bins=n_bins, range=(0, 1))
        source_probs = source_hist / max(source_hist.sum(), 1)

        participant_baselines[participant] = {
            "regression_rate": reg_rate,
            "total_regressions": total_regressions,
            "total_saccades": total_saccades,
            "mean_dAOI": p_regs["dAOI"].mean() if len(p_regs) > 0 else 0,
            "std_dAOI": p_regs["dAOI"].std() if len(p_regs) > 0 else 0,
            "target_probs": target_probs,
            "source_probs": source_probs,
            "mean_fd": all_mean_fds[participant],
        }

    return participant_baselines, mean_fd_all


def compute_zscores(
    text_ids,
    texts,
    participants,
    text_dfs,
    all_regressions_df,
    all_reg_metrics_df,
    participant_baselines,
    mean_fd_all,
    n_bins=10,
    direction="target",
):
    """Compute z-scores for regression targets or sources across all texts.

    Args:
        direction: "target" or "source" — which regression endpoint to analyze

    Returns:
        zscore_df: DataFrame with text_id, token_idx, token, observed, expected, z_score, n_participants, p_value, q_value
        salient_df: subset where q_value < 0.05
    """
    idx_col = "target_idx" if direction == "target" else "source_idx"
    prob_key = "target_probs" if direction == "target" else "source_probs"

    records = []
    for text_id in text_ids:
        n_tokens = len(texts[text_id].split())
        participants_reading = [p for p in participants if text_id in text_dfs.get(p, {})]
        for token_idx in range(n_tokens):
            obs_total = 0
            exp_total = 0
            for participant in participants_reading:
                obs = len(
                    all_regressions_df[
                        (all_regressions_df["participant"] == participant)
                        & (all_regressions_df["text_id"] == text_id)
                        & (all_regressions_df[idx_col] == token_idx)
                    ]
                )

                bl = participant_baselines[participant]
                rel_pos = token_idx / max(n_tokens - 1, 1)
                bin_idx = min(int(rel_pos * n_bins), n_bins - 1)

                n_sacc = all_reg_metrics_df[
                    (all_reg_metrics_df["participant"] == participant)
                    & (all_reg_metrics_df["text_id"] == text_id)
                ]["total_saccades"].values
                n_sacc = n_sacc[0] if len(n_sacc) > 0 else 0

                speed_factor = bl["mean_fd"] / mean_fd_all
                exp = bl["regression_rate"] * n_sacc * bl[prob_key][bin_idx] * speed_factor

                obs_total += obs
                exp_total += exp

            z = (obs_total - exp_total) / np.sqrt(max(exp_total, 1e-6))
            records.append(
                {
                    "text_id": text_id,
                    "token_idx": token_idx,
                    "token": texts[text_id].split()[token_idx],
                    "observed": obs_total,
                    "expected": round(exp_total, 2),
                    "z_score": round(z, 3),
                    "n_participants": len(participants_reading),
                }
            )

    zscore_df = pd.DataFrame(records)
    zscore_df["p_value"] = 2 * (1 - stats.norm.cdf(np.abs(zscore_df["z_score"])))
    zscore_df["q_value"] = bh_fdr(zscore_df["p_value"].values)

    salient_df = zscore_df[zscore_df["q_value"] < 0.05].sort_values(
        "z_score", ascending=False
    )
    return zscore_df, salient_df


def precompute_aoi_arrays(text_dfs, text_ids):
    """Pre-compute AOI index arrays for fast permutation shuffling.

    Returns:
        pre_arrays: dict {(participant, text_id): np.array of AOI indices}
        text_ids_with_data: list of text_ids that have at least one participant
    """
    pre_arrays = {}
    for participant in list(text_dfs.keys()):
        for text_id in text_ids:
            if text_id not in text_dfs.get(participant, {}):
                continue
            fixations = extract_fixations(text_dfs[participant][text_id])
            fix_with_aoi = fixations.dropna(subset=["AOI"])
            aoi_idx = fix_with_aoi["AOI"].str.split(".", n=1).str[0].astype(int).values
            pre_arrays[(participant, text_id)] = aoi_idx

    text_ids_with_data = [
        t for t in text_ids if any((p, t) in pre_arrays for p in list(text_dfs.keys()))
    ]
    return pre_arrays, text_ids_with_data


def run_permutation_test(text_ids_with_data, texts, text_dfs, pre_arrays, n_perms=10000):
    """Run permutation test for target and source regression counts.

    Returns:
        obs_target_counts: dict {text_id: np.array}
        obs_source_counts: dict {text_id: np.array}
        null_target_counts: dict {text_id: list of np.arrays}
        null_source_counts: dict {text_id: list of np.arrays}
    """
    # Observed counts
    obs_target_counts = {}
    obs_source_counts = {}
    for text_id in text_ids_with_data:
        n_tokens = len(texts[text_id].split())
        target_counts = np.zeros(n_tokens)
        source_counts = np.zeros(n_tokens)
        for participant in list(text_dfs.keys()):
            key = (participant, text_id)
            if key not in pre_arrays:
                continue
            aoi_idx = pre_arrays[key]
            dAOI = np.diff(aoi_idx)
            reg_mask = dAOI < 0
            targets = aoi_idx[1:][reg_mask]
            sources = aoi_idx[:-1][reg_mask]
            for t_idx in targets:
                if 0 <= t_idx < n_tokens:
                    target_counts[t_idx] += 1
            for s_idx in sources:
                if 0 <= s_idx < n_tokens:
                    source_counts[s_idx] += 1
        obs_target_counts[text_id] = target_counts
        obs_source_counts[text_id] = source_counts

    # Null distributions
    null_target_counts = {t: [] for t in text_ids_with_data}
    null_source_counts = {t: [] for t in text_ids_with_data}

    t_start = time.time()
    for perm in range(n_perms):
        for text_id in text_ids_with_data:
            n_tokens = len(texts[text_id].split())
            target_counts = np.zeros(n_tokens)
            source_counts = np.zeros(n_tokens)
            for participant in list(text_dfs.keys()):
                key = (participant, text_id)
                if key not in pre_arrays:
                    continue
                aoi_idx = pre_arrays[key]
                perm_aoi_idx = aoi_idx[np.random.permutation(len(aoi_idx))]
                dAOI = np.diff(perm_aoi_idx)
                reg_mask = dAOI < 0
                targets = perm_aoi_idx[1:][reg_mask]
                sources = perm_aoi_idx[:-1][reg_mask]
                for t_idx in targets:
                    if 0 <= t_idx < n_tokens:
                        target_counts[t_idx] += 1
                for s_idx in sources:
                    if 0 <= s_idx < n_tokens:
                        source_counts[s_idx] += 1
            null_target_counts[text_id].append(target_counts)
            null_source_counts[text_id].append(source_counts)
        if (perm + 1) % 1000 == 0:
            elapsed = time.time() - t_start
            rate = (perm + 1) / elapsed
            remaining = (n_perms - perm - 1) / rate
            print(
                f"  Perm {perm + 1}/{n_perms}: {elapsed:.1f}s elapsed, ~{remaining:.0f}s remaining"
            )

    elapsed_total = time.time() - t_start
    print(f"\nPermutation test done in {elapsed_total:.1f}s")
    return obs_target_counts, obs_source_counts, null_target_counts, null_source_counts


def compute_permutation_pvalues(
    text_ids_with_data,
    obs_target_counts,
    obs_source_counts,
    null_target_counts,
    null_source_counts,
):
    """Compute permutation p-values with BH correction.

    Returns:
        perm_target_qvalues: dict {text_id: np.array of q-values}
        perm_source_qvalues: dict {text_id: np.array of q-values}
        perm_sig_targets: int count of significant target tokens
        perm_sig_sources: int count of significant source tokens
    """
    perm_target_pvalues = {}
    perm_source_pvalues = {}
    for text_id in text_ids_with_data:
        null_arr = np.array(null_target_counts[text_id])
        obs = obs_target_counts[text_id]
        perm_target_pvalues[text_id] = (null_arr >= obs[np.newaxis, :]).mean(axis=0)

        null_arr = np.array(null_source_counts[text_id])
        obs = obs_source_counts[text_id]
        perm_source_pvalues[text_id] = (null_arr >= obs[np.newaxis, :]).mean(axis=0)

    all_pvals_target = np.concatenate([perm_target_pvalues[t] for t in text_ids_with_data])
    all_qvals_target = bh_fdr(all_pvals_target)
    perm_target_qvalues = {}
    offset = 0
    for text_id in text_ids_with_data:
        n = len(perm_target_pvalues[text_id])
        perm_target_qvalues[text_id] = all_qvals_target[offset : offset + n]
        offset += n

    all_pvals_source = np.concatenate([perm_source_pvalues[t] for t in text_ids_with_data])
    all_qvals_source = bh_fdr(all_pvals_source)
    perm_source_qvalues = {}
    offset = 0
    for text_id in text_ids_with_data:
        n = len(perm_source_pvalues[text_id])
        perm_source_qvalues[text_id] = all_qvals_source[offset : offset + n]
        offset += n

    perm_sig_targets = sum(
        1 for t in text_ids_with_data for q in perm_target_qvalues[t] if q < 0.05
    )
    perm_sig_sources = sum(
        1 for t in text_ids_with_data for q in perm_source_qvalues[t] if q < 0.05
    )
    return perm_target_qvalues, perm_source_qvalues, perm_sig_targets, perm_sig_sources


def compare_zscore_permutation(
    salient_target_tokens,
    salient_source_tokens,
    text_ids_with_data,
    perm_target_qvalues,
    perm_source_qvalues,
    zscore_target_df,
    zscore_source_df,
    texts,
):
    """Compare z-score vs permutation significant tokens and cluster into hotspots.

    Returns:
        all_sig_tokens: set of (text_id, token_idx) significant by either method
        hotspots_df: DataFrame with regression hotspots per text
    """
    z_target_sig = set(
        zip(salient_target_tokens["text_id"], salient_target_tokens["token_idx"])
    )
    perm_target_sig = set()
    for text_id in text_ids_with_data:
        for idx, q in enumerate(perm_target_qvalues[text_id]):
            if q < 0.05:
                perm_target_sig.add((text_id, idx))

    z_source_sig = set(
        zip(salient_source_tokens["text_id"], salient_source_tokens["token_idx"])
    )
    perm_source_sig = set()
    for text_id in text_ids_with_data:
        for idx, q in enumerate(perm_source_qvalues[text_id]):
            if q < 0.05:
                perm_source_sig.add((text_id, idx))

    all_sig_tokens = z_target_sig | perm_target_sig | z_source_sig | perm_source_sig

    # Cluster into hotspots
    hotspot_records = []
    for text_id in text_ids_with_data:
        sig_target_z = {}
        sig_source_z = {}
        for text_id_s, tok_idx in all_sig_tokens:
            if text_id_s != text_id:
                continue
            zt = zscore_target_df[
                (zscore_target_df["text_id"] == text_id)
                & (zscore_target_df["token_idx"] == tok_idx)
            ]["z_score"].values
            zs = zscore_source_df[
                (zscore_source_df["text_id"] == text_id)
                & (zscore_source_df["token_idx"] == tok_idx)
            ]["z_score"].values
            if len(zt) > 0:
                sig_target_z[tok_idx] = float(zt[0])
            if len(zs) > 0:
                sig_source_z[tok_idx] = float(zs[0])

        if sig_target_z:
            peaks = cluster_salient(list(sig_target_z.keys()), list(sig_target_z.values()))
            for p in peaks:
                hotspot_records.append(
                    {
                        "text_id": text_id,
                        "direction": "TO",
                        "token": texts[text_id].split()[p["peak_position"]],
                        "position": p["peak_position"],
                        "z_score": p["peak_z"],
                        "cluster_size": p["cluster_size"],
                    }
                )

        if sig_source_z:
            peaks = cluster_salient(list(sig_source_z.keys()), list(sig_source_z.values()))
            for p in peaks:
                hotspot_records.append(
                    {
                        "text_id": text_id,
                        "direction": "FROM",
                        "token": texts[text_id].split()[p["peak_position"]],
                        "position": p["peak_position"],
                        "z_score": p["peak_z"],
                        "cluster_size": p["cluster_size"],
                    }
                )

    hotspots_df = pd.DataFrame(hotspot_records)
    return all_sig_tokens, hotspots_df


def compute_fisher_overlap(all_sig_tokens, annotations, texts):
    """Compute overlap of significant tokens with annotated spans using Fisher's exact test.

    Returns:
        overlap_df: DataFrame with token-level overlap info
        fisher_df: DataFrame with per-text Fisher's exact test results
    """
    all_sig_by_text = {}
    for text_id, token_idx in all_sig_tokens:
        all_sig_by_text.setdefault(text_id, set()).add(token_idx)

    overlap_records = []
    fisher_results = []

    for text_id in sorted(all_sig_by_text):
        if text_id not in annotations:
            continue
        ann = annotations[text_id]
        n_tokens = len(ann)
        sig_toks = all_sig_by_text[text_id]

        a = sum(1 for i in sig_toks if i < n_tokens and ann[i] > 0)
        b = sum(1 for i in sig_toks if i < n_tokens and ann[i] == 0)
        not_sig = set(range(n_tokens)) - sig_toks
        c = sum(1 for i in not_sig if i < n_tokens and ann[i] > 0)
        d = sum(1 for i in not_sig if i < n_tokens and ann[i] == 0)

        for i in sorted(sig_toks):
            if i < n_tokens:
                overlap_records.append(
                    {
                        "text_id": text_id,
                        "token_idx": i,
                        "token": texts[text_id].split()[i],
                        "in_span": ann[i] > 0,
                        "annotation_value": round(float(ann[i]), 3),
                    }
                )

        if (a + b) > 0 and (c + d) > 0:
            odds, p = fisher_exact([[a, b], [c, d]], alternative="greater")
            fisher_results.append(
                {
                    "text_id": text_id,
                    "a_sig_in_span": a,
                    "b_sig_not_in_span": b,
                    "c_not_sig_in_span": c,
                    "d_not_sig_not_in_span": d,
                    "odds_ratio": odds,
                    "p_value": p,
                }
            )

    overlap_df = pd.DataFrame(overlap_records)
    fisher_df = pd.DataFrame(fisher_results)
    return overlap_df, fisher_df


def compute_span_coverage(
    text_ids, texts, data, per_label_annotations, all_reg_metrics_df, text_dfs, label_cols
):
    """Compute span coverage vs regression rate per label type.

    Returns:
        coverage_df: DataFrame with text_id, label, coverage, has_label, mean_regression_rate
        spearman_df: DataFrame with label, rho, p, n_texts, n_with_coverage
    """
    tag_map = {
        "sexist": None,
        "irony": "IRONY",
        "humor": "JOKE",
        "implicit sexism": "IMPLICIT SEXISM",
        "stereotypes": "STEREOTYPE",
        "inequality": "INEQUALITY/DISCRIMINATION",
        "discrimination": "INEQUALITY/DISCRIMINATION",
        "objectification": "OBJECTIFICATION",
        "critique": "CRITIQUE",
        "reported_sexism": "REPORTED SEXISM",
    }

    coverage_records = []
    for text_id in text_ids:
        for col in label_cols:
            if col not in data.columns:
                continue
            row = data[data["num_id"] == text_id]
            if len(row) == 0:
                continue
            has_label = row[col].values[0] == 1

            tag = tag_map.get(col)
            if (
                tag
                and tag in per_label_annotations
                and text_id in per_label_annotations[tag]
            ):
                coverage = per_label_annotations[tag][text_id].mean()
            else:
                coverage = 1.0 if has_label else 0.0

            rates = []
            for participant in list(text_dfs.keys()):
                m = all_reg_metrics_df[
                    (all_reg_metrics_df["participant"] == participant)
                    & (all_reg_metrics_df["text_id"] == text_id)
                ]
                if len(m) > 0:
                    rates.append(m["regression_rate"].values[0])

            coverage_records.append(
                {
                    "text_id": text_id,
                    "label": col,
                    "coverage": coverage,
                    "has_label": has_label,
                    "mean_regression_rate": np.mean(rates) if rates else 0,
                }
            )

    coverage_df = pd.DataFrame(coverage_records)

    spearman_results = []
    for col in label_cols:
        if col not in coverage_df["label"].values:
            continue
        sub = coverage_df[coverage_df["label"] == col]
        if sub["coverage"].std() == 0:
            spearman_results.append(
                {
                    "label": col,
                    "rho": 0,
                    "p": 1,
                    "n_texts": len(sub),
                    "n_with_coverage": (sub["coverage"] > 0).sum(),
                }
            )
            continue
        rho, p = spearmanr(sub["coverage"], sub["mean_regression_rate"])
        spearman_results.append(
            {
                "label": col,
                "rho": round(rho, 3),
                "p": round(p, 4),
                "n_texts": len(sub),
                "n_with_coverage": (sub["coverage"] > 0).sum(),
            }
        )

    spearman_df = pd.DataFrame(spearman_results).sort_values("p")
    return coverage_df, spearman_df


def compute_span_hotspots(per_label_annotations, zscore_target_df, zscore_source_df, texts):
    """Find salient regression hotspots within annotated spans, weighted by 1/span_length.

    Returns:
        span_hotspots_df: DataFrame with label, text_id, direction, token, position, z_score, cluster_size, span_length, weight
    """
    span_hotspot_results = []

    for tag in sorted(per_label_annotations.keys()):
        for text_id, tag_ann in per_label_annotations[tag].items():
            in_tok_indices = np.where(tag_ann > 0)[0]
            if len(in_tok_indices) == 0:
                continue

            span_length = len(in_tok_indices)
            weight = 1.0 / span_length

            for zdf, direction in [(zscore_target_df, "TO"), (zscore_source_df, "FROM")]:
                text_z = zdf[zdf["text_id"] == text_id]
                span_tokens = text_z[text_z["token_idx"].isin(in_tok_indices)]

                if len(span_tokens) == 0:
                    continue

                peaks = cluster_salient(
                    span_tokens["token_idx"].tolist(),
                    span_tokens["z_score"].tolist(),
                    max_gap=3,
                )

                for p in peaks:
                    span_hotspot_results.append(
                        {
                            "label": tag,
                            "text_id": text_id,
                            "direction": direction,
                            "token": texts[text_id].split()[p["peak_position"]],
                            "position": p["peak_position"],
                            "z_score": p["peak_z"],
                            "cluster_size": p["cluster_size"],
                            "span_length": span_length,
                            "weight": weight,
                        }
                    )

    return pd.DataFrame(span_hotspot_results)


def get_top_hotspots_per_text(
    zscore_target_df,
    zscore_source_df,
    salient_target_tokens,
    salient_source_tokens,
    texts,
    text_ids,
    n_top=10,
):
    """Get top N significant regression tokens per text (TO and FROM combined).

    Filters stopwords and reports the percentage.

    Returns:
        hotspots_df: DataFrame with text_id, direction, token, position, z_score, is_stopword
        stopword_pct: float, percentage of significant tokens that are stopwords
    """
    import nltk

    nltk.download("stopwords", quiet=True)
    es_stopwords = set(nltk.corpus.stopwords.words("spanish"))

    records = []
    for text_id in text_ids:
        tokens = texts[text_id].split()

        # Merge target and source significant tokens for this text
        tgt = salient_target_tokens[salient_target_tokens["text_id"] == text_id][
            ["token_idx", "z_score"]
        ].copy()
        tgt["direction"] = "TO"
        src = salient_source_tokens[salient_source_tokens["text_id"] == text_id][
            ["token_idx", "z_score"]
        ].copy()
        src["direction"] = "FROM"

        combined = pd.concat([tgt, src], ignore_index=True)
        if len(combined) == 0:
            continue

        # Merge TO and FROM for the same token: keep the one with higher |z|
        combined["abs_z"] = combined["z_score"].abs()
        combined = combined.sort_values("abs_z", ascending=False).drop_duplicates(
            subset="token_idx", keep="first"
        )

        # Add token text and stopword flag
        combined["token"] = combined["token_idx"].apply(
            lambda i: tokens[i] if i < len(tokens) else "?"
        )
        combined["is_stopword"] = (
            combined["token"].str.lower().str.strip(".,;:!?()\"'¿¡").isin(es_stopwords)
        )
        combined["text_id"] = text_id

        records.append(
            combined[
                ["text_id", "direction", "token", "token_idx", "z_score", "is_stopword"]
            ]
        )

    if not records:
        return pd.DataFrame(), 0.0

    all_hotspots = pd.concat(records, ignore_index=True)

    # Compute stopword percentage across ALL significant tokens
    total_sig = len(all_hotspots)
    total_sw = all_hotspots["is_stopword"].sum()
    stopword_pct = 100 * total_sw / max(total_sig, 1)

    # For each text, take top n_top non-stopword tokens (if available), else top n_top overall
    top_records = []
    for text_id in text_ids:
        text_all = all_hotspots[all_hotspots["text_id"] == text_id].copy()
        if len(text_all) == 0:
            continue

        # Prefer lexical words
        lexical = text_all[~text_all["is_stopword"]].head(n_top)
        if len(lexical) < n_top:
            # Fill remaining with stopwords if needed
            remaining = text_all[text_all["is_stopword"]].head(n_top - len(lexical))
            lexical = pd.concat([lexical, remaining])

        top_records.append(lexical)

    hotspots_df = (
        pd.concat(top_records, ignore_index=True) if top_records else pd.DataFrame()
    )
    return hotspots_df, stopword_pct
