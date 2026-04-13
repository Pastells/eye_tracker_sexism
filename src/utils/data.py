import warnings
from pathlib import Path
import os

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import numpy as np
import pandas as pd


def z_score(df, col, participant_col):
    """Compute z-score of col for each participant"""
    stats = df[[participant_col, col]].groupby(participant_col)[col].agg(["mean", "std"])
    df = df.merge(stats, on=participant_col, suffixes=("", "_stats"))
    df[col + " z-score"] = (df[col] - df["mean"]) / df["std"]
    df = df.drop(columns=["mean", "std"])
    return df


def get_hits(df, aoi_hit, part_col, normalize=True, max_texts=None) -> dict:
    text_hits = {}
    if max_texts is None:
        texts = {t.split(" - ")[1].rsplit("]", 1)[0] for t in aoi_hit}
    else:
        texts = {t.split(" - ")[1].rsplit("]", 1)[0] for t in aoi_hit[:max_texts]}

    for text in texts:
        aoi_hit_text = [col for col in aoi_hit if f"{text}" in col]
        text_hits[text] = get_hits_text(df, aoi_hit_text, part_col, normalize=normalize)
    return text_hits


def get_hits_text(df, aoi_hit_text, part_col, normalize=True) -> dict:
    res = {}
    participants = df[part_col].unique().tolist()
    res["tokens"] = [t.split("- ")[1].split("]")[0] for t in aoi_hit_text]
    hits = df[[part_col] + aoi_hit_text].groupby(part_col).sum()[aoi_hit_text]
    hits = hits.to_numpy()
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        if normalize:
            try:
                hits = hits / hits.sum(axis=1, keepdims=True)
            except (RuntimeWarning, ZeroDivisionError):
                hits = 0
    res["hits"] = hits

    z_pupil = []
    for token in aoi_hit_text:
        hit_1 = df[df[token] == 1]["Pupil diameter filtered z-score"]
        z = 0 if hit_1.empty else hit_1.mean()
        z_pupil.append(z)
    res["z_pupil"] = np.array(z_pupil)

    # Create a MultiIndex for all combinations of participants and tokens
    idx = pd.MultiIndex.from_product([participants, aoi_hit_text], names=[part_col, "token"])

    # Calculate z_pupil for each participant and token
    hit_mask = df[aoi_hit_text] == 1
    filtered_df = df.loc[
        hit_mask.any(axis=1), [part_col, "Pupil diameter filtered z-score"] + aoi_hit_text
    ]
    z_pupil_df = (
        filtered_df.melt(
            id_vars=[part_col, "Pupil diameter filtered z-score"],
            value_vars=aoi_hit_text,
            var_name="token",
        )
        .query("value == 1")
        .groupby([part_col, "token"])["Pupil diameter filtered z-score"]
        .mean()
    )
    z_pupil_df = z_pupil_df.reindex(idx, fill_value=0)
    z_pupil = z_pupil_df.unstack().to_numpy()
    res["z_pupil"] = z_pupil
    return res


def convert_to_parquet(tsv_file):
    tsv_path = Path("../data", tsv_file + ".tsv")
    parquet_file = tsv_path.with_suffix(".parquet")
    if not os.path.exists(parquet_file):
        print("creant parquet i tsv amb només primera línia")
        df = pd.read_csv(tsv_path, sep="\t", low_memory=False)
        df.to_parquet(parquet_file, index=False)
        # os.rename(tsv_path, tsv_path.with_stem(tsv_file + "_original"))
        # df.iloc[:1].to_csv(tsv_path, sep="\t", index=False)
    return parquet_file


def read_data(tsv_file):
    parquet_file = convert_to_parquet(tsv_file)

    cols = pq.read_schema(parquet_file).names
    aoi_size = [col for col in cols if "AOI size" in col]
    aoi_hit = [col for col in cols if "AOI hit" in col]
    calibration_cols = [col for col in cols if "calibration" in col] + [
        col for col in cols if "validation" in col
    ]
    unwanted_cols = aoi_size + [
        "Client area position X (DACSpx)",
        "Client area position Y (DACSpx)",
        "Viewport position X",
        "Viewport position Y",
        "Viewport width",
        "Viewport height",
        "Full page width",
        "Full page height",
        "Mouse position X",
        "Mouse position Y",
        "Sensor",
        "Project name",
        "Export date",
        "Recording name",
        "Recording date",
        "Recording date UTC",
        "Recording start time",
        "Recording start time UTC",
        "Recording duration",
        "Timeline name",
        "Recording Fixation filter name",
        "Recording software version",
        "Recording resolution height",
        "Recording resolution width",
        "Recording monitor latency",
        "Presented Media width",
        "Presented Media height",
        "Presented Media position X (DACSpx)",
        "Presented Media position Y (DACSpx)",
        "Original Media width",
        "Original Media height",
    ]
    cols_to_keep = [col for col in cols if col not in unwanted_cols]

    column_mapping = {
        "Recording timestamp": "timestamp",
        "Gaze point X": "gaze_x",
        "Gaze point Y": "gaze_y",
        "Eye movement type": "event_type",  # Fixation, Saccade, etc.
        "Eye movement type index": "event_index",
        "Fixation point X": "fix_x",
        "Fixation point Y": "fix_y",
        "Eye movement event duration": "duration",
        "Participant name": "participant",
    }
    raw_df = pd.read_parquet(
        parquet_file,
        engine="pyarrow",
        dtype_backend="numpy_nullable",
        columns=cols_to_keep,
    ).rename(columns={k: v for k, v in column_mapping.items()})

    # optimize memory a bit
    raw_df[aoi_hit] = raw_df[aoi_hit].astype(pd.BooleanDtype())
    object_columns = raw_df.select_dtypes(include="string").columns
    raw_df[object_columns] = raw_df[object_columns].astype("category")
    float_columns = raw_df.select_dtypes(include="float").columns
    raw_df[float_columns] = raw_df[float_columns].astype(pd.Float32Dtype())

    if not raw_df[calibration_cols].empty:
        calibration_df = raw_df[["participant"] + calibration_cols].drop_duplicates()
        raw_df = raw_df.drop(calibration_cols, axis=1)
    else:
        calibration_df = None

    participants = raw_df["participant"].unique().tolist()

    # raw_df = raw_df.dropna(subset=["gaze_x", "gaze_y"])
    # drop [nan, 'Unclassified', 'EyesNotFound']
    df = raw_df[raw_df["event_type"].isin(("Fixation", "Saccade"))].copy()
    return aoi_hit, calibration_df, participants, df


# ======================================
# process raw data into list of dfs
# ======================================


def split_by_text(df):
    """
    Split the main DataFrame into a list of DataFrames,
    one per unique stimulus in 'Presented Stimulus name'.
    Preserves the order of appearance.
    """
    text_dfs = {}

    # Get unique stimulus names in order of appearance, excluding NaN
    stimuli = set(df["Presented Stimulus name"].dropna().unique()) - {
        "Text",
        "photo-ground-texture-pattern",
    }

    for stimulus in stimuli:
        text_df = df[df["Presented Stimulus name"] == stimulus].copy()
        text_df.attrs["stimulus_name"] = stimulus
        text_dfs[stimulus] = text_df

    return text_dfs


def drop_irrelevant_aois(text_df, stimulus, aoi_hit):
    """Keep columns for an AOI"""

    def clean_affixes(string, idx):
        string = f"{idx}." + string.removeprefix("AOI hit [" + stimulus + " - ")
        if string[-1] == "]":
            return string[:-1]
        if string[-2].isnumeric():
            return string[:-4] + string[-3:]
        return string[:-3] + string[-2:]

    aois_to_keep = [a for a in aoi_hit if a.startswith("AOI hit [" + stimulus)]
    aois_to_keep = {a: clean_affixes(a, idx) for idx, a in enumerate(aois_to_keep)}
    cols_to_drop = set(aoi_hit) - set(aois_to_keep.keys())
    text_df = text_df.drop(columns=cols_to_drop).rename(columns=aois_to_keep)
    return text_df, list(aois_to_keep.values())


def collapse_aoi_columns(text_df, aoi_cols):
    """
    Collapse AOI hit columns into a single 'AOI' column (vectorized).
    """
    aoi_data = text_df[aoi_cols].fillna(0).astype(int)

    # Check for simultaneous hits
    hits_per_row = aoi_data.sum(axis=1)
    multi_hits = hits_per_row[hits_per_row > 1]

    if len(multi_hits) > 0:
        stimulus = text_df.attrs.get("stimulus_name", "?")
        print(f"  ⚠ '{stimulus}': {len(multi_hits)} rows have multiple AOI hits!")

        unique_overlaps = set()
        for idx in multi_hits.index:
            hit_aois = tuple(col for col in aoi_cols if aoi_data.loc[idx, col] == 1)
            unique_overlaps.add(hit_aois)
        for overlap in unique_overlaps:
            print(f"    Overlapping: {list(overlap)}")

    text_df = text_df.copy()
    has_hit = hits_per_row >= 1
    text_df["AOI"] = None
    text_df.loc[has_hit, "AOI"] = aoi_data.loc[has_hit].idxmax(axis=1)
    text_df = text_df.drop(columns=aoi_cols)

    n_hits = text_df["AOI"].notna().sum()
    n_unique = text_df["AOI"].nunique()
    print(f"  {n_hits} samples with AOI hits across {n_unique} unique AOIs")

    return text_df


def process_all_texts(df, aoi_hit):
    """
    Full pipeline: split by text, then clean AOI columns per text.
    Returns a list of clean DataFrames.
    """
    text_dfs = split_by_text(df)
    aoi_cols_dict = {}

    for stimulus, text_df in text_dfs.items():
        text_df, aoi_cols = drop_irrelevant_aois(text_df, stimulus, aoi_hit)
        text_dfs[stimulus] = collapse_aoi_columns(text_df, aoi_cols)
        aoi_cols_dict[stimulus] = aoi_cols

    return text_dfs, aoi_cols_dict


# ======================================
# fixations and regressions
# ======================================


def extract_fixations(df):
    """
    Extract fixation events from Tobii data.
    Drops first fixation
    """
    if "event_type" in df.columns:
        # Tobii already classified events
        fixations = df[df["event_type"] == "Fixation"].dropna(subset=["AOI"]).copy()

        # Group by fixation index to get one row per fixation
        fixations = (
            fixations.groupby("event_index")
            .agg(
                {
                    "timestamp": "first",
                    "fix_x": "mean",
                    "fix_y": "mean",
                    "duration": "first",
                    "AOI": "first",
                }
            )
            .reset_index()
        ).rename(columns={"fix_x": "x", "fix_y": "y"})
    else:
        # TODO: check if it happens
        # Use raw gaze points as-is (consider applying I-VT or I-DT filter)
        fixations = df[["timestamp", "gaze_x", "gaze_y"]].copy()
        fixations.rename(columns={"gaze_x": "x", "gaze_y": "y"}, inplace=True)

    fixations = fixations.sort_values("timestamp").reset_index(drop=True).drop([0])
    return fixations


def detect_regressions(fixations, line_height_threshold=50):
    fix = fixations.dropna(subset=["AOI"]).copy()
    fix["aoi_idx"] = fix["AOI"].str.split(".", n=1).str[0].astype(int)

    fix["dAOI"] = fix["aoi_idx"].diff()
    fix["dx"] = fix["x"].diff()
    fix["dy"] = fix["y"].diff()

    reg_mask = fix["dAOI"] < 0
    reg = fix[reg_mask].copy()
    reg["from_AOI"] = fix["AOI"].shift(1).loc[reg_mask]
    reg["to_AOI"] = fix["AOI"].loc[reg_mask]
    reg["timestamp_from"] = fix["timestamp"].shift(1).loc[reg_mask]
    reg["timestamp_to"] = fix["timestamp"].loc[reg_mask]
    reg["saccade_length"] = np.sqrt(reg["dx"] ** 2 + reg["dy"] ** 2)
    reg["regression_type"] = np.where(
        reg["dy"].abs() < line_height_threshold, "within-line", "between-line"
    )

    return reg[
        [
            "timestamp_from",
            "timestamp_to",
            "from_AOI",
            "to_AOI",
            "dAOI",
            "dy",
            "saccade_length",
            "regression_type",
        ]
    ].reset_index(drop=True)


def compute_regression_metrics(fixations, regressions):
    """Compute summary statistics about regressions."""
    total_fixations = len(fixations)
    total_saccades = total_fixations - 1
    total_regressions = len(regressions)

    metrics = {
        "total_fixations": total_fixations,
        "total_saccades": total_saccades,
        "total_regressions": total_regressions,
        "regression_rate": total_regressions / total_saccades if total_saccades > 0 else 0,
        "within_line_regressions": (
            len(regressions[regressions["regression_type"] == "within-line"])
            if len(regressions) > 0
            else 0
        ),
        "between_line_regressions": (
            len(regressions[regressions["regression_type"] == "between-line"])
            if len(regressions) > 0
            else 0
        ),
        "mean_regression_distance": regressions["saccade_length"].mean()
        if len(regressions) > 0
        else 0,
        "max_regression_distance": regressions["saccade_length"].max()
        if len(regressions) > 0
        else 0,
    }

    return metrics
