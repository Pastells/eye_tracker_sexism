import glob
import os
import re
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq


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
    idx = pd.MultiIndex.from_product(
        [participants, aoi_hit_text], names=[part_col, "token"]
    )

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


def read_data(tsv_file: str):
    if tsv_file.endswith(".parquet"):
        parquet_file = tsv_file
    else:
        parquet_file = convert_to_parquet(tsv_file)

    cols = pq.read_schema(parquet_file).names
    aoi_size = [col for col in cols if "AOI size" in col]
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
        **{x: x.replace("76)", "76") for x in cols_to_keep if x.startswith("AOI hit [76)")},
    }

    raw_df = pd.read_parquet(
        parquet_file,
        engine="pyarrow",
        dtype_backend="numpy_nullable",
        columns=cols_to_keep,
    ).rename(columns=column_mapping)

    # Compute aoi_hit after rename (76) -> 76)
    aoi_hit = [col for col in raw_df.columns if "AOI hit" in col]

    # optimize memory a bit
    raw_df[aoi_hit] = raw_df[aoi_hit].astype(pd.BooleanDtype())
    object_columns = raw_df.select_dtypes(include="string").columns
    raw_df[object_columns] = raw_df[object_columns].astype("category")
    float_columns = raw_df.select_dtypes(include="float").columns
    raw_df[float_columns] = raw_df[float_columns].astype(pd.Float32Dtype())
    raw_df["Presented Stimulus name"] = (
        raw_df["Presented Stimulus name"]
        .map({"76)": "76"})
        .fillna(raw_df["Presented Stimulus name"])
        .astype(int)
    )

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


def read_all_data(parquet_folder: str = "all_parquets", n_debug=False):
    """
    Reads all parquet files from a folder by calling `read_data` on each one
    Returns lists aggregated across all files.

    Returns:
        flat_aoi_hit: deduplicated flat list of all AOI hit column names
        all_calibration_dfs: list of calibration DataFrames (one per file),
                             if exported seperately it will be None
        all_participants: list of participant lists (one per file)
        all_dfs: list of filtered DataFrames (one per file)
    """
    parquet_files = sorted(glob.glob(os.path.join(parquet_folder, "*.parquet")))
    if n_debug:
        parquet_files = parquet_files[:n_debug]

    all_aoi_hit = []
    all_calibration_dfs = []
    all_participants = []
    all_dfs = []

    for parquet_file in parquet_files:
        # `read_data` expects a tsv path, but internally calls `convert_to_parquet`.
        # We pass the parquet path; assuming `convert_to_parquet` is idempotent
        # (returns the parquet path if it already exists), this works.
        # Otherwise, pass the corresponding .tsv path instead.
        aoi_hit, calibration_df, participants, df = read_data(parquet_file)

        all_aoi_hit.append(aoi_hit)
        all_calibration_dfs.append(calibration_df)
        all_participants.append(participants)
        all_dfs.append(df)

    # Flatten aoi_hit into a deduplicated list across all files
    flat_aoi_hit = list(dict.fromkeys(col for sublist in all_aoi_hit for col in sublist))

    return flat_aoi_hit, all_calibration_dfs, all_participants, all_dfs


# ======================================
# process raw data into list of dfs
# ======================================


def split_by_text(df) -> dict["str", pd.DataFrame]:
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


def _extract_aoi_token(col):
    """Extract the token name from an AOI hit column."""
    idx = col.find(" - ")
    if idx != -1:
        string = col[idx + 3 :]
    else:
        string = col
    match = re.search(r"\](?:\.\d+)?$", string)
    if match:
        return string[: match.start()]
    return string


def _sort_aois_by_text(aois, text_tokens):
    """Sort AOI columns by their position in the original text.

    Duplicate tokens are matched to consecutive occurrences in the text.
    Tokens not found in the text are placed at the end.
    """
    token_counters = {}
    positions = {}

    for col in aois:
        token = _extract_aoi_token(col)
        if token not in token_counters:
            token_counters[token] = 0

        count = token_counters[token]
        position = None
        for i, t in enumerate(text_tokens):
            if t == token:
                if count == 0:
                    position = i
                    break
                count -= 1

        if position is not None:
            positions[col] = position
            token_counters[token] += 1
        else:
            positions[col] = len(text_tokens) + len(positions)

    return sorted(aois, key=lambda c: positions[c])


def drop_irrelevant_aois(text_df, stimulus, aoi_hit, text_tokens=None):
    """Keep columns for an AOI"""

    def clean_affixes(string, idx):
        string = string.removeprefix("AOI hit [" + str(stimulus) + " - ")
        match = re.search(r"\](?:\.\d+)?$", string)
        if match:
            token = string[: match.start()]
        else:
            token = string
        return f"{idx}.{token}"

    aois_to_keep = [
        a
        for a in aoi_hit
        if a.startswith("AOI hit [" + str(stimulus) + " - ") and a in text_df.columns
    ]

    if text_tokens is not None:
        aois_to_keep = _sort_aois_by_text(aois_to_keep, text_tokens)

    aois_to_keep = {a: clean_affixes(a, idx) for idx, a in enumerate(aois_to_keep)}
    cols_to_drop = [
        c for c in text_df.columns if c.startswith("AOI hit") and c not in aois_to_keep
    ]
    text_df = text_df.drop(columns=cols_to_drop).rename(columns=aois_to_keep)
    return text_df, list(aois_to_keep.values())


def collapse_aoi_columns(text_df, aoi_cols, stimulus, verbose=False):
    """
    Collapse AOI hit columns into a single 'AOI' column (vectorized).
    """
    aoi_data = text_df[aoi_cols].fillna(False).astype(bool)

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

    if verbose:
        n_hits = text_df["AOI"].notna().sum()
        n_unique = text_df["AOI"].nunique()
        print(f"{stimulus}: {n_hits} samples with AOI hits across {n_unique} unique AOIs")

    return text_df


def process_all_texts(
    dfs: pd.DataFrame | list[pd.DataFrame],
    aoi_hit,
    # TODO: this file is old, should update with the edited texts with "[HABLANTE 0], etc."
    texts_csv="../data/mused_chosen_data.csv",
):
    """
    Full pipeline: split by text, then clean AOI columns per text.
    Returns a list of clean DataFrames.

    Args:
        texts_csv: path to CSV with columns 'id' (e.g. 'video_04') and 'text_clean'.
                   If provided, AOI columns are sorted by token position in the text.
    """
    if isinstance(dfs, list):
        text_dfs = {}
        for df in dfs:
            text_dfs = {**text_dfs, **split_by_text(df)}
    else:
        text_dfs = split_by_text(dfs)

    text_tokens_map = {}
    if texts_csv is not None:
        texts_df = pd.read_csv(texts_csv)
        texts_df["num_id"] = texts_df["id"].str.replace("video_", "").astype(int)
        for _, row in texts_df.iterrows():
            text_tokens_map[str(row["num_id"])] = row["text_clean"].split()

    aoi_cols_dict = {}

    for stimulus, text_df in text_dfs.items():
        tokens = text_tokens_map.get(str(stimulus))
        text_df, aoi_cols = drop_irrelevant_aois(
            text_df, stimulus, aoi_hit, text_tokens=tokens
        )
        text_dfs[stimulus] = collapse_aoi_columns(text_df, aoi_cols, stimulus)
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


def get_anotacions(tsv_file):
    anotacions = pd.read_csv(tsv_file, sep="\t").drop(
        ["Recording timestamp", "Computer timestamp"], axis=1
    )
    anotacions = anotacions[
        anotacions["Event"].isin(["TextStart", "TextEnd", "KeyboardEvent"])
    ]
    records = []

    for _, row in anotacions.iterrows():
        event = row["Event"]
        value = str(row["Event value"])

        if event == "TextStart":
            # Check if value is a plain number (text ID) or "Text (X)" format
            if not value.startswith("Text") and value not in ("confidence", "buit"):
                current_text_num = value
                keyboard_events = []  # Reset keyboard events for new text number

        elif event == "KeyboardEvent":
            keyboard_events.append(value)
            if len(keyboard_events) == 2:
                records.append(
                    {
                        "user": row["Participant name"],
                        "text": current_text_num,
                        "sexist": keyboard_events[0],
                        "confidence": keyboard_events[1],
                    }
                )
        elif len(keyboard_events) > 2:
            raise ValueError

    anotacions = pd.DataFrame(records, columns=["user", "text", "sexist", "confidence"])
    anotacions.text = anotacions.text.map({"76)": "76"}).fillna(anotacions.text).astype(int)
    # map when Numlock wasn't active
    anotacions.sexist = (
        anotacions.sexist.map({"End": 1, "Insert": 0, "PageDown": 1})
        .fillna(anotacions.sexist)
        .astype(int)
    )
    anotacions.confidence = (
        anotacions.confidence.map({"End": 1, "Down": 2, "PageDown": 3})
        .fillna(anotacions.confidence)
        .astype(int)
    )
    # Correccions manuals
    anotacions.loc[(anotacions.user == "javi") & (anotacions.text == 78), "sexist"] = 0
    anotacions.loc[(anotacions.user == "clara") & (anotacions.text == 62), "sexist"] = 1
    anotacions.loc[(anotacions.user == "alo") & (anotacions.text == 214), "sexist"] = 0

    assert anotacions.shape[0] == anotacions.user.nunique() * anotacions.text.nunique()
    anotacions = anotacions.sort_values(by=["user", "text"])
    return anotacions
