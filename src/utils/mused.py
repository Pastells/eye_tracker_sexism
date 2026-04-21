import re
from operator import itemgetter

import numpy as np
import pandas as pd

# ==================
# Read textual data
# ==================


def get_df(file):
    df = pd.read_csv(file)
    cols = df.columns
    df = df.rename(columns={col: col.lower() for col in cols})
    return df


def join_dfs(dfs, cols_left, cols_right, id_col, binary_cols, num_cols, label_col):
    df = (
        pd.concat(
            [
                (
                    dfs["m"][cols_left]
                    .merge(dfs["c"][cols_right], on=id_col, how="left")
                    .merge(dfs["g"][cols_right], on=id_col, how="left")
                ),
                (
                    dfs["a"][cols_left]
                    .merge(dfs["r"][cols_right], on=id_col, how="left")
                    .merge(dfs["n"][cols_right], on=id_col, how="left")
                ),
            ]
        )
        .reset_index()
        .drop("index", axis=1)
    )
    for col in binary_cols:
        df[col + "_soft"] = np.round((df[col] + df[col + "_x"] + df[col + "_y"]) / 3, 2)
        df[col] = (df[col + "_soft"] > 0.5).astype(int)
        df = df.drop(columns=[col + "_x", col + "_y"])

    for col in num_cols:
        df[col] = np.round((df[col] + df[col + "_x"] + df[col + "_y"]) / 3, 2)
        df = df.drop(columns=[col + "_x", col + "_y"])

    df[label_col] = df[label_col] + df[label_col + "_x"] + df[label_col + "_y"]
    df[label_col] = df[label_col].apply(lambda x: sorted(x, key=itemgetter("start")))
    df = df.drop(columns=[label_col + "_x", label_col + "_y"])

    df = df.rename(columns={id_col: "id"})
    return df


def get_lens(_df, IDS, sentiment="SEXIST"):
    """
    filter (no-)sexist texts and add spans
    """
    df = _df.copy()
    df["len"] = df.text.str.len()
    df = df[(df.sentiment == sentiment) & (df.video_id.isin(IDS)) & (df.len < 2000)]
    # df = df[(df.video_id.isin(IDS))].sort_values(by="len")

    df["label"] = df["label"].apply(
        lambda x: eval(x) if isinstance(x, str) else (x if isinstance(x, list) else [])
    )
    df[["text_clean", "label_clean"]] = df.apply(clean, axis=1)
    df["span_len"] = df.label_clean.apply(lambda x: sum([ann["end"] - ann["start"] for ann in x]))
    df["span_num"] = df.label_clean.str.len()
    df["span_lens"] = df.label_clean.apply(lambda x: [ann["end"] - ann["start"] for ann in x])
    return df


# ==================
# Clean text
# ==================


def clean_one_speaker_tag(content, remove_all=False):
    speaker_tags = re.findall(r"(\[SPEAKER_\d{2}\]: )", content)

    if remove_all or len(set(speaker_tags)) == 1:
        return re.sub(r"\[SPEAKER_\d{2}\]: ", "", content)
    return content


def clean_speaker_tags(content, remove_all=False):
    speaker_tags = re.findall(r"(\[SPEAKER_\d{2}\]: )", content)

    if remove_all or len(set(speaker_tags)) == 1:
        return re.sub(r"\[SPEAKER_\d{2}\]: ", "", content)

    # Initialize variables to track the previous tag and the result content
    previous_tag = None
    result_content = ""
    last_index = 0

    for tag in speaker_tags:
        # Find the position of the current tag in the content
        tag_index = content.find(tag, last_index)

        # Add the segment of content up to the current tag to the result
        result_content += content[last_index:tag_index]

        # If the current tag is different from the previous one, add it to the result
        if tag != previous_tag:
            result_content += tag
            previous_tag = tag

        # Update the last index to continue searching after the current tag
        last_index = tag_index + len(tag)

    # Add the remaining content after the last tag
    result_content += content[last_index:]

    return result_content


def clean_text(text, hablante=True):
    """Remove time stamps and line breaks"""
    texts = text.strip().split("\n")
    text = "".join(
        [t.split(maxsplit=4)[-1] if not t.startswith("<OCR>") else t for t in texts]
    ).replace("\r", "")
    text = re.sub(r"<\/?OCR>", "", text).strip()
    text = clean_speaker_tags(text)
    if hablante:
        text = re.sub(r"\[SPEAKER_0([0-9])\]", r"\n[HABLANTE \1]", text).strip()
    return text


# ==================================================
# Offset annotations to agree with clean text
# ==================================================


def build_offset_mapping(original: str, cleaned: str) -> dict:
    """
    Maps each character index in the original string to the
    corresponding index in the cleaned string using SequenceMatcher.
    Returns {old_index: new_index} for all matched characters.
    """
    from difflib import SequenceMatcher

    matcher = SequenceMatcher(None, original, cleaned, autojunk=False)

    mapping = {}
    for op, i1, i2, j1, j2 in matcher.get_opcodes():
        if op == "equal":
            for old, new in zip(range(i1, i2), range(j1, j2)):
                mapping[old] = new
    return mapping


def remap_span(start: int, end: int, mapping: dict) -> tuple[int, int] | None:
    if not mapping:
        return None

    sorted_keys = sorted(mapping.keys())
    max_key = sorted_keys[-1]

    # Find nearest valid start (expand RIGHT)
    new_start = None
    for i in range(start, max_key + 1):
        if i in mapping:
            new_start = mapping[i]
            break

    # Find nearest valid end (expand LEFT)
    new_end = None
    for i in range(end - 1, start - 1, -1):
        if i in mapping:
            new_end = mapping[i] + 1
            break

    if new_start is None or new_end is None or new_start >= new_end:
        return None
    return new_start, new_end


def clean_text_with_mapping(text: str) -> tuple[str, dict]:
    """
    Returns (cleaned_text, offset_mapping).
    """
    cleaned = clean_text(text)
    # cleaned = clean_one_speaker_tag(cleaned)
    mapping = build_offset_mapping(text, cleaned)
    return cleaned, mapping


TIMESTAMP_PATTERN = re.compile(r"\d+\s+\d{2}:\d{2},\d+\s+-->\s+\d{2}:\d{2},\d+\s+")
PATTERN = re.compile(r"<\/?OCR>|\[SPEAKER_0([0-9])\]:")


def clean_text_with_mapping_new(text: str, hablante: bool = True) -> tuple[str, dict[int, int]]:
    """
    Clean subtitle-like text while building a mapping from original indices
    to cleaned indices.

    Instead of 1) cleaning, and 2) building the offset mapping, this does it together
    The main benefit would be to have a mapping from SPEAKER to HABLANTE, but it's not
    working great, so for now I stick with the old one, even if this tag is lost on the spans
    """

    mapping: dict[int, int] = {}
    cleaned = []

    clean_i = 0
    orig_i = 0
    n = len(text)

    def map_span(start, end, target):
        for i in range(start, end):
            mapping[i] = target

    while orig_i < n:
        # --- 1. Remove timestamp prefix (only at line starts) ---
        if orig_i == 0 or text[orig_i - 1] == "\n":
            m = TIMESTAMP_PATTERN.match(text, orig_i)
            if m:
                start, end = m.span()
                map_span(start, end, clean_i)
                orig_i = end
                continue

        # --- 2. Apply structured replacements ---
        m = PATTERN.match(text, orig_i)
        if m:
            start, end = m.span()
            matched = m.group(0)

            # OCR tags → delete
            if matched.startswith("<OCR") or matched.startswith("</OCR"):
                map_span(start, end, clean_i)

            # Speaker tags
            elif matched.startswith("[SPEAKER_0"):
                if hablante:
                    replacement = f"\n[HABLANTE {m.group(1)}]"
                    cleaned.append(replacement)
                    map_span(start, end, clean_i)
                    clean_i += len(replacement) - 1
                else:
                    # keep original
                    for i in range(start, end):
                        mapping[i] = clean_i
                        cleaned.append(text[i])
                        clean_i += 1

            orig_i = end
            continue

        # --- 3. Copy normal character ---
        mapping[orig_i] = clean_i
        cleaned.append(text[orig_i])
        orig_i += 1
        clean_i += 1

    final_text = (
        "".join(cleaned)
        .strip()
        .replace("\n\n", "__newline__")
        .replace("\n", " ")
        .replace("__newline__", "\n")
    )

    return final_text, mapping


def clean(row):
    cleaned_text, mapping = clean_text_with_mapping(row["text"])

    new_labels = []
    for annotation in row["label"]:  # each is a dict with "start", "end", "labels", etc.
        result = remap_span(int(annotation["start"]), int(annotation["end"]), mapping)
        if result:
            new_start, new_end = result
            new_labels.append(
                {
                    **annotation,
                    "text": cleaned_text[new_start:new_end],
                    "start": new_start,
                    "end": new_end,
                }
            )
        # if result is None, the annotation was in a deleted region -> drop it

    return pd.Series(
        {
            "text_clean": cleaned_text,
            "label_clean": new_labels,
        }
    )
