import spacy
from spacy import displacy

nlp = spacy.blank("es")  # Spanish blank model


def annotations_to_spacy_doc(text: str, annotations: list[dict]) -> spacy.tokens.Doc:
    doc = nlp.make_doc(text)

    all_spans = []

    for ann in annotations:
        char_start = ann["start"]
        char_end = ann["end"]
        labels = ann["labels"]

        for label in labels:
            span = doc.char_span(char_start, char_end, label=label, alignment_mode="expand")

            if span is None:
                print(f"⚠️  Could not align span [{char_start}:{char_end}] for label '{label}'")
                continue

            all_spans.append(span)

    # ✅ Single span group with all spans
    doc.spans["sc"] = all_spans

    return doc


def visualize_annotations(
    text: str, annotations: list[dict], colors: dict = None, jupyter: bool = True
) -> None:
    doc = annotations_to_spacy_doc(text, annotations)

    default_colors = {
        "INEQUALIT\/DISCRIMINATION": "#ff9561",
        "IMPLICIT SEXISM": "#aa9cfc",
        "EXPLICIT SEXISM": "#fc9caa",
        "STEREOTYPING": "#9cfc9c",
        "OBJECTIFICATION": "#fcf09c",
    }
    colors = {**default_colors, **(colors or {})}

    options = {
        "spans_key": "sc",  # ✅ single string key, not a list
        "colors": colors,
    }

    displacy.render(doc, style="span", options=options, jupyter=jupyter)
