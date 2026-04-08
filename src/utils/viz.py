import seaborn as sns
import spacy
from spacy import displacy
from weasyprint import HTML

palette = sns.color_palette("colorblind").as_hex()


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
                print(f"ÔÜá´©Å  Could not align span [{char_start}:{char_end}] for label '{label}'")
                continue

            all_spans.append(span)

    # Ô£à Single span group with all spans
    doc.spans["sc"] = all_spans

    return doc


def visualize_annotations(
    text: str,
    annotations: list[dict],
    file=None,
) -> None:
    doc = annotations_to_spacy_doc(text, annotations)
    colors = dict(
        zip(
            [
                "INEQUALITY\\/DISCRIMINATION",
                "IMPLICIT SEXISM",
                "EXPLICIT SEXISM",
                "STEREOTYPE",
                "OBJECTIFICATION",
                "IRONY",
                "JOKE",
            ],
            palette,
        )
    )

    options = {
        "spans_key": "sc",  # single string key, not a list
        "colors": colors,
    }
    if file is None:
        kwargs = {"jupyter": True}
        displacy.render(doc, style="span", options=options, **kwargs)
        return

    options = {
        **options,
        # --- vertical layout ---
        "top_offset": 20,  # distance from text to first underline
        "span_label_offset": 10,  # label position
        "line_height": 1,  # distance between lines
        # --- horizontal layout ---
        "offset_x": 5,
    }

    kwargs = {"jupyter": False, "page": True}
    html = (
        displacy.render(doc, style="span", options=options, **kwargs)
        .replace("font-size: 16px", "font-size: 10px")
        .replace("font-weight: bold;", "font-weight: normal;")
        .replace("display: inline-block;", "display: inline-block; margin-right: -10px;")
        .replace("margin-bottom: 6rem", "margin-bottom: 2rem")
    )
    HTML(string=html).write_pdf(file)
