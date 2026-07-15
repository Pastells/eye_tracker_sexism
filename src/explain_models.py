"""
explain_models.py

Run Captum explainability methods (Saliency, InputXGradient, IG) on trained models
for the 40 chosen test texts. Saves per-token and per-word attributions.

Usage:
    uv run python explain_models.py                              # all checkpoints
    uv run python explain_models.py --checkpoint mrbert_filtered  # one checkpoint
    uv run python explain_models.py --methods saliency ig         # specific methods
    uv run python explain_models.py --n_texts 5                  # limit texts (debug)
"""

import argparse
import json
import os
import sys

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["NCCL_P2P_DISABLE"] = "1"
os.environ["NCCL_IB_DISABLE"] = "1"

import pandas as pd
import torch
from tqdm import tqdm

from explain_captum import (
    ModelWrapper,
    aggregate_to_words,
    get_tokenizer_model,
    tokenize,
)

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
CHOSEN_PATH = os.path.join(DATA_DIR, "chosen_data_full.csv")
CHECKPOINTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "checkpoints")

METHODS = ["saliency", "input_x_gradient", "ig"]

MODELS = {
    "mrbert": "BSC-LT/MrBERT-es",
    "mbert": "bert-base-multilingual-cased",
    "beto": "dccuchile/bert-base-spanish-wwm-cased",
}


def find_checkpoints():
    """Discover available trained model checkpoints."""
    checkpoints = {}
    if not os.path.exists(CHECKPOINTS_DIR):
        print(f"WARNING: Checkpoints directory not found: {CHECKPOINTS_DIR}")
        return checkpoints

    for entry in os.listdir(CHECKPOINTS_DIR):
        entry_path = os.path.join(CHECKPOINTS_DIR, entry)
        if not os.path.isdir(entry_path):
            continue
        # Look for checkpoint-* subdirs
        ckpt_subdirs = [d for d in os.listdir(entry_path) if d.startswith("checkpoint-")]
        if ckpt_subdirs:
            # Use the latest checkpoint
            latest = sorted(ckpt_subdirs, key=lambda x: int(x.split("-")[-1]))[-1]
            checkpoints[entry] = os.path.join(entry_path, latest)
        else:
            # Maybe the dir itself is the checkpoint
            if os.path.exists(os.path.join(entry_path, "config.json")):
                checkpoints[entry] = entry_path

    return checkpoints


def load_test_texts():
    """Load the 40 chosen test texts."""
    df = pd.read_csv(CHOSEN_PATH)
    texts = df.text_clean.str.replace("\n", " ").tolist()
    ids = df.id.tolist()
    return ids, texts


def explain_single_text(wrapper, text, methods):
    """Run explainability methods on a single text, return results dict."""
    input_ids, attention_mask, tokens = tokenize(wrapper, text)
    pred_class, pred_prob = wrapper.predict(input_ids, attention_mask)

    results = {
        "pred_class": pred_class,
        "pred_prob": pred_prob,
        "methods": {},
    }

    for method_name in methods:
        try:
            from explain_captum import METHODS

            method_cls = METHODS[method_name]
            method = method_cls(wrapper)
            result = method.attribute(input_ids, attention_mask)

            if isinstance(result, tuple):
                scores, delta = result
                delta_val = delta.item() if hasattr(delta, "item") else delta
            else:
                scores = result
                delta_val = None

            scores_list = scores.tolist() if hasattr(scores, "tolist") else list(scores)
            text_words = text.split()
            word_names, word_scores = aggregate_to_words(tokens, scores_list, text_words)

            results["methods"][method_name] = {
                "tokens": tokens,
                "scores": scores_list,
                "words": word_names,
                "word_scores": word_scores,
                "delta": delta_val,
            }
        except Exception as e:
            print(f"  Error with {method_name}: {e}")
            results["methods"][method_name] = {"error": str(e)}

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Run Captum explanations on trained models"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Specific checkpoint to explain (default: all found)",
    )
    parser.add_argument(
        "--methods",
        type=str,
        nargs="+",
        default=METHODS,
        help="Methods to run (default: saliency input_x_gradient ig)",
    )
    parser.add_argument(
        "--n_texts", type=int, default=None, help="Limit number of texts (default: all 40)"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="explanations",
        help="Output directory for results",
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Discover checkpoints
    all_checkpoints = find_checkpoints()
    if not all_checkpoints:
        print("No checkpoints found. Run train_models.py first.")
        sys.exit(1)

    if args.checkpoint:
        if args.checkpoint not in all_checkpoints:
            print(
                f"Checkpoint '{args.checkpoint}' not found. Available: {list(all_checkpoints.keys())}"
            )
            sys.exit(1)
        checkpoints = {args.checkpoint: all_checkpoints[args.checkpoint]}
    else:
        checkpoints = all_checkpoints

    print(f"Found {len(checkpoints)} checkpoints: {list(checkpoints.keys())}")

    # Load test texts
    text_ids, texts = load_test_texts()
    if args.n_texts:
        text_ids = text_ids[: args.n_texts]
        texts = texts[: args.n_texts]
    print(f"Explaining {len(texts)} texts with methods: {args.methods}")

    os.makedirs(args.output_dir, exist_ok=True)

    # Process each checkpoint
    for ckpt_name, ckpt_path in checkpoints.items():
        print(f"\n{'=' * 60}")
        print(f"Checkpoint: {ckpt_name}")
        print(f"Path: {ckpt_path}")
        print(f"{'=' * 60}")

        # Determine model_id from checkpoint name
        model_key = ckpt_name.split("_")[0]
        model_id = MODELS.get(model_key)
        if model_id is None:
            print(f"WARNING: Cannot determine model_id for '{ckpt_name}', skipping")
            continue

        try:
            tokenizer, model = get_tokenizer_model(model_id, checkpoint=ckpt_path)
            model = model.to(device)
            model.eval()
            wrapper = ModelWrapper(model, tokenizer, device)
        except Exception as e:
            print(f"ERROR loading model: {e}")
            continue

        all_results = {}
        for idx, (text_id, text) in enumerate(tqdm(zip(text_ids, texts), total=len(texts))):
            try:
                results = explain_single_text(wrapper, text, args.methods)
                all_results[text_id] = {
                    "text": text,
                    **results,
                }
            except Exception as e:
                print(f"  Error processing {text_id}: {e}")
                all_results[text_id] = {"text": text, "error": str(e)}

            # Save incrementally
            output_path = os.path.join(args.output_dir, f"{ckpt_name}.json")
            with open(output_path, "w") as f:
                json.dump(all_results, f, indent=2, ensure_ascii=False)

        # Also save as CSV for easy loading
        rows = []
        for text_id, res in all_results.items():
            if "error" in res and "methods" not in res:
                continue
            for method_name, method_res in res.get("methods", {}).items():
                if "error" in method_res:
                    continue
                for word, score in zip(method_res["words"], method_res["word_scores"]):
                    rows.append(
                        {
                            "text_id": text_id,
                            "method": method_name,
                            "word": word,
                            "salience": score,
                            "pred_class": res.get("pred_class"),
                            "pred_prob": res.get("pred_prob"),
                        }
                    )

        csv_path = os.path.join(args.output_dir, f"{ckpt_name}.csv")
        pd.DataFrame(rows).to_csv(csv_path, index=False)
        print(f"Saved {len(rows)} rows to {csv_path}")

    print(f"\nDone. Results in {args.output_dir}/")


if __name__ == "__main__":
    main()
