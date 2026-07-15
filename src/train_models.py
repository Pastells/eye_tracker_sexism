"""
train_models.py

Train 3 BERT models (MrBERT-es, RoBERTa-BNE, BETO) on two data conditions:
  - Filtered: mused_all_clean.csv (~124 train texts, unanimous agreement)
  - Full: text_with_mmlabel.csv (~360 train texts, all with text-video agreement)

Usage:
    uv run python train_models.py                    # all models, both conditions
    uv run python train_models.py --model mrbert     # one model
    uv run python train_models.py --condition filtered  # one condition
"""

import argparse
import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["NCCL_P2P_DISABLE"] = "1"
os.environ["NCCL_IB_DISABLE"] = "1"

import pandas as pd

from utils.train import get_tokenizer_model, train

MODELS = {
    "mrbert": "BSC-LT/MrBERT-es",
    "mbert": "bert-base-multilingual-cased",
    "beto": "dccuchile/bert-base-spanish-wwm-cased",
}

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
CHOSEN_PATH = os.path.join(DATA_DIR, "mused_chosen_data.csv")
FILTERED_PATH = os.path.join(DATA_DIR, "mused_all_clean.csv")
FULL_PATH = os.path.join(DATA_DIR, "01_Text", "text_with_mmlabel.csv")

MAX_TOKENS = 256
DROPOUT = 0.0
LABELS = "hard"


def load_filtered_data():
    """Load filtered set (~164 texts with unanimous annotator agreement)."""
    df = pd.read_csv(FILTERED_PATH)
    df = df[["id", "text_clean", "sexist"]]
    chosen = pd.read_csv(CHOSEN_PATH)
    chosen_ids = set(chosen.id.tolist())
    df_train = (
        df[~df.id.isin(chosen_ids)].sample(frac=1, random_state=42).reset_index(drop=True)
    )
    print(f"[Filtered] {len(df_train)} train texts")
    return df_train


def load_full_data():
    """Load full MuSeD corpus (400 texts with text-video agreement)."""
    df = pd.read_csv(FULL_PATH)
    df = df[["id", "text_clean", "sexist_text"]].rename(columns={"sexist_text": "sexist"})
    chosen = pd.read_csv(CHOSEN_PATH)
    chosen_ids = set(chosen.id.tolist())
    df_train = (
        df[~df.id.isin(chosen_ids)].sample(frac=1, random_state=42).reset_index(drop=True)
    )
    print(f"[Full] {len(df_train)} train texts")
    return df_train


def main():
    parser = argparse.ArgumentParser(description="Train BERT models for sexism detection")
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        choices=list(MODELS.keys()),
        help="Train a specific model (default: all)",
    )
    parser.add_argument(
        "--condition",
        type=str,
        default=None,
        choices=["filtered", "full"],
        help="Train on a specific data condition (default: both)",
    )
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=5e-6)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--dropout", type=float, default=0.1)
    args = parser.parse_args()

    models_to_train = [args.model] if args.model else list(MODELS.keys())
    conditions = [args.condition] if args.condition else ["filtered", "full"]

    # Load data once
    data = {}
    if "filtered" in conditions:
        data["filtered"] = load_filtered_data()
    if "full" in conditions:
        data["full"] = load_full_data()

    results = {}

    for condition in conditions:
        df_train = data[condition]
        print(f"\n{'=' * 60}")
        print(f"Condition: {condition} ({len(df_train)} train texts)")
        print(f"{'=' * 60}")

        for model_name in models_to_train:
            model_id = MODELS[model_name]
            output_dir = f"checkpoints/{model_name}_{condition}"

            print(f"\n--- Training {model_name} ({model_id}) on {condition} ---")
            print(f"Output: {output_dir}")

            tokenizer, model = get_tokenizer_model(
                model_id, dropout=args.dropout, labels=LABELS
            )

            # Override training args
            import utils.train as train_module

            train_module.training_args.output_dir = output_dir
            train_module.training_args.learning_rate = args.lr
            train_module.training_args.per_device_train_batch_size = args.batch_size
            train_module.training_args.per_device_eval_batch_size = args.batch_size
            train_module.training_args.num_train_epochs = args.epochs

            try:
                _, best_checkpoint = train(
                    df_train=df_train,
                    df_test=None,
                    tokenizer=tokenizer,
                    model=model,
                    labels=LABELS,
                    max_tokens=MAX_TOKENS,
                )
                results[f"{model_name}_{condition}"] = {
                    "status": "ok",
                    "best_checkpoint": best_checkpoint,
                }
                print(f"Best checkpoint: {best_checkpoint}")
            except Exception as e:
                print(f"ERROR training {model_name} on {condition}: {e}")
                results[f"{model_name}_{condition}"] = {
                    "status": "error",
                    "error": str(e),
                }

    print(f"\n{'=' * 60}")
    print("Summary:")
    print(f"{'=' * 60}")
    for key, val in results.items():
        status = val["status"]
        ckpt = val.get("best_checkpoint", "N/A")
        print(f"  {key}: {status} -> {ckpt}")


if __name__ == "__main__":
    main()
