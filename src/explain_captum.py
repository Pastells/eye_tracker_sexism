"""
Captum Interpretability Suite for BERT models

TODO:
- Fix GPU CUDA errors for some methods (index out of bounds on attention_mask)
- Implement DeepLift, GuidedBackprop, GuidedGradCam with proper model wrapping
  (these require torch.nn.Module, not function wrapper)
- Optimize FeatureAblation (too slow - embeddings are high dimensional)
- Add input_x_gradient method (currently broken)
- Test NoiseTunnel with faster base methods
"""

import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
os.environ["NCCL_P2P_DISABLE"] = "1"
os.environ["NCCL_IB_DISABLE"] = "1"

import argparse
import html as html_lib
import json
from tqdm import tqdm

import numpy as np
import torch
from captum.attr import (
    DeepLift,
    FeatureAblation,
    GuidedBackprop,
    GuidedGradCam,
    InputXGradient,
    IntegratedGradients,
    LayerGradCam,
    NoiseTunnel,
    Saliency,
)
from IPython.display import HTML, display
from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer

from utils.mused import load_data


def get_tokenizer_model(model_id, checkpoint=None, dropout=0.0, num_labels=2):
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if checkpoint:
        model = AutoModelForSequenceClassification.from_pretrained(checkpoint)
    else:
        config = AutoConfig.from_pretrained(model_id)
        config.hidden_dropout_prob = dropout
        config.attention_probs_dropout_prob = dropout
        config.num_labels = num_labels
        model = AutoModelForSequenceClassification.from_pretrained(model_id, config=config)
    return tokenizer, model


class ModelWrapper:
    def __init__(self, model, tokenizer, device):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device

    def get_embeddings(self, input_ids):
        return self.model.model.embeddings(input_ids)

    def forward_with_embeddings(self, embeddings, attention_mask):
        if attention_mask is not None and attention_mask.device != embeddings.device:
            attention_mask = attention_mask.to(embeddings.device)
        outputs = self.model(inputs_embeds=embeddings, attention_mask=attention_mask)
        return outputs.logits[:, 1]

    def forward(self, input_ids, attention_mask):
        with torch.no_grad():
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        return outputs.logits

    def forward_for_captum(self, input_ids, attention_mask=None):
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        embeddings = self.model.model.embeddings(input_ids)
        embeddings = embeddings.requires_grad_(True)
        if attention_mask.device != embeddings.device:
            attention_mask = attention_mask.to(embeddings.device)
        outputs = self.model(inputs_embeds=embeddings, attention_mask=attention_mask)
        return outputs.logits

    def forward_embeddings_for_captum(self, embeddings, attention_mask=None):
        if attention_mask is None:
            attention_mask = torch.ones_like(embeddings[:, :, 0])
        embeddings = embeddings.requires_grad_(True)
        if attention_mask.device != embeddings.device:
            attention_mask = attention_mask.to(embeddings.device)
        outputs = self.model(inputs_embeds=embeddings, attention_mask=attention_mask)
        return outputs.logits

    def predict(self, input_ids, attention_mask):
        logits = self.forward(input_ids, attention_mask)
        return logits.argmax(dim=-1).item(), torch.softmax(logits, dim=-1).max().item()


def tokenize(wrapper, text):
    inputs = wrapper.tokenizer(text, return_tensors="pt")
    input_ids = inputs["input_ids"].to(wrapper.device)
    attention_mask = inputs["attention_mask"].to(wrapper.device)
    tokens = wrapper.tokenizer.convert_ids_to_tokens(input_ids.squeeze(0))
    return input_ids, attention_mask, tokens


def get_baseline(wrapper, input_ids):
    embeddings = wrapper.get_embeddings(input_ids)
    return torch.zeros_like(embeddings)


def summarize_attributions(attributions):
    if attributions is None:
        return None
    if hasattr(attributions, "shape"):
        if len(attributions.shape) == 3:
            return attributions.squeeze(0).sum(dim=-1)
        elif len(attributions.shape) == 2:
            return attributions.squeeze(0)
    return attributions


class AttributionMethod:
    name = "base"

    def __init__(self, wrapper):
        self.wrapper = wrapper

    def attribute(self, input_ids, attention_mask):
        raise NotImplementedError


class SaliencyMethod(AttributionMethod):
    name = "saliency"

    def attribute(self, input_ids, attention_mask):
        embeddings = self.wrapper.get_embeddings(input_ids)
        saliency = Saliency(self.wrapper.forward_embeddings_for_captum)
        attributions = saliency.attribute(embeddings, target=1)
        return summarize_attributions(attributions)


class InputXGradientMethod(AttributionMethod):
    name = "input_x_gradient"

    def attribute(self, input_ids, attention_mask):
        embeddings = self.wrapper.get_embeddings(input_ids)
        method = InputXGradient(self.wrapper.forward_embeddings_for_captum)
        attributions = method.attribute(embeddings, target=1)
        return summarize_attributions(attributions)


class IntegratedGradientsMethod(AttributionMethod):
    name = "integrated_gradients"

    def attribute(self, input_ids, attention_mask):
        embeddings = self.wrapper.get_embeddings(input_ids)
        baseline = torch.zeros_like(embeddings)

        def forward_fn(embeddings, attention_mask):
            return self.wrapper.forward_with_embeddings(embeddings, attention_mask)

        ig = IntegratedGradients(forward_fn)
        attributions, delta = ig.attribute(
            embeddings,
            baselines=baseline,
            additional_forward_args=(attention_mask,),
            return_convergence_delta=True,
        )
        return summarize_attributions(attributions), delta


class DeepLiftMethod(AttributionMethod):
    name = "deep_lift"

    def attribute(self, input_ids, attention_mask):
        embeddings = self.wrapper.get_embeddings(input_ids)
        baseline = torch.zeros_like(embeddings)
        method = DeepLift(self.wrapper.forward_embeddings_for_captum)
        attributions = method.attribute(embeddings, baselines=baseline, target=1)
        return summarize_attributions(attributions)


class GuidedBackpropMethod(AttributionMethod):
    name = "guided_backprop"

    def attribute(self, input_ids, attention_mask):
        embeddings = self.wrapper.get_embeddings(input_ids)
        method = GuidedBackprop(self.wrapper.forward_embeddings_for_captum)
        attributions = method.attribute(embeddings, target=1)
        return summarize_attributions(attributions)


class LayerGradCamMethod(AttributionMethod):
    name = "layer_gradcam"

    def __init__(self, wrapper, layer_name=None):
        super().__init__(wrapper)
        if layer_name:
            self.layer = get_layer_by_name(wrapper.model, layer_name)
        else:
            self.layer = wrapper.model.model.layers[-1]

    def attribute(self, input_ids, attention_mask):
        embeddings = self.wrapper.get_embeddings(input_ids)
        method = LayerGradCam(self.wrapper.forward_embeddings_for_captum, self.layer)
        attributions = method.attribute(embeddings, target=1)
        return summarize_attributions(attributions)


class GuidedGradCamMethod(AttributionMethod):
    name = "guided_gradcam"

    def __init__(self, wrapper, layer_name=None):
        super().__init__(wrapper)
        if layer_name:
            self.layer = get_layer_by_name(wrapper.model, layer_name)
        else:
            self.layer = wrapper.model.model.layers[-1]

    def attribute(self, input_ids, attention_mask):
        embeddings = self.wrapper.get_embeddings(input_ids)
        method = GuidedGradCam(self.wrapper.forward_embeddings_for_captum, self.layer)
        attributions = method.attribute(embeddings, target=1)
        return summarize_attributions(attributions)


class FeatureAblationMethod(AttributionMethod):
    name = "feature_ablation"

    def attribute(self, input_ids, attention_mask):
        embeddings = self.wrapper.get_embeddings(input_ids)
        method = FeatureAblation(self.wrapper.forward_embeddings_for_captum)
        attributions = method.attribute(embeddings, target=1)
        return summarize_attributions(attributions)


class NoiseTunnelMethod(AttributionMethod):
    name = "noise_tunnel"

    def __init__(self, wrapper, base_method="saliency", n_samples=5):
        super().__init__(wrapper)
        self.base_method_str = base_method
        self.n_samples = n_samples

    def attribute(self, input_ids, attention_mask):
        embeddings = self.wrapper.get_embeddings(input_ids)
        if self.base_method_str == "saliency":
            base = Saliency(self.wrapper.forward_embeddings_for_captum)
        elif self.base_method_str == "ig":
            baseline = torch.zeros_like(embeddings)

            def forward_fn(emb):
                return self.wrapper.forward_with_embeddings(emb, attention_mask)

            base = IntegratedGradients(forward_fn)
        else:
            base = Saliency(self.wrapper.forward_embeddings_for_captum)

        nt = NoiseTunnel(base)
        attributions = nt.attribute(embeddings, target=1)
        return summarize_attributions(attributions)


METHODS = {
    "saliency": SaliencyMethod,
    "input_x_gradient": InputXGradientMethod,
    "ig": IntegratedGradientsMethod,
    "deep_lift": DeepLiftMethod,
    "guided_backprop": GuidedBackpropMethod,
    "layer_gradcam": LayerGradCamMethod,
    "guided_gradcam": GuidedGradCamMethod,
    "feature_ablation": FeatureAblationMethod,
    "noise_tunnel": NoiseTunnelMethod,
}


def get_layer_by_name(model, layer_name):
    parts = layer_name.split(".")
    layer = model
    for part in parts:
        layer = getattr(layer, part)
    return layer


def format_token(token):
    if token.startswith("##"):
        return token[2:]
    elif token.startswith("Ġ") or token.startswith("▁"):
        return token[1:]
    return token


def build_token_to_word(tokens: list[str], words: list[str]) -> list[int]:
    """
    Alinea tokens BERT (amb prefix ##) a paraules del text original.
    Retorna llista de longitud len(tokens), amb -1 per tokens especials.
    """
    special = {"[CLS]", "[SEP]", "[PAD]", "[MASK]", "<s>", "</s>"}
    token_to_word = []
    word_idx = 0
    char_pos_in_word = 0

    for tok in tokens:
        if tok in special:
            token_to_word.append(-1)
            continue

        is_continuation = tok.startswith("##")
        tok_clean = tok[2:] if is_continuation else tok.lstrip("▁Ġ")

        if not is_continuation and char_pos_in_word > 0:
            # Nou token que comença paraula nova
            word_idx += 1
            char_pos_in_word = 0

        # Avança word_idx si el token no encaixa a la paraula actual
        while word_idx < len(words) and not words[word_idx][char_pos_in_word:].startswith(
            tok_clean
        ):
            word_idx += 1
            char_pos_in_word = 0

        if word_idx >= len(words):
            token_to_word.append(-1)
            continue

        token_to_word.append(word_idx)
        char_pos_in_word += len(tok_clean)
        if char_pos_in_word >= len(words[word_idx]):
            # Paraula completada, però no avancem fins al pròxim token no-##
            pass

    return token_to_word


def aggregate_to_words(
    tokens: list[str],
    scores: list[float],
    words: list[str],
    reduce: str = "sum",
) -> tuple[list[str], list[float]]:
    """Agrega scores de tokens a paraules. Retorna (words, word_scores)."""
    scores = np.abs(scores)
    token_to_word = build_token_to_word(tokens, words)
    word_scores = np.zeros(len(words), dtype=float)
    counts = np.zeros(len(words), dtype=int)

    for i, w_idx in enumerate(token_to_word):
        if w_idx < 0:
            continue
        word_scores[w_idx] += scores[i]
        counts[w_idx] += 1

    if reduce == "mean":
        word_scores = np.where(counts > 0, word_scores / np.maximum(counts, 1), 0.0)

    # Normalitza a [0, 1]
    max_s = word_scores.max()
    if max_s > 0:
        word_scores = word_scores / max_s

    return words, word_scores.tolist()


def visualize_attributions(tokens, scores, cmap_intensity=1.0):
    scores = np.array(scores, dtype=float)
    if np.abs(scores).max() > 0:
        scores = scores / (np.abs(scores).max() + 1e-9)

    parts = [
        '<div style="font-family: monospace; font-size: 16px; line-height: 2.2; text-decoration: none;">'
    ]

    for tok, s in zip(tokens, scores):
        if s > 0:
            color = f"rgba(0, 200, 0, {abs(s) * cmap_intensity:.3f})"
        else:
            color = f"rgba(220, 0, 0, {abs(s) * cmap_intensity:.3f})"

        display_tok = format_token(tok)
        if tok.startswith("##"):
            sep = ""
        elif tok.startswith("Ġ") or tok.startswith("▁"):
            sep = " "
        else:
            sep = " "

        safe_tok = html_lib.escape(display_tok)
        safe_title = html_lib.escape(f"{s:.4f}")

        parts.append(
            f'{sep}<span style="background-color: {color}; '
            f"padding: 2px 4px; border-radius: 3px; "
            f'text-decoration: none; display: inline-block;" '
            f'title="{safe_title}">{safe_tok}</span>'
        )

    parts.append("</div>")
    display(HTML("".join(parts)))


def run_all_methods(wrapper, text, methods=None):
    input_ids, attention_mask, tokens = tokenize(wrapper, text)
    pred_class, pred_prob = wrapper.predict(input_ids, attention_mask)

    print(f"Text: {text[:200]}...")
    print(f"Prediction: class={pred_class}, prob={pred_prob:.4f}")

    if methods is None:
        methods = list(METHODS.keys())

    results = {}

    for method_name in methods:
        if method_name not in METHODS:
            print(f"Unknown method: {method_name}")
            continue

        print(f"\n--- {method_name} ---")
        try:
            method_cls = METHODS[method_name]
            method = method_cls(wrapper)

            result = method.attribute(input_ids, attention_mask)

            if isinstance(result, tuple):
                scores, delta = result
                delta_val = delta.item() if hasattr(delta, "item") else delta
                print(f"Convergence delta: {delta_val:.4f}")
            else:
                scores = result

            scores_list = scores.tolist() if hasattr(scores, "tolist") else list(scores)
            text_words = text.split()
            word_names, word_scores = aggregate_to_words(tokens, scores_list, text_words)
            results[method_name] = {
                "tokens": tokens,
                "scores": scores_list,
                "words": word_names,
                "word_scores": word_scores,
                "pred_class": pred_class,
                "pred_prob": pred_prob,
            }

        except Exception as e:
            print(f"Error with {method_name}: {e}")
            results[method_name] = {"error": str(e)}

    return results


def save_results(results, output_path):
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Captum interpretability suite for BERT models")
    parser.add_argument("--model_id", type=str, default="BSC-LT/MrBERT-es")
    parser.add_argument("--checkpoint", type=str, default="baseline_es/checkpoint-20")
    parser.add_argument("--methods", type=str, nargs="+", default=None, help="Methods to run")
    parser.add_argument(
        "--output", type=str, default="captum_results.json", help="Output JSON path"
    )
    parser.add_argument(
        "--n_texts", type=int, default=None, help="Limit number of texts (default: all)"
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    tokenizer, model = get_tokenizer_model(args.model_id, checkpoint=args.checkpoint)
    model = model.to(device)
    model.eval()

    wrapper = ModelWrapper(model, tokenizer, device)

    _, df_test = load_data()
    texts = df_test.text_clean.str.replace("\n", " ").to_list()
    if args.n_texts is not None:
        texts = texts[: args.n_texts]

    all_results = {}
    for idx, text in tqdm(enumerate(texts)):
        print(f"\n{'=' * 60}")
        print(f"Text {idx + 1}/{len(texts)}")
        print(f"{'=' * 60}")
        try:
            results = run_all_methods(wrapper, text, args.methods)
            all_results[str(idx)] = {
                "text": text,
                "methods": results,
            }
        except Exception as e:
            print(f"Error processing text {idx}: {e}")
            all_results[str(idx)] = {"text": text, "error": str(e)}

        # Guarda incrementalment per no perdre dades si peta
        with open(args.output, "w") as f:
            json.dump(all_results, f, indent=2, ensure_ascii=False)

    print(f"\n✓ All results saved to {args.output}")


if __name__ == "__main__":
    main()
