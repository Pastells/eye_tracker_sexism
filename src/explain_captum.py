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
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["NCCL_P2P_DISABLE"] = "1"
os.environ["NCCL_IB_DISABLE"] = "1"

import argparse
import html as html_lib
import json

import numpy as np
import pandas as pd
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


def load_data(path="../data/mused_all_clean.csv", chosen_path="../data/mused_chosen_data.csv"):
    joined = pd.read_csv(path)[["id", "text_clean", "sexist", "sexist_soft"]]
    chosen = pd.read_csv(chosen_path)
    chosen_ids = chosen.id.to_list()
    df_test = joined[joined.id.isin(chosen_ids)]
    return df_test


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
    "integrated_gradients": IntegratedGradientsMethod,
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


def print_attributions(tokens, scores, top_k=10):
    pairs = list(zip(tokens, scores))
    pairs = [(format_token(t), s) for t, s in pairs]
    pairs_sorted = sorted(pairs, key=lambda x: abs(x[1]), reverse=True)

    print(f"\nTop {top_k} most important tokens:")
    for i, (tok, score) in enumerate(pairs_sorted[:top_k]):
        print(f"  {i + 1}. {tok:30s} {score:+.6f}")


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
            results[method_name] = {
                "tokens": tokens,
                "scores": scores_list,
                "pred_class": pred_class,
                "pred_prob": pred_prob,
            }

            print_attributions(tokens, scores_list)

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
    parser.add_argument("--text", type=str, default=None)
    parser.add_argument("--text_index", type=int, default=0, help="Index of text from test set")
    parser.add_argument("--methods", type=str, nargs="+", default=None, help="Methods to run")
    parser.add_argument("--output", type=str, default=None, help="Output JSON path")
    parser.add_argument("--layer", type=str, default=None, help="Layer for gradcam methods")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    tokenizer, model = get_tokenizer_model(args.model_id, checkpoint=args.checkpoint)
    model = model.to(device)
    model.eval()

    wrapper = ModelWrapper(model, tokenizer, device)

    if args.text:
        text = args.text
    else:
        df_test = load_data()
        texts = df_test.text_clean.to_list()
        text = texts[args.text_index]

    results = run_all_methods(wrapper, text, args.methods)

    if args.output:
        save_results(results, args.output)


if __name__ == "__main__":
    main()
