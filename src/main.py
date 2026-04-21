from __future__ import annotations

from explain.ig import IntegratedGradients
from models.mlm import MLMClozeWrapper

DEVICE = "cuda"

# from models.seq_cls import SeqClassifierWrapper

# --- Fine-tuned ---
# ft_wrapper = SeqClassifierWrapper.from_pretrained_finetuned(
#     "projecte-aina/roberta-base-ca-v2",  # substitueix pel teu model fine-tuned
# )
# print(ft_wrapper.predict_proba(TEXT))

# --- NLI zero-shot ---
# nli_wrapper = SeqClassifierWrapper.from_pretrained_nli(
#     "MoritzLaurer/mDeBERTa-v3-base-mnli-xnli",
#     hypothesis="Este texto es sexista.",
# )
# print(nli_wrapper.predict_proba(TEXT))


# --- Totes tres generen inputs compatibles amb qualsevol Explainer ---
# for wrapper in [ft_wrapper, nli_wrapper, mlm_wrapper]:
#     inputs, alignment = wrapper.build_inputs(TEXT)
#     print(f"\n{wrapper.name}")
#     print(alignment)
#     print(f"  Tokens:      {alignment.tokens}")
#     print(f"  Paraules:    {alignment.words}")
#     print(f"  tok→word:    {alignment.token_to_word}")


MODEL_NAME = "BSC-LT/MrBERT-es"

TEXTS = [
    "Las mujeres no deberían trabajar fuera de casa.",
    "Todos merecen igualdad de oportunidades en el trabajo.",
    "Las chicas no sirven para las matemáticas.",
    "El equipo está formado por personas muy competentes.",
]


def print_explanation(exp):
    print(f"\n  Mètode: {exp.method}")
    print(f"  Score:  {exp.target_score:+.4f}")
    for word, sal in zip(exp.words, exp.word_saliency):
        bar = "█" * int(sal * 20)
        print(f"    {word:20s} {sal:.3f} {bar}")


def mlm():
    wrapper = MLMClozeWrapper(model_name=MODEL_NAME)

    # grad_explainer = SimpleGradient(wrapper, aggregation="sum", normalize=True)
    # ixg_explainer = InputXGradient(wrapper, aggregation="sum", normalize=True)
    ig_explainer = IntegratedGradients(
        wrapper,
        n_steps=50,
        baseline="zero",
        aggregation="sum",
        normalize=True,
    )

    print("\n" + "=" * 70)
    print(f"Model: {MODEL_NAME}")
    print("=" * 70)

    for text in TEXTS:
        # --- Classificació (mitjana dels 2 prompts) ---
        scores = wrapper.target_score(text)  # tensor [2]
        mean_score = scores.mean().item()
        pred = "SEXISTA" if mean_score > 0 else "NO SEXISTA"

        print(f"\nText: {text}")
        for i, s in enumerate(scores):
            print(f"  Prompt {i}: {s.item():+.4f}")
        print(f"  Mitjana: {mean_score:+.4f}  → {pred}")

        # --- Explicabilitat per cada prompt ---
        for prompt_idx in range(len(wrapper.PROMPT_TEMPLATES)):
            # IG ja sobreescriu explain() i gestiona mask_pos internament
            # Però necessitem dir-li quin prompt fer servir.
            # Usem target_class=prompt_idx com a convenció,
            # o millor: cridem explain amb el text construït manualment.
            inputs, mask_pos = wrapper.build_inputs(text, prompt_idx=prompt_idx)
            inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

            # exp_grad = grad_explainer.explain(text, mask_pos=mask_pos)
            # exp_ixg = ixg_explainer.explain(text, mask_pos=mask_pos)
            exp_ig = ig_explainer.explain(text, prompt_idx=prompt_idx)
            print(f"\n  [Prompt {prompt_idx}]", end="")
            # print_explanation(exp_grad)
            # print_explanation(exp_ixg)
            print_explanation(exp_ig)
            gap = exp_ig.extra.get("completeness_gap", float("nan"))
            print(f"    IG completeness gap: {gap:.4f}")

    print("\n" + "=" * 70)


if __name__ == "__main__":
    mlm()
