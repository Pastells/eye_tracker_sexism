from models.mlm import MLMClozeWrapper

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


MODEL_NAME = "dccuchile/bert-base-spanish-wwm-cased"
MODEL_NAME = "BSC-LT/MrBERT-es"
# MODEL_NAME = "PlanTL-GOB-ES/roberta-base-bne"

TEXTS = [
    "Las mujeres no deberían trabajar fuera de casa.",
    "Todos merecen igualdad de oportunidades en el trabajo.",
    "Las chicas no sirven para las matemáticas.",
    "El equipo está formado por personas muy competentes.",
]


def mlm():
    wrapper = MLMClozeWrapper(model_name=MODEL_NAME)

    print("\n" + "=" * 70)
    print(f"Model: {MODEL_NAME}")
    print("=" * 70)

    for text in TEXTS:
        scores = wrapper.target_score(text)  # tensor [2]
        mean_score = scores.mean().item()
        pred = "SEXISTA" if mean_score > 0 else "NO SEXISTA"

        print(f"\nText: {text}")
        for i, s in enumerate(scores):
            print(f"  Prompt {i}: {s.item():+.4f}")
        print(f"  Mitjana: {mean_score:+.4f}  → {pred}")

    print("\n" + "=" * 70)


if __name__ == "__main__":
    mlm()
