"""
models/mlm.py

ModelWrapper per a RoBERTa (o qualsevol model MLM) en mode zero-shot cloze.

Estratègia
----------
Donat un text, construïm un input de la forma:

    [CLS] <prompt_prefix> [MASK] <prompt_suffix> [SEP] <text> [SEP]

Per exemple:
    "[CLS] Aquest text és [MASK]. [SEP] El director va dir que les dones
     no saben conduir. [SEP]"

El target_score és el logit de la classe positiva (p.ex. "sexista")
a la posició del token [MASK].

El text original és la part després el separador, i és l'única part que
s'alinea amb l'eye-tracking (la resta és prompt → token_to_word = -1).
"""

from __future__ import annotations

import torch
from torch import Tensor
from transformers import (
    AutoModelForMaskedLM,
    AutoTokenizer,
)

from explain.base import ModelWrapper

# ── Tokens per model ──────────────────────────────────────────────────────────
# Executa el bloc de diagnòstic primer per verificar quins tokens fer servir


class MLMClozeWrapper(ModelWrapper):
    """
    Zero-shot cloze via Masked Language Modeling.

    score = log p([MASK] = '1') - log p([MASK] = '0')

    Parameters
    ----------
    model, tokenizer : model HuggingFace MLM (RoBERTa, XLM-R, BERT...)
    prompt_prefix : text ABANS del [MASK], p.ex. "Aquest text és"
    prompt_suffix : text DESPRÉS del [MASK], p.ex. "."
    text_first     : si True, el text va ABANS del prompt (algunes tasques
                     funcionen millor així). Default: False (prompt primer).
    device, name   : com a ModelWrapper.
    """

    def __init__(self, model_name: str, device: str = "cpu"):
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForMaskedLM.from_pretrained(model_name)
        super().__init__(model=model, tokenizer=tokenizer, device=device)

        # Obtenim els token IDs de '0' i '1'
        # for word in ["0", "1", "Sí", "No", "si", "no", "S", "N"]:
        #     for variant in [word, " " + word]:
        #         ids = self.tokenizer.encode(variant, add_special_tokens=False)
        #         print(f"{repr(variant):10} → {ids} ({len(ids)} tokens)")
        print(f"[DIAGNÒSTIC] mask_token: {repr(self.tokenizer.mask_token)}")

        if model_name == "dccuchile/bert-base-spanish-wwm-cased":
            TOKEN_POS = "Sí"
            TOKEN_NEG = "No"
            MASK = "[MASK]"
        elif model_name == "PlanTL-GOB-ES/roberta-base-bne":
            TOKEN_POS = "Sí"
            TOKEN_NEG = "No"
            MASK = "[MASK]"
        else:
            TOKEN_POS = "Sí"
            TOKEN_NEG = "No"
            MASK = "<mask>"

        self.PROMPT_TEMPLATES = [
            f"Responde con Sí o No. ¿El siguiente texto es sexista? '{{text}}' {MASK}.",
            f"Analiza si el texto contiene sexismo o discriminación de género. Texto: '{{text}}' El texto es sexista: {MASK}.",
        ]
        self.token_id_0 = self._get_single_token_id(TOKEN_POS)
        self.token_id_1 = self._get_single_token_id(TOKEN_NEG)
        print(
            f"[MLMClozeWrapper] token_id('0') = {self.token_id_0}, "
            f"token_id('1') = {self.token_id_1}"
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _get_single_token_id(self, word: str) -> int:

        for variant in [word, " " + word]:
            ids = self.tokenizer.encode(variant, add_special_tokens=False)
            if len(ids) == 1:
                return ids[0]

        raise ValueError(
            f"'{word}' no es pot tokenitzar com a un sol token. Prova un altre verbalizador."
        )

    def build_inputs(self, text: str, prompt_idx: int = 0) -> tuple[dict[str, Tensor], int]:
        prompt = self.PROMPT_TEMPLATES[prompt_idx].format(text=text)
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        mask_pos = (
            (inputs["input_ids"][0] == self.tokenizer.mask_token_id)
            .nonzero(as_tuple=True)[0][0]
            .item()
        )
        return inputs, mask_pos

    def embed_inputs(self, input_ids: Tensor) -> Tensor:
        """input_ids [1, S] → embeddings [1, S, H]"""
        emb_layer = self.model.get_input_embeddings()
        return emb_layer(input_ids)

    # ------------------------------------------------------------------
    # Scoring
    # ------------------------------------------------------------------

    def target_score(self, text: str, prompt_idx: int | None = None) -> Tensor:
        """
        Si prompt_idx és None, retorna tensor [N] amb el score de cada prompt.
        Si prompt_idx és un enter, retorna escalar per aquell prompt.
        """
        indices = [prompt_idx] if prompt_idx is not None else range(len(self.PROMPT_TEMPLATES))
        scores = []
        for i in indices:
            inputs, mask_pos = self.build_inputs(text, prompt_idx=i)
            with torch.no_grad():
                logits = self.model(**inputs).logits
            log_probs = torch.log_softmax(logits[0, mask_pos, :], dim=-1)
            # print(log_probs[self.token_id_1], log_probs[self.token_id_0])
            scores.append(log_probs[self.token_id_1] - log_probs[self.token_id_0])
        result = torch.stack(scores)
        return result[0] if prompt_idx is not None else result

    def target_score_from_embeds(
        self,
        embeds: Tensor,  # [1, S, H]
        attention_mask: Tensor,  # [1, S]
        mask_pos: int,
    ) -> Tensor:
        """Escalar diferenciable respecte a embeds."""
        logits = self.model(
            inputs_embeds=embeds,
            attention_mask=attention_mask,
        ).logits  # [1, S, V]
        log_probs = torch.log_softmax(logits[0, mask_pos, :], dim=-1)
        return log_probs[self.token_id_1] - log_probs[self.token_id_0]

    def target_score_from_embeds_batch(
        self,
        embeds: Tensor,  # [B, S, H]
        attention_mask: Tensor,  # [B, S]
        mask_pos: int,
    ) -> Tensor:
        """Vector [B] diferenciable respecte a embeds. Per a IG batch."""
        logits = self.model(
            inputs_embeds=embeds,
            attention_mask=attention_mask,
        ).logits  # [B, S, V]
        log_probs = torch.log_softmax(logits[:, mask_pos, :], dim=-1)
        return log_probs[:, self.token_id_1] - log_probs[:, self.token_id_0]
