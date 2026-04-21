"""
models/seq_cls.py

ModelWrapper per a models de classificació de seqüències (SequenceClassification).

Cobreix dos casos:
  1. RoBERTa fine-tuned en la teva tasca (classification head entrenat).
  2. RoBERTa-NLI zero-shot (classification head entrenat en MNLI/XNLI,
     adaptat a zero-shot via premissa-hipòtesi).

Diferència clau respecte a MLMClozeWrapper
------------------------------------------
Aquí NO hi ha [MASK]. El model retorna logits directament sobre classes.
El target_score és el logit de la classe d'interès (índex configurable).

Per al cas NLI, l'input és un parell (text, hipòtesi) i el logit
d'interès és el de 'entailment' (típicament índex 2 en models MNLI).
"""

from __future__ import annotations

import torch
from torch import Tensor
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizerBase,
)

from explain.base import ModelWrapper, TokenWordAlignment
from models.utils import _assign_word_indices_simple, _whitespace_tokenize


class SeqClassifierWrapper(ModelWrapper):
    """
    Wrapper per a `AutoModelForSequenceClassification`.

    Funciona tant per a fine-tuned com per a NLI zero-shot.

    Parameters
    ----------
    model, tokenizer : model HuggingFace SeqClassification
    target_class_idx : índex de la classe d'interès en `model.config.id2label`.
                       - Fine-tuned binary: 0 o 1 (1 = sexista típicament).
                       - NLI: índex d'entailment (veure `nli_entailment_idx`).
    nli_hypothesis   : si no és None, activa el mode NLI. El text va com
                       a premissa i `nli_hypothesis` com a hipòtesi.
                       Ex: "Aquest text és sexista."
    max_length       : longitud màxima de tokenització.
    device, name     : com a ModelWrapper.
    """

    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizerBase,
        target_class_idx: int = 1,
        nli_hypothesis: str | None = None,
        max_length: int = 256,
        device: str | torch.device = "cuda",
        name: str | None = None,
    ):
        super().__init__(model, tokenizer, device, name)
        self.target_class_idx = target_class_idx
        self.nli_hypothesis = nli_hypothesis
        self.max_length = max_length
        self._is_nli = nli_hypothesis is not None

        # Per NLI, pre-tokenitzem la hipòtesi per saber quants tokens té
        if self._is_nli:
            self._hypothesis_tokens = tokenizer.tokenize(nli_hypothesis)

    # ------------------------------------------------------------------
    # build_inputs
    # ------------------------------------------------------------------

    def build_inputs(self, text: str) -> tuple[dict[str, Tensor], TokenWordAlignment]:
        """
        Mode fine-tuned
        ---------------
        Input: [CLS] text [SEP]
        Tots els tokens del text s'alineen amb paraules.

        Mode NLI
        --------
        Input: [CLS] text [SEP] hipòtesi [SEP]
        Només els tokens del text (premissa) s'alineen amb paraules.
        Els tokens de la hipòtesi → -1.
        """
        tok = self.tokenizer
        words = _whitespace_tokenize(text)
        text_tokens = tok.tokenize(text)

        if not self._is_nli:
            # [CLS] text [SEP]
            all_tokens = [tok.cls_token] + text_tokens + [tok.sep_token]
            text_start, text_end = 1, 1 + len(text_tokens)
        else:
            # [CLS] text [SEP] hipòtesi [SEP]
            hyp_tokens = self._hypothesis_tokens
            all_tokens = (
                [tok.cls_token] + text_tokens + [tok.sep_token] + hyp_tokens + [tok.sep_token]
            )
            text_start, text_end = 1, 1 + len(text_tokens)

        # Truncat si cal
        if len(all_tokens) > self.max_length:
            # Truncem el text (no el prompt/hipòtesi)
            overflow = len(all_tokens) - self.max_length
            text_tokens = text_tokens[: len(text_tokens) - overflow]
            text_end = text_start + len(text_tokens)
            if not self._is_nli:
                all_tokens = [tok.cls_token] + text_tokens + [tok.sep_token]
            else:
                all_tokens = (
                    [tok.cls_token]
                    + text_tokens
                    + [tok.sep_token]
                    + self._hypothesis_tokens
                    + [tok.sep_token]
                )

        input_ids = tok.convert_tokens_to_ids(all_tokens)
        attention_mask = [1] * len(input_ids)

        inputs = {
            "input_ids": torch.tensor([input_ids]),
            "attention_mask": torch.tensor([attention_mask]),
        }

        # --- Alineació ---
        token_to_word = [-1] * len(all_tokens)
        _assign_word_indices_simple(tok, all_tokens, text_start, text_end, words, token_to_word)

        alignment = TokenWordAlignment(
            tokens=all_tokens,
            words=words,
            token_to_word=token_to_word,
        )
        return inputs, alignment

    # ------------------------------------------------------------------
    # target_score
    # ------------------------------------------------------------------

    def target_score(
        self,
        inputs: dict[str, Tensor],
        target_class: int | str | None = None,
    ) -> Tensor:
        """
        Retorna el logit de `target_class_idx` (o de `target_class` si
        s'especifica com a enter).

        Per al cas NLI, `target_class_idx` hauria de ser l'índex
        d'entailment del model (veure `_find_entailment_idx`).
        """
        idx = target_class if isinstance(target_class, int) else self.target_class_idx

        logits = self.forward_from_embeds(
            self.embed_inputs(inputs["input_ids"]),
            inputs["attention_mask"],
        )
        # logits shape: [1, num_classes]
        return logits[0, idx]

    def target_score_from_embeds(self, embeds, attention_mask, target_class=None):
        idx = target_class if isinstance(target_class, int) else self.target_class_idx
        out = self.model(
            inputs_embeds=embeds,
            attention_mask=attention_mask.to(self.device),
        )
        return out.logits[0, idx]

    def target_score_from_embeds_batch(self, embeds, attention_mask, target_class=None):
        idx = target_class if isinstance(target_class, int) else self.target_class_idx
        out = self.model(
            inputs_embeds=embeds,
            attention_mask=attention_mask.to(self.device),
        )
        return out.logits[:, idx]  # [batch]

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------

    @torch.no_grad()
    def predict_proba(self, text: str) -> dict[str, float]:
        """Retorna probabilitats per a totes les classes."""
        import torch.nn.functional as F

        inputs, _ = self.build_inputs(text)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        logits = self.model(**inputs).logits
        probs = F.softmax(logits[0], dim=-1)

        id2label = self.model.config.id2label
        return {id2label[i]: probs[i].item() for i in range(len(probs))}

    @staticmethod
    def _find_entailment_idx(model: PreTrainedModel) -> int:
        """
        Intenta trobar automàticament l'índex d'entailment mirant
        `model.config.id2label`.
        Retorna 0 si no el troba (log warning).
        """
        id2label = model.config.id2label
        for idx, label in id2label.items():
            if "entail" in label.lower():
                return idx
        print(
            "[SeqClassifierWrapper] Warning: no s'ha trobat 'entailment' a "
            f"id2label={id2label}. Fent servir idx=0."
        )
        return 0

    # ------------------------------------------------------------------
    # Factory classmethods
    # ------------------------------------------------------------------

    @classmethod
    def from_pretrained_finetuned(
        cls,
        model_name: str,
        target_class_idx: int = 1,
        device: str = "cuda",
        **kwargs,
    ) -> "SeqClassifierWrapper":
        """Carrega un model fine-tuned per a classificació binària."""
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSequenceClassification.from_pretrained(model_name)
        return cls(
            model=model,
            tokenizer=tokenizer,
            target_class_idx=target_class_idx,
            nli_hypothesis=None,
            device=device,
            name=model_name,
            **kwargs,
        )

    @classmethod
    def from_pretrained_nli(
        cls,
        model_name: str,
        hypothesis: str = "Aquest text és sexista.",
        device: str = "cuda",
        **kwargs,
    ) -> "SeqClassifierWrapper":
        """
        Carrega un model NLI per a zero-shot classification.

        Models recomanats:
          - "MoritzLaurer/mDeBERTa-v3-base-mnli-xnli"  (multilingüe)
          - "cross-encoder/nli-deberta-v3-large"        (anglès, molt bo)
          - "joeddav/xlm-roberta-large-xnli"            (multilingüe)
        """
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSequenceClassification.from_pretrained(model_name)
        entailment_idx = cls._find_entailment_idx.__func__(model)
        return cls(
            model=model,
            tokenizer=tokenizer,
            target_class_idx=entailment_idx,
            nli_hypothesis=hypothesis,
            device=device,
            name=f"{model_name}[NLI]",
            **kwargs,
        )
