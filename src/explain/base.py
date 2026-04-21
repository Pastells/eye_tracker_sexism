"""
explain/base.py

Interfície unificada per a mètodes d'explicabilitat a través de diferents
arquitectures (encoder MLM, encoder NLI, encoder fine-tuned, LLM generatiu).

Idea clau
---------
Tot mètode de saliència (gradient, IG, attention, occlusion...) necessita:

  1. Un *model forward* que, donat un text, produeixi un **escalar** (el
     "target" sobre el qual mesurar la importància). Aquest escalar
     representa la confiança del model en la classe d'interès.
  2. Un mapping *token → paraula* per agregar saliències de subtokens a
     paraules (unitat comparable amb l'eye-tracking).
  3. Accés als embeddings d'entrada (per gradient-based) o a les matrius
     d'atenció (per attention-based).

Aquest mòdul defineix:
  - `ModelWrapper`: abstracció sobre HF models que exposa (1), (2), (3).
  - `Explainer`:    classe base per a mètodes d'explicabilitat.
  - `Explanation`:  estructura de dades per al resultat (saliència per paraula).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Literal, Protocol

import torch
from torch import Tensor, nn
from transformers import PreTrainedModel, PreTrainedTokenizerBase

# =============================================================================
# Estructures de dades
# =============================================================================


@dataclass
class Explanation:
    """
    Resultat d'un mètode d'explicabilitat sobre un text.

    Attributes
    ----------
    text : str
        Text original.
    words : list[str]
        Paraules (unitat d'anàlisi, alineable amb eye-tracking).
    word_saliency : list[float]
        Saliència per paraula (mateixa longitud que `words`).
    tokens : list[str]
        Subtokens del model.
    token_saliency : list[float]
        Saliència per subtoken (abans d'agregar).
    target_score : float
        Valor de l'escalar-target utilitzat (útil per debugging).
    method : str
        Nom del mètode ("grad", "ig", "attention", "occlusion"...).
    model_name : str
        Identificador del model.
    extra : dict
        Informació addicional específica del mètode.
    """

    text: str
    words: list[str]
    word_saliency: list[float]
    tokens: list[str]
    token_saliency: list[float]
    target_score: float
    method: str
    model_name: str
    extra: dict = field(default_factory=dict)

    def normalized(self, method: Literal["minmax", "sum", "none"] = "minmax") -> "Explanation":
        """Retorna una còpia amb `word_saliency` normalitzat."""
        import numpy as np

        s = np.array(self.word_saliency, dtype=float)
        if method == "minmax":
            rng = s.max() - s.min()
            s = (s - s.min()) / rng if rng > 0 else np.zeros_like(s)
        elif method == "sum":
            tot = s.sum()
            s = s / tot if tot > 0 else np.zeros_like(s)
        return Explanation(
            text=self.text,
            words=self.words,
            word_saliency=s.tolist(),
            tokens=self.tokens,
            token_saliency=self.token_saliency,
            target_score=self.target_score,
            method=self.method,
            model_name=self.model_name,
            extra={**self.extra, "normalization": method},
        )


@dataclass
class TokenWordAlignment:
    """
    Alineació entre subtokens del tokenitzador i paraules del text original.

    `token_to_word[i] = j` vol dir que el subtoken `i` pertany a la paraula `j`.
    Els tokens especials (CLS, SEP, PAD, tokens de prompt) tenen `-1`.
    """

    tokens: list[str]
    words: list[str]
    token_to_word: list[int]  # -1 per tokens especials/prompt

    def aggregate(
        self,
        token_scores: Tensor | list[float],
        reduce: Literal["sum", "mean", "max"] = "sum",
    ) -> list[float]:
        """Agrega saliències de subtokens a nivell de paraula."""
        import numpy as np

        scores = np.asarray(token_scores, dtype=float)
        word_scores = np.zeros(len(self.words), dtype=float)
        counts = np.zeros(len(self.words), dtype=int)

        for tok_idx, w_idx in enumerate(self.token_to_word):
            if w_idx < 0:
                continue
            if reduce == "max":
                word_scores[w_idx] = (
                    max(word_scores[w_idx], scores[tok_idx])
                    if counts[w_idx] > 0
                    else scores[tok_idx]
                )
            else:  # sum o mean (acumulem i dividim al final si cal)
                word_scores[w_idx] += scores[tok_idx]
            counts[w_idx] += 1

        if reduce == "mean":
            word_scores = np.where(counts > 0, word_scores / np.maximum(counts, 1), 0.0)

        return word_scores.tolist()


# =============================================================================
# ModelWrapper: abstracció sobre els diferents casos d'ús
# =============================================================================


class ModelWrapper:
    """
    Abstracció unificada sobre un model HuggingFace per a explicabilitat.

    Cada subclasse implementa:
        - target_score(text) -> scalar_tensor
    """

    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizerBase,
        device: str | torch.device = "cuda",
        name: str | None = None,
    ):
        self.model = model.to(device).eval()
        self.tokenizer = tokenizer
        self.device = torch.device(device)
        self.name = name or model.config._name_or_path

    @property
    def embeddings_layer(self) -> nn.Module:
        return self.model.get_input_embeddings()

    def embed_inputs(self, input_ids: Tensor) -> Tensor:
        """input_ids [B, S] → embeddings [B, S, H]"""
        return self.embeddings_layer(input_ids.to(self.device))

    def forward_from_embeds(
        self,
        inputs_embeds: Tensor,
        attention_mask: Tensor,
        **extra,
    ) -> Tensor:
        """Forward des d'embeddings (per IG). Retorna logits."""
        return self.model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            **extra,
        ).logits

    def forward_with_attentions(
        self, inputs: dict[str, Tensor]
    ) -> tuple[Tensor, tuple[Tensor, ...]]:
        """Forward retornant (logits, attentions). Per attention-based explainers."""
        outputs = self.model(**inputs, output_attentions=True)
        return outputs.logits, outputs.attentions


# =============================================================================
# Explainer: classe base per a mètodes d'explicabilitat
# =============================================================================


class Explainer(ABC):
    """
    Classe base. Un `Explainer` pren un `ModelWrapper` i produeix `Explanation`s.

    Els mètodes concrets (IG, Grad, Attention, Occlusion) hereten d'aquí.
    """

    def __init__(
        self,
        model: ModelWrapper,
        aggregation: Literal["sum", "mean", "max"] = "sum",
    ):
        self.model = model
        self.aggregation = aggregation

    @property
    @abstractmethod
    def method_name(self) -> str: ...

    @abstractmethod
    def _compute_token_saliency(
        self,
        inputs: dict[str, Tensor],
        alignment: TokenWordAlignment,
        target_class: int | str | None,
    ) -> tuple[list[float], float, dict]:
        """
        Implementació específica del mètode.

        Returns
        -------
        token_saliency : list[float]
            Saliència per subtoken (mateixa longitud que `alignment.tokens`).
        target_score : float
            Valor de l'escalar-target (per debug).
        extra : dict
            Informació addicional.
        """
        ...

    def explain(
        self,
        text: str,
        target_class: int | str | None = None,
    ) -> Explanation:
        """Pipeline estàndard: build inputs → compute saliency → aggregate."""
        inputs, alignment = self.model.build_inputs(text)
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

        token_saliency, target_score, extra = self._compute_token_saliency(
            inputs, alignment, target_class
        )

        word_saliency = alignment.aggregate(token_saliency, reduce=self.aggregation)

        return Explanation(
            text=text,
            words=alignment.words,
            word_saliency=word_saliency,
            tokens=alignment.tokens,
            token_saliency=token_saliency,
            target_score=target_score,
            method=self.method_name,
            model_name=self.model.name,
            extra=extra,
        )

    def explain_batch(
        self,
        texts: list[str],
        target_class: int | str | None = None,
    ) -> list[Explanation]:
        """Versió batch ingènua (la pots sobreescriure per eficiència)."""
        return [self.explain(t, target_class) for t in texts]


# =============================================================================
# Protocol per tipatge extern (ex: pipelines de comparació)
# =============================================================================


class ExplainerProtocol(Protocol):
    """Per tipatge estructural, si algú no vol heretar d'`Explainer`."""

    def explain(self, text: str, target_class: int | str | None = ...) -> Explanation: ...
