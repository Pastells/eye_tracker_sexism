"""
explain/gradient.py

Mètodes de saliència basats en gradients simples per a MLMClozeWrapper:
  1. SimpleGradient (Grad):        ||∂score/∂embedding||_2
  2. InputXGradient (Grad×Input):  ||embedding ⊙ ∂score/∂embedding||_2

Aquests són els mètodes més ràpids i la base conceptual d'IG.
SimpleGradient és equivalent a IG amb un sol pas i baseline=0,
però Grad×Input sol ser més estable i és el que més es fa servir
com a baseline de comparació.

Referència
----------
- Simonyan et al. (2013) "Deep Inside Convolutional Networks"
- Kindermans et al. (2016) "Investigating the influence of noise..."
- Samek et al. (2017) comparativa de mètodes de saliència
"""

from __future__ import annotations

from typing import Literal

import numpy as np
from torch import Tensor

from explain.base import Explainer, Explanation, TokenWordAlignment
from models.mlm import MLMClozeWrapper


class SimpleGradient(Explainer):
    """
    Saliència = norma L2 del gradient de l'score respecte als embeddings.

        s(t_i) = ||∂score / ∂e_i||_2

    Intuïció: mesura quant canviaria la predicció si pertorbéssim
    infinitesimalment l'embedding del token i.

    Avantatge: molt ràpid (1 forward + 1 backward).
    Desavantatge: no té en compte la magnitud de l'embedding
                  (pot saturar-se en zones planes de la funció).
    """

    method_name = "simple_gradient"

    def __init__(
        self,
        wrapper: MLMClozeWrapper,
        aggregation: Literal["sum", "mean", "max"] = "sum",
        normalize: bool = True,
    ):
        self.wrapper = wrapper
        self.aggregation = aggregation
        self.normalize = normalize

    def explain(self, text: str, prompt_idx: int = 0) -> Explanation:
        inputs, mask_pos = self.wrapper.build_inputs(text, prompt_idx=prompt_idx)
        inputs = {k: v.to(self.wrapper.device) for k, v in inputs.items()}

        embeds = self.wrapper.embed_inputs(inputs["input_ids"])  # [1, S, H]
        embeds.retain_grad()

        score = self.wrapper.target_score_from_embeds(
            embeds, inputs["attention_mask"], mask_pos
        )

        self.wrapper.model.zero_grad()
        score.backward()

        grad = embeds.grad
        if grad is None:
            raise RuntimeError(
                "El gradient és None. Comprova que no hi hagi .detach() "
                "o torch.no_grad() actiu al camí."
            )

        token_saliency = self._token_saliency(embeds, grad)
        return self._build_explanation(
            text, token_saliency, inputs, prompt_idx, score.item()
        )

    def _token_saliency(self, embeds: Tensor, grad: Tensor) -> Tensor:
        """Grad: ||∂score/∂e_i||_2"""
        return grad[0].norm(dim=-1)  # [S]

    def _build_explanation(
        self,
        text: str,
        token_saliency: Tensor,
        inputs: dict,
        prompt_idx: int,
        target_score: float,
    ) -> Explanation:
        # TODO: afegir _build_alignment o posar a base, ara no hi és
        alignment: TokenWordAlignment = self._build_alignment(text, inputs, prompt_idx)

        token_sal_arr = np.nan_to_num(token_saliency.detach().cpu().numpy(), nan=0.0)

        # Zero als tokens especials
        special_ids = set(self.wrapper.tokenizer.all_special_ids)
        input_ids_list = inputs["input_ids"][0].tolist()
        for i, tid in enumerate(input_ids_list):
            if tid in special_ids:
                token_sal_arr[i] = 0.0

        token_sal_list = token_sal_arr.tolist()
        word_saliency = alignment.aggregate(token_sal_list, reduce=self.aggregation)

        if self.normalize:
            arr = np.nan_to_num(np.array(word_saliency, dtype=float), nan=0.0)
            max_sal = arr.max() if arr.size else 1.0
            if max_sal > 0:
                arr = arr / max_sal
            word_saliency = arr.tolist()

        return Explanation(
            text=text,
            words=alignment.words,
            word_saliency=word_saliency,
            tokens=alignment.tokens,
            token_saliency=token_sal_list,
            target_score=target_score,
            method=self.method_name,
            model_name=self.wrapper.name,
            extra={"prompt_idx": prompt_idx},
        )


class InputXGradient(SimpleGradient):
    """
    Saliència = norma L2 de (embedding ⊙ gradient).

        s(t_i) = ||e_i ⊙ ∂score/∂e_i||_2

    Millora sobre SimpleGradient: té en compte la magnitud de l'embedding.
    Si el gradient és gran però l'embedding és petit, la saliència
    resultant és petita — típicament més informatiu.
    """

    method_name = "input_x_gradient"

    def _token_saliency(self, embeds: Tensor, grad: Tensor) -> Tensor:
        input_x_grad = embeds[0].detach() * grad[0]  # [S, H]
        return input_x_grad.norm(dim=-1)  # [S]
