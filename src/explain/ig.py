"""
explain/ig.py

Integrated Gradients (IG) per a MLMClozeWrapper.

IG resol el problema de la "saturació" dels gradients simples:
en zones de la funció on el gradient és quasi zero (el model és molt
segur o molt equivocat), Grad retorna saliències baixes per a tot,
fins i tot si el token és important.

IG integra el gradient al llarg d'una línia recta entre un
baseline (embedding de referència, típicament zeros o [PAD]) i
l'embedding real:

    IG(x_i) = (x_i - x'_i) · ∫₀¹ ∂F(x' + α(x-x')) / ∂x_i  dα

On:
  - x_i  : embedding del token i
  - x'_i : embedding del baseline (típicament zeros)
  - F    : target_score (escalar)
  - α    : paràmetre d'interpolació ∈ [0, 1]

L'integral s'aproxima amb suma de Riemann sobre `n_steps` passos,
processats en mini-batches de mida `batch_size` per evitar OOM.

Propietats importants d'IG
--------------------------
1. Completeness: sum(IG) ≈ F(x) - F(x')  [verificable → "completeness gap"]
2. Sensitivity:  si canviar x_i canvia F, llavors IG(x_i) ≠ 0
3. Linearity:    IG és lineal respecte a F

Baselines suportats
-------------------
"zero"  → zeros (més comú a la literatura NLP)
"pad"   → embedding del token PAD (semànticament neutre)
"mask"  → embedding del token MASK (interessant per MLM)
"noise" → soroll gaussià petit

Referència
----------
Sundararajan et al. (2017) "Axiomatic Attribution for Deep Networks"
https://arxiv.org/abs/1703.01365
"""

from __future__ import annotations

import math
from typing import Literal

import numpy as np
import torch
from torch import Tensor

from explain.base import Explanation, TokenWordAlignment
from models.mlm import MLMClozeWrapper


class IntegratedGradients:
    """
    Integrated Gradients adaptat a MLMClozeWrapper.

    Parameters
    ----------
    wrapper      : MLMClozeWrapper
    n_steps      : nombre de passos de Riemann (50–300 recomanat)
    baseline     : "zero" | "pad" | "mask" | "noise"
    batch_size   : passos per forward pass (evita OOM)
    aggregation  : com agregar subtokens a paraules
    normalize    : normalitza saliències a [0, 1]
    """

    method_name = "integrated_gradients"

    def __init__(
        self,
        wrapper: MLMClozeWrapper,
        n_steps: int = 100,
        baseline: Literal["zero", "pad", "mask", "noise"] = "zero",
        batch_size: int = 16,
        aggregation: Literal["sum", "mean", "max"] = "sum",
        normalize: bool = True,
    ):
        self.wrapper = wrapper
        self.n_steps = n_steps
        self.baseline_type = baseline
        self.batch_size = batch_size
        self.aggregation = aggregation
        self.normalize = normalize

    # ------------------------------------------------------------------
    # Baseline
    # ------------------------------------------------------------------

    def _build_baseline(self, embeds: Tensor, input_ids: Tensor) -> Tensor:
        """Retorna baseline [1, S, H] del mateix device que embeds."""
        device = embeds.device
        seq_len = embeds.shape[1]

        if self.baseline_type == "zero":
            return torch.zeros_like(embeds)

        elif self.baseline_type == "pad":
            pad_id = self.wrapper.tokenizer.pad_token_id or 0
            pad_ids = torch.full((1, seq_len), pad_id, dtype=torch.long, device=device)
            with torch.no_grad():
                return self.wrapper.embed_inputs(pad_ids)

        elif self.baseline_type == "mask":
            mask_id = getattr(self.wrapper.tokenizer, "mask_token_id", None)
            if mask_id is None:
                print("[IG] Warning: mask_token_id no existeix, fent servir zeros.")
                return torch.zeros_like(embeds)
            mask_ids = torch.full((1, seq_len), mask_id, dtype=torch.long, device=device)
            with torch.no_grad():
                return self.wrapper.embed_inputs(mask_ids)

        elif self.baseline_type == "noise":
            return torch.randn_like(embeds) * 0.01

        else:
            raise ValueError(f"Baseline desconegut: {self.baseline_type}")

    # ------------------------------------------------------------------
    # Core IG
    # ------------------------------------------------------------------

    def _compute_ig(
        self,
        embeds: Tensor,  # [1, S, H]
        baseline: Tensor,  # [1, S, H]
        attention_mask: Tensor,  # [1, S]
        mask_pos: int,
    ) -> tuple[Tensor, float, float]:
        """
        Retorna:
          ig            : [1, S, H]  gradients integrats
          score_real    : float
          score_base    : float
        """
        device = embeds.device
        # Interpolació: alpha = 0/n ... n/n
        alphas = torch.linspace(1 / self.n_steps, 1, self.n_steps, device=device)

        # Acumulem gradients
        grad_accum = torch.zeros_like(embeds)  # [1, S, H]

        for start in range(0, self.n_steps, self.batch_size):
            end = min(start + self.batch_size, self.n_steps + 1)
            alpha_batch = alphas[start:end]  # [b]
            b = alpha_batch.shape[0]

            # Interpolar: [b, S, H]
            interp = (
                (baseline + alpha_batch[:, None, None] * (embeds - baseline))
                .detach()
                .requires_grad_(True)
            )

            # Expandir attention_mask a [b, S]
            attn = attention_mask.expand(b, -1)

            scores = self.wrapper.target_score_from_embeds_batch(interp, attn, mask_pos)  # [b]
            scores.sum().backward()

            grad_accum += interp.grad.sum(dim=0, keepdim=True)  # [1, S, H]

        # Riemann: (x - x') * mean_gradient
        ig = (embeds - baseline) * grad_accum / self.n_steps

        # Scores per completeness gap
        with torch.no_grad():
            score_real = self.wrapper.target_score_from_embeds(
                embeds, attention_mask, mask_pos
            ).item()
            score_base = self.wrapper.target_score_from_embeds(
                baseline, attention_mask, mask_pos
            ).item()

        return ig, score_real, score_base

    # ------------------------------------------------------------------
    # TokenWordAlignment per MLM
    # ------------------------------------------------------------------

    def _build_alignment(
        self, text: str, inputs: dict, prompt_idx: int, debug=False
    ) -> TokenWordAlignment:
        tokenizer = self.wrapper.tokenizer
        template = self.wrapper.PROMPT_TEMPLATES[prompt_idx]
        before, after = template.split("{text}")

        input_ids = inputs["input_ids"][0].tolist()
        tokens = tokenizer.convert_ids_to_tokens(input_ids)
        words = text.split()

        before_ids = tokenizer(before, add_special_tokens=False)["input_ids"]
        before_plus_text_ids = tokenizer(before + text, add_special_tokens=False)["input_ids"]
        n_text_tokens = len(before_plus_text_ids) - len(before_ids)

        n_special_start = 1
        text_start = n_special_start + len(before_ids)
        text_end = text_start + n_text_tokens

        if debug:
            print("\n=== _build_alignment DEBUG ===")
            print(f"text_start={text_start}, text_end={text_end}")
            print(f"Text tokens: {tokens[text_start:text_end]}")

        # Alineació manual: reconstruïm el text token a token i trobem a quina paraula pertany cada token
        # Convertim els tokens del text a strings netes (sense ▁)
        text_tokens = tokens[text_start:text_end]

        # Construïm una cadena concatenada de paraules amb els seus índexs de caràcter
        # word_char_ranges[j] = (start, end) del caràcter de la paraula j dins la concatenació
        concat = ""
        word_char_ranges = []
        for _, w in enumerate(words):
            start = len(concat)
            concat += w
            word_char_ranges.append((start, len(concat)))

        if debug:
            print(f"concat: {concat!r}")
            print(f"word_char_ranges: {word_char_ranges}")

        # Recorrem els tokens del text, reconstruint la posició de caràcter
        token_to_word_text = []
        char_pos = 0
        for tok in text_tokens:
            # Neteja: treu el prefix ▁ de SentencePiece
            tok_clean = tok.replace("▁", "")
            # Treu signes de puntuació que no són part de les paraules? No, les paraules inclouen '.'
            # Avança char_pos fins que el token encaixi
            # Busquem la primera aparició de tok_clean a partir de char_pos
            if not tok_clean:
                token_to_word_text.append(-1)
                continue

            idx = concat.find(tok_clean, char_pos)
            if idx == -1:
                # Potser el token conté caràcters no presents (ex: "'" del prompt barrejat)
                # Intentem buscar caràcter a caràcter
                token_to_word_text.append(-1)
                print(
                    f"  WARN: token {tok!r} (clean={tok_clean!r}) no trobat des de char_pos={char_pos}"
                )
                continue

            # Assignem a la paraula que conté aquest caràcter
            mid_char = idx  # primer caràcter del token
            wid = -1
            for j, (s, e) in enumerate(word_char_ranges):
                if s <= mid_char < e:
                    wid = j
                    break
            token_to_word_text.append(wid)
            char_pos = idx + len(tok_clean)
            if debug:
                print(
                    f"  token={tok!r} clean={tok_clean!r} idx={idx} → word {wid} ({words[wid] if wid >= 0 else '?'})"
                )

        # Construeix token_to_word complet
        token_to_word = []
        for i in range(len(tokens)):
            if text_start <= i < text_end:
                token_to_word.append(token_to_word_text[i - text_start])
            else:
                token_to_word.append(-1)

        if debug:
            print(f"token_to_word: {token_to_word}")
            print(f"words: {words}")
            print("=== END DEBUG ===\n")

        return TokenWordAlignment(
            tokens=tokens,
            words=words,
            token_to_word=token_to_word,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def explain(self, text: str, prompt_idx: int = 0, debug=False) -> Explanation:
        """Calcula IG per a un text i un prompt concret."""
        inputs, mask_pos = self.wrapper.build_inputs(text, prompt_idx=prompt_idx)
        inputs = {k: v.to(self.wrapper.device) for k, v in inputs.items()}

        with torch.no_grad():
            embeds = self.wrapper.embed_inputs(inputs["input_ids"])  # [1, S, H]
        embeds = embeds.detach()

        baseline = self._build_baseline(embeds, inputs["input_ids"])

        ig, score_real, score_base = self._compute_ig(
            embeds, baseline, inputs["attention_mask"], mask_pos
        )

        # Saliència per token: L2 sobre la dimensió hidden
        token_saliency = ig[0].norm(dim=-1)  # [S]

        # Alineament
        alignment = self._build_alignment(text, inputs, prompt_idx, debug=debug)

        # Agregació paraules
        token_sal_arr = np.nan_to_num(token_saliency.detach().cpu().numpy(), nan=0.0)
        token_sal_list = token_sal_arr.tolist()

        # Màscara de tokens especials
        special_ids = set(self.wrapper.tokenizer.all_special_ids)
        input_ids_list = inputs["input_ids"][0].tolist()

        # Posa a zero els tokens especials i el <mask>
        for i, tid in enumerate(input_ids_list):
            if tid in special_ids:
                token_sal_list[i] = 0.0

        word_saliency = alignment.aggregate(token_sal_list, reduce=self.aggregation)

        if self.normalize:
            valid = [s for s in word_saliency if not math.isnan(s)]
            max_sal = max(valid) if valid else 1.0
            if max_sal > 0:
                word_saliency = [0.0 if math.isnan(s) else s / max_sal for s in word_saliency]

        gap = abs(ig[0].sum().item() - (score_real - score_base))

        return Explanation(
            text=text,
            words=alignment.words,
            word_saliency=word_saliency,
            tokens=alignment.tokens,
            token_saliency=token_sal_list,
            target_score=score_real,
            method=self.method_name,
            model_name=self.wrapper.name,
            extra={
                "prompt_idx": prompt_idx,
                "completeness_gap": gap,
                "score_baseline": score_base,
                "baseline_type": self.baseline_type,
                "n_steps": self.n_steps,
            },
        )
