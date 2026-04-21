"""
models/utils.py

Funcions auxiliars compartides pels wrappers.
"""


def _whitespace_tokenize(text: str) -> list[str]:
    """
    Tokenització per espais + neteja de puntuació adherida.

    Consistent amb la manera com l'eye-tracker defineix AOIs per paraula:
    la paraula visual és la seqüència de caràcters entre espais.

    Exemple:
        "Hola, món!" → ["Hola,", "món!"]

    Nota: NO separarem la puntuació de la paraula, perquè l'eye-tracker
    tampoc no ho fa (la puntuació forma part del AOI visual).
    """
    return text.split()


def _assign_word_indices_simple(
    tok,
    all_tokens: list[str],
    text_start: int,
    text_end: int,
    words: list[str],
    token_to_word: list[int],
) -> None:
    """
    Assigna índexs de paraula als tokens del segment [text_start, text_end).

    Mateix algorisme greedy que MLMClozeWrapper._assign_word_indices,
    extret aquí per reutilitzar-lo.
    """
    word_token_counts = []
    for i, word in enumerate(words):
        prefix = " " if i > 0 else ""
        n = len(tok.tokenize(prefix + word))
        word_token_counts.append(n)

    cursor = text_start
    for w_idx, count in enumerate(word_token_counts):
        for _ in range(count):
            if cursor < text_end:
                token_to_word[cursor] = w_idx
                cursor += 1
