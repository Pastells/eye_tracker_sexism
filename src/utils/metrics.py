import numpy as np
from scipy import stats

EPS = 1e-12  # avoid log(0)


def safe_cross_entropy(p, q):
    """Cross-entropy H(p, q) = -sum(p * log(q)), with epsilon for numerical stability."""
    p = np.asarray(p, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)
    q = np.clip(q, EPS, None)
    return -np.sum(p * np.log(q))


def safe_kl_divergence(p, q):
    """KL(p || q) = sum(p * log(p / q)), with epsilon for numerical stability."""
    p = np.asarray(p, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)
    p = np.clip(p, EPS, None)
    q = np.clip(q, EPS, None)
    return np.sum(p * np.log(p / q))


def jensen_shannon_divergence(p, q, base2=True):
    """JSD(p || q) = 0.5 * KL(p || m) + 0.5 * KL(q || m), where m = 0.5 * (p + q)."""
    p = np.asarray(p, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)
    m = 0.5 * (p + q)
    jsd = 0.5 * safe_kl_divergence(p, m) + 0.5 * safe_kl_divergence(q, m)
    if base2:
        jsd /= np.log(2)
    return jsd


def compute_metrics(eye, ann):
    """Compute Spearman, cross-entropy, and KL-divergence for two aligned vectors."""
    n = min(len(eye), len(ann))
    eye, ann = eye[:n], ann[:n]
    if n == 0:
        return None
    rho, _ = stats.spearmanr(eye, ann)
    return {
        "spearman": rho,
        "cross_entropy": safe_cross_entropy(ann, eye),
        "kl_div": safe_kl_divergence(ann, eye),
        "js_div": jensen_shannon_divergence(ann, eye),
    }
