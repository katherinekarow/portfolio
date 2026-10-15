"""Process capability indices and expected out-of-spec ppm under a normal model."""
from __future__ import annotations

from .descriptive import normal_cdf


def indices(mu: float, sigma_within: float, sigma_overall: float, lsl: float, usl: float) -> dict:
    """Cp = (USL − LSL)/6σ_w, Cpk = min(USL − μ, μ − LSL)/3σ_w; Pp, Ppk use the overall (long-term) σ."""
    def pair(s):
        return (usl - lsl) / (6 * s), min(usl - mu, mu - lsl) / (3 * s)
    cp, cpk = pair(sigma_within)
    pp, ppk = pair(sigma_overall)
    return {"Cp": cp, "Cpk": cpk, "Pp": pp, "Ppk": ppk,
            "ppm_within": ppm(mu, sigma_within, lsl, usl), "ppm_overall": ppm(mu, sigma_overall, lsl, usl)}


def ppm(mu: float, sigma: float, lsl: float, usl: float) -> float:
    """Expected parts per million outside spec: [Φ((LSL − μ)/σ) + 1 − Φ((USL − μ)/σ)] × 10⁶."""
    return float((normal_cdf(lsl, mu, sigma) + 1 - normal_cdf(usl, mu, sigma)) * 1e6)
