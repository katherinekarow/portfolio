"""Cleaning rules that turn vendor free text into the internal data model."""
from __future__ import annotations

import re

import pandas as pd

from .synth import FX_TO_USD, TIERS, VENDOR_LABELS

# Build the label -> tier map from known spellings, matched case- and punctuation-insensitively.
_LABEL_MAP = {re.sub(r"[^a-z0-9]", "", lbl.lower()): tier for tier, lbls in VENDOR_LABELS.items() for lbl in lbls}
_LABEL_MAP.update({t.lower(): t for t in TIERS})


def tier_from_label(label: str) -> str | None:
    if label is None or (isinstance(label, float) and pd.isna(label)):
        return None
    key = re.sub(r"[^a-z0-9]", "", str(label).lower())
    if key in _LABEL_MAP:
        return _LABEL_MAP[key]
    for k, tier in _LABEL_MAP.items():          # fall back to prefix match ("premiumgradefancy" -> PREMIUM)
        if len(k) >= 3 and key.startswith(k):
            return tier
    return None


def parse_amount(text) -> float | None:
    """'€1,234.5' -> 1234.5 ; ' 99.00 ' -> 99.0 ; garbage -> None"""
    if text is None or (isinstance(text, float) and pd.isna(text)):
        return None
    cleaned = re.sub(r"[^0-9.\-]", "", str(text).replace(",", ""))
    try:
        return float(cleaned)
    except ValueError:
        return None


def normalize_vendor(raw: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (clean offers, rejected rows with reason codes).

    * trims and upper-cases status
    * maps grade labels to tiers (unmapped labels are rejected, never guessed)
    * converts every amount to USD
    * keeps only the latest version of re-sent offers
    """
    df = raw.copy()
    df.columns = [c.strip() for c in df.columns]
    out = pd.DataFrame({
        "offer_id": df["Offer ID"].str.strip(),
        "order_ref": df["Order Ref"].str.strip(),
        "lot_id": df["Lot"].str.strip().str.upper(),
        "from_tier": df["From Grade"].map(tier_from_label),
        "to_tier": df["To Grade"].map(tier_from_label),
        "amount_local": df["Bid"].map(parse_amount),
        "currency": df["Currency"].str.strip().str.upper(),
        "status": df["Status"].str.strip().str.upper(),
        "updated_at": pd.to_datetime(df["Updated At"]),
    })
    out["fx_rate"] = out.currency.map(FX_TO_USD)
    out["amount_usd"] = (out.amount_local * out.fx_rate).round(2)

    reasons = pd.Series("", index=out.index)
    reasons[out.from_tier.isna() | out.to_tier.isna()] = "UNMAPPED_GRADE"
    reasons[out.amount_local.isna()] = "UNPARSEABLE_AMOUNT"
    reasons[out.fx_rate.isna()] = "UNKNOWN_CURRENCY"
    bad = reasons != ""
    rejects = raw[bad].assign(reason_code=reasons[bad])

    clean = out[~bad].sort_values("updated_at")
    n_before = len(clean)
    clean = clean.drop_duplicates("offer_id", keep="last")
    clean.attrs["duplicates_removed"] = n_before - len(clean)
    return clean.reset_index(drop=True), rejects.reset_index(drop=True)
