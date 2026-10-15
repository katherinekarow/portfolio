"""Reconcile the vendor's view of substitution offers against the internal system of record.

Exception classes (one per offer, most severe wins):

| class                  | meaning                                                         |
|------------------------|-----------------------------------------------------------------|
| CANCELLED_BUT_CHARGED  | vendor says cancelled, internal system still charged the buyer  |
| MISSING_INTERNAL       | vendor accepted, internal file already covers that day, no charge |
| MISSING_VENDOR         | internal charge with no vendor record at all                    |
| PRICE_MISMATCH         | both sides agree it happened but amounts differ beyond tolerance |
| IN_TRANSIT             | vendor accepted after the latest internal file's cut-off (timing) |
| MATCHED                | agrees within tolerance (small FX / rounding noise is allowed)  |
"""
from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

SEVERITY = {"CANCELLED_BUT_CHARGED": 1, "MISSING_INTERNAL": 2, "MISSING_VENDOR": 2,
            "PRICE_MISMATCH": 3, "IN_TRANSIT": 9, "MATCHED": 9}
EXCEPTIONS = [k for k, v in SEVERITY.items() if v < 9]


@dataclass
class Tolerance:
    absolute_usd: float = 0.25        # rounding / FX display noise
    relative: float = 0.005           # 0.5% of the vendor amount

    def limit(self, amount: pd.Series) -> pd.Series:
        return np.maximum(self.absolute_usd, self.relative * amount.abs())


@dataclass
class ReconResult:
    detail: pd.DataFrame
    internal_file_date: dt.date
    warnings: list[str] = field(default_factory=list)
    control: dict = field(default_factory=dict)

    def ids(self, cls: str) -> set[str]:
        return set(self.detail.loc[self.detail.recon_class == cls, "offer_id"])

    @property
    def exceptions(self) -> pd.DataFrame:
        return self.detail[self.detail.recon_class.isin(EXCEPTIONS)].sort_values(
            ["severity", "abs_diff_usd"], ascending=[True, False])


def next_business_day(d: dt.date) -> dt.date:
    while d.weekday() >= 5:
        d += dt.timedelta(days=1)
    return d


def load_internal(daily: dict[dt.date, pd.DataFrame], run_date: dt.date,
                  lookback_days: int = 35) -> tuple[pd.DataFrame, dt.date, list[str]]:
    """Stack the daily internal files inside the look-back window.

    If today's file has not landed yet, fall back to the most recent business day that
    exists and say so - the run continues instead of failing on a Monday morning.
    """
    warnings = []
    available = sorted(d for d in daily if run_date - dt.timedelta(days=lookback_days) <= d <= run_date)
    if not available:
        raise FileNotFoundError(f"no internal files in the {lookback_days} days before {run_date}")
    latest = available[-1]
    if latest != run_date:
        warnings.append(f"internal file for {run_date} not available; using {latest} as the latest cut-off")
    frames = [daily[d].assign(report_date=d) for d in available]
    internal = pd.concat(frames, ignore_index=True).rename(columns=str.lower)
    n = len(internal)
    internal = internal.drop_duplicates("offer_id", keep="last")
    if len(internal) < n:
        warnings.append(f"{n - len(internal)} offer(s) appeared in more than one internal file; kept the latest")
    return internal, latest, warnings


def reconcile(vendor_clean: pd.DataFrame, internal: pd.DataFrame, cutoff: dt.date,
              tol: Tolerance | None = None) -> pd.DataFrame:
    tol = tol or Tolerance()
    v = vendor_clean.rename(columns=lambda c: f"v_{c}" if c != "offer_id" else c)
    i = internal.rename(columns=lambda c: f"i_{c}" if c != "offer_id" else c)
    m = v.merge(i, on="offer_id", how="outer", indicator="side")

    in_v, in_i = m.side.isin(["left_only", "both"]), m.side.isin(["right_only", "both"])
    status = m.v_status.fillna("")
    vendor_day = pd.to_datetime(m.v_updated_at).dt.date
    awaiting = vendor_day.map(lambda d: next_business_day(d) > cutoff if pd.notna(d) else False)

    m["diff_usd"] = (m.i_charged_usd - m.v_amount_usd).round(2)
    m["abs_diff_usd"] = m.diff_usd.abs().fillna(m.v_amount_usd.where(~in_i)).fillna(m.i_charged_usd)
    beyond = m.diff_usd.abs() > tol.limit(m.v_amount_usd.fillna(0))

    conds = [
        in_i & (status == "CANCELLED"),
        in_v & ~in_i & (status == "ACCEPTED") & ~awaiting,
        in_i & ~in_v,
        in_v & in_i & beyond,
        in_v & ~in_i & (status == "ACCEPTED") & awaiting,
        in_v & in_i,
    ]
    classes = ["CANCELLED_BUT_CHARGED", "MISSING_INTERNAL", "MISSING_VENDOR", "PRICE_MISMATCH",
               "IN_TRANSIT", "MATCHED"]
    m["recon_class"] = np.select(conds, classes, default="NO_CHARGE_EXPECTED")
    m["severity"] = m.recon_class.map(SEVERITY).fillna(9).astype(int)
    m["revenue_at_risk_usd"] = np.select(
        [m.recon_class == "MISSING_INTERNAL", m.recon_class == "PRICE_MISMATCH",
         m.recon_class == "CANCELLED_BUT_CHARGED", m.recon_class == "MISSING_VENDOR"],
        [m.v_amount_usd, m.diff_usd.abs(), m.i_charged_usd, m.i_charged_usd], default=0.0)
    m["action"] = m.recon_class.map({
        "CANCELLED_BUT_CHARGED": "Refund buyer; confirm cancellation reached billing",
        "MISSING_INTERNAL": "Raise charge; check the vendor -> billing interface for this day",
        "MISSING_VENDOR": "Ask vendor for the offer record or reverse the charge",
        "PRICE_MISMATCH": "Correct the charge to the accepted bid (check FX / manual override)",
    }).fillna("")
    m["lot_id"] = m.v_lot_id.fillna(m.i_lot_id)
    m["from_tier"] = m.v_from_tier.fillna(m.i_from_tier)
    m["to_tier"] = m.v_to_tier.fillna(m.i_to_tier)
    cols = ["offer_id", "recon_class", "severity", "lot_id", "from_tier", "to_tier", "v_status",
            "v_amount_usd", "i_charged_usd", "diff_usd", "abs_diff_usd", "revenue_at_risk_usd",
            "v_updated_at", "i_report_date", "action"]
    return m[cols].sort_values(["severity", "offer_id"]).reset_index(drop=True)


def control_totals(detail: pd.DataFrame, vendor_clean: pd.DataFrame, internal: pd.DataFrame) -> dict:
    """Every input row must land in exactly one class, and the money must add up."""
    vendor_ids, internal_ids = set(vendor_clean.offer_id), set(internal.offer_id)
    counts = detail.recon_class.value_counts().to_dict()
    both = detail.v_amount_usd.notna() & detail.i_charged_usd.notna()
    ctl = {
        "vendor_offers": len(vendor_ids),
        "internal_charges": len(internal_ids),
        "union_offers": len(vendor_ids | internal_ids),
        "classified_rows": len(detail),
        "counts": counts,
        "internal_total_usd": round(float(internal.charged_usd.sum()), 2),
        "detail_internal_total_usd": round(float(detail.i_charged_usd.sum()), 2),
        "net_diff_on_matched_pairs_usd": round(float(detail.loc[both, "diff_usd"].sum()), 2),
        "revenue_at_risk_usd": round(float(detail.revenue_at_risk_usd.sum()), 2),
    }
    ctl["rows_reconciled"] = ctl["classified_rows"] == ctl["union_offers"] and detail.offer_id.is_unique
    ctl["money_reconciled"] = abs(ctl["internal_total_usd"] - ctl["detail_internal_total_usd"]) < 0.005
    return ctl


def run(vendor_clean: pd.DataFrame, daily: dict, run_date: dt.date, tol: Tolerance | None = None) -> ReconResult:
    internal, cutoff, warnings = load_internal(daily, run_date)
    detail = reconcile(vendor_clean, internal, cutoff, tol)
    return ReconResult(detail=detail, internal_file_date=cutoff, warnings=warnings,
                       control=control_totals(detail, vendor_clean, internal))
