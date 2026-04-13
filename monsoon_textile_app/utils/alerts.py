"""
Societal Impact Modules for Monsoon-Textile Volatility System.

Translates ensemble risk scores into actionable advisories for three
stakeholder groups: smallholder cotton farmers, MSME textile
manufacturers, and state/central policy-makers.

Author: Monsoon-Textile Volatility Research Team
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from loguru import logger


# ======================================================================
# Default configurations
# ======================================================================

_DEFAULT_FARMER_CONFIG: Dict[str, Any] = {
    "risk_threshold": 0.5,
    "premium_range_pct": {"LOW": (2.0, 4.0), "MODERATE": (4.0, 7.0), "HIGH": (7.0, 12.0), "EXTREME": (12.0, 18.0)},
    "yield_drop_estimates": {"LOW": (0, 5), "MODERATE": (5, 15), "HIGH": (15, 30), "EXTREME": (30, 50)},
    "insurance_deadline_buffer_days": 14,
    "supported_districts": [
        "Vidarbha", "Marathwada", "Saurashtra", "Telangana",
        "Rajkot", "Surendranagar", "Guntur", "Adilabad",
        "Akola", "Amravati", "Yavatmal", "Wardha",
    ],
}

_DEFAULT_MSME_CONFIG: Dict[str, Any] = {
    "risk_threshold": 0.5,
    "hedge_horizon_weeks": {"MODERATE": 4, "HIGH": 8, "EXTREME": 12},
    "expected_price_impact_pct": {"LOW": (-2, 3), "MODERATE": (3, 10), "HIGH": (10, 25), "EXTREME": (25, 50)},
    "sector_employment_lakhs": 45.0,
    "annual_cotton_consumption_cr": 85000.0,
}

_DEFAULT_POLICY_CONFIG: Dict[str, Any] = {
    "risk_threshold": 0.4,
    "employment_base_lakhs": 45.0,
    "employment_impact_pct": {"LOW": 0.0, "MODERATE": 2.0, "HIGH": 8.0, "EXTREME": 15.0},
    "states": [
        "Maharashtra", "Gujarat", "Telangana", "Andhra Pradesh",
        "Madhya Pradesh", "Rajasthan", "Karnataka", "Haryana",
    ],
}


def _classify_risk(score: float) -> str:
    """Classify a risk score into a categorical level."""
    if score < 0.3:
        return "LOW"
    elif score < 0.6:
        return "MODERATE"
    elif score < 0.8:
        return "HIGH"
    return "EXTREME"


# ======================================================================
# Farmer Advisory System
# ======================================================================


class FarmerAdvisorySystem:
    """Generate crop-insurance and agronomic advisories for smallholder
    cotton farmers based on ensemble risk scores and monsoon deficit data.

    Parameters
    ----------
    config : dict, optional
        Configuration overriding default thresholds, premium ranges,
        yield-drop estimates, and supported districts.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        self.config: Dict[str, Any] = {**_DEFAULT_FARMER_CONFIG, **(config or {})}
        logger.info(
            "FarmerAdvisorySystem initialised | threshold={t} districts={n}",
            t=self.config["risk_threshold"],
            n=len(self.config["supported_districts"]),
        )

    # ------------------------------------------------------------------
    # Advisory generation
    # ------------------------------------------------------------------

    def generate_advisory(
        self,
        risk_score: float,
        district: str,
        deficit_pct: float,
        date: str,
    ) -> Dict[str, Any]:
        """Generate an advisory for a single district on a given date.

        Parameters
        ----------
        risk_score : float
            Composite ensemble risk score (0-1).
        district : str
            District name.
        deficit_pct : float
            Current cumulative monsoon rainfall deficit (negative = deficit).
        date : str
            Advisory date (ISO format).

        Returns
        -------
        dict
            Advisory containing message, severity, recommended actions,
            estimated yield drop, and insurance deadline.
        """
        level = _classify_risk(risk_score)
        yield_range = self.config["yield_drop_estimates"].get(level, (0, 0))
        premium_range = self.config["premium_range_pct"].get(level, (0, 0))
        buffer_days = self.config["insurance_deadline_buffer_days"]

        advisory_date = pd.Timestamp(date)
        insurance_deadline = advisory_date + pd.Timedelta(days=buffer_days)

        messages = {
            "LOW": f"{district}: Monsoon conditions normal. Continue regular crop management.",
            "MODERATE": (
                f"{district}: Moderate drought risk detected (deficit {deficit_pct:.1f}%). "
                f"Consider enrolling in PMFBY crop insurance within {buffer_days} days."
            ),
            "HIGH": (
                f"{district}: HIGH drought risk (deficit {deficit_pct:.1f}%). "
                f"Strongly recommended: enrol in crop insurance immediately. "
                f"Estimated yield drop {yield_range[0]}-{yield_range[1]}%."
            ),
            "EXTREME": (
                f"{district}: EXTREME drought risk (deficit {deficit_pct:.1f}%). "
                f"URGENT: Enrol in crop insurance NOW. Consider drought-resistant seed varieties. "
                f"Expected yield loss {yield_range[0]}-{yield_range[1]}%."
            ),
        }

        recommended_actions: Dict[str, List[str]] = {
            "LOW": ["Monitor IMD forecasts weekly"],
            "MODERATE": [
                "Enrol in PMFBY crop insurance",
                "Review water management practices",
                "Contact local KVK for advisory",
            ],
            "HIGH": [
                "Enrol in PMFBY crop insurance immediately",
                "Activate micro-irrigation if available",
                "Reduce fertiliser application",
                "Consider intercropping with drought-tolerant legumes",
            ],
            "EXTREME": [
                "Enrol in PMFBY crop insurance NOW",
                "Switch to drought-resistant Bt cotton varieties",
                "Implement deficit irrigation schedule",
                "Contact district agriculture officer",
                "Consider partial crop diversification",
            ],
        }

        advisory: Dict[str, Any] = {
            "district": district,
            "date": date,
            "risk_score": round(risk_score, 4),
            "severity": level,
            "deficit_pct": round(deficit_pct, 2),
            "message": messages[level],
            "recommended_actions": recommended_actions[level],
            "estimated_yield_drop_pct": {"min": yield_range[0], "max": yield_range[1]},
            "insurance_premium_range_pct": {"min": premium_range[0], "max": premium_range[1]},
            "insurance_deadline": str(insurance_deadline.date()),
        }

        logger.debug(
            "Advisory | {district} | {level} | risk={rs:.3f}",
            district=district,
            level=level,
            rs=risk_score,
        )
        return advisory

    def batch_advisories(
        self,
        risk_scores_by_district: Dict[str, Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Generate advisories for all districts above the risk threshold.

        Parameters
        ----------
        risk_scores_by_district : dict
            Mapping of district name to dict with keys ``risk_score``,
            ``deficit_pct``, and ``date``.

        Returns
        -------
        list[dict]
            List of advisory dicts for districts exceeding the threshold.
        """
        advisories: List[Dict[str, Any]] = []
        threshold = self.config["risk_threshold"]

        for district, data in risk_scores_by_district.items():
            score = data.get("risk_score", 0.0)
            if score >= threshold:
                advisory = self.generate_advisory(
                    risk_score=score,
                    district=district,
                    deficit_pct=data.get("deficit_pct", 0.0),
                    date=data.get("date", str(pd.Timestamp.now().date())),
                )
                advisories.append(advisory)

        logger.info(
            "Batch advisories | {issued}/{total} districts above threshold {t}",
            issued=len(advisories),
            total=len(risk_scores_by_district),
            t=threshold,
        )
        return advisories

    # ------------------------------------------------------------------
    # Impact estimation
    # ------------------------------------------------------------------

    def estimate_savings(
        self,
        advisories_issued: int,
        historical_claims: pd.DataFrame,
    ) -> Dict[str, Any]:
        """Estimate savings from early insurance adoption driven by advisories.

        Parameters
        ----------
        advisories_issued : int
            Total number of advisories issued across all districts.
        historical_claims : pd.DataFrame
            Historical crop-insurance claims data with columns
            ``['year', 'district', 'claims_rs_cr', 'insured_farmers',
            'total_farmers']``.

        Returns
        -------
        dict
            Estimated savings in Rs (crores) from higher insurance adoption.
        """
        if historical_claims.empty:
            return {"estimated_savings_rs_cr": 0.0, "additional_farmers_covered": 0}

        avg_claim_per_farmer = (
            historical_claims["claims_rs_cr"].sum()
            / max(historical_claims["insured_farmers"].sum(), 1)
        )
        avg_coverage_rate = (
            historical_claims["insured_farmers"].sum()
            / max(historical_claims["total_farmers"].sum(), 1)
        )

        # Assume advisories increase coverage by 15-25 percentage points
        adoption_uplift = 0.20
        additional_farmers = int(
            advisories_issued * (1 - avg_coverage_rate) * adoption_uplift * 1000
        )
        estimated_savings = additional_farmers * float(avg_claim_per_farmer)

        result: Dict[str, Any] = {
            "estimated_savings_rs_cr": round(estimated_savings, 2),
            "additional_farmers_covered": additional_farmers,
            "current_coverage_rate_pct": round(avg_coverage_rate * 100, 1),
            "projected_coverage_rate_pct": round((avg_coverage_rate + adoption_uplift) * 100, 1),
            "advisories_issued": advisories_issued,
        }
        logger.info(
            "Estimated savings: Rs {s:.2f} Cr | +{f} farmers covered",
            s=result["estimated_savings_rs_cr"],
            f=additional_farmers,
        )
        return result

    def district_level_report(
        self,
        risk_data: pd.DataFrame,
    ) -> pd.DataFrame:
        """Generate a district-level risk and advisory report.

        Parameters
        ----------
        risk_data : pd.DataFrame
            DataFrame with columns ``['district', 'risk_score', 'deficit_pct',
            'date']``.

        Returns
        -------
        pd.DataFrame
            Enriched DataFrame with severity, estimated yield drop,
            advisory status, and recommended actions.
        """
        if risk_data.empty:
            return pd.DataFrame()

        df = risk_data.copy()
        df["severity"] = df["risk_score"].apply(_classify_risk)
        df["advisory_issued"] = df["risk_score"] >= self.config["risk_threshold"]

        df["yield_drop_min_pct"] = df["severity"].map(
            lambda s: self.config["yield_drop_estimates"].get(s, (0, 0))[0]
        )
        df["yield_drop_max_pct"] = df["severity"].map(
            lambda s: self.config["yield_drop_estimates"].get(s, (0, 0))[1]
        )
        df["insurance_premium_min_pct"] = df["severity"].map(
            lambda s: self.config["premium_range_pct"].get(s, (0, 0))[0]
        )
        df["insurance_premium_max_pct"] = df["severity"].map(
            lambda s: self.config["premium_range_pct"].get(s, (0, 0))[1]
        )

        logger.info(
            "District report | {n} districts | {a} advisories issued",
            n=len(df),
            a=int(df["advisory_issued"].sum()),
        )
        return df


# ======================================================================
# MSME Hedging Advisor
# ======================================================================


class MSMEHedgingAdvisor:
    """Hedging and procurement advisories for MSME textile manufacturers.

    Translates risk scores and cotton-price trends into actionable alerts
    for small and medium enterprises in the textile value chain.

    Parameters
    ----------
    config : dict, optional
        Configuration overriding default thresholds and sector parameters.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        self.config: Dict[str, Any] = {**_DEFAULT_MSME_CONFIG, **(config or {})}
        logger.info(
            "MSMEHedgingAdvisor initialised | threshold={t}",
            t=self.config["risk_threshold"],
        )

    def generate_alert(
        self,
        risk_score: float,
        cotton_trend: str,
        current_price: float,
    ) -> Dict[str, Any]:
        """Generate a hedging alert for MSME textile manufacturers.

        Parameters
        ----------
        risk_score : float
            Composite ensemble risk score (0-1).
        cotton_trend : str
            Recent price trend: ``'rising'``, ``'stable'``, or ``'falling'``.
        current_price : float
            Current MCX cotton price (Rs per candy).

        Returns
        -------
        dict
            Alert with message, severity, recommended hedge, expected
            price range, and potential savings percentage.
        """
        level = _classify_risk(risk_score)
        price_impact = self.config["expected_price_impact_pct"].get(level, (0, 0))
        hedge_weeks = self.config["hedge_horizon_weeks"].get(level, 0)

        price_lo = current_price * (1 + price_impact[0] / 100.0)
        price_hi = current_price * (1 + price_impact[1] / 100.0)

        # Potential savings from hedging = midpoint of expected price increase
        midpoint_increase = (price_impact[0] + price_impact[1]) / 2.0
        potential_savings_pct = max(midpoint_increase, 0.0)

        hedge_recommendations: Dict[str, str] = {
            "LOW": "No immediate hedging required. Monitor weekly.",
            "MODERATE": (
                f"Consider forward contracts for {hedge_weeks}-week horizon. "
                f"Lock in 30-50% of quarterly cotton requirements."
            ),
            "HIGH": (
                f"Strongly recommend hedging via MCX cotton futures for {hedge_weeks} weeks. "
                f"Lock in 50-75% of quarterly requirements at current prices."
            ),
            "EXTREME": (
                f"URGENT: Maximise forward cover for {hedge_weeks}+ weeks. "
                f"Lock in 75-100% of quarterly requirements immediately. "
                f"Consider inventory build at current levels."
            ),
        }

        messages: Dict[str, str] = {
            "LOW": "Cotton market stable. No action needed.",
            "MODERATE": (
                f"Moderate volatility risk detected. Cotton trend: {cotton_trend}. "
                f"Expected price range Rs {price_lo:,.0f}-{price_hi:,.0f}/candy."
            ),
            "HIGH": (
                f"HIGH volatility risk. Cotton trend: {cotton_trend}. "
                f"Expected price range Rs {price_lo:,.0f}-{price_hi:,.0f}/candy. "
                f"Hedge immediately."
            ),
            "EXTREME": (
                f"EXTREME volatility risk. Cotton trend: {cotton_trend}. "
                f"Expected price range Rs {price_lo:,.0f}-{price_hi:,.0f}/candy. "
                f"Critical: lock in supply now."
            ),
        }

        alert: Dict[str, Any] = {
            "risk_score": round(risk_score, 4),
            "severity": level,
            "cotton_trend": cotton_trend,
            "current_price_rs": round(current_price, 2),
            "message": messages[level],
            "recommended_hedge": hedge_recommendations[level],
            "expected_price_range_rs": {"low": round(price_lo, 2), "high": round(price_hi, 2)},
            "hedge_horizon_weeks": hedge_weeks,
            "potential_savings_pct": round(potential_savings_pct, 2),
        }

        logger.debug(
            "MSME alert | {level} | price={p:,.0f} | savings={s:.1f}%",
            level=level,
            p=current_price,
            s=potential_savings_pct,
        )
        return alert

    def compute_hedging_savings(
        self,
        spot_prices: pd.Series,
        forward_prices: pd.Series,
        alert_dates: List[str],
    ) -> Dict[str, Any]:
        """Compute actual savings from hedging at alert dates vs. spot purchase.

        Parameters
        ----------
        spot_prices : pd.Series
            Daily spot cotton prices (DatetimeIndex, Rs/candy).
        forward_prices : pd.Series
            Forward/futures prices at the time of alert (DatetimeIndex).
        alert_dates : list[str]
            ISO-format dates when hedging alerts were issued.

        Returns
        -------
        dict
            Total savings, per-alert savings, and percentage saved.
        """
        savings_records: List[Dict[str, Any]] = []
        total_saved_rs = 0.0
        total_spot_cost = 0.0

        for date_str in alert_dates:
            dt = pd.Timestamp(date_str)
            # Price at alert (forward lock-in price)
            if dt not in forward_prices.index:
                nearest_idx = forward_prices.index.get_indexer([dt], method="nearest")[0]
                if nearest_idx < 0:
                    continue
                forward_dt = forward_prices.index[nearest_idx]
            else:
                forward_dt = dt

            forward_price = float(forward_prices.loc[forward_dt])

            # Spot price 8 weeks later (delivery date)
            delivery_dt = dt + pd.Timedelta(weeks=8)
            if delivery_dt > spot_prices.index.max():
                continue
            nearest_delivery_idx = spot_prices.index.get_indexer([delivery_dt], method="nearest")[0]
            if nearest_delivery_idx < 0:
                continue
            delivery_spot = float(spot_prices.iloc[nearest_delivery_idx])

            saving = delivery_spot - forward_price
            total_saved_rs += saving
            total_spot_cost += delivery_spot

            savings_records.append({
                "alert_date": date_str,
                "forward_price_rs": round(forward_price, 2),
                "delivery_spot_rs": round(delivery_spot, 2),
                "saving_rs_per_candy": round(saving, 2),
            })

        savings_pct = (total_saved_rs / total_spot_cost * 100) if total_spot_cost > 0 else 0.0

        result: Dict[str, Any] = {
            "total_savings_rs": round(total_saved_rs, 2),
            "total_spot_cost_rs": round(total_spot_cost, 2),
            "savings_pct": round(savings_pct, 2),
            "n_alerts_evaluated": len(savings_records),
            "per_alert_details": savings_records,
        }

        logger.info(
            "Hedging savings | Rs {s:,.0f} saved ({pct:.1f}%) across {n} alerts",
            s=total_saved_rs,
            pct=savings_pct,
            n=len(savings_records),
        )
        return result

    def sector_impact_estimate(
        self,
        risk_scores: pd.Series,
    ) -> Dict[str, Any]:
        """Estimate aggregate MSME textile sector impact from monsoon risk.

        Parameters
        ----------
        risk_scores : pd.Series
            Risk score series (DatetimeIndex).

        Returns
        -------
        dict
            Aggregate impact in Rs crores, affected employment, and
            risk distribution.
        """
        if risk_scores.empty:
            return {"error": "Empty risk scores series"}

        levels = risk_scores.apply(_classify_risk)
        level_distribution = levels.value_counts(normalize=True).to_dict()

        # Weighted average price impact
        avg_score = float(risk_scores.mean())
        max_score = float(risk_scores.max())
        avg_level = _classify_risk(avg_score)

        price_impact = self.config["expected_price_impact_pct"].get(avg_level, (0, 0))
        midpoint_impact_pct = (price_impact[0] + price_impact[1]) / 2.0
        annual_consumption = self.config["annual_cotton_consumption_cr"]
        estimated_cost_increase_cr = annual_consumption * midpoint_impact_pct / 100.0

        result: Dict[str, Any] = {
            "avg_risk_score": round(avg_score, 4),
            "max_risk_score": round(max_score, 4),
            "avg_risk_level": avg_level,
            "risk_distribution": {k: round(v, 4) for k, v in level_distribution.items()},
            "estimated_raw_material_cost_increase_rs_cr": round(estimated_cost_increase_cr, 2),
            "annual_cotton_consumption_rs_cr": annual_consumption,
            "sector_employment_lakhs": self.config["sector_employment_lakhs"],
        }

        logger.info(
            "Sector impact | avg_risk={ar} | cost_increase=Rs {ci:,.0f} Cr",
            ar=avg_level,
            ci=estimated_cost_increase_cr,
        )
        return result


# ======================================================================
# Policy Dashboard Generator
# ======================================================================


class PolicyDashboardGenerator:
    """Generate state-level risk dashboards and policy recommendations
    for government decision-makers.

    Parameters
    ----------
    config : dict, optional
        Configuration overriding default thresholds and state parameters.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        self.config: Dict[str, Any] = {**_DEFAULT_POLICY_CONFIG, **(config or {})}
        self._farmer_system = FarmerAdvisorySystem()
        self._msme_advisor = MSMEHedgingAdvisor()
        logger.info(
            "PolicyDashboardGenerator initialised | states={n}",
            n=len(self.config["states"]),
        )

    def weekly_risk_report(
        self,
        state_risk_data: Dict[str, Dict[str, Any]],
        date: str,
    ) -> Dict[str, Any]:
        """Generate a weekly risk report with state-level heatmap data.

        Parameters
        ----------
        state_risk_data : dict
            Mapping of state name to dict containing ``risk_score``,
            ``deficit_pct``, ``districts`` (list of district-level dicts).
        date : str
            Report date (ISO format).

        Returns
        -------
        dict
            Report with state-level heatmap data, district drill-downs,
            national summary, and historical comparison fields.
        """
        heatmap_data: List[Dict[str, Any]] = []
        district_drilldown: Dict[str, List[Dict[str, Any]]] = {}
        all_scores: List[float] = []

        for state, data in state_risk_data.items():
            score = data.get("risk_score", 0.0)
            all_scores.append(score)
            level = _classify_risk(score)

            heatmap_data.append({
                "state": state,
                "risk_score": round(score, 4),
                "risk_level": level,
                "deficit_pct": data.get("deficit_pct"),
            })

            # District drill-down
            districts = data.get("districts", [])
            if districts:
                district_drilldown[state] = [
                    {
                        "district": d.get("name", "Unknown"),
                        "risk_score": round(d.get("risk_score", 0.0), 4),
                        "risk_level": _classify_risk(d.get("risk_score", 0.0)),
                        "deficit_pct": d.get("deficit_pct"),
                    }
                    for d in districts
                ]

        national_avg = float(np.mean(all_scores)) if all_scores else 0.0
        states_at_risk = sum(1 for s in all_scores if s >= self.config["risk_threshold"])

        report: Dict[str, Any] = {
            "date": date,
            "national_avg_risk": round(national_avg, 4),
            "national_risk_level": _classify_risk(national_avg),
            "states_at_risk": states_at_risk,
            "total_states_monitored": len(state_risk_data),
            "heatmap_data": sorted(heatmap_data, key=lambda x: x["risk_score"], reverse=True),
            "district_drilldown": district_drilldown,
            "historical_comparison": {
                "note": "Populate with historical same-week risk scores for year-over-year comparison.",
            },
        }

        logger.info(
            "Weekly report | date={d} | national_avg={na:.3f} | states_at_risk={sr}/{total}",
            d=date,
            na=national_avg,
            sr=states_at_risk,
            total=len(state_risk_data),
        )
        return report

    def employment_impact_estimate(
        self,
        risk_level: str,
    ) -> Dict[str, Any]:
        """Estimate the number of textile-sector jobs at risk.

        Parameters
        ----------
        risk_level : str
            Risk classification: ``'LOW'``, ``'MODERATE'``, ``'HIGH'``,
            or ``'EXTREME'``.

        Returns
        -------
        dict
            Employment impact estimate with jobs at risk and percentage.
        """
        impact_pct = self.config["employment_impact_pct"].get(risk_level, 0.0)
        base_employment = self.config["employment_base_lakhs"]
        jobs_at_risk_lakhs = base_employment * impact_pct / 100.0

        result: Dict[str, Any] = {
            "risk_level": risk_level,
            "employment_base_lakhs": base_employment,
            "impact_pct": impact_pct,
            "jobs_at_risk_lakhs": round(jobs_at_risk_lakhs, 2),
            "jobs_at_risk_approx": int(jobs_at_risk_lakhs * 100_000),
        }

        logger.info(
            "Employment impact | {level} | {j:.2f} lakh jobs at risk ({p}%)",
            level=risk_level,
            j=jobs_at_risk_lakhs,
            p=impact_pct,
        )
        return result

    def automated_recommendations(
        self,
        risk_level: str,
        state: str,
    ) -> Dict[str, Any]:
        """Generate policy action recommendations based on risk level.

        Parameters
        ----------
        risk_level : str
            Risk classification.
        state : str
            State name for context-specific recommendations.

        Returns
        -------
        dict
            Policy recommendations categorised by urgency.
        """
        recommendations: Dict[str, Dict[str, List[str]]] = {
            "LOW": {
                "immediate": [],
                "short_term": [
                    "Continue routine monsoon monitoring",
                    f"Review {state} cotton procurement readiness",
                ],
                "medium_term": [
                    "Update state disaster management plans",
                    "Conduct farmer awareness campaigns on crop insurance",
                ],
            },
            "MODERATE": {
                "immediate": [
                    f"Activate {state} drought monitoring cell",
                    "Issue PMFBY enrolment reminder via SMS/IVR",
                ],
                "short_term": [
                    "Pre-position drought relief materials",
                    "Brief district collectors on situation",
                    "Review MSP procurement logistics",
                ],
                "medium_term": [
                    "Prepare SDRF activation paperwork",
                    "Coordinate with textile industry associations",
                ],
            },
            "HIGH": {
                "immediate": [
                    f"Convene {state} crisis management meeting",
                    "Activate SDRF (State Disaster Response Fund)",
                    "Issue public advisory via all channels",
                    "Extend PMFBY enrolment deadline by 2 weeks",
                ],
                "short_term": [
                    "Deploy mobile water tankers to deficit districts",
                    "Activate MGNREGA works in affected areas",
                    "Coordinate with CCI for cotton buffer stock release",
                    "Brief textile MSMEs on expected supply disruption",
                ],
                "medium_term": [
                    "Prepare NDRF assistance request",
                    "Plan input subsidy disbursement",
                ],
            },
            "EXTREME": {
                "immediate": [
                    f"Declare drought in affected {state} districts",
                    "Activate NDRF (National Disaster Response Fund) request",
                    "Issue emergency PMFBY deadline extension",
                    "Deploy emergency water supply teams",
                    "Convene inter-ministerial coordination meeting",
                ],
                "short_term": [
                    "Release cotton from CCI buffer stocks",
                    "Activate free seed distribution programme",
                    "Scale up MGNREGA allocation by 50%",
                    "Coordinate with RBI on MSME credit relaxation",
                    "Issue textile sector employment protection advisory",
                ],
                "medium_term": [
                    "Prepare PM-KISAN special disbursement",
                    "Plan medium-term water infrastructure projects",
                    "Review and update National Fibre Policy contingencies",
                ],
            },
        }

        level_recs = recommendations.get(risk_level, recommendations["LOW"])

        result: Dict[str, Any] = {
            "state": state,
            "risk_level": risk_level,
            "recommendations": level_recs,
            "total_actions": sum(len(v) for v in level_recs.values()),
        }

        logger.info(
            "Policy recommendations | {state} | {level} | {n} actions",
            state=state,
            level=risk_level,
            n=result["total_actions"],
        )
        return result

    def impact_metrics_summary(self) -> Dict[str, Any]:
        """Aggregate impact metrics across all stakeholder groups.

        Returns
        -------
        dict
            Summary of farmer savings potential, MSME savings potential,
            early-warning lead time, and advisory coverage statistics.
        """
        summary: Dict[str, Any] = {
            "system_name": "Monsoon-Textile Volatility Early Warning System",
            "farmer_impact": {
                "target_coverage_districts": len(
                    self._farmer_system.config["supported_districts"]
                ),
                "insurance_adoption_uplift_target_pct": 20.0,
                "estimated_savings_potential_rs_cr": "Computed per-season via estimate_savings()",
            },
            "msme_impact": {
                "sector_employment_lakhs": self._msme_advisor.config["sector_employment_lakhs"],
                "annual_cotton_consumption_rs_cr": self._msme_advisor.config[
                    "annual_cotton_consumption_cr"
                ],
                "hedging_savings_potential_pct": "5-15% depending on severity",
            },
            "early_warning": {
                "target_lead_time_weeks": "4-8 weeks before price spike",
                "signal_threshold": 0.6,
                "historical_accuracy": "Computed via DroughtYearBacktester",
            },
            "coverage": {
                "states_monitored": len(self.config["states"]),
                "states": self.config["states"],
                "update_frequency": "weekly",
            },
        }

        logger.info(
            "Impact metrics summary | {n} states | {d} districts",
            n=len(self.config["states"]),
            d=len(self._farmer_system.config["supported_districts"]),
        )
        return summary
