"""
ai_insights.py  (was gemini_insights.py)
-----------------------------------------
Generates an AI "Executive Readout" using the Groq API.

Model  : llama-3.3-70b-versatile  (free, no credit card needed)
Sign up: https://console.groq.com

The function is intentionally kept as generate_executive_readout()
so the rest of the codebase needs zero changes.
"""

import os
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

_MODEL = "llama-3.3-70b-versatile"   # free on Groq, 128k context


def generate_executive_readout(ab_results: dict) -> str:
    """
    Builds a structured prompt from the ab_results dict produced by
    ABTestEngine.run_full_analysis() and calls the Groq API.

    Returns a formatted Markdown string ready for display in Streamlit.
    """
    api_key = os.getenv("GROQ_API_KEY", "").strip()
    if not api_key:
        return (
            "### ⚠️  Groq API Key Not Configured\n\n"
            "Add `GROQ_API_KEY=your_key` to your `.env` file and restart the app.\n\n"
            "Get a **free** key (no credit card) at **https://console.groq.com**"
        )

    client = Groq(api_key=api_key)

    srm  = ab_results["srm_check"]
    conv = ab_results["conversion_rate"]
    aov  = ab_results["average_order_value"]
    summ = ab_results["experiment_summary"]

    prompt = f"""You are a Senior Data Scientist and A/B Testing expert presenting to the C-suite
and product leadership at a major e-commerce company.
Write a precise, confident, data-driven "Executive Readout" for the experiment below.
Use professional business language. No jargon overload. Quantify everything.

══════════════════════════════════════════════════
EXPERIMENT BRIEF
══════════════════════════════════════════════════
Name       : Checkout Page Redesign
Hypothesis : A redesigned checkout page with a high-contrast CTA button and a
             personalised dynamic-pricing banner will increase conversion rate
             and average order value compared to the existing design.
Duration   : {summ["test_duration_days"]} days
Total users: {srm["n_total"]:,}  (Control: {srm["n_control"]:,}  |  Treatment: {srm["n_treatment"]:,})
Alpha      : {conv["alpha"]} → 95% confidence threshold

══════════════════════════════════════════════════
GUARD-RAIL — SAMPLE RATIO MISMATCH (SRM)
══════════════════════════════════════════════════
Expected split : {srm["expected_split"]*100:.0f}% / {(1-srm["expected_split"])*100:.0f}%
Observed split : {srm["actual_split"]*100:.1f}% / {(1-srm["actual_split"])*100:.1f}%
Chi-Square     : {srm["chi2_statistic"]}   p-value: {srm["p_value"]}
SRM Status     : {"🔴 SRM DETECTED" if srm["srm_detected"] else "🟢 CLEAN — no SRM"}

══════════════════════════════════════════════════
PRIMARY METRIC — CONVERSION RATE (Chi-Square Test)
══════════════════════════════════════════════════
Control   : {conv["ctrl_rate"]*100:.2f}%  ({conv["ctrl_conversions"]:,} / {conv["ctrl_total"]:,} users)
Treatment : {conv["trt_rate"]*100:.2f}%  ({conv["trt_conversions"]:,} / {conv["trt_total"]:,} users)
Absolute uplift  : {conv["absolute_uplift"]*100:+.2f} pp
Relative uplift  : {conv["relative_uplift_pct"]:+.1f}%
95% CI (control)   : [{conv["ctrl_ci_lower"]*100:.2f}%, {conv["ctrl_ci_upper"]*100:.2f}%]
95% CI (treatment) : [{conv["trt_ci_lower"]*100:.2f}%, {conv["trt_ci_upper"]*100:.2f}%]
Chi-Square stat  : {conv["chi2_statistic"]}   p-value: {conv["p_value"]}
Cohen's h        : {conv["cohens_h"]}
Statistical power: {conv["statistical_power"]*100:.1f}%
Result           : {"✅ STATISTICALLY SIGNIFICANT" if conv["is_significant"] else "❌ NOT SIGNIFICANT"}

══════════════════════════════════════════════════
SECONDARY METRIC — AVERAGE ORDER VALUE (Welch's T-Test)
══════════════════════════════════════════════════
Control   : ${aov["ctrl_mean"]:.2f}  (σ=${aov["ctrl_std"]:.2f},  n={aov["ctrl_n"]:,})
Treatment : ${aov["trt_mean"]:.2f}  (σ={aov["trt_std"]:.2f},  n={aov["trt_n"]:,})
Mean difference : ${aov["mean_difference"]:+.2f}
95% CI on diff  : [${aov["ci_lower"]:.2f}, ${aov["ci_upper"]:.2f}]
Relative uplift : {aov["relative_uplift_pct"]:+.1f}%
t-statistic     : {aov["t_statistic"]}   df={aov["degrees_of_freedom"]:.0f}   p-value: {aov["p_value"]}
Cohen's d       : {aov["cohens_d"]}
MDE @ 80% power : ${aov["mde_dollars"]:.2f}
Statistical power: {aov["statistical_power"]*100:.1f}%
Result          : {"✅ STATISTICALLY SIGNIFICANT" if aov["is_significant"] else "❌ NOT SIGNIFICANT"}

══════════════════════════════════════════════════
SYSTEM RECOMMENDATION
══════════════════════════════════════════════════
{summ["overall_recommendation"]}

══════════════════════════════════════════════════

Now write the Executive Readout with EXACTLY these six sections.
Use clean Markdown. Be specific — cite numbers, not vague adjectives.

## 1. 🔍 Experiment Validity
Was the test set up correctly? Address SRM and data quality.

## 2. 📊 Results Snapshot
3–5 bullet points: the most important findings an executive needs in 30 seconds.

## 3. 📐 Statistical Interpretation
Explain what the p-values, confidence intervals, and effect sizes mean
in plain language that a product manager could understand.

## 4. 💰 Business Impact Estimation
If treatment is shipped to 100% of users, what is the estimated incremental
annual revenue impact? Show your arithmetic clearly.
Assume monthly traffic of 50,000 visitors. Current average revenue per session = $12.

## 5. ✅ Decision & Recommendation
Clear GO / NO-GO / ITERATE with explicit business rationale.
What should the product team do on Monday morning?

## 6. 🛡️ Guardrails & Next Steps
Post-launch KPIs to monitor. Risks. Any follow-up experiments recommended.
"""

    try:
        response = client.chat.completions.create(
            model=_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a world-class data scientist specialising in A/B experimentation "
                        "and business analytics. You write precise, quantitative executive readouts."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            temperature=0.3,
            max_tokens=2048,
        )
        return response.choices[0].message.content

    except Exception as exc:
        return (
            f"### ❌ Groq API Error\n\n"
            f"```\n{exc}\n```\n\n"
            "Check your `GROQ_API_KEY` in `.env` and try again.\n\n"
            "Get a free key at: **https://console.groq.com**"
        )
