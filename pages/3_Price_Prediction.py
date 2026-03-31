"""
pages/3_Price_Prediction.py — AI Price Prediction Engine
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Uses sklearn Pipeline for inference (no manual encode/scale).
Includes: validation → prediction → explanation → model comparison → SHAP
"""

import os, sys, time
import streamlit as st
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from config import ML_FEATURES, OWNER_MAP, APP_NAME, CURRENT_YEAR
from utils.helpers import (html, fmt_price, sec_title, page_header, badge,
                            chart_feature_importance, chart_model_comparison)
from utils.validation import validate_prediction_input, ValidationError
from utils.feature_engineering import generate_prediction_explanation
from model.predict import predict_price
from model.evaluate import feature_importance_analysis, generate_comparison_table
from utils.logger import get_logger

logger = get_logger("page.prediction")

# ── CSS ───────────────────────────────────────────────────────────────────────
_CSS = os.path.join(os.path.dirname(os.path.dirname(__file__)), "assets", "styles.css")
if os.path.exists(_CSS):
    with open(_CSS, encoding="utf-8") as f:
        html(f"<style>\n{f.read()}\n</style>")

# ── Session state ─────────────────────────────────────────────────────────────
dff    = st.session_state.get("dff",    pd.DataFrame())
df_raw = st.session_state.get("df",     pd.DataFrame())
mdl    = st.session_state.get("mdl",    None)
enc    = st.session_state.get("enc",    None)
scaler = st.session_state.get("scaler", None)
meta   = st.session_state.get("meta",   None)

if dff.empty:
    st.warning("⚠️ No data loaded. Visit the Home page first.")
    st.stop()

html(page_header(
    "🤖 AI Price Prediction",
    "sklearn Pipeline · ColumnTransformer · Cross-Validation · SHAP Explainability",
))

col_form, col_out = st.columns([1, 1], gap="large")

# ══════════════════════════════════════════════════════════════════════════════
# LEFT COLUMN — INPUT FORM
# ══════════════════════════════════════════════════════════════════════════════
with col_form:
    html(sec_title("🔧", "Vehicle Specifications"))

    brands = sorted(dff["brand_clean"].dropna().unique())
    _brand = st.selectbox("Car Brand", brands, key="p_brand",
                          help="Brand extracted from car name — top 25 brands + 'Other'")

    yr_min = max(2000, int(dff["year"].min()))
    yr_max = min(CURRENT_YEAR, int(dff["year"].max()))
    _year  = st.slider("Manufacturing Year", yr_min, yr_max, min(2019, yr_max), key="p_year",
                       help="Year the car was manufactured — newer = higher price")
    car_age = CURRENT_YEAR - _year

    fuels  = sorted(dff["fuel"].dropna().unique())
    _fuel  = st.selectbox("Fuel Type", fuels, key="p_fuel",
                          help="Diesel/Electric command premium; CNG = affordable; Petrol = standard")

    trans  = sorted(dff["transmission"].dropna().unique())
    _trans = st.selectbox("Transmission", trans, key="p_trans",
                          help="Automatics command 20-40% premium in urban markets")

    sellers = sorted(dff["seller_type"].dropna().unique())
    _seller = st.selectbox("Seller Type", sellers, key="p_seller",
                           help="Trustmark Dealer > Dealer > Individual (trust factor)")

    _km_raw = st.text_input("KM Driven", "30000", placeholder="e.g. 35000", key="p_km",
                            help="Total odometer reading — directly impacts valuation")
    km_ok, _km = True, 0
    try:
        _km = int(_km_raw.replace(",", "").strip())
        if not (0 < _km < 1_000_000):
            st.error("Enter KM between 1 and 10,00,000")
            km_ok = False
    except ValueError:
        st.error("Enter a valid number for KM")
        km_ok = False

    owner_labels = [l for l in OWNER_MAP if l != "Test Drive Car"]
    _owner_lbl = st.selectbox("Ownership", owner_labels, key="p_owner",
                              help="First owner = 15-25% premium over multi-owner vehicles")
    _owner_num = OWNER_MAP[_owner_lbl]

    # Predict button
    go = st.button("🔮  Get AI Price Estimate", use_container_width=True, disabled=(not km_ok),
                   type="primary")

# ══════════════════════════════════════════════════════════════════════════════
# RIGHT COLUMN — PREDICTION OUTPUT
# ══════════════════════════════════════════════════════════════════════════════
with col_out:
    html(sec_title("📊", "Prediction Result"))

    if go and km_ok:
        if mdl is None or meta is None:
            st.error("Model not loaded. Ensure the Home page initialised the app.")
        else:
            with st.spinner("⚡ Running AI valuation…"):
                time.sleep(0.25)

                try:
                    # 1. Validate inputs
                    validated = validate_prediction_input(
                        brand=_brand, year=_year, fuel=_fuel,
                        transmission=_trans, seller_type=_seller,
                        km_driven=_km, owner_label=_owner_lbl,
                        valid_brands=brands, valid_fuels=fuels,
                        valid_trans=trans, valid_sellers=sellers,
                    )

                    # 2. Build input dict (raw names — Pipeline handles encoding)
                    input_row = {
                        "car_age":      validated["car_age"],
                        "km_driven":    validated["km_driven"],
                        "km_per_year":  validated["km_per_year"],
                        "owner_num":    validated["owner_num"],
                        "fuel":         validated["fuel"],
                        "seller_type":  validated["seller_type"],
                        "transmission": validated["transmission"],
                        "brand_clean":  validated["brand_clean"],
                    }

                    # 3. Predict via Pipeline
                    result = predict_price(mdl, meta, enc, scaler, input_row)
                    pred = result["price"]
                    r2   = result["r2"]
                    low  = result["price_low"]
                    high = result["price_high"]

                    # 4. Business explanation
                    explanation = generate_prediction_explanation(
                        car_age=validated["car_age"],
                        km_driven=validated["km_driven"],
                        fuel=validated["fuel"],
                        transmission=validated["transmission"],
                        owner_num=validated["owner_num"],
                        predicted_price=pred,
                        r2_score=r2,
                    )
                    conf, clab, ccol = explanation["confidence_label"]

                    reasons_html = ""
                    for icon, reason, sentiment in explanation["reasons"]:
                        reasons_html += f'<div style="margin:4px 0;font-size:.83rem;">{icon} {reason}</div>'

                    # 5. Render prediction card
                    pipeline_badge = badge("sklearn Pipeline", "emerald")
                    html(f"""
                    <div class="pred-card">
                      <div class="pred-label-top">Estimated Market Value</div>
                      <div class="pred-price">{fmt_price(pred)}</div>
                      <div class="pred-range">Likely range: {fmt_price(low)} – {fmt_price(high)}</div>
                      <div class="pred-badges">
                        {badge(meta['best_name'],'violet')}
                        {badge(f"R² {r2:.3f}",'cyan')}
                        {badge(clab, ccol)}
                        {pipeline_badge}
                        {badge(f"v{meta.get('version','?')}",'amber')}
                      </div>
                      <div style="margin-top:1rem;text-align:left;position:relative;z-index:1;">
                        <div style="font-size:.72rem;color:#64748B;text-transform:uppercase;
                                    letter-spacing:.07em;margin-bottom:.4rem;">
                          Model Confidence ({conf}%)
                        </div>
                        <div class="confidence-bar">
                          <div class="confidence-fill" style="width:{conf}%;"></div>
                        </div>
                      </div>
                      <div class="pred-explanation">
                        <b>Why this price?</b><br>
                        {reasons_html}
                        <div style="margin-top:8px;padding-top:8px;border-top:1px solid rgba(6,182,212,0.15);
                                    font-size:.82rem;color:#94A3B8;">
                          <b>Summary:</b> {explanation['summary']}
                        </div>
                      </div>
                    </div>
                    """)

                    logger.info(f"Prediction: ₹{pred:,.0f} for {_brand} {_year}")

                except ValidationError as e:
                    st.error(f"⚠️ Validation Error: {e}")
                except Exception as e:
                    st.error(f"❌ Prediction failed: {e}")
                    logger.error(f"Prediction error: {e}", exc_info=True)

            # Similar cars reference
            sim = dff[
                (dff["fuel"] == _fuel) &
                (dff["transmission"] == _trans) &
                (dff["brand_clean"] == _brand) &
                (dff["car_age"].between(max(0, car_age - 2), car_age + 2))
            ].copy()
            if sim.empty:
                sim = dff[
                    (dff["fuel"] == _fuel) &
                    (dff["transmission"] == _trans) &
                    (dff["car_age"].between(max(0, car_age - 2), car_age + 2))
                ].copy()

            if not sim.empty:
                html(sec_title("🔍", "Similar Cars in Dataset", f"{len(sim)} found"))
                show = [c for c in ["name", "year", "km_driven", "fuel",
                                    "transmission", "seller_type", "selling_price"]
                        if c in sim.columns]
                show_df = sim[show].rename(columns={"selling_price": "Price (₹)"}).copy()
                show_df["Price (₹)"] = show_df["Price (₹)"].apply(fmt_price)
                st.dataframe(show_df.head(8).reset_index(drop=True), use_container_width=True)

            # Download prediction report
            if 'pred' in dir():
                rep = pd.DataFrame([{
                    "Brand": _brand, "Year": _year, "Car Age": car_age,
                    "Fuel": _fuel, "Transmission": _trans, "Seller": _seller,
                    "KM Driven": _km, "Ownership": _owner_lbl,
                    "Predicted Price": round(pred, 2),
                    "Range Low": round(low, 2), "Range High": round(high, 2),
                    "Model": meta["best_name"], "R2": r2, "Confidence": f"{conf}%",
                    "Pipeline": meta.get("pipeline_type", "N/A"),
                    "Model Version": meta.get("version", "?"),
                }]).T.reset_index()
                rep.columns = ["Parameter", "Value"]
                st.download_button("⬇️  Download Prediction Report",
                                   rep.to_csv(index=False).encode("utf-8"),
                                   "cariq_prediction.csv", "text/csv")
    else:
        html("""
        <div style="background:rgba(21,33,58,0.8);border:1px dashed rgba(124,58,237,0.3);
                    border-radius:14px;padding:3.5rem 2rem;text-align:center;color:#64748B;">
          <div style="font-size:3rem;margin-bottom:1rem;">🔮</div>
          <div style="font-size:.95rem;line-height:1.6;">
            Configure vehicle specs on the left<br>and click
            <b style="color:#A78BFA">Get AI Price Estimate</b>.
          </div>
        </div>
        """)

# ══════════════════════════════════════════════════════════════════════════════
# MODEL PERFORMANCE & EXPLAINABILITY
# ══════════════════════════════════════════════════════════════════════════════
if meta:
    st.markdown("---")
    html(sec_title("🏆", "Model Performance & Comparison"))

    # Pipeline info badge
    pipeline_type = meta.get("pipeline_type", "N/A")
    html(f"""
    <div style="background:linear-gradient(135deg,rgba(16,185,129,0.08),rgba(124,58,237,0.06));
                border:1px solid rgba(16,185,129,0.2);border-radius:10px;
                padding:.7rem 1.2rem;margin-bottom:1rem;font-size:.82rem;color:#94A3B8;">
      <b style="color:#10B981;">Pipeline:</b> {pipeline_type} &nbsp;|&nbsp;
      <b style="color:#A78BFA;">Seed:</b> {meta.get('random_seed', '?')} &nbsp;|&nbsp;
      <b style="color:#06B6D4;">CV:</b> {meta.get('cv_folds', '?')}-fold &nbsp;|&nbsp;
      <b style="color:#F59E0B;">Test:</b> {meta.get('test_size', 0.2)*100:.0f}% holdout &nbsp;|&nbsp;
      <b style="color:#F43F5E;">SHAP:</b> {'✅' if meta.get('shap_importance') else '❌'}
    </div>
    """)

    c1, c2 = st.columns([3, 2], gap="large")
    with c1:
        st.plotly_chart(chart_model_comparison(meta["results"]), use_container_width=True)
    with c2:
        for name, res in meta["results"].items():
            is_best = name == meta["best_name"]
            mae_html = ""
            if "mae" in res and res["mae"]:
                mae_html = f"""
                <div class="score-item">
                  <span class="score-label">MAE</span>
                  <span class="score-val" style="color:#F43F5E">&#8377;{res['mae']/1e5:.2f}L</span>
                </div>"""
            cv_html = ""
            if "cv_r2_mean" in res:
                cv_html = f"""
                <div class="score-item">
                  <span class="score-label">CV R²</span>
                  <span class="score-val" style="color:#A78BFA">{res['cv_r2_mean']:.4f}±{res.get('cv_r2_std',0):.3f}</span>
                </div>"""
            html(f"""
            <div class="model-card {'best' if is_best else ''}">
              <div class="model-name">{"🏆 " if is_best else "📊 "}{name}</div>
              <div class="model-scores">
                <div class="score-item">
                  <span class="score-label">R² Score</span>
                  <span class="score-val" style="color:#10B981">{res['r2']:.4f}</span>
                </div>
                <div class="score-item">
                  <span class="score-label">RMSE</span>
                  <span class="score-val" style="color:#F59E0B">&#8377;{res['rmse']/1e5:.2f}L</span>
                </div>
                {mae_html}
                {cv_html}
              </div>
            </div>""")

    # Feature Importance / SHAP
    fi_df = feature_importance_analysis(meta)
    if fi_df is not None and not fi_df.empty:
        shap_tag = "SHAP" if meta.get("shap_importance") else "Tree Importance"
        html(sec_title("🎯", f"Feature Importance ({shap_tag})", meta["best_name"]))

        fi_series = pd.Series(
            fi_df["Importance"].values,
            index=fi_df["Feature Code"].values,
        )
        st.plotly_chart(chart_feature_importance(fi_series), use_container_width=True)

        with st.expander("📖 Feature Business Impact Analysis", expanded=False):
            display_cols = ["Feature", "Importance", "Contribution (%)", "Source", "Business Impact"]
            display_cols = [c for c in display_cols if c in fi_df.columns]
            st.dataframe(fi_df[display_cols], use_container_width=True, hide_index=True)

    # Comparison table download
    with st.expander("📋 Full Model Comparison Table", expanded=False):
        comp_df = generate_comparison_table(meta)
        st.dataframe(comp_df, use_container_width=True, hide_index=True)
        st.download_button("⬇️ Download Comparison", comp_df.to_csv(index=False).encode(),
                           "cariq_model_comparison.csv", "text/csv")
