"""
pages/5_Model_Lab.py — Automated Model Training & Tuning Lab
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
One-click training, hyperparameter tuning, live progress,
model comparison, and artifact management — all from the UI.
"""

import os, sys, json, time
from datetime import datetime
import streamlit as st
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from config import (
    APP_NAME, ARTIFACTS_DIR, META_JSON, MODEL_PKL,
    CV_FOLDS, TEST_SIZE, RANDOM_SEED, HYPERPARAM_GRIDS,
    NUM_FEATURES, CAT_FEATURES,
)
from utils.helpers import (html, fmt_price, sec_title, page_header, badge,
                            chart_model_comparison, chart_feature_importance)
from model.evaluate import feature_importance_analysis, generate_comparison_table, model_summary
from utils.logger import get_logger

logger = get_logger("page.model_lab")

# ── CSS ────────────────────────────────────────────────────────────────────────
_CSS = os.path.join(os.path.dirname(os.path.dirname(__file__)), "assets", "styles.css")
if os.path.exists(_CSS):
    with open(_CSS, encoding="utf-8") as f:
        html(f"<style>\n{f.read()}\n</style>")

# ── Page Header ───────────────────────────────────────────────────────────────
html(page_header(
    "🧪 Model Lab",
    "Automated training · Hyperparameter tuning · Live progress · Model comparison",
))

# ── Load current model info ───────────────────────────────────────────────────
meta = st.session_state.get("meta", None)
if meta is None and os.path.exists(META_JSON):
    with open(META_JSON) as f:
        meta = json.load(f)

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 1 — Current Model Status
# ══════════════════════════════════════════════════════════════════════════════
html(sec_title("📋", "Current Model Status"))

if meta:
    best      = meta["best_name"]
    best_res  = meta["results"][best]
    ver       = meta.get("version", "?")
    t_sec     = meta.get("training_time_sec", "?")
    n_rows    = meta.get("dataset_rows", "?")
    n_models  = len(meta.get("results", {}))
    pipe_type = meta.get("pipeline_type", "N/A")
    is_tuned  = meta.get("tuned", False)
    has_shap  = meta.get("shap_importance") is not None

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        html(f"""
        <div class="kpi-card violet">
          <div class="kpi-orb"></div>
          <div class="kpi-icon-wrap">🏆</div>
          <div class="kpi-label">Best Model</div>
          <div class="kpi-value">{best}</div>
          <div class="kpi-sub">v{ver}</div>
        </div>""")
    with c2:
        html(f"""
        <div class="kpi-card cyan">
          <div class="kpi-orb"></div>
          <div class="kpi-icon-wrap">🎯</div>
          <div class="kpi-label">R² Score</div>
          <div class="kpi-value">{best_res['r2']:.4f}</div>
          <div class="kpi-sub">RMSE: {fmt_price(best_res['rmse'])}</div>
        </div>""")
    with c3:
        html(f"""
        <div class="kpi-card amber">
          <div class="kpi-orb"></div>
          <div class="kpi-icon-wrap">📊</div>
          <div class="kpi-label">Models Trained</div>
          <div class="kpi-value">{n_models}</div>
          <div class="kpi-sub">{n_rows:,} rows · {t_sec}s</div>
        </div>""")
    with c4:
        status_parts = []
        if is_tuned: status_parts.append("Tuned")
        if has_shap: status_parts.append("SHAP")
        status_text = " + ".join(status_parts) if status_parts else "Standard"
        html(f"""
        <div class="kpi-card emerald">
          <div class="kpi-orb"></div>
          <div class="kpi-icon-wrap">⚙️</div>
          <div class="kpi-label">Pipeline</div>
          <div class="kpi-value" style="font-size:1rem;">{status_text}</div>
          <div class="kpi-sub">{pipe_type.split('+')[0].strip()}</div>
        </div>""")

    # Pipeline metadata strip
    html(f"""
    <div style="background:linear-gradient(135deg,rgba(16,185,129,0.08),rgba(124,58,237,0.06));
                border:1px solid rgba(16,185,129,0.2);border-radius:10px;
                padding:.6rem 1.2rem;margin:1rem 0;font-size:.8rem;color:#94A3B8;">
      <b style="color:#10B981;">Pipeline:</b> {pipe_type} &nbsp;|&nbsp;
      <b style="color:#A78BFA;">Seed:</b> {RANDOM_SEED} &nbsp;|&nbsp;
      <b style="color:#06B6D4;">CV:</b> {meta.get('cv_folds', CV_FOLDS)}-fold &nbsp;|&nbsp;
      <b style="color:#F59E0B;">Split:</b> {meta.get('test_size', TEST_SIZE)*100:.0f}% holdout &nbsp;|&nbsp;
      <b style="color:#F43F5E;">SHAP:</b> {'Yes' if has_shap else 'No'} &nbsp;|&nbsp;
      <b style="color:#EC4899;">Tuned:</b> {'Yes' if is_tuned else 'No'}
    </div>
    """)
else:
    html("""
    <div style="background:rgba(21,33,58,0.8);border:1px dashed rgba(245,158,11,0.4);
                border-radius:14px;padding:2rem;text-align:center;color:#F59E0B;">
      <div style="font-size:2rem;margin-bottom:.8rem;">⚠️</div>
      <div style="font-size:.95rem;">No trained model found.<br>
      Use the training panel below to train your first model!</div>
    </div>
    """)

st.markdown("---")

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 2 — Training Control Panel
# ══════════════════════════════════════════════════════════════════════════════
html(sec_title("🚀", "Training Control Panel"))

html("""
<div style="background:rgba(21,33,58,0.6);border:1px solid rgba(124,58,237,0.2);
            border-radius:12px;padding:1.2rem 1.5rem;margin-bottom:1.2rem;">
  <div style="font-size:.85rem;color:#94A3B8;line-height:1.7;">
    <b style="color:#E2E8F0;">How it works:</b><br>
    1. Choose your training mode below<br>
    2. Click the train button — the pipeline runs <b>right here in the dashboard</b><br>
    3. Watch live progress: data loading → model training → cross-validation → SHAP → artifact saving<br>
    4. When done, the new model instantly replaces the old one across all pages
  </div>
</div>
""")

col_cfg, col_action = st.columns([2, 1], gap="large")

with col_cfg:
    # Training mode
    mode = st.radio(
        "Training Mode",
        ["Standard Training", "Hyperparameter Tuning"],
        captions=[
            "Train all 4 models with default hyperparameters (~15s)",
            "Train + RandomizedSearchCV for each model (~3-10 min)",
        ],
        key="train_mode",
        horizontal=True,
    )
    tune = mode == "Hyperparameter Tuning"

    c_cv, c_split = st.columns(2)
    with c_cv:
        cv_folds = st.slider("Cross-Validation Folds", 3, 15, CV_FOLDS, key="cv_slider",
                             help="More folds = more accurate evaluation but slower training")
    with c_split:
        test_pct = st.slider("Test Split %", 10, 40, int(TEST_SIZE * 100), key="split_slider",
                             help="Percentage of data held out for final evaluation")

    # Show tuning grids
    if tune:
        with st.expander("🔧 Hyperparameter Search Grids", expanded=False):
            for model_name, grid in HYPERPARAM_GRIDS.items():
                st.markdown(f"**{model_name}**")
                for param, values in grid.items():
                    st.code(f"{param}: {values}", language="python")

with col_action:
    html(f"""
    <div style="background:rgba(124,58,237,0.08);border:1px solid rgba(124,58,237,0.25);
                border-radius:12px;padding:1.5rem;text-align:center;margin-top:.5rem;">
      <div style="font-size:1.5rem;margin-bottom:.5rem;">
        {'⚡' if not tune else '🔬'}
      </div>
      <div style="font-size:.85rem;color:#A78BFA;font-weight:600;margin-bottom:.3rem;">
        {'Standard' if not tune else 'Tuning'} Mode
      </div>
      <div style="font-size:.75rem;color:#64748B;">
        {'~15 seconds' if not tune else '~3-10 minutes'}<br>
        {cv_folds}-fold CV · {test_pct}% holdout<br>
        4 models · {'20 iter/model' if tune else 'default params'}
      </div>
    </div>
    """)

    train_btn = st.button(
        f"{'⚡' if not tune else '🔬'}  Start {'Training' if not tune else 'Tuning'}",
        use_container_width=True,
        type="primary",
        key="train_btn",
    )

    if meta:
        retrain_warning = st.checkbox(
            "I understand this replaces the current model",
            key="retrain_confirm",
        )
    else:
        retrain_warning = True  # no existing model to worry about

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 3 — Live Training Execution
# ══════════════════════════════════════════════════════════════════════════════
if train_btn:
    if not retrain_warning and meta:
        st.warning("⚠️ Please confirm you want to replace the current model.")
    else:
        st.markdown("---")
        html(sec_title("⏳", "Training in Progress..."))

        # Progress containers
        progress_bar = st.progress(0, text="Initializing pipeline...")
        status_box   = st.empty()
        log_expander = st.expander("📋 Training Log (live)", expanded=True)
        log_lines    = []

        def streamlit_callback(info: dict):
            """Receive progress updates from model.train and render them live."""
            pct      = info.get("pct", 0)
            message  = info.get("message", "")
            stage    = info.get("stage", "")

            # Update progress bar
            progress_bar.progress(min(pct / 100, 1.0), text=message)

            # Build status card
            stage_icons = {
                "load": "📥", "split": "✂️", "training": "🔧",
                "tuning": "🔬", "trained": "✅", "best": "🏆",
                "shap": "🔬", "save": "💾", "done": "🎉",
            }
            icon = stage_icons.get(stage, "⏳")

            # Add to log
            timestamp = datetime.now().strftime("%H:%M:%S")
            log_entry = f"`{timestamp}` {icon} {message}"

            # Add metrics if available
            if stage == "trained":
                r2  = info.get("r2", "")
                mae = info.get("mae", "")
                if r2:
                    log_entry += f" · **R²={r2:.4f}** · MAE={fmt_price(mae)}"

            log_lines.append(log_entry)
            with log_expander:
                st.markdown("\n\n".join(log_lines))

            # Status card
            if stage == "trained":
                model_name = info.get("model_name", "")
                r2         = info.get("r2", 0)
                rmse       = info.get("rmse", 0)
                cv_mean    = info.get("cv_r2_mean", 0)
                color      = "#10B981" if r2 > 0.7 else "#F59E0B" if r2 > 0.5 else "#F43F5E"
                status_box.markdown(f"""
                <div style="background:rgba(21,33,58,0.8);border-left:3px solid {color};
                            border-radius:0 8px 8px 0;padding:.6rem 1rem;margin:.3rem 0;
                            font-size:.82rem;color:#E2E8F0;">
                  ✅ <b>{model_name}</b> &nbsp;
                  R²=<b style="color:{color}">{r2:.4f}</b> &nbsp;
                  RMSE={fmt_price(rmse)} &nbsp;
                  CV={cv_mean:.4f}
                </div>
                """, unsafe_allow_html=True)

        # ── RUN TRAINING ──────────────────────────────────────────────────────
        try:
            from model.train import train as run_train

            new_meta = run_train(
                verbose=False,
                tune_hyperparams=tune,
                cv_folds=cv_folds,
                progress_callback=streamlit_callback,
            )

            # ── SUCCESS ───────────────────────────────────────────────────────
            progress_bar.progress(1.0, text="Training complete!")

            # Reload artifacts into session state
            from model.predict import load_artifacts
            mdl_new, enc_new, scaler_new, meta_new = load_artifacts()
            st.session_state.update({
                "mdl":    mdl_new,
                "enc":    enc_new,
                "scaler": scaler_new,
                "meta":   meta_new,
            })
            meta = meta_new  # update local reference

            # Success banner
            best_r2   = new_meta["results"][new_meta["best_name"]]["r2"]
            best_rmse = new_meta["results"][new_meta["best_name"]]["rmse"]
            elapsed   = new_meta.get("training_time_sec", "?")

            html(f"""
            <div style="background:linear-gradient(135deg,rgba(16,185,129,0.15),rgba(124,58,237,0.1));
                        border:1px solid rgba(16,185,129,0.3);border-radius:14px;
                        padding:1.5rem 2rem;margin:1.5rem 0;text-align:center;">
              <div style="font-size:2rem;margin-bottom:.5rem;">🎉</div>
              <div style="font-size:1.1rem;color:#10B981;font-weight:700;margin-bottom:.5rem;">
                Training Complete!
              </div>
              <div style="font-size:.9rem;color:#E2E8F0;">
                {badge(new_meta['best_name'], 'violet')}
                {badge(f"R² {best_r2:.4f}", 'cyan')}
                {badge(f"RMSE {fmt_price(best_rmse)}", 'amber')}
                {badge(f"v{new_meta.get('version','?')}", 'emerald')}
                {badge(f"{elapsed}s", 'violet')}
              </div>
              <div style="font-size:.78rem;color:#64748B;margin-top:.8rem;">
                Model automatically loaded across all dashboard pages.
                {' Hyperparameter tuning was applied.' if tune else ''}
              </div>
            </div>
            """)

            logger.info(f"UI training complete: {new_meta['best_name']} R²={best_r2:.4f}")

        except Exception as e:
            progress_bar.progress(0, text="Training failed!")
            st.error(f"❌ Training failed: {e}")
            logger.error(f"UI training error: {e}", exc_info=True)

st.markdown("---")

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 4 — Model Comparison (if trained)
# ══════════════════════════════════════════════════════════════════════════════
if meta:
    html(sec_title("🏆", "Model Comparison"))

    st.plotly_chart(chart_model_comparison(meta["results"]), use_container_width=True)

    # Detailed comparison table
    comp_df = generate_comparison_table(meta)
    st.dataframe(comp_df, use_container_width=True, hide_index=True)

    # Feature importance
    fi_df = feature_importance_analysis(meta)
    if fi_df is not None and not fi_df.empty:
        shap_tag = "SHAP" if meta.get("shap_importance") else "Tree Importance"
        html(sec_title("🎯", f"Feature Importance ({shap_tag})"))

        fi_series = pd.Series(fi_df["Importance"].values, index=fi_df["Feature Code"].values)
        st.plotly_chart(chart_feature_importance(fi_series), use_container_width=True)

    st.markdown("---")

    # ── Artifact Management ───────────────────────────────────────────────────
    html(sec_title("💾", "Artifact Management"))

    ac1, ac2, ac3 = st.columns(3)

    with ac1:
        # Download model metadata
        meta_str = json.dumps(meta, indent=2)
        st.download_button("📥 Download Metadata (JSON)", meta_str.encode(),
                           "model_meta.json", "application/json",
                           use_container_width=True)

    with ac2:
        # Download model summary report
        summary = model_summary(meta)
        st.download_button("📄 Download Report (TXT)", summary.encode(),
                           "cariq_model_report.txt", "text/plain",
                           use_container_width=True)

    with ac3:
        # Download comparison CSV
        if comp_df is not None:
            st.download_button("📊 Download Comparison (CSV)",
                               comp_df.to_csv(index=False).encode(),
                               "model_comparison.csv", "text/csv",
                               use_container_width=True)

    # Metadata viewer
    with st.expander("🔍 Full Model Metadata", expanded=False):
        st.json(meta)

    # Training history tip
    html("""
    <div style="background:rgba(21,33,58,0.6);border:1px solid rgba(6,182,212,0.2);
                border-radius:10px;padding:.8rem 1.2rem;margin-top:1rem;
                font-size:.8rem;color:#94A3B8;line-height:1.6;">
      <b style="color:#06B6D4;">Tip:</b>
      Every training run generates a new versioned model (timestamp-based).
      Check <code>model/artifacts/model_meta.json</code> for the full training config,
      or inspect <code>logs/cariq.log</code> for detailed training logs.
    </div>
    """)
