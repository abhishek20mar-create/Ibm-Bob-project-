"""
frontend.py — Streamlit Frontend for Heart Attack Prediction
Run:  streamlit run frontend.py
Requires the FastAPI backend to be running at http://localhost:8000
"""

import requests
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

API_BASE = "http://localhost:8000"

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Heart Attack Prediction",
    page_icon="🫀",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main-title {font-size:2.2rem; font-weight:700; color:#c0392b; margin-bottom:0;}
    .subtitle   {font-size:1rem;   color:#7f8c8d;  margin-bottom:1.5rem;}
    .risk-high  {background:#fdecea; border-left:5px solid #c0392b; padding:1rem; border-radius:6px;}
    .risk-mod   {background:#fff8e1; border-left:5px solid #f39c12; padding:1rem; border-radius:6px;}
    .risk-low   {background:#e8f5e9; border-left:5px solid #27ae60; padding:1rem; border-radius:6px;}
    .metric-card{background:#f7f8fa; border-radius:8px; padding:1rem; text-align:center;}
</style>
""", unsafe_allow_html=True)


# ── Helpers ───────────────────────────────────────────────────────────────────
@st.cache_data(ttl=300)
def fetch_stats():
    try:
        return requests.get(f"{API_BASE}/model/stats", timeout=5).json()
    except Exception:
        return None

@st.cache_data(ttl=300)
def fetch_metadata():
    try:
        return requests.get(f"{API_BASE}/model/features", timeout=5).json()
    except Exception:
        return None

@st.cache_data(ttl=300)
def fetch_comparison():
    try:
        return requests.get(f"{API_BASE}/model/comparison", timeout=5).json()
    except Exception:
        return None


def gauge_chart(probability: float) -> go.Figure:
    pct = probability * 100
    color = "#c0392b" if pct >= 60 else "#f39c12" if pct >= 30 else "#27ae60"
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=pct,
        number={"suffix": "%", "font": {"size": 40}},
        delta={"reference": 50, "increasing": {"color": "#c0392b"}, "decreasing": {"color": "#27ae60"}},
        gauge={
            "axis": {"range": [0, 100], "tickwidth": 1},
            "bar":  {"color": color},
            "steps": [
                {"range": [0,  30], "color": "#e8f5e9"},
                {"range": [30, 60], "color": "#fff8e1"},
                {"range": [60,100], "color": "#fdecea"},
            ],
            "threshold": {
                "line": {"color": "black", "width": 3},
                "thickness": 0.75,
                "value": 50,
            },
        },
        title={"text": "Heart Attack Risk Probability"},
    ))
    fig.update_layout(height=280, margin=dict(t=40, b=10, l=20, r=20))
    return fig


def feature_importance_chart(importances: dict) -> go.Figure:
    df = pd.DataFrame(importances.items(), columns=["Feature", "Importance"])
    df = df.sort_values("Importance")
    fig = px.bar(
        df, x="Importance", y="Feature", orientation="h",
        color="Importance", color_continuous_scale="Reds",
        title="Feature Importances (Random Forest)"
    )
    fig.update_layout(
        height=380, coloraxis_showscale=False,
        margin=dict(t=50, b=20, l=10, r=10),
        yaxis_title="", xaxis_title="Importance Score"
    )
    return fig


def confusion_matrix_chart(cm: list) -> go.Figure:
    labels = ["No Disease", "Disease"]
    fig = px.imshow(
        cm, text_auto=True, color_continuous_scale="Reds",
        x=labels, y=labels,
        labels={"x": "Predicted", "y": "Actual"},
        title="Confusion Matrix (Test Set)"
    )
    fig.update_layout(height=300, margin=dict(t=50, b=20, l=20, r=20))
    return fig


# ── Sidebar — Patient Input Form ──────────────────────────────────────────────
st.sidebar.markdown("## 🩺 Patient Details")
st.sidebar.markdown("Fill in the patient's clinical parameters:")

meta = fetch_metadata() or {}

def sidebar_input(key, label, meta_item):
    if meta_item.get("type") == "select":
        opts = {int(k): v for k, v in meta_item["options"].items()}
        choice = st.sidebar.selectbox(label, options=list(opts.keys()), format_func=lambda x: opts[x], key=key)
        return float(choice)
    elif meta_item.get("type") == "float":
        return st.sidebar.slider(label, min_value=float(meta_item["min"]), max_value=float(meta_item["max"]),
                                  value=float((meta_item["min"] + meta_item["max"]) / 2),
                                  step=float(meta_item["step"]), key=key)
    else:
        return float(st.sidebar.number_input(label, min_value=int(meta_item["min"]), max_value=int(meta_item["max"]),
                                              value=int((meta_item["min"] + meta_item["max"]) / 2),
                                              step=int(meta_item["step"]), key=key))

# Default values used if metadata API is unavailable
defaults = {
    "age": 54.0, "sex": 1.0, "cp": 2.0, "trestbps": 130.0, "chol": 250.0,
    "fbs": 0.0, "restecg": 0.0, "thalach": 160.0, "exang": 0.0,
    "oldpeak": 1.4, "slope": 2.0, "ca": 0.0, "thal": 3.0
}

FIELD_ORDER = [
    "age", "sex", "cp", "trestbps", "chol", "fbs",
    "restecg", "thalach", "exang", "oldpeak", "slope", "ca", "thal"
]

input_values = {}
if meta:
    for key in FIELD_ORDER:
        m = meta.get(key, {})
        input_values[key] = sidebar_input(key, m.get("label", key), m)
else:
    st.sidebar.warning("⚠️ Could not load feature metadata from API. Using manual inputs.")
    input_values["age"]      = float(st.sidebar.number_input("Age",            29,  77, 54))
    input_values["sex"]      = float(st.sidebar.selectbox("Sex",               [0, 1], format_func=lambda x: "Female" if x == 0 else "Male"))
    input_values["cp"]       = float(st.sidebar.selectbox("Chest Pain Type",   [1,2,3,4]))
    input_values["trestbps"] = float(st.sidebar.number_input("Resting BP",     80, 200, 130))
    input_values["chol"]     = float(st.sidebar.number_input("Cholesterol",    100,600, 250))
    input_values["fbs"]      = float(st.sidebar.selectbox("Fasting BS>120",    [0, 1]))
    input_values["restecg"]  = float(st.sidebar.selectbox("Resting ECG",       [0, 1, 2]))
    input_values["thalach"]  = float(st.sidebar.number_input("Max Heart Rate", 60, 220, 160))
    input_values["exang"]    = float(st.sidebar.selectbox("Exercise Angina",   [0, 1]))
    input_values["oldpeak"]  = st.sidebar.slider("Oldpeak", 0.0, 7.0, 1.4, 0.1)
    input_values["slope"]    = float(st.sidebar.selectbox("ST Slope",          [1, 2, 3]))
    input_values["ca"]       = float(st.sidebar.selectbox("Major Vessels",     [0, 1, 2, 3]))
    input_values["thal"]     = float(st.sidebar.selectbox("Thalassemia",       [3, 6, 7]))

predict_btn = st.sidebar.button("🔍 Predict Risk", use_container_width=True, type="primary")

# ── Main Content ──────────────────────────────────────────────────────────────
st.markdown('<div class="main-title">🫀 Heart Attack Prediction</div>', unsafe_allow_html=True)
model_name = (stats or {}).get("model_name", "Best Model")
st.markdown(
    f'<div class="subtitle">AI-powered clinical decision support — Cleveland Heart Disease Dataset'
    f' &nbsp;·&nbsp; Active model: <strong>{model_name}</strong></div>',
    unsafe_allow_html=True,
)

# ── Model Stats ───────────────────────────────────────────────────────────────
stats = fetch_stats()
if stats:
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("🎯 Test Accuracy",    f"{stats['accuracy']*100:.1f}%")
    with col2:
        st.metric("📈 ROC-AUC Score",    f"{stats['roc_auc']:.4f}")
    with col3:
        st.metric("🔄 5-Fold CV Accuracy", f"{stats['cv_accuracy']*100:.1f}%")
    with col4:
        st.metric("📊 Training Samples", f"{stats['n_train']}")

st.divider()

# ── Prediction Result ─────────────────────────────────────────────────────────
if predict_btn:
    with st.spinner("Running prediction..."):
        try:
            resp = requests.post(f"{API_BASE}/predict", json=input_values, timeout=10)
            resp.raise_for_status()
            result = resp.json()
        except requests.exceptions.ConnectionError:
            st.error("❌ Cannot connect to the API. Make sure the FastAPI backend is running (`uvicorn app:app --reload`).")
            result = None
        except Exception as e:
            st.error(f"❌ Prediction failed: {e}")
            result = None

    if result:
        res_col, gauge_col = st.columns([1, 1])
        with res_col:
            risk = result["risk_level"]
            cls  = {"High": "risk-high", "Moderate": "risk-mod", "Low": "risk-low"}[risk]
            icon = {"High": "🔴", "Moderate": "🟡", "Low": "🟢"}[risk]
            st.markdown(f"""
            <div class="{cls}">
                <h2>{icon} {result['label']}</h2>
                <p><strong>Risk Level:</strong> {risk}</p>
                <p><strong>Probability:</strong> {result['probability']*100:.1f}%</p>
            </div>
            """, unsafe_allow_html=True)

            st.markdown("#### 🔑 Top Contributing Features")
            tf_df = pd.DataFrame(result["top_features"])
            tf_df.columns = ["Feature", "Importance"]
            st.dataframe(tf_df, use_container_width=True, hide_index=True)

        with gauge_col:
            st.plotly_chart(gauge_chart(result["probability"]), use_container_width=True)

        st.divider()

# ── Model Insights ─────────────────────────────────────────────────────────────
if stats:
    st.markdown("## 📊 Model Insights")
    imp_col, cm_col = st.columns([3, 2])
    with imp_col:
        st.plotly_chart(feature_importance_chart(stats["feature_importances"]), use_container_width=True)
    with cm_col:
        st.plotly_chart(confusion_matrix_chart(stats["confusion_matrix"]), use_container_width=True)

    st.markdown("## 📋 Classification Report")
    report = stats.get("classification_report", {})
    rows = []
    for k, v in report.items():
        if isinstance(v, dict):
            rows.append({"Class": k, **{kk: round(vv, 3) for kk, vv in v.items()}})
    if rows:
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    # ── Model Comparison ───────────────────────────────────────────────────────
    comparison = fetch_comparison()
    if comparison:
        with st.expander("📊 Model Comparison — LR vs Random Forest vs XGBoost", expanded=False):
            comp_df = pd.DataFrame(comparison)
            comp_df.columns = [c.upper() if c in ("f1",) else c.replace("_", " ").title()
                                for c in comp_df.columns]
            # Highlight the best row (highest ROC-AUC = first since train.py sorts by AUC)
            st.dataframe(
                comp_df.style.highlight_max(
                    subset=["Accuracy", "Precision", "Recall", "F1", "Roc Auc"],
                    color="#d4edda",
                ),
                use_container_width=True,
                hide_index=True,
            )

# ── Footer ─────────────────────────────────────────────────────────────────────
st.divider()
st.markdown(
    "<center><small>Heart Attack Prediction App · Logistic Regression · Random Forest · XGBoost · Cleveland Heart Disease Dataset</small></center>",
    unsafe_allow_html=True
)
