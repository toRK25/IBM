import streamlit as st
import pickle
import numpy as np
import pandas as pd
import time
import os
from datetime import datetime

# ─────────────────────────────────────────────
#  PAGE CONFIG  (must be first Streamlit call)
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Kiko – Live Chat Filter",
    page_icon="🔮",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
#  GLOBAL STYLES
# ─────────────────────────────────────────────
st.markdown("""
<style>
/* ── Google Fonts ── */
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Sans:ital,wght@0,300;0,400;0,500;1,300&display=swap');

/* ── Root tokens ── */
:root {
    --bg:        #0b0e1a;
    --surface:   #131729;
    --surface2:  #1a1f35;
    --border:    #252b45;
    --accent:    #6c63ff;
    --accent2:   #00d4aa;
    --pos:       #00d4aa;
    --neg:       #ff4d6d;
    --neu:       #f4c542;
    --text:      #e8eaf6;
    --muted:     #6b7194;
    --radius:    14px;
}

/* ── Base reset ── */
html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
    color: var(--text);
    background-color: var(--bg) !important;
}
.stApp { background-color: var(--bg) !important; }

/* ── Hide default Streamlit chrome ── */
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding: 2rem 2.5rem 4rem; max-width: 1300px; }

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: var(--surface) !important;
    border-right: 1px solid var(--border);
}
[data-testid="stSidebar"] * { color: var(--text) !important; }
[data-testid="stSidebar"] .stMarkdown h1 { font-family: 'Syne', sans-serif; }

/* ── Typography ── */
h1, h2, h3, .brand { font-family: 'Syne', sans-serif; letter-spacing: -0.5px; }

/* ── Textarea ── */
textarea {
    background: var(--surface2) !important;
    border: 1px solid var(--border) !important;
    border-radius: var(--radius) !important;
    color: var(--text) !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 15px !important;
    line-height: 1.7 !important;
    transition: border-color .25s;
}
textarea:focus { border-color: var(--accent) !important; box-shadow: 0 0 0 3px rgba(108,99,255,.18) !important; }

/* ── Metrics ── */
[data-testid="metric-container"] {
    background: var(--surface) !important;
    border: 1px solid var(--border) !important;
    border-radius: var(--radius) !important;
    padding: 1rem 1.2rem !important;
}
[data-testid="metric-container"] label { color: var(--muted) !important; font-size: 11px !important; text-transform: uppercase; letter-spacing: 1px; }
[data-testid="metric-container"] [data-testid="stMetricValue"] { font-family: 'Syne', sans-serif !important; font-size: 26px !important; }

/* ── Buttons ── */
.stButton > button {
    background: var(--surface2) !important;
    border: 1px solid var(--border) !important;
    color: var(--text) !important;
    border-radius: 8px !important;
    font-family: 'DM Sans', sans-serif !important;
    transition: all .2s;
}
.stButton > button:hover {
    border-color: var(--accent) !important;
    background: rgba(108,99,255,.12) !important;
}

/* ── Dataframe ── */
[data-testid="stDataFrame"] {
    border: 1px solid var(--border) !important;
    border-radius: var(--radius) !important;
    overflow: hidden;
}

/* ── Progress bar ── */
.stProgress > div > div { background-color: var(--accent) !important; border-radius: 999px; }
.stProgress > div { background-color: var(--border) !important; border-radius: 999px; }

/* ── Divider ── */
hr { border-color: var(--border) !important; margin: 1.5rem 0; }

/* ── Prediction card ── */
.pred-card {
    padding: 1.5rem 2rem;
    border-radius: var(--radius);
    border-left: 5px solid transparent;
    margin: .5rem 0 1.5rem;
    animation: fadeSlide .35s ease;
}
.pred-pos  { background: rgba(0,212,170,.08);  border-color: var(--pos); }
.pred-neg  { background: rgba(255,77,109,.08); border-color: var(--neg); }
.pred-neu  { background: rgba(244,197,66,.08); border-color: var(--neu); }
.pred-label { font-family: 'Syne', sans-serif; font-size: 2rem; font-weight: 800; }
.pred-sub   { font-size: 13px; color: var(--muted); margin-top: .25rem; }

@keyframes fadeSlide {
    from { opacity: 0; transform: translateY(8px); }
    to   { opacity: 1; transform: translateY(0);   }
}

/* ── Confidence pill ── */
.conf-pill {
    display: inline-block;
    padding: 2px 12px;
    border-radius: 999px;
    font-size: 12px;
    font-weight: 600;
    letter-spacing: .5px;
    margin-top: .5rem;
}
.conf-high { background: rgba(0,212,170,.2);  color: var(--pos); }
.conf-med  { background: rgba(244,197,66,.2); color: var(--neu); }
.conf-low  { background: rgba(255,77,109,.2); color: var(--neg); }

/* ── Brand header ── */
.brand-bar {
    display: flex; align-items: center; gap: 12px;
    margin-bottom: 2rem;
}
.brand-icon {
    width: 44px; height: 44px;
    background: linear-gradient(135deg, var(--accent), var(--accent2));
    border-radius: 12px;
    display: flex; align-items: center; justify-content: center;
    font-size: 22px;
}
.brand-name { font-family: 'Syne', sans-serif; font-size: 1.6rem; font-weight: 800; }
.brand-tag  { font-size: 12px; color: var(--muted); letter-spacing: 1px; text-transform: uppercase; }

/* ── Section headings ── */
.sec-head {
    font-family: 'Syne', sans-serif;
    font-size: .75rem;
    letter-spacing: 2px;
    text-transform: uppercase;
    color: var(--muted);
    margin: 1.5rem 0 .75rem;
}

/* ── History table row colors ── */
.row-pos { color: var(--pos) !important; }
.row-neg { color: var(--neg) !important; }

/* ── Toggle ── */
[data-testid="stCheckbox"] label { font-size: 13px; color: var(--muted) !important; }

/* ── Sidebar info blocks ── */
.info-block {
    background: var(--surface2);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: .85rem 1rem;
    margin-bottom: .6rem;
    font-size: 13px;
}
.info-block b { color: var(--accent2); }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
#  MODEL LOADING
# ─────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_model():
    """Load TF-IDF vectorizer and LinearSVC model from disk."""
    with open("vector.pkl", "rb") as f:
        vectorizer = pickle.load(f)
    with open("model.pkl", "rb") as f:
        model = pickle.load(f)
    return vectorizer, model


# ─────────────────────────────────────────────
#  PREDICTION LOGIC
# ─────────────────────────────────────────────
def predict_sentiment(text: str, vectorizer, model):
    """
    Transform text → predict label + confidence.
    Returns (label, confidence_score, raw_scores).
    """
    vec = vectorizer.transform([text])
    label = model.predict(vec)[0]

    # decision_function for confidence
    scores = model.decision_function(vec)[0]

    # Normalise to [0,1] probability-like value
    if isinstance(scores, np.ndarray):
        exp_s = np.exp(scores - scores.max())
        prob  = exp_s / exp_s.sum()
        conf  = float(prob.max())
    else:
        # Binary case
        conf = float(1 / (1 + np.exp(-abs(scores))))

    return label, conf, scores


def label_meta(label: str):
    """Return (emoji, css_class, color_name) for a label."""
    lbl = str(label).strip().lower()
    if "pos" in lbl:
        return "😊", "pred-pos", "#00d4aa", "Positive"
    elif "neg" in lbl:
        return "😔", "pred-neg", "#ff4d6d", "Negative"
    else:
        return "😐", "pred-neu", "#f4c542", "Neutral"


def text_stats(text: str):
    """Return (words, chars, sentences) counts."""
    words     = len(text.split()) if text.strip() else 0
    chars     = len(text)
    sentences = text.count(".") + text.count("!") + text.count("?")
    return words, chars, max(sentences, 1 if text.strip() else 0)


# ─────────────────────────────────────────────
#  SESSION STATE INIT
# ─────────────────────────────────────────────
def init_state():
    defaults = {
        "history":      [],          # list of dicts
        "pred_counts":  {"Positive": 0, "Negative": 0, "Neutral": 0},
        "total_preds":  0,
        "show_conf":    True,
        "last_input":   "",
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

init_state()


# ─────────────────────────────────────────────
#  SIDEBAR
# ─────────────────────────────────────────────
def render_sidebar():
    with st.sidebar:
        st.markdown("""
        <div style="padding:1.2rem 0 1rem;">
            <div style="display:flex;align-items:center;gap:10px;margin-bottom:.3rem;">
                <div style="width:38px;height:38px;background:linear-gradient(135deg,#6c63ff,#00d4aa);
                            border-radius:10px;display:flex;align-items:center;justify-content:center;font-size:20px;">
                    🔮
                </div>
                <div>
                    <div style="font-family:'Syne',sans-serif;font-size:1.3rem;font-weight:800;">Kiko</div>
                    <div style="font-size:10px;color:#6b7194;letter-spacing:1.5px;text-transform:uppercase;">Live Chat Filter</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.divider()

        # ── Model Info ──
        st.markdown('<div class="sec-head">⚙ Model</div>', unsafe_allow_html=True)
        st.markdown("""
        <div class="info-block">
            <b>Classifier</b><br>LinearSVC (Support Vector)<br><br>
            <b>Vectorizer</b><br>TF-IDF (n-gram 1–2)<br><br>
            <b>Accuracy</b><br>88% (validation set)
        </div>
        """, unsafe_allow_html=True)

        # ── Dataset Info ──
        st.markdown('<div class="sec-head">📦 Dataset</div>', unsafe_allow_html=True)
        st.markdown("""
        <div class="info-block">
            <b>Source</b><br>Custom chat corpus<br><br>
            <b>Size</b><br>~50,000 labelled samples<br><br>
            <b>Classes</b><br>Positive · Negative · Neutral
        </div>
        """, unsafe_allow_html=True)

        # ── Options ──
        st.markdown('<div class="sec-head">🎛 Options</div>', unsafe_allow_html=True)
        st.session_state.show_conf = st.checkbox(
            "Show confidence score", value=st.session_state.show_conf
        )
        show_stats = st.checkbox("Show text statistics", value=True)
        show_chart = st.checkbox("Show distribution chart", value=True)
        show_hist  = st.checkbox("Show prediction history", value=True)

        st.divider()
        st.markdown(
            '<div style="font-size:11px;color:#6b7194;text-align:center;">Kiko v1.0 · Built with Streamlit</div>',
            unsafe_allow_html=True,
        )
        return show_stats, show_chart, show_hist


# ─────────────────────────────────────────────
#  MAIN HEADER
# ─────────────────────────────────────────────
def render_header():
    st.markdown("""
    <div class="brand-bar">
        <div class="brand-icon">🔮</div>
        <div>
            <div class="brand-name">Kiko — Live Chat Filter</div>
            <div class="brand-tag">Real-time Sentiment Intelligence · LinearSVC + TF-IDF</div>
        </div>
    </div>
    """, unsafe_allow_html=True)


# ─────────────────────────────────────────────
#  METRICS ROW
# ─────────────────────────────────────────────
def render_metrics(words: int, chars: int):
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("🎯 Model Accuracy", "88%")
    with c2:
        st.metric("📊 Predictions Made", st.session_state.total_preds)
    with c3:
        st.metric("📝 Word Count", words)
    with c4:
        st.metric("🔤 Characters", chars)


# ─────────────────────────────────────────────
#  PREDICTION CARD
# ─────────────────────────────────────────────
def render_prediction(label, conf, show_conf):
    emoji, css, color, display = label_meta(label)

    conf_label = "High" if conf > .70 else ("Medium" if conf > .45 else "Low")
    conf_class = "conf-high" if conf > .70 else ("conf-med" if conf > .45 else "conf-low")

    st.markdown(f"""
    <div class="pred-card {css}">
        <div class="pred-label" style="color:{color};">{emoji} {display}</div>
        <div class="pred-sub">Sentiment detected by LinearSVC classifier</div>
        {f'<span class="conf-pill {conf_class}">Confidence: {conf:.1%} — {conf_label}</span>' if show_conf else ""}
    </div>
    """, unsafe_allow_html=True)

    if show_conf:
        st.progress(conf)


# ─────────────────────────────────────────────
#  DISTRIBUTION CHART
# ─────────────────────────────────────────────
def render_chart():
    counts = st.session_state.pred_counts
    if sum(counts.values()) == 0:
        st.markdown(
            '<div style="color:#6b7194;font-size:13px;text-align:center;padding:1rem 0;">'
            'No predictions yet — chart will update as you type.'
            '</div>',
            unsafe_allow_html=True,
        )
        return

    df = pd.DataFrame({
        "Sentiment": list(counts.keys()),
        "Count":     list(counts.values()),
    })

    # Colour map that matches our palette
    colors = {"Positive": "#00d4aa", "Negative": "#ff4d6d", "Neutral": "#f4c542"}
    df["Color"] = df["Sentiment"].map(colors)

    # Use st.bar_chart for simplicity (no extra deps)
    chart_df = df.set_index("Sentiment")[["Count"]]
    st.bar_chart(chart_df, color="#6c63ff", height=220)


# ─────────────────────────────────────────────
#  HISTORY TABLE
# ─────────────────────────────────────────────
def render_history():
    hist = st.session_state.history
    if not hist:
        st.markdown(
            '<div style="color:#6b7194;font-size:13px;">No history yet.</div>',
            unsafe_allow_html=True,
        )
        return

    df = pd.DataFrame(hist[::-1])  # newest first
    st.dataframe(
        df,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Time":      st.column_config.TextColumn("⏱ Time",      width="small"),
            "Sentiment": st.column_config.TextColumn("🏷 Sentiment", width="small"),
            "Confidence":st.column_config.ProgressColumn("📈 Conf", format="%.0f%%", min_value=0, max_value=100),
            "Text":      st.column_config.TextColumn("💬 Input",     width="large"),
        },
    )

    if st.button("🗑  Clear History"):
        st.session_state.history     = []
        st.session_state.pred_counts = {"Positive": 0, "Negative": 0, "Neutral": 0}
        st.session_state.total_preds = 0
        st.rerun()


# ─────────────────────────────────────────────
#  RECORD HISTORY (deduplicated)
# ─────────────────────────────────────────────
def record(text: str, label: str, conf: float):
    """Append to session history, avoiding duplicate back-to-back entries."""
    _, _, _, display = label_meta(label)

    last = st.session_state.history[-1] if st.session_state.history else None
    if last and last["Text"] == text[:80]:
        return  # same input, skip

    st.session_state.history.append({
        "Time":       datetime.now().strftime("%H:%M:%S"),
        "Sentiment":  display,
        "Confidence": round(conf * 100, 1),
        "Text":       text[:80] + ("…" if len(text) > 80 else ""),
    })
    st.session_state.pred_counts[display] = st.session_state.pred_counts.get(display, 0) + 1
    st.session_state.total_preds += 1


# ─────────────────────────────────────────────
#  MAIN APP
# ─────────────────────────────────────────────
def main():
    # ── Sidebar ──
    show_stats, show_chart, show_hist = render_sidebar()

    # ── Load model (cached) ──
    try:
        vectorizer, model = load_model()
        model_loaded = True
    except FileNotFoundError:
        model_loaded = False

    # ── Header ──
    render_header()

    if not model_loaded:
        st.error(
            "⚠️  **model.pkl** or **vector.pkl** not found in the current directory.\n\n"
            "Place both files alongside `app.py` and restart the app."
        )
        st.stop()

    # ── Input area ──
    st.markdown('<div class="sec-head">💬 Input</div>', unsafe_allow_html=True)
    user_input = st.text_area(
        label="",
        placeholder="Type or paste a message here — results update instantly…",
        height=130,
        key="user_input",
        label_visibility="collapsed",
    )

    words, chars, sentences = text_stats(user_input)

    # ── Metrics row ──
    render_metrics(words, chars)

    # ── Prediction pipeline ──
    st.markdown('<div class="sec-head">🔍 Prediction</div>', unsafe_allow_html=True)

    if not user_input.strip():
        st.markdown(
            '<div style="color:#6b7194;font-size:14px;padding:.75rem 0;">'
            '✏️  Start typing above to see live sentiment analysis…'
            '</div>',
            unsafe_allow_html=True,
        )
    else:
        label, conf, raw = predict_sentiment(user_input, vectorizer, model)
        render_prediction(label, conf, st.session_state.show_conf)
        record(user_input, label, conf)

    # ── Stats expander ──
    if show_stats and user_input.strip():
        with st.expander("📐 Text Statistics", expanded=False):
            s1, s2, s3 = st.columns(3)
            s1.metric("Words",     words)
            s2.metric("Characters", chars)
            s3.metric("Sentences", sentences)

    st.divider()

    # ── Bottom section: chart + history ──
    col_a, col_b = st.columns([1, 1.6], gap="large")

    with col_a:
        if show_chart:
            st.markdown('<div class="sec-head">📊 Distribution</div>', unsafe_allow_html=True)
            render_chart()

    with col_b:
        if show_hist:
            st.markdown('<div class="sec-head">🕑 History</div>', unsafe_allow_html=True)
            render_history()


if __name__ == "__main__":
    main()