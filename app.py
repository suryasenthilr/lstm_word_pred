import streamlit as st
import numpy as np
import pickle
import time
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

st.set_page_config(
    page_title="NeuralScribe · Next Word",
    page_icon="🧠",
    layout="centered",
)

st.markdown("""
<style>
html, body, [class*="css"] {
    font-family: 'Courier New', Courier, monospace;
    background-color: #020c14;
    color: #cff5ff;
}
.stApp { background: #020c14; }

.grid-bg {
    position: fixed;
    inset: 0;
    background-image:
        linear-gradient(rgba(0,255,255,0.03) 1px, transparent 1px),
        linear-gradient(90deg, rgba(0,255,255,0.03) 1px, transparent 1px);
    background-size: 40px 40px;
    pointer-events: none;
    z-index: 0;
}

.orb {
    position: fixed;
    top: -250px; left: 50%;
    transform: translateX(-50%);
    width: 800px; height: 800px;
    border-radius: 50%;
    background: radial-gradient(circle at center,
        rgba(0,220,255,0.12) 0%,
        rgba(0,80,255,0.08) 45%,
        transparent 70%);
    pointer-events: none;
    z-index: 0;
    animation: pulse 5s ease-in-out infinite;
}
@keyframes pulse {
    0%,100% { transform: translateX(-50%) scale(1); opacity:0.6; }
    50%      { transform: translateX(-50%) scale(1.15); opacity:1; }
}

.header-wrap { text-align: center; padding: 2.5rem 0 1.2rem; position: relative; z-index:1; }
.header-title {
    font-family: 'Courier New', Courier, monospace;
    font-size: 5rem;
    font-weight: 800;
    letter-spacing: 0.05em;
    background: linear-gradient(135deg, #00eaff 0%, #0080ff 50%, #00eaff 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin: 0;
    line-height: 1.1;
    filter: drop-shadow(0 0 20px rgba(0,220,255,0.5));
}
.header-sub {
    font-size: 0.78rem;
    letter-spacing: 0.28em;
    text-transform: uppercase;
    color: rgba(0,220,255,0.4);
    margin-top: 0.5rem;
}

.stats-row {
    display: flex; gap: 0.8rem; flex-wrap: wrap;
    margin-top: 1.2rem; justify-content: center;
}
.stat-chip {
    background: rgba(0,220,255,0.05);
    border: 1px solid rgba(0,220,255,0.2);
    border-radius: 4px;
    padding: 0.25rem 0.9rem;
    font-size: 0.72rem;
    color: rgba(0,220,255,0.5);
    letter-spacing: 0.1em;
}
.stat-chip span { color: #00eaff; font-weight: 600; }

/* Model toggle */
.model-toggle-wrap {
    display: flex; gap: 0; justify-content: center;
    margin: 1.2rem auto 0;
    border: 1px solid rgba(0,220,255,0.3);
    border-radius: 6px;
    overflow: hidden;
    width: fit-content;
}
.model-badge {
    padding: 0.4rem 1.4rem;
    font-size: 0.75rem;
    letter-spacing: 0.15em;
    text-transform: uppercase;
}
.model-badge-active {
    background: rgba(0,220,255,0.15);
    color: #00eaff;
    border-right: 1px solid rgba(0,220,255,0.3);
}
.model-badge-inactive {
    background: transparent;
    color: rgba(0,220,255,0.3);
}

textarea {
    background: rgba(0,20,40,0.8) !important;
    border: 1px solid rgba(0,220,255,0.3) !important;
    border-radius: 6px !important;
    color: #00eaff !important;
    font-family: 'Courier New', Courier, monospace !important;
    font-size: 1rem !important;
    resize: vertical !important;
    transition: border-color 0.3s, box-shadow 0.3s !important;
    caret-color: #00eaff !important;
}
textarea:focus {
    border-color: #00eaff !important;
    box-shadow: 0 0 0 2px rgba(0,220,255,0.2), 0 0 20px rgba(0,220,255,0.1) !important;
}

.stSlider > div > div { color: #00eaff !important; }

.stButton > button {
    width: 100%;
    background: transparent !important;
    color: #00eaff !important;
    border: 1px solid #00eaff !important;
    border-radius: 6px !important;
    font-family: 'Courier New', Courier, monospace !important;
    font-size: 1rem !important;
    font-weight: 700 !important;
    letter-spacing: 0.2em !important;
    padding: 0.75rem 0 !important;
    cursor: pointer;
    transition: all 0.2s ease !important;
    position: relative; z-index:1;
    text-transform: uppercase !important;
}
.stButton > button:hover {
    background: rgba(0,220,255,0.08) !important;
    box-shadow: 0 0 20px rgba(0,220,255,0.3), inset 0 0 20px rgba(0,220,255,0.05) !important;
    transform: translateY(-1px) !important;
}
.stButton > button:active { transform: translateY(0) !important; }

.result-box {
    margin-top: 1.6rem;
    padding: 1.6rem 2rem;
    background: rgba(0,20,50,0.6);
    border: 1px solid rgba(0,220,255,0.25);
    border-radius: 8px;
    text-align: center;
    position: relative; z-index:1;
    animation: fadeUp 0.4s ease both;
    box-shadow: 0 0 30px rgba(0,220,255,0.05);
}
@keyframes fadeUp {
    from { opacity:0; transform: translateY(14px); }
    to   { opacity:1; transform: translateY(0); }
}
.result-label {
    font-size: 0.62rem;
    letter-spacing: 0.3em;
    text-transform: uppercase;
    color: rgba(0,220,255,0.4);
    margin-bottom: 0.6rem;
}
.result-tokens {
    display: flex;
    flex-wrap: wrap;
    gap: 0.5rem;
    justify-content: center;
    margin-top: 0.8rem;
}
.token {
    background: rgba(0,220,255,0.06);
    border: 1px solid rgba(0,220,255,0.5);
    border-radius: 4px;
    padding: 0.3rem 0.85rem;
    font-size: 1.5rem;
    font-family: 'Courier New', Courier, monospace;
    font-weight: 700;
    color: #00eaff;
    letter-spacing: 0.08em;
    animation: popIn 0.35s cubic-bezier(.34,1.56,.64,1) both;
    text-shadow: 0 0 10px rgba(0,220,255,0.6);
    box-shadow: 0 0 12px rgba(0,220,255,0.15);
}
@keyframes popIn {
    from { opacity:0; transform: scale(0.6); }
    to   { opacity:1; transform: scale(1); }
}

.prob-bar-wrap { margin-top: 1rem; text-align: left; }
.prob-label { font-size: 0.72rem; color: rgba(0,220,255,0.5); margin-bottom: 0.3rem; letter-spacing:0.1em; }
.prob-bar-bg {
    background: rgba(0,220,255,0.06);
    border-radius: 2px; height: 6px; overflow: hidden;
}
.prob-bar-fill {
    height: 100%; border-radius: 2px;
    background: linear-gradient(90deg, #0050ff, #00eaff);
    box-shadow: 0 0 8px rgba(0,220,255,0.5);
}

.full-text-box {
    margin-top: 1.4rem;
    padding: 1rem 1.4rem;
    background: rgba(0,10,30,0.6);
    border-left: 2px solid #00eaff;
    border-radius: 4px;
    font-size: 0.95rem;
    color: rgba(0,220,255,0.8);
    word-break: break-word;
    text-align: left;
    box-shadow: inset 0 0 20px rgba(0,220,255,0.03);
}

.footer {
    text-align: center; font-size: 0.65rem;
    color: rgba(0,220,255,0.2); margin-top: 2.5rem;
    letter-spacing: 0.2em; text-transform: uppercase;
    position: relative; z-index:1;
}
</style>
<div class="grid-bg"></div>
<div class="orb"></div>
""", unsafe_allow_html=True)


@st.cache_resource(show_spinner=False)
def load_lstm():
    model     = load_model("lstm_model.h5")
    tokenizer = pickle.load(open("tokenizer.pkl", "rb"))
    max_len   = pickle.load(open("max_len.pkl",   "rb"))
    return model, tokenizer, max_len

@st.cache_resource(show_spinner=False)
def load_gru():
    model     = load_model("gru_model.h5")
    tokenizer = pickle.load(open("tokenizer.pkl", "rb"))
    max_len   = pickle.load(open("max_len.pkl",   "rb"))
    return model, tokenizer, max_len


def predict_next_words(model, tokenizer, max_len, seed_text, n_words):
    result = []
    current = seed_text
    for _ in range(n_words):
        seq = tokenizer.texts_to_sequences([current])[0]
        seq = pad_sequences([seq], maxlen=max_len - 1, padding="pre")
        probs = model.predict(seq, verbose=0)[0]
        pred_idx = np.argmax(probs)
        max_prob = float(probs[pred_idx])
        word = ""
        for w, idx in tokenizer.word_index.items():
            if idx == pred_idx:
                word = w
                break
        result.append((word, max_prob))
        current += " " + word
    return result


# Header
st.markdown("""
<div class="header-wrap">
  <p class="header-title">NeuralScribe</p>
  <p class="header-sub">Next Word Prediction Engine</p>
</div>
""", unsafe_allow_html=True)

# Model selector
model_choice = st.radio(
    "Select Model",
    ["LSTM", "GRU"],
    horizontal=True,
    label_visibility="collapsed",
)

# Load selected model
with st.spinner(f"Loading {model_choice} model..."):
    try:
        if model_choice == "LSTM":
            model, tokenizer, max_len = load_lstm()
        else:
            model, tokenizer, max_len = load_gru()
        vocab_size = len(tokenizer.word_index) + 1
        loaded_ok = True
    except Exception as e:
        st.error(f"Could not load model files: {e}")
        loaded_ok = False

if loaded_ok:
    st.markdown(f"""
    <div class="stats-row">
      <div class="stat-chip">Model <span>{model_choice}</span></div>
      <div class="stat-chip">Vocab size <span>{vocab_size:,}</span></div>
      <div class="stat-chip">Max sequence len <span>{max_len}</span></div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<div style='height:1.5rem'></div>", unsafe_allow_html=True)

    seed_text = st.text_area(
        "Seed text",
        placeholder="Type your opening phrase here...",
        height=110,
        label_visibility="collapsed",
        key="seed",
    )

    n_words = st.slider(
        "Words to predict",
        min_value=1, max_value=20, value=1,
        help="How many words should the model generate?",
    )

    predict_btn = st.button("[ PREDICT ]", use_container_width=True)

    if predict_btn:
        if not seed_text.strip():
            st.warning("Please enter some seed text first.")
        else:
            with st.spinner(f"Running {model_choice}..."):
                time.sleep(0.3)
                predictions = predict_next_words(model, tokenizer, max_len, seed_text.strip(), n_words)

            words_html = "".join(
                f'<div class="token" style="animation-delay:{i*0.07}s">{w}</div>'
                for i, (w, _) in enumerate(predictions)
            )
            label = "word" if n_words == 1 else "words"
            st.markdown(
                f'<div class="result-box">'
                f'<div class="result-label">// {model_choice} · Predicted {label}</div>'
                f'<div class="result-tokens">{words_html}</div>'
                f'</div>',
                unsafe_allow_html=True
            )

            for w, p in predictions:
                pct = round(p * 100, 1)
                st.markdown(
                    f'<div class="prob-bar-wrap">'
                    f'<div class="prob-label">"{w}" - confidence {pct}%</div>'
                    f'<div class="prob-bar-bg">'
                    f'<div class="prob-bar-fill" style="width:{pct}%"></div>'
                    f'</div></div>',
                    unsafe_allow_html=True
                )

            full_output = seed_text.strip() + " " + " ".join(w for w, _ in predictions)
            st.markdown(
                f'<div class="full-text-box">'
                f'<span style="color:rgba(0,220,255,0.35);font-size:0.7rem;letter-spacing:0.2em;">// FULL OUTPUT</span>'
                f'<br>{full_output}'
                f'</div>',
                unsafe_allow_html=True
            )

    st.markdown('<div class="footer">Powered by TensorFlow · Keras · Streamlit</div>', unsafe_allow_html=True)