"""
AI-Powered Cybersecurity: Using AI to Detect Cyber Attacks  -  Streamlit UI

Run:   streamlit run app.py
Needs in the same folder: attack_detector.pkl  (and sample_traffic.csv for the demo)
"""
import time

import joblib
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="AI Cyber Attack Detector", page_icon="🛡️", layout="wide")


@st.cache_resource
def load_model():
    model, cols = joblib.load("attack_detector.pkl")
    return model, cols


def prepare(df, cols):
    """Clean uploaded data and return (features, error message)."""
    df = df.copy()
    df.columns = df.columns.str.strip()
    missing = [c for c in cols if c not in df.columns]
    if missing:
        return None, f"Missing required columns: {missing}"
    X = df[cols].replace([np.inf, -np.inf], np.nan)
    keep = X.notna().all(axis=1)
    return X[keep], None


# ------------------------------------------------------------------ header
st.title("🛡️ AI-Powered Cybersecurity: Detecting Cyber Attacks")
st.caption("Random Forest classifier trained on CIC-IDS2017 network traffic. "
           "Classifies each traffic flow as Normal or Malicious and raises an alert.")

try:
    model, cols = load_model()
except Exception as e:
    st.error("Could not load attack_detector.pkl. Keep it in the same folder as app.py.")
    st.code(str(e))
    st.stop()

with st.sidebar:
    st.header("Model information")
    st.write("**Algorithm:** Random Forest (100 trees)")
    st.write("**Dataset:** CIC-IDS2017 (100,000-record sample)")
    st.write(f"**Features used:** {len(cols)}")
    st.markdown("**Test results (20,000 records)**")
    st.write("Accuracy: 99.66%")
    st.write("Precision: 0.9919")
    st.write("Recall: 0.9878")
    st.write("F1-score: 0.9898")
    with st.expander("Selected features"):
        for c in cols:
            st.write("- " + c)

tab1, tab2 = st.tabs(["📂 Upload and Detect", "📡 Simulate Live Traffic"])

# ------------------------------------------------------------------ tab 1
with tab1:
    st.subheader("Upload network traffic (CSV)")
    st.write("Use the provided `sample_traffic.csv`, or any CSV with CIC-IDS2017 flow features.")
    up = st.file_uploader("Choose a CSV file", type="csv")
    if up is not None:
        raw = pd.read_csv(up)
        X, err = prepare(raw, cols)
        if err:
            st.error(err)
        else:
            pred = model.predict(X[cols])
            proba = model.predict_proba(X[cols])[:, 1]
            out = X.copy()
            out["Prediction"] = np.where(pred == 1, "Malicious", "Normal")
            out["Attack probability"] = proba.round(4)

            n_total, n_mal = len(out), int((pred == 1).sum())
            c1, c2, c3 = st.columns(3)
            c1.metric("Flows analysed", n_total)
            c2.metric("Normal", n_total - n_mal)
            c3.metric("Malicious", n_mal)

            if n_mal > 0:
                st.error(f"🚨 ALERT: {n_mal} malicious flow(s) detected!")
            else:
                st.success("✅ All traffic is normal.")

            st.bar_chart(out["Prediction"].value_counts())

            show = st.radio("Show", ["Malicious only", "All records"], horizontal=True)
            view = out[out["Prediction"] == "Malicious"] if show == "Malicious only" else out
            st.dataframe(view.head(500), use_container_width=True)
            st.download_button("Download results (CSV)", out.to_csv(index=False),
                               "detection_results.csv", "text/csv")

# ------------------------------------------------------------------ tab 2
with tab2:
    st.subheader("Simulated live network traffic")
    st.write("Flows from the sample file are fed one by one, as if captured from a network.")
    n_flows = st.slider("Number of flows", 10, 50, 20)
    delay = st.slider("Delay between flows (seconds)", 0.1, 1.0, 0.3)
    if st.button("▶ Start simulation"):
        try:
            sample = pd.read_csv("sample_traffic.csv")
        except Exception:
            st.error("sample_traffic.csv not found in the app folder.")
            st.stop()
        X, err = prepare(sample, cols)
        if err:
            st.error(err)
            st.stop()
        X = X.sample(min(n_flows, len(X)), random_state=int(time.time()) % 1000)

        log, alerts = [], 0
        box = st.empty()
        bar = st.progress(0)
        for n, (i, row) in enumerate(X.iterrows(), 1):
            p = model.predict(row.to_frame().T[cols])[0]
            if p == 1:
                alerts += 1
                log.append(f"🚨 Flow {n:02d}: ALERT - Malicious traffic detected")
            else:
                log.append(f"✅ Flow {n:02d}: Normal traffic")
            box.markdown("\n\n".join(log[-12:]))
            bar.progress(n / len(X))
            time.sleep(delay)
        if alerts:
            st.error(f"Simulation finished: {len(X)} flows processed, {alerts} alert(s) raised.")
        else:
            st.success(f"Simulation finished: {len(X)} flows processed, no alerts.")

st.divider()
st.caption("Prototype: offline, dataset-based detection. Live packet capture is future scope.")
