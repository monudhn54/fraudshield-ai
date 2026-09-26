import streamlit as st
import numpy as np
import pandas as pd
import joblib

# ----------------------------
# CONFIG
# ----------------------------
st.set_page_config(page_title="Fraud Detector 🚀", page_icon="🛡️", layout="centered")

st.title("🛡️ FraudSheild AI")
st.caption("Detect suspicious transactions in real-time ⚡")

# ----------------------------
# LOAD MODEL
# ----------------------------
@st.cache_resource
def load_artifact():
    try:
        return joblib.load("best_fraud_model_tuned.pkl")
    except:
        return None

artifact = load_artifact()

if artifact is None:
    st.error("⚠️ Upload your trained model (.pkl)")
    uploaded = st.file_uploader("Upload model", type=["pkl"])
    if uploaded:
        import io
        artifact = joblib.load(io.BytesIO(uploaded.read()))
    else:
        st.stop()

model = artifact["model"]
features = artifact["features"]
threshold_default = artifact.get("threshold", 0.5)
scaler = artifact.get("scaler", None)
num_cols = artifact.get("num_cols", features)
p99_log_amount = artifact.get("p99_log_amount", 13.35)

# ----------------------------
# THRESHOLD CONTROL
# ----------------------------
st.sidebar.header("⚙️ Controls")
threshold = st.sidebar.slider("Fraud Sensitivity", 0.0, 1.0, float(threshold_default))

st.sidebar.info("Lower threshold = catch more fraud 🔥 (but more false alarms)")

# ----------------------------
# INPUT FORM
# ----------------------------
TRANSACTION_TYPES = {
    "CASH_IN": 0,
    "CASH_OUT": 1,
    "DEBIT": 2,
    "PAYMENT": 3,
    "TRANSFER": 4,
}

with st.form("input_form"):
    st.subheader("💳 Enter Transaction Details")

    col1, col2 = st.columns(2)

    with col1:
        step = st.number_input("Step (Hour)", 1, 744, 1)
        txn_type = st.selectbox("Transaction Type", list(TRANSACTION_TYPES.keys()))
        amount = st.number_input("Amount", 0.0, value=1000.0)
        old_org = st.number_input("Sender Old Balance", 0.0, value=5000.0)

    with col2:
        new_org = st.number_input("Sender New Balance", 0.0, value=4000.0)
        old_dest = st.number_input("Receiver Old Balance", 0.0, value=0.0)
        new_dest = st.number_input("Receiver New Balance", 0.0, value=1000.0)

    submit = st.form_submit_button("🚀 Predict")

# ----------------------------
# FEATURE ENGINEERING
# ----------------------------
def create_features():
    hour = step % 24
    log_amount = np.log1p(amount)
    is_high_amount = int(log_amount > p99_log_amount)
    is_night = int(hour in [0,1,2,3,4,5,22,23])

    balance_diff_orig = old_org - new_org
    balance_diff_dest = old_dest - new_dest

    return {
        "step": step,
        "amount": amount,
        "isFlaggedFraud": 0,
        "log_amount": log_amount,
        "is_high_amount": is_high_amount,
        "hour": hour,
        "is_night": is_night,
        "balance_diff_orig": balance_diff_orig,
        "balance_diff_dest": balance_diff_dest,
        "type_encoded": TRANSACTION_TYPES[txn_type],
    }

# ----------------------------
# PREDICTION
# ----------------------------
if submit:

    # Basic validation
    if new_org > old_org:
        st.warning("⚠️ Sender balance increased after transaction (suspicious input)")

    row = create_features()
    X = pd.DataFrame([row])[features]

    # Scaling
    if scaler:
        cols = [c for c in num_cols if c in X.columns]
        X[cols] = scaler.transform(X[cols])

    prob = model.predict_proba(X)[0,1]
    pred = int(prob >= threshold)

    st.divider()

    # ----------------------------
    # RESULT UI
    # ----------------------------
    st.subheader("📊 Result")

    st.metric("Fraud Probability", f"{prob:.2%}")

    if pred:
        st.error("🚨 FRAUD DETECTED")
    else:
        st.success("✅ Looks Safe")

    st.progress(int(prob*100))

    # ----------------------------
    # GEN-Z EXPLANATION
    # ----------------------------
    st.subheader("🧠 Why this prediction?")

    reasons = []

    if row["is_high_amount"]:
        reasons.append("💰 Unusually high transaction")

    if row["is_night"]:
        reasons.append("🌙 Happens at night (risky time)")

    if abs(row["balance_diff_orig"] - amount) > 1:
        reasons.append("⚠️ Sender balance mismatch")

    if abs(row["balance_diff_dest"] + amount) > 1:
        reasons.append("⚠️ Receiver balance mismatch")

    if len(reasons) == 0:
        reasons.append("Normal transaction pattern")

    for r in reasons:
        st.write("•", r)

    # ----------------------------
    # RAW DATA
    # ----------------------------
    with st.expander("🔍 Feature Values"):
        st.dataframe(pd.DataFrame([row]))

# ----------------------------
# FOOTER
# ----------------------------
st.divider()
st.caption("⚡ Built for learning ML deployment | Not for real banking use")
