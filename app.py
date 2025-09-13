import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import numpy as np

st.set_page_config(page_title="Stock Trend Prediction", layout="centered")
st.title("Stock Market Trend Prediction (demo)")
st.markdown("Upload historical OHLC CSV (columns: Date,Open,High,Low,Close,Volume). App creates simple features and predicts Up/Down next day.")

uploaded = st.file_uploader("Upload CSV", type=["csv"])
if uploaded is not None:
    df = pd.read_csv(uploaded, parse_dates=["Date"])
    df = df.sort_values("Date").reset_index(drop=True)
    st.write("Preview:", df.head())
    # Feature engineering
    df["return"] = df["Close"].pct_change()
    df["ma5"] = df["Close"].rolling(5).mean()
    df["ma10"] = df["Close"].rolling(10).mean()
    df = df.dropna().copy()
    df["target"] = (df["Close"].shift(-1) > df["Close"]).astype(int)
    features = ["return","ma5","ma10","Volume"]
    X = df[features]
    y = df["target"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    st.metric("Accuracy", f"{acc:.4f}")
    st.subheader("Classification Report")
    st.text(classification_report(y_test, preds))
    st.subheader("Add latest row to see prediction")
    latest = X.tail(1)
    if st.button("Predict Next Day Trend"):
        p = model.predict(latest)[0]
        st.write("Prediction:", "Up (1)" if p==1 else "Down (0)")
else:
    st.info("No CSV uploaded. A sample synthetic CSV is included in the folder `sample_stock.csv` for testing.")
