import streamlit as st
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk
import pandas as pd

# Ensure required NLTK data exists
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')
try:
    nltk.data.find('sentiment/vader_lexicon')
except LookupError:
    nltk.download('vader_lexicon')

st.set_page_config(page_title="Sentiment Analysis Dashboard", layout="wide")
st.title("Sentiment Analysis Dashboard")
st.markdown("Enter text or paste social media posts (one per line). Uses NLTK VADER for polarity scoring.")

text = st.text_area("Input text (one post per line)", height=200)
if st.button("Analyze"):
    if not text.strip():
        st.warning("Please enter some text.")
    else:
        posts = [p.strip() for p in text.splitlines() if p.strip()]
        sia = SentimentIntensityAnalyzer()
        results = []
        for p in posts:
            score = sia.polarity_scores(p)
            results.append({"text": p, "neg": score["neg"], "neu": score["neu"], "pos": score["pos"], "compound": score["compound"]})
        df = pd.DataFrame(results)
        st.subheader("Results")
        st.dataframe(df)
        st.metric("Average compound", round(df["compound"].mean(),4))
        st.bar_chart(df[["neg","neu","pos"]])
