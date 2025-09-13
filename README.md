# Stock Market Trend Prediction (demo)

Streamlit app that trains a RandomForest to predict whether the next day's Close will be higher (Up) or lower (Down).

## How to run
1. `pip install -r requirements.txt`
2. `streamlit run app.py`
3. Upload a CSV with columns: Date,Open,High,Low,Close,Volume
4. Or use sample_stock.csv provided.

## Notes
- This is a simple demo. For production, add feature engineering, walk-forward validation, and hyperparameter tuning.
