from xgboost import XGBClassifier
import pandas as pd
import yfinance as yf
from datetime import datetime
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
from joblib import dump

"""
Loading data
"""
ticker_symbol = "VOO"
START = "2010-01-01"
TODAY = datetime.today().strftime("%Y-%m-%d")
data = yf.download(ticker_symbol, START, TODAY)
data.reset_index(inplace=True)
df = pd.DataFrame(data=data)

"""
Data Preprocessing
"""

df["Tomorrow"] = df["Adj Close"].shift(-1)
df["Target"] = (df["Tomorrow"] > df["Adj Close"]).astype(int)
df["MA_50"] = df["Adj Close"].rolling(50).mean()
df["MA_200"] = df["Adj Close"].rolling(200).mean()

"""
Data Splitting
"""
def splitter(trans_func):
    X_train, X_test, y_train, y_test = train_test_split(
        X, 
        y, 
        test_size=0.2, 
        random_state=101)
    return X_train, X_test, y_train, y_test

"""
Backtesting and Prediction Process
"""
def predict(train, test, predictors, model):
    model.fit(train[predictors], train["Target"])
    predictions = model.predict(test[predictors])
    predictions = pd.Series(predictions, index=test.index, name="Predictions")
    combined = pd.concat([test["Target"], predictions], axis=1)
    return combined

def backtest(data, model, predictors, predictions, start=2500, step=250):
    all_predictions = []
    for i in range(start, data.shape[0], step):
        training_data = data.iloc[0:i].copy()
        testing_data = data.iloc[i:(i+step)].copy()
        predictions = predict(training_data, testing_data, predictors, model)
        all_predictions.append(predictions)
    return pd.concat(all_predictions)

xgb_model = XGBClassifier(eta=0.2, max_depth=6, n_estimators=1000)

predictors = ['Open', 'High', 'Low', 'Adj Close', 'Volume', 'MA_50', 'MA_200']

pipeline = Pipeline(steps=[
    ('data_split', FunctionTransformer(splitter, validate=False)),
    ('model', XGBClassifier(eta=0.2, max_depth=6, n_estimators=800))
])

X = df[predictors]
y = df['Target']

pipeline.fit(X, y)

dump(pipeline, 'xgboost_pipeline.joblib')