from dotenv import load_dotenv
import os
import streamlit as st
import yfinance as yf
import openai
import pandas as pd
import requests
import plotly.express as px
from datetime import datetime
from plotly import graph_objs as go
from xgboost import XGBClassifier
from prophet import Prophet
from plotly.subplots import make_subplots
from prophet.plot import plot_plotly, plot_components_plotly
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, accuracy_score, f1_score

START = '2010-01-01'
TODAY = datetime.today().strftime("%Y-%m-%d")

#Enter your api keys
load_dotenv()
NEWS_API_KEY = os.getenv("NEWS_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

class StockDataLoader:

    @staticmethod
    @st.cache_data # saves whatever that is downloaded so we dont need to download again
    def load_data(ticker):
        data = yf.download(ticker, START, TODAY)
        data.reset_index(inplace=True)
        return data

class StockPlotter:

    @staticmethod
    def plot_raw_data(data):
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=data['Date'], y=data['Open'], name='stock_open'))
        fig.add_trace(go.Scatter(x=data['Date'], y=data['Close'], name='stock_close'))
        fig.layout.update(title_text="Time Series Data", xaxis_rangeslider_visible=True, width=900, height=800)
        st.plotly_chart(fig)

    @staticmethod
    def plot_MA_Vol(data, ticker_df):
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True)
        fig.add_trace(go.Candlestick(x=data['Date'], open=data['Open'], high=data['High'], low=data['Low'], close=data['Close']))
        ticker_df['MA50'] = ticker_df['Close'].rolling(50).mean()
        ticker_df['MA200'] = ticker_df['Close'].rolling(200).mean()
        fig.add_trace(go.Scatter(x=ticker_df['Date'], y=ticker_df['MA50'], name='MA 50'))
        fig.add_trace(go.Scatter(x=ticker_df['Date'], y=ticker_df['MA200'], name="MA 200"))
        fig.add_trace(go.Bar(x=ticker_df['Date'], y=ticker_df['Volume'], name='Volume'),row=2, col=1)
        fig.layout.update(title_text="Moving Avg [50 days vs 200 days] Analysis", xaxis_rangeslider_visible=True, width=900, height=1600)
        st.plotly_chart(fig)

class StockStatistics:

    @staticmethod
    def stock_description(ticker_df):
        ticker_desc = ticker_df.drop(labels=['Date', 'MA50', 'MA200'], axis=1)
        st.write(ticker_desc.describe())

    @staticmethod
    def daily_return_average(ticker_df):
        ticker_df['Daily Average Return'] = ticker_df['Adj Close'].pct_change() * 100
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=ticker_df['Date'], y=ticker_df['Daily Average Return']))
        fig.layout.update(title_text=f"Average Daily Return Rate of {selected_stocks}", xaxis_rangeslider_visible=True)
        st.plotly_chart(fig)
        st.write("**Average Daily Return Rate Statistics**")
        st.write(ticker_df['Daily Average Return'].describe())

    @staticmethod
    def monthly_return_average(ticker_df):
        ticker_df['Monthly Average Return'] = ticker_df['Adj Close'].pct_change(30) * 100
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=ticker_df['Date'], y=ticker_df['Monthly Average Return']))
        fig.layout.update(title_text=f"Monthly Average Return Rate of {selected_stocks}", xaxis_rangeslider_visible=True)
        st.plotly_chart(fig)
        st.write("**Monthly Average Return Rate Statistics**")
        st.write(ticker_df['Monthly Average Return'].describe())

    @staticmethod
    def corr_report_daily(ticker_df, stocks):
        for stock in stocks:
            if selected_stocks != stock:
                compare_df = pd.DataFrame(data=StockDataLoader.load_data(stock))
                compare_df['Daily Average Return'] = compare_df['Adj Close'].pct_change() * 100
                unregistered_values = len(ticker_df['Daily Average Return']) - len(compare_df['Daily Average Return'])
                if unregistered_values < 0:
                    compare_df = compare_df.drop(compare_df.index[:abs(unregistered_values)], axis=0)
                elif unregistered_values > 0:
                    ticker_df = ticker_df.drop(ticker_df.index[:unregistered_values], axis=0)
                corr = ticker_df['Daily Average Return'].corr(compare_df['Daily Average Return'])
                st.write(f"{selected_stocks} & {stock}: {corr}")

    @staticmethod
    def corr_report_monthly(ticker_df, stocks):
        for stock in stocks:
            if selected_stocks != stock:
                compare_df = pd.DataFrame(data=StockDataLoader.load_data(stock))
                compare_df['Monthly Average Return'] = compare_df['Adj Close'].pct_change(30) * 100
                unregistered_values = len(ticker_df['Monthly Average Return']) - len(compare_df['Monthly Average Return'])
                if unregistered_values < 0:
                    compare_df = compare_df.drop(compare_df.index[:abs(unregistered_values)], axis=0)
                elif unregistered_values > 0:
                    ticker_df = ticker_df.drop(ticker_df.index[:unregistered_values], axis=0)
                corr = ticker_df['Monthly Average Return'].corr(compare_df['Monthly Average Return'])
                st.write(f"{selected_stocks} & {stock}: {corr}")

class NewsFetcher:

    @staticmethod
    def fetch_news(company_name, news_api_key, num_of_news_to_retrieve):
        news_lst_copy = []
        news_dict = {}

        news_endpoint_url = "https://newsapi.org/v2/everything"
        news_param = {
            'apiKey': news_api_key,
            'q': company_name,
            'excludeDomains': "yahoo.com",
            'language': "en",
            'publishedAt': f"{str(datetime.now().date())}",
            'sortBy': "publishedAt, relevancy"
        }

        news_response = requests.get(news_endpoint_url, params=news_param)
        news_response.raise_for_status()
        news_retrieved = 0

        for content in news_response.json()['articles']:
            if news_retrieved < num_of_news_to_retrieve:
                news_dict["Headline"] = content['title']
                news_dict["Description"] = content['description']
                news_dict["Source"] = content['url']
                article_published_time = content['publishedAt'].split('T')
                article_published_date, article_published_time = article_published_time[0], article_published_time[1]
                news_dict["Published Date"] = article_published_date
                news_dict["Published Time"] = article_published_time
                news_lst_copy.append(news_dict.copy())
                news_retrieved+=1
            else:
                break
        
        return news_lst_copy

    @staticmethod
    def display_news(news_lst):
        count = 0
        for news in news_lst:
            count += 1
            st.write(count)
            for category, content in news.items():
                st.write(f"\n{category}: {content}")

class StockForecasting:

    @staticmethod
    def fbprophet_forecast(ticker_df, periods):
        df_train = ticker_df[['Date', 'Adj Close']]
        df_train = df_train.rename(columns={"Date": "ds", "Adj Close": "y"})
        model = Prophet()
        model.fit(df_train)
        future = model.make_future_dataframe(periods=periods)
        forecast = model.predict(future)
        fig_forecast = plot_plotly(model, forecast, xlabel="Timeline", ylabel="Price")
        fig_forecast_components = plot_components_plotly(model, forecast)
        st.plotly_chart(fig_forecast)
        st.plotly_chart(fig_forecast_components)

class StockPrediction:

    def __init__(self, ticker_df):
        self.modified_df = ticker_df
        self.predictors = ['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume', 'MA_50', 'MA_200']
        self.xgb_model = XGBClassifier(eta=0.2, max_depth=6, n_estimators=1000)

    def data_preprocessing(self, ticker_df):
        self.modified_df = ticker_df
        self.modified_df["Tomorrow"] = self.modified_df["Adj Close"].shift(-1)
        self.modified_df["Target"] = (self.modified_df["Tomorrow"] > self.modified_df["Adj Close"]).astype(int)
        self.modified_df["MA_50"] = self.modified_df["Adj Close"].rolling(50).mean()
        self.modified_df["MA_200"] = self.modified_df["Adj Close"].rolling(200).mean()
        self.X = self.modified_df[self.predictors]
        self.y = self.modified_df["Target"]
        X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, test_size=0.2, random_state=101)
        return X_train, X_test, y_train, y_test
    
    def model_prediction(self, X_train, X_test, y_train):
        self.xgb_model.fit(X_train, y_train)
        model_prediction = self.xgb_model.predict(X_test)
        return model_prediction

    def backtest_predictions(self, train, test):
        self.xgb_model.fit(train[self.predictors], train["Target"])
        predictions = self.xgb_model.predict(test[self.predictors])
        predictions = pd.Series(predictions, index=test.index, name="Predictions")
        combined = pd.concat([test["Target"], predictions], axis=1)
        return combined

    # assign backtest function to a variable to use precision score
    def run_backtest(self, start=2500, step=250) -> list:
        all_predictions = []
        for i in range(start, self.modified_df.shape[0], step):
            training_data = self.modified_df.iloc[0:i].copy()
            testing_data = self.modified_df.iloc[i:(i+step)].copy()
            predictions = self.backtest_predictions(training_data, testing_data)
            all_predictions.append(predictions)
        return pd.concat(all_predictions)
    
    def plot_backtest(self, backtest_prediction_df):
        concatenated_df = pd.concat([self.modified_df['Date'], backtest_prediction_df], axis=1)
        fig = px.line(concatenated_df, x='Date', y=['Target', 'Predictions'], color='variable')
        fig.layout.update(title_text="Backtest Results", xaxis_rangeslider_visible=True)
        st.plotly_chart(fig)

class SentimentAnalysisGPT:

    @staticmethod
    def gpt_market_sentiment_analysis(selected_stocks, news_snippet, openai_api_key):
        aggregated_news_snippet = ""
        for news in news_snippet:
            for category, content in news.items():
                if category == "Description":
                    aggregated_news_snippet += content + "\n"

        openai.api_key = openai_api_key
        response = openai.ChatCompletion.create(
            model= "gpt-3.5-turbo-1106",
            messages=[
                {"role": "system", "content": "You are an expert in analyzing stock market sentiments based on news snippets and news descriptions who will lead your company to a conclusion of the sentiment in the market."},
                {"role": "user", "content": f"Analyze and tell me the market conditions of {selected_stocks}, based on the news snippet: \"{aggregated_news_snippet}\""}
                ]
        )

        st.write(response['choices'][0]['message']['content'])
    
class StockAnalyzer:

    def __init__(self, selected_stocks):
        self.selected_stocks = selected_stocks
        self.data_loader = StockDataLoader()
        self.plotter = StockPlotter()
        self.statistics = StockStatistics()
        self.forecasting = StockForecasting()
        self.news_fetcher = NewsFetcher()
        self.sentiment_analysis = SentimentAnalysisGPT()

    def main(self):
        data = self.data_loader.load_data(self.selected_stocks)
        ticker_df = pd.DataFrame(data=data)
        self.plotter.plot_raw_data(data)
        self.plotter.plot_MA_Vol(data, ticker_df)

        stocks_lst = ["AAPL", "GOOG", "MSFT", "TSLA", "VOO", "SCCO"]

        st.write(f"**{self.selected_stocks} Statistics Summary**")
        self.statistics.stock_description(ticker_df)
        self.statistics.daily_return_average(ticker_df)
        st.write("**Correlation Between Stocks (Daily Returns)**")
        self.statistics.corr_report_daily(ticker_df, stocks_lst)
        self.statistics.monthly_return_average(ticker_df)
        st.write("**Correlation Between Stocks (Monthly Returns)**")
        self.statistics.corr_report_monthly(ticker_df, stocks_lst)

        n_years = st.slider("Years for prediction:", 1, 5)
        period = n_years * 365
        self.forecasting.fbprophet_forecast(ticker_df, period)

        """
        ## Stock Direction Prediction
        """
        self.stock_prediction = StockPrediction(ticker_df)
        X_train, X_test, y_train, y_test = self.stock_prediction.data_preprocessing(ticker_df)
        st.write("\n")
        st.write("**Model Prediction Metrics**")
        model_prediction = self.stock_prediction.model_prediction(X_train, X_test, y_train)
        st.write(f"The Accuracy Score of Predictions Made from XGBoost is:  \n{round(accuracy_score(y_test, model_prediction) * 100, 2)}%")
        st.write(f"The F1-Score of Predictions Made from XGBoost is:  \n{round(f1_score(y_test, model_prediction) * 100 ,2)}%")
        st.write(f"The Precision Score of Predictions Made from XGBoost is:  \n{round(precision_score(y_test, model_prediction) * 100, 2)}%")
        st.write("\n")
        st.write("**Market Price Nature**")
        backtest_prediction = self.stock_prediction.run_backtest()
        st.write(f"Probability of {selected_stocks} price increase from {START} until today:  \n{round((backtest_prediction['Target'].sum() / backtest_prediction.shape[0]) * 100, 2)}%")
        st.write(f"Probability of {selected_stocks} price decrease from {START} until today:  \n{round((1 - (backtest_prediction['Target'].sum() / backtest_prediction.shape[0]))* 100, 2)}%")
        st.write("\n")
        self.stock_prediction.plot_backtest(backtest_prediction)
        st.write(f"The Precision Score of the backtest is: {round(precision_score(backtest_prediction['Target'], backtest_prediction['Predictions']) * 100, 2)}%")
        st.write("\n")
        if backtest_prediction["Predictions"].iloc[-1] == 0:
            direction = "LOWER"
        else:
            direction = "HIGHER"
        st.write(f"The next market trading day for {selected_stocks}, stock price will close {direction}") 

        """
        ## Retrieved News
        """
        news_lst = self.news_fetcher.fetch_news(self.selected_stocks, NEWS_API_KEY, 10)
        self.news_fetcher.display_news(news_lst)

        """
        ## Market Sentiment Analysis Based on Retrieved News Powered By GPT
        """
        try: 
            self.sentiment_analysis.gpt_market_sentiment_analysis(self.selected_stocks, news_lst, OPENAI_API_KEY)
        except:
            st.write("There is an error loading GPT's reponse.")

if __name__ == '__main__':
    st.title("Stock Prediction Web App")
    selected_stocks = st.selectbox(label="Select ticker", options=("AAPL", "GOOG", "MSFT", "TSLA", "VOO", "SCCO"))
    analyzer = StockAnalyzer(selected_stocks)
    analyzer.main()