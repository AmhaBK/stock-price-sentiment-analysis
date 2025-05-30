import pandas as pd

def load_data(news_path, stock_path):
    news_df = pd.read_csv(news_path, parse_dates=['date'])
    stock_df = pd.read_csv(stock_path, parse_dates=['date'])
    return news_df, stock_df

def align_dates(news_df, stock_df):
    news_df['date'] = news_df['date'].dt.date
    stock_df['date'] = stock_df['date'].dt.date
    return news_df, stock_df

def save_data(df, path):
    df.to_csv(path, index=False)
