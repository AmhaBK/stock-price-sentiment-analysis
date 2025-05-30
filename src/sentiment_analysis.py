import pandas as pd
from textblob import TextBlob

def get_sentiment(text):
    return TextBlob(text).sentiment.polarity

def apply_sentiment(df, text_column='headline'):
    df['sentiment'] = df[text_column].apply(get_sentiment)
    return df