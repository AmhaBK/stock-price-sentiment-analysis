# module for sentiment analysis

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from textblob import TextBlob
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation

class FinancialNewsEDA:
    def __init__(self, news_data: pd.DataFrame):
        """
        Initialize the EDA object with the dataset.
        """
        self.news_data = news_data.copy()

    def convert_dates(self):
        """
        Convert the 'date' column to datetime with timezone awareness.
        """
        self.news_data['date'] = pd.to_datetime(self.news_data['date'], errors='coerce', utc=True)
        self.news_data['date_only'] = self.news_data['date'].dt.date
        print("Date conversion completed. Here are a few examples:")
        print(self.news_data[['date', 'date_only']].head())

    def headline_length_stats(self):
        """
        Calculate basic statistics of headline lengths and plot distribution.
        """
        self.news_data['headline_length'] = self.news_data['headline'].apply(len)
        print("Headline Length Stats:")
        print(self.news_data['headline_length'].describe())

        plt.figure(figsize=(8,5))
        sns.histplot(self.news_data['headline_length'], bins=30, kde=True, color='skyblue')
        plt.title('Distribution of Headline Lengths')
        plt.xlabel('Number of Characters')
        plt.ylabel('Frequency')
        plt.show()

    def articles_per_publisher(self, top_n=10):
        """
        Count articles per publisher and plot top N.
        """
        publisher_counts = self.news_data['publisher'].value_counts()
        print("\nTop Publishers:")
        print(publisher_counts.head(top_n))

        plt.figure(figsize=(10,6))
        sns.barplot(x=publisher_counts.head(top_n).index, y=publisher_counts.head(top_n).values, palette='viridis')
        plt.xticks(rotation=45)
        plt.title('Top Publishers by Article Count')
        plt.ylabel('Number of Articles')
        plt.show()

    def publication_date_trends(self):
        """
        Plot article publication frequency over time.
        """
        articles_per_day = self.news_data.groupby('date_only').size()

        plt.figure(figsize=(12,6))
        articles_per_day.plot()
        plt.title('Article Publication Frequency Over Time')
        plt.xlabel('Date')
        plt.ylabel('Number of Articles')
        plt.xticks(rotation=45)
        plt.show()

    def analyze_publisher_domains(self, top_n=10):
        """
        Identify unique email domains and visualize top N.
        """
        self.news_data['publisher_domain'] = self.news_data['publisher'].apply(
            lambda x: x.split('@')[-1] if isinstance(x, str) and '@' in x else 'No Email'
        )
        domain_counts = self.news_data['publisher_domain'].value_counts()

        print("\nTop Publisher Domains:")
        print(domain_counts.head(top_n))

        plt.figure(figsize=(10,6))
        sns.barplot(x=domain_counts.head(top_n).index, y=domain_counts.head(top_n).values, palette='magma')
        plt.title('Top 10 Publisher Email Domains')
        plt.ylabel('Number of Articles')
        plt.xlabel('Email Domain')
        plt.xticks(rotation=45)
        plt.show()

    def plot_publication_hour(self):
        """
        Plot the number of articles by hour of publication.
        """
        self.news_data['hour'] = self.news_data['date'].dt.hour
        hourly_counts = self.news_data.groupby('hour').size()

        plt.figure(figsize=(10,6))
        hourly_counts.plot(kind='bar', color='teal')
        plt.title('Articles by Hour of Publication')
        plt.xlabel('Hour of Day')
        plt.ylabel('Number of Articles')
        plt.show()

    def plot_rolling_average(self, window=7):
        """
        Plot rolling average of article publication frequency.
        """
        articles_per_day = self.news_data.groupby('date_only').size()
        rolling_articles = articles_per_day.rolling(window=window).mean()

        plt.figure(figsize=(12,6))
        plt.plot(articles_per_day.index, articles_per_day, alpha=0.5, label='Daily')
        plt.plot(articles_per_day.index, rolling_articles, color='red', label=f'{window}-Day Rolling Avg')
        plt.title('Publication Frequency with Rolling Average')
        plt.xlabel('Date')
        plt.ylabel('Number of Articles')
        plt.legend()
        plt.show()

    def preprocess_text(self):
        """
        Basic text cleaning and preprocessing for topic modeling.
        """
        self.news_data['clean_headline'] = self.news_data['headline'].str.lower().str.replace(r'[^a-zA-Z]', ' ', regex=True)
        return self.news_data

    def perform_lda(self, n_topics=5, n_top_words=10):
        """
        Perform LDA topic modeling and display top words per topic.
        """
        vectorizer = CountVectorizer(stop_words='english', max_df=0.9, min_df=2)
        dtm = vectorizer.fit_transform(self.news_data['clean_headline'])

        lda = LatentDirichletAllocation(n_components=n_topics, random_state=42)
        lda.fit(dtm)

        feature_names = vectorizer.get_feature_names_out()
        for idx, topic in enumerate(lda.components_):
            print(f"\nTopic {idx+1}:")
            print([feature_names[i] for i in topic.argsort()[-n_top_words:]])

        topic_values = lda.transform(dtm)
        self.news_data['dominant_topic'] = topic_values.argmax(axis=1)
        return self.news_data

    def sentiment_analysis(self):
        """
        Perform sentiment analysis on headlines using TextBlob.
        """
        def get_sentiment(text):
            if pd.isna(text):
                return 0  # neutral for empty headlines
            blob = TextBlob(text)
            return blob.sentiment.polarity

        self.news_data['sentiment'] = self.news_data['headline'].apply(get_sentiment)
        print("\nSample of Sentiment Scores:")
        print(self.news_data[['headline', 'sentiment']].head())
        return self.news_data




