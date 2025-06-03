# src/stock_analysis.py

import pandas as pd
import matplotlib.pyplot as plt
import talib
import mplfinance as mpf

class StockAnalysis:
    def __init__(self, ticker, df = None, file_path = None):
        """
        Initialize the StockAnalysis object.
        """
        self.ticker = ticker

        if df is not None:
            # Use the provided DataFrame directly
            self.df = df.copy()
        elif file_path is not None:
            # Load data from file
            self.df = pd.read_csv(file_path, parse_dates=['Date'])
        else:
            raise ValueError("Must provide either a file_path or a DataFrame.")
        self.df.columns = self.df.columns.str.lower()
        self.df['ticker'] = ticker
        # Sort by date
        self.df.sort_values('date', inplace=True)
        self.df.reset_index(drop=True, inplace=True)

        print(f"Data for {ticker} loaded and sorted by date successfully.")

    def plot_ohlc_prices(self):
        """
        Plot open, high, low, and close prices as line plots.
        """
        plt.figure(figsize=(12,6))
        plt.plot(self.df['date'], self.df['open'], label='Open', alpha=0.6)
        plt.plot(self.df['date'], self.df['high'], label='High', alpha=0.6)
        plt.plot(self.df['date'], self.df['low'], label='Low', alpha=0.6)
        plt.plot(self.df['date'], self.df['close'], label='Close', alpha=0.8, linewidth=2)
        plt.title(f'{self.ticker} OHLC Prices')
        plt.xlabel('Date')
        plt.ylabel('Price')
        plt.legend()
        plt.show()

    def plot_candlestick(self):
        """
        Plot candlestick chart using mplfinance.
        """
        import mplfinance as mpf
        df_candle = self.df.set_index('date')
        mpf.plot(df_candle,
                type='candle',
                volume=True,
                title=f'{self.ticker} Candlestick Chart',
                ylabel='Price',
                ylabel_lower='Volume',
                style='yahoo')

    def add_technical_indicators(self):
        """
        Add key technical indicators: Moving Averages, RSI, MACD.
        """
        self.df['MA_20'] = talib.SMA(self.df['close'], timeperiod=20)
        self.df['MA_50'] = talib.SMA(self.df['close'], timeperiod=50)
        self.df['RSI'] = talib.RSI(self.df['close'], timeperiod=14)
        self.df['MACD'], self.df['MACD_signal'], self.df['MACD_hist'] = talib.MACD(self.df['close'])
        print(f"Technical indicators added for {self.ticker}.")

    def compute_daily_returns(self):
        """
        Compute daily returns.
        """
        self.df['daily_return'] = self.df['close'].pct_change()
        print(f"Daily returns computed for {self.ticker}.")

    def adjust_for_splits(self):
        """
        Adjust closing price for stock splits.
        """
        if 'stock splits' in self.df.columns:
            self.df['adjusted_close'] = self.df['close'] / (self.df['stock splits'].replace(0, 1))
            print(f"Adjusted for stock splits for {self.ticker}.")
        else:
            self.df['adjusted_close'] = self.df['close']
            print("No stock splits data found.")

    def plot_price_and_ma(self):
        """
        Plot closing price and moving averages.
        """
        plt.figure(figsize=(12,6))
        plt.plot(self.df['date'], self.df['close'], label='Close Price', alpha=0.6)
        plt.plot(self.df['date'], self.df['MA_20'], label='20-day MA', color='red')
        plt.plot(self.df['date'], self.df['MA_50'], label='50-day MA', color='green')
        plt.title(f'{self.ticker} Price with Moving Averages')
        plt.xlabel('Date')
        plt.ylabel('Price')
        plt.legend()
        plt.show()

    def plot_volume(self):
        """
        Plot trading volume over time.
        """
        plt.figure(figsize=(12,4))
        plt.bar(self.df['date'], self.df['volume'], color='skyblue')
        plt.title(f'{self.ticker} Trading Volume')
        plt.xlabel('Date')
        plt.ylabel('Volume')
        plt.show()

    def plot_macd(self):
        """
        Plot MACD indicator.
        """
        plt.figure(figsize=(12,4))
        plt.plot(self.df['date'], self.df['MACD'], label='MACD', color='blue')
        plt.plot(self.df['date'], self.df['MACD_signal'], label='Signal Line', color='orange')
        plt.title(f'{self.ticker} MACD')
        plt.xlabel('Date')
        plt.ylabel('MACD')
        plt.legend()
        plt.show()

    def plot_rsi(self):
        """
        Plot RSI indicator.
        """
        plt.figure(figsize=(12,4))
        plt.plot(self.df['date'], self.df['RSI'], color='purple')
        plt.axhline(70, color='red', linestyle='--', label='Overbought (70)')
        plt.axhline(30, color='green', linestyle='--', label='Oversold (30)')
        plt.title(f'{self.ticker} RSI')
        plt.xlabel('Date')
        plt.ylabel('RSI')
        plt.legend()
        plt.show()


    def calculate_pynance_metrics(self):
        """
        Calculate additional financial metrics using PyNance.
        """
        from pynance import indicators

        # Calculate volatility (standard deviation of returns)
        volatility = indicators.volatility(self.df['close'])
        self.df['volatility'] = volatility

        # Calculate momentum (rate of change)
        momentum = indicators.momentum(self.df['close'], window=10)
        self.df['momentum'] = momentum

        print(f"PyNance metrics (volatility, momentum) added for {self.ticker}.")

    def plot_volatility(self):
        """
        Plot the volatility metric over time.
        """
        if 'volatility' in self.df.columns:
            plt.figure(figsize=(12,4))
            plt.plot(self.df['date'], self.df['volatility'], color='teal')
            plt.title(f'{self.ticker} Volatility Over Time')
            plt.xlabel('Date')
            plt.ylabel('Volatility')
            plt.show()
        else:
            print("Volatility data not found. Please run calculate_pynance_metrics() first.")

    def plot_momentum(self):
        """
        Plot the momentum metric over time.
        """
        if 'momentum' in self.df.columns:
            plt.figure(figsize=(12,4))
            plt.plot(self.df['date'], self.df['momentum'], color='orange')
            plt.title(f'{self.ticker} Momentum Over Time')
            plt.xlabel('Date')
            plt.ylabel('Momentum')
            plt.show()
        else:
            print("Momentum data not found. Please run calculate_pynance_metrics() first.")


    def plot_dividends(self):
        """
        Plot dividend payouts over time if available.
        """
        if 'dividend' in self.df.columns and self.df['dividend'].sum() > 0:
            plt.figure(figsize=(12,4))
            plt.bar(self.df['date'], self.df['dividend'], color='gold')
            plt.title(f'{self.ticker} Dividend Payouts')
            plt.xlabel('Date')
            plt.ylabel('Dividend')
            plt.show()
        else:
            print("No dividend data found.")

    def get_data(self):
        """
        Return the DataFrame.
        """
        return self.df
    

