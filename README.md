
# Financial Sentiment Analysis

Predicting stock price movements using news sentiment analysis.

---

## 📈 Project Overview

This project explores the relationship between financial news sentiment and stock price movements. It includes:

- Sentiment analysis of news headlines using NLP.
- Technical indicators with TA-Lib and PyNance.
- Correlation analysis between sentiment scores and daily stock returns.
- Actionable insights for investment strategies.

---

## 🚀 Project Structure

```
financial-sentiment-analysis/
├── .vscode/               # VSCode settings (optional)
├── .github/               # CI/CD workflows
├── data/                  # Datasets (financial news & stock prices)
├── notebooks/             # Jupyter notebooks for EDA
├── scripts/               # Data preprocessing scripts
├── src/                   # Core sentiment analysis module
├── tests/                 # Unit tests
├── requirements.txt       # Python dependencies
├── README.md              # Project documentation
└── .gitignore
```

---

## 🛠️ Setup Instructions

1️⃣ **Clone the repository:**
```bash
git clone https://github.com/AmhaBK/stock-price-sentiment-analysis.git
```

2️⃣ **Create a virtual environment:**
```bash
python -m venv venv
```

3️⃣ **Activate the environment:**

- Windows:
```bash
venv\Scripts\activate
```

4️⃣ **Install dependencies:**
```bash
pip install -r requirements.txt
```

---

## ⚙️ Running the Analysis

- Use the `notebooks/EDA.ipynb` notebook to explore data and visualize insights.
- Use the `src/sentiment_analysis.py` module to compute sentiment scores.
- Use the `scripts/data_preprocessing.py` script for data cleaning and alignment.

---

## ✅ Testing

Run unit tests:
```bash
python -m unittest discover -s tests
```

---

## 🚀 CI/CD

The project uses GitHub Actions to run unit tests automatically on each push and PR.


---

## 📚 References

- [TextBlob](https://textblob.readthedocs.io/en/dev/)
- [TA-Lib](https://github.com/ta-lib/ta-lib-python)
- [PyNance](https://github.com/mqandil/pynance)
- [Investopedia Stock Market](https://www.investopedia.com/terms/s/stockmarket.asp)

---