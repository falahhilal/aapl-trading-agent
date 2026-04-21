# 📈 AAPL Trading Agent

An end-to-end AI-powered algorithmic trading system for Apple Inc. (AAPL) stock.
The agent observes daily market conditions and autonomously decides whether to
**BUY**, **SELL**, or **HOLD** shares to maximize financial return.

🔗 **Live API:** https://aapl-trading-agent.onrender.com/
🔗 **Live App:** https://aapl-trading-agent.streamlit.app/

---

## What it does
Real AAPL market data (2015–2024)
↓
Feature engineering — 32 technical indicators + VIX macro feature
↓
MLP classifier predicts UP / DOWN / NEUTRAL for next trading day
↓
Heuristic agent filters by confidence threshold, RSI, and VIX
↓
BUY / SELL / HOLD decision with reasoning
↓
Backtesting simulation — $10,000 virtual capital, 0.1% transaction cost
↓
Results compared against passive Buy-and-Hold baseline
---

## Results

| Metric | Agent | Buy and Hold |
|---|---|---|
| Final Portfolio Value | $13,840 | $20,387 |
| Total Return | +38.41% | +103.87% |
| Sharpe Ratio | 1.334 | 1.799 |
| Max Drawdown | -14.15% | -16.55% |
| Win Rate | 76.5% (13/17 trades) | N/A |
| Total Trades | 17 | 1 |

---

## Features engineered (32 total)

| Category | Features |
|---|---|
| Price-derived | daily_return, log_return, high_low_range, close_open_gap, prev_close |
| Macro | vix_close, vix_change |
| RSI | rsi (14-day) |
| MACD | macd, macd_signal, macd_histogram |
| Bollinger Bands | bb_width, bb_position |
| Rolling statistics | rolling_std + price_vs_mean (5, 10, 20 day windows) |
| Lag returns | lag_return (1, 2, 3, 5 days) |
| Volume | volume_ratio, log_volume, volume_change |
| Cyclic date encoding | dow_sin, dow_cos, month_sin, month_cos |
| Momentum | roc_5, roc_10 |

---

## Models trained

| Model | Accuracy | F1 Macro |
|---|---|---|
| **MLP (selected)** | **0.369** | **0.365** |
| Random Forest | 0.363 | 0.361 |
| KNN | 0.343 | 0.341 |
| Logistic Regression | 0.349 | 0.332 |
| SVM | 0.335 | 0.326 |

F1 scores around 0.33–0.37 are expected for stock direction prediction.
Random baseline on a balanced 3-class problem is 0.33.
Walk-forward validation across 5 years (2019–2023) achieved mean F1 of 0.318.

---

## API Endpoints

| Endpoint | Description |
|---|---|
| `GET /` | Health check — returns model status |
| `GET /predict/{date}` | BUY/SELL/HOLD for any date (YYYY-MM-DD) |
| `GET /predict-range?start=&end=` | Predictions for a date range |
| `GET /backtest` | Full backtest metrics vs Buy-and-Hold |
| `GET /trades` | All 17 completed trades as JSON |


---

## Project structure
aapl-trading-agent/
├── data/
│   ├── collector.py       # downloads AAPL + VIX from Yahoo Finance
│   └── preprocessor.py    # cleans data, merges VIX, generates labels
├── features/
│   └── technical.py       # engineers 32 features from scratch
├── agent/
│   ├── classifier.py      # trains 5 ML models, walk-forward validation
│   ├── heuristic.py       # TradingAgent class — stateful decision logic
│   └── backtester.py      # portfolio simulation with transaction costs
├── evaluation/
│   └── metrics.py         # Sharpe ratio, drawdown, equity curve plots
├── raw_data/              # auto-created, gitignored
├── outputs/               # plots and results
├── models/                # trained model saved here
├── app.py                 # Streamlit frontend
├── api.py                 # FastAPI backend
├── predict.py             # lightweight inference module
├── config.py              # all settings in one place
├── main.py                # runs full pipeline end to end
├── Dockerfile             # containerizes the API
├── requirements.txt
└── .github/
└── workflows/
└── ci.yml         # CI/CD — build, test, deploy on every push
---

## Setup & run locally

### 1. Clone the repo
```bash
git clone https://github.com/falahhilal/aapl-trading-agent.git
cd aapl-trading-agent
```

### 2. Create virtual environment
```bash
python -m venv venv
source venv/Scripts/activate   # Git Bash on Windows
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Run the full pipeline
```bash
python main.py
```

### 5. Start the API
```bash
uvicorn api:app --reload
```

### 6. Start the Streamlit app
```bash
streamlit run app.py
```

---

## Run with Docker

```bash
docker build -t aapl-trading-agent .
docker run -p 8000:8000 aapl-trading-agent
```

API available at `http://localhost:8000/docs`

---

## CI/CD Pipeline

Every push to `main` automatically:

1. Builds the Docker image on GitHub Actions
2. Starts the container
3. Runs smoke tests on all 4 API endpoints
4. Deploys to Render if all tests pass

Pipeline status visible in the **Actions** tab.

---

## Tech stack

| Category | Tools |
|---|---|
| ML & Data | Python, scikit-learn, pandas, numpy |
| Market Data | yfinance |
| API | FastAPI, uvicorn |
| Containerization | Docker |
| Deployment | Render |
| CI/CD | GitHub Actions |
| Frontend | Streamlit |
| Visualization | matplotlib, seaborn |

---

## What this is NOT

- Not live trading with real money
- Not financial advice
- Not reinforcement learning