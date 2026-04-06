# AAPL Trading Agent

AI-powered BUY/SELL/HOLD decision system for Apple Inc. stock.

## Setup

1. Clone the repo
2. Create virtual environment:
   python -m venv venv
   source venv/Scripts/activate  (Git Bash)
   venv\Scripts\activate         (Command Prompt)

3. Install dependencies:
   pip install -r requirements.txt

4. Add your Finnhub API key in config.py:
   FINNHUB_API_KEY = "your_key_here"
   Get a free key at https://finnhub.io

## Run the pipeline

python data/collector.py
python data/preprocessor.py
python features/technical.py

## Project Structure

data/          — data collection and preprocessing
features/      — technical indicator engineering
agent/         — ML models, heuristic agent, backtester
evaluation/    — metrics and visualizations
config.py      — all settings in one place
main.py        — runs entire pipeline end to end