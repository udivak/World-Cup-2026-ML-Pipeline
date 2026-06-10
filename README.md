# World Cup 2026 Prediction System

A multi-stage soccer match prediction system for the 2026 World Cup. Learns from ~47k historical
international matches (1872–present) to produce calibrated Win/Draw/Loss probabilities and
Monte-Carlo–simulates the tournament for advancement and title odds.

**Stack:** Python + scikit-learn, pandas/numpy, SQLAlchemy → Supabase Postgres.

## Quickstart

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env            # then fill DATABASE_URL with your Supabase password
pytest tests/                   # fixture-backed; DB tests skip without DATABASE_URL
python -m src.collect.matches_loader   # download + load matches
```

## Setup

1. Copy `.env.example` to `.env` and fill `DATABASE_URL` with your Supabase Postgres password.
2. The connection string format: `postgresql://postgres:<password>@db.rdaxlpoeuaeivreziaio.supabase.co:5432/postgres?sslmode=require`
3. Run `python -m src.common.db` to verify the connection and create the `wc2026` schema.

## Project structure

```
src/
  common/    config.py, db.py, io.py, teams.py
  collect/   matches_loader.py
  features/  elo.py, form.py, context.py, build_features.py
  models/    baselines.py, train.py, calibrate.py, evaluate.py
  predict/   predict.py, simulate_tournament.py
data/raw/    cached CSV downloads (gitignored)
notebooks/   EDA and reporting notebooks
tests/
config.yaml
```

## Phases

| Phase | Description |
|-------|-------------|
| 0 | Scaffold + data pipeline |
| 1 | MVP backtest (Elo baseline → beat it) |
| 2 | Transfermarkt squad enrichment |
| 3 | Live 2026 predictions + tournament simulation |
