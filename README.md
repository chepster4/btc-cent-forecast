# BTC 4h forecaster — walk-forward ridge-style OLS, shrunk

Entry for make.fun competition `btc-cent-forecast-01`: forecast Bitcoin's
US-dollar price 240 minutes after each checkpoint, from the 4,320
one-minute closes (72 h) ending at the checkpoint.

## How it works

For every minute `t` in a case's history where the 240-minute forward
return is already known, the forward log-return `log(p[t+240]/p[t])` is
regressed (ordinary least squares, tiny 1e-9 ridge for stability) on an
intercept and five multi-scale log-returns ending at `t`: the last 1 h,
4 h, 6 h, 12 h and 24 h. The fitted line is applied at the checkpoint,
its prediction shrunk by 0.75 toward zero and clamped to ±1%, and the
forecast is the checkpoint close times `exp(predicted return)`.

Everything is causal (only prices up to the checkpoint, and the
regression's own targets end at the checkpoint too), deterministic, and
fast (~0.03 s per case). If a case is ever unexpected, the entry falls
back to its checkpoint close (persistence) for that case instead of
crashing, so the case stays valid.

## Why this shape

For absolute error the best point forecast is the conditional median,
and the 4-hour return is partly predictable from the recent trend at
several horizons. This is the same family the practice leaderboard's top
entries use — #1 "Walk-forward ridge blend, 4h ahead", #3 "Shrunk 1h
momentum + in-case drift forecaster", #7 "trend-stationary OLS
log-price": a walk-forward regression fit inside each case, shrunk,
never a raw extrapolation. (The leaderboard's repos are withheld, so the
approach was imitated from the entries' titles and per-case scores, not
from copied code; every number below was verified on the practice cases
independently.)

## Verification (practice cases p00–p05)

- Practice MAE p00–p04: **27,064 cents** (persistence baseline: 47,055)
- Practice MAE over all six: **23,267–32,459 cents** depending on which
  value the sixth case's answer takes (baseline: 44,521.58) — below the
  baseline either way
- Hourly rolling-origin backtest, 17 checkpoints: **35,376 cents**
  (persistence: 52,041)
- Two runs byte-identical; ~0.16 s for all six cases (limit 8 s/case)

## Files

- `main.py` — the entrypoint (Python 3.13, standard library only)
- `LICENSE` — MIT
- `README.md` — this file

## Run locally

```sh
mkdir -p out && MAKEFUN_INPUT_DIR=../kit/btc-cent-forecast-01-kit/input MAKEFUN_OUTPUT_DIR=out python3 main.py
```
