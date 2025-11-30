# app.py
# Streamlit app - Simple 30-day WTI (CL=F) Monte Carlo + shock scenarios + simple position sizing (1:2)
# Copy this file into your repo and run: streamlit run app.py

import streamlit as st
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import yfinance as yf

st.set_page_config(page_title="WTI 30d Predictor — Simple", layout="centered")

st.title("WTI 30-d Predictor — Én knap, klar til swing trade")
st.markdown("En simpel app: henter live WTI (CL=F), estimerer volatilitet, kører to simulationer (normal + shock), og foreslår stop-loss / take-profit + positionstørrelse (1:2 gearing).")

# -----------------------
# Inputs (kept minimal)
# -----------------------
col1, col2 = st.columns(2)
with col1:
    ticker = st.text_input("Ticker for WTI (yfinance)", value="CL=F")
    days = st.number_input("Horisont (dage)", value=30, min_value=1)
    sims = st.number_input("Antal simulationer", value=7000, min_value=1000)
with col2:
    account_usd = st.number_input("Konto (USD)", value=2880.0, step=100.0)
    risk_pct = st.number_input("Risiko pr. trade (% af konto)", value=1.0, min_value=0.1) / 100.0
    allocation_pct = st.number_input("Max margin allocation (% af konto)", value=50.0, min_value=1.0) / 100.0
    leverage = st.number_input("Gearing (fx 2 = 1:2)", value=2.0, min_value=1.0)

run = st.button("Forudsig 30 dage — Kør")

if not run:
    st.info("Tryk på knappen for at hente live data og lave forudsigelsen.")
    st.stop()

# -----------------------
# Fetch live data
# -----------------------
st.info("Henter seneste prisdata...")
try:
    data = yf.download(ticker, period="180d", interval="1d", progress=False)
    if data.empty:
        raise RuntimeError("Ingen data fra yfinance for ticker.")
except Exception as e:
    st.error(f"Hentning af data mislykkedes: {e}")
    st.stop()

# Use adjusted close if available, else close
if "Adj Close" in data.columns:
    series = data["Adj Close"].dropna()
else:
    series = data["Close"].dropna()

if series.empty:
    st.error("Ingen prisdata tilgængeligt.")
    st.stop()

S0 = float(series.iloc[-1])
st.write(f"**Aktuel pris ({ticker}):** {S0:.2f} USD (dato {series.index[-1].date()})")

# -----------------------
# Estimate volatility (daily)
# -----------------------
returns = series.pct_change().dropna()
# Use recent window (e.g., 60d) if available
window = min(60, len(returns))
recent_ret = returns.iloc[-window:]
daily_sigma = float(recent_ret.std(ddof=0))
if daily_sigma <= 0:
    daily_sigma = 0.02  # fallback
st.write(f"Estimeret daglig volatilitet (historisk, seneste {window} dage): {daily_sigma:.4f} ({daily_sigma*100:.2f}% per dag)")

# -----------------------
# Monte Carlo functions
# -----------------------
def simulate_gbm(S0, mu, sigma, days, sims, rng):
    """
    Geometric Brownian Motion simulation.
    Returns array shape (sims, days+1)
    """
    dt = 1.0
    paths = np.zeros((sims, days + 1), dtype=float)
    paths[:, 0] = S0
    drift = (mu - 0.5 * sigma ** 2) * dt
    for t in range(1, days + 1):
        z = rng.normal(size=sims)
        paths[:, t] = paths[:, t - 1] * np.exp(drift + sigma * z)
    return paths

def simulate_with_jumps(S0, mu, sigma, days, sims, rng, jump_lambda=0.15, jump_mu=0.0, jump_sigma=0.08):
    """
    GBM with Poisson jumps: occasional jumps added multiplicatively.
    jump_lambda = expected number of jumps over whole horizon per path (converted to daily inside)
    jump_mu/jump_sigma = jump size distribution (log-normal-like additive to log-price)
    """
    dt = 1.0
    daily_lambda = jump_lambda / days if days > 0 else jump_lambda
    paths = np.zeros((sims, days + 1), dtype=float)
    paths[:, 0] = S0
    for t in range(1, days + 1):
        z = rng.normal(size=sims)
        # Poisson events
        jumps = rng.uniform(size=sims) < daily_lambda
        # jump magnitude (additive on log scale)
        jump_sizes = np.zeros(sims)
        if jumps.any():
            jump_sizes[jumps] = rng.normal(loc=jump_mu, scale=jump_sigma, size=jumps.sum())
        paths[:, t] = paths[:, t - 1] * np.exp((mu - 0.5 * sigma ** 2) + sigma * z + jump_sizes)
    return paths

# -----------------------
# Run simulations: normal + shock
# -----------------------
rng = np.random.default_rng(seed=42)
mu = 0.0  # neutral drift assumption
sims = int(sims)
days = int(days)

# Normal scenario (historical vol)
paths_normal = simulate_gbm(S0=S0, mu=mu, sigma=daily_sigma, days=days, sims=sims, rng=rng)
final_normal = paths_normal[:, -1]

# Shock scenario: higher vol + occasional jumps
shock_sigma = max(daily_sigma * 1.8, daily_sigma + 0.02)
paths_shock = simulate_with_jumps(S0=S0, mu=mu, sigma=shock_sigma, days=days, sims=sims, rng=rng,
                                  jump_lambda=0.6, jump_mu=0.0, jump_sigma=0.10)
final_shock = paths_shock[:, -1]

# -----------------------
# Summary stats + recommended SL / TP
# -----------------------
def summary_and_levels(final_prices, pct_low=5, pct_high=95):
    median = float(np.median(final_prices))
    p_low = float(np.percentile(final_prices, pct_low))
    p_high = float(np.percentile(final_prices, pct_high))
    return {"median": median, "low": p_low, "high": p_high}

norm_stats = summary_and_levels(final_normal, pct_low=5, pct_high=95)
shock_stats = summary_and_levels(final_shock, pct_low=5, pct_high=95)

# We'll recommend:
# - Stop-loss = normal 5% percentile (conservative)
# - Take-profit = normal 95% percentile
stoploss_price = norm_stats["low"]
takeprofit_price = norm_stats["high"]
median_price = norm_stats["median"]

# Dates
start_date = datetime.utcnow().date()
end_date = (datetime.utcnow() + timedelta(days=days)).date()

# -----------------------
# Position sizing (simple, risk-based)
# -----------------------
# Desired USD risk
risk_usd_target = account_usd * risk_pct

# If stop is above entry price (unlikely), flip sign to avoid div by zero:
if S0 <= stoploss_price:
    # fallback: set stoploss at S0 * 0.95 if weird
    stoploss_price = S0 * 0.95

price_diff = S0 - stoploss_price
if price_diff <= 0:
    units = 0.0
else:
    units = risk_usd_target / price_diff  # number of barrels (USD exposure = units * price_diff)
# Apply leverage constraints: margin requirement = (units * S0) / leverage
margin_required = (units * S0) / leverage
max_margin = account_usd * allocation_pct

# If margin_required exceeds allocation, scale units down
if margin_required > max_margin and margin_required > 0:
    scale = max_margin / margin_required
    units *= scale
    margin_required = (units * S0) / leverage

# Round down units to a practical whole number (barrels); many brokers trade contracts - keep integer
units = int(np.floor(units))
if units < 0:
    units = 0
margin_required = (units * S0) / leverage
estimated_risk = units * price_diff  # USD

# -----------------------
# Output
# -----------------------
st.subheader("Resultat — normal vs shock scenarie (30 dage)")
coln, cols = st.columns(2)
with coln:
    st.markdown("**Normal scenarie (historisk vol)**")
    st.write(f"Median slutpris: **{median_price:.2f} USD**")
    st.write(f"5% worst case: **{norm_stats['low']:.2f} USD**")
    st.write(f"95% best case: **{norm_stats['high']:.2f} USD**")
with cols:
    st.markdown("**Shock scenarie (højere vol + jumps)**")
    st.write(f"Median slutpris: **{shock_stats['median']:.2f} USD**")
    st.write(f"5% worst case: **{shock_stats['low']:.2f} USD**")
    st.write(f"95% best case: **{shock_stats['high']:.2f} USD**")

st.subheader("Foreslået trade-opsætning (én knap-output)")
st.write(f"Entry dato: **{start_date}** — Target dato: **{end_date}**")
st.write(f"Entry pris: **{S0:.2f} USD**")
st.write(f"Anbefalet Stop-Loss (konservativ, 5% percentil normal): **{stoploss_price:.2f} USD**")
st.write(f"Anbefalet Take-Profit (95% percentil normal): **{takeprofit_price:.2f} USD**")

st.markdown("**Position sizing (simpel, baseret på ønsket risiko)**")
st.write(f"Konto: {account_usd:.2f} USD — Risiko pr. trade: {risk_pct*100:.2f}% → Risiko USD mål: {risk_usd_target:.2f} USD")
st.write(f"Foreslået antal enheder (barrels/kontrakter): **{units}**")
st.write(f"Estimeret USD-risiko for position: **{estimated_risk:.2f} USD**")
st.write(f"Margin required (ved {leverage:.1f}x): **{margin_required:.2f} USD** (max allocation: {max_margin:.2f} USD)")

if units == 0:
    st.warning("Udregnet units = 0 (sandsynligvis fordi stoploss er for tæt på entry eller risiko for lav). Juster inputs (risiko%, allocation eller drift).")

# -----------------------
# Plots
# -----------------------
st.subheader("Visualisering (eksempelbaner + distribution)")

# Show a handful of random sample paths (normal)
sample_n = min(30, sims)
idx = rng.choice(sims, sample_n, replace=False)
fig1, ax1 = plt.subplots(figsize=(6, 3.8))
for i in idx:
    ax1.plot(paths_normal[i], linewidth=1, alpha=0.9)
ax1.axhline(S0, color="black", linewidth=0.8, linestyle="--", label="Entry")
ax1.set_title(f"Eksempel på {sample_n} simulerede prisbaner (normal)")
ax1.set_xlabel("Dage")
ax1.set_ylabel("Pris (USD)")
ax1.legend(loc="upper left", fontsize="small")
st.pyplot(fig1)

# Histogram of final prices (normal)
fig2, ax2 = plt.subplots(figsize=(6, 3.8))
ax2.hist(final_normal, bins=60)
ax2.axvline(stoploss_price, color="red", linestyle="--", label="Stop-loss (5% quantile)")
ax2.axvline(takeprofit_price, color="green", linestyle="--", label="Take-profit (95% quantile)")
ax2.set_title("Distribution af slutpriser (normal scenarie)")
ax2.set_xlabel("Slutpris (USD)")
ax2.set_ylabel("Antal simulationer")
ax2.legend()
st.pyplot(fig2)

st.success("Færdig — brug disse levels som et statistisk input til din trading. Husk altid at tjekke live-nyheder og margin-krav hos din broker.")
