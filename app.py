import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

st.title("WTI Crude Oil – Monte Carlo Simulation (30 dage)")

# ---- Inputs ----
S0 = st.number_input("Startpris (USD)", value=60.0)
sigma = st.number_input("Daglig volatilitet (fx 0.02 = 2%)", value=0.02, step=0.01)
mu = st.number_input("Daglig drift (typisk 0)", value=0.0)
N = st.number_input("Antal dage", value=30)
M = st.number_input("Antal simulationer", value=5000)

run = st.button("Kør simulation")

if run:
    # Monte Carlo Simulation
    paths = np.zeros((int(M), int(N)+1))
    paths[:, 0] = S0

    for i in range(int(M)):
        for t in range(1, int(N)+1):
            paths[i, t] = paths[i, t-1] * np.exp((mu - 0.5 * sigma**2) + sigma * np.random.normal())

    final_prices = paths[:, -1]

    st.subheader("📊 Resultater")
    st.write(f"**Median slutpris:** {np.median(final_prices):.2f} USD")
    st.write(f"**5% worst case:** {np.percentile(final_prices, 5):.2f} USD")
    st.write(f"**95% best case:** {np.percentile(final_prices, 95):.2f} USD")

    # ---- Plot random paths ----
    st.subheader("📈 Eksempel på prisbaner")
    fig1, ax1 = plt.subplots()
    for i in range(20):  # vis 20 tilfældige stier
        ax1.plot(paths[i])
    ax1.set_title("Simulerede WTI-prisbaner (20 stk.)")
    ax1.set_ylabel("Pris (USD)")
    ax1.set_xlabel("Dage")
    st.pyplot(fig1)

    # ---- Distribution ----
    st.subheader("📉 Distribution af slutpriser")
    fig2, ax2 = plt.subplots()
    ax2.hist(final_prices, bins=50)
    ax2.set_title("Fordeling af priser efter 30 dage")
    ax2.set_xlabel("Slutpris (USD)")
    st.pyplot(fig2)
