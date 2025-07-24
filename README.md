# CrossImpactAnalyzer

**Quantitative Modeling of Trading Signals using Cross-Impact and Order Flow Imbalance**

[![Project Slides](https://github.com/A-Binoy/CrossImpactAnalyzer/blob/main/CIA/Highlights.pdf)  | 🔗 [Original Paper](https://doi.org/10.1080/14697688.2023.2236159)

---

## 📌 Overview

This project explores whether **cross-asset order flow imbalance (OFI)** can be used to generate **predictive trading signals** in equity markets. Inspired by the work of Cont, Cucuringu, and Zhang (2023) [[1]](#references), we replicate core findings using real Level 2 data from the LOBSTER dataset and test a long-only strategy based on OFI signal strength.

Key questions explored:

- Can cross-impact coefficients be reliably estimated?
- Do lagged OFI signals have predictive power?
- Can we translate this signal into profitable trades?

---

## 🧠 Core Idea

**Order Flow Imbalance (OFI)** quantifies net buying/selling pressure using limit order book dynamics.  
This project:
- Constructs a PCA-based **integrated OFI** from multi-level LOB data.
- Builds cross-sectional regression models to study **contemporaneous** and **predictive** cross-impact.
- Simulates a trading strategy that takes long positions when integrated OFI exceeds the 70th percentile.

---

## ⚙️ Methods

- **Data**: LOBSTER message and orderbook files for 5 Nasdaq stocks: `AAPL`, `AMZN`, `GOOG`, `INTC`, `MSFT`
- **Signal Construction**:
  - Parse Level 2 LOB data using C++ (`ofi_tools.cpp`) via Pybind11.
  - Compute per-level OFI and apply PCA to obtain an integrated OFI per stock.
- **Modeling**:
  - Linear regression for cross-impact estimation.
  - Strategy: go long on stocks with high lagged OFI (top 30%).

---

## 📈 Results

- 📊 **R² scores** for contemporaneous cross-impact were high for self-impact, but weak for other stocks — consistent with theory.
- ⏱️ **Predictive power** of OFI was significantly higher at 1-minute horizons than at 5-minute.
- 💸 Strategy based on top-30% OFI signals yielded **~7% cumulative return** over the test period.

See [impact_results.txt](./impact_results.txt) for full regression outputs.

---

## 📊 Visualizations

| Figure | Description |
|--------|-------------|
| ![cumulative_pnl](./cumulative_pnl.png) | Cumulative returns from the OFI-based strategy |
| ![cross_impact_heatmap](./cross_impact_heatmap.png) | Heatmap of cross-impact coefficients |
| ![predictive_power](./predictive_power.png) | Predictive R² at 1m and 5m horizons |
| ![pnl_high_vs_low_ofi_AAPL](./pnl_high_vs_low_ofi_AAPL.png) | AAPL PnL in high vs low OFI periods |
| ![pnl_vs_ofi_AMZN](./pnl_vs_ofi_AMZN.png) | Scatter of AMZN PnL vs lagged OFI |
| ... | (See `/Viz` folder for full set) |

---

## 🔬 Future Work

- Test cross-impact in **sector-specific portfolios** (e.g., tech vs energy).
- Incorporate **transaction costs** and slippage models.
- Add **multi-horizon** signals and expand to full S&P100.
- Explore **non-linear models** (e.g., Random Forests) for capturing nonlinear OFI effects.

---


