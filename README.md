# Cross-Impact Analysis of Order Flow Imbalance in Equity Markets

## Overview

This project analyzes high-frequency equity market data to compute **Order Flow Imbalance (OFI) metrics** across multiple levels of the **Limit Order Book (LOB)** and assess **cross-asset impacts** on short-term price changes. The methodology follows the paper **"Cross-Impact of Order Flow Imbalance in Equity Markets."**

Key components of the project:
- Computation of **multi-level OFI metrics**
- Analysis of **cross-impact** between stocks
- Regression-based **quantification** of impact relationships
- **Visualization** of findings through time series, heatmaps, and predictive power plots

## Task Description

### 1. Compute OFI Metrics
- Derive **multi-level OFI metrics** (up to 5 levels) for each stock.
- Integrate these metrics using **Principal Component Analysis (PCA)** or another **dimensionality reduction method**.

### 2. Analyze Cross-Impact
- Assess **contemporaneous cross-impact** of OFI on short-term price changes across stocks.
- Evaluate the **predictive power** of lagged cross-asset OFI on **future price changes** (e.g., 1-minute and 5-minute horizons).

### 3. Quantify Results
- Use **regression models** to assess the **explanatory power** of contemporaneous and lagged OFI.
- Compare **self-impact** (within the same stock) vs. **cross-impact** (between stocks).

### 4. Visualization and Reporting
- Generate **visualizations** to illustrate relationships:
  - **OFI time series** for each stock
  - **Cross-impact heatmap**
  - **Predictive power analysis**
  - **Cumulative impact plots**

