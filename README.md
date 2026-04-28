# 📊 ETF Directional Bias Prediction with MLP & Technical Indicators

A machine learning project that predicts the **next-day directional bias** (up or down) of the **iShares Core S&P 500 ETF (IVV)** using a Multi-Layer Perceptron (MLP) trained on a comprehensive set of technical indicators. The study evaluates how feature selection via correlation ranking affects classification accuracy using 10-fold stratified cross-validation.

---

## 💡 Core Idea

The target variable Γ(t) is defined as:

$$\Gamma(t) = \begin{cases} +1 & \text{if } \text{Close}(t) - \text{Close}(t-1) > 0 \\ -1 & \text{otherwise} \end{cases}$$

The model is trained to predict **Γ(t+1)** — tomorrow's direction — using today's technical indicator values. This frames price direction forecasting as a binary classification problem.

---

## 🔬 What's Covered

| Section | Description |
|---|---|
| Data Download | IVV daily OHLCV data via `yfinance` (Jan 2009 – Jan 2020) |
| Feature Extraction | All available `pandas_ta` indicators applied automatically (~323 features) |
| Data Cleaning | Columns with any NaN dropped; data trimmed to December 2009 onwards |
| Target Definition | Binary directional label Γ(t+1) computed from close-to-close returns |
| Normalisation | Min-Max scaling applied to all features |
| Feature Selection | Absolute Pearson correlation with target used to rank features |
| Model Training | MLP with single hidden layer, logistic activation, LBFGS solver |
| Evaluation | 10-fold stratified cross-validation across 7 feature subset sizes |

---

## 🏗️ Model Architecture

The MLP hidden layer size is derived dynamically from the number of input features:

```
hidden_size = (num_features + num_classes) / 2
activation  = logistic (sigmoid)
solver      = lbfgs
learning_rate = adaptive (init = 0.03)
momentum    = 0.2
max_iter    = 5000
```

---

## 📊 Feature Selection Experiment

Seven feature subset sizes are evaluated against the full-feature baseline:

| Subset Size | Description |
|---|---|
| 5 | Top 5 most correlated features |
| 7 | Top 7 |
| 10 | Top 10 |
| 20 | Top 20 |
| 50 | Top 50 |
| 100 | Top 100 |
| 323 | All features (baseline) |

Results are reported as **median CV accuracy**, **mean CV accuracy**, and **gain vs. baseline (pp)** for each subset size, with a plot of accuracy vs. number of features.

---

## ⚙️ Configuration

| Parameter | Value |
|---|---|
| Ticker | `IVV` |
| Data range | 2009-01-01 to 2020-01-01 |
| CV folds | 10 (Stratified, no shuffle) |
| Feature ranking | Absolute Pearson correlation with target |
| Scaler | MinMaxScaler |

---

## 🛠️ Installation

```bash
pip install numpy pandas matplotlib scikit-learn yfinance pandas_ta TA-Lib
```

> **Note:** `TA-Lib` requires a C library to be installed first. See the [TA-Lib installation guide](https://github.com/TA-Lib/ta-lib-python) for platform-specific instructions.

---

## 🚀 Usage

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/your-repo-name.git
   cd your-repo-name
   ```
2. Install dependencies (see above).
3. Open the notebook:
   ```bash
   jupyter notebook etf-directional-bias-prediction.ipynb
   ```
4. Run all cells (`Kernel → Restart & Run All`).

> **Note:** An active internet connection is required to download IVV data via `yfinance`.

---

## 📦 Dependencies

| Package | Purpose |
|---|---|
| `yfinance` | Download IVV historical OHLCV data |
| `pandas_ta` | Compute all technical indicators automatically |
| `TA-Lib` | Underlying C library required by `pandas_ta` |
| `scikit-learn` | MLP classifier, MinMaxScaler, stratified CV, accuracy metric |
| `pandas` / `numpy` | Data manipulation and numerical operations |
| `matplotlib` | Accuracy vs. feature count plot and cumulative Gamma chart |

---

## 📁 Repository Structure

```
.
├── etf-directional-bias-prediction.ipynb    # Main analysis notebook
└── README.md                                # This file
```

---

## ⚠️ Disclaimer

This project is for **educational and research purposes only** and does not constitute financial or investment advice. Predicting market direction is inherently uncertain, and past model performance does not guarantee future results.
