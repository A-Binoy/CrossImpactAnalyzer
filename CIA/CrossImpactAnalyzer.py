import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.linear_model import LinearRegression
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import yfinance as yf
import seaborn as sns
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import time
import os
import ofi_tools as ot

class OrderFlowAnalysis:
    def __init__(self, symbols, start_date, end_date, root_path):
        self.symbols = symbols
        self.start_date = start_date
        self.end_date = end_date
        self.data = {}
        self.ofi_metrics = {}
        self.pca_components = None
        self.root = root_path
        self.pnl = pd.Series(dtype=float)

    def input_data(self, choice):
        if choice == "lb":
            return self.lb_fetch_data()
        elif choice == "yf":
            return self.yf_fetch_data()
        else:
            print("Invalid input.")
            return None

    def lb_fetch_data(self):
        data_path = os.path.join(self.root, "data")
        for symbol in self.symbols:
            message_file = None
            orderbook_file = None
            for file in os.listdir(data_path):
                if symbol in file and "message" in file:
                    message_file = os.path.join(data_path, file)
                if symbol in file and "orderbook" in file:
                    orderbook_file = os.path.join(data_path, file)
            if not message_file or not orderbook_file:
                raise FileNotFoundError(f"Could not find LOBSTER files for {symbol} in {data_path}.")
            data = ot.read_lobster_files(message_file, orderbook_file, 10)

            df = pd.DataFrame([
                {
                    "time": row.time,
                    "type": row.type,
                    "price": row.price,
                    "direction": row.direction
                }
                for row in data
            ])
            df["returns"] = df["price"].pct_change().fillna(0)

            ofi_list = ot.compute_ofi(data)  # Returns list of lists
            ofi_df = pd.DataFrame(ofi_list, columns=[f"ofi_level_{i+1}" for i in range(10)])
            df = df.iloc[1:].reset_index(drop=True)
            df = pd.concat([df, ofi_df], axis=1)

            df["forward_1m"] = df["returns"].shift(-1)
            df["forward_5m"] = df["returns"].rolling(5).sum().shift(-5)
            self.data[symbol] = df
            self.ofi_metrics[symbol] = df
        return self.data

    def yf_fetch_data(self, standardization='zscore', fill_method='ffill', depth_level=5):
        self.check_time()
        interval = '1m'
        all_data = {}

        for symbol in self.symbols:
            df = yf.download(symbol, start=self.start_date, end=self.end_date, interval=interval, progress=False)
            time.sleep(2)
            if df.empty:
                print(f"No data found for {symbol}. Skipping.")
                continue

            # Clean column names
            df.columns = df.columns.str.strip().str.lower()
            df.rename(columns={'close': 'price'}, inplace=True)
            df['returns'] = df['price'].pct_change().fillna(0)
            df['volume_imbalance'] = df['volume'].rolling(depth_level, min_periods=1).sum() \
                                                - df['volume'].rolling(depth_level, min_periods=1).mean()

            if fill_method == 'ffill':
                df.fillna(method='ffill', inplace=True)
            elif fill_method == 'zero':
                df.fillna(0, inplace=True)
            elif fill_method == 'drop':
                df.dropna(inplace=True)

            if standardization == 'zscore':
                scaler = StandardScaler()
                df['volume_imbalance'] = scaler.fit_transform(df[['volume_imbalance']])
            elif standardization == 'minmax':
                scaler = MinMaxScaler()
                df['volume_imbalance'] = scaler.fit_transform(df[['volume_imbalance']])

            df['forward_1m'] = df['returns'].shift(-1)
            df['forward_5m'] = df['returns'].rolling(5).sum().shift(-5)

            all_data[symbol] = df
        self.data = all_data

    def gen_returns(self):
        self.returns = pd.DataFrame({sym: self.ofi_metrics[sym]['returns'] for sym in self.symbols})

    def check_time(self):
        if (self.end_date - self.start_date).days > 8:
            raise ValueError("Time to long for 1m data")

    def compute_ofi_proxy(self, window_sizes=[1, 5]):
        for symbol in self.symbols:
            df = self.data[symbol].copy()

            # Calculate OFI proxies at different time windows
            for window in window_sizes:
                ofi_list = df['volume_imbalance'].tolist()
                rolling_sum = ot.rolling_ofi_sum(ofi_list, window)
                df[f'ofi_{window}m'] = rolling_sum

            self.ofi_metrics[symbol] = df

    def integrate_ofi_metrics(self):
        for symbol in self.symbols:
            df = self.ofi_metrics[symbol]
            ofi_cols = [col for col in df.columns if col.startswith('ofi_')]

            # Integrating ofi
            scaler = StandardScaler()
            ofi_standardized = scaler.fit_transform(df[ofi_cols])
            pca = PCA(n_components=1)
            integrated_ofi = pca.fit_transform(ofi_standardized)

            df['integrated_ofi'] = integrated_ofi
            self.ofi_metrics[symbol] = df

    def analyze_cross_impact(self):
        combined_data = pd.DataFrame()

        for symbol in self.symbols:
            df = self.ofi_metrics[symbol]
            ofi_col = f'ofi_{symbol}'
            return_col = f'return_{symbol}'
            combined_data[ofi_col] = df['integrated_ofi']
            combined_data[return_col] = df['returns']

        combined_data = combined_data.dropna()

        self.impact_results = {
            'contemporaneous': {},
            'predictive': {}
        }

        for target_symbol in self.symbols:
            target_returns = f'return_{target_symbol}'
            features = [f'ofi_{sym}' for sym in self.symbols] # Prepare features
            X = combined_data[features]
            y = combined_data[target_returns]

            model = LinearRegression()
            model.fit(X, y)
            self.impact_results['contemporaneous'][target_symbol] = {
                'coefficients': dict(zip(self.symbols, model.coef_)),
                'r2_score': model.score(X, y)
            }

        # Analyze predictive impact (lagged)
        for target_symbol in self.symbols:
            X = pd.DataFrame(index=self.ofi_metrics[self.symbols[0]].index)
            for sym in self.symbols:
                X[f'ofi_{sym}_lag1'] = self.ofi_metrics[sym]['integrated_ofi'].shift(1)

            df_target = self.ofi_metrics[target_symbol]
            y_1m = df_target['forward_1m']
            y_5m = df_target['forward_5m']

            # Align X and y to the same index first
            X, y_1m = X.align(y_1m, join='inner', axis=0)
            X, y_5m = X.align(y_5m, join='inner', axis=0)

            # Then drop rows with any NaN
            valid_idx = ~(X.isna().any(axis=1) | y_1m.isna() | y_5m.isna())
            X = X.loc[valid_idx]
            y_1m = y_1m.loc[valid_idx]
            y_5m = y_5m.loc[valid_idx]

            model_1m = LinearRegression().fit(X, y_1m)
            model_5m = LinearRegression().fit(X, y_5m)
            self.impact_results['predictive'][target_symbol] = {
                '1m': {
                    'coefficients': dict(zip(self.symbols, model_1m.coef_)),
                    'r2_score': model_1m.score(X, y_1m)
                },
                '5m': {
                    'coefficients': dict(zip(self.symbols, model_5m.coef_)),
                    'r2_score': model_5m.score(X, y_5m)
                }
            }

    def export_data_and_ofi_metrics(self, folder_name="exported_data"):
        export_path = os.path.join(self.root, folder_name)
        os.makedirs(export_path, exist_ok=True)

        for symbol in self.symbols:
            data_df = self.data[symbol].copy()
            data_df.to_csv(os.path.join(export_path, f"{symbol}_raw_data.csv"))

            ofi_df = self.ofi_metrics[symbol].copy()
            ofi_df.to_csv(os.path.join(export_path, f"{symbol}_ofi_metrics.csv"))

    def print_results(self, filename="impact_results.txt"):
        with open(f"{self.root}/{filename}", "w") as file:
            def write_and_print(text=""):
                file.write(text + "\n")

            write_and_print("\nContemporaneous Impact Results:")
            for target in self.impact_results['contemporaneous']:
                write_and_print(f"\nTarget Stock: {target}")
                write_and_print(f"R² Score: {self.impact_results['contemporaneous'][target]['r2_score']:.4f}")
                write_and_print("Impact Coefficients:")
                for sym, coef in self.impact_results['contemporaneous'][target]['coefficients'].items():
                    write_and_print(f"{sym}: {coef:.6f}")

            write_and_print("\nPredictive Impact Results:")
            for target in self.impact_results['predictive']:
                write_and_print(f"\nTarget Stock: {target}")
                write_and_print("1-minute horizon:")
                write_and_print(f"R² Score: {self.impact_results['predictive'][target]['1m']['r2_score']:.4f}")
                write_and_print("5-minute horizon:")
                write_and_print(f"R² Score: {self.impact_results['predictive'][target]['5m']['r2_score']:.4f}")

        print(f"\nResults saved to {filename}")

    def gen_signals(self, threshold_quantile=0.8):
        signals = {}
        for target in self.symbols:
            df = self.ofi_metrics[target]
            threshold = df['integrated_ofi'].quantile(threshold_quantile)
            signals[target] = df['integrated_ofi'].shift(1).apply(lambda x: 1 if x > threshold else 0)
        signals_df = pd.DataFrame(signals)
        return signals_df

    def export_signals_and_pnl(self, signals, filename="signals_and_pnl.csv") :
        results_df = signals.copy()
        results_df.columns = [f"Signal_{col}" for col in results_df.columns]

        results_df["Cumulative_PnL"] = self.pnl

        results_df = results_df.reset_index()
        results_df.rename(columns={"index": "Timestamp"}, inplace=True)

        results_df.to_csv(os.path.join(self.root, filename), index=False)
        print(f"Signals and PnL exported")

    def simulate_trading(self, threshold_quantile):
        self.gen_returns()
        signals = self.gen_signals(threshold_quantile)
        pnl_series = pd.Series(index=self.returns.index, dtype=float)

        for timestep in signals.index:
            current_signals = signals.loc[timestep]
            active_longs = current_signals[current_signals > 0].index.tolist()

            if active_longs:
                weights = pd.Series(1 / len(active_longs), index=active_longs)
                pnl_step = (self.returns.loc[timestep, active_longs] * weights).sum()
            else:
                pnl_step = 0
            pnl_series[timestep] = pnl_step

        pnl = pnl_series.fillna(0).cumsum()
        self.pnl = pnl
        self.export_signals_and_pnl(signals)

        return pnl

class OrderFlowVisualization:
    def __init__(self, analysis, save_path):
        self.analysis = analysis
        self.save_path = save_path
        os.makedirs(save_path, exist_ok=True)

    def create_all_visualizations(self):
        self.plot_ofi_time_series()
        self.plot_cross_impact_heatmap()
        self.plot_predictive_power()
        self.plot_cumulative_impact()

    def plot_ofi_time_series(self):
        # Plot OFI time series for each stock
        plt.figure(figsize=(15, 10))

        for i, symbol in enumerate(self.analysis.symbols, 1):
            plt.subplot(len(self.analysis.symbols), 1, i)
            df = self.analysis.ofi_metrics[symbol]
            plt.plot(df.index, df['integrated_ofi'],
                    label='Integrated OFI', color='blue', alpha=0.7)
            ax2 = plt.gca().twinx()
            ax2.plot(df.index, df['returns'],
                    label='Returns', color='red', alpha=0.5)

            plt.title(f'{symbol} - OFI and Returns')
            plt.legend()

        plt.tight_layout()
        plt.savefig(os.path.join(self.save_path, 'ofi_time_series.png'))
        plt.close()

    def plot_cross_impact_heatmap(self):
        # Prepare contemporaneous impact data
        impact_matrix = np.zeros((len(self.analysis.symbols), len(self.analysis.symbols)))

        for i, target in enumerate(self.analysis.symbols):
            coeffs = self.analysis.impact_results['contemporaneous'][target]['coefficients']
            for j, source in enumerate(self.analysis.symbols):
                impact_matrix[i, j] = coeffs[source]

        # Create heatmap
        plt.figure(figsize=(12, 8))
        sns.heatmap(impact_matrix,
                   xticklabels=self.analysis.symbols,
                   yticklabels=self.analysis.symbols,
                   annot=True, cmap='RdYlBu', center=0)

        plt.title('Cross-Impact Coefficients Heatmap')
        plt.xlabel('Source Stock')
        plt.ylabel('Target Stock')
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_path, 'cross_impact_heatmap.png'))
        plt.close()

    def plot_predictive_power(self):
        plt.figure(figsize=(12, 6))

        r2_1m = []
        r2_5m = []

        for symbol in self.analysis.symbols:
            r2_1m.append(self.analysis.impact_results['predictive'][symbol]['1m']['r2_score'])
            r2_5m.append(self.analysis.impact_results['predictive'][symbol]['5m']['r2_score'])

        x = np.arange(len(self.analysis.symbols))
        width = 0.35

        plt.bar(x - width/2, r2_1m, width, label='1-timestep horizon')
        plt.bar(x + width/2, r2_5m, width, label='5-timestep horizon')

        plt.xlabel('Stock')
        plt.ylabel('R² Score')
        plt.title('Predictive Power (R² Score) by Horizon')
        plt.xticks(x, self.analysis.symbols)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_path, 'predictive_power.png'))
        plt.close()

    def plot_cumulative_impact(self):
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 12))

        # --- Plot cumulative self-impact ---
        ax1.set_title('Cumulative Self-Impact (OFI vs Returns)')
        for symbol in self.analysis.symbols:
            df = self.analysis.ofi_metrics[symbol]
            cum_returns = df['returns'].cumsum()
            cum_ofi = df['integrated_ofi'].cumsum()

            ax1.plot(cum_ofi, cum_returns, label=symbol)

        ax1.set_xlabel('Cumulative OFI (Integrated)')
        ax1.set_ylabel('Cumulative Returns')
        ax1.legend()
        ax1.grid(False)
        ax2.set_title('Cumulative Cross-Impact on AAPL')
        target_df = self.analysis.ofi_metrics['AAPL']
        target_returns = target_df['returns'].cumsum()

        for symbol in self.analysis.symbols:
            if symbol != 'AAPL':
                df = self.analysis.ofi_metrics[symbol]
                cum_ofi = df['integrated_ofi'].cumsum()

                # Align the indexes so lengths match for plotting
                cum_ofi_aligned, target_returns_aligned = cum_ofi.align(target_returns, join='inner')

                ax2.plot(cum_ofi_aligned, target_returns_aligned, label=f'{symbol} OFI -> AAPL Returns')

        ax2.set_xlabel('Cumulative OFI (Integrated)')
        ax2.set_ylabel('AAPL Cumulative Returns')
        ax2.legend()
        ax2.grid(False)

        plt.tight_layout()
        plt.savefig(os.path.join(self.save_path, 'cumulative_impact.png'))
        plt.close()

    def plot_cumulative_pnl(self):
        plt.figure(figsize=(10, 5))
        (self.analysis.pnl).plot(title="Cumulative PnL from OFI-based Strategy (%)")
        plt.xlabel("Timepoints since Market open ")
        plt.ylabel("Cumulative Return (%)")
        plt.grid(True)
        plt.gca().xaxis.set_major_locator(plt.MaxNLocator(10))
        plt.gcf().autofmt_xdate()
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_path, 'cumulative_pnl.png'))
        plt.close()

    # 2️⃣ PnL vs. Integrated OFI Threshold
    def plot_pnl_vs_ofi(self):
        for symbol in self.analysis.symbols:
            df = self.analysis.ofi_metrics[symbol]
            plt.figure(figsize=(10, 5))
            plt.scatter(df['integrated_ofi'], self.analysis.pnl.loc[df.index], alpha=0.5)
            plt.xlabel("Integrated OFI (Lagged)")
            plt.ylabel("Cumulative PnL")
            plt.title(f"Relationship Between OFI Strength and PnL ({symbol})")
            plt.grid(True)
            plt.tight_layout()
            plt.savefig(os.path.join(self.save_path, f'pnl_vs_ofi_{symbol}.png'))
            plt.close()

    # 3️⃣ PnL Attribution by Symbol
    def plot_pnl_by_symbol(self):
        signals = self.analysis.gen_signals()
        pnl_by_symbol = (self.analysis.returns * signals).cumsum().iloc[-1]
        plt.figure(figsize=(10, 6))
        pnl_by_symbol.plot(kind='bar', title='Final PnL by Symbol')
        plt.ylabel('Cumulative Return')
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_path, 'pnl_by_symbol.png'))
        plt.close()

    # 4️⃣ PnL During Strong vs. Weak OFI Regimes
    def plot_pnl_high_vs_low_ofi(self):
        signals = self.analysis.gen_signals()
        pnl_by_symbol = (self.analysis.returns * signals).cumsum()

        for symbol in self.analysis.symbols:
            df = self.analysis.ofi_metrics[symbol]
            pnl = pnl_by_symbol[symbol]
            df, pnl = df.align(pnl, join='inner', axis=0)
            high_ofi = df['integrated_ofi'] > df['integrated_ofi'].quantile(0.8)
            pnl_high_ofi = pnl[high_ofi]
            pnl_low_ofi = pnl[~high_ofi]

            plt.figure(figsize=(10, 6))
            plt.plot(pnl_high_ofi.index, pnl_high_ofi, label="High OFI Periods")
            plt.plot(pnl_low_ofi.index, pnl_low_ofi, label="Low OFI Periods")
            plt.legend()
            plt.title(f"PnL in High vs. Low OFI Periods ({symbol})")
            plt.xlabel("Time")
            plt.ylabel("Cumulative PnL")
            plt.tight_layout()
            plt.savefig(os.path.join(self.save_path, f'pnl_high_vs_low_ofi_{symbol}.png'))
            plt.close()

    def plot_pnl_visualizations(self):
        self.plot_cumulative_pnl()
        self.plot_pnl_vs_ofi()
        self.plot_pnl_by_symbol()
        self.plot_pnl_high_vs_low_ofi()

if __name__ == "__main__":
    # Define parameters
    symbols = ['AAPL', 'AMZN', 'GOOG', 'INTC', 'MSFT']
    end_date = datetime.now()
    start_date = end_date - timedelta(days=7)  # Get 8 days worth of 1m data
    root_path = "."
    standardization='zscore'
    fill_method='ffill'
    depth_level = 5
    para =  0.7
    # Initialize and run analysis

    analysis = OrderFlowAnalysis(symbols, start_date, end_date, root_path)
    print(para)
    # Run the analysis pipeline
    dfs = analysis.input_data("lb")
    # analysis.compute_ofi_proxy()
    analysis.integrate_ofi_metrics()
    analysis.analyze_cross_impact()
    analysis.print_results()
    analysis.simulate_trading(para)
    analysis.export_data_and_ofi_metrics()
    viz = OrderFlowVisualization(analysis, f"{root_path}/Viz")
    viz.plot_pnl_visualizations()
    viz.create_all_visualizations()
