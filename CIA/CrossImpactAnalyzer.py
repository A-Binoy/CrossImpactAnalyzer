import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
import yfinance as yf
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os

class OrderFlowAnalysis:
    def __init__(self, symbols, start_date, end_date, root_path):
        self.symbols = symbols
        self.start_date = start_date
        self.end_date = end_date
        self.data = {}
        self.ofi_metrics = {}
        self.pca_components = None
        self.root = root_path

    def check_time(self):
        if (self.end_date - self.start_date).days > 8:
            raise ValueError("Time to long for 1m data")

    def fetch_data(self):
        for symbol in self.symbols:
            try:
                ticker = yf.Ticker(symbol)
                # Get 1-minute interval data
                df = ticker.history(start=self.start_date, end=self.end_date, interval='1m')
                df['volume_imbalance'] = (df['Volume'] *np.where(df['Close'] > df['Open'], 1, -1))
                df['returns'] = df['Close'].pct_change()

                df['forward_1m'] = df['returns'].shift(-1) # forward rates for prediction
                df['forward_5m'] = df['Close'].shift(-5).div(df['Close']) - 1

                self.data[symbol] = df

            except Exception as e:
                print(f"Error fetching data for {symbol}: {str(e)}")

    def compute_ofi_proxy(self, window_sizes=[1, 5, 10, 15, 20]):
        for symbol in self.symbols:
            df = self.data[symbol].copy()

            # Calculate OFI proxies at different time windows
            for window in window_sizes:
                df[f'ofi_{window}m'] = (
                    df['volume_imbalance']
                    .rolling(window=window)
                    .sum()
                    .fillna(0)
                )

            self.ofi_metrics[symbol] = df

    def integrate_ofi_metrics(self): # integrate OFI metreic
        for symbol in self.symbols:
            df = self.ofi_metrics[symbol]
            ofi_cols = [col for col in df.columns if col.startswith('ofi_')]

            # Standardize the features
            scaler = StandardScaler()
            ofi_standardized = scaler.fit_transform(df[ofi_cols])

            pca = PCA(n_components=1) # Apply PCA
            integrated_ofi = pca.fit_transform(ofi_standardized)

            df['integrated_ofi'] = integrated_ofi # Add integrated ofi
            self.ofi_metrics[symbol] = df

    def analyze_cross_impact(self):
        combined_data = pd.DataFrame()

        for symbol in self.symbols:
            df = self.ofi_metrics[symbol]
            ofi_col = f'ofi_{symbol}'
            return_col = f'return_{symbol}'
            combined_data[ofi_col] = df['integrated_ofi']
            combined_data[return_col] = df['returns']

        combined_data = combined_data.dropna() #Remove NaN

        # Result dictionary
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
            df = self.ofi_metrics[target_symbol]
            X = pd.DataFrame()
            for sym in self.symbols:
                X[f'ofi_{sym}_lag1'] = self.ofi_metrics[sym]['integrated_ofi'].shift(1)
            y_1m = df['forward_1m']
            y_5m = df['forward_5m']


            valid_idx = ~(X.isna().any(axis=1) | y_1m.isna() | y_5m.isna()) # Remove NaN
            X = X[valid_idx]
            y_1m = y_1m[valid_idx]
            y_5m = y_5m[valid_idx]


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
    def print_results(self, filename="impact_results.txt"):
        with open(f"{self.root}/{filename}", "w") as file:
            # Redirect both to console and file
            def write_and_print(text=""):
                print(text)
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
        r2_1m = []
        r2_5m = []

        for symbol in self.analysis.symbols:
            r2_1m.append(self.analysis.impact_results['predictive'][symbol]['1m']['r2_score'])
            r2_5m.append(self.analysis.impact_results['predictive'][symbol]['5m']['r2_score'])

        # Create grouped bar plot
        fig = go.Figure(data=[
            go.Bar(name='1-min horizon', x=self.analysis.symbols, y=r2_1m),
            go.Bar(name='5-min horizon', x=self.analysis.symbols, y=r2_5m)
        ])

        fig.update_layout(
            title='Predictive Power (R² Score) by Horizon',
            barmode='group',
            xaxis_title='Stock',
            yaxis_title='R² Score'
        )

        fig.write_html(os.path.join(self.save_path, 'predictive_power.html'))

    def plot_cumulative_impact(self):
        fig = make_subplots(rows=2, cols=1,
                          subplot_titles=('Cumulative Self-Impact',
                                        'Cumulative Cross-Impact'))

        for symbol in self.analysis.symbols:
            df = self.analysis.ofi_metrics[symbol]

            # Cumulative self-impact
            cum_returns = df['returns'].cumsum()
            cum_ofi = df['integrated_ofi'].cumsum()

            fig.add_trace(
                go.Scatter(x=cum_ofi, y=cum_returns,
                          mode='lines', name=symbol),
                row=1, col=1
            )

            # Cumulative cross-impact (using AAPL as example target)
            if symbol != 'AAPL':
                target_returns = self.analysis.ofi_metrics['AAPL']['returns'].cumsum()
                fig.add_trace(
                    go.Scatter(x=cum_ofi, y=target_returns,
                              mode='lines', name=f'{symbol}->AAPL'),
                    row=2, col=1
                )

        fig.update_layout(height=800, title_text="Cumulative Impact Analysis")
        fig.write_html(os.path.join(self.save_path, 'cumulative_impact.html'))

if __name__ == "__main__":
    # Define parameters
    symbols = ['AAPL', 'AMGN', 'TSLA', 'JPM', 'XOM']
    end_date = datetime.now()
    start_date = end_date - timedelta(days=8)  # Get 8 days worth of 1m data
    root_path = "./CIA"
    # Initialize and run analysis
    analysis = OrderFlowAnalysis(symbols, start_date, end_date, root_path)

    # Run the analysis pipeline
    analysis.fetch_data()
    analysis.compute_ofi_proxy()
    analysis.integrate_ofi_metrics()
    analysis.analyze_cross_impact()
    analysis.print_results()
    viz = OrderFlowVisualization(analysis, f"{root_path}/Viz")
    viz.create_all_visualizations()


