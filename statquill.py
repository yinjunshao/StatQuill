#!/usr/bin/env python3
"""
StatQuill CLI - Predictive Analytics Engine
A terminal-based prediction system for CSV/Excel data with AI-enhanced commentary.
"""

import os
import sys
import json
import math
import time
import argparse
import configparser
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, field
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# DEPENDENCY CHECK & AUTO-INSTALL PROMPT
# =============================================================================
REQUIRED_PACKAGES = {
    'pandas': 'pandas>=1.5.0',
    'numpy': 'numpy>=1.21.0',
    'requests': 'requests>=2.28.0',
    'rich': 'rich>=13.0.0',
    'openpyxl': 'openpyxl>=3.0.0',
    'scipy': 'scipy>=1.9.0',
}

missing = []
for pkg, spec in REQUIRED_PACKAGES.items():
    try:
        __import__(pkg)
    except ImportError:
        missing.append(spec)

if missing:
    print("\n[StatQuill] Missing dependencies detected.")
    print(f"Run: pip install {' '.join(missing)}")
    print("Or:  pip install statquill[all]\n")
    sys.exit(1)

import numpy as np
import pandas as pd
import requests
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.prompt import Prompt, Confirm
from rich.layout import Layout
from rich.syntax import Syntax
from rich.progress import Progress, SpinnerColumn, TextColumn
from scipy import stats
from scipy.linalg import svd, pinv

console = Console()

# =============================================================================
# CONFIGURATION MANAGEMENT
# =============================================================================
CONFIG_DIR = Path.home() / ".statquill"
CONFIG_FILE = CONFIG_DIR / "config.ini"
ENV_FILE = CONFIG_DIR / ".env"

class ConfigManager:
    """Handles .env / JSON configuration persistence."""

    def __init__(self):
        CONFIG_DIR.mkdir(exist_ok=True)
        self.config = configparser.ConfigParser()
        self._load()

    def _load(self):
        if CONFIG_FILE.exists():
            self.config.read(CONFIG_FILE)
        else:
            self.config['openrouter'] = {}
            self.config['preferences'] = {'model': 'anthropic/claude-3.5-sonnet'}

    def save(self):
        with open(CONFIG_FILE, 'w') as f:
            self.config.write(f)

    @property
    def api_key(self) -> Optional[str]:
        # Check env var first, then config file
        env_key = os.environ.get('OPENROUTER_API_KEY')
        if env_key:
            return env_key
        return self.config.get('openrouter', 'api_key', fallback=None)

    @api_key.setter
    def api_key(self, value: str):
        self.config.set('openrouter', 'api_key', value)
        self.save()

    @property
    def model(self) -> str:
        return self.config.get('openrouter', 'model', 
                               fallback='anthropic/claude-3.5-sonnet')

    @model.setter
    def model(self, value: str):
        self.config.set('openrouter', 'model', value)
        self.save()

    def is_configured(self) -> bool:
        return self.api_key is not None and len(self.api_key) > 20


# =============================================================================
# DATA PARSING ENGINE
# =============================================================================
@dataclass
class ColumnMeta:
    name: str
    dtype: str  # 'numeric', 'datetime', 'categorical', 'string'
    is_target: bool = False
    missing_count: int = 0
    unique_count: int = 0
    mean: Optional[float] = None
    std: Optional[float] = None
    min_val: Optional[float] = None
    max_val: Optional[float] = None

class DataParser:
    """Ingests and analyzes spreadsheet files."""

    SUPPORTED_FORMATS = {'.csv', '.xlsx', '.xls', '.tsv', '.txt'}
    TIME_HINTS = {'date', 'time', 'timestamp', 'datetime', 'year', 'month', 'day'}

    def __init__(self, filepath: str):
        self.filepath = Path(filepath)
        self.df: Optional[pd.DataFrame] = None
        self.columns: List[ColumnMeta] = []
        self.time_column: Optional[str] = None
        self.numeric_columns: List[str] = []
        self._load()

    def _load(self):
        if not self.filepath.exists():
            raise FileNotFoundError(f"File not found: {self.filepath}")

        ext = self.filepath.suffix.lower()
        if ext not in self.SUPPORTED_FORMATS:
            raise ValueError(f"Unsupported format: {ext}. Use: {self.SUPPORTED_FORMATS}")

        # Load with appropriate engine
        if ext == '.csv' or ext == '.txt':
            self.df = pd.read_csv(self.filepath, encoding='utf-8')
        elif ext in {'.xlsx', '.xls'}:
            self.df = pd.read_excel(self.filepath)
        elif ext == '.tsv':
            self.df = pd.read_csv(self.filepath, sep='\t', encoding='utf-8')

        self._infer_types()
        self._normalize_headers()

    def _infer_types(self):
        """Infer column types and detect time column."""
        for col in self.df.columns:
            series = self.df[col]
            meta = ColumnMeta(name=col, missing_count=series.isna().sum(),
                            unique_count=series.nunique())

            # Try datetime first
            if any(hint in col.lower() for hint in self.TIME_HINTS):
                try:
                    parsed = pd.to_datetime(series, errors='coerce')
                    if parsed.notna().sum() / len(series) > 0.8:
                        self.df[col] = parsed
                        meta.dtype = 'datetime'
                        self.time_column = col
                        self.columns.append(meta)
                        continue
                except:
                    pass

            # Try numeric
            coerced = pd.to_numeric(series, errors='coerce')
            if coerced.notna().sum() / len(series) > 0.8:
                self.df[col] = coerced
                meta.dtype = 'numeric'
                meta.mean = coerced.mean()
                meta.std = coerced.std()
                meta.min_val = coerced.min()
                meta.max_val = coerced.max()
                self.numeric_columns.append(col)
            else:
                # Check if categorical (few unique values)
                if series.nunique() / len(series) < 0.1 and series.nunique() < 50:
                    meta.dtype = 'categorical'
                else:
                    meta.dtype = 'string'

            self.columns.append(meta)

    def _normalize_headers(self):
        """Strip whitespace and handle duplicates."""
        self.df.columns = [c.strip() for c in self.df.columns]
        # Handle duplicates by appending index
        seen = {}
        new_cols = []
        for c in self.df.columns:
            if c in seen:
                seen[c] += 1
                new_cols.append(f"{c}_{seen[c]}")
            else:
                seen[c] = 0
                new_cols.append(c)
        self.df.columns = new_cols
        for i, col in enumerate(self.columns):
            col.name = self.df.columns[i]

    def get_numeric_df(self) -> pd.DataFrame:
        """Return only numeric columns, dropping NaNs."""
        return self.df[self.numeric_columns].dropna()

    def has_time_data(self) -> bool:
        return self.time_column is not None

    def summary(self) -> Dict:
        return {
            'rows': len(self.df),
            'columns': len(self.df.columns),
            'time_column': self.time_column,
            'numeric_columns': self.numeric_columns,
            'column_meta': [vars(c) for c in self.columns]
        }


# =============================================================================
# QUANTITATIVE MATHEMATICS ENGINE
# =============================================================================
@dataclass
class ModelResult:
    """Container for any model's output."""
    predictions: np.ndarray
    coefficients: Optional[np.ndarray] = None
    intercept: Optional[float] = None
    rss: float = 0.0
    sigma2: float = 0.0
    prediction_variance: np.ndarray = field(default_factory=lambda: np.array([]))
    standard_error: np.ndarray = field(default_factory=lambda: np.array([]))
    prediction_interval_lower: np.ndarray = field(default_factory=lambda: np.array([]))
    prediction_interval_upper: np.ndarray = field(default_factory=lambda: np.array([]))
    cv: np.ndarray = field(default_factory=lambda: np.array([]))
    model_type: str = "unknown"
    target_column: str = ""
    feature_columns: List[str] = field(default_factory=list)

class AutoregressiveModel:
    """AR(p) model for time-series prediction."""

    def __init__(self, max_lag: int = 5):
        self.max_lag = max_lag
        self.phi: np.ndarray = None
        self.c: float = 0.0
        self.sigma2: float = 0.0
        self.p: int = 0

    def fit(self, series: np.ndarray) -> ModelResult:
        """Fit AR model with AIC-based order selection."""
        series = np.asarray(series).flatten()
        n = len(series)

        best_aic = np.inf
        best_result = None

        for p in range(1, min(self.max_lag + 1, n // 3)):
            try:
                # Build lag matrix
                X = np.zeros((n - p, p))
                y = series[p:]
                for i in range(p):
                    X[:, i] = series[p - i - 1:n - i - 1]

                # OLS via SVD for stability
                beta, residuals, rank, s = np.linalg.lstsq(
                    np.column_stack([np.ones(len(X)), X]), y, rcond=None
                )

                c = beta[0]
                phi = beta[1:]
                y_pred = np.column_stack([np.ones(len(X)), X]) @ beta
                rss = np.sum((y - y_pred) ** 2)
                sigma2 = rss / (len(y) - p - 1) if len(y) > p + 1 else rss

                # AIC = n*ln(sigma2) + 2*k
                aic = len(y) * np.log(sigma2 + 1e-10) + 2 * (p + 1)

                if aic < best_aic:
                    best_aic = aic
                    best_result = (p, c, phi, X, y, y_pred, rss, sigma2)
            except:
                continue

        if best_result is None:
            raise ValueError("Could not fit AR model to data")

        self.p, self.c, self.phi, X, y, y_pred, rss, self.sigma2 = best_result

        # Calculate uncertainty for in-sample predictions
        XtX_inv = np.linalg.pinv(X.T @ X)
        pred_var = np.array([1 + x @ XtX_inv @ x for x in X]) * self.sigma2
        se = np.sqrt(pred_var)
        df = len(y) - self.p - 1
        t_crit = stats.t.ppf(0.975, max(df, 1))

        return ModelResult(
            predictions=y_pred,
            coefficients=self.phi,
            intercept=self.c,
            rss=rss,
            sigma2=self.sigma2,
            prediction_variance=pred_var,
            standard_error=se,
            prediction_interval_lower=y_pred - t_crit * se,
            prediction_interval_upper=y_pred + t_crit * se,
            cv=se / np.abs(y_pred),
            model_type="AR({self.p})",
            target_column="time_series"
        )

    def predict_next(self, series: np.ndarray, steps: int = 1) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Predict next value(s) with uncertainty."""
        series = np.asarray(series).flatten()
        predictions = []
        variances = []

        # Use last p values
        history = list(series[-self.p:])

        for _ in range(steps):
            pred = self.c + sum(self.phi[i] * history[-(i+1)] for i in range(self.p))
            predictions.append(pred)
            # Variance accumulates for multi-step
            var = self.sigma2 * (1 + sum(self.phi[i]**2 for i in range(self.p)))
            variances.append(var)
            history.append(pred)

        preds = np.array(predictions)
        se = np.sqrt(variances)
        df = len(series) - self.p - 1
        t_crit = stats.t.ppf(0.975, max(df, 1))

        return preds, preds - t_crit * se, preds + t_crit * se


class MatrixLinearRegression:
    """Multi-axis matrix linear regression with full uncertainty quantification."""

    def __init__(self, regularization: float = 1e-5):
        self.reg = regularization
        self.beta: np.ndarray = None
        self.XtX_inv: np.ndarray = None
        self.sigma2: float = 0.0
        self.feature_cols: List[str] = []
        self.target_col: str = ""
        self.X_mean: np.ndarray = None
        self.X_std: np.ndarray = None
        self.y_mean: float = 0.0
        self.y_std: float = 1.0

    def fit(self, df: pd.DataFrame, target_col: str, feature_cols: List[str]) -> ModelResult:
        """Fit multivariate linear regression."""
        self.target_col = target_col
        self.feature_cols = feature_cols

        # Extract and clean data
        data = df[feature_cols + [target_col]].dropna()
        X_raw = data[feature_cols].values
        y = data[target_col].values

        # Standardize for numerical stability
        self.X_mean = X_raw.mean(axis=0)
        self.X_std = X_raw.std(axis=0) + 1e-10
        X = (X_raw - self.X_mean) / self.X_std

        self.y_mean = y.mean()
        self.y_std = y.std() + 1e-10
        y_norm = (y - self.y_mean) / self.y_std

        # Add intercept
        X_design = np.column_stack([np.ones(len(X)), X])

        # Ridge-regularized normal equation for stability
        XtX = X_design.T @ X_design
        XtX += np.eye(XtX.shape[0]) * self.reg

        # Solve via SVD for maximum stability
        self.XtX_inv = np.linalg.pinv(XtX)
        self.beta = self.XtX_inv @ X_design.T @ y_norm

        # Predictions (in normalized space)
        y_pred_norm = X_design @ self.beta
        y_pred = y_pred_norm * self.y_std + self.y_mean

        # Residuals and variance
        residuals = y - y_pred
        n, k = len(y), len(feature_cols)
        self.sigma2 = np.sum(residuals**2) / max(n - k - 1, 1)

        # Prediction variance for each point
        pred_var = np.array([x @ self.XtX_inv @ x for x in X_design]) * self.sigma2 * (self.y_std**2)
        se = np.sqrt(np.maximum(pred_var, 0))

        # 95% Prediction intervals
        df = max(n - k - 1, 1)
        t_crit = stats.t.ppf(0.975, df)

        return ModelResult(
            predictions=y_pred,
            coefficients=self.beta[1:],  # exclude intercept
            intercept=self.beta[0] * self.y_std + self.y_mean,
            rss=np.sum(residuals**2),
            sigma2=self.sigma2,
            prediction_variance=pred_var,
            standard_error=se,
            prediction_interval_lower=y_pred - t_crit * se,
            prediction_interval_upper=y_pred + t_crit * se,
            cv=se / np.abs(y_pred + 1e-10),
            model_type="Matrix-OLS",
            target_column=target_col,
            feature_columns=feature_cols
        )

    def predict(self, x_new: np.ndarray) -> Tuple[float, float, float, float]:
        """Predict single row: returns (prediction, lower, upper, cv)."""
        x_norm = (x_new - self.X_mean) / self.X_std
        x_design = np.concatenate([[1], x_norm])

        pred_norm = x_design @ self.beta
        pred = pred_norm * self.y_std + self.y_mean

        var = x_design @ self.XtX_inv @ x_design * self.sigma2 * (self.y_std**2)
        se = np.sqrt(max(var, 0))

        df = max(len(self.X_mean) - len(self.feature_cols) - 1, 1)
        t_crit = stats.t.ppf(0.975, df)

        return pred, pred - t_crit * se, pred + t_crit * se, se / abs(pred + 1e-10)


class CovarianceEngine:
    """Computes and analyzes covariance matrices."""

    @staticmethod
    def compute(df: pd.DataFrame) -> Tuple[np.ndarray, List[str], np.ndarray]:
        """Returns (cov_matrix, column_names, correlation_matrix)."""
        numeric_df = df.select_dtypes(include=[np.number]).dropna()
        cols = list(numeric_df.columns)
        cov = numeric_df.cov().values
        corr = numeric_df.corr().values
        return cov, cols, corr

    @staticmethod
    def find_collinear_pairs(corr: np.ndarray, cols: List[str], threshold: float = 0.9) -> List[Tuple[str, str, float]]:
        """Find highly correlated pairs to warn about multicollinearity."""
        pairs = []
        n = len(cols)
        for i in range(n):
            for j in range(i+1, n):
                if abs(corr[i, j]) > threshold:
                    pairs.append((cols[i], cols[j], corr[i, j]))
        return sorted(pairs, key=lambda x: abs(x[2]), reverse=True)


# =============================================================================
# AI ENHANCEMENT ENGINE (OpenRouter)
# =============================================================================
class AIEnhancer:
    """Sends raw statistics to OpenRouter for humanized commentary."""

    API_URL = "https://openrouter.ai/api/v1/chat/completions"

    def __init__(self, api_key: str, model: str):
        self.api_key = api_key
        self.model = model
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://statquill.local",
            "X-Title": "StatQuill CLI"
        }

    def generate_commentary(self, stats_payload: Dict, context: str = "") -> str:
        """Send stats to AI and return humanized commentary."""

        system_prompt = """You are StatQuill, an expert data analyst. Analyze the provided statistical summary and prediction results. Provide:
1. A brief executive summary (2-3 sentences)
2. Key trends and patterns observed
3. Notable risks or uncertainties
4. Actionable insights based on the predictions
Be concise, professional, and data-driven. Use markdown formatting."""

        user_content = f"""## Statistical Analysis Results

```json
{json.dumps(stats_payload, indent=2, default=str)}
```

{"## Domain Context\n" + context if context else ""}

Please provide your analysis."""

        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content}
            ],
            "temperature": 0.3,
            "max_tokens": 1500
        }

        try:
            response = requests.post(
                self.API_URL,
                headers=self.headers,
                json=payload,
                timeout=60
            )
            response.raise_for_status()
            data = response.json()
            return data['choices'][0]['message']['content']
        except requests.exceptions.RequestException as e:
            return f"[AI Commentary unavailable: {str(e)}]"
        except KeyError:
            return "[AI Commentary unavailable: Unexpected response format]"


# =============================================================================
# TERMINAL DISPLAY ENGINE
# =============================================================================
class DisplayEngine:
    """Rich terminal visualizations."""

    def __init__(self):
        self.console = Console()

    def show_banner(self):
        banner = """
   ███████╗████████╗ █████╗ ████████╗ ██████╗ ██╗   ██╗██╗██╗     
   ██╔════╝╚══██╔══╝██╔══██╗╚══██╔══╝██╔═══██╗██║   ██║██║██║     
   ███████╗   ██║   ███████║   ██║   ██║   ██║██║   ██║██║██║     
   ╚════██║   ██║   ██╔══██║   ██║   ██║▄▄ ██║██║   ██║██║██║     
   ███████║   ██║   ██║  ██║   ██║   ╚██████╔╝╚██████╔╝██║███████╗
   ╚══════╝   ╚═╝   ╚═╝  ╚═╝   ╚═╝    ╚══▀▀═╝  ╚═════╝ ╚═╝╚══════╝
        Predictive Analytics CLI v1.0
        """
        self.console.print(Panel(banner, style="bold cyan", border_style="cyan"))

    def show_data_summary(self, parser: DataParser):
        table = Table(title="Dataset Overview", show_header=True, header_style="bold magenta")
        table.add_column("Property", style="cyan")
        table.add_column("Value", style="green")

        table.add_row("File", str(parser.filepath))
        table.add_row("Rows", str(len(parser.df)))
        table.add_row("Columns", str(len(parser.df.columns)))
        table.add_row("Time Column", parser.time_column or "None detected")
        table.add_row("Numeric Columns", ", ".join(parser.numeric_columns) or "None")

        self.console.print(table)

        # Column details
        col_table = Table(title="Column Analysis", show_header=True, header_style="bold magenta")
        col_table.add_column("Column", style="cyan")
        col_table.add_column("Type", style="yellow")
        col_table.add_column("Missing", style="red")
        col_table.add_column("Mean/Mode", style="green")

        for meta in parser.columns:
            if meta.dtype == 'numeric':
                mean_str = f"{meta.mean:.4f}" if meta.mean else "N/A"
            else:
                mean_str = f"{meta.unique_count} unique"
            col_table.add_row(meta.name, meta.dtype, str(meta.missing_count), mean_str)

        self.console.print(col_table)

    def show_model_results(self, result: ModelResult):
        table = Table(title=f"Model Results: {result.model_type}", show_header=True)
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green")

        table.add_row("Target", result.target_column)
        table.add_row("RSS", f"{result.rss:.4f}")
        table.add_row("σ² (Error Variance)", f"{result.sigma2:.6f}")
        table.add_row("Mean SE", f"{np.mean(result.standard_error):.4f}")
        table.add_row("Mean CV", f"{np.mean(result.cv):.4f}")

        if result.coefficients is not None and len(result.feature_columns) > 0:
            coef_table = Table(title="Coefficients (β)", show_header=True)
            coef_table.add_column("Feature", style="cyan")
            coef_table.add_column("Weight", style="green")
            for feat, coef in zip(result.feature_columns, result.coefficients):
                coef_table.add_row(feat, f"{coef:.6f}")
            self.console.print(coef_table)

        self.console.print(table)

    def show_covariance(self, cov: np.ndarray, cols: List[str], corr: np.ndarray):
        if len(cols) > 10:
            self.console.print("[yellow]Covariance matrix too large to display (>10 columns). Showing top correlations.[/]")
            engine = CovarianceEngine()
            pairs = engine.find_collinear_pairs(corr, cols, 0.5)
            if pairs:
                table = Table(title="Top Correlations", show_header=True)
                table.add_column("Variable A", style="cyan")
                table.add_column("Variable B", style="cyan")
                table.add_column("Correlation", style="green")
                for a, b, r in pairs[:10]:
                    color = "red" if abs(r) > 0.9 else "yellow" if abs(r) > 0.7 else "green"
                    table.add_row(a, b, f"[{color}]{r:.4f}[/{color}]")
                self.console.print(table)
            return

        table = Table(title="Covariance Matrix", show_header=True)
        table.add_column("", style="bold")
        for c in cols:
            table.add_column(c[:8], justify="right")

        for i, row_col in enumerate(cols):
            row = [row_col[:12]]
            for j in range(len(cols)):
                val = cov[i, j]
                row.append(f"{val:.4f}")
            table.add_row(*row)

        self.console.print(table)

    def show_ai_commentary(self, text: str):
        self.console.print(Panel(text, title="AI Analysis", border_style="green", style="white"))

    def show_prediction_table(self, predictions: List[Dict]):
        table = Table(title="Predictions (Sorted by Lowest CV)", show_header=True)
        table.add_column("Target", style="cyan")
        table.add_column("Prediction", style="green")
        table.add_column("95% Lower", style="yellow")
        table.add_column("95% Upper", style="yellow")
        table.add_column("CV", style="red")
        table.add_column("Confidence", style="blue")

        for pred in sorted(predictions, key=lambda x: x['cv']):
            conf = "High" if pred['cv'] < 0.1 else "Medium" if pred['cv'] < 0.3 else "Low"
            conf_color = "green" if pred['cv'] < 0.1 else "yellow" if pred['cv'] < 0.3 else "red"
            table.add_row(
                pred['target'],
                f"{pred['value']:.4f}",
                f"{pred['lower']:.4f}",
                f"{pred['upper']:.4f}",
                f"{pred['cv']:.4f}",
                f"[{conf_color}]{conf}[/{conf_color}]"
            )

        self.console.print(table)


# =============================================================================
# INTERACTIVE PREDICTION LOOP
# =============================================================================
class PredictionLoop:
    """Interactive CLI for entering partial data and getting predictions."""

    def __init__(self, parser: DataParser, display: DisplayEngine):
        self.parser = parser
        self.display = display
        self.models: Dict[str, Any] = {}
        self.results: Dict[str, ModelResult] = {}

    def run(self):
        """Main interactive loop."""
        console = self.display.console

        # Train models on all numeric columns
        numeric_df = self.parser.get_numeric_df()
        if len(numeric_df.columns) < 2:
            console.print("[red]Need at least 2 numeric columns for prediction.[/]")
            return

        console.print("\n[bold cyan]Training models on all numeric columns...[/]")

        with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}")) as progress:
            task = progress.add_task("Fitting models...", total=len(numeric_df.columns))

            for target in numeric_df.columns:
                features = [c for c in numeric_df.columns if c != target]
                try:
                    model = MatrixLinearRegression()
                    result = model.fit(numeric_df, target, features)
                    self.models[target] = model
                    self.results[target] = result
                except Exception as e:
                    console.print(f"[yellow]Warning: Could not fit model for {target}: {e}[/]")
                progress.advance(task)

        # Interactive input loop
        while True:
            console.print("\n[bold green]═" * 60)
            console.print("[bold]PREDICTION MODE[/]")
            console.print("Enter known values. Leave blank to predict. Type 'q' to quit.")
            console.print("═" * 60 + "[/]")

            # Show current data as reference
            console.print(f"\n[dim]Available columns: {', '.join(numeric_df.columns)}[/]")
            console.print(f"[dim]Data range: {len(numeric_df)} rows[/]")

            user_inputs = {}
            for col in numeric_df.columns:
                val = Prompt.ask(f"  {col}", default="", show_default=False)
                if val.lower() == 'q':
                    return
                if val.strip():
                    try:
                        user_inputs[col] = float(val)
                    except ValueError:
                        console.print(f"[red]Invalid number for {col}, skipping.[/]")

            if not user_inputs:
                console.print("[yellow]No values entered. Please provide at least one known value.[/]")
                continue

            # Predict missing columns
            predictions = []
            for target in numeric_df.columns:
                if target in user_inputs:
                    continue

                model = self.models.get(target)
                if not model:
                    continue

                # Build feature vector from known values
                feature_values = []
                for feat in model.feature_cols:
                    if feat in user_inputs:
                        feature_values.append(user_inputs[feat])
                    else:
                        # Use mean if unknown
                        feature_values.append(numeric_df[feat].mean())

                try:
                    pred, lower, upper, cv = model.predict(np.array(feature_values))
                    predictions.append({
                        'target': target,
                        'value': pred,
                        'lower': lower,
                        'upper': upper,
                        'cv': cv
                    })
                except Exception as e:
                    console.print(f"[red]Error predicting {target}: {e}[/]")

            if predictions:
                self.display.show_prediction_table(predictions)
            else:
                console.print("[yellow]No predictions could be generated.[/]")

            if not Confirm.ask("\nMake another prediction?"):
                break


# =============================================================================
# TIME-SERIES PREDICTION LOOP
# =============================================================================
class TimeSeriesLoop:
    """Predict next time step using AR model."""

    def __init__(self, parser: DataParser, display: DisplayEngine):
        self.parser = parser
        self.display = display

    def run(self):
        console = self.display.console
        time_col = self.parser.time_column
        numeric_df = self.parser.get_numeric_df()

        console.print(f"\n[bold cyan]Time Series Mode[/] (Time column: {time_col})")

        # Let user pick which numeric column to predict
        targets = [c for c in numeric_df.columns if c != time_col]
        if not targets:
            console.print("[red]No valid target columns found.[/]")
            return

        console.print("\nAvailable targets:")
        for i, t in enumerate(targets, 1):
            console.print(f"  {i}. {t}")

        choice = Prompt.ask("Select target column (number or name)", default=targets[0])
        target = targets[int(choice)-1] if choice.isdigit() and 1 <= int(choice) <= len(targets) else choice

        if target not in targets:
            console.print(f"[red]Invalid target: {target}[/]")
            return

        series = numeric_df[target].dropna().values

        console.print(f"\n[bold]Fitting AR model to {len(series)} observations...[/]")

        try:
            ar_model = AutoregressiveModel(max_lag=min(10, len(series)//4))
            result = ar_model.fit(series)
            self.display.show_model_results(result)

            # Predict next steps
            steps = int(Prompt.ask("How many future steps to predict?", default="1"))
            preds, lower, upper = ar_model.predict_next(series, steps)

            pred_table = Table(title=f"Future Predictions for '{target}'", show_header=True)
            pred_table.add_column("Step", style="cyan")
            pred_table.add_column("Prediction", style="green")
            pred_table.add_column("95% Lower", style="yellow")
            pred_table.add_column("95% Upper", style="yellow")

            for i, (p, l, u) in enumerate(zip(preds, lower, upper), 1):
                pred_table.add_row(f"+{i}", f"{p:.4f}", f"{l:.4f}", f"{u:.4f}")

            console.print(pred_table)

        except Exception as e:
            console.print(f"[red]AR model failed: {e}[/]")
            console.print("[yellow]Falling back to Matrix Regression...[/]")
            # Fallback to matrix regression
            loop = PredictionLoop(self.parser, self.display)
            loop.run()


# =============================================================================
# MAIN APPLICATION
# =============================================================================
class StatQuillApp:
    """Main StatQuill CLI application."""

    def __init__(self):
        self.config = ConfigManager()
        self.display = DisplayEngine()
        self.ai: Optional[AIEnhancer] = None

    def setup(self):
        """Initial configuration wizard."""
        self.display.show_banner()
        console = self.display.console

        console.print("\n[bold yellow]Welcome! Let's set up StatQuill.[/]")

        api_key = Prompt.ask("Enter your OpenRouter API Key", password=True)
        if len(api_key) < 20:
            console.print("[red]That doesn't look like a valid API key.[/]")
            if not Confirm.ask("Continue anyway?"):
                sys.exit(0)

        self.config.api_key = api_key

        model = Prompt.ask(
            "Enter OpenRouter Model",
            default="anthropic/claude-3.5-sonnet"
        )
        self.config.model = model

        console.print("[green]Configuration saved!\n[/]")

    def analyze_file(self, filepath: str, context: str = ""):
        """Full analysis pipeline for a single file."""
        console = self.display.console

        # Parse data
        with console.status("[bold green]Parsing data file..."):
            parser = DataParser(filepath)

        self.display.show_data_summary(parser)

        # Compute covariance
        numeric_df = parser.get_numeric_df()
        if len(numeric_df.columns) >= 2:
            cov, cols, corr = CovarianceEngine.compute(numeric_df)
            self.display.show_covariance(cov, cols, corr)

            # Warn about multicollinearity
            collinear = CovarianceEngine.find_collinear_pairs(corr, cols, 0.9)
            if collinear:
                console.print("[bold red]⚠ High multicollinearity detected:[/]")
                for a, b, r in collinear:
                    console.print(f"  {a} ↔ {b}: r={r:.3f}")

        # Build stats payload for AI
        stats_payload = {
            "file": str(parser.filepath),
            "summary": parser.summary(),
            "numeric_summary": {
                col: {
                    "mean": float(numeric_df[col].mean()),
                    "std": float(numeric_df[col].std()),
                    "min": float(numeric_df[col].min()),
                    "max": float(numeric_df[col].max())
                }
                for col in numeric_df.columns
            }
        }

        # AI Enhancement
        if self.config.is_configured() and self.ai is None:
            self.ai = AIEnhancer(self.config.api_key, self.config.model)

        if self.ai:
            with console.status("[bold green]Consulting AI for insights..."):
                commentary = self.ai.generate_commentary(stats_payload, context)
            self.display.show_ai_commentary(commentary)

        # Enter prediction mode
        console.print("\n[bold]Analysis complete. Entering prediction mode...[/]")

        if parser.has_time_data() and len(numeric_df.columns) >= 2:
            if Confirm.ask("Data contains time. Run time-series prediction?", default=True):
                ts_loop = TimeSeriesLoop(parser, self.display)
                ts_loop.run()
            else:
                pred_loop = PredictionLoop(parser, self.display)
                pred_loop.run()
        else:
            pred_loop = PredictionLoop(parser, self.display)
            pred_loop.run()

    def run(self):
        """Main entry point."""
        parser = argparse.ArgumentParser(
            description="StatQuill - Predictive Analytics CLI",
            formatter_class=argparse.RawDescriptionHelpFormatter,
            epilog="""
Examples:
  statquill data.csv
  statquill sales.xlsx --context "Q3 retail sales forecast"
  statquill --setup
            """
        )
        parser.add_argument('file', nargs='?', help='Path to data file (CSV, XLSX, etc.)')
        parser.add_argument('--context', '-c', default='', help='Domain context for AI analysis')
        parser.add_argument('--setup', action='store_true', help='Run configuration wizard')
        parser.add_argument('--model', '-m', help='Override OpenRouter model')

        args = parser.parse_args()

        self.display.show_banner()

        # Setup mode
        if args.setup:
            self.setup()
            return

        # Check configuration
        if not self.config.is_configured():
            console = self.display.console
            console.print("[yellow]OpenRouter API key not configured.[/]")
            if Confirm.ask("Run setup now?"):
                self.setup()
            else:
                console.print("[red]StatQuill requires an API key for AI features.[/]")
                console.print("Run with --setup or set OPENROUTER_API_KEY env var.")
                sys.exit(1)

        if args.model:
            self.config.model = args.model

        # File mode
        if args.file:
            self.analyze_file(args.file, args.context)
        else:
            # Interactive file selection
            console = self.display.console
            filepath = Prompt.ask("Enter path to data file")
            context = Prompt.ask("Optional context/description", default="")
            self.analyze_file(filepath, context)


# =============================================================================
# ENTRY POINT
# =============================================================================
def main():
    app = StatQuillApp()
    app.run()

if __name__ == "__main__":
    main()
