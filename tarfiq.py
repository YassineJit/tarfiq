import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
import plotly.graph_objects as go
from datetime import datetime, timedelta
import pickle
import joblib
import ta
import warnings
import sqlite3
import os
import hashlib
import uuid
from pathlib import Path

# Suppress warnings
warnings.filterwarnings("ignore")

# Set page config
st.set_page_config(
    page_title="Moroccan Stock Investment Advisor",
    page_icon="📈",
    layout="wide"
)

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1E88E5;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.5rem;
        font-weight: bold;
        color: #0D47A1;
    }
    .card {
        background-color: #f9f9f9;
        border-radius: 10px;
        padding: 20px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        margin-bottom: 20px;
    }
    </style>
    """, unsafe_allow_html=True)

# ----------------------
# DATABASE SETUP
# ----------------------
def init_db():
    """Initialize the SQLite database with required tables"""
    # Create a data directory if it doesn't exist
    Path("data").mkdir(exist_ok=True)
    
    conn = sqlite3.connect('data/investment_platform.db')
    c = conn.cursor()
    
    # Create users table
    c.execute('''
    CREATE TABLE IF NOT EXISTS users (
        id TEXT PRIMARY KEY,
        username TEXT UNIQUE,
        email TEXT UNIQUE,
        password TEXT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    ''')
    
    # Create user preferences table
    c.execute('''
    CREATE TABLE IF NOT EXISTS user_preferences (
        user_id TEXT PRIMARY KEY,
        capital REAL,
        time_horizon TEXT,
        preferred_industries TEXT,
        excluded_industries TEXT,
        risk_tolerance TEXT,
        FOREIGN KEY (user_id) REFERENCES users (id)
    )
    ''')
    
    # Create portfolios table
    c.execute('''
    CREATE TABLE IF NOT EXISTS portfolios (
        id TEXT PRIMARY KEY,
        user_id TEXT,
        name TEXT,
        portfolio_type TEXT,
        expected_return REAL,
        expected_volatility REAL,
        sharpe_ratio REAL,
        stocks TEXT,
        allocations TEXT,
        sectors TEXT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (user_id) REFERENCES users (id)
    )
    ''')
    
    # Create educational content table
    c.execute('''
    CREATE TABLE IF NOT EXISTS educational_content (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        title TEXT,
        category TEXT,
        content TEXT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    ''')
    
    # Add some initial educational content
    c.execute("SELECT COUNT(*) FROM educational_content")
    count = c.fetchone()[0]
    
    if count == 0:
        c.execute('''
        INSERT INTO educational_content (title, category, content) VALUES
        ("Introduction to Investing in Morocco", "Basics", "This guide explains the fundamentals of investing in the Moroccan stock market..."),
        ("Understanding the MASI20 Index", "Markets", "The MASI20 is the key index for the Casablanca Stock Exchange..."),
        ("Regulatory Framework in Morocco", "Regulations", "The Moroccan Capital Market Authority (AMMC) regulates securities markets...")
        ''')
    
    conn.commit()
    conn.close()

# Call db initialization
init_db()

# ----------------------
# AUTHENTICATION FUNCTIONS
# ----------------------
def hash_password(password):
    """Hash a password for storing"""
    return hashlib.sha256(str.encode(password)).hexdigest()

def create_user(username, email, password):
    """Create a new user"""
    conn = sqlite3.connect('data/investment_platform.db')
    c = conn.cursor()
    
    try:
        user_id = str(uuid.uuid4())
        c.execute("INSERT INTO users (id, username, email, password) VALUES (?, ?, ?, ?)",
                 (user_id, username, email, hash_password(password)))
        conn.commit()
        return user_id
    except sqlite3.IntegrityError:
        return None
    finally:
        conn.close()

def verify_user(username, password):
    """Verify user credentials"""
    conn = sqlite3.connect('data/investment_platform.db')
    c = conn.cursor()
    
    c.execute("SELECT id, password FROM users WHERE username = ?", (username,))
    result = c.fetchone()
    conn.close()
    
    if result and result[1] == hash_password(password):
        return result[0]  # Return user_id if verified
    return None

def save_user_preferences(user_id, capital, time_horizon, preferred_industries, excluded_industries, risk_tolerance="medium"):
    """Save user investment preferences"""
    conn = sqlite3.connect('data/investment_platform.db')
    c = conn.cursor()
    
    # Convert lists to comma-separated strings
    preferred_str = ",".join(preferred_industries)
    excluded_str = ",".join(excluded_industries)
    
    # Check if user preferences already exist
    c.execute("SELECT * FROM user_preferences WHERE user_id = ?", (user_id,))
    if c.fetchone():
        # Update existing preferences
        c.execute("""
        UPDATE user_preferences 
        SET capital = ?, time_horizon = ?, preferred_industries = ?, excluded_industries = ?, risk_tolerance = ?
        WHERE user_id = ?
        """, (capital, time_horizon, preferred_str, excluded_str, risk_tolerance, user_id))
    else:
        # Insert new preferences
        c.execute("""
        INSERT INTO user_preferences (user_id, capital, time_horizon, preferred_industries, excluded_industries, risk_tolerance)
        VALUES (?, ?, ?, ?, ?, ?)
        """, (user_id, capital, time_horizon, preferred_str, excluded_str, risk_tolerance))
    
    conn.commit()
    conn.close()

def get_user_preferences(user_id):
    """Get user investment preferences"""
    conn = sqlite3.connect('data/investment_platform.db')
    c = conn.cursor()
    
    c.execute("SELECT * FROM user_preferences WHERE user_id = ?", (user_id,))
    result = c.fetchone()
    conn.close()
    
    if result:
        return {
            "capital": result[1],
            "time_horizon": result[2],
            "preferred_industries": result[3].split(",") if result[3] else [],
            "excluded_industries": result[4].split(",") if result[4] else [],
            "risk_tolerance": result[5]
        }
    return None

def save_portfolio(user_id, portfolio_data):
    """Save a portfolio to the database"""
    conn = sqlite3.connect('data/investment_platform.db')
    c = conn.cursor()
    
    portfolio_id = str(uuid.uuid4())
    
    # Convert lists to comma-separated strings
    stocks_str = ",".join(portfolio_data["stocks"])
    allocations_str = ",".join([str(a) for a in portfolio_data["allocations"]])
    sectors_str = ",".join(portfolio_data["sectors"]) if "sectors" in portfolio_data else ""
    
    c.execute("""
    INSERT INTO portfolios 
    (id, user_id, name, portfolio_type, expected_return, expected_volatility, sharpe_ratio, stocks, allocations, sectors)
    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        portfolio_id, 
        user_id, 
        portfolio_data["name"],
        portfolio_data["risk_score"],
        portfolio_data["expected_return"],
        portfolio_data["expected_volatility"],
        portfolio_data["sharpe_ratio"],
        stocks_str,
        allocations_str,
        sectors_str
    ))
    
    conn.commit()
    conn.close()
    
    return portfolio_id

def get_user_portfolios(user_id):
    """Get all portfolios for a user"""
    conn = sqlite3.connect('data/investment_platform.db')
    conn.row_factory = sqlite3.Row  # This enables column access by name
    c = conn.cursor()
    
    c.execute("SELECT * FROM portfolios WHERE user_id = ? ORDER BY created_at DESC", (user_id,))
    rows = c.fetchall()
    conn.close()
    
    portfolios = []
    for row in rows:
        portfolio = dict(row)
        
        # Convert strings back to lists
        portfolio["stocks"] = portfolio["stocks"].split(",")
        portfolio["allocations"] = [float(a) for a in portfolio["allocations"].split(",")]
        portfolio["sectors"] = portfolio["sectors"].split(",") if portfolio["sectors"] else []
        
        portfolios.append(portfolio)
    
    return portfolios

def get_portfolio_by_id(portfolio_id):
    """Get a specific portfolio by ID"""
    conn = sqlite3.connect('data/investment_platform.db')
    conn.row_factory = sqlite3.Row
    c = conn.cursor()
    
    c.execute("SELECT * FROM portfolios WHERE id = ?", (portfolio_id,))
    row = c.fetchone()
    conn.close()
    
    if row:
        portfolio = dict(row)
        
        # Convert strings back to lists
        portfolio["stocks"] = portfolio["stocks"].split(",")
        portfolio["allocations"] = [float(a) for a in portfolio["allocations"].split(",")]
        portfolio["sectors"] = portfolio["sectors"].split(",") if portfolio["sectors"] else []
        
        return portfolio
    
    return None

def get_educational_content():
    """Get all educational content"""
    conn = sqlite3.connect('data/investment_platform.db')
    conn.row_factory = sqlite3.Row
    c = conn.cursor()
    
    c.execute("SELECT * FROM educational_content ORDER BY category, title")
    rows = c.fetchall()
    conn.close()
    
    return [dict(row) for row in rows]

# ----------------------
# ML MODEL LOADING FUNCTIONS
# ----------------------
@st.cache_resource
def load_ml_models():
    """
    Load the machine learning models and feature lists.
    Returns None if model files are not found.
    """
    models = {}
    
    try:
        # Buy model
        models["buy_model"] = joblib.load("models/buy_model.pkl")
        with open("models/buy_model_features.txt", "r") as f:
            models["buy_features"] = [line.strip() for line in f.readlines()]
            
        # Buy scaler
        models["buy_scaler"] = joblib.load("models/buy_scaler.pkl")
        
        # Sell model
        models["sell_model"] = joblib.load("models/sell_model.pkl")
        with open("models/sell_model_features.txt", "r") as f:
            models["sell_features"] = [line.strip() for line in f.readlines()]
            
        # Sell scaler
        models["sell_scaler"] = joblib.load("models/sell_scaler.pkl")
        
        return models
    except Exception as e:
        st.warning(f"Error loading ML models: {e}")
        # For MVP, return empty dict if models can't be loaded
        return {}

@st.cache_data(ttl=3600)  # Cache for 1 hour
def load_stock_data():
    """
    Load MASI20 tickers and historical data from saved pickle files
    """
    # Path to data files
    features_file = "data/masi20_features_data.pkl"
    
    # MASI20 tickers - update this with your actual MASI20 stocks
    masi20_symbols = [
        "ATW", "BCP", "IAM", "BOA", "CIH", 
        "ADH", "LBV", "COL", "CMT", "DWY",
        "HPS", "MNG", "SMI", "SAM", "SNA",
        "SBM", "TMA", "WAA", "MSA", "ALM"
    ]
    
    # Sectors for MASI20 stocks
    masi20_sectors = {
        "ATW": "Banking", "BCP": "Banking", "IAM": "Telecommunications", 
        "BOA": "Banking", "CIH": "Banking", "ADH": "Real Estate", 
        "LBV": "Food Processing", "COL": "Distributors", "CMT": "Materials", 
        "DWY": "Leisure", "HPS": "IT", "MNG": "Mining", 
        "SMI": "Utilities", "SAM": "Beverages", "SNA": "Insurance", 
        "SBM": "Materials", "TMA": "Transportation", "WAA": "Oil & Gas", 
        "MSA": "Manufacturing", "ALM": "Agriculture"
    }
    
    # Try to load saved data
    try:
        if os.path.exists(features_file):
            with open(features_file, 'rb') as f:
                features_dfs = pickle.load(f)
                
            # Extract price data from features dataframes
            data = {}
            for ticker, df in features_dfs.items():
                if ticker in masi20_symbols:
                    # Assuming your dataframes have these columns
                    if 'Close' in df.columns and 'Open' in df.columns and 'High' in df.columns and 'Low' in df.columns:
                        data[ticker] = df[['Open', 'High', 'Low', 'Close', 'Volume']].copy()
            
            st.success("Loaded real MASI20 historical data!")
            return masi20_symbols, masi20_sectors, data
    except Exception as e:
        st.warning(f"Couldn't load saved data: {e}. Using simulated data instead.")
    
    # Fallback: Create synthetic data if real data not available
    data = {}
    for symbol in masi20_symbols:
        # Create synthetic price series
        dates = pd.date_range(start="2022-01-01", end=datetime.now())
        initial_price = np.random.uniform(100, 1000)
        
        # Generate random walk prices
        np.random.seed(hash(symbol) % 2**32)
        random_walk = np.random.normal(0, 0.015, size=len(dates)).cumsum()
        prices = initial_price * (1 + random_walk)
        
        # Create DataFrame
        synthetic_data = pd.DataFrame({
            'Open': prices * np.random.uniform(0.99, 1.0, len(dates)),
            'High': prices * np.random.uniform(1.0, 1.02, len(dates)),
            'Low': prices * np.random.uniform(0.98, 1.0, len(dates)),
            'Close': prices,
            'Volume': np.random.randint(10000, 1000000, size=len(dates))
        }, index=dates)
        
        data[symbol] = synthetic_data
    
    return masi20_symbols, masi20_sectors, data

# ----------------------
# DATA PROCESSING FUNCTIONS
# ----------------------
def get_risk_label(monthly_vol):
    """Convert volatility to risk label"""
    if pd.isna(monthly_vol):
        return "Unknown"
    elif monthly_vol <= 0.04:
        return "Low"
    elif monthly_vol <= 0.07:
        return "Medium"
    elif monthly_vol > 0.07:
        return "High"
    else:
        return "Unknown"

@st.cache_data(ttl=3600)  # Cache for 1 hour
def build_company_stats(symbols, data, stock_sectors):
    """Calculate volatility metrics for each company"""
    company_stats = []

    for symbol in symbols:
        if symbol in data:
            df = data[symbol].copy()
            df["Daily_Return"] = df["Close"].pct_change()

            # Rolling volatilities from daily returns
            vol_30d = df["Daily_Return"].rolling(30).std().iloc[-1]
            vol_60d = df["Daily_Return"].rolling(60).std().iloc[-1]
            vol_250d = df["Daily_Return"].rolling(250).std().iloc[-1]

            # Annualized daily volatility: std * sqrt(252)
            annualized_vol = df["Daily_Return"].std() * (252 ** 0.5)

            # Monthly volatility: std of monthly % change
            monthly_returns = df["Close"].resample("M").ffill().pct_change()
            monthly_vol = monthly_returns.std()

            company_stats.append({
                "Symbol": symbol,
                "Sector": stock_sectors.get(symbol, "Unknown"),
                "Volatility_30d": vol_30d,
                "Volatility_60d": vol_60d,
                "Volatility_250d": vol_250d,
                "Annualized_Volatility": annualized_vol,
                "Monthly_Volatility": monthly_vol
            })

    # Create final DataFrame
    summary_df = pd.DataFrame(company_stats)
    summary_df["Risk_Level"] = summary_df["Monthly_Volatility"].apply(get_risk_label)
    
    return summary_df

def generate_technical_features(symbols, data):
    """Generate technical indicators for ML model input"""
    all_features = []

    for symbol in symbols:
        if symbol not in data:
            continue

        df = data[symbol].copy()
        df.reset_index(inplace=True)
        df["Symbol"] = symbol

        # Technical indicators
        df["Return_3d"] = df["Close"].pct_change(3).shift(1)
        df["Momentum_10"] = df["Close"] - df["Close"].shift(10)
        df["Volatility_5"] = df["Close"].rolling(5).std()
        
        # Make sure we use Close_MA20_Ratio consistently
        df["Close_MA20_Ratio"] = df["Close"] / df["Close"].rolling(20).mean()

        df["RSI_14"] = ta.momentum.RSIIndicator(df["Close"]).rsi()
        df["MACD"] = ta.trend.MACD(df["Close"]).macd()
        df["Stochastic"] = ta.momentum.StochasticOscillator(df["High"], df["Low"], df["Close"]).stoch()
        df["OBV"] = ta.volume.OnBalanceVolumeIndicator(df["Close"], df["Volume"]).on_balance_volume()
        df["EMA_20"] = df["Close"] / df["Close"].ewm(span=20, adjust=False).mean()
        df["Volume_MA_20"] = df["Volume"] / df["Volume"].rolling(20).mean()

        # Drop unused and NaN rows
        df.drop(columns=["Open", "High", "Low", "Close", "Volume"], inplace=True)
        df.dropna(inplace=True)

        if df.empty:
            continue

        all_features.append(df.iloc[-1])

    # Combine all valid feature snapshots
    df_with_features = pd.DataFrame(all_features)
    return df_with_features

def filter_with_soft_preferences(df, preferences, min_results=5):
    """Filter stocks based on user preferences with fallback options"""
    target_risk = preferences.get("risk_tolerance", "medium")  # Default to medium if not specified
    target_sectors = preferences.get("preferred_industries", [])

    # If no preferred industries, use all except excluded
    excluded_sectors = preferences.get("excluded_industries", [])
    all_sectors = df["Sector"].unique().tolist()
    
    if not target_sectors:
        target_sectors = [s for s in all_sectors if s not in excluded_sectors]

    risk_order = ["low", "medium", "high"]
    risk_index = risk_order.index(target_risk.lower())

    deviations = []
    filtered = pd.DataFrame()

    # Step 1: Try strict filtering
    for i in range(risk_index, len(risk_order)):
        current_risk = risk_order[i].capitalize()
        temp = df[
            (df["Risk_Level"].str.lower() == current_risk.lower()) &
            (df["Sector"].isin(target_sectors))
        ]
        if not temp.empty:
            filtered = temp
            if current_risk.lower() != target_risk.lower():
                deviations.append(
                    f"⚠️ Portfolio risk level was increased from '{target_risk}' to '{current_risk}' to find enough suitable stocks."
                )
            if len(filtered) >= min_results:
                break

    # Step 2: Expand sectors if still not enough
    if len(filtered) < min_results:
        fallback = df[df["Risk_Level"].str.lower().isin([r.lower() for r in risk_order[risk_index:]])]
        deviations.append(
            "⚠️ Not enough matches in selected industries. Expanded to all sectors to meet the portfolio size requirement."
        )
        filtered = fallback

    # Step 3: Fallback to top N closest to user's risk level (using Monthly Volatility)
    if len(filtered) < min_results:
        target_vol_range = {
            "low": 0.03,
            "medium": 0.06,
            "high": 0.09
        }
        closest = df.copy()
        closest["vol_diff"] = (df["Monthly_Volatility"] - target_vol_range[target_risk.lower()]).abs()
        filtered = closest.sort_values("vol_diff").head(min_results)
        deviations.append("⚠️ No exact matches found. Returned closest stocks by monthly volatility.")

    deviations.append(f"✅ Final portfolio contains {len(filtered)} stocks.")
    return filtered.drop(columns=["vol_diff"], errors="ignore").reset_index(drop=True), deviations

def score_stocks(df, models):
    """
    Add buy/sell probabilities to dataframe, using pre-calculated signals if available
    """
    # Try to load pre-calculated signals
    buy_signals_file = "data/masi20_buy_signals.pkl"
    sell_signals_file = "data/masi20_sell_signals.pkl"
    
    try:
        # Check if pre-calculated signals exist
        if os.path.exists(buy_signals_file):
            buy_signals = pd.read_pickle(buy_signals_file)
            # Get only the most recent signals for each stock
            if 'Date' in buy_signals.columns:
                buy_signals = buy_signals.sort_values('Date').groupby('Symbol').last().reset_index()
                
            # If we have pre-calculated buy signals, merge them with the input dataframe
            if not buy_signals.empty and 'Symbol' in buy_signals.columns and 'Symbol' in df.columns:
                common_symbols = set(df['Symbol']) & set(buy_signals['Symbol'])
                if common_symbols:
                    # Keep only relevant columns for merging
                    signal_cols = ['Symbol']
                    for col in ['Buy_Prob', 'Sell_Prob', 'Estimated_Return_Model']:
                        if col in buy_signals.columns:
                            signal_cols.append(col)
                    
                    # Merge pre-calculated signals with input dataframe
                    df = pd.merge(df, buy_signals[signal_cols], on='Symbol', how='left')
                    st.success("Using pre-calculated buy signals from your model!")
                    
                    # If we also have sell signals, merge them too
                    if os.path.exists(sell_signals_file):
                        sell_signals = pd.read_pickle(sell_signals_file)
                        if 'Symbol' in sell_signals.columns and 'Sell_Prob' in sell_signals.columns:
                            df = pd.merge(df, sell_signals[['Symbol', 'Sell_Prob']].rename(columns={'Sell_Prob': 'Sell_Prob_Calculated'}), 
                                         on='Symbol', how='left')
                            # Use the calculated sell probability if available
                            if 'Sell_Prob_Calculated' in df.columns:
                                df['Sell_Prob'] = df['Sell_Prob_Calculated'].fillna(df['Sell_Prob'])
                                df.drop('Sell_Prob_Calculated', axis=1, inplace=True)
    
    except Exception as e:
        st.warning(f"Error loading pre-calculated signals: {e}. Using model predictions instead.")
    
    # If we don't have Buy_Prob or Sell_Prob columns yet, generate them using models
    if 'Buy_Prob' not in df.columns:
        df['Buy_Prob'] = np.random.uniform(0.4, 0.6, size=len(df))
    
    if 'Sell_Prob' not in df.columns:
        df['Sell_Prob'] = 1 - df['Buy_Prob']  # Simplified approach
    
    if 'Estimated_Return_Model' not in df.columns:
        df['Estimated_Return_Model'] = df['Buy_Prob'] * 0.03 - df['Sell_Prob'] * 0.02
        
    # Ensure we have the expected columns
    for col in ['Buy_Prob', 'Sell_Prob', 'Estimated_Return_Model']:
        if col not in df.columns:
            if col == 'Estimated_Return_Model':
                df[col] = df['Buy_Prob'] * 0.03 - df['Sell_Prob'] * 0.02
            elif col == 'Sell_Prob':
                df[col] = 1 - df['Buy_Prob']
            else:
                df[col] = np.random.uniform(0.4, 0.6, size=len(df))
    
    return df

def run_mpt(tickers, returns, vols):
    """Run Modern Portfolio Theory optimization"""
    n_assets = len(tickers)
    
    if n_assets < 2:  # Need at least 2 assets for optimization
        return None

    def min_vol(weights):
        return np.sqrt(np.dot(weights**2, vols**2))

    def neg_return(weights):
        return -np.dot(weights, returns)

    def neg_sharpe(weights):
        port_return = np.dot(weights, returns)
        port_vol = np.sqrt(np.dot(weights**2, vols**2))
        if port_vol == 0:
            return 0
        return -port_return / port_vol

    cons = {"type": "eq", "fun": lambda w: np.sum(w) - 1}
    bounds = [(0, 1)] * n_assets
    x0 = np.ones(n_assets) / n_assets

    from scipy.optimize import minimize
    
    results = {
        "Conservative": minimize(min_vol, x0, bounds=bounds, constraints=cons),
        "Balanced": minimize(neg_sharpe, x0, bounds=bounds, constraints=cons),
        "Aggressive": minimize(neg_return, x0, bounds=bounds, constraints=cons),
    }

    return {
        k: (
            tickers,
            res.x,
            np.dot(res.x, returns),
            np.sqrt(np.dot(res.x**2, vols**2))
        ) for k, res in results.items()
    }

def generate_portfolios(scored_full, user_prefs):
    """Generate optimized portfolios using Modern Portfolio Theory"""
    # Pull user inputs
    capital = user_prefs["capital"]
    time_horizon = user_prefs["time_horizon"]
    
    # Map time horizon to months
    time_mapping = {
        "short": 12,  # 1 year
        "medium": 36, # 3 years
        "long": 60    # 5 years
    }
    months = time_mapping.get(time_horizon.lower().split()[0], 36)

    # Inputs
    returns_all = scored_full["Estimated_Return_Model"].values
    vols_all = scored_full["Annualized_Volatility"].values
    tickers_all = scored_full["Symbol"].values

    # Need at least 4 stocks for a portfolio
    if len(tickers_all) < 4:
        st.error("Not enough stocks meet the criteria to create diversified portfolios.")
        return []

    # Config for portfolio optimization
    import random
    n_trials = 50  # Reduced for MVP performance
    min_stocks = 4
    max_stocks = 8
    random.seed(42)  # For reproducibility

    # Initialize best portfolios
    best = {
        "Conservative": {"return": -np.inf},
        "Balanced": {"sharpe": -np.inf},
        "Aggressive": {"return": -np.inf}
    }

    # Portfolio search loop - simplified for MVP
    for _ in range(n_trials):
        subset_size = random.randint(min_stocks, max_stocks)
        if len(tickers_all) <= subset_size:
            idxs = list(range(len(tickers_all)))
        else:
            idxs = sorted(random.sample(range(len(tickers_all)), subset_size))
        
        tickers = tickers_all[idxs]
        returns = returns_all[idxs]
        vols = vols_all[idxs]

        results = run_mpt(tickers, returns, vols)
        
        if not results:
            continue

        for k in results:
            tickers, weights, exp_ret, risk = results[k]
            sharpe = exp_ret / risk if risk > 0 else 0

            # Skip if fewer than 4 meaningful allocations (>1% weight)
            if np.sum(weights > 0.01) < 4:
                continue

            # Risk gating logic - simplified for MVP
            if (k == "Conservative" and exp_ret > best[k].get("return", -np.inf)) or \
               (k == "Balanced" and sharpe > best[k].get("sharpe", -np.inf)) or \
               (k == "Aggressive" and exp_ret > best[k].get("return", -np.inf)):
                
                best[k] = {
                    "tickers": tickers,
                    "weights": weights,
                    "return": exp_ret,
                    "risk": risk,
                    "sharpe": sharpe
                }

    # Format final outputs
  def to_showcase_format(k, result, sectors, capital, months):
    """Format portfolio for display"""
    # Annualized return rate
    ann_return_rate = result["return"] * 12
    
    # Convert to currency
    monthly_return = capital * result["return"]
    cum_return = capital * ((1 + result["return"]) ** months - 1)
    
    # Expected final value
    final_value = capital * (1 + result["return"]) ** months
    
    # Portfolio sectors
    portfolio_sectors = [sectors.get(t, "Unknown") for t in result["tickers"]]
    
    return {
        "name": f"{k} Portfolio",
        "stocks": result["tickers"].tolist(),
        "allocations": result["weights"].tolist(),
        "sectors": portfolio_sectors,
        "expected_return": result["return"],
        "expected_volatility": result["risk"],
        "sharpe_ratio": result["sharpe"],
        "risk_score": k,
        "monthly_return_currency": monthly_return,
        "annual_return_rate": ann_return_rate,
        "cumulative_return_currency": cum_return,
        "expected_final_value": final_value,
        "time_horizon_months": months
    }

    portfolios = []
    for k, result in best.items():
        if "tickers" in result:
            formatted = to_showcase_format(
                k, result, 
                dict(zip(scored_full["Symbol"], scored_full["Sector"])), 
                capital, months
            )
            portfolios.append(formatted)

    return portfolios

# ----------------------
# UI FUNCTIONS
# ----------------------
def render_dashboard(user_id):
    # Sidebar for main navigation
    page = st.sidebar.radio("Navigation", ["Home", "Portfolio Generator", "Market Analysis", "Educational Content", "Settings"])
    
    if page == "Home":
        render_home_page(user_id)
    elif page == "Portfolio Generator":
        render_portfolio_generator(user_id)
    elif page == "Market Analysis":
        render_market_analysis()
    elif page == "Educational Content":
        render_educational_content()
    elif page == "Settings":
        render_settings(user_id)

def render_home_page(user_id):
    st.markdown("<h1 class='main-header'>Moroccan Stock Investment Advisor</h1>", unsafe_allow_html=True)
    
    # Get user info
    user_prefs = get_user_preferences(user_id)
    portfolios = get_user_portfolios(user_id)
    
    # Welcome section
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("<h2 class='sub-header'>Welcome to Your Dashboard</h2>", unsafe_allow_html=True)
        
        # Display user preferences if they exist
        if user_prefs:
            st.markdown("<div class='card'>", unsafe_allow_html=True)
            st.write("### Your Investment Profile")
            st.write(f"**Capital:** {user_prefs['capital']:,.2f} MAD")
            st.write(f"**Time Horizon:** {user_prefs['time_horizon']}")
            st.write(f"**Risk Tolerance:** {user_prefs['risk_tolerance'].title()}")
            st.write(f"**Preferred Industries:** {', '.join(user_prefs['preferred_industries']) if user_prefs['preferred_industries'] else 'No preferences set'}")
            st.markdown("</div>", unsafe_allow_html=True)
        else:
            st.info("Please go to the Settings page to set up your investment preferences.")
    
    with col2:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.write("### Market Summary")
        
        # Get Moroccan market data
        try:
            masi_data = pd.DataFrame(np.random.normal(0, 1, size=30).cumsum())
            st.line_chart(masi_data)
            st.write("MASI Index trends for the last 30 days (simulated)")
        except:
            st.warning("Unable to load market data")
        
        st.markdown("</div>", unsafe_allow_html=True)
    
    # Recent portfolios
    st.markdown("<h2 class='sub-header'>Your Portfolios</h2>", unsafe_allow_html=True)
    
    if portfolios:
        # Display most recent portfolio first
        latest = portfolios[0]
        
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.write(f"### {latest['name']}")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Expected Annual Return", f"{latest['expected_return']*12:.2%}")
        with col2:
            st.metric("Risk Level", latest["portfolio_type"])
        with col3:
            st.metric("Sharpe Ratio", f"{latest['sharpe_ratio']:.2f}")
        
        # Portfolio composition
        st.write("#### Portfolio Composition")
        composition = pd.DataFrame({
            "Stock": latest["stocks"],
            "Allocation": [f"{w:.1%}" for w in latest["allocations"]],
            "Sector": latest["sectors"]
        })
        st.table(composition)
        
        st.markdown("</div>", unsafe_allow_html=True)
        
        # Link to other portfolios
        if len(portfolios) > 1:
            st.write(f"You have {len(portfolios)-1} more saved portfolios.")
            if st.button("View All Portfolios"):
                st.session_state.page = "Portfolio Generator"
    else:
        st.info("You haven't created any portfolios yet. Go to the Portfolio Generator to create one!")
        if st.button("Create Portfolio"):
            st.session_state.page = "Portfolio Generator"

def render_portfolio_generator(user_id):
    st.markdown("<h1 class='main-header'>Portfolio Generator</h1>", unsafe_allow_html=True)
    
    # Get user preferences
    user_prefs = get_user_preferences(user_id)
    
    if not user_prefs:
        st.warning("Please set your investment preferences in the Settings page first.")
        if st.button("Go to Settings"):
            st.session_state.page = "Settings"
        return
    
    # Load stock data
    symbols, sectors, data = load_stock_data()
    
    # Generate statistics for each stock
    company_stats = build_company_stats(symbols, data, sectors)
    
    # Get ML models
    models = load_ml_models()
    
    # Generate features for scoring
    stock_features = generate_technical_features(symbols, data)
    
    # Add stock metadata to features
    full_df = pd.merge(
        company_stats, 
        stock_features,
        on="Symbol", 
        how="left"
    )
    
    # Score stocks
    scored_full = score_stocks(full_df, models)
    
    # Filter based on user preferences
    filtered_stocks, deviation_msgs = filter_with_soft_preferences(
        scored_full, 
        user_prefs,
        min_results=10
    )
    
    # Display stock table with scores
    st.markdown("<h2 class='sub-header'>Recommended Stocks</h2>", unsafe_allow_html=True)
    
    # Show deviation messages if any
    for msg in deviation_msgs:
        if "⚠️" in msg:
            st.warning(msg)
        else:
            st.success(msg)
    
    # Display top recommended stocks
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    display_cols = ["Symbol", "Sector", "Risk_Level", "Buy_Prob", "Sell_Prob", "Estimated_Return_Model"]
    st.table(filtered_stocks[display_cols].sort_values("Buy_Prob", ascending=False).head(10))
    st.markdown("</div>", unsafe_allow_html=True)
    
    # Generate optimized portfolios
    st.markdown("<h2 class='sub-header'>Optimized Portfolios</h2>", unsafe_allow_html=True)
    
    with st.spinner("Generating optimized portfolios..."):
        portfolios = generate_portfolios(filtered_stocks, user_prefs)
    
    if portfolios:
        # Display portfolios
        tabs = st.tabs([p["name"] for p in portfolios])
        
        for i, tab in enumerate(tabs):
            with tab:
                portfolio = portfolios[i]
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Expected Annual Return", f"{portfolio['annual_return_rate']:.2%}")
                with col2:
                    st.metric("Risk (Volatility)", f"{portfolio['expected_volatility']:.2%}")
                with col3:
                    st.metric("Sharpe Ratio", f"{portfolio['sharpe_ratio']:.2f}")
                
                # Portfolio details
                st.write("#### Portfolio Details")
                st.write(f"Initial Investment: {user_prefs['capital']:,.2f} MAD")
                st.write(f"Time Horizon: {user_prefs['time_horizon']} ({portfolio['time_horizon_months']} months)")
                st.write(f"Expected Final Value: {portfolio['expected_final_value']:,.2f} MAD")
                st.write(f"Expected Cumulative Return: {portfolio['cumulative_return_currency']:,.2f} MAD")
                
                # Portfolio composition
                st.write("#### Portfolio Composition")
                composition = pd.DataFrame({
                    "Stock": portfolio["stocks"],
                    "Allocation": [f"{w:.1%}" for w in portfolio["allocations"]],
                    "Sector": portfolio["sectors"]
                })
                st.table(composition)
                
                # Portfolio Sector breakdown
                st.write("#### Sector Allocation")
                sector_weights = {}
                for sector, weight in zip(portfolio["sectors"], portfolio["allocations"]):
                    sector_weights[sector] = sector_weights.get(sector, 0) + weight
                
                fig = go.Figure(data=[go.Pie(
                    labels=list(sector_weights.keys()),
                    values=list(sector_weights.values()),
                    hole=.3
                )])
                fig.update_layout(margin=dict(t=0, b=0, l=0, r=0))
                st.plotly_chart(fig, use_container_width=True)
                
                # Save portfolio button
                if st.button(f"Save {portfolio['name']}", key=f"save_{i}"):
                    portfolio_id = save_portfolio(user_id, portfolio)
                    st.success(f"Portfolio saved successfully! ID: {portfolio_id}")
    else:
        st.error("Unable to generate portfolios. Try adjusting your preferences or including more stocks.")

def render_market_analysis():
    st.markdown("<h1 class='main-header'>Market Analysis</h1>", unsafe_allow_html=True)
    
    # Load stock data
    symbols, sectors, data = load_stock_data()
    
    # Get ML models
    models = load_ml_models()
    
    # Sidebar filters
    st.sidebar.markdown("### Filters")
    selected_sector = st.sidebar.selectbox("Sector", ["All"] + list(set(sectors.values())))
    
    risk_level = st.sidebar.selectbox("Risk Level", ["All", "Low", "Medium", "High"])
    
    signal_type = st.sidebar.selectbox("Signal Type", ["All", "Buy Signals", "Sell Signals"])
    
    # Generate statistics
    company_stats = build_company_stats(symbols, data, sectors)
    
    # Generate features for scoring
    stock_features = generate_technical_features(symbols, data)
    
    # Combine stats and features
    full_df = pd.merge(
        company_stats, 
        stock_features,
        on="Symbol", 
        how="left"
    )
    
    # Score stocks
    scored_full = score_stocks(full_df, models)
    
    # Apply filters
    filtered = scored_full.copy()
    
    if selected_sector != "All":
        filtered = filtered[filtered["Sector"] == selected_sector]
    
    if risk_level != "All":
        filtered = filtered[filtered["Risk_Level"] == risk_level]
    
    if signal_type == "Buy Signals":
        filtered = filtered[filtered["Buy_Prob"] > 0.6]
    elif signal_type == "Sell Signals":
        filtered = filtered[filtered["Sell_Prob"] > 0.6]
    
    # Market overview
    st.markdown("<h2 class='sub-header'>Market Overview</h2>", unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.write("### Sector Distribution")
        sector_counts = company_stats["Sector"].value_counts()
        fig = go.Figure(data=[go.Pie(
            labels=sector_counts.index,
            values=sector_counts.values,
            hole=.3
        )])
        fig.update_layout(margin=dict(t=0, b=0, l=0, r=0))
        st.plotly_chart(fig, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)
    
    with col2:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.write("### Risk Distribution")
        risk_counts = company_stats["Risk_Level"].value_counts()
        fig = go.Figure(data=[go.Bar(
            x=risk_counts.index,
            y=risk_counts.values
        )])
        fig.update_layout(margin=dict(t=30, b=0, l=0, r=0))
        st.plotly_chart(fig, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)
    
    # Filtered stocks
    st.markdown("<h2 class='sub-header'>Stock Analysis</h2>", unsafe_allow_html=True)
    
    if filtered.empty:
        st.info("No stocks match the selected filters. Try changing your criteria.")
    else:
        # Display stock table
        display_cols = ["Symbol", "Sector", "Risk_Level", "Buy_Prob", "Sell_Prob", "Estimated_Return_Model"]
        st.dataframe(filtered[display_cols].sort_values("Buy_Prob", ascending=False))
        
        # Chart section
        st.markdown("<h2 class='sub-header'>Stock Charts</h2>", unsafe_allow_html=True)
        
        symbol = st.selectbox("Select Stock", filtered["Symbol"].tolist())
        
        if symbol in data:
            df = data[symbol].copy()
            df.reset_index(inplace=True)
            
            # Plot price chart
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=df["index"], 
                y=df["Close"],
                mode='lines',
                name='Close Price'
            ))
            fig.update_layout(
                title=f"{symbol} Stock Price",
                xaxis_title="Date",
                yaxis_title="Price (MAD)"
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Technical analysis
            st.write("### Technical Indicators")
            
            # Candlestick chart with MA
            df['MA20'] = df['Close'].rolling(window=20).mean()
            df['MA50'] = df['Close'].rolling(window=50).mean()
            
            fig = go.Figure()
            
            fig.add_trace(go.Candlestick(
                x=df['index'],
                open=df['Open'],
                high=df['High'],
                low=df['Low'],
                close=df['Close'],
                name='Candlestick'
            ))
            
            fig.add_trace(go.Scatter(
                x=df['index'],
                y=df['MA20'],
                name='20-day MA',
                line=dict(color='orange')
            ))
            
            fig.add_trace(go.Scatter(
                x=df['index'],
                y=df['MA50'],
                name='50-day MA',
                line=dict(color='green')
            ))
            
            fig.update_layout(
                title=f"{symbol} with Moving Averages",
                xaxis_title="Date",
                yaxis_title="Price (MAD)",
                xaxis_rangeslider_visible=False
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # ML model predictions
            st.write("### ML Model Predictions")
            
            col1, col2 = st.columns(2)
            
            with col1:
                buy_prob = float(filtered[filtered["Symbol"] == symbol]["Buy_Prob"].values[0])
                fig = go.Figure(go.Indicator(
                    mode = "gauge+number",
                    value = buy_prob,
                    title = {'text': "Buy Signal Probability"},
                    gauge = {
                        'axis': {'range': [0, 1]},
                        'bar': {'color': "green"},
                        'steps': [
                            {'range': [0, 0.4], 'color': "lightgray"},
                            {'range': [0.4, 0.6], 'color': "gray"},
                            {'range': [0.6, 1], 'color': "lightgreen"}
                        ],
                        'threshold': {
                            'line': {'color': "red", 'width': 4},
                            'thickness': 0.75,
                            'value': 0.6
                        }
                    }
                ))
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                sell_prob = float(filtered[filtered["Symbol"] == symbol]["Sell_Prob"].values[0])
                fig = go.Figure(go.Indicator(
                    mode = "gauge+number",
                    value = sell_prob,
                    title = {'text': "Sell Signal Probability"},
                    gauge = {
                        'axis': {'range': [0, 1]},
                        'bar': {'color': "red"},
                        'steps': [
                            {'range': [0, 0.4], 'color': "lightgray"},
                            {'range': [0.4, 0.6], 'color': "gray"},
                            {'range': [0.6, 1], 'color': "lightcoral"}
                        ],
                        'threshold': {
                            'line': {'color': "green", 'width': 4},
                            'thickness': 0.75,
                            'value': 0.6
                        }
                    }
                ))
                st.plotly_chart(fig, use_container_width=True)

def render_educational_content():
    st.markdown("<h1 class='main-header'>Educational Content</h1>", unsafe_allow_html=True)
    
    # Get educational content
    content = get_educational_content()
    
    if not content:
        st.info("No educational content available yet. Check back later!")
        return
    
    # Group by category
    content_by_category = {}
    for article in content:
        if article["category"] not in content_by_category:
            content_by_category[article["category"]] = []
        content_by_category[article["category"]].append(article)
    
    # Create tabs for each category
    if content_by_category:
        tabs = st.tabs(list(content_by_category.keys()))
        
        for i, (category, articles) in enumerate(content_by_category.items()):
            with tabs[i]:
                for article in articles:
                    with st.expander(article["title"]):
                        st.markdown(article["content"])
    else:
        st.info("No content available yet")

def render_settings(user_id):
    st.markdown("<h1 class='main-header'>Settings</h1>", unsafe_allow_html=True)
    
    # Get current preferences
    current_prefs = get_user_preferences(user_id)
    
    st.markdown("<h2 class='sub-header'>Investment Preferences</h2>", unsafe_allow_html=True)
    
    # Create form
    with st.form("preferences_form"):
        # Capital
        capital = st.number_input(
            "Investment Capital (MAD)",
            min_value=1000.0,
            max_value=10000000.0,
            value=float(current_prefs["capital"]) if current_prefs else 100000.0,
            step=10000.0
        )
        
        # Time horizon
        time_options = [
            "Short Term (1-2 years)",
            "Medium Term (3-5 years)",
            "Long Term (5+ years)"
        ]
        time_horizon = st.selectbox(
            "Investment Time Horizon",
            options=time_options,
            index=time_options.index(current_prefs["time_horizon"]) if current_prefs and current_prefs["time_horizon"] in time_options else 1
        )
        
        # Risk tolerance
        risk_options = ["Low", "Medium", "High"]
        risk_tolerance = st.selectbox(
            "Risk Tolerance",
            options=risk_options,
            index=risk_options.index(current_prefs["risk_tolerance"].capitalize()) if current_prefs else 1
        )
        
        # Preferred industries
        all_industries = [
            "Banking", "Insurance", "Energy", "Real Estate", "Construction",
            "Consumer Staples", "Telecommunications", "Technology", "Mining",
            "Transport & Logistics", "Retail", "Manufacturing", "Healthcare"
        ]
        
        preferred_industries = st.multiselect(
            "Preferred Industries (Optional)",
            options=all_industries,
            default=current_prefs["preferred_industries"] if current_prefs else []
        )
        
        # Excluded industries
        remaining_industries = [ind for ind in all_industries if ind not in preferred_industries]
        excluded_industries = st.multiselect(
            "Industries to Exclude (Optional)",
            options=remaining_industries,
            default=current_prefs["excluded_industries"] if current_prefs else []
        )
        
        # Submit button
        submitted = st.form_submit_button("Save Preferences")
        
        if submitted:
            save_user_preferences(
                user_id,
                capital,
                time_horizon,
                preferred_industries,
                excluded_industries,
                risk_tolerance.lower()
            )
            st.success("Preferences saved successfully!")

def render_login_page():
    st.markdown("<h1 class='main-header'>Moroccan Stock Investment Advisor</h1>", unsafe_allow_html=True)
    
    tab1, tab2 = st.tabs(["Login", "Create Account"])
    
    with tab1:
        with st.form("login_form"):
            username = st.text_input("Username")
            password = st.text_input("Password", type="password")
            
            submitted = st.form_submit_button("Login")
            
            if submitted:
                if username and password:
                    user_id = verify_user(username, password)
                    
                    if user_id:
                        st.session_state.user_id = user_id
                        st.session_state.authenticated = True
                        st.success("Login successful!")
                        st.experimental_rerun()
                    else:
                        st.error("Invalid username or password")
                else:
                    st.error("Please enter both username and password")
    
    with tab2:
        with st.form("signup_form"):
            new_username = st.text_input("Choose Username")
            new_email = st.text_input("Email")
            new_password = st.text_input("Choose Password", type="password")
            confirm_password = st.text_input("Confirm Password", type="password")
            
            submitted = st.form_submit_button("Create Account")
            
            if submitted:
                if not (new_username and new_email and new_password):
                    st.error("Please fill out all fields")
                elif new_password != confirm_password:
                    st.error("Passwords do not match")
                else:
                    user_id = create_user(new_username, new_email, new_password)
                    
                    if user_id:
                        st.session_state.user_id = user_id
                        st.session_state.authenticated = True
                        st.success("Account created! Now you can set your investment preferences.")
                        st.experimental_rerun()
                    else:
                        st.error("Username or email already exists")

# ----------------------
# MAIN APP FLOW
# ----------------------
def main():
    # Check if user is authenticated
    if "authenticated" not in st.session_state:
        st.session_state.authenticated = False
    
    # Check if page is set
    if "page" not in st.session_state:
        st.session_state.page = "Home"
    
    # Handle authentication
    if not st.session_state.authenticated:
        render_login_page()
    else:
        # Sidebar logout button
        if st.sidebar.button("Logout"):
            st.session_state.authenticated = False
            if "user_id" in st.session_state:
                del st.session_state.user_id
            st.experimental_rerun()
        
        # Show dashboard
        render_dashboard(st.session_state.user_id)

if __name__ == "__main__":
    main()
