"""
Data validation utilities to ensure sufficient data for training and prediction
"""
import pandas as pd
from datetime import datetime
from typing import Tuple, Optional


# Minimum data requirements (in calendar days)
MIN_DAYS_FOR_TRAINING = 365  # Need at least 1 year of calendar days for training
MIN_DAYS_FOR_PREDICTION = 120  # Need at least 120 days for prediction with features
MIN_ROWS_AFTER_FEATURES = 50  # Minimum rows after feature engineering


class InsufficientDataError(Exception):
    """Raised when there is not enough data for training or prediction"""
    pass


def validate_training_data(
    df: pd.DataFrame,
    ticker: str,
    start_date: datetime,
    end_date: datetime,
    min_days: int = MIN_DAYS_FOR_TRAINING
) -> None:
    """
    Validate that there is sufficient data for training.
    
    Args:
        df: DataFrame with training data
        ticker: Stock ticker symbol
        start_date: Training start date
        end_date: Training end date
        min_days: Minimum number of days required
        
    Raises:
        InsufficientDataError: If data is insufficient
    """
    if df is None or df.empty:
        raise InsufficientDataError(
            f"❌ {ticker}: No data available for training period {start_date.date()} to {end_date.date()}"
        )
    
    num_rows = len(df)
    date_range = (end_date - start_date).days
    
    # Allow for some data gaps (weekends/holidays)
    if num_rows < min_days * 0.5:  # Expect at least 50% of calendar days (accounting for weekends)
        raise InsufficientDataError(
            f"❌ {ticker}: Insufficient training data. "
            f"Got {num_rows} rows, need at least {int(min_days * 0.5)} rows "
            f"(~{min_days} calendar days) for reliable model training.\n"
            f"   📅 Period: {start_date.date()} to {end_date.date()} ({date_range} days)\n"
            f"   💡 Suggestion: Increase training period or check data quality"
        )
    
    # Check for required columns
    required_cols = ['Close', 'High', 'Low', 'Open', 'Volume']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise InsufficientDataError(
            f"❌ {ticker}: Missing required columns: {missing_cols}\n"
            f"   Available columns: {list(df.columns)}"
        )
    
    # Check for NaN values in critical columns
    critical_nans = df[['Close', 'Volume']].isna().sum()
    if critical_nans['Close'] > num_rows * 0.1:  # More than 10% NaN
        raise InsufficientDataError(
            f"❌ {ticker}: Too many NaN values in Close price: {critical_nans['Close']} / {num_rows} rows\n"
            f"   💡 Data quality issue - try different data source or date range"
        )
    
    print(f"✅ {ticker}: Training data validated - {num_rows} rows over {date_range} days")


def validate_prediction_data(
    df: pd.DataFrame,
    ticker: str,
    min_days: int = MIN_DAYS_FOR_PREDICTION
) -> None:
    """
    Validate that there is sufficient data for making predictions.
    
    Args:
        df: DataFrame with recent data for prediction
        ticker: Stock ticker symbol
        min_days: Minimum number of days required
        
    Raises:
        InsufficientDataError: If data is insufficient
    """
    if df is None or df.empty:
        raise InsufficientDataError(
            f"❌ {ticker}: No data available for prediction"
        )
    
    num_rows = len(df)
    
    # Allow for some data gaps (weekends/holidays)
    if num_rows < min_days * 0.5:  # Expect at least 50% of calendar days
        raise InsufficientDataError(
            f"❌ {ticker}: Insufficient prediction data. "
            f"Got {num_rows} rows, need at least {int(min_days * 0.5)} rows "
            f"(~{min_days} calendar days) for feature engineering.\n"
            f"   💡 Feature calculation (RSI, MACD, SMA50, etc.) requires historical data"
        )
    
    # Check date range
    if hasattr(df.index, 'min') and hasattr(df.index, 'max'):
        date_span = (df.index.max() - df.index.min()).days
        print(f"✅ {ticker}: Prediction data validated - {num_rows} rows spanning {date_span} days")


def validate_features_after_engineering(
    df: pd.DataFrame,
    ticker: str,
    min_rows: int = MIN_ROWS_AFTER_FEATURES,
    context: str = "training"
) -> None:
    """
    Validate that there are enough rows after feature engineering.
    
    Args:
        df: DataFrame after feature engineering
        ticker: Stock ticker symbol
        min_rows: Minimum number of rows required after features
        context: Context string ("training" or "prediction")
        
    Raises:
        InsufficientDataError: If insufficient rows remain
    """
    if df is None or df.empty:
        raise InsufficientDataError(
            f"❌ {ticker}: All rows dropped during {context} feature engineering!\n"
            f"   💡 This usually means:\n"
            f"      - Not enough historical data for technical indicators\n"
            f"      - Too many NaN/missing values in source data\n"
            f"      - Feature calculation window is too large for available data"
        )
    
    num_rows = len(df)
    
    if num_rows < min_rows:
        raise InsufficientDataError(
            f"❌ {ticker}: Only {num_rows} rows remain after {context} feature engineering, "
            f"need at least {min_rows}\n"
            f"   💡 Solutions:\n"
            f"      - Increase data period (use more historical data)\n"
            f"      - Reduce feature complexity (shorter moving average windows)\n"
            f"      - Check for data quality issues"
        )
    
    print(f"✅ {ticker}: {num_rows} rows available after {context} feature engineering")


def get_data_summary(df: pd.DataFrame, ticker: str) -> dict:
    """
    Get a summary of data availability for diagnostics.
    
    Args:
        df: DataFrame to summarize
        ticker: Stock ticker symbol
        
    Returns:
        Dictionary with data summary
    """
    if df is None or df.empty:
        return {
            'ticker': ticker,
            'rows': 0,
            'status': 'EMPTY',
            'message': 'No data available'
        }
    
    summary = {
        'ticker': ticker,
        'rows': len(df),
        'columns': len(df.columns),
        'date_range': None,
        'missing_values': {},
        'status': 'OK'
    }
    
    # Date range
    if hasattr(df.index, 'min') and hasattr(df.index, 'max'):
        summary['date_range'] = f"{df.index.min().date()} to {df.index.max().date()}"
        summary['days_span'] = (df.index.max() - df.index.min()).days
    
    # Missing values
    for col in ['Close', 'Volume', 'High', 'Low', 'Open']:
        if col in df.columns:
            nan_count = df[col].isna().sum()
            if nan_count > 0:
                summary['missing_values'][col] = nan_count
    
    # Determine status
    if len(df) < MIN_DAYS_FOR_TRAINING * 0.7:
        summary['status'] = 'INSUFFICIENT'
        summary['message'] = f"Only {len(df)} rows, need ~{int(MIN_DAYS_FOR_TRAINING * 0.7)}"
    elif summary['missing_values']:
        summary['status'] = 'WARNING'
        summary['message'] = f"Has missing values: {summary['missing_values']}"
    
    return summary


def print_data_diagnostics(summaries: list, max_display: int = 100):
    """
    Print a formatted table of data diagnostics for multiple tickers.
    
    Args:
        summaries: List of data summary dictionaries
        max_display: Maximum number of tickers to display (default: 100)
    """
    print("\n" + "=" * 100)
    print("📊 DATA VALIDATION DIAGNOSTICS")
    print("=" * 100)
    
    # Summary statistics first
    total = len(summaries)
    ok_count = sum(1 for s in summaries if s['status'] == 'OK')
    insufficient = sum(1 for s in summaries if s['status'] == 'INSUFFICIENT')
    warning = sum(1 for s in summaries if s['status'] == 'WARNING')
    empty = sum(1 for s in summaries if s['status'] == 'EMPTY')
    error = sum(1 for s in summaries if s['status'] == 'ERROR')
    
    print(f"Total: {total} tickers | ✅ {ok_count} OK | ⚠️ {warning} warnings | ❌ {insufficient} insufficient | 🚫 {empty} empty")
    
    # Separate problematic tickers
    problematic = [s for s in summaries if s['status'] not in ['OK']]
    good = [s for s in summaries if s['status'] == 'OK']
    
    # Show problematic tickers first (always show these)
    if problematic:
        print("\n⚠️  PROBLEMATIC TICKERS:")
        print("-" * 100)
        print(f"{'Ticker':<10} {'Rows':<8} {'Days':<8} {'Status':<15} {'Message':<40}")
        print("-" * 100)
        
        for s in problematic[:50]:  # Show first 50 problematic
            days = s.get('days_span', 'N/A')
            message = s.get('message', '')
            print(f"{s['ticker']:<10} {s['rows']:<8} {str(days):<8} {s['status']:<15} {message:<40}")
        
        if len(problematic) > 50:
            print(f"... and {len(problematic) - 50} more problematic tickers")
    
    # Show sample of good tickers (limited)
    if good and len(good) <= 20:
        print("\n✅ GOOD TICKERS:")
        print("-" * 100)
        print(f"{'Ticker':<10} {'Rows':<8} {'Days':<8} {'Status':<15} {'Message':<40}")
        print("-" * 100)
        
        for s in good:
            days = s.get('days_span', 'N/A')
            message = s.get('message', '')
            print(f"{s['ticker']:<10} {s['rows']:<8} {str(days):<8} {s['status']:<15} {message:<40}")
    elif good:
        print(f"\n✅ {len(good)} tickers have good data quality (not shown for brevity)")
    
    print("=" * 100 + "\n")
    
    # Final summary with warnings
    if insufficient > total * 0.3:
        print(f"⚠️  WARNING: {insufficient}/{total} tickers have insufficient data!")
        print(f"💡 Consider using a longer data period or filtering these tickers\n")







