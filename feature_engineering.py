"""
Feature Engineering for Fraud Detection
Creates advanced features from raw transaction data
"""

import pandas as pd
import numpy as np
from datetime import datetime
from typing import Tuple
import warnings
warnings.filterwarnings('ignore')


class FraudFeatureEngine:
    """
    Feature engineering pipeline for fraud detection
    """
    
    def __init__(self):
        self.user_profiles = {}
        self.feature_columns = []
    
    def create_features(self, df: pd.DataFrame, 
                       is_training: bool = True) -> pd.DataFrame:
        """
        Create all fraud detection features
        
        Args:
            df: Raw transaction DataFrame
            is_training: Whether this is training data
            
        Returns:
            DataFrame with engineered features
        """
        df = df.copy()
        
        # Basic transaction features
        df = self._create_transaction_features(df)
        
        # Temporal features
        df = self._create_temporal_features(df)
        
        # User behavior features
        df = self._create_user_behavior_features(df)
        
        # Amount-based features
        df = self._create_amount_features(df)
        
        # Velocity features
        df = self._create_velocity_features(df)
        
        # Risk indicator features
        df = self._create_risk_indicators(df)
        
        # Interaction features
        df = self._create_interaction_features(df)
        
        if is_training:
            self.feature_columns = [col for col in df.columns 
                                   if col not in ['Time', 'Class']]
        
        return df
    
    def _create_transaction_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Basic transaction-level features"""
        
        # Amount statistics
        df['amount_log'] = np.log1p(df['Amount'])
        df['amount_squared'] = df['Amount'] ** 2
        df['amount_sqrt'] = np.sqrt(df['Amount'])
        
        # Amount percentile
        df['amount_percentile'] = df['Amount'].rank(pct=True)
        
        return df
    
    def _create_temporal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Time-based features"""
        
        # Convert seconds to hours
        df['hour'] = (df['Time'] / 3600) % 24
        df['day'] = (df['Time'] / 86400).astype(int)
        
        # Time of day categories
        df['is_night'] = ((df['hour'] >= 23) | (df['hour'] < 5)).astype(int)
        df['is_morning'] = ((df['hour'] >= 5) & (df['hour'] < 12)).astype(int)
        df['is_afternoon'] = ((df['hour'] >= 12) & (df['hour'] < 18)).astype(int)
        df['is_evening'] = ((df['hour'] >= 18) & (df['hour'] < 23)).astype(int)
        
        # Cyclical encoding for hour (24-hour cycle)
        df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
        
        # Day of period
        df['day_sin'] = np.sin(2 * np.pi * df['day'] / 7)
        df['day_cos'] = np.cos(2 * np.pi * df['day'] / 7)
        
        return df
    
    def _create_user_behavior_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """User-specific behavior patterns"""
        
        # Sort by time to ensure correct rolling calculations
        df = df.sort_values('Time').reset_index(drop=True)
        
        # Rolling statistics (simulating user history)
        # In production, this would use actual user IDs
        
        # Rolling mean and std of amounts (window of 10 transactions)
        df['amount_rolling_mean_10'] = df['Amount'].rolling(
            window=10, min_periods=1
        ).mean()
        df['amount_rolling_std_10'] = df['Amount'].rolling(
            window=10, min_periods=1
        ).std().fillna(0)
        
        # Transaction count in rolling window
        df['transaction_count_10'] = df['Amount'].rolling(
            window=10, min_periods=1
        ).count()
        
        # Deviation from user's typical behavior
        df['amount_deviation'] = np.abs(
            df['Amount'] - df['amount_rolling_mean_10']
        ) / (df['amount_rolling_std_10'] + 1e-6)
        
        # Z-score of amount
        overall_mean = df['Amount'].mean()
        overall_std = df['Amount'].std()
        df['amount_zscore'] = (df['Amount'] - overall_mean) / overall_std
        
        return df
    
    def _create_amount_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Amount-specific features"""
        
        # Binned amounts
        amount_bins = [0, 10, 50, 100, 500, np.inf]
        df['amount_bin'] = pd.cut(
            df['Amount'], 
            bins=amount_bins, 
            labels=[0, 1, 2, 3, 4]
        ).astype(int)
        
        # High value flags
        percentiles = df['Amount'].quantile([0.75, 0.90, 0.95, 0.99])
        df['is_high_value_75'] = (df['Amount'] > percentiles[0.75]).astype(int)
        df['is_high_value_90'] = (df['Amount'] > percentiles[0.90]).astype(int)
        df['is_high_value_95'] = (df['Amount'] > percentiles[0.95]).astype(int)
        df['is_high_value_99'] = (df['Amount'] > percentiles[0.99]).astype(int)
        
        # Low value flag
        df['is_low_value'] = (df['Amount'] < df['Amount'].quantile(0.25)).astype(int)
        
        return df
    
    def _create_velocity_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Transaction velocity features"""
        
        df = df.sort_values('Time').reset_index(drop=True)
        
        # Time since last transaction
        df['time_since_last'] = df['Time'].diff().fillna(0)
        
        # Moving average of time between transactions
        df['time_between_ma_5'] = df['time_since_last'].rolling(
            window=5, min_periods=1
        ).mean()
        
        # Transaction frequency (transactions per hour)
        df['transaction_frequency'] = 3600 / (df['time_since_last'] + 1)
        
        # Flag for rapid transactions
        df['is_rapid_transaction'] = (df['time_since_last'] < 300).astype(int)  # < 5 min
        
        # Cumulative transaction count
        df['cumulative_transactions'] = range(1, len(df) + 1)
        
        return df
    
    def _create_risk_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Risk indicator flags"""
        
        # Unusual time flag (late night/early morning)
        df['unusual_time'] = df['is_night'].copy()
        
        # Unusual amount flag (extreme values)
        df['unusual_amount'] = (
            (df['amount_zscore'].abs() > 3) | 
            (df['is_high_value_99'] == 1)
        ).astype(int)
        
        # Combined risk score (simple heuristic)
        df['risk_score_heuristic'] = (
            df['unusual_time'] * 1 +
            df['unusual_amount'] * 2 +
            df['is_rapid_transaction'] * 1.5 +
            df['amount_deviation'] * 0.5
        )
        
        # Normalize risk score
        if df['risk_score_heuristic'].max() > 0:
            df['risk_score_normalized'] = (
                df['risk_score_heuristic'] / df['risk_score_heuristic'].max()
            )
        else:
            df['risk_score_normalized'] = 0
        
        return df
    
    def _create_interaction_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Feature interactions"""
        
        # Amount x Time interactions
        df['amount_x_night'] = df['Amount'] * df['is_night']
        df['amount_x_hour'] = df['Amount'] * df['hour']
        
        # Velocity x Amount
        df['frequency_x_amount'] = df['transaction_frequency'] * df['Amount']
        
        # Deviation x Time
        df['deviation_x_night'] = df['amount_deviation'] * df['is_night']
        
        return df
    
    def get_feature_importance_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Get summary of features for importance analysis
        
        Args:
            df: DataFrame with engineered features
            
        Returns:
            Summary statistics for each feature
        """
        feature_cols = [col for col in df.columns 
                       if col not in ['Time', 'Class', 'Amount']]
        
        summary = pd.DataFrame({
            'feature': feature_cols,
            'mean': df[feature_cols].mean().values,
            'std': df[feature_cols].std().values,
            'min': df[feature_cols].min().values,
            'max': df[feature_cols].max().values,
            'missing_pct': (df[feature_cols].isna().sum() / len(df) * 100).values
        })
        
        return summary.sort_values('std', ascending=False)


def create_fraud_features_pipeline(df: pd.DataFrame, 
                                   target_col: str = 'Class') -> Tuple[pd.DataFrame, pd.Series]:
    """
    Complete feature engineering pipeline
    
    Args:
        df: Raw transaction DataFrame
        target_col: Name of target column
        
    Returns:
        Tuple of (features_df, target_series)
    """
    engine = FraudFeatureEngine()
    
    # Create features
    df_features = engine.create_features(df, is_training=True)
    
    # Separate features and target
    X = df_features[engine.feature_columns]
    y = df_features[target_col] if target_col in df_features.columns else None
    
    # Handle any remaining missing values
    X = X.fillna(X.median())
    
    return X, y


def get_feature_descriptions() -> dict:
    """Get descriptions of all engineered features"""
    return {
        'Transaction Features': [
            'amount_log: Log-transformed transaction amount',
            'amount_squared: Squared transaction amount',
            'amount_sqrt: Square root of transaction amount',
            'amount_percentile: Percentile rank of transaction amount'
        ],
        'Temporal Features': [
            'hour: Hour of day (0-23)',
            'day: Day number in dataset',
            'is_night/morning/afternoon/evening: Time of day indicators',
            'hour_sin/cos: Cyclical encoding of hour',
            'day_sin/cos: Cyclical encoding of day'
        ],
        'User Behavior Features': [
            'amount_rolling_mean_10: Rolling mean of last 10 transactions',
            'amount_rolling_std_10: Rolling std of last 10 transactions',
            'transaction_count_10: Count of transactions in rolling window',
            'amount_deviation: Deviation from user typical behavior',
            'amount_zscore: Z-score of transaction amount'
        ],
        'Amount Features': [
            'amount_bin: Binned amount categories',
            'is_high_value_XX: Flags for high-value transactions',
            'is_low_value: Flag for low-value transactions'
        ],
        'Velocity Features': [
            'time_since_last: Time since previous transaction',
            'time_between_ma_5: Moving average of time between transactions',
            'transaction_frequency: Transactions per hour',
            'is_rapid_transaction: Flag for rapid succession',
            'cumulative_transactions: Total transaction count'
        ],
        'Risk Indicators': [
            'unusual_time: Late night/early morning flag',
            'unusual_amount: Extreme amount flag',
            'risk_score_heuristic: Combined risk score',
            'risk_score_normalized: Normalized risk score'
        ],
        'Interaction Features': [
            'amount_x_night: Amount during night hours',
            'amount_x_hour: Amount-hour interaction',
            'frequency_x_amount: Velocity-amount interaction',
            'deviation_x_night: Deviation during night hours'
        ]
    }


# Example usage
if __name__ == "__main__":
    print("Feature Engineering Module for Fraud Detection")
    print("=" * 50)
    
    # Display feature categories
    descriptions = get_feature_descriptions()
    for category, features in descriptions.items():
        print(f"\n{category}:")
        for feature in features:
            print(f"  • {feature}")
    
    print("\n" + "=" * 50)
    print("Total feature categories:", len(descriptions))
    print("Ready to process transaction data!")
