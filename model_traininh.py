"""
Model Training Pipeline for Fraud Detection
Handles class imbalance, hyperparameter tuning, and model evaluation
"""

import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score,
    precision_recall_curve, average_precision_score, roc_curve
)
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline as ImbPipeline
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')


class FraudModelTrainer:
    """
    Complete training pipeline for fraud detection models
    """
    
    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.model = None
        self.scaler = None
        self.anomaly_model = None
        self.metrics = {}
        self.feature_importance = None
    
    def prepare_data(self, X: pd.DataFrame, y: pd.Series, 
                    test_size: float = 0.3) -> tuple:
        """
        Prepare and split data for training
        
        Args:
            X: Feature DataFrame
            y: Target variable
            test_size: Proportion of data for testing
            
        Returns:
            X_train, X_test, y_train, y_test
        """
        # Split data with stratification
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, 
            test_size=test_size, 
            random_state=self.random_state,
            stratify=y
        )
        
        # Scale features
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Convert back to DataFrame to preserve column names
        X_train_scaled = pd.DataFrame(
            X_train_scaled, 
            columns=X_train.columns,
            index=X_train.index
        )
        X_test_scaled = pd.DataFrame(
            X_test_scaled, 
            columns=X_test.columns,
            index=X_test.index
        )
        
        print(f"Training set: {len(X_train)} samples")
        print(f"Test set: {len(X_test)} samples")
        print(f"Fraud rate in training: {y_train.mean():.4%}")
        print(f"Fraud rate in test: {y_test.mean():.4%}")
        
        return X_train_scaled, X_test_scaled, y_train, y_test
    
    def handle_imbalance(self, X_train: pd.DataFrame, y_train: pd.Series,
                        method: str = 'smote') -> tuple:
        """
        Handle class imbalance using various techniques
        
        Args:
            X_train: Training features
            y_train: Training labels
            method: 'smote', 'undersample', or 'combined'
            
        Returns:
            Resampled X_train, y_train
        """
        print(f"\nApplying {method} for class imbalance...")
        print(f"Original class distribution: {y_train.value_counts().to_dict()}")
        
        if method == 'smote':
            sampler = SMOTE(random_state=self.random_state, k_neighbors=5)
            X_resampled, y_resampled = sampler.fit_resample(X_train, y_train)
            
        elif method == 'undersample':
            sampler = RandomUnderSampler(random_state=self.random_state)
            X_resampled, y_resampled = sampler.fit_resample(X_train, y_train)
            
        elif method == 'combined':
            # SMOTE followed by undersampling
            over = SMOTE(sampling_strategy=0.3, random_state=self.random_state)
            under = RandomUnderSampler(sampling_strategy=0.5, random_state=self.random_state)
            
            X_temp, y_temp = over.fit_resample(X_train, y_train)
            X_resampled, y_resampled = under.fit_resample(X_temp, y_temp)
        else:
            X_resampled, y_resampled = X_train, y_train
        
        print(f"Resampled class distribution: {pd.Series(y_resampled).value_counts().to_dict()}")
        
        return X_resampled, y_resampled
    
    def train_xgboost(self, X_train: pd.DataFrame, y_train: pd.Series,
                     use_resampling: bool = True) -> XGBClassifier:
        """
        Train XGBoost model optimized for fraud detection
        
        Args:
            X_train: Training features
            y_train: Training labels
            use_resampling: Whether to apply SMOTE
            
        Returns:
            Trained XGBoost model
        """
        print("\n" + "="*50)
        print("Training XGBoost Model")
        print("="*50)
        
        # Handle imbalance if requested
        if use_resampling:
            X_train_res, y_train_res = self.handle_imbalance(
                X_train, y_train, method='smote'
            )
        else:
            X_train_res, y_train_res = X_train, y_train
        
        # Calculate scale_pos_weight for imbalanced data
        fraud_ratio = (y_train == 0).sum() / (y_train == 1).sum()
        
        # XGBoost parameters optimized for fraud detection
        params = {
            'max_depth': 6,
            'learning_rate': 0.1,
            'n_estimators': 200,
            'scale_pos_weight': fraud_ratio if not use_resampling else 1,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'min_child_weight': 1,
            'gamma': 0.1,
            'reg_alpha': 0.1,
            'reg_lambda': 1,
            'random_state': self.random_state,
            'eval_metric': 'auc',
            'tree_method': 'hist'
        }
        
        print("\nModel Parameters:")
        for key, value in params.items():
            print(f"  {key}: {value}")
        
        # Train model
        self.model = XGBClassifier(**params)
        self.model.fit(
            X_train_res, y_train_res,
            verbose=False
        )
        
        # Feature importance
        self.feature_importance = pd.DataFrame({
            'feature': X_train.columns,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        print("\nTop 10 Most Important Features:")
        print(self.feature_importance.head(10).to_string(index=False))
        
        return self.model
    
    def train_isolation_forest(self, X_train: pd.DataFrame, 
                              contamination: float = 0.02) -> IsolationForest:
        """
        Train Isolation Forest for anomaly detection
        
        Args:
            X_train: Training features
            contamination: Expected proportion of anomalies
            
        Returns:
            Trained Isolation Forest model
        """
        print("\n" + "="*50)
        print("Training Isolation Forest (Anomaly Detection)")
        print("="*50)
        
        self.anomaly_model = IsolationForest(
            contamination=contamination,
            random_state=self.random_state,
            n_estimators=100,
            max_samples='auto',
            max_features=1.0
        )
        
        self.anomaly_model.fit(X_train)
        
        # Test anomaly detection on training set
        anomaly_predictions = self.anomaly_model.predict(X_train)
        anomaly_count = (anomaly_predictions == -1).sum()
        
        print(f"Detected {anomaly_count} anomalies in training set ({anomaly_count/len(X_train):.2%})")
        
        return self.anomaly_model
    
    def evaluate_model(self, X_test: pd.DataFrame, y_test: pd.Series,
                      model_name: str = "XGBoost") -> dict:
        """
        Comprehensive model evaluation
        
        Args:
            X_test: Test features
            y_test: Test labels
            model_name: Name of model being evaluated
            
        Returns:
            Dictionary of evaluation metrics
        """
        print("\n" + "="*50)
        print(f"Evaluating {model_name} Model")
        print("="*50)
        
        # Predictions
        y_pred_proba = self.model.predict_proba(X_test)[:, 1]
        y_pred = (y_pred_proba > 0.5).astype(int)
        
        # Metrics
        roc_auc = roc_auc_score(y_test, y_pred_proba)
        avg_precision = average_precision_score(y_test, y_pred_proba)
        
        # Confusion Matrix
        cm = confusion_matrix(y_test, y_pred)
        tn, fp, fn, tp = cm.ravel()
        
        # Calculate rates
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        fpr = fp / (fp + tn)
        fnr = fn / (fn + tp)
        
        metrics = {
            'roc_auc': roc_auc,
            'average_precision': avg_precision,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'false_positive_rate': fpr,
            'false_negative_rate': fnr,
            'confusion_matrix': cm
        }
        
        # Print results
        print(f"\nROC-AUC Score: {roc_auc:.4f}")
        print(f"Average Precision: {avg_precision:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1-Score: {f1:.4f}")
        print(f"False Positive Rate: {fpr:.4f}")
        print(f"False Negative Rate: {fnr:.4f}")
        
        print("\nConfusion Matrix:")
        print(f"  True Negatives:  {tn:,}")
        print(f"  False Positives: {fp:,}")
        print(f"  False Negatives: {fn:,}")
        print(f"  True Positives:  {tp:,}")
        
        # Classification Report
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred, target_names=['Legitimate', 'Fraud']))
        
        self.metrics = metrics
        return metrics
    
    def cross_validate(self, X: pd.DataFrame, y: pd.Series, 
                      cv_folds: int = 5) -> dict:
        """
        Perform cross-validation
        
        Args:
            X: Features
            y: Labels
            cv_folds: Number of CV folds
            
        Returns:
            Cross-validation scores
        """
        print("\n" + "="*50)
        print(f"Performing {cv_folds}-Fold Cross-Validation")
        print("="*50)
        
        skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, 
                             random_state=self.random_state)
        
        # Cross-validate
        cv_scores = cross_val_score(
            self.model, X, y, 
            cv=skf, 
            scoring='roc_auc',
            n_jobs=-1
        )
        
        print(f"\nCross-Validation ROC-AUC Scores: {cv_scores}")
        print(f"Mean ROC-AUC: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
        
        return {
            'cv_scores': cv_scores,
            'mean_score': cv_scores.mean(),
            'std_score': cv_scores.std()
        }
    
    def plot_evaluation_metrics(self, X_test: pd.DataFrame, y_test: pd.Series,
                               save_path: str = None):
        """
        Create visualization of model performance
        
        Args:
            X_test: Test features
            y_test: Test labels
            save_path: Path to save the plot
        """
        y_pred_proba = self.model.predict_proba(X_test)[:, 1]
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # ROC Curve
        fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
        roc_auc = roc_auc_score(y_test, y_pred_proba)
        
        axes[0, 0].plot(fpr, tpr, label=f'ROC (AUC = {roc_auc:.4f})', linewidth=2)
        axes[0, 0].plot([0, 1], [0, 1], 'k--', label='Random')
        axes[0, 0].set_xlabel('False Positive Rate')
        axes[0, 0].set_ylabel('True Positive Rate')
        axes[0, 0].set_title('ROC Curve')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Precision-Recall Curve
        precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)
        avg_precision = average_precision_score(y_test, y_pred_proba)
        
        axes[0, 1].plot(recall, precision, label=f'PR (AP = {avg_precision:.4f})', linewidth=2)
        axes[0, 1].set_xlabel('Recall')
        axes[0, 1].set_ylabel('Precision')
        axes[0, 1].set_title('Precision-Recall Curve')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Confusion Matrix
        y_pred = (y_pred_proba > 0.5).astype(int)
        cm = confusion_matrix(y_test, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[1, 0])
        axes[1, 0].set_xlabel('Predicted')
        axes[1, 0].set_ylabel('Actual')
        axes[1, 0].set_title('Confusion Matrix')
        
        # Feature Importance (Top 15)
        top_features = self.feature_importance.head(15)
        axes[1, 1].barh(top_features['feature'], top_features['importance'])
        axes[1, 1].set_xlabel('Importance')
        axes[1, 1].set_title('Top 15 Feature Importances')
        axes[1, 1].invert_yaxis()
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"\nPlot saved to {save_path}")
        
        plt.show()
    
    def save_models(self, model_path: str = 'models/xgboost_model.pkl',
                   scaler_path: str = 'models/scaler.pkl',
                   anomaly_path: str = 'models/isolation_forest.pkl'):
        """Save trained models"""
        with open(model_path, 'wb') as f:
            pickle.dump(self.model, f)
        print(f"Model saved to {model_path}")
        
        if self.scaler:
            with open(scaler_path, 'wb') as f:
                pickle.dump(self.scaler, f)
            print(f"Scaler saved to {scaler_path}")
        
        if self.anomaly_model:
            with open(anomaly_path, 'wb') as f:
                pickle.dump(self.anomaly_model, f)
            print(f"Anomaly model saved to {anomaly_path}")


# Main training pipeline
def train_fraud_detection_pipeline(X: pd.DataFrame, y: pd.Series,
                                   save_models: bool = True) -> FraudModelTrainer:
    """
    Complete training pipeline
    
    Args:
        X: Feature DataFrame
        y: Target Series
        save_models: Whether to save trained models
        
    Returns:
        Trained FraudModelTrainer instance
    """
    trainer = FraudModelTrainer()
    
    # Prepare data
    X_train, X_test, y_train, y_test = trainer.prepare_data(X, y)
    
    # Train XGBoost
    trainer.train_xgboost(X_train, y_train, use_resampling=True)
    
    # Train Isolation Forest
    trainer.train_isolation_forest(X_train)
    
    # Evaluate
    trainer.evaluate_model(X_test, y_test)
    
    # Cross-validation
    trainer.cross_validate(X_train, y_train)
    
    # Plot metrics
    trainer.plot_evaluation_metrics(X_test, y_test)
    
    # Save models
    if save_models:
        trainer.save_models()
    
    return trainer


if __name__ == "__main__":
    print("Fraud Detection Model Training Pipeline")
    print("=" * 50)
    print("This module provides complete training functionality")
    print("Import and use train_fraud_detection_pipeline() function")
