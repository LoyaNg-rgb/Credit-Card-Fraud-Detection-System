# Complete Setup Guide - Credit Card Fraud Detection System

## 📋 Table of Contents
1. [Prerequisites](#prerequisites)
2. [Project Structure](#project-structure)
3. [Installation Steps](#installation-steps)
4. [Data Preparation](#data-preparation)
5. [Running the Pipeline](#running-the-pipeline)
6. [Using the Dashboard](#using-the-dashboard)
7. [API Usage](#api-usage)
8. [Troubleshooting](#troubleshooting)

---

## Prerequisites

### System Requirements
- **Python:** 3.8 or higher
- **RAM:** Minimum 8GB (16GB recommended)
- **Storage:** 2GB free space
- **OS:** Windows, macOS, or Linux

### Required Skills
- Basic Python programming
- Understanding of machine learning concepts
- Familiarity with command line/terminal

---

## Project Structure

Create the following directory structure:

```
fraud-detection-system/
│
├── data/
│   ├── raw/                    # Place creditcard.csv here
│   ├── processed/              # Processed data files
│   └── README.md
│
├── notebooks/
│   ├── 01_EDA.ipynb
│   ├── 02_Feature_Engineering.ipynb
│   ├── 03_Model_Training.ipynb
│   ├── 04_Model_Evaluation.ipynb
│   └── Complete_Fraud_Detection_Pipeline.ipynb
│
├── src/
│   ├── __init__.py
│   ├── data_preprocessing.py
│   ├── feature_engineering.py
│   ├── model_training.py
│   ├── fraud_detection.py
│   └── utils.py
│
├── models/
│   ├── xgboost_model.pkl       # Trained model (generated)
│   ├── scaler.pkl              # Feature scaler (generated)
│   ├── isolation_forest.pkl   # Anomaly model (generated)
│   ├── feature_importance.csv # Feature analysis (generated)
│   └── README.md
│
├── dashboard/
│   └── app.py                  # Streamlit dashboard
│
├── tests/
│   ├── __init__.py
│   ├── test_features.py
│   ├── test_models.py
│   └── test_detection.py
│
├── requirements.txt
├── setup.py
├── config.yaml
├── README.md
├── SETUP_GUIDE.md             # This file
└── .gitignore
```

---

## Installation Steps

### Step 1: Clone or Create Repository

**Option A: Clone from GitHub**
```bash
git clone https://github.com/LoyaNg-rgb/fraud-detection-system.git
cd fraud-detection-system
```

**Option B: Create Manually**
```bash
mkdir fraud-detection-system
cd fraud-detection-system
```

### Step 2: Create Virtual Environment

**On Windows:**
```bash
python -m venv venv
venv\Scripts\activate
```

**On macOS/Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

### Step 3: Install Dependencies

```bash
# Upgrade pip
pip install --upgrade pip

# Install required packages
pip install -r requirements.txt

# Verify installation
python -c "import pandas, sklearn, xgboost; print('All packages installed successfully!')"
```

### Step 4: Create Directory Structure

```bash
# Create all necessary directories
mkdir -p data/raw data/processed models notebooks dashboard tests src

# Create __init__.py files
touch src/__init__.py tests/__init__.py
```

---

## Data Preparation

### Step 1: Download Dataset

1. Visit [Kaggle Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
2. Download `creditcard.csv`
3. Place it in `data/raw/` directory

**Alternative:** Use a sample dataset for testing
```python
# Create sample data for testing (if needed)
import pandas as pd
import numpy as np

# Generate synthetic data
np.random.seed(42)
n_samples = 10000
data = {
    'Time': np.random.randint(0, 172800, n_samples),
    'Amount': np.random.exponential(88, n_samples),
    'Class': np.random.choice([0, 1], n_samples, p=[0.998, 0.002])
}

# Add V1-V28 features
for i in range(1, 29):
    data[f'V{i}'] = np.random.randn(n_samples)

df = pd.DataFrame(data)
df.to_csv('data/raw/creditcard.csv', index=False)
```

### Step 2: Verify Data

```python
import pandas as pd

df = pd.read_csv('data/raw/creditcard.csv')
print(f"Shape: {df.shape}")
print(f"Columns: {list(df.columns)}")
print(f"Fraud rate: {df['Class'].mean():.4%}")
```

---

## Running the Pipeline

### Method 1: Complete Notebook (Recommended for First Time)

```bash
# Start Jupyter Notebook
jupyter notebook

# Open and run:
# notebooks/Complete_Fraud_Detection_Pipeline.ipynb
```

**What it does:**
- Loads and explores data
- Engineers features
- Trains models
- Evaluates performance
- Saves models and metrics

### Method 2: Python Scripts

```python
# Run step by step

# 1. Feature Engineering
from src.feature_engineering import create_fraud_features_pipeline
import pandas as pd

df = pd.read_csv('data/raw/creditcard.csv')
X, y = create_fraud_features_pipeline(df)

# 2. Train Models
from src.model_training import train_fraud_detection_pipeline

trainer = train_fraud_detection_pipeline(X, y, save_models=True)

# 3. Make Predictions
from src.fraud_detection import FraudDetector

detector = FraudDetector(
    model_path='models/xgboost_model.pkl',
    scaler_path='models/scaler.pkl',
    anomaly_model_path='models/isolation_forest.pkl'
)

# Test prediction
sample = X.iloc[0:1]
result = detector.predict(sample)
print(f"Fraud Probability: {result['fraud_probability']:.2%}")
```

### Method 3: Command Line Interface

Create a CLI script `train.py`:

```python
import argparse
from src.feature_engineering import create_fraud_features_pipeline
from src.model_training import train_fraud_detection_pipeline
import pandas as pd

def main():
    parser = argparse.ArgumentParser(description='Train fraud detection model')
    parser.add_argument('--data', type=str, required=True, help='Path to data file')
    parser.add_argument('--test-size', type=float, default=0.3, help='Test set size')
    args = parser.parse_args()
    
    print(f"Loading data from {args.data}...")
    df = pd.read_csv(args.data)
    
    print("Creating features...")
    X, y = create_fraud_features_pipeline(df)
    
    print("Training models...")
    trainer = train_fraud_detection_pipeline(X, y, save_models=True)
    
    print("Training completed!")

if __name__ == "__main__":
    main()
```

Run it:
```bash
python train.py --data data/raw/creditcard.csv --test-size 0.3
```

---

## Using the Dashboard

### Step 1: Start Dashboard

```bash
# From project root directory
streamlit run dashboard/app.py

# Dashboard will open at http://localhost:8501
```

### Step 2: Dashboard Features

**1. Dashboard Page**
- View overall statistics
- See transaction distributions
- Monitor fraud trends

**2. Real-time Monitoring**
- Analyze individual transactions
- Get instant fraud scores
- View recommendations

**3. Batch Analysis**
- Upload CSV files
- Process multiple transactions
- Download results

**4. Model Info**
- View model architecture
- Check performance metrics
- See feature importance

**5. Settings**
- Adjust risk thresholds
- Configure alerts
- Customize preferences

---

## API Usage

### Creating a Simple API

Create `api/app.py`:

```python
from flask import Flask, request, jsonify
import pandas as pd
import sys
sys.path.append('..')
from src.fraud_detection import FraudDetector

app = Flask(__name__)

# Load detector
detector = FraudDetector(
    model_path='../models/xgboost_model.pkl',
    scaler_path='../models/scaler.pkl',
    anomaly_model_path='../models/isolation_forest.pkl'
)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        transaction = pd.DataFrame([data])
        
        result = detector.predict(transaction)
        
        return jsonify({
            'success': True,
            'fraud_probability': float(result['fraud_probability']),
            'risk_level': result['risk_level'],
            'recommendation': result['recommendation']
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 400

@app.route('/batch-predict', methods=['POST'])
def batch_predict():
    try:
        data = request.json
        transactions = pd.DataFrame(data)
        
        results = detector.predict_batch(transactions)
        
        return jsonify({
            'success': True,
            'results': results.to_dict('records')
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True, port=5000)
```

### Testing API

```bash
# Start API
python api/app.py

# Test with curl
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{"V1": -1.35, "V2": -0.072, ..., "Amount": 149.62}'
```

---

## Troubleshooting

### Common Issues

**1. Import Error: Module not found**
```bash
# Solution: Install missing package
pip install <package-name>

# Or reinstall all dependencies
pip install -r requirements.txt --force-reinstall
```

**2. Memory Error during training**
```python
# Solution: Reduce data size or use sampling
df_sample = df.sample(n=100000, random_state=42)
```

**3. Model files not found**
```bash
# Solution: Ensure models directory exists and train models
mkdir -p models
python train.py --data data/raw/creditcard.csv
```

**4. Streamlit dashboard not starting**
```bash
# Solution: Check if streamlit is installed
pip install streamlit

# Try alternate port
streamlit run dashboard/app.py --server.port 8502
```

**5. CUDA/GPU errors with XGBoost**
```python
# Solution: Use CPU version
# In model_training.py, set:
params = {
    ...
    'tree_method': 'hist',  # Use CPU
    'gpu_id': -1
}
```

### Performance Optimization

**1. Speed up training**
```python
# Use fewer estimators for testing
params = {
    'n_estimators': 100,  # Reduced from 200
    'n_jobs': -1          # Use all CPU cores
}
```

**2. Reduce memory usage**
```python
# Use sparse matrices
from scipy.sparse import csr_matrix
X_sparse = csr_matrix(X.values)
```

**3. Faster predictions**
```python
# Batch predictions instead of loops
results = detector.predict_batch(transactions)
# Instead of:
# for transaction in transactions:
#     result = detector.predict(transaction)
```

---

## Testing

### Run Unit Tests

```bash
# Run all tests
pytest tests/

# Run with coverage
pytest tests/ --cov=src

# Run specific test
pytest tests/test_models.py::test_fraud_detection
```

### Create Test Cases

Create `tests/test_models.py`:

```python
import pytest
import pandas as pd
import numpy as np
from src.fraud_detection import FraudDetector

def test_fraud_detector_initialization():
    # Test model loading
    detector = FraudDetector(
        model_path='models/xgboost_model.pkl',
        scaler_path='models/scaler.pkl'
    )
    assert detector.model is not None

def test_prediction():
    detector = FraudDetector('models/xgboost_model.pkl')
    
    # Create sample transaction
    sample = pd.DataFrame({
        'V1': [-1.35], 'V2': [-0.07], 'Amount': [149.62]
        # Add other features...
    })
    
    result = detector.predict(sample)
    
    assert 'fraud_probability' in result
    assert 0 <= result['fraud_probability'] <= 1
    assert result['risk_level'] in ['MINIMAL', 'LOW', 'MEDIUM', 'HIGH', 'CRITICAL']
```

---

## Next Steps

### 1. Model Improvement
- Experiment with different algorithms
- Tune hyperparameters
- Add more features
- Try ensemble methods

### 2. Deployment
- Deploy to AWS/GCP/Azure
- Set up CI/CD pipeline
- Create Docker container
- Implement monitoring

### 3. Production Readiness
- Add logging and monitoring
- Implement A/B testing
- Create feedback loop
- Set up automated retraining

### 4. Documentation
- Add API documentation
- Create user guide
- Write technical specifications
- Document model decisions

---

## Resources

### Documentation
- [XGBoost Documentation](https://xgboost.readthedocs.io/)
- [scikit-learn User Guide](https://scikit-learn.org/stable/user_guide.html)
- [Streamlit Documentation](https://docs.streamlit.io/)

### Tutorials
- [Handling Imbalanced Data](https://imbalanced-learn.org/stable/)
- [Feature Engineering Guide](https://www.kaggle.com/learn/feature-engineering)
- [Model Deployment](https://www.fullstackpython.com/deployment.html)

### Community
- [Kaggle Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
- [Stack Overflow](https://stackoverflow.com/questions/tagged/fraud-detection)
- [GitHub Issues](https://github.com/LoyaNg-rgb/fraud-detection-system/issues)

---

## Support

For questions or issues:
- Email: loyanganba.ngathem@gmail.com
- LinkedIn: [Loyanganba Ngathem](https://linkedin.com/in/loyanganba-ngathem-315327378)
- GitHub: [Open an Issue](https://github.com/LoyaNg-rgb/fraud-detection-system/issues)

---

## License

This project is licensed under the MIT License. See LICENSE file for details.

---

**Last Updated:** November 2025  
**Version:** 1.0  
**Author:** Loyanganba Ngathem
