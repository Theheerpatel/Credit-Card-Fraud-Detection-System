# Credit-Card-Fraud-Detection-System
Deep Learning based Credit Card Fraud Detection System

# Credit Card Fraud Detection using Deep Learning

## 📋 Project Overview

This project implements a **Deep Learning-based Credit Card Fraud Detection System** that can identify fraudulent transactions in real-time. The system uses Artificial Neural Networks (ANN) to analyze transaction patterns and distinguish between legitimate and fraudulent behavior with high accuracy.

### 🎯 Project Goals
- **Real-time Fraud Detection**: Process transactions in under 1 second
- **High Accuracy**: Achieve ≥95% overall accuracy
- **Balanced Performance**: Maintain high precision and recall for fraud cases
- **Scalability**: Adapt to new fraud patterns through continuous learning

## 📊 Dataset Information

### Source
- **Dataset**: Credit Card Fraud Detection Dataset from Kaggle
- **Link**: [Kaggle Dataset](https://www.kaggle.com/mlg-ulb/creditcardfraud)
- **Transactions**: European cardholders from September 2013

### Dataset Statistics
- **Total Transactions**: 284,807
- **Fraudulent Transactions**: 492 (0.17%)
- **Legitimate Transactions**: 284,315 (99.83%)
- **Features**: 30 numerical attributes (28 PCA components + Time + Amount)
- **Target Variable**: Class (0 = Legitimate, 1 = Fraud)

## 🏗️ Model Architecture

### Deep Neural Network Structure


Input Layer (30 features)
↓
Dense Layer (64 neurons, ReLU) + Dropout (0.3)
↓
Dense Layer (32 neurons, ReLU) + Dropout (0.3)
↓
Dense Layer (16 neurons, ReLU) + Dropout (0.2)
↓
Output Layer (1 neuron, Sigmoid)


### Key Components
- **Activation**: ReLU for hidden layers, Sigmoid for output
- **Regularization**: Dropout layers to prevent overfitting
- **Optimizer**: Adam optimizer
- **Loss Function**: Binary Cross-Entropy
- **Class Imbalance Handling**: SMOTE (Synthetic Minority Over-sampling Technique)

## 📈 Performance Results

### Model Performance Metrics
| Metric | Value | Goal | Status |
|--------|-------|------|---------|
| **Accuracy** | 99.88% | ≥95% | ✅ **Achieved** |
| **Precision** | 61.03% | ≥90% | ⚠️ **Needs Improvement** |
| **Recall** | 84.69% | ≥85% | ✅ **Achieved** |
| **F1-Score** | 70.94% | ≥90% | ⚠️ **Needs Improvement** |
| **ROC-AUC** | 97.44% | ≥95% | ✅ **Achieved** |

### Detailed Classification Report

| Class       | Precision | Recall | F1-Score | Support |
|-------------|-----------|--------|----------|---------|
| Legitimate  | 1.00      | 1.00   | 1.00     | 56,864  |
| Fraud       | 0.61      | 0.85   | 0.71     | 98      |


|               | Precision | Recall | F1-Score | Support |
|---------------|-----------|--------|----------|---------|
| Accuracy      | -         | -      | 1.00     | 56,962  |
| Macro Avg     | 0.81      | 0.92   | 0.85     | 56,962  |
| Weighted Avg  | 1.00      | 1.00   | 1.00     | 56,962  |

### Confusion Matrix Analysis
- **True Negatives**: 56,830 (Correctly identified legitimate transactions)
- **False Positives**: 34 (Legitimate transactions flagged as fraud)
- **False Negatives**: 15 (Fraud transactions missed)
- **True Positives**: 83 (Correctly identified fraud transactions)


## 🧪 Usage

### ✅ Running in Google Colab (Recommended)

1. Upload the notebook file: `credit_card_fraud_detection.ipynb` to Google Colab.
2. Upload the dataset file: `creditcard.csv` to the Colab environment.
3. Run all the cells sequentially.

The model will automatically:

- Load and preprocess the dataset
- Handle class imbalance using **SMOTE**
- Train the deep learning model
- Evaluate performance metrics
- Save the trained model for future inference


## 🔧 Key Features

### ✅ Data Preprocessing
- **Feature Scaling:** StandardScaler for normalization
- **Class Imbalance Handling:** SMOTE oversampling
- **Train-Test Split:** 80–20 split with stratification

### ✅ Model Training
- **Early Stopping:** Prevents overfitting
- **Learning Rate Reduction:** Adaptive learning rate scheduling
- **Batch Training:** Efficient memory usage for large datasets

### ⚡ Real-time Capabilities
- **Fast Inference:** < 1 second per transaction
- **Scalable Architecture:** Handles high-volume transactions
- **Continuous Learning:** Can be retrained on new data


## 📚 Technical Details

### 🧩 Libraries Used

| Category          | Libraries                 | Purpose                                        |
|-------------------|----------------------------|------------------------------------------------|
| Data Handling     | pandas, numpy             | Data manipulation & numerical operations       |
| Visualization     | matplotlib, seaborn       | Data exploration & result plotting             |
| Preprocessing     | scikit-learn              | Scaling, splitting, metrics                    |
| Deep Learning     | TensorFlow, Keras         | Neural network model building                  |
| Imbalance         | imbalanced-learn          | SMOTE for class balancing                      |

### ⚙️ Model Configuration
- **Batch Size:** 256
- **Epochs:** 100 (with early stopping)
- **Validation Split:** 20%
- **Random State:** 42 (for reproducibility)


## 🎯 Business Impact

### 💰 Benefits
- **Financial Loss Reduction:** Early detection minimizes monetary damage
- **Enhanced Security:** Identifies complex fraud patterns
- **Customer Trust:** Secure transactions build loyalty
- **Regulatory Compliance:** Meets financial security standards

### 🏦 Applications
- Banks & Financial Institutions
- E-commerce Payment Gateways
- Mobile Payment Applications
- Credit Card Issuers


## 🔮 Future Enhancements

### 🚀 Planned Improvements
- **Ensemble Methods:** Combine models for improved accuracy
- **Real-time Deployment:** API development for production use
- **Anomaly Detection:** Unsupervised models for unseen fraud patterns
- **Feature Engineering:** Additional transaction metadata
- **Model Explainability:** SHAP values for interpretability

### ⚙️ Performance Optimization
- **Hyperparameter Tuning:** Grid search for optimal parameters
- **Alternative Architectures:** LSTM for sequential pattern learning
- **Cost-sensitive Learning:** Weighted loss for fraud cases


## 👥 Team Members
- **Gauri Rana** (22000382)
- **Heer Patel** (22000384)
- **Dhruvi Patel** (22000402)
- **Zainab Khokhawala** (22000425)

**Navrachana University**  
Department of Computer Science and Engineering  
Bachelor of Technology (CSE) – 4th Year  
**Autumn Semester (July–Nov 2025)**


## 🙏 Acknowledgments

- **Kaggle** for providing the credit card fraud detection dataset
- **TensorFlow** and **Keras** teams for their excellent deep learning tools
- **Navrachana University** for academic guidance and support
