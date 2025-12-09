# DDoS Detection Using Decision Trees

A machine learning project that implements a Decision Tree-based algorithm for detecting DDoS (Distributed Denial of Service) attacks in network traffic data.

## 📋 Project Overview

This project implements a complete machine learning pipeline for DDoS attack detection using a CART Decision Tree classifier. It includes data preprocessing, feature selection, model training, cross-validation, and comprehensive evaluation metrics.

## ✨ Features

- **Data Preprocessing**: Handles missing values, categorical encoding, and normalization
- **Feature Selection**: Chi-square test to select top-K most relevant features
- **Model Training**: CART Decision Tree with configurable depth for interpretability
- **Cross-Validation**: K-fold stratified cross-validation for robust performance estimation
- **Comprehensive Evaluation**: Accuracy, Precision, Recall, F1-Score, and Confusion Matrix
- **Model Persistence**: Save and load trained models for future use
- **Visualization**: Confusion matrix visualization

## 🚀 Installation

1. Clone this repository:
```bash
git clone <your-repo-url>
cd pythonProject2
```

2. Install required dependencies:
```bash
# On Windows:
python -m pip install -r requirements.txt

# On Linux/Mac:
pip install -r requirements.txt
```

## 📊 Usage

### Basic Usage

```bash
python decision_tree.py --csv path/to/dataset.csv --label Label --top_k 20 --max_depth 8
```

### Command Line Arguments

- `--csv` (required): Path to input CSV dataset
- `--label` (optional): Name of the label column (auto-detected if not provided)
- `--test_size` (optional): Test split fraction (default: 0.2 = 80% train, 20% test)
- `--top_k` (optional): Select top-K features via chi-square (default: 20)
- `--max_depth` (optional): Max depth of the Decision Tree (default: 8)
- `--n_folds` (optional): Number of folds for cross-validation (default: 5)

### Example

```bash
python decision_tree.py --csv large_synthetic_dataset.csv --label label --top_k 10 --max_depth 4 --n_folds 5
```

## 📁 Project Structure

```
pythonProject2/
├── decision_tree.py          # Main implementation
├── requirements.txt          # Python dependencies
├── README.md                 # This file
├── .gitignore               # Git ignore rules
├── PSEUDOCODE.md            # Algorithm pseudocode
├── EVALUATION_FORMULAS.md   # Mathematical formulas
├── PROFESSOR_SUMMARY.md     # Project summary
└── [data files]             # Dataset files (not tracked in git)
```

## 🔧 Technical Details

### Algorithm Components

1. **Data Loading**: CSV file processing with automatic label detection
2. **Preprocessing**: 
   - Missing value imputation (median for numeric, mode for categorical)
   - MinMax normalization for numeric features
   - One-hot encoding for categorical features
3. **Feature Selection**: Chi-square test to select top-K most relevant features
4. **Model Training**: CART Decision Tree with limited depth for interpretability
5. **Evaluation**: Comprehensive metrics with confusion matrix visualization
6. **Model Persistence**: Saved model for future use

### Model Characteristics

- **Algorithm**: CART Decision Tree (Gini impurity)
- **Class Weight**: Balanced (handles imbalanced datasets)
- **Feature Selection**: Chi-square test
- **Cross-Validation**: Stratified K-fold

## 📈 Evaluation Metrics

The project evaluates models using:
- **Accuracy**: Overall correctness
- **Precision**: True positives / (True positives + False positives)
- **Recall**: True positives / (True positives + False negatives)
- **F1-Score**: Harmonic mean of precision and recall
- **Confusion Matrix**: Visual representation of classification results

## 📝 Output Files

After running the script, the following files are generated:
- `ddos_decision_tree.joblib`: Trained model and preprocessing pipeline
- `confusion_matrix.png`: Confusion matrix visualization

## 📚 Documentation

Additional documentation files:
- `PSEUDOCODE.md`: Complete algorithm pseudocode
- `EVALUATION_FORMULAS.md`: Mathematical formulas and examples
- `PROFESSOR_SUMMARY.md`: Detailed project summary
- `REAL_DATASET_GUIDE.md`: Guide for using real datasets
- `PRESENTATION_GUIDE.md`: Presentation tips and explanations

## 🎓 Educational Value

This project demonstrates:
- Data preprocessing and feature engineering
- Machine learning model training and evaluation
- Cross-validation for robust performance estimation
- Model interpretation and visualization
- Best practices in ML project organization

## 🔬 Requirements

- Python 3.7+
- scikit-learn
- pandas
- numpy
- matplotlib
- joblib

See `requirements.txt` for specific versions.

## 📄 License

This project is part of a college assignment. Please use responsibly and cite appropriately if used for academic purposes.

## 🤝 Contributing

This is an academic project. For questions or suggestions, please open an issue.

## 📧 Contact

For questions about this project, please refer to the documentation files or open an issue.

