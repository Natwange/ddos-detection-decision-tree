# Real Dataset Setup Guide for DDoS Detection

## Recommended Real Datasets

### 1. CIC-IDS2017 (Recommended)
- **Size**: ~2.8 GB
- **Records**: ~2.8 million network flows
- **Features**: 78 features
- **Attack Types**: DDoS, DoS, Brute Force, XSS, SQL Injection, Infiltration, Botnet
- **Download**: https://www.unb.ca/cic/datasets/ids-2017.html

### 2. NSL-KDD
- **Size**: ~25 MB
- **Records**: ~125,000 records
- **Features**: 41 features
- **Attack Types**: DoS, Probe, R2L, U2R
- **Download**: https://www.unb.ca/cic/datasets/nsl.html

### 3. CICDDoS2019 (Alternative)
- **Size**: ~16 GB
- **Records**: ~5+ million records
- **Features**: 88 features
- **Attack Types**: Multiple DDoS variants
- **Download**: https://www.unb.ca/cic/datasets/ddos-2019.html

## Quick Setup Instructions

### Option 1: NSL-KDD (Easiest - Small Dataset)
```bash
# Download NSL-KDD
wget https://www.unb.ca/cic/datasets/nsl.html
# Extract the files and use KDDTrain+.txt and KDDTest+.txt
```

### Option 2: CIC-IDS2017 (Recommended - Realistic Size)
```bash
# Download CIC-IDS2017
# Go to: https://www.unb.ca/cic/datasets/ids-2017.html
# Download the zip file and extract
```

## File Structure After Download

### CIC-IDS2017 Structure:
```
CIC-IDS2017/
├── MachineLearningCSV/
│   ├── MachineLearningCVE.csv
│   ├── Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv
│   ├── Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv
│   ├── Friday-WorkingHours-Morning.pcap_ISCX.csv
│   ├── Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv
│   ├── Friday-WorkingHours-Evening.pcap_ISCX.csv
│   └── ...
```

### NSL-KDD Structure:
```
NSL-KDD/
├── KDDTrain+.txt
├── KDDTest+.txt
├── KDDTrain+_20Percent.txt
└── KDDTest-21.txt
```

## How to Use with Your Algorithm

### For CIC-IDS2017:
```bash
# Use the main CSV file
python decision_tree.py --csv "CIC-IDS2017/MachineLearningCSV/MachineLearningCVE.csv" --label Label --top_k 20 --max_depth 8 --n_folds 5
```

### For NSL-KDD:
```bash
# First convert to CSV format, then use
python decision_tree.py --csv "NSL-KDD/KDDTrain+.csv" --label label --top_k 15 --max_depth 6 --n_folds 5
```

## Expected Results with Real Datasets

### CIC-IDS2017:
- **Training Size**: ~2.2M samples (80%)
- **Test Size**: ~560K samples (20%)
- **Expected Accuracy**: 85-95%
- **Expected F1-Score**: 80-90%
- **Processing Time**: 10-30 minutes

### NSL-KDD:
- **Training Size**: ~100K samples (80%)
- **Test Size**: ~25K samples (20%)
- **Expected Accuracy**: 90-98%
- **Expected F1-Score**: 85-95%
- **Processing Time**: 2-5 minutes

## Memory Requirements

### Minimum Requirements:
- **RAM**: 8GB (for CIC-IDS2017), 4GB (for NSL-KDD)
- **Storage**: 5GB free space
- **CPU**: Any modern processor

### Recommended:
- **RAM**: 16GB
- **Storage**: 10GB free space
- **CPU**: Multi-core processor

## Troubleshooting

### If you get memory errors:
1. Reduce the dataset size by sampling
2. Use fewer features (reduce --top_k)
3. Use a smaller tree depth (reduce --max_depth)

### If download is slow:
1. Use a university network
2. Download during off-peak hours
3. Use a download manager

## Alternative: Use Sample of Real Dataset

If you can't download the full dataset, I can help you create a larger synthetic dataset that mimics the real one:

```bash
# Generate a larger synthetic dataset (10K samples)
python test_real_dataset.py  # This creates realistic_network_data.csv with 2000 samples
# We can modify it to create 10K+ samples
```

## What Your Professor Will See

With a real dataset, your results will show:
1. **Large Training Set**: 80% of millions of records
2. **Realistic Performance**: Not 100% accuracy (more believable)
3. **Proper Cross-Validation**: 5-fold CV on large dataset
4. **Feature Selection**: Chi-square on real network features
5. **Normalization**: MinMax scaling on real data
6. **Comprehensive Metrics**: F1, Precision, Recall with confidence intervals

This demonstrates real-world machine learning skills!
