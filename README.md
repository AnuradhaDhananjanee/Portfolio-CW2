# Portfolio CW2 — E-Commerce Shipping Analysis

## Option 2: Real-World Data Analysis & Business Problem Solving

---

## Business Problem
An e-commerce company is experiencing high late delivery rates, leading to poor customer satisfaction. This project builds a machine learning model to **predict late deliveries** and identify the key factors causing them.

---

## Project Structure

```
Portfolio_CW2/
├── data/
│   └── E Commerce.csv          # Raw dataset from Kaggle
├── notebooks/
│   ├── 01_data_collection.ipynb  # ETL & Data Loading
│   ├── 02_eda.ipynb              # Exploratory Data Analysis
│   ├── 03_clustering.ipynb       # K-Means Clustering
│   └── 04_model.ipynb            # Predictive Model
├── outputs/
│   ├── cleaned_data.csv          # Cleaned dataset
│   ├── clustered_data.csv        # Dataset with cluster labels
│   └── *.png                     # All saved charts
├── docs/
│   ├── methodology.md            # Project methodology
│   └── cloud_setup.md            # Azure cloud setup notes
├── reports/
│   └── final_report.md           # Final report
├── README.md
└── requirements.txt
```

---

## Dataset
- **Source:** Kaggle — E-Commerce Shipping Dataset by Prachi13
- **Link:** https://www.kaggle.com/datasets/prachi13/customer-churn-and-shipping
- **Size:** ~11,000 rows, 12 columns
- **Target:** Predict late delivery (1=Late, 0=On Time)

---

## How to Run

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Run notebooks in order
```
01_data_collection.ipynb  → First
02_eda.ipynb              → Second
03_clustering.ipynb       → Third
04_model.ipynb            → Fourth
```

---

## Cloud Integration (Azure)
- **Azure Blob Storage** — Raw dataset storage
- **Azure SQL Database** — Structured data storage
- **Azure VM** — Computation environment
- **Azure ML** — Model deployment (optional)

---

## Results Summary
| Model | Accuracy |
|-------|----------|
| Logistic Regression | ~67% |
| Random Forest | ~69% |

---

## Tools & Technologies
- Python 3.9+
- Pandas, NumPy, Matplotlib, Seaborn
- Scikit-learn (Random Forest, KMeans, Logistic Regression)
- Jupyter Notebooks
- GitHub (Version Control)
- Azure (Cloud Integration)
