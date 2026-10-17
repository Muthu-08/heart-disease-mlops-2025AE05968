import urllib.request
import pandas as pd

UCI_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data"
COLUMNS = [
    "age", "sex", "cp", "trestbps", "chol", "fbs", 
    "restecg", "thalach", "exang", "oldpeak", "slope", 
    "ca", "thal", "target"
]

def download_and_clean_data(output_path="data/processed_heart.csv"):
    print("Downloading UCI Heart Disease dataset...")
    df = pd.read_csv(UCI_URL, header=None, names=COLUMNS, na_values="?")
    
    # Handle missing values (ca and thal contain a small count of '?')
    df["ca"] = df["ca"].fillna(df["ca"].median())
    df["thal"] = df["thal"].fillna(df["thal"].mode()[0])
    
    # Binarize target: 0 = No Disease, 1..4 = Heart Disease Present
    df["target"] = (df["target"] > 0).astype(int)
    
    df.to_csv(output_path, index=False)
    print(f"Data successfully cleaned and saved to {output_path}")

if __name__ == "__main__":
    download_and_clean_data()