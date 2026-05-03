from pathlib import Path
import pandas as pd


DATA_PATH = Path("data/raw/telco_churn.csv")


def load_raw_data():
    df = pd.read_csv(DATA_PATH)
    return df