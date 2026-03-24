from importing_lib import *
import os
from sklearn.model_selection import train_test_split

# ── Function 1: Read Data ────────────────────────────────────────
def read_data():
    df = pd.read_csv("diabetes.csv", index_col=False)
    
    print("First 5 rows:")
    print(df.head())
    print("************************")
    
    print("Data Info:")
    print(df.info())
    print("************************")
    
    return df

# ── Function 2: Check & Remove Nulls and Duplicates ─────────────
def clean_data(df):
    print("Null values per column:")
    print(df.isnull().sum())
    print("************************")
    
    print(f"Duplicates found: {df.duplicated().sum()}")
    df = df.drop_duplicates()
    print(f"Duplicates after removal: {df.duplicated().sum()}")
    print("************************")
    
    return df

# ── Function 3: Save Raw Train/Test Split Data ───────────────────
def save_raw_data(df):
    # Create raw_data folder if it doesn't exist
    raw_data_path = os.path.join(os.path.dirname(__file__), '..', 'raw_data')
    os.makedirs(raw_data_path, exist_ok=True)
    
    # Split data — 80% train, 20% test
    train_data, test_data = train_test_split(df, test_size=0.2, random_state=42)
    
    # Save to raw_data folder
    train_data.to_csv(os.path.join(raw_data_path, 'train_data.csv'), index=False)
    test_data.to_csv(os.path.join(raw_data_path, 'test_data.csv'), index=False)
    
    print(f"Train data size: {train_data.shape}")
    print(f"Test data size:  {test_data.shape}")
    print(f"Raw data saved to: {raw_data_path}")
    print("************************")
    
    return train_data, test_data

# ── Main Function ────────────────────────────────────────────────
def main():
    print("========== STEP 1: Reading Data ==========")
    df = read_data()
    
    print("========== STEP 2: Cleaning Data ==========")
    df = clean_data(df)
    
    print("========== STEP 3: Saving Raw Data ==========")
    train_data, test_data = save_raw_data(df)
    
    return df, train_data, test_data

# ── Call Main ────────────────────────────────────────────────────
if __name__ == "__main__":
    df, train_data, test_data = main()