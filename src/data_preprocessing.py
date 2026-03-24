from importing_lib import *
from data_ingestion import main as ingestion_main
import os

# ── Function 1: Scale Data ───────────────────────────────────────
def scale_data(df):
    scaleIt = MinMaxScaler()
    
    columns_to_be_scaled = [c for c in df.columns if df[c].max() > 1]
    print("Columns to be scaled:", columns_to_be_scaled)
    print("************************")
    
    scaled_columns = scaleIt.fit_transform(df[columns_to_be_scaled])
    scaled_columns = pd.DataFrame(scaled_columns, columns=columns_to_be_scaled)
    scaled_columns['Outcome'] = df['Outcome'].values
    
    return scaled_columns

# ── Function 2: Save Processed Data ─────────────────────────────
def save_processed_data(scaled_train, scaled_test):
    # Create processed folder if it doesn't exist
    processed_path = os.path.join(os.path.dirname(__file__), '..', 'processed')
    os.makedirs(processed_path, exist_ok=True)
    
    # Save both files
    scaled_train.to_csv(os.path.join(processed_path, 'train_processed.csv'), index=False)
    scaled_test.to_csv(os.path.join(processed_path, 'test_processed.csv'), index=False)
    
    print(f"Processed train data size : {scaled_train.shape}")
    print(f"Processed test data size  : {scaled_test.shape}")
    print(f"Processed data saved to   : {processed_path}")
    print("************************")

# ── Main Function ────────────────────────────────────────────────
def main():
    # Get train and test data from data_ingestion pipeline
    print("========== STEP 1: Loading Train & Test Data ==========")
    _, train_data, test_data = ingestion_main()
    
    print("========== STEP 2: Scaling Train Data ==========")
    scaled_train = scale_data(train_data)
    
    print("========== STEP 3: Scaling Test Data ==========")
    scaled_test = scale_data(test_data)
    
    print("========== STEP 4: Saving Processed Data ==========")
    save_processed_data(scaled_train, scaled_test)
    
    return scaled_train, scaled_test

# ── Call Main ────────────────────────────────────────────────────
if __name__ == "__main__":
    scaled_train, scaled_test = main()