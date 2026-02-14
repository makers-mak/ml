"""
Download and prepare test data for the Streamlit app
"""
import pandas as pd
from utils import COLUMN_NAMES

# Download test data
print("Downloading test data from UCI repository...")
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.test"

try:
    df = pd.read_csv(url, names=COLUMN_NAMES, na_values=' ?', skipinitialspace=True, skiprows=1)
    
    # Clean the data
    df = df.dropna()
    
    # Clean income column (remove periods)
    if df['income'].dtype == object:
        df['income'] = df['income'].str.replace('.', '', regex=False)
    
    # Save
    output_file = "test_data.csv"
    df.to_csv(output_file, index=False)
    
    print(f"✓ Test data downloaded successfully!")
    print(f"✓ Saved as: {output_file}")
    print(f"✓ Rows: {len(df)}")
    print(f"✓ Columns: {len(df.columns)}")
    print(f"\nColumn names:")
    print(df.columns.tolist())
    print(f"\nFirst few rows:")
    print(df.head())
    print(f"\nYou can now upload '{output_file}' in the Streamlit app!")
    
except Exception as e:
    print(f"Error downloading test data: {e}")
    print("\nAlternative: Use a portion of training data as test")
    print("Run: python app.py --train")
    print("Then manually split the training data for testing")
