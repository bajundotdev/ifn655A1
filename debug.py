import pandas as pd
import numpy as np

FOLDER_PATH = 'output'
ACCIDENT_FILENAME = 'accident/accident_cleaned.csv'
VEHICLE_FILENAME = 'vehicle/vehicle-1_cleaned_v2.csv'

def simulate_adf_debug_issue():
    print("\n========================================================")
    print(" ADF DATA FLOW DEBUG SIMULATOR ")
    print("========================================================\n")

    print("1. Loading full datasets...")
    try:
        accident_df = pd.read_csv(f"{FOLDER_PATH}/{ACCIDENT_FILENAME}", encoding='unicode_escape', low_memory=False)
        vehicle_df = pd.read_csv(f"{FOLDER_PATH}/{VEHICLE_FILENAME}", encoding='unicode_escape', low_memory=False)
    except FileNotFoundError:
        print(f"Error: Could not find files in {FOLDER_PATH}/. Please ensure they exist.")
        return

    print(f"   -> Accidents loaded: {len(accident_df):,} rows")
    print(f"   -> Vehicles loaded:  {len(vehicle_df):,} rows\n")

    print("2. Simulating ADF's 1,000-row debug sampling...")
    
    # We take a random sample just like ADF does when you click "Data Preview"
    sample_size = 1000
    accident_sample = accident_df.sample(n=sample_size, random_state=42)
    vehicle_sample = vehicle_df.sample(n=sample_size, random_state=99) # Different random state mimics independent source sampling

    print("3. Attempting the Inner Join on the random samples (joinAll)...")
    
    # Extract just the ACCIDENT_NO sets from our tiny samples
    accident_sample_ids = set(accident_sample['ACCIDENT_NO'])
    vehicle_sample_ids = set(vehicle_sample['ACCIDENT_NO'])
    
    # Find the intersection (the Inner Join)
    overlap = accident_sample_ids.intersection(vehicle_sample_ids)
    
    print(f"   -> Unique Accident IDs in Source 1 sample: {len(accident_sample_ids)}")
    print(f"   -> Unique Accident IDs in Source 2/3 sample: {len(vehicle_sample_ids)}")
    print(f"   -> MATCHING IDs FOUND (The Data Preview): {len(overlap)}")
    
    if len(overlap) == 0:
        print("\n   [RESULT]: The Data Preview is empty! This proves why your ADF preview fails.")
        print("   Because ADF grabbed 1,000 random rows from 200,000+ total rows independently,")
        print("   the mathematical chance of grabbing the exact same crash in both samples is near 0%.")

    print("\n========================================================")
    print(" FINDING VALID IDs FOR YOUR ADF DEBUG FILTER ")
    print("========================================================\n")
    
    print("4. Scanning full database for perfect overlaps...")
    # Find IDs that exist in BOTH full datasets
    full_accident_ids = set(accident_df['ACCIDENT_NO'])
    full_vehicle_ids = set(vehicle_df['ACCIDENT_NO'])
    
    full_overlap = full_accident_ids.intersection(full_vehicle_ids)
    
    # Convert to list so we can grab a few examples
    valid_ids_list = list(full_overlap)
    
    if len(valid_ids_list) > 0:
        print(f"   -> Found {len(valid_ids_list):,} crashes that exist perfectly in all tables!")
        print("   -> Copy and paste any of these into your ADF Source Debug Filter:")
        
        # Print 5 guaranteed valid IDs
        for i in range(min(5, len(valid_ids_list))):
            print(f"      ACCIDENT_NO == '{valid_ids_list[i]}'")
    else:
        print("   -> CRITICAL ERROR: No overlapping IDs found. Your datasets are completely disconnected.")

if __name__ == "__main__":
    simulate_adf_debug_issue()