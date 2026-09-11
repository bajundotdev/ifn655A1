import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# 1. Load the dataset
df = pd.read_csv("data/vehicle-original.csv", encoding="unicode_escape", low_memory=False)

# ----------------------------------------------------
# Step 1: Basic Structure & Dimensions
# ----------------------------------------------------
print("--- 1. DATASET DIMENSIONS ---")
print(f"Total Rows: {df.shape[0]:,}")
print(f"Total Columns: {df.shape[1]}")

print("\n--- DATA TYPES & NON-NULL COUNTS ---")
df.info()

# ----------------------------------------------------
# Step 2: Missing Values Check
# ----------------------------------------------------
print("\n--- 2. MISSING VALUES PER COLUMN ---")
missing_counts = df.isnull().sum()
# Filter to display only columns that contain missing entries
print(missing_counts[missing_counts > 0].sort_values(ascending=False))

# ----------------------------------------------------
# Step 3: Numeric Summary (Min, Max, Mean, Median)
# ----------------------------------------------------
print("\n--- 3. NUMERICAL STATISTICS ---")
# describe() includes count, mean, std, min, 25%, 50% (median), 75%, and max
print(df.describe().round(2))

# ----------------------------------------------------
# Step 4: Frequent Categories
# ----------------------------------------------------
print("\n--- 4. TOP CATEGORIES ---")
print("\nTop 5 Vehicle Types:")
# print(df["VEHICLE_TYPE_DESC"].value_counts().head(5))
print('\nDistinct values: ', df["VEHICLE_TYPE_DESC"].nunique())
print(df["VEHICLE_TYPE_DESC"].value_counts().sort_index())

print("\nTop 5 Vehicle Makes:")
# print(df["VEHICLE_MAKE"].value_counts().head(5))
print(df["VEHICLE_MAKE"].value_counts())
