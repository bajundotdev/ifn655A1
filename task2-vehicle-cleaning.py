from pathlib import Path
import numpy as np
import pandas as pd

# ---------------------------------------------------------
# 1. Load Data
# ---------------------------------------------------------
data_path = Path("data/vehicle-original.csv")
df = pd.read_csv(data_path, encoding="unicode_escape", low_memory=False)

# ---------------------------------------------------------
# 2. Outlier Detection & Replacement (NaN)
# ---------------------------------------------------------
# Mask invalid manufacturing years (< 1900 or > 2025)
year_outliers_mask = (~df["VEHICLE_YEAR_MANUF"].between(1900, 2025)) & df["VEHICLE_YEAR_MANUF"].notna()
print(f"Identified {year_outliers_mask.sum():,} manufacturing year outliers. Setting to NaN...")
df.loc[year_outliers_mask, "VEHICLE_YEAR_MANUF"] = np.nan

# ---------------------------------------------------------
# 3. Hierarchical Median Imputation for Manufacturing Year
# ---------------------------------------------------------
# Step 1: Median by [VEHICLE_MAKE, VEHICLE_MODEL]
model_median = df.groupby(["VEHICLE_MAKE", "VEHICLE_MODEL"])["VEHICLE_YEAR_MANUF"].transform("median")

# Step 2: Fallback to VEHICLE_MAKE median (for models with no valid years)
make_median = df.groupby("VEHICLE_MAKE")["VEHICLE_YEAR_MANUF"].transform("median")

# Step 3: Global dataset median fallback
global_median = df["VEHICLE_YEAR_MANUF"].median()

df["VEHICLE_YEAR_MANUF"] = (
    df["VEHICLE_YEAR_MANUF"]
    .fillna(model_median)
    .fillna(make_median)
    .fillna(global_median)
    .round()
)

# ---------------------------------------------------------
# 4. Reference Lookups
# ---------------------------------------------------------
wheel_map = {
    1: 4, 2: 4, 3: 4, 4: 4, 5: 4, 6: 6, 7: 6, 8: 4, 9: 4,
    10: 2, 11: 2, 12: 2, 13: 2, 14: 0, 15: 0, 16: 0, 17: 4,
    18: 4, 19: 4, 20: 4, 21: 2, 27: 4, 60: 6, 61: 6, 62: 6,
    63: 6, 71: 4, 72: 6, 99: 4
}

impact_map = {
    "F": 1,
    "R": 8,
    "N": 9,
    "S": 6,
    "T": 9,
}

traffic_control_map = {
    0: "No control", 1: "Stop-go lights", 2: "Flashing lights",
    3: "Out of order", 4: "Ped. lights", 5: "Ped. crossing",
    6: "RX Gates/Booms", 7: "X Bells/lights", 8: "RX no control",
    9: "Roundabout", 10: "Stop sign", 11: "giveaway sign",
    12: "School flags", 13: "School No Flags", 14: "Police",
    15: "Other", 99: "Unknown",
}
traffic_desc_to_code = {v: k for k, v in traffic_control_map.items()}

# ---------------------------------------------------------
# 5. Data Cleaning & Remaining Imputation
# ---------------------------------------------------------
# Impute missing wheels using VEHICLE_TYPE lookup
imputed_wheels = df["VEHICLE_TYPE"].map(wheel_map)
df["NO_OF_WHEELS"] = df["NO_OF_WHEELS"].fillna(imputed_wheels)

# Standardize INITIAL_IMPACT (convert mixed strings to numeric codes)
df["INITIAL_IMPACT"] = df["INITIAL_IMPACT"].replace(impact_map)
df["INITIAL_IMPACT"] = pd.to_numeric(df["INITIAL_IMPACT"], errors="coerce")

# Reconcile TRAFFIC_CONTROL and TRAFFIC_CONTROL_DESC bidirectionally
df["TRAFFIC_CONTROL"] = df["TRAFFIC_CONTROL"].fillna(
    df["TRAFFIC_CONTROL_DESC"].map(traffic_desc_to_code)
).fillna(0)

df["TRAFFIC_CONTROL_DESC"] = df["TRAFFIC_CONTROL_DESC"].fillna(
    df["TRAFFIC_CONTROL"].map(traffic_control_map)
)

# Impute cylinders with scalar mode
cyl_mode = df["NO_OF_CYLINDERS"].mode()[0]
df["NO_OF_CYLINDERS"] = df["NO_OF_CYLINDERS"].fillna(cyl_mode)

# Drop redundant or heavily missing columns
COLS_TO_DROP = ["VEHICLE_POWER", "CARRY_CAPACITY", "CUBIC_CAPACITY", "VEHICLE_COLOUR_2"]
dropped_df = df[COLS_TO_DROP].copy()
df = df.drop(columns=COLS_TO_DROP)

# ---------------------------------------------------------
# 6. Automated Data Profiling
# ---------------------------------------------------------
print("\n" + "=" * 60)
print("DATA PROFILE SUMMARY")
print("=" * 60)
print(f"Total Records : {len(df):,}")
print(f"Total Columns : {df.shape[1]}")
print(f"Duplicate Rows: {df.duplicated().sum():,}\n")

# Tabular missing value overview
missing_summary = pd.DataFrame({
    "Missing Count": df.isna().sum(),
    "Missing Pct (%)": (df.isna().mean() * 100).round(2),
    "Dtype": df.dtypes
})
print(missing_summary[missing_summary["Missing Count"] > 0].sort_values("Missing Count", ascending=False))

# Numerical distribution profile
print("\nNumerical Feature Summary:")
print(df.describe().T[["count", "mean", "std", "min", "50%", "max"]])

# ---------------------------------------------------------
# 7. Export Cleaned & Dropped Data
# ---------------------------------------------------------
output_dir = Path("output/vehicle")
output_dir.mkdir(parents=True, exist_ok=True)

dropped_df.to_csv(output_dir / "vehicle_dropped.csv", index=False)
df.to_csv(output_dir / "vehicle_cleaned.csv", index=False)
print(f"\nFiles exported successfully to '{output_dir}/'.")


# To be deleted later
print("----------")
nulls = df.isna().sum()
print(nulls)