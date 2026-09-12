"""
Task 2 - Vehicle data cleaning.

Convention used throughout this script:
  A CATEGORY that is blank becomes an explicit category, because an unknown is
  a real fact about the crash and should stay visible.
  A MEASURE that is blank stays blank, because any number invented for it gets
  arithmetic done to it later.
"""
from pathlib import Path
import numpy as np
import pandas as pd

# ---------------------------------------------------------
# 1. Load Data
# ---------------------------------------------------------
data_path = Path("data/vehicle-original.csv")
df = pd.read_csv(data_path, encoding="unicode_escape", low_memory=False)

raw_row_count = len(df)
raw_occupants = df["TOTAL_NO_OCCUPANTS"].sum()
print(f'{data_path.name} loaded: {raw_row_count:,} rows x {df.shape[1]} columns')

# ---------------------------------------------------------
# 2. Manufacturing years that are not real years
# ---------------------------------------------------------
# between() is INCLUSIVE, so 1900 has to sit OUTSIDE the range. The dictionary
# gives this column no unknown code, and 1900 is used as one on 253 rows.
year_outliers_mask = (~df["VEHICLE_YEAR_MANUF"].between(1901, 2025)) & df["VEHICLE_YEAR_MANUF"].notna()
print(f'\nManufacturing years that are not real years: {year_outliers_mask.sum():,}')
df.loc[year_outliers_mask, "VEHICLE_YEAR_MANUF"] = np.nan

# ---------------------------------------------------------
# 3. VEHICLE_YEAR_MANUF is left missing, not imputed
# ---------------------------------------------------------
# Table 6 measures vehicle age. Filling the age column with the median age of
# other vehicles would make part of the answer a product of the fill.
missing_year = df["VEHICLE_YEAR_MANUF"].isnull().sum()
print(f'VEHICLE_YEAR_MANUF left missing: {missing_year:,} ({missing_year / len(df) * 100:.2f}%)')

# ---------------------------------------------------------
# 4. A zero weight is not a weight
# ---------------------------------------------------------
zero_tare = (df["TARE_WEIGHT"] == 0)
df.loc[zero_tare, "TARE_WEIGHT"] = np.nan
print(f'\nTARE_WEIGHT zeros set to missing: {zero_tare.sum():,}')
print(f'TARE_WEIGHT now missing: {df["TARE_WEIGHT"].isnull().sum():,}')

# ---------------------------------------------------------
# 5. Reference lookups
# ---------------------------------------------------------
# A bicycle has two wheels - that is a fact about bicycles, not a guess about
# this bicycle. Types with no knowable answer are left out on purpose:
# 14 Horse, 15 Tram, 16 Train, 18 Not Applicable, 99 Not Known.
wheel_map = {
    1: 4, 2: 4, 3: 4, 4: 4, 5: 4, 6: 6, 7: 6, 8: 4, 9: 4,
    10: 2, 11: 2, 12: 2, 13: 2, 17: 4, 19: 4, 20: 4, 21: 2,
    27: 4, 60: 6, 61: 6, 62: 6, 63: 6, 71: 4, 72: 6
}

# INITIAL_IMPACT is a position code, not a number. Digits are corner positions,
# letters are whole faces, 0 is a towed unit. Published list, plus two codes
# that occur in the file but not in the dictionary.
initial_impact_desc = {
    "0": "Towed unit",           "8": "Left rear corner",
    "1": "Right front corner",   "9": "Not known",
    "2": "Right side forwards",  "F": "Front",
    "3": "Right side rearwards", "N": "None",
    "4": "Right rear corner",    "R": "Rear",
    "5": "Left front corner",    "S": "Sidecar",
    "6": "Left side forwards",   "T": "Top/roof",
    "7": "Left side rearwards",  "U": "Undercarriage",
    "L": "Undocumented code L",  "M": "Undocumented code M",
    "_": "Not recorded",
}

# These labels are spelled exactly as they appear in TRAFFIC_CONTROL_DESC.
traffic_control_map = {
    0: "No control", 1: "Stop-go lights", 2: "Flashing lights",
    3: "Out of order", 4: "Ped. lights", 5: "Ped. crossing",
    6: "RX Gates/Booms", 7: "RX Bells/Lights", 8: "RX No control",
    9: "Roundabout", 10: "Stop sign", 11: "Giveway sign",
    12: "School Flags", 13: "School No flags", 14: "Police",
    15: "Other", 99: "Unknown",
}
traffic_desc_to_code = {v: k for k, v in traffic_control_map.items()}

# ---------------------------------------------------------
# 6. Cleaning
# ---------------------------------------------------------
imputed_wheels = df["VEHICLE_TYPE"].map(wheel_map)
df["NO_OF_WHEELS"] = df["NO_OF_WHEELS"].fillna(imputed_wheels)
print(f'\nNO_OF_WHEELS left missing: {df["NO_OF_WHEELS"].isnull().sum():,}')

# Keep INITIAL_IMPACT as text. Forcing it to a number turned 135,274 "Front"
# rows into "Right front corner" and silently nulled U, L and M.
df["INITIAL_IMPACT"] = df["INITIAL_IMPACT"].astype(str).str.strip()
df.loc[df["INITIAL_IMPACT"].isin(["nan", "NaN", ""]), "INITIAL_IMPACT"] = np.nan

# 0 means "No control", which is a claim about the scene. It is not a stand-in
# for not knowing, so the chain stops at the lookup.
df["TRAFFIC_CONTROL"] = df["TRAFFIC_CONTROL"].fillna(
    df["TRAFFIC_CONTROL_DESC"].map(traffic_desc_to_code))

df["TRAFFIC_CONTROL_DESC"] = df["TRAFFIC_CONTROL_DESC"].fillna(
    df["TRAFFIC_CONTROL"].map(traffic_control_map))

# NO_OF_CYLINDERS is left missing. The mode is 4, and 19,921 of the missing
# rows are bicycles, which have no engine at all.
print(f'NO_OF_CYLINDERS left missing: {df["NO_OF_CYLINDERS"].isnull().sum():,}')

# VEHICLE_POWER    - 0 populated values in 364,297 rows.
# VEHICLE_COLOUR_2 - 'ZZ' (unknown / not applicable) on 355,821 rows, 97.7%.
# CARRY_CAPACITY and CUBIC_CAPACITY are KEPT: they are missing by design,
# populated for exactly the commercial vehicles Table 6 is about.
COLS_TO_DROP = ["VEHICLE_POWER", "VEHICLE_COLOUR_2"]
dropped_df = df[COLS_TO_DROP].copy()
df = df.drop(columns=COLS_TO_DROP)

# ---------------------------------------------------------
# 7. A blank category becomes an explicit category
# ---------------------------------------------------------
# A blank and a "not known" code are two different facts, and the dictionary
# keeps them apart - REG_STATE has both 'Z' Not known and '_' Blank. So a blank
# gets its own value and sits beside the not-known code instead of replacing it.
NOT_RECORDED_TEXT = "_"
NOT_RECORDED_CODE = -1

text_categories = [
    "INITIAL_DIRECTION", "FINAL_DIRECTION", "REG_STATE", "VEHICLE_BODY_STYLE",
    "VEHICLE_MAKE", "VEHICLE_MODEL", "CONSTRUCTION_TYPE", "FUEL_TYPE",
    "TRAILER_TYPE", "VEHICLE_COLOUR_1", "INITIAL_IMPACT",
    "ROAD_SURFACE_TYPE_DESC", "VEHICLE_TYPE_DESC", "TRAFFIC_CONTROL_DESC",
]

code_categories = [
    "VEHICLE_DCA_CODE", "ROAD_SURFACE_TYPE", "VEHICLE_TYPE", "DRIVER_INTENT",
    "VEHICLE_MOVEMENT", "CAUGHT_FIRE", "LAMPS", "LEVEL_OF_DAMAGE",
    "TOWED_AWAY_FLAG", "TRAFFIC_CONTROL",
]

print("\nLabelling blank categories")
print("-" * 50)
filled_total = 0
for column in text_categories:
    count = df[column].isnull().sum()
    df[column] = df[column].fillna(NOT_RECORDED_TEXT)
    filled_total = filled_total + count
    print(f'{column:<24} blanks labelled: {count:>7,}')

for column in code_categories:
    count = df[column].isnull().sum()
    df[column] = df[column].fillna(NOT_RECORDED_CODE)
    filled_total = filled_total + count
    print(f'{column:<24} blanks labelled: {count:>7,}')

print(f'\nBlank categories labelled in total: {filled_total:,}')

# ---------------------------------------------------------
# 8. Description columns for the codes that will be charted
# ---------------------------------------------------------
# Only where the dictionary gives a COMPLETE code list. VEHICLE_MAKE (686),
# VEHICLE_BODY_STYLE (118) and VEHICLE_MODEL (13,798) stay as codes - the
# published lists are partial, and a half-filled _DESC looks authoritative and
# is mostly blank.
level_of_damage_desc = {
    1: "Minor", 2: "Moderate (driveable)", 3: "Moderate (towed away)",
    4: "Major (towed away)", 5: "Extensive (unrepairable)", 6: "Nil damage",
    9: "Not known", -1: "Not recorded",
}

fuel_type_desc = {
    "D": "Diesel", "E": "Electric", "G": "Gas", "M": "Multi", "P": "Petrol",
    "R": "Rotary", "Z": "Unknown",
    "O": "Undocumented code O", "S": "Undocumented code S",
    "_": "Not recorded",
}

caught_fire_desc = {
    0: "Not applicable", 1: "Yes", 2: "No", 9: "Not known", -1: "Not recorded",
}

trailer_type_desc = {
    "A": "Caravan", "B": "Trailer (general)", "C": "Trailer (boat)",
    "D": "Horse float", "E": "Machinery", "F": "Farm/agricultural equipment",
    "G": "Not known what is being towed", "H": "Not applicable",
    "I": "Trailer (Exempt)", "J": "Semi Trailer", "K": "Pig Trailer",
    "L": "Dog Trailer", "_": "Not recorded",
}

driver_intent_desc = {
    1: "Going straight ahead", 2: "Turning right", 3: "Turning left",
    4: "Leaving a driveway", 5: "U-turning", 6: "Changing lanes",
    7: "Overtaking", 8: "Merging", 9: "Reversing", 10: "Parking or unparking",
    11: "Parked legally", 12: "Parked illegally", 13: "Stationary accident",
    14: "Stationary broken down", 15: "Other stationary", 16: "Avoiding animals",
    17: "Slow/stopping", 18: "Out of control", 19: "Wrong way",
    99: "Not known", -1: "Not recorded",
}

df["LEVEL_OF_DAMAGE_DESC"] = df["LEVEL_OF_DAMAGE"].map(level_of_damage_desc)
df["FUEL_TYPE_DESC"]       = df["FUEL_TYPE"].map(fuel_type_desc)
df["CAUGHT_FIRE_DESC"]     = df["CAUGHT_FIRE"].map(caught_fire_desc)
df["TRAILER_TYPE_DESC"]    = df["TRAILER_TYPE"].map(trailer_type_desc)
df["DRIVER_INTENT_DESC"]   = df["DRIVER_INTENT"].map(driver_intent_desc)
df["INITIAL_IMPACT_DESC"]  = df["INITIAL_IMPACT"].map(initial_impact_desc)

new_desc = ["LEVEL_OF_DAMAGE_DESC", "FUEL_TYPE_DESC", "CAUGHT_FIRE_DESC",
            "TRAILER_TYPE_DESC", "DRIVER_INTENT_DESC", "INITIAL_IMPACT_DESC"]
print("\nCodes no description map covered (should all be 0):")
print(df[new_desc].isnull().sum().to_string())

# ---------------------------------------------------------
# 9. Integers, so the CSV says 2 and not 2.0
# ---------------------------------------------------------
# A column became float64 the moment it held one blank, and fillna() does not
# change a dtype. to_csv() then writes 2.0, which SQL Server cannot convert to
# an integer. These have no blanks left, so a plain int is correct.
for column in code_categories:
    df[column] = df[column].astype("int64")

# The measures still have blanks on purpose, so they need pandas' nullable
# integer type. Int64 writes an empty field for a blank, not the text 'nan'.
integer_measures = [
    "VEHICLE_YEAR_MANUF", "TARE_WEIGHT", "VEHICLE_WEIGHT", "CARRY_CAPACITY",
    "CUBIC_CAPACITY", "NO_OF_WHEELS", "NO_OF_CYLINDERS", "SEATING_CAPACITY",
    "TOTAL_NO_OCCUPANTS",
]
for column in integer_measures:
    df[column] = df[column].astype("Int64")

# ---------------------------------------------------------
# 10. Vehicles whose parent accident was rejected
# ---------------------------------------------------------
# These rows are not faulty. Their parent crash was quarantined during the
# accident cleaning, so the foreign key has nothing to point at. They are set
# aside with the reason recorded, the same way the accident side does it.
parents = pd.read_csv(Path("output/accident/accident_cleaned.csv"),
                      encoding="unicode_escape", usecols=["ACCIDENT_NO"],
                      low_memory=False)
parent_keys = set(parents["ACCIDENT_NO"].astype(str).str.strip())

has_parent = df["ACCIDENT_NO"].astype(str).str.strip().isin(parent_keys)

orphan_df = df[~has_parent].copy()
orphan_df["REJECT_REASON"] = "Parent accident rejected during accident cleaning"
df = df[has_parent]

print(f'\nVehicle rows with no parent accident: {len(orphan_df):,}')
if len(orphan_df) > 0:
    print(orphan_df.groupby("ACCIDENT_NO").size().to_string())

# ---------------------------------------------------------
# 11. Profile - measures and codes are read differently
# ---------------------------------------------------------
print("\n" + "=" * 60)
print("DATA PROFILE SUMMARY")
print("=" * 60)
print(f'Total Records : {len(df):,}')
print(f'Total Columns : {df.shape[1]}')
print(f'Duplicate Rows: {df.duplicated().sum():,}\n')

missing_summary = pd.DataFrame({
    "Missing Count": df.isna().sum(),
    "Missing Pct (%)": (df.isna().mean() * 100).round(2),
    "Dtype": df.dtypes,
})
print(missing_summary[missing_summary["Missing Count"] > 0].sort_values("Missing Count", ascending=False))

print("\nMeasures - these are quantities, so a mean means something:")
print(df[integer_measures].astype("float64").describe().T[["count", "mean", "std", "min", "50%", "max"]])

print("\nCode columns - these are labels stored as numbers, so they are counted,")
print("not averaged. -1 marks a blank that was never recorded.")
for column in code_categories:
    print(f'\n{column}')
    print(df[column].value_counts().sort_index().to_string())

# ---------------------------------------------------------
# 12. Reconcile - nothing was deleted
# ---------------------------------------------------------
ledger = pd.DataFrame(
    [
        ("Rows", raw_row_count, len(df), len(orphan_df),
            len(df) + len(orphan_df)),
        ("Occupants", int(raw_occupants), int(df["TOTAL_NO_OCCUPANTS"].sum()),
            int(orphan_df["TOTAL_NO_OCCUPANTS"].sum()),
            int(df["TOTAL_NO_OCCUPANTS"].sum()
                + orphan_df["TOTAL_NO_OCCUPANTS"].sum())),
    ],
    columns=["Measure", "Started with", "Cleaned", "Set aside", "Total now"],
)
ledger["Adds up"] = np.where(ledger["Started with"] == ledger["Total now"], "yes", "NO")
print()
print(ledger.to_string(index=False))
print("\nNothing was deleted - every vehicle is still accounted for.")

# ---------------------------------------------------------
# 13. Export
# ---------------------------------------------------------
output_dir = Path("output/vehicle")
output_dir.mkdir(parents=True, exist_ok=True)

df.to_csv(output_dir / "vehicle_cleaned.csv", index=False)
orphan_df.to_csv(output_dir / "vehicle_rejected.csv", index=False)
dropped_df.to_csv(output_dir / "vehicle_dropped.csv", index=False)
print(f"\nFiles exported to '{output_dir}/'.")

# ---------------------------------------------------------
# 14. Verification
# ---------------------------------------------------------
print("\n--- verification ---")
print(f'Rows: {len(df):,}   (expect 364,295)')
print(f'Cols: {df.shape[1]}          (expect 41)')

category_columns = text_categories + code_categories
still_blank = df[category_columns].isnull().sum()
print("\nCategory columns still holding a blank (should be none):")
if still_blank.sum() == 0:
    print("  none")
else:
    print(still_blank[still_blank > 0].to_string())

print("\nMeasures, which are allowed to be missing:")
print(df[integer_measures].isnull().sum().to_string())
