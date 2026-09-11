import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

FOLDER_PATH = 'data'
ACCIDENT_FILENAME = 'accident-original.csv'
OUTPUT_FOLDER = 'output/accident'

os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# STEP 01: This step loads the raw accident data into memory, preserves an untouched copy for later reconciliation,
# and prints a basic profile of the dataset so we can confirm that the file has been read correctly before cleaning.
print("\nSTEP 01 Load data source files")
print("=" * 95)

raw = pd.read_csv(f"{FOLDER_PATH}/{ACCIDENT_FILENAME}", encoding='unicode_escape', low_memory=False)
accident_df = raw.copy()   # raw is kept untouched so the totals can be reconciled at the end
print(accident_df.info())

print(f'{ACCIDENT_FILENAME} loaded: {len(accident_df):,} rows, {accident_df.shape[1]} columns')

# STEP 02: This step checks for hidden sentinel codes (such as 777, 888, 999), investigates where missing values 
# are concentrated,creates more readable speed-zone groupings, and fills missing road-management-area values 
# so later analysis is consistent.
print("\nSTEP 02 Handling Missing Data")
print("=" * 95)

# Unknown codes hidden inside categorical columns
print("Before conversion")
print(accident_df[["SPEED_ZONE", "LIGHT_CONDITION", "POLICE_ATTEND", "ROAD_GEOMETRY", "RMA"]].isnull().sum())

speed_zone_codes = {777: "Other speed limit", 888: "Camping grounds/ off road", 999: "Not known"}

coded = accident_df["SPEED_ZONE"].isin(speed_zone_codes.keys())
mean_before = accident_df["SPEED_ZONE"].mean()
mean_after = accident_df.loc[~coded, "SPEED_ZONE"].mean()

print(f'\nSPEED_ZONE codes found: {coded.sum():,} rows ({coded.mean() * 100:.2f}%)')
print(f'\nMean SPEED_ZONE before codes: {mean_before:.1f}')
print(f'Mean SPEED_ZONE excluding codes: {mean_after:.1f}')
# The first figure is not a speed limit - it is what happens when 999 is averaged in

# Apply binning into a new SPEED_ZONE_DESC column
bins = [0, 40, 60, 80, 100, 150]
labels = ['0-40 km/h', '41-60 km/h', '61-80 km/h', '81-100 km/h', '100+ km/h']

accident_df["SPEED_ZONE_DESC"] = pd.cut(
    pd.to_numeric(accident_df["SPEED_ZONE"]), 
    bins=bins, 
    labels=labels
).astype(str)

# Map the known sentinel text back into the description column
accident_df.loc[coded, "SPEED_ZONE_DESC"] = accident_df.loc[coded, "SPEED_ZONE"].map(speed_zone_codes)
accident_df["SPEED_ZONE_DESC"] = accident_df["SPEED_ZONE_DESC"].replace('nan', 'Unknown')

print("\nBinned SPEED_ZONE_DESC distribution:")
print(accident_df["SPEED_ZONE_DESC"].value_counts().to_string())

# Report the single-digit 9 codes but leave them intact for the mapping phase
for column in ["LIGHT_CONDITION", "POLICE_ATTEND", "ROAD_GEOMETRY"]:
    count = (accident_df[column] == 9).sum()
    print(f'{column:<18} code 9 (Unknown) found in {count:,} rows. Retaining as category 9.')

print("\nAfter conversion, isnull().sum() now reports:")
print(accident_df[["SPEED_ZONE", "LIGHT_CONDITION", "POLICE_ATTEND", "ROAD_GEOMETRY", "RMA"]].isnull().sum())

# RMA
missing_before = accident_df['RMA'].isnull().sum()
print(f'\nRMA missing values before: {missing_before:,} ({missing_before / len(accident_df) * 100:.2f}%)')

accident_year = pd.to_datetime(accident_df['ACCIDENT_DATE']).dt.year

# Is the missingness spread evenly, or concentrated in particular years?
by_year = accident_df.groupby(accident_year)['RMA'].apply(lambda s: s.isnull().mean() * 100)
print('\nPercent of RMA missing by accident year:')
print(by_year.round(2).to_string())

accident_df["RMA"] = accident_df["RMA"].fillna("Unknown")
print(f'\nRMA missing values after: {accident_df["RMA"].isnull().sum()}')

# The dictionary defines NODE_ID as a location identifier that increments with new locations. 
# We retain the zero/negative codes rather than forcing NaN.
bad_node = (accident_df["NODE_ID"] <= 0)
print(f'\nNODE_ID values that are zero or negative (retained as codes): {bad_node.sum():,}')


# STEP 03: This step looks for exact duplicate rows and duplicate accident numbers so we can detect data quality issues
# before any rows are removed or transformed, preserving a clear record of the source-quality checks.
print("\nSTEP 03 Removing Duplicates")
print("=" * 95)

identical = accident_df.duplicated().sum()
repeated_key = accident_df.duplicated(subset=['ACCIDENT_NO']).sum()
print(f'Identical rows: {identical}')
print(f'Repeated ACCIDENT_NO values: {repeated_key}')

# STEP 04: This step reviews each column for low-information content, such as single-value fields, highly empty fields,
# and mismatched code/description pairings, so redundant columns can be removed without losing essential data.
print("\nSTEP 04 Dropping Unnecessary Columns")
print("=" * 95)

print("Columns with only one distinct value:")
unary = [c for c in accident_df.columns if accident_df[c].nunique(dropna=True) <= 1]
print(f"  {unary if unary else 'none'}")

print("\nColumns more than 90 percent empty:")
mostly_empty = [c for c in accident_df.columns if accident_df[c].isnull().mean() > 0.9]
print(f"  {mostly_empty if mostly_empty else 'none'}")

# Code column and Code description matching 
print("\nDoes each code map to exactly one description?")
pairs = [("ACCIDENT_TYPE", "ACCIDENT_TYPE_DESC"),
         ("DCA_CODE", "DCA_DESC"),
         ("ROAD_GEOMETRY", "ROAD_GEOMETRY_DESC")]

for code_column, desc_column in pairs:
    most = accident_df.groupby(code_column)[desc_column].nunique().max()
    verdict = "consistent" if most == 1 else "INCONSISTENT"
    print(f"{code_column:<16} at most {most} description(s) per code {verdict}")

# STEP 05: This step converts date and time fields into pandas-friendly formats and casts code columns to strings
# to prevent accidental numeric operations on categorical identifiers, while preserving the actual code values 
# for mapping later.
print("\nSTEP 05 Data Type Conversion")
print("=" * 95)

print("Data types before conversion:")
print(accident_df[["ACCIDENT_DATE", "ACCIDENT_TIME"]].dtypes)

# Date conversion is kept generic as requested
accident_df['ACCIDENT_DATE'] = pd.to_datetime(accident_df['ACCIDENT_DATE'])
failed_dates = accident_df['ACCIDENT_DATE'].isnull().sum()

time_as_datetime = pd.to_datetime(accident_df['ACCIDENT_TIME'], format='%H:%M:%S')
failed_times = time_as_datetime.isnull().sum()
accident_df['ACCIDENT_TIME'] = time_as_datetime.dt.time

print("\nAfter conversion:")
print(accident_df[["ACCIDENT_DATE", "ACCIDENT_TIME"]].dtypes)
print(f'\nDates that failed to convert: {failed_dates}')
print(f'Times that failed to convert: {failed_times}')

# Convert nominal and ordinal codes to strings to prevent invalid mathematical scaling
# We strip the ".0" suffix to prevent the float-to-string conversion bug caused by NaNs
categorical_cols = [
    'ACCIDENT_TYPE', 'DCA_CODE', 'LIGHT_CONDITION', 
    'NODE_ID', 'POLICE_ATTEND', 'ROAD_GEOMETRY', 'SEVERITY'
]

for col in categorical_cols:
    accident_df[col] = accident_df[col].astype(str).str.replace(r'\.0$', '', regex=True).replace('nan', np.nan)
    
print(f"\nConverted {len(categorical_cols)} nominal/ordinal columns from numeric to string types safely.")
print(accident_df.info())

# STEP 06: This step clarifies the injury count columns and adds human-readable labels for coded categories 
# such as severity, police attendance, lighting, and road geometry so downstream analysis and reports 
# are easier to interpret.
print("\nSTEP 06 Renaming Columns & Mapping Descriptions")
print("=" * 95)

renames = {
    'NO_PERSONS_INJ_2': 'NO_PERSONS_SERIOUS_INJ',
    'NO_PERSONS_INJ_3': 'NO_PERSONS_OTHER_INJ',
}

accident_df.rename(columns=renames, inplace=True)

# Mapping missing descriptions from the Data Dictionary
severity_map = {
    '1': 'Fatal accident', 
    '2': 'Serious injury accident', 
    '3': 'Other injury accident', 
    '4': 'Non injury accident'
}

police_attend_map = {
    '1': 'Yes', 
    '2': 'No',
    '9': 'Unknown'
}
light_condition_map = {
    '1': 'Day', 
    '2': 'Dusk/Dawn', 
    '3': 'Dark Street lights on', 
    '4': 'Dark Street lights off', 
    '5': 'Dark No street lights', 
    '6': 'Dark Street lights unknown',
    '9': 'Unknown'
}
road_geometry_map = {
    '1': 'Cross intersection',
    '2': 'T intersection',
    '3': 'Y intersection',
    '4': 'Multiple intersection',
    '5': 'Not at intersection',
    '6': 'Dead end',
    '7': 'Road closure',
    '8': 'Private property',
    '9': 'Unknown'
}

accident_df['SEVERITY_DESC'] = accident_df['SEVERITY'].map(severity_map)
accident_df['POLICE_ATTEND_DESC'] = accident_df['POLICE_ATTEND'].map(police_attend_map)
accident_df['LIGHT_CONDITION_DESC'] = accident_df['LIGHT_CONDITION'].map(light_condition_map)
accident_df['ROAD_GEOMETRY_DESC'] = accident_df['ROAD_GEOMETRY'].map(road_geometry_map)

print("\nAdded description columns for SEVERITY, POLICE_ATTEND, LIGHT_CONDITION, and ROAD_GEOMETRY.")


# STEP 07: This step applies data-quality validation rules to identify records that are internally inconsistent 
# or unlinked to vehicle data, moves those invalid records into a rejected dataset, and retains a clear reason 
# for each rejection.
print("\nSTEP 07 Filtering and Selecting Data")
print("=" * 95)

# Records that contradict their own definition and are moved
# to a rejected file with the reason. 
# Note: SEVERITY is now a string type due to the Step 05 conversion.
no_vehicles = accident_df['NO_OF_VEHICLES'] == 0
contradiction = (accident_df['SEVERITY'] == '2') & (accident_df['NO_PERSONS_SERIOUS_INJ'] == 0)

# Cross-reference with vehicle.csv to find orphan records
vehicle_filename = 'vehicle-original.csv'
vehicle_df = pd.read_csv(f"{FOLDER_PATH}/{vehicle_filename}", encoding='unicode_escape', low_memory=False)
orphan_crash = ~accident_df['ACCIDENT_NO'].isin(vehicle_df['ACCIDENT_NO'])

print(f'Crashes reporting zero vehicles: {no_vehicles.sum()}')
print(f'Serious injury coded with no such injury: {contradiction.sum()}')
print(f'Orphan crashes (no matching vehicle record): {orphan_crash.sum()}')
print(f'Rows meeting any rejection condition: {(no_vehicles | contradiction | orphan_crash).sum()}')

reject_df = accident_df[no_vehicles | contradiction | orphan_crash].copy()

# Assign specific reject reasons hierarchically using np.select
# Evaluate the conditions specifically on reject_df so the dimensions match
reject_df["REJECT_REASON"] = np.select(
    [
        reject_df['NO_OF_VEHICLES'] == 0, 
        (reject_df['SEVERITY'] == '2') & (reject_df['NO_PERSONS_SERIOUS_INJ'] == 0), 
        ~reject_df['ACCIDENT_NO'].isin(vehicle_df['ACCIDENT_NO'])
    ],
    [
        "Crash record states zero vehicles were involved",
        "Severity says serious injury but none is recorded",
        "Orphan record: No matching vehicle data found"
    ],
    default="Unknown"
)

clean_df = accident_df[~(no_vehicles | contradiction | orphan_crash)].copy()

print(f"\nrows before filtering: {len(accident_df):,}")
print(f"rows set aside: {len(reject_df):,}")
print(f"rows kept: {len(clean_df):,}")
print("\nReasons recorded on the rejected rows:")
print(reject_df["REJECT_REASON"].value_counts().to_string())

# STEP 08: This step identifies extreme outliers in the person and vehicle count fields using a z-score threshold,
# flags them for review, and plots the impact so we can assess whether they represent legitimate events or 
# data anomalies.
print("\nSTEP 08 Handling Outliers")
print("=" * 95)
# Detect unusually large crashes with z-scores and flag them without deleting them.

print("Z-score method, threshold |z| > 3\n")
for column in ["NO_PERSONS", "NO_PERSONS_KILLED", "NO_OF_VEHICLES"]:
    values = clean_df[column].dropna()
    zscore = np.abs(stats.zscore(values))
    print(f"  {column:<20} flags {(zscore > 3).sum():>8,} rows ({(zscore > 3).mean() * 100:5.2f}%)")

persons = clean_df["NO_PERSONS"].dropna()
persons_zscore = np.abs(stats.zscore(persons))

clean_df["NO_PERSONS_ZSCORE"] = np.nan
clean_df.loc[persons.index, "NO_PERSONS_ZSCORE"] = persons_zscore
clean_df["IS_LARGE_CRASH"] = (clean_df["NO_PERSONS_ZSCORE"] > 3).astype(int)

print(f'\nRows flagged as unusually large: {clean_df["IS_LARGE_CRASH"].sum():,}')
print("\nThe largest crashes the z-score flags in NO_PERSONS:")
print(clean_df[clean_df["IS_LARGE_CRASH"] == 1].nlargest(6, "NO_PERSONS")[
    ["ACCIDENT_NO", "ACCIDENT_DATE", "SEVERITY", "NO_OF_VEHICLES",
     "NO_PERSONS", "NO_PERSONS_KILLED"]].to_string(index=False))
# These rows are flagged, not removed so removing them would delete the events we wanted to see

fig, axes = plt.subplots(1, 2, figsize=(12, 3.6))
sns.boxplot(x=clean_df["NO_PERSONS"], ax=axes[0])
axes[0].set_title("NO_PERSONS - all crashes")
sns.boxplot(x=clean_df[clean_df["IS_LARGE_CRASH"] == 0]["NO_PERSONS"], ax=axes[1])
axes[1].set_title("NO_PERSONS - excluding the flagged crashes")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_FOLDER, 'fig5_outlier_boxplots.png'), bbox_inches='tight')

# STEP 09: This step checks text-based fields for accidental leading/trailing spaces and inconsistent letter case,
# so values can be compared and grouped reliably before reporting or downstream modelling.
print("\nSTEP 09 Standardising text columns")
print("=" * 95)
# Check text fields for inconsistent whitespace and casing.

text_columns = ["ACCIDENT_TYPE_DESC", "ROAD_GEOMETRY_DESC", "RMA"]
needs_work = 0
for column in text_columns:
    values = clean_df[column].astype(str)
    distinct_now = values.nunique()
    distinct_after = values.str.strip().str.upper().nunique()
    padded = (values != values.str.strip()).sum()
    verdict = "clean" if (distinct_now == distinct_after and padded == 0) else "NEEDS WORK"
    if verdict == "NEEDS WORK":
        needs_work += 1
    print(f"{column:<28} distinct {distinct_now:>3} after strip+upper {distinct_after:>3} "
          f"padded {padded} {verdict}")

# STEP 10: This step recalculates the weekday using the actual date field, checks whether the numbering matches 
# the published calendar mapping, and verifies that the rebuilt values are consistent with the observed weekday names.
print("\nSTEP 10 Rebuild DAY_OF_WEEK")
print("=" * 95)
# Recalculate the weekday from the accident date and verify its numbering.

# The published dictionary states the numbering is 1 = Sunday ... 7 = Saturday.
old_day_of_week = clean_df["DAY_OF_WEEK"].copy()
clean_df["DAY_OF_WEEK"] = (clean_df["ACCIDENT_DATE"].dt.dayofweek + 1) % 7 + 1   # Sunday = 1

changed = int((old_day_of_week != clean_df["DAY_OF_WEEK"]).sum())
print(f'Values that changed: {changed:,} ({changed / len(clean_df) * 100:.1f}%)')

# Rebuilding DAY_OF_WEEK 
day_name = clean_df["ACCIDENT_DATE"].dt.day_name()
print("\nChecking the rebuilt column against the calendar day name:")
check_values = {}
names_per_number_values = {}
for day_number, rows in clean_df.groupby("DAY_OF_WEEK"):
    names = day_name.loc[rows.index]
    check_values[day_number] = names.value_counts().index[0]
    names_per_number_values[day_number] = names.nunique()
check = pd.Series(check_values)
print(check.to_string())

names_per_number = max(names_per_number_values.values(), default=0)
severe = clean_df["SEVERITY"].isin(['1', '2']) # Now comparing as strings
print(f'\nSevere crashes (fatal or serious): {severe.sum():,} ({severe.mean() * 100:.1f}%)')

# The rejected rows must carry the same columns as the cleaned rows
for column in clean_df.columns:
    if column not in reject_df.columns:
        reject_df[column] = np.nan
reject_df = reject_df[list(clean_df.columns) + ["REJECT_REASON"]]

# STEP 11: This step compares the original totals against the cleaned and rejected datasets so we can verify 
# that the split is complete, and that every row and important casualty measure is still accounted for after filtering.
print("\nSTEP 11 Reconcile the row and casualty totals")
print("=" * 95)
# Confirm that cleaned and rejected records preserve the original totals.

ledger = pd.DataFrame(
    [
        ("Rows", len(raw), len(clean_df), len(reject_df), len(clean_df) + len(reject_df)),
        ("Persons killed", int(raw["NO_PERSONS_KILLED"].sum()),
            int(clean_df["NO_PERSONS_KILLED"].sum()),
            int(reject_df["NO_PERSONS_KILLED"].sum()),
            int(clean_df["NO_PERSONS_KILLED"].sum() + reject_df["NO_PERSONS_KILLED"].sum())),
        ("Persons involved", int(raw["NO_PERSONS"].sum()),
            int(clean_df["NO_PERSONS"].sum()),
            int(reject_df["NO_PERSONS"].sum()),
            int(clean_df["NO_PERSONS"].sum() + reject_df["NO_PERSONS"].sum())),
    ],
    columns=["Measure", "Started with", "Cleaned", "Rejected", "Total now"],
)

ledger["Adds up"] = np.where(ledger["Started with"] == ledger["Total now"], "yes", "NO")
print(ledger.to_string(index=False))
# Nothing was deleted and every data are acccounted

# STEP 12: This final step removes redundant or derived fields that are not needed for the final dataset 
# and writes the cleaned and rejected files to disk so they are ready for downstream analysis or reporting.
print("\nSTEP 12 Drop derivable columns and save")
print("=" * 95)
# Remove redundant derived fields and write the cleaned and rejected datasets.

# Drop derivable columns AND redundant columns identified in Task 1 and FK NODE_ID
derivable = ['DAY_WEEK_DESC', 'DAY_OF_WEEK', 'NO_PERSONS', 'NO_PERSONS_ZSCORE', 'ACCIDENT_TYPE', 'NODE_ID']
clean_df = clean_df.drop(columns=derivable)
reject_df = reject_df.drop(columns=derivable)

print(f'\ncleaned columns : {clean_df.shape[1]}')
print(f'rejected columns: {reject_df.shape[1]}')

clean_df.to_csv(os.path.join(OUTPUT_FOLDER, 'accident_cleaned.csv'), index=False)
reject_df.to_csv(os.path.join(OUTPUT_FOLDER, 'accident_rejected.csv'), index=False)