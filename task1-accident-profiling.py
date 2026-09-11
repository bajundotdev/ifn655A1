import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

FOLDER_PATH = 'data'
ACCIDENT_FILENAME = 'accident-original.csv'
OUTPUT_FOLDER = 'output/accident'

issues = []

# Create the output folder before saving charts and issue reports.
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# STEP 01: This step processes the raw accident dataset by reading the CSV into memory and checking
# the row count, column count, and overall dataset footprint before any profiling work begins.
print("\nSTEP 01 Load data source files")
print("=" * 95)
accident_df = pd.read_csv(f"{FOLDER_PATH}/{ACCIDENT_FILENAME}", encoding='unicode_escape', low_memory=False)
print(f'{ACCIDENT_FILENAME} loaded: {len(accident_df):,} rows x {accident_df.shape[1]} columns')

# STEP 02: This step processes the schema by reviewing each column's inferred pandas dtype and the
# structure of the dataset to detect coding and date/time fields that may have been read incorrectly.
print("\nSTEP 02 Structure and Data types of every column of accident.csv file")
print("=" * 95)
print(accident_df.info())

issues.append({
    'Column': 'ACCIDENT_TYPE, DCA_CODE, ROAD_GEOMETRY, LIGHT_CONDITION, POLICE_ATTEND, SEVERITY', 
    'Issue': 'Nominal/ordinal categorical codes incorrectly inferred as int64'
})
issues.append({
    'Column': 'ACCIDENT_DATE / ACCIDENT_TIME', 
    'Issue': 'Temporal data loaded as strings instead of datetime objects'
})
# ACCIDENT_DATE and ACCIDENT_TIME should be datetime, default to str
# ACCIDENT_TYPE (code 1-9)
# DAY_OF_WEEK (code 1 'Sunday' to 7 'Saturday')
# DCA_CODE (code 100-781)
# LIGHT_CONDITION (code 1-9, no 7 and 8)
# POLICE_ATTEND (code 1 yes, 2 no, 9 not known)
# ROAD_GEOMETRY (code 1-9)
# SEVERITY (code 1-4)
# SPEED_ZONE (string; padded number code 000)
print("How many rows, and how many distinct accident numbers?")
print(f'{ACCIDENT_FILENAME}: {len(accident_df):,} rows)')
print(f'distinct ACCIDENT_NO: {accident_df["ACCIDENT_NO"].nunique():,}')

# This step processes the date field by converting ACCIDENT_DATE to datetime and verifying the
# minimum and maximum dates to confirm the data's temporal coverage and any obvious date anomalies.
dates = pd.to_datetime(accident_df["ACCIDENT_DATE"], errors='coerce')
valid_dates = dates.notna()
if valid_dates.any():
    print(f"Coverage: {dates[valid_dates].min().date()} to {dates[valid_dates].max().date()}")
else:
    print("Coverage: no valid ACCIDENT_DATE values found")


# STEP 03: This step processes the original file values and the loaded values side by side to check whether
# pandas has coerced coded fields into the wrong type or whether placeholder/sentinel values are hidden.
print("\nSTEP 03 Compare how the loaded values are stored in the file")
print("=" * 95)

# Read the same column as strings to inspect the original CSV values before pandas type coercion.
text_copy = pd.read_csv(f"{FOLDER_PATH}/{ACCIDENT_FILENAME}", encoding='unicode_escape', low_memory=False, dtype=str)
print("as stored in the file :", sorted(text_copy["SPEED_ZONE"].unique().tolist()))
print("as pandas loaded it   :", sorted(accident_df["SPEED_ZONE"].unique().tolist()))

# Normalise to numeric before checking sentinel values to avoid string-vs-numeric mismatches.
accident_df["SPEED_ZONE"] = pd.to_numeric(accident_df["SPEED_ZONE"], errors='coerce')

# Show the loaded speed-zone values and summary stats to identify invalid or placeholder coding.
print(sorted(accident_df["SPEED_ZONE"].dropna().unique().tolist()))
print(accident_df["SPEED_ZONE"].describe())

# Sentinel codes (777/888/999) are often placeholders and can indicate missing or invalid speed data.
sentinels = accident_df["SPEED_ZONE"].isin([777, 888, 999]).sum()
print(f'\nSPEED_ZONE sentinel codes (777/888/999): {sentinels:,} rows')

issues.append({
    "Column": "SPEED_ZONE", 
    "Issue": f"{sentinels:,} rows use codes 777/888/999 in a numeric column"
})

# Visualise the SPEED_ZONE distribution to spot suspicious or unexpected coding patterns.
plt.figure(figsize=(9, 4.5))
sns.countplot(data=accident_df, x="SPEED_ZONE")
plt.title("Crashes by SPEED_ZONE")
plt.xlabel("SPEED_ZONE as loaded")
plt.ylabel("crashes")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_FOLDER, 'fig1_speed_zone_chart.png'), bbox_inches='tight')
plt.close()

# Check NODE_ID for impossible location identifiers, which often indicate bad or missing node references.
bad_node = int((accident_df["NODE_ID"] <= 0).sum())
if bad_node > 0:
    print(f'\nNODE_ID values that are zero or negative: {bad_node:,}')
    print(f'Distinct bad values are: {sorted(accident_df.loc[accident_df["NODE_ID"] <= 0, "NODE_ID"].unique().tolist())}')
    
    issues.append({
        "Column": "NODE_ID", 
        "Issue": "Contains zero or negative values"
    })
 
# STEP 04: This step processes the dataset for missing values by counting nulls in each column and identifying
# which fields are incomplete enough to require remediation before model development or aggregation.
print("\nSTEP 04 Count missing values in every column")
print("=" * 95)

nulls = accident_df.isna().sum()  # Counts NaN, None, and other pandas missing values
print(nulls)

print("\nPercent of rows that have any missing values:")
has_nulls = nulls[nulls > 0].sort_values(ascending=False)
for column in has_nulls.index:
    count = has_nulls[column]
    print(f"{column:<24} {format(count, ','):>10} {count / len(accident_df) * 100:6.2f}%")

    issues.append({
        "Column": column, 
        "Issue": f"{count:,} missing values"
    })

# Identify columns containing the sentinel value 9 or the category "unknown".
print("\nColumns containing values coded as 9 or 'unknown':")
for column in accident_df.columns:
    values = accident_df[column].astype("string").str.strip().str.lower()
    sentinel_count = int(values.isin(["9", "unknown"]).sum())

    if sentinel_count > 0:
        codes_found = values[values.isin(["9", "unknown"])].value_counts().to_dict()
        print(
            f"{column:<24} {format(sentinel_count, ','):>10} "
            f"{sentinel_count / len(accident_df) * 100:6.2f}% "
            f"{list(codes_found)[0]:>8}"
        )
        issues.append({
            "Column": column,
            "Issue": f"{sentinel_count:,} values coded as 9 or unknown"
        })

# The RMA column has a high number of missing records, 7673, so we keep this visible
# in the audit output for later review in the data-cleaning stage.
if len(has_nulls) == 0:
    print("none")
 
# STEP 05: This step processes categorical code columns by listing each distinct value and its frequency,
# helping reveal invalid codes, unexpected categories, and domain mismatches before cleaning.
print("\nSTEP 05 Unique values and how often each one occurs")
print("=" * 95)

columns = ['ACCIDENT_TYPE', 'DAY_OF_WEEK', 'DCA_CODE', 'LIGHT_CONDITION', 'POLICE_ATTEND', 'ROAD_GEOMETRY', 'SEVERITY', 'SPEED_ZONE', 'RMA']

for col in columns:
    print(f'\n-----{col}-----')
    print(f'distinct values: {accident_df[col].nunique()}')
    print(f'{accident_df[col].value_counts().sort_index()}')

# DAY_OF_WEEK has values that does not represent the assigned DAY_WEEK_DESC
# SPEED_ZONE has (categorical) values that does not defined in the dictionary


# ==============================
# STEP 06: This step processes numeric variables by calculating descriptive measures such as min, max,
# median, quartiles, and distributions to reveal skewness, outliers, and suspicious ranges.
print("\nSTEP 06 Summary statistics for the numeric columns")
print("=" * 95)

print(accident_df.describe(include='all').T)

print("\nMedian of every numeric column in accident.csv:")
numeric_df = accident_df.select_dtypes(include=["number"])
print(numeric_df.median())
 
# STEP 07: This step processes the dataset for duplication by checking whether rows are repeated exactly
# or whether accident identifiers are reused, which can distort totals and create data integrity issues.
print("\nSTEP 07 Check for duplicate records")
print("=" * 95)

print(f"Duplicate rows: {accident_df.duplicated().sum()}")
print(f"Repeated ACCIDENT_NO: {accident_df.duplicated(subset=['ACCIDENT_NO']).sum()}")
 
# STEP 08: This step processes numeric and severity fields by evaluating skewness, z-score outliers, and
# class distribution to find extreme crash records and anomalous severity patterns.
print("\nSTEP 08 Outlier checks")
print("=" * 95)

# Outliers are checked to identify unusually extreme crash counts or casualty totals that may indicate
# data-entry errors, coding problems, or genuinely rare but valid events that need review.
print("Z-score method, threshold |z| > 3\n")
for name, df, column in [("accident.csv", accident_df, "NO_PERSONS"),
                         ("accident.csv", accident_df, "NO_PERSONS_KILLED")]:
    values = df[column].dropna()
    zscore = np.abs(stats.zscore(values))
    flagged = (zscore > 3).sum()
    print(f"  {name:<13} {column:<20} flags {flagged:>8,} rows ({(flagged / len(values) * 100):5.2f}%)")

print("\nQuartiles of the casualty columns:\n")
for column in ["NO_PERSONS", "NO_PERSONS_KILLED", "NO_PERSONS_INJ_2"]:
    values = accident_df[column]
    q1 = values.quantile(0.25)
    q3 = values.quantile(0.75)
    print(f"  {column:<20} Q1 = {q1:5.1f}   median = {values.median():5.1f}   Q3 = {q3:5.1f}")

fatal = (accident_df["SEVERITY"] == 1).sum()
print(f"\nTotal number of fatal crashes: {fatal:,}")

issues.append({
    "Column": "SEVERITY", 
    "Issue": f"Severe target class imbalance: Fatal crashes (1) represent only {(fatal/len(accident_df)*100):.2f}% of data, class 4 (none) is entirely missing"
})
issues.append({
    "Column": "NO_PERSONS", 
    "Issue": "Extreme right-skew distribution containing severe mass-casualty outliers (up to ~100 persons) that will warp distance-based algorithms"
})

# severity chart
plt.figure(figsize=(7, 4.5))
sns.countplot(data=accident_df, x="SEVERITY")
plt.title("Number of crashes by severity code")
plt.xlabel("SEVERITY (1 = fatal, 2 = serious injury, 3 = other injury, 4 = none)")
plt.ylabel("crashes")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_FOLDER, 'fig2_severity_chart.png'), bbox_inches='tight')
plt.close()
 
# The boxplot helps visualise the spread and central tendency of persons involved per crash,
# and highlights whether a few crashes have unusually large counts.
plt.figure(figsize=(8, 3.2))
sns.boxplot(x=accident_df["NO_PERSONS"])
plt.title("Persons involved per crash - box plot")
plt.xlabel("NO_PERSONS")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_FOLDER, 'fig3_persons_boxplot.png'), bbox_inches='tight')
plt.close()
 
# STEP 09: This step processes relationships between fields by checking whether codes and descriptions match,
# whether date-based enumerations align with the day-of-week field, and whether counts across related
# columns are logically consistent.
print("\nSTEP 09 Consistency checks")
print("=" * 95)

pairs = [("ACCIDENT_TYPE", "ACCIDENT_TYPE_DESC"), ("DAY_OF_WEEK", "DAY_WEEK_DESC"), ("DCA_CODE", "DCA_DESC"), ("ROAD_GEOMETRY", "ROAD_GEOMETRY_DESC")]

for code, desc in pairs:
    most = accident_df.groupby(code)[desc].nunique().max()
    print(f"  {code:<16} at most {most} description(s) per code   {'OK' if most == 1 else 'INCONSISTENT'}")

# convert dates to temporary series for profiling without modifying the mainframe
accident_dates = pd.to_datetime(accident_df["ACCIDENT_DATE"])
iso_number = accident_dates.dt.dayofweek + 1
sunday_first = (accident_dates.dt.dayofweek + 1) % 7 + 1

agrees_iso = (accident_df["DAY_OF_WEEK"] == iso_number).sum()
agrees_sunday = (accident_df["DAY_OF_WEEK"] == sunday_first).sum()
agrees_neither = len(accident_df) - agrees_iso - agrees_sunday

print("\nWhich numbering system does DAY_OF_WEEK follow?")
print(f"  Monday = 1  : {agrees_iso:9,}  ({(agrees_iso / len(accident_df) * 100):5.2f}%)")
print(f"  Sunday = 1  : {agrees_sunday:9,}  ({(agrees_sunday / len(accident_df) * 100):5.2f}%)")
print(f"  neither     : {agrees_neither:9,}  ({(agrees_neither / len(accident_df) * 100):5.2f}%)")

print("\nWhat day are the 'neither' rows actually on?")
print(accident_dates[(accident_df["DAY_OF_WEEK"] != iso_number) & (accident_df["DAY_OF_WEEK"] != sunday_first)].dt.day_name().value_counts())
# There 2 different numbering system
# It should be Sunday == 1 per the data dictionary and
# Monday == 1 (non-compliant)

parts = (accident_df["NO_PERSONS_KILLED"] + accident_df["NO_PERSONS_INJ_2"]
         + accident_df["NO_PERSONS_INJ_3"] + accident_df["NO_PERSONS_NOT_INJ"])
mismatch = (parts != accident_df["NO_PERSONS"]).sum()

print("\nTotal number of casualty columns is equal to NO_PERSONS?")
print(f"Rows where they do not: {mismatch:,}")

no_vehicles = (accident_df["NO_OF_VEHICLES"] == 0).sum()
print(f"\nCrashes recorded with NO_OF_VEHICLES equals to 0 : {no_vehicles:,}")

contradiction = ((accident_df["SEVERITY"] == 2) & (accident_df["NO_PERSONS_INJ_2"] == 0)).sum()
print(f'Crashes recorded as serious injury but with no serious injuries: {contradiction:,}')

issues.append({
    "Column": "DAY_OF_WEEK", 
    "Issue": f"{agrees_neither:,} rows fail to match ISO numbering."
})
issues.append({
    "Column": "NO_OF_VEHICLES", 
    "Issue": f"{no_vehicles:,} crashes recorded with zero vehicles."
})
issues.append({
    "Column": "SEVERITY / NO_PERSONS_INJ_2", 
    "Issue": f"Severity scale coding contradicts actual injury counts in {contradiction:,} records"
})
# Issue "DAY_OF_WEEK", "NO_OF_VEHICLES", "SEVERITY / NO_PERSONS_INJ_2"
 
# STEP 10: This step processes the relationship between the accident and vehicle files by checking whether
# each accident has a corresponding vehicle record and whether any vehicle records are orphaned.
print("\nSTEP 10 The relationship between accident.csv and vehicle.csv")
print("=" * 95)

vehicle_filename = 'vehicle-original.csv'
vehicle_df = pd.read_csv(f"{FOLDER_PATH}/{vehicle_filename}", encoding='unicode_escape', low_memory=False)

accident_ids = set(accident_df["ACCIDENT_NO"])
vehicle_ids = set(vehicle_df["ACCIDENT_NO"])

print(f"distinct ACCIDENT_NO in accident.csv : {len(accident_ids):,}")
print(f"distinct ACCIDENT_NO in vehicle.csv  : {len(vehicle_ids):,}")
print(f"vehicle records with no matching accident : {len(vehicle_ids - accident_ids):,}")
print(f"accidents with no matching vehicle record : {len(accident_ids - vehicle_ids):,}")

issues.append({
    "Column": "ACCIDENT_NO", 
    "Issue": f"{len(accident_ids - vehicle_ids):,} accidents have no matching vehicle record"})
# Issue "ACCIDENT_NO"

 
# STEP 11: This step processes the numeric feature space by calculating a correlation matrix to identify
# relationships, redundancies, and potentially problematic feature interactions.
print("\nSTEP 11 Correlation matrix")
print("=" * 95)

corr_matrix = numeric_df.corr()

issues.append({
    'Column': 'ACCIDENT_TYPE / DCA_CODE',
    'Issue': 'Strong positive multicollinearity indicating redundant feature information'
})

issues.append({
    'Column': 'SEVERITY / NO_PERSONS_INJ_2',
    'Issue': 'Strong negative correlation due to inverted severity scale coding'
})

plt.figure(figsize=(11, 9))
sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f", annot_kws={"size": 7})
plt.title("Correlation between true continuous columns - accident.csv")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_FOLDER, 'fig4_correlation_heatmap.png'), bbox_inches='tight')
# plt.show()

# Save to OUTPUT Folder
# Export the issues log so the data-quality findings are captured in a structured output file.
issues_df = pd.DataFrame(issues)
issues_df.to_csv(f'{OUTPUT_FOLDER}/accident_issues.csv', index=False)