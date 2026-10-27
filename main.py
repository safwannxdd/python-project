import pandas as pd
import numpy as np
import statsmodels.api as sm

# ---- 1) Load data ----
path = "/Users/muhammadsafwan/Downloads/Applied Empirical Project(Sheet1) (1).csv"
df = pd.read_csv(path)

# ---- 2) Clean / handle missing ----
# Convert common missing markers to NaN
df = df.replace(["N/A", "NA", "na", "-", "", " ", "n/a"], np.nan)

# Set variable names (edit these if your column names differ)
y_col = "Gini"
x_main = "EduSpend"
controls = ["LnGDPPC", "GOVSpend", "Unemployment rate", "Urban"]  # add/remove controls here

# Force numeric (turns non-numeric strings into NaN)
for c in [y_col, x_main] + controls:
    df[c] = pd.to_numeric(df[c], errors="coerce")

# Keep only rows with all needed variables present
reg_df = df[[y_col, x_main] + controls].dropna()

print("Rows before:", len(df), "Rows used in regression:", len(reg_df))

# ---- 3) Run OLS regression ----
Y = reg_df[y_col]
X = reg_df[[x_main] + controls]
X = sm.add_constant(X)

model = sm.OLS(Y, X).fit(cov_type="HC3")  # robust SE (HC3). You can use "HC1" too.
print(model.summary())
