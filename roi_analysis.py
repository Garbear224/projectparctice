import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Load the data
# Skip rows 1 and 2 (indices 1 and 2) to keep the first row (Q-codes) as header
df = pd.read_csv('Alternative CPA Pathways Survey_December 31, 2025_09.45.csv', header=0, skiprows=[1, 2])

# Clean columns
# Strip whitespace from relevant columns
if 'Q27' in df.columns:
    df['Q27'] = df['Q27'].str.strip()
if 'Q55' in df.columns:
    df['Q55'] = df['Q55'].str.strip()
if 'Q44' in df.columns:
    df['Q44'] = df['Q44'].str.strip()

# Filter dataset to include only Undergraduate and Graduate students
df_filtered = df[df['Q27'].isin(['Undergraduate', 'Graduate'])].copy()

# Convert Q55 responses to a numerical scale
mapping = {
    "Definitely yes": 5,
    "Probably yes": 4,
    "Might or might not": 3,
    "Probably not": 2,
    "Definitely not": 1
}

# Apply mapping for Q55
if 'Q55' in df_filtered.columns:
    df_filtered['Q55_numeric'] = df_filtered['Q55'].map(mapping)
else:
    df_filtered['Q55_numeric'] = np.nan

# Apply mapping for Q44 if it exists
if 'Q44' in df_filtered.columns:
    df_filtered['Q44_numeric'] = df_filtered['Q44'].map(mapping)
else:
    df_filtered['Q44_numeric'] = np.nan

# Combine Q55 (Undergraduate) and Q44 (Graduate) into a single target variable
# If Q55 is NaN, try Q44.
df_filtered['Perceived_ROI'] = df_filtered['Q55_numeric'].fillna(df_filtered['Q44_numeric'])

# Remove rows with missing values in Q27 or the combined target variable
df_clean = df_filtered.dropna(subset=['Q27', 'Perceived_ROI']).copy()

# --- STATISTICAL ANALYSIS ---

# Calculate n, mean, std for both groups
grouped = df_clean.groupby('Q27')['Perceived_ROI']
desc_stats = grouped.agg(['count', 'mean', 'std']).reset_index()
desc_stats.columns = ['Group', 'n', 'Mean', 'Standard Deviation']

# Conduct Independent Samples T-Test
undergrad_scores = df_clean[df_clean['Q27'] == 'Undergraduate']['Perceived_ROI']
grad_scores = df_clean[df_clean['Q27'] == 'Graduate']['Perceived_ROI']

# Using standard independent t-test
t_stat, p_val = stats.ttest_ind(undergrad_scores, grad_scores)

# Calculate Cohen's d
n1 = len(undergrad_scores)
n2 = len(grad_scores)
s1 = undergrad_scores.std()
s2 = grad_scores.std()
mean1 = undergrad_scores.mean()
mean2 = grad_scores.mean()

pooled_std = np.sqrt(((n1 - 1) * s1**2 + (n2 - 1) * s2**2) / (n1 + n2 - 2))
cohens_d = (mean1 - mean2) / pooled_std

# --- VISUALIZATION ---

plt.figure(figsize=(10, 6))
# Seaborn barplot with error bars (95% CI)
sns.barplot(data=df_clean, x='Q27', y='Perceived_ROI', hue='Q27', legend=False, errorbar=('ci', 95), capsize=.1, palette='muted')

plt.title("Perceived ROI of Graduate Degree: Undergrad vs Graduate Students")
plt.ylabel("Perceived Lifetime Earnings Benefit (1-5)")
plt.xlabel("Student Status")

# Save the plot
plt.savefig('roi_perception_analysis.png')
plt.close()

# --- OUTPUT SUMMARY ---

print("--- Descriptive Statistics ---")
print(desc_stats.to_string(index=False))
print("\n--- T-Test Results ---")
print(f"T-statistic: {t_stat:.4f}")
print(f"P-value: {p_val:.4f}")
print(f"Cohen's d: {cohens_d:.4f}")

print("\n--- Conclusion ---")
alpha = 0.05
if p_val < alpha:
    conclusion = (
        "There is a statistically significant difference in the perceived financial ROI of a Master's degree "
        "between Undergraduate and Graduate students (p < 0.05). "
        "A 'perception gap' exists between the two student populations."
    )
else:
    conclusion = (
        "There is no statistically significant difference in the perceived financial ROI of a Master's degree "
        "between Undergraduate and Graduate students (p >= 0.05). "
        "No significant 'perception gap' was found."
    )
print(conclusion)
