import pandas as pd
import numpy as np
import seaborn as sns

print("Pandas version:", pd.__version__)

import seaborn as sns
df = sns.load_dataset('titanic')
print('Shape', df.shape)
df.head()

df.info()
df.describe().round(2)

# Missing values at a glance
missing = df.isnull().sum()
missing_percent = 100 * df.isnull().sum() / len(df)
pd.DataFrame({'missing': missing, 'missing_percent': missing_percent.round(2)}).sort_values(by='missing', ascending=False)

# ==========================================
# Part 2 - Handling missing values
# ==========================================

# Strategy 1: Dropping rows where any column is null
print('Before dropna():', df.shape)
df_dropped = df.dropna()
print('After dropna():', df_dropped.shape)

# Strategy 2: Drop only if specific column is null
df_dropped2 = df.dropna(subset=['age', 'embarked'])
print('After dropna(subset=[age, embarked]):', df_dropped2.shape)

# Strategy 3: Fill missing values
df2 = df.copy() # never mutate the original while exploring

df2['age'] = df2['age'].fillna(df2['age'].mean())
df2['embarked'] = df2['embarked'].fillna(df2['embarked'].mode()[0])

# Drop 'deck' - too many missing values to be useful
df2 = df2.drop(columns=['deck'])

print('Missing after cleaning')
print(df2.isnull().sum())

# ==========================================
# Extension Task 2:
# #1. Find all male passengers who paid a fare above 200 and survived.
# #2. Using .iloc, extract rows 100 to 109 and the last 3 columns.
# ==========================================

wealthy_male_survivors = df2[(df2['sex'] == 'male') & (df2['fare'] > 200) & (df2['survived'] == 1)]
print("wealthy_male_survivors: ", wealthy_male_survivors)

# Extension Task 1:
# #1. How many unique values does the 'embarked' column has?
# #2. What is the most common passenger class (pclass)?

subset_iloc = df2.iloc[100:110, -3:]
print("\n Rows 100-109 and the last 3 columns: ", subset_iloc)