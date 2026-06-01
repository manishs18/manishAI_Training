import pandas as pd
import numpy as np
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