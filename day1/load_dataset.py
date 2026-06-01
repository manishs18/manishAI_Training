import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Notebook display settings
# %matplotlib inline
plt.rcParams['figure.dpi'] = 120
sns.set_theme(style='whitegrid', palette='muted')

# Load dataset
df = sns.load_dataset('titanic').dropna(subset=['age', 'embarked'])
df['age_group'] = pd.cut(
    df['age'], 
    bins=[0, 12, 18, 35, 60, 120], 
    labels=['Child', 'Teen', 'Young Adult', 'Adult', 'Senior']
)
print('Dataset shape:', df.shape)