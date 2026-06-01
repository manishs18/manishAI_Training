import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# # Anatomy demo – labelled blank figure
fig, ax = plt.subplots(figsize=(7, 4))

ax.set_title('Figure title (ax.set_title)', fontsize=14)
ax.set_xlabel('X-axis label (ax.set_xlabel)')
ax.set_ylabel('Y-axis label (ax.set_ylabel)')

ax.text(
    0.5, 0.5, 'Axes area', 
    transform=ax.transAxes,
    ha='center', va='center', 
    fontsize=16, color='lightgray'
)

# # Annotation arrow
ax.annotate(
    'Annotation (ax.annotate)',
    xy=(0.2, 0.3), xycoords='axes fraction',
    xytext=(0.5, 0.2), textcoords='axes fraction',
    arrowprops=dict(arrowstyle='->', color='steelblue'),
    color='steelblue', fontsize=10
)

plt.tight_layout()
plt.show()