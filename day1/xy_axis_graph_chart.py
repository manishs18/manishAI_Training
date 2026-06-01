from load_dataset import *

# ==========================================
# Part 4: Distribution plot (Histogram + KDE)
# ==========================================

fig, ax = plt.subplots(figsize=(7, 4))

# Plot histogram for age distribution
sns.histplot(
    data=df,
    x='age',
    kde=True,
    bins=25,
    color='purple',
    ax=ax
)

ax.set_title('Passenger Age Distribution')
ax.set_xlabel('Age')
ax.set_ylabel('Count')

plt.tight_layout()
plt.show()

# ==========================================
# Extension Task 3:
# #1. Plot a distribution of 'fare' using sns.histplot.
# #2. Use a different color and add a KDE line.
# ==========================================

fig, ax = plt.subplots(figsize=(7, 4))

# Plot histogram for fare distribution
sns.histplot(
    data=df,
    x='fare',
    kde=True,
    bins=30,
    color='teal',
    ax=ax
)

ax.set_title('Passenger Fare Distribution')
ax.set_xlabel('Fare')
ax.set_ylabel('Count')

plt.tight_layout()
plt.show()