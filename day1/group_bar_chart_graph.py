from bar_chart import *

# # Create a grouped bar chart showing survival rate by BOTH class and sex
fig, ax = plt.subplots(figsize=(7, 5))

# # Using seaborn to easily handle the grouped categories (hue)
sns.barplot(
    data=df,
    x='pclass',
    y='survived',
    hue='sex',
    ax=ax,
    errorbar=None  # Removes the default error bars for a cleaner look
)

# # Customize the chart labels and appearance
ax.set_title('Survival Rate by Passenger Class and Sex', pad=15, fontsize=14)
ax.set_xlabel('Passenger Class', fontsize=12)
ax.set_ylabel('Survival Rate', fontsize=12)
ax.set_ylim(0, 1)

# # Map x-ticks if you want it to look exactly like the first plot ("1st", "2nd", "3rd")
ax.set_xticklabels(['1st', '2nd', '3rd'])

plt.tight_layout()
plt.show()