from xy_axis_graph_chart import *

# Part 2 - Chart 1: Bar chart

# Best for comparing a numeric value across discrete categories

# Group by 'pclass' and calculate the mean of 'survived'
survival_by_class = df.groupby('pclass')['survived'].mean().reset_index()

# Rename columns properly to 'Class' and 'Survival Rate'
survival_by_class = survival_by_class.rename(columns={'pclass': 'Class', 'survived': 'Survival Rate'})

# Map numeric classes to string labels
survival_by_class['Class'] = survival_by_class['Class'].map({1: '1st', 2: '2nd', 3: '3rd'})

# Initialize the plot
fig, ax = plt.subplots(figsize=(6, 4))

# Create the bar chart
bars = ax.bar(survival_by_class['Class'], survival_by_class['Survival Rate'],
              color=['tab:blue', 'tab:orange', 'tab:green'])

# Add data labels on top of each bar
for bar, val in zip(bars, survival_by_class['Survival Rate']):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
            f'{val:.0%}', ha='center', va='bottom', fontsize=11, fontweight='bold')

# Set chart titles and labels
ax.set_title('Survival rate by passenger class', pad=12)
ax.set_xlabel('Passenger class')
ax.set_ylabel('Survival rate')
ax.set_ylim(0, 1)

# Display the chart
plt.show()