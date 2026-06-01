from group_bar_chart_graph import *

# # Part 3 : Scatter plot

fig, ax = plt.subplots(figsize=(7, 5))

colors = {1: '#1976D2', 2: '#43A047', 3: '#E53935'}
labels = {1: '1st class', 2: '2nd class', 3: '3rd class'}

for cls in [1, 2, 3]:
    subset = df[df['pclass'] == cls]
    ax.scatter(
        subset['age'], 
        subset['fare'],
        c=colors[cls],
        alpha=0.5, 
        s=40, 
        edgecolors='white', 
        linewidth=0.3,
        label=labels[cls]
    )

ax.set_title('Fare vs age by passenger class')
ax.set_xlabel('Age (years)')
ax.set_ylabel('Ticket fare (£)')
ax.legend(title='Class')
ax.set_yscale('log')  # log scale because a few very high fares skew the axis

# # Annotate an outlier
max_fare_idx = df['fare'].idxmax()
ax.annotate(
    f"Highest fare: £{df.loc[max_fare_idx, 'fare']:.0f}",
    xy=(df.loc[max_fare_idx, 'age'], df.loc[max_fare_idx, 'fare']),
    xytext=(55, 450), 
    arrowprops=dict(arrowstyle='->', color='gray'),
    fontsize=9, 
    color='gray'
)

plt.tight_layout()
plt.show()