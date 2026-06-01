from scatter_plot_graph import *

# # Survival rate pivot: class x embarkation port
pivot = df.pivot_table(values='survived', index='pclass', columns='embarked', aggfunc='mean')

fig, ax = plt.subplots(figsize=(6, 4))
sns.heatmap(
    pivot, 
    annot=True, 
    fmt='.0%', 
    cmap='RdYlGn',
    linewidths=0.5, 
    ax=ax, 
    vmin=0, 
    vmax=1,
    cbar_kws={'label': 'Survival rate'}
)

ax.set_title('Survival rate by class & embarkation port')
ax.set_xlabel('Embarkation port (C=Cherbourg, Q=Queenstown, S=Southampton)')
ax.set_ylabel('Passenger class')

plt.tight_layout()
plt.show()