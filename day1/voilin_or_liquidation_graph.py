from heat_map_graph import *

# # Part 5: Violin + Strip Plot using seaborn

fig, ax = plt.subplots(figsize=(8, 5))

sns.violinplot(
    data=df, 
    x='pclass', 
    y='age', 
    hue='survived',
    split=True, 
    palette={0: '#EF9A9A', 1: '#A5D6A7'},
    inner='quartile', 
    ax=ax
)

ax.set_title('Age distribution by class and survival (violin)')
ax.set_xlabel('Passenger class')
ax.set_ylabel('Age (years)')

# Correcting the typo from 'get_legend_handles_labels'
handles, _ = ax.get_legend_handles_labels()
ax.legend(handles, ['Did not survive', 'Survived'], title='Outcome')

plt.tight_layout()
plt.show()