from multi_panel_dashboard_graph import *

# --- Part 6 : Multi-Panel Dashboard with plt.subplots ---

# 1. Change the grid to 3 rows and 2 columns to accommodate 5 plots
fig, axes = plt.subplots(3, 2, figsize=(14, 15))
fig.suptitle("Titanic passenger data – a visual story", fontsize=16, fontweight='bold', y=1.01)

# ## --- Panel A: survival rate by class ---
ax = axes[0, 0]
surv = df.groupby('pclass')['survived'].mean()
ax.bar(['1st', '2nd', '3rd'], surv.values, color=['#1565C0', '#1976D2', '#90CAF9'])
ax.set_title('A. Survival rate by class')
ax.set_ylabel('Rate')
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.0%}'))

# ## --- Panel B: age histogram ---
ax = axes[0, 1]
ax.hist(df[df['survived'] == 0]['age'], bins=25, alpha=0.6, color='#E53935', label='Not survived')
ax.hist(df[df['survived'] == 1]['age'], bins=25, alpha=0.6, color='#43A047', label='Survived')
ax.set_title('B. Age distribution by outcome')
ax.set_xlabel('Age')
ax.set_ylabel('Count')
ax.legend(fontsize=8)

# ## --- Panel C: fare vs age scatter ---
ax = axes[1, 0]
for cls, c, lbl in zip([1, 2, 3], ['#1976D2', '#43A047', '#E53935'], ['1st', '2nd', '3rd']):
    s = df[df['pclass'] == cls]
    ax.scatter(s['age'], s['fare'], c=c, alpha=0.4, s=20, label=lbl)
ax.set_yscale('log')
ax.set_title('C. Fare vs age (log scale)')
ax.set_xlabel('Age')
ax.set_ylabel('Fare (log)')
ax.legend(title='Class', fontsize=8)

# ## --- Panel D: survival heatmap ---
ax = axes[1, 1]
pivot = df.pivot_table(values='survived', index='pclass', columns='embarked', aggfunc='mean')
sns.heatmap(pivot, annot=True, fmt='.0%', cmap='RdYlGn', linewidths=0.5, ax=ax, vmin=0, vmax=1, cbar=False)
ax.set_title('D. Survival rate: class x port')
ax.set_xlabel('Port')
ax.set_ylabel('Class')

# ## --- Panel E: passenger count by age group & sex (The 5th Panel) ---
ax = axes[2, 0]
sns.countplot(
    data=df,
    x='age_group',
    hue='sex',
    ax=ax,
    order=['Child', 'Teen', 'Young Adult', 'Adult', 'Senior'] # Ensures consistent order
)
ax.set_title('E. Passenger Count by Age Group and Sex')
ax.set_xlabel('Age Group')
ax.set_ylabel('Number of Passengers')
ax.legend(title='Sex')

# 2. Hide the remaining unused 6th subplot slot (bottom right)
axes[2, 1].axis('off')

# 3. Clean up formatting and save the figure
plt.tight_layout()
plt.savefig('titanic_dashboard.png', dpi=150, bbox_inches='tight')
plt.show()