"""Regenerate README figures from the raw Kaggle Stroke dataset.

Run: python generate_figures.py
Reads: datasets/raw/dataset.csv
Writes: figures/*.png
"""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style='whitegrid')
plt.rcParams.update({
    'figure.dpi': 110,
    'savefig.dpi': 160,
    'font.family': 'DejaVu Sans',
    'axes.titleweight': 'bold',
})

STROKE_COLORS = {0: '#4C8BF5', 1: '#E74C3C'}
FIG = Path('figures')
FIG.mkdir(exist_ok=True)

df = pd.read_csv('datasets/raw/dataset.csv')

# -------- Fig 1: Class imbalance --------
fig, ax = plt.subplots(figsize=(7.5, 4.5))
counts = df['stroke'].value_counts().sort_index()
labels = ['No stroke', 'Stroke']
colors = [STROKE_COLORS[0], STROKE_COLORS[1]]
bars = ax.bar(labels, counts, color=colors, edgecolor='white', linewidth=2)
for b, n in zip(bars, counts):
    pct = n / counts.sum() * 100
    ax.text(b.get_x() + b.get_width() / 2, b.get_height(),
            f'{n:,}\n({pct:.1f}%)', ha='center', va='bottom', fontsize=11)
ax.set_ylabel('Patients')
ax.set_title('Target class distribution — heavy imbalance')
ax.set_ylim(0, counts.max() * 1.18)
plt.tight_layout()
plt.savefig(FIG / 'class-imbalance.png', bbox_inches='tight')
plt.close()

# -------- Fig 2: Age distribution by stroke --------
fig, ax = plt.subplots(figsize=(10, 4.5))
for label, name in [(0, 'No stroke'), (1, 'Stroke')]:
    sub = df[df['stroke'] == label]
    ax.hist(sub['age'], bins=40, color=STROKE_COLORS[label], alpha=0.6,
            label=f'{name} (n={len(sub):,})', density=True, edgecolor='white')
ax.set_xlabel('Age (years)')
ax.set_ylabel('Density')
ax.set_title('Age distribution — stroke vs. no stroke')
ax.legend()
plt.tight_layout()
plt.savefig(FIG / 'age-by-stroke.png', bbox_inches='tight')
plt.close()

# -------- Fig 3: Stroke rate by categorical features --------
cat_features = ['hypertension', 'heart_disease', 'ever_married',
                'smoking_status', 'gender', 'work_type']

fig, axes = plt.subplots(2, 3, figsize=(16, 8.5))
axes = axes.flatten()
for ax, col in zip(axes, cat_features):
    rate = (df.groupby(col)['stroke'].mean() * 100).sort_values(ascending=False)
    bars = ax.bar(rate.index.astype(str), rate.values,
                  color='#A23B72', edgecolor='white', linewidth=1.5)
    for b, v in zip(bars, rate):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height(),
                f'{v:.1f}%', ha='center', va='bottom', fontsize=10)
    ax.set_title(col.replace('_', ' ').title())
    ax.set_ylabel('Stroke rate (%)')
    ax.tick_params(axis='x', rotation=20)
    ax.set_ylim(0, max(rate.values) * 1.25)

plt.suptitle('Stroke rate by feature', fontsize=14, fontweight='bold', y=1.01)
plt.tight_layout()
plt.savefig(FIG / 'stroke-rate-by-feature.png', bbox_inches='tight')
plt.close()

# -------- Fig 4: Correlation heatmap (numeric features) --------
num_features = ['age', 'hypertension', 'heart_disease', 'avg_glucose_level', 'bmi', 'stroke']
corr = df[num_features].corr()
fig, ax = plt.subplots(figsize=(7.5, 6))
sns.heatmap(corr, annot=True, fmt='.2f', cmap='RdBu_r', center=0, square=True,
            cbar_kws={'label': 'Pearson correlation'}, ax=ax,
            annot_kws={'size': 10})
ax.set_title('Correlation heatmap — numeric features', pad=12)
plt.tight_layout()
plt.savefig(FIG / 'correlation-heatmap.png', bbox_inches='tight')
plt.close()

print('Generated figures:')
for f in sorted(FIG.glob('*.png')):
    print(f'  {f.name} ({f.stat().st_size / 1024:.0f} KB)')
