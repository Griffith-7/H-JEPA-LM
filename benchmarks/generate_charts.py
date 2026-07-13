import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import os

plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.size': 11,
    'axes.titlesize': 14,
    'axes.labelsize': 12,
    'figure.facecolor': 'white',
    'axes.facecolor': '#fafafa',
    'axes.grid': True,
    'grid.alpha': 0.3,
    'grid.linestyle': '--',
})

COLORS = {
    'GPT-NTP': '#6366f1',
    'BERT-MLM': '#f59e0b',
    'LLM-JEPA': '#ef4444',
    'JEPA-LM': '#10b981',
    'H-JEPA-LM': '#3b82f6',
}

LABELS = {
    'GPT-NTP': 'GPT (NTP)',
    'BERT-MLM': 'BERT (MLM)',
    'LLM-JEPA': 'LLM-JEPA',
    'JEPA-LM': 'JEPA-LM',
    'H-JEPA-LM': 'H-JEPA-LM (Ours)',
}

with open('benchmark_hjepa_results/results.json', 'r') as f:
    data = json.load(f)

models = ['GPT-NTP', 'BERT-MLM', 'LLM-JEPA', 'JEPA-LM', 'H-JEPA-LM']
cosine = [data[m]['diversity']['cosine_sim'] for m in models]
embed_std = [data[m]['diversity']['embedding_std'] for m in models]
sv_ratio = [data[m]['diversity']['sv_ratio'] for m in models]
params = [data[m]['params'] / 1e6 for m in models]
colors = [COLORS[m] for m in models]
labels = [LABELS[m] for m in models]

fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.suptitle('Embedding Diversity Benchmarks', fontsize=16, fontweight='bold', y=1.02)

bars = axes[0].bar(labels, cosine, color=colors, edgecolor='white', linewidth=1.5, width=0.6)
axes[0].set_title('Mean Cosine Similarity (lower = more diverse)')
axes[0].set_ylabel('Cosine Similarity')
axes[0].set_ylim(0.7, 1.02)
axes[0].tick_params(axis='x', rotation=25)
for bar, val in zip(bars, cosine):
    axes[0].text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.005,
                 f'{val:.3f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
best_idx = np.argmin(cosine)
bars[best_idx].set_edgecolor('#16a34a')
bars[best_idx].set_linewidth(2.5)

bars = axes[1].bar(labels, embed_std, color=colors, edgecolor='white', linewidth=1.5, width=0.6)
axes[1].set_title('Embedding Std Dev (higher = more diverse)')
axes[1].set_ylabel('Std Dev')
axes[1].set_ylim(0, 0.4)
axes[1].tick_params(axis='x', rotation=25)
for bar, val in zip(bars, embed_std):
    axes[1].text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.005,
                 f'{val:.3f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
best_idx = np.argmax(embed_std)
bars[best_idx].set_edgecolor('#16a34a')
bars[best_idx].set_linewidth(2.5)

bars = axes[2].bar(labels, sv_ratio, color=colors, edgecolor='white', linewidth=1.5, width=0.6)
axes[2].set_title('Singular Value Ratio (lower = more balanced)')
axes[2].set_ylabel('SV Ratio')
axes[2].set_ylim(0.3, 1.0)
axes[2].tick_params(axis='x', rotation=25)
for bar, val in zip(bars, sv_ratio):
    axes[2].text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.005,
                 f'{val:.3f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
best_idx = np.argmin(sv_ratio)
bars[best_idx].set_edgecolor('#16a34a')
bars[best_idx].set_linewidth(2.5)

plt.tight_layout()
os.makedirs('benchmarks', exist_ok=True)
plt.savefig('benchmarks/diversity_benchmarks.png', dpi=200, bbox_inches='tight', facecolor='white')
plt.close()

fig, ax = plt.subplots(figsize=(10, 6))
fig.suptitle('Cosine Similarity by Model (lower = more diverse embeddings)', fontsize=14, fontweight='bold')

bars = ax.bar(labels, cosine, color=colors, edgecolor='white', linewidth=1.5, width=0.5)
ax.set_ylabel('Mean Cosine Similarity')
ax.set_ylim(0.7, 1.02)
ax.axhline(y=1.0, color='gray', linestyle=':', alpha=0.5, label='Collapse (1.0)')
ax.axhline(y=0.85, color='gray', linestyle='--', alpha=0.3, label='BERT baseline')

for bar, val in zip(bars, cosine):
    ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.005,
            f'{val:.3f}', ha='center', va='bottom', fontsize=11, fontweight='bold')

legend_elements = [plt.Rectangle((0,0),1,1, facecolor=c, edgecolor='white', label=l) 
                   for c, l in zip(colors, labels)]
ax.legend(handles=legend_elements, loc='upper right', fontsize=9)
ax.annotate('23% improvement\nover LLM-JEPA',
            xy=(4, 0.774), xytext=(3.2, 0.82),
            arrowprops=dict(arrowstyle='->', color='#3b82f6', lw=2),
            fontsize=10, fontweight='bold', color='#3b82f6')

plt.tight_layout()
plt.savefig('benchmarks/cosine_similarity.png', dpi=200, bbox_inches='tight', facecolor='white')
plt.close()

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle('H-JEPA-LM Architecture Overview', fontsize=16, fontweight='bold')

ax = axes[0]
ax.set_xlim(0, 10)
ax.set_ylim(0, 10)
ax.axis('off')
ax.set_title('Prediction Architecture', fontsize=13)

boxes = [
    (1, 8.5, 'Input Text\n"The cat sat on the ___"', '#e0e7ff'),
    (1, 6.5, 'Hierarchical Encoder\n(bidirectional)', '#dbeafe'),
    (1, 4.5, 'Hierarchical Predictor\n(multi-level)', '#d1fae5'),
    (1, 2.5, 'EMA Target Encoder\n(stop-gradient)', '#fef3c7'),
    (1, 0.5, 'JEPA Loss\n(cosine similarity)', '#fee2e2'),
]
for x, y, text, color in boxes:
    rect = plt.Rectangle((x-0.8, y-0.5), 3.6, 1.0, facecolor=color, edgecolor='#374151', 
                         linewidth=1.5, transform=ax.transData, zorder=2)
    ax.add_patch(rect)
    ax.text(x+1, y, text, ha='center', va='center', fontsize=9, fontweight='bold', zorder=3)

for i in range(len(boxes)-1):
    ax.annotate('', xy=(1, boxes[i+1][1]+0.5), xytext=(1, boxes[i][1]-0.5),
                arrowprops=dict(arrowstyle='->', color='#6b7280', lw=1.5))

ax2 = axes[1]
ax2.set_xlim(0, 10)
ax2.set_ylim(0, 10)
ax2.axis('off')
ax2.set_title('World Model & Planning', fontsize=13)

boxes2 = [
    (1, 8.5, 'Action Input\n"Move forward"', '#e0e7ff'),
    (1, 6.5, 'Action Encoder\n(fused into highest level)', '#dbeafe'),
    (1, 4.5, 'World Model\n(latent dynamics)', '#d1fae5'),
    (1, 2.5, 'Action Sequence\nOptimizer (K=10)', '#fef3c7'),
    (1, 0.5, 'Best Action\n"Move right then up"', '#fee2e2'),
]
for x, y, text, color in boxes2:
    rect = plt.Rectangle((x-0.8, y-0.5), 3.6, 1.0, facecolor=color, edgecolor='#374151',
                         linewidth=1.5, transform=ax2.transData, zorder=2)
    ax2.add_patch(rect)
    ax2.text(x+1, y, text, ha='center', va='center', fontsize=9, fontweight='bold', zorder=3)

for i in range(len(boxes2)-1):
    ax2.annotate('', xy=(1, boxes2[i+1][1]+0.5), xytext=(1, boxes2[i][1]-0.5),
                 arrowprops=dict(arrowstyle='->', color='#6b7280', lw=1.5))

plt.tight_layout()
plt.savefig('benchmarks/architecture_overview.png', dpi=200, bbox_inches='tight', facecolor='white')
plt.close()

fig, ax = plt.subplots(figsize=(10, 5))
fig.suptitle('Parameter Count Comparison', fontsize=14, fontweight='bold')

bars = ax.barh(labels, params, color=colors, edgecolor='white', linewidth=1.5, height=0.5)
ax.set_xlabel('Parameters (millions)')
for bar, val in zip(bars, params):
    ax.text(bar.get_width() + 0.1, bar.get_y() + bar.get_height()/2.,
            f'{val:.1f}M', ha='left', va='center', fontsize=10, fontweight='bold')

plt.tight_layout()
plt.savefig('benchmarks/parameter_comparison.png', dpi=200, bbox_inches='tight', facecolor='white')
plt.close()

print("Generated 4 charts:")
print("  benchmarks/diversity_benchmarks.png")
print("  benchmarks/cosine_similarity.png")
print("  benchmarks/architecture_overview.png")
print("  benchmarks/parameter_comparison.png")
