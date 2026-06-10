import pandas as pd, matplotlib.pyplot as plt
df = pd.read_csv('results/tier3_results.csv')
vetted = df[df['vetting_passed']]
fig, ax = plt.subplots(figsize=(10, 6))
ax.hist(vetted['cnn_probability'], bins=50, color='#4A90E2', edgecolor='black')
ax.axvline(0.9, color='red', linestyle='--', label='Strong endorsement (≥0.9)')
ax.axvline(0.95, color='darkred', linestyle='--', label='Very strong endorsement(≥0.95)')
ax.set_xlabel('CNN planet probability', fontsize=12)
ax.set_ylabel('Number of TESS candidates', fontsize=12)
ax.set_title('Stellaris CNN scores: 4,002 unresolved TESS candidates', fontsize=13)
ax.legend()
plt.tight_layout()
plt.savefig('linkedin_score_distribution.png', dpi=150)