
import matplotlib.pyplot as plt
import numpy as np


categories = ['GWP100', 'HOFP', 'PMFP', 'AP', 'EOFP', 'FFP']
vertical = [5.1879, 0.00347, 0.00224, 0.00393, 0.00133, 97.83]
traditional = [0.61193, 0.00036402, 0.00023526, 0.0006483, 0.0001865, 10.481]

x = np.arange(len(categories))
width = 0.35

plt.figure(figsize=(8,4))
plt.bar(x - width/2, vertical, width, label='Vertical Farming', color='#007acc')
plt.bar(x + width/2, traditional, width, label='Traditional Farming', color='#4CAF50')

plt.yscale('log')
plt.ylabel('Impact per kg (category units, log scale)')
plt.title('Total LCIA Impact by Category — Vertical vs Traditional Farming')
plt.xticks(x, categories)
plt.legend()
plt.grid(True, which="major", axis="y", linestyle="--", alpha=0.5)
plt.savefig('charts_both/VF_TF_totals_bar.png', dpi=300)
# Add numeric labels
for i, v in enumerate(vertical):
    plt.text(i - width/2, v * 1.3, f'{v:.3f}', ha='center', fontsize=7)
for i, v in enumerate(traditional):
    plt.text(i + width/2, v * 1.3, f'{v:.3f}', ha='center', fontsize=7)

plt.tight_layout()
plt.show()