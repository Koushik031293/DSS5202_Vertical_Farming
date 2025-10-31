import numpy as np
import matplotlib.pyplot as plt

categories = ['GWP100','HOFP','PMFP','AP','EOFP','FFP']

vf_raw = [5.187870303, 0.003474978, 0.002242795, 0.003934676, 0.001329544, 97.82866033]
tf_raw = [0.61193, 0.00036402, 0.00023526, 0.0006483, 0.0001865, 10.481]

# Normalize each category relative to its max value
vf_norm, tf_norm = [], []
for v, t in zip(vf_raw, tf_raw):
    hi = max(v, t)
    vf_norm.append(v / hi)
    tf_norm.append(t / hi)

# Close polygons
vf_plot = vf_norm + vf_norm[:1]
tf_plot = tf_norm + tf_norm[:1]
angles = np.linspace(0, 2*np.pi, len(categories), endpoint=False).tolist()
angles += angles[:1]

# Plot
fig, ax = plt.subplots(figsize=(6,6), subplot_kw=dict(polar=True))
ax.plot(angles, vf_plot, color='#007acc', linewidth=2, label='Vertical Farming')
ax.fill(angles, vf_plot, color='#007acc', alpha=0.25)
ax.plot(angles, tf_plot, color='#4CAF50', linewidth=2, label='Traditional Farming')
ax.fill(angles, tf_plot, color='#4CAF50', alpha=0.25)

ax.set_xticks(angles[:-1])
ax.set_xticklabels(categories)
ax.set_yticklabels([])
ax.set_title('Normalized LCIA Comparison (Relative Scale)', size=13)
ax.legend(
    loc='lower center',
    bbox_to_anchor=(0.5, -0.25),
    ncol=2,
    frameon=False
)
# ax.legend(loc='upper right', bbox_to_anchor=(1.25, 1.1))

plt.tight_layout()
plt.savefig('charts_both/VF_TF_radar.png', dpi=300)
plt.show()