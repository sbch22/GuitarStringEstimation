import matplotlib.pyplot as plt
import numpy as np

features = [
    # ("F0",                     0.4254, 0.0055),
    # ("Rel. Frequenzabweichungen",    0.1464, 0.0045),
    # ("Rel. Partialtonamplituden", 0.0919, 0.0038),
    # ("Betas",         0.0847, 0.0043),
    # ("Valide Partialtöne",         0.0568, 0.0016),
    ("SD",                    0.0467, 0.0033),
    ("Min",                    0.0376, 0.0018),
    ("Max",                    0.0369, 0.0030),
    ("Mode",                   0.0331, 0.0027),
    ("Var",                    0.0319, 0.0030),
    ("Mittel",                   0.0282, 0.0022),
    # ("Spectral Centroid",      0.0222, 0.0014),
    # ("Amplituden-Abklingkoeffizienten", 0.0202, 0.0022),
    ("Median",                 0.0193, 0.0028),
    ("Kurtosis",               0.0130, 0.0028),
    ("Schiefe",               0.0116, 0.0018),
]

labels = [f[0] for f in features]
importances = np.array([f[1] for f in features])
errors = np.array([f[2] for f in features])

fig, ax = plt.subplots(figsize=(8, 6))

y = np.arange(len(labels))
bars = ax.barh(y, importances, xerr=errors, align="center",
               color="#534AB7", ecolor="#3C3489",
               capsize=3, height=0.6, error_kw={"linewidth": 1.2})

ax.set_yticks(y)
ax.set_yticklabels(labels, fontfamily="monospace", fontsize=10)
ax.invert_yaxis()

ax.set_xlabel(
    "Mittlere Permutation Feature-Importance \n(Mittel über Solo & Comp, sowie 10 Wdh.)",
    fontsize=10
)
ax.set_xlim(left=0)
ax.xaxis.grid(True, linestyle="--", linewidth=0.5, alpha=0.5)
ax.set_axisbelow(True)
ax.spines[["top", "right"]].set_visible(False)

plt.tight_layout()
plt.savefig("feature_importance.pdf", bbox_inches="tight")
plt.show()