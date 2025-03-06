# %%

import numpy as np
import matplotlib.pyplot as plt

from utilitaires import *

# %%


def afficher_score_et_deviation(ax, data):
    data[~np.isfinite(data)] = np.nan
    x = np.arange(data.shape[1])
    est = np.nanmean(data, axis=0)
    sd = np.nanstd(data, axis=0)
    dmin = np.nanmin(data, axis=0)
    dmax = np.nanmax(data, axis=0)
    ax.fill_between(x, dmin, dmax, alpha=0.2, label="valeurs extrêmes", color="firebrick", lw=0)
    ax.fill_between(
        x, est - sd, est + sd, alpha=0.3, label="déviation standard", color="olivedrab", lw=0
    )
    ax.plot(x, est, label="moyenne", c="black")
    ax.legend()
    ax.margins(x=0)


instance_name = "inst1"
save_name = f"{instance_name}_sans_hybridation"
save_file_scores = f"best_scores/{save_name}_scores.txt"
save_file_distances = f"best_scores/{save_name}_distances.txt"

scores_array = np.array(lire_resultats(save_file_scores))

# Plot the data with error bars
fig, ax = plt.subplots(figsize=(7, 7))
afficher_score_et_deviation(ax, scores_array)
ax.set_xlabel("Itération")
ax.set_ylabel("Score")
ax.set_title("Score avec barres d'erreurs")
fig.tight_layout()
plt.show()

scores_array = np.array(lire_resultats(save_file_distances))

# Plot the data with error bars
fig, ax = plt.subplots(figsize=(7, 7))
afficher_score_et_deviation(ax, scores_array)
ax.set_xlabel("Itération")
ax.set_ylabel("Distance")
ax.set_title("Distance avec barres d'erreurs")
fig.tight_layout()
plt.show()
