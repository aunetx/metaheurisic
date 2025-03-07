# %%

import numpy as np
import matplotlib.pyplot as plt

from utilitaires import *

# %%


def afficher_score_et_deviation(ax, data, N_agents):
    data[~np.isfinite(data)] = np.nan
    x = np.arange(data.shape[1]) * N_agents
    ax.fill_between(
        x,
        np.nanmin(data, axis=0),
        np.nanmax(data, axis=0),
        alpha=0.2,
        label="valeurs extrêmes",
        color="firebrick",
        lw=0,
    )
    data[~np.isfinite(data)] = np.inf
    ax.fill_between(
        x,
        np.nanquantile(data, 0.25, axis=0),
        np.nanquantile(data, 0.75, axis=0),
        alpha=0.3,
        label="quartiles",
        color="blue",
        lw=0,
    )
    data[~np.isfinite(data)] = np.nan
    ax.plot(x, np.nanmean(data, axis=0), label="moyenne", c="black", ls=":")
    data[~np.isfinite(data)] = np.inf
    ax.plot(x, np.nanmedian(data, axis=0), label="médiane", c="olivedrab", ls="--")

    ax.set_xlim((0, x[-1]))
    ax.legend()


instance_name = "inst1"
N_agents = 50
save_name = f"{instance_name}_avec_hybridation_{N_agents}A"
save_file_scores = f"best_scores/{save_name}_scores.txt"
save_file_distances = f"best_scores/{save_name}_distances.txt"

scores_array = np.array(lire_resultats(save_file_scores))
distances_array = np.array(lire_resultats(save_file_distances))
n_iterations = scores_array.shape[1] - 1

plot_separes = False
if plot_separes:
    for arr, name in zip([scores_array, distances_array], ["Score", "Distance"]):
        fig, ax = plt.subplots(figsize=(7, 5))

        afficher_score_et_deviation(ax, arr, N_agents)
        ax.set_xlabel(f"Nombre d'évaluations ({N_agents} agents, {n_iterations} itérations)")
        ax.set_ylabel(name)
        ax.set_title(f"{name} et sa dispersion")

        fig.tight_layout()
        plt.show()
else:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 3.5))

    afficher_score_et_deviation(ax1, scores_array, N_agents)
    ax1.set_xlabel(f"Nombre d'évaluations ({N_agents} agents, {n_iterations} itérations)")
    ax1.set_ylabel("Score")
    ax1.set_title("Score et sa dispersion")

    afficher_score_et_deviation(ax2, distances_array, N_agents)
    ax2.set_xlabel(f"Nombre d'évaluations ({N_agents} agents, {n_iterations} itérations)")
    ax2.set_ylabel("Distance")
    ax2.set_title("Distance et sa dispersion")

    fig.tight_layout()
    plt.show()
