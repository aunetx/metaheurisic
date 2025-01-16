# %%

import numpy as np
import matplotlib.pyplot as plt
import math
import copy
from tqdm import tqdm

from utilitaires import *

# %%


def mutation_echange(parcours):
    """
    Échange deux villes du parcours au hasard (sauf la première).
    """
    i1 = np.random.randint(1, len(parcours))
    i2 = np.random.randint(1, len(parcours))
    while i2 == i1:
        i2 = np.random.randint(1, len(parcours))

    nouveau_parcours = parcours.copy()
    nouveau_parcours[i1] = parcours[i2]
    nouveau_parcours[i2] = parcours[i1]
    return nouveau_parcours


def mutation_insertion(parcours):
    """
    Choisit une ville au hasard (sauf la première), et l'insert à une place
    au hasard sur le parcours (sauf au départ).
    """
    i_ville = np.random.randint(1, len(parcours))
    i_insertion = np.random.randint(1, len(parcours))

    ville = parcours[i_ville]
    nouveau_parcours = np.delete(parcours, i_ville)
    nouveau_parcours = np.insert(nouveau_parcours, i_insertion, ville)
    return nouveau_parcours


def selectionner_tranche(agents, n_agents, ratio_meilleurs_scores, ratio_meilleurs_penalites):
    n_meilleurs_scores = int(n_agents * ratio_meilleurs_scores)
    n_meilleurs_penalites = int(n_agents * ratio_meilleurs_penalites)
    n_hasard = n_agents - n_meilleurs_scores - n_meilleurs_penalites

    if n_meilleurs_scores > 0:
        agents = sorted(agents, key=lambda agent: agent.score)
    meilleurs_scores = agents[:n_meilleurs_scores]
    agents = agents[n_meilleurs_scores:]

    if n_meilleurs_penalites > 0:
        agents = sorted(agents, key=lambda agent: agent.n_penalites)
    meilleurs_penalites = agents[:n_meilleurs_penalites]
    agents = agents[n_meilleurs_penalites:]

    probabilites = np.array([1 / agent.score**2 for agent in agents])
    probabilites /= np.sum(probabilites)
    hasard = list(np.random.choice(agents, size=n_hasard, replace=False, p=probabilites))

    return meilleurs_scores + meilleurs_penalites + hasard


class Agent:
    score = 0
    distance = 0
    iteration = 0
    p_mutation = 0.8
    p_mutation_echange = 0.5

    def __init__(self, instance, dist_mat, parcours=None, p_mutation=0.8, p_mutation_echange=0.5):
        self.instance = instance
        self.dist_mat = dist_mat
        self.p_mutation = p_mutation
        self.p_mutation_echange = p_mutation_echange

        if parcours:
            self.parcours = parcours
        else:
            self.parcours = [1] + np.random.permutation(
                np.arange(2, len(self.instance) + 1)
            ).tolist()
        self.reevaluer()

    def act_erreur(self, erreur):
        return 1000 + erreur * self.iteration

    def iterer(self):
        self.iteration += 1
        self.reevaluer()
        return self

    def reevaluer(self):
        self.distance, self.score, self.n_penalites = evaluation(
            self.instance, self.dist_mat, self.parcours, lambda x: self.act_erreur(x)
        )

        return self

    def muter(self):
        if np.random.rand() < self.p_mutation:
            if np.random.rand() < self.p_mutation_echange:
                self.parcours = mutation_echange(self.parcours)
            else:
                self.parcours = mutation_insertion(self.parcours)
        self.iterer()
        return self

    def afficher_parcours(self):
        import igraph as ig

        coords = [(self.instance[i]["x"], self.instance[i]["y"]) for i in self.instance]
        p = self.parcours
        v = [(p[i] - 1, p[i + 1] - 1) for i in range(len(p) - 1)] + [(p[-1] - 1, p[0] - 1)]

        g = ig.Graph()
        g.add_vertices(self.parcours)
        g.add_edges(v)

        fig, ax = plt.subplots()

        g.vs["label"] = g.vs["name"]
        g.vs["label_size"] = 7
        ig.plot(g, target=ax, layout=coords)

    def copier(self):
        return copy.copy(self)


# %%

fig, (ax1, ax2, ax3) = plt.subplots(3, figsize=(8, 16), sharex=True)

instance = charger_instance("data/inst2")
dist_mat = compute_dist_mat(instance)

N_agents = 70

ratio_parents = 0.3
ratio_meilleurs_scores_parents = 0.2
ratio_meilleurs_penalites_parents = 0.1
ratio_meilleurs_scores_enfants = 0.2
ratio_meilleurs_penalites_enfants = 0.1

n_parents = int(N_agents * ratio_parents)
n_enfants = N_agents - n_parents

continuer = False
if not continuer:
    agents = [Agent(instance, dist_mat) for i in range(N_agents)]

    iterations = [[agent.iteration for agent in agents]]
    scores = [[agent.score for agent in agents]]
    distances = [[agent.distance for agent in agents]]
    n_penalites = [[agent.n_penalites for agent in agents]]

ax1.scatter(
    [agent.iteration for agent in agents],
    [agent.n_penalites for agent in agents],
    c="black",
    marker=".",
    alpha=0.4,
)
ax2.scatter(
    [agent.iteration for agent in agents],
    [agent.score for agent in agents],
    c="black",
    marker=".",
    alpha=0.05,
)

N_iteration = 4000
tqdm_range = tqdm(range(1, N_iteration))
for i in tqdm_range:
    parents = [agent.copier().iterer() for agent in agents]
    enfants = [agent.muter() for agent in agents]

    parents_selectionnes = selectionner_tranche(
        parents,
        n_parents,
        ratio_meilleurs_scores_parents,
        ratio_meilleurs_penalites_parents,
    )
    enfants_selectionnes = selectionner_tranche(
        enfants,
        n_enfants,
        ratio_meilleurs_scores_enfants,
        ratio_meilleurs_penalites_enfants,
    )

    agents = parents_selectionnes + enfants_selectionnes
    assert len(agents) == N_agents

    iterations.append([agent.iteration for agent in agents])
    scores.append([agent.score for agent in agents])
    distances.append([agent.distance for agent in agents])
    n_penalites.append([agent.n_penalites for agent in agents])

ax1.scatter(
    iterations,
    n_penalites,
    c="black",
    marker=".",
    alpha=0.4,
)
ax1.set_ylim([0, None])
ax1.set_ylabel("N pénalités")

ax2.scatter(
    iterations,
    scores,
    c="black",
    marker=".",
    alpha=0.05,
)
ax2.set_ylabel("Scores")
ax2.set_yscale(("log"))

ax3.plot([min(score) for score in scores], c="red")
ax3.set_ylabel("Meilleur score")
ax3.set_xlabel("Itération")
ax3t = ax3.twinx()
ax3t.plot([min(distance) for distance in distances], c="black", ls="--")
ax3t.set_ylabel("Meilleure distance")

fig.tight_layout()


# %%

meilleur_agent = sorted(agents, key=lambda agent: agent.score)[0]
print(meilleur_agent.parcours)
meilleur_agent.afficher_parcours()
