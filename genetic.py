# %%

import numpy as np
import matplotlib.pyplot as plt
import math
import copy

from utilitaires import *

# %%


def mutation_echange(parcours, p=0.1):
    """
    Échange deux villes du parcours au hasard (sauf la première)
    avec une probabilité p.
    """
    if np.random.rand() > p:
        return parcours

    i1 = np.random.randint(1, len(parcours))
    i2 = np.random.randint(1, len(parcours))
    while i2 == i1:
        i2 = np.random.randint(1, len(parcours))

    nouveau_parcours = parcours.copy()
    nouveau_parcours[i1] = parcours[i2]
    nouveau_parcours[i2] = parcours[i1]
    return nouveau_parcours


def mutation_insertion(parcours, p=0.1):
    """
    Choisit une ville au hasard (sauf la première), et l'insert à une place
    au hasard sur le parcours (sauf au départ) avec une probabilité p.
    """
    if np.random.rand() > p:
        return parcours

    i_ville = np.random.randint(1, len(parcours))
    i_insertion = np.random.randint(1, len(parcours))

    ville = parcours[i_ville]
    nouveau_parcours = np.delete(parcours, i_ville)
    nouveau_parcours = np.insert(nouveau_parcours, i_insertion, ville)
    return nouveau_parcours


def selectionner_tranche(agents, n_meilleurs_scores, n_meilleurs_penalites, n_hasard):
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
    p_mutation = 0.1

    def __init__(self, instance, dist_mat, parcours=None, p_mutation=0.1):
        self.instance = instance
        self.dist_mat = dist_mat
        self.p_mutation = p_mutation

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
        distance, score, n_penalites = evaluation(
            self.instance, self.dist_mat, self.parcours, lambda x: self.act_erreur(x)
        )

        self.score = score
        self.n_penalites = n_penalites
        self.distance = distance if n_penalites == 0 else np.inf

        return self

    def muter(self):
        self.parcours = mutation_echange(self.parcours, self.p_mutation)
        self.parcours = mutation_insertion(self.parcours, self.p_mutation)
        self.iterer()
        return self

    def enfanter(self):
        enfant = self.copier().iterer()
        return enfant

    def copier(self):
        return copy.deepcopy(self)


# %%

fig, (ax1, ax2, ax3) = plt.subplots(3, figsize=(8, 16), sharex=True)

instance = charger_instance("data/inst1")
dist_mat = compute_dist_mat(instance)

N_agents = 300

garder_n_parents = 70
garder_n_meilleurs_scores_parents = 20
garder_n_meilleurs_penalites_parents = 10
garder_n_autres_parents = (
    garder_n_parents - garder_n_meilleurs_scores_parents - garder_n_meilleurs_penalites_parents
)

garder_n_enfants = N_agents - garder_n_parents
garder_n_meilleurs_scores_enfants = 50
garder_n_meilleurs_penalites_enfants = 25
garder_n_autres_enfants = (
    garder_n_enfants - garder_n_meilleurs_scores_enfants - garder_n_meilleurs_penalites_enfants
)

agents = [Agent(instance, dist_mat, p_mutation=0.65) for i in range(N_agents)]

scores = [min(agent.score for agent in agents)]
distances = [min(agent.distance for agent in agents)]
n_penalites = [min(agent.n_penalites for agent in agents)]

ax1.scatter(
    [0] * N_agents, [agent.n_penalites for agent in agents], c="black", marker=".", alpha=0.4
)
ax2.scatter([0] * N_agents, [agent.score for agent in agents], c="black", marker=".", alpha=0.05)

N_iteration = 250
for i in range(1, N_iteration):
    print(f"{i} / {N_iteration}", end="\r")

    parents = [agent.copier().iterer() for agent in agents]
    enfants = [agent.copier().muter() for agent in agents]

    parents_selectionnes = selectionner_tranche(
        parents,
        garder_n_meilleurs_scores_parents,
        garder_n_meilleurs_penalites_parents,
        garder_n_autres_parents,
    )
    enfants_selectionnes = selectionner_tranche(
        enfants,
        garder_n_meilleurs_scores_enfants,
        garder_n_meilleurs_penalites_enfants,
        garder_n_autres_enfants,
    )

    agents = parents_selectionnes + enfants_selectionnes
    assert len(agents) == N_agents

    scores.append(min(agent.score for agent in agents))
    distances.append(min(agent.distance for agent in agents))
    n_penalites.append(min(agent.n_penalites for agent in agents))

    ax1.scatter(
        [i] * N_agents, [agent.n_penalites for agent in agents], c="black", marker=".", alpha=0.4
    )
    ax2.scatter(
        [i] * N_agents, [agent.score for agent in agents], c="black", marker=".", alpha=0.05
    )


ax1.set_ylim([0, None])
ax1.set_ylabel("N pénalités")
ax2.set_ylabel("Scores")
ax2.set_yscale(("log"))

ax3.plot(scores, c="red")
ax3.set_ylabel("Meilleur score")
ax3.set_xlabel("Itération")
ax3t = ax3.twinx()
ax3t.plot(n_penalites, c="black", ls="--")
ax3t.set_ylabel("Meilleure pénalité")

fig.tight_layout()


# %%

meilleur_agent = sorted(agents, key=lambda agent: agent.score)[0]
meilleur_agent.score
