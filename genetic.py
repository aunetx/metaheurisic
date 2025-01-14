# %%

import numpy as np
import matplotlib.pyplot as plt
import math
import copy

from utilitaires import *
import scoreEtudiant

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


def act_erreur(erreur, iteration):
    return erreur


def trier_agents(agents):
    return sorted(agents, key=lambda agent: agent.score)


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
            self.parcours = np.random.permutation(np.arange(1, len(self.instance) + 1)).tolist()
        self.recalculer_distance()
        self.recalculer_score()

    def act_erreur(self, erreur):
        return erreur * self.iteration

    def iterer(self):
        self.iteration += 1
        return self

    def recalculer_distance(self):
        self.distance = evaluation(
            self.instance, self.dist_mat, self.parcours, lambda x: np.inf if x != 0 else 0
        )
        return self

    def recalculer_score(self):
        self.score = evaluation(
            self.instance, self.dist_mat, self.parcours, lambda x: self.act_erreur(x)
        )
        return self

    def muter(self):
        self.parcours = mutation_echange(self.parcours,self.p_mutation)
        self.parcours = mutation_insertion(self.parcours,self.p_mutation)
        self.iterer()
        self.recalculer_distance()
        self.recalculer_score()
        return self

    def enfanter(self):
        enfant = self.copier().iterer()
        enfant.recalculer_score()
        return enfant

    def copier(self):
        return copy.deepcopy(self)


# %%

instance = charger_instance("data/inst1")
dist_mat = compute_dist_mat(instance)

N_agents = 300
garder_n_parents = 50
agents = trier_agents(Agent(instance, dist_mat,p_mutation=0.2) for i in range(N_agents))

scores = [min(agent.score for agent in agents)]
distances = [min(agent.distance for agent in agents)]

N_iteration = 200
for i in range(N_iteration):
    parents = [agent.enfanter() for agent in agents]

    for agent in agents:
        agent.muter()
    enfants = trier_agents(agents)

    agents = trier_agents(parents[:garder_n_parents] + enfants[:-garder_n_parents])

    scores.append(min(agent.score for agent in agents))
    distances.append(min(agent.distance for agent in agents))


plt.plot(scores, c="red")
# plt.twinx()
# plt.plot(distances, c="black", ls="--")


# %%

N_iteration = 10000
scores = []
scores = [evaluation(instance, dist_mat, parcours, lambda x: act_erreur(x, i))]
distances = [evaluation(instance, dist_mat, parcours)]

for i in range(N_iteration):
    nouveau_parcours = mutation_echange(parcours)
    nouveau_parcours = mutation_insertion(parcours)
    nouveau_score = evaluation(instance, dist_mat, nouveau_parcours, lambda x: act_erreur(x, i))

    if nouveau_score < min(scores[-10:]):
        parcours = nouveau_parcours
        scores.append(nouveau_score)
        distances.append(evaluation(instance, dist_mat, nouveau_parcours))

plt.plot(scores, c="red")
plt.twinx()
plt.plot(distances, c="black", ls="--")

print(min(distances))

# %%
