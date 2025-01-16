# %%
#imports

import numpy as np
import matplotlib.pyplot as plt
import math
import copy
import igraph
from time import time

from utilitaires import *
import scoreEtudiant

# %% definitions 

# definitions
def mutation_echange(parcours):
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

def hybridation_ox2(parcours1, parcours2):
    
    """Ref ; Genetic Algorithms for the Travelling Salesman Problem: A
Review of Representations and Operators"""

    n = len(parcours1)
    a,b  = np.random.choice(range(1,n),2,replace=False)
    a,b = min(a,b),max(a,b)
        
    parcours_start = parcours1[:a]
    parcours_middle = parcours1[a:b]
    parcours_end = parcours1[b:]
    parcours_middle_new = []
    
    for ville in parcours2:
        if ville in parcours_middle:
            parcours_middle_new.append(ville)
    
    # print("here")     
    # print(len(parcours_middle_new))
    # print(len(parcours_middle))
    # print()
    # print(type(parcours_start))
    # print(type(parcours_middle_new))
    # print(type(parcours_end))
    parcours_middle_new = np.array(parcours_middle_new)
    parcours_new = np.concatenate((parcours_start, parcours_middle_new, parcours_end))
    assert len(parcours_new) == n
    
    return parcours_new

def enfanter (agent1, agent2):
    parcours1 = agent1.parcours
    parcours2 = agent2.parcours
    
    parcours = hybridation_ox2(parcours1, parcours2)
    enfant = agent1.copier()
    enfant.parcours = parcours
    enfant.reevaluer()
    return enfant

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
    p_mutation_echange = 0.1
    p_mutation_insertion = 0.1

    def __init__(self, instance, dist_mat, parcours=None, p_mutation_echange=0.1, pm_mutation_insertion=0.1):
        self.instance = instance
        self.dist_mat = dist_mat
        self.p_mutation_echange = p_mutation_echange
        self.p_mutation_insertion = pm_mutation_insertion

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
        if np.random.rand() < self.p_mutation_echange:
            self.parcours = mutation_echange(self.parcours)
        if np.random.rand() < self.p_mutation_insertion:
            self.parcours = mutation_insertion(self.parcours)
        self.iterer()
        self.recalculer_distance()
        self.recalculer_score()
        return self

    def copier(self):
        return copy.deepcopy(self)


# %%
# Run

total_time = 0
start_time = time()
instance = charger_instance("data/inst2")
dist_mat = compute_dist_mat(instance)

N_agents = 100

ratio_parents = 0.3
ratio_meilleurs_scores_parents = 0.2
ratio_meilleurs_penalites_parents = 0.1


n_parents = int(N_agents * ratio_parents)
n_enfants = N_agents - n_parents

N_iteration = 1500
continuer = False

fig, (ax1, ax2, ax3) = plt.subplots(3, figsize=(8, 15), sharex=True)

if not continuer:
    
    total_time = 0
    agents = [Agent(instance, dist_mat, p_mutation_echange=0.35, pm_mutation_insertion=0.35) for i in range(N_agents)]

    scores = [min(agent.score for agent in agents)]
    distances = [min(agent.distance for agent in agents)]
    n_penalites = [min(agent.n_penalites for agent in agents)]

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

for i in range(1, N_iteration):
    print(f"{i} / {N_iteration}", end="\r")

    parents_selectionnes = selectionner_tranche(
        agents,
        n_parents,
        ratio_meilleurs_scores_parents,
        ratio_meilleurs_penalites_parents,
    )
    
    enfants = []
    for i in range(n_enfants):
        parent1, parent2 = np.random.choice(parents_selectionnes, size=2, replace=False)
        enfants.append(enfanter(parent1, parent2))

    agents = parents_selectionnes + enfants
    agents = [agent.muter() for agent in agents]
    
    
    assert len(agents) == N_agents

    scores.append(min(agent.score for agent in agents))
    distances.append(min(agent.distance for agent in agents))
    n_penalites.append(min(agent.n_penalites for agent in agents))

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


run_time = time() - start_time
total_time += run_time

ax1.set_ylim([0, None])
ax1.set_ylabel("N pénalités")
ax2.set_ylabel("Scores")
ax2.set_yscale(("log"))

ax3.plot(scores, c="red")
ax3.set_yscale("log")
ax3.set_ylabel("Meilleur score")
ax3.set_xlabel("Itération")
ax3t = ax3.twinx()
ax3t.plot(distances, c="black", ls="--")
ax3t.set_ylabel("Meilleure distance")

fig.tight_layout()


# %%
#End
meilleur_agent = sorted(agents, key=lambda agent: agent.score)[0]
print(f"Temps d'exécution total du run : {run_time:.2f} s")
print("Meilleur score:", meilleur_agent.score)
print("parcours:", meilleur_agent.parcours)
meilleur_agent.afficher_parcours()



# %%
