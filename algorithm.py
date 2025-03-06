import copy
from time import time
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from utilitaires import *


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


def hybridation_ox2(parcours1, parcours2):
    """
    Hybride deux parcours en un seul parcours résultant.

    Référence : "Genetic Algorithms for the Travelling Salesman Problem:
    A Review of Representations and Operators"
    """

    n = len(parcours1)
    a, b = np.random.choice(range(1, n), 2, replace=False)
    a, b = min(a, b), max(a, b)

    parcours_start = parcours1[:a]
    parcours_middle = parcours1[a:b]
    parcours_end = parcours1[b:]
    parcours_middle_new = []

    for ville in parcours2:
        if ville in parcours_middle:
            parcours_middle_new.append(ville)

    parcours_middle_new = np.array(parcours_middle_new)
    parcours_new = np.concatenate((parcours_start, parcours_middle_new, parcours_end))
    assert len(parcours_new) == n

    return parcours_new


def enfanter(agent1, agent2):
    """
    Utilise l'hybridation de parcours pour enfanter un agent à partir de deux parents.
    """
    parcours1 = agent1.parcours
    parcours2 = agent2.parcours

    parcours = hybridation_ox2(parcours1, parcours2)
    enfant = agent1.copier()
    enfant.parcours = parcours
    return enfant


def selectionner_tranche(agents, n_agents, ratio_meilleurs_scores, ratio_meilleurs_penalites):
    """
    Sélectionne parmis la list `agents` une tranche avec `n_agents`, avec le ratio des meilleurs
    scores et des meilleures pénalités données ; et sélectionne le reste des agents au hasard
    avec une probabilité de poids 1/score**2.
    """
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
    def __init__(
        self,
        instance,
        dist_mat,
        parcours=None,
        p_mutation_0=0,
        p_mutation_f=0.8,
        iter_mutation_max=2000,
        p_mutation_echange=0.5,
    ):
        self.instance = instance
        self.dist_mat = dist_mat
        self.p_mutation_0 = p_mutation_0
        self.p_mutation_f = p_mutation_f
        self.p_mutation_echange = p_mutation_echange
        self.iter_mutation_max = iter_mutation_max

        self.score = 0
        self.distance = 0
        self.p_mutation = self.p_mutation_0
        self.iteration = 0

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
        self.p_mutation = (self.p_mutation_f - self.p_mutation_0) * min(
            (self.iteration / self.iter_mutation_max), 1
        ) + self.p_mutation_0
        return self

    def reevaluer(self):
        self.distance, self.score, self.n_penalites = evaluation(
            self.instance, self.dist_mat, self.parcours, lambda x: self.act_erreur(x)
        )

        return self

    def muter(self):
        self.iterer()
        i = 0
        while np.random.rand() < self.p_mutation and i < 10:
            if np.random.rand() < self.p_mutation_echange:
                self.parcours = mutation_echange(self.parcours)
            else:
                self.parcours = mutation_insertion(self.parcours)
            i += 1
        self.reevaluer()
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

        fig.tight_layout()

    def copier(self):
        return copy.copy(self)


class Algorithme:
    def __init__(
        self,
        instance,
        dist_mat,
        N_agents,
        p_mutation_0=0,
        p_mutation_f=0.8,
        iter_mutation_max=1,
        p_mutation_echange=0.5,
        ratio_parents=0.3,
        ratio_meilleurs_scores_parents=0.2,
        ratio_meilleurs_penalites_parents=0.1,
        ratio_meilleurs_scores_enfants=0.2,
        ratio_meilleurs_penalites_enfants=0.1,
        utiliser_hybridation=True,
    ):
        self.instance = instance
        self.dist_mat = dist_mat
        self.start_time = time()
        self.total_time = 0

        self.utiliser_hybridation = utiliser_hybridation
        self.N_agents = N_agents

        self.ratio_parents = ratio_parents
        self.ratio_meilleurs_scores_parents = ratio_meilleurs_scores_parents
        self.ratio_meilleurs_penalites_parents = ratio_meilleurs_penalites_parents

        self.ratio_meilleurs_scores_enfants = ratio_meilleurs_scores_enfants
        self.ratio_meilleurs_penalites_enfants = ratio_meilleurs_penalites_enfants

        self.n_parents = int(self.N_agents * self.ratio_parents)
        self.n_enfants = self.N_agents - self.n_parents

        self.agents = [
            Agent(
                instance,
                dist_mat,
                p_mutation_0=p_mutation_0,
                p_mutation_f=p_mutation_f,
                iter_mutation_max=iter_mutation_max,
                p_mutation_echange=p_mutation_echange,
            )
            for i in range(N_agents)
        ]

        self.iterations = [[agent.iteration for agent in self.agents]]
        self.scores = [[agent.score for agent in self.agents]]
        self.distances = [[agent.distance for agent in self.agents]]
        self.n_penalites = [[agent.n_penalites for agent in self.agents]]

    def iterer(self):
        parents_selectionnes = selectionner_tranche(
            [agent.copier().iterer().reevaluer() for agent in self.agents],
            self.n_parents,
            self.ratio_meilleurs_scores_parents,
            self.ratio_meilleurs_penalites_parents,
        )

        if self.utiliser_hybridation:
            piscine_enfants = []
            parents_possibles = [agent.copier() for agent in self.agents]
            for _ in range(self.N_agents):
                parent1, parent2 = np.random.choice(parents_possibles, size=2, replace=False)
                piscine_enfants.append(enfanter(parent1, parent2).muter())
        else:
            piscine_enfants = [agent.copier().muter() for agent in self.agents]

        enfants_selectionnes = selectionner_tranche(
            piscine_enfants,
            self.n_enfants,
            self.ratio_meilleurs_scores_enfants,
            self.ratio_meilleurs_penalites_enfants,
        )

        self.agents = parents_selectionnes + enfants_selectionnes
        assert len(self.agents) == self.N_agents

        self.iterations.append([agent.iteration for agent in self.agents])
        self.scores.append([agent.score for agent in self.agents])
        self.distances.append([agent.distance for agent in self.agents])
        self.n_penalites.append([agent.n_penalites for agent in self.agents])

    def lancer_simulation(self, N_iterations, afficher_progression=True):
        self.start_time = time()

        if afficher_progression:
            iterations = tqdm(range(1, N_iterations + 1))
        else:
            iterations = range(1, N_iterations + 1)
        for _ in iterations:
            self.iterer()

        self.total_time += time() - self.start_time
        self.start_time = (
            time() - self.start_time
        )  # pas une bonne pratique mais flemme d'ajouter un attribut

        if afficher_progression:
            self.afficher_simulation()

    def afficher_simulation(self):
        fig, (ax1, ax2) = plt.subplots(2, figsize=(7, 8), sharex=True)
        ax2t = ax2.twinx()

        ax1.scatter(
            self.iterations,
            self.scores,
            c="black",
            marker=".",
            alpha=0.05,
        )
        ax1.set_ylabel("Scores")
        ax1.set_yscale(("log"))

        ax2.plot([min(score) for score in self.scores], c="red")
        ax2.set_ylabel("Meilleur score")
        ax2.set_xlabel("Itération")

        ax2t.plot([min(distance) for distance in self.distances], c="black", ls="--")
        ax2t.set_ylabel("Meilleure distance")

        fig.tight_layout()
        plt.show()
