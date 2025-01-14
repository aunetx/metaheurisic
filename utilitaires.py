import numpy as np
import math


def charger_instance(inst_name):
    """
    Charge une instance depuis un fichier.
    """
    f = open(inst_name, "r")
    inst = {}

    # ignoring the beginning
    vals = list(filter(None, f.readline().split(" ")))
    while not vals[0].isdigit():
        vals = list(filter(None, f.readline().split(" ")))

    # reading the file
    inst[int(vals[0])] = {
        "x": float(vals[1]),
        "y": float(vals[2]),
        "wstart": float(vals[4]),
        "wend": float(vals[5]),
    }
    while len(vals) > 0 and vals[0].isdigit() and int(vals[0]) < 999:
        inst[int(vals[0])] = {
            "x": float(vals[1]),
            "y": float(vals[2]),
            "wstart": float(vals[4]),
            "wend": float(vals[5]),
        }
        vals = list(filter(None, f.readline().split(" ")))

    return inst


def charger_solution(sol_name):
    """
    Charge la solution et son score à partir d'un fichier.
    """
    with open(sol_name, "r") as f:
        sol_list = list(map(int, f.readline().split()))
        sol_val = f.readline()
        if sol_val != "":
            sol_val = int(sol_val)
        else:
            sol_val = None

    return sol_list, sol_val


def dist(instance, node1, node2, distances_flottantes):
    """
    La distance géométique entre deux villes. N'est pas la distance topologique !
    (ne prend pas en compte l'inégalité triangulaire).
    Si concours est True, alors les distances sont tronquées à l'entier inférieur,
    comme dans scoreEtudiant.py.
    """
    x1 = instance[node1]["x"]
    y1 = instance[node1]["y"]
    x2 = instance[node2]["x"]
    y2 = instance[node2]["y"]
    if distances_flottantes:
        return math.sqrt((x1 - x2) * (x1 - x2) + (y1 - y2) * (y1 - y2))
    else:
        return math.floor(math.sqrt((x1 - x2) * (x1 - x2) + (y1 - y2) * (y1 - y2)))


def compute_dist_mat(instance, distances_flottantes=False):
    """
    La matrice des distances topologiques entre deux villes, donnée en flottant.
    Si concours est True, alors les distances sont tronquées à l'entier inférieur,
    comme dans scoreEtudiant.py.
    """
    mat_dist = np.zeros((len(instance) + 1, len(instance) + 1))
    for i in instance:
        for j in instance:
            mat_dist[i, j] = dist(instance, i, j, distances_flottantes=distances_flottantes)

    for i in instance:
        for j in instance:
            for k in instance:
                if mat_dist[i, j] > mat_dist[i, k] + mat_dist[k, j]:
                    mat_dist[i, j] = mat_dist[i, k] + mat_dist[k, j]
    return mat_dist


def evaluation(instance, dist_mat, parcours, g=lambda x: 0):
    """
    Évalue un parcours donné. La fonction g de pénalité à zéro signifie
    que l'on ne prend pas en compte le temps.
    """
    distance = 0
    penalite = 0
    n_penalites = 0
    t = 0
    for i in range(len(parcours)):
        # on revient au départ à la fin
        j = i if i != len(parcours) - 1 else -1

        d_parcours = dist_mat[int(parcours[j]), int(parcours[j + 1])]
        distance += d_parcours
        t += d_parcours
        next_start = instance[parcours[j + 1]]["wstart"]
        end_window = instance[parcours[j + 1]]["wend"]
        if t < next_start:
            t = next_start
        if t > end_window:
            erreur = t - end_window
            penalite += g(erreur)
            n_penalites += 1

    return distance, distance + penalite, n_penalites
