# %%

import multiprocessing
from functools import partial
from tqdm import tqdm

from utilitaires import *
from algorithm import *


# %%

instance_name = "inst1"
instance = charger_instance(f"data/{instance_name}")
dist_mat = compute_dist_mat(instance)
SCORES_FILE_LOCK = multiprocessing.Lock()
DISTANCES_FILE_LOCK = multiprocessing.Lock()

N_agents = 50
N_batches = 10
N_iterations_par_batch = 50
N_runs = 50

save_name = f"{instance_name}_avec_hybridation_{N_agents}A"
save_file_scores = f"best_scores/{save_name}_scores.txt"
save_file_distances = f"best_scores/{save_name}_distances.txt"


def run(run_number, verbose=True):
    if verbose:
        print("New simulation")
    algo = Algorithme(
        instance,
        dist_mat,
        N_agents,
        p_mutation_0=0.2,
        p_mutation_f=0.6,
        iter_mutation_max=2000,
        p_mutation_echange=0.5,
        ratio_parents=0.3,
        ratio_meilleurs_scores_parents=0.2,
        ratio_meilleurs_penalites_parents=0.2,
        ratio_meilleurs_scores_enfants=0.2,
        ratio_meilleurs_penalites_enfants=0.2,
        utiliser_hybridation=True,
    )

    for batch in range(N_batches):
        algo.lancer_simulation(N_iterations_par_batch, afficher_progression=verbose)
        if verbose:
            print(f"Run {run_number+1} / {N_runs}")
            print(f"Batch {batch + 1} / {N_batches}")
            print(f"Batch time : {algo.start_time:.2f}")
            print(f"Total simulation time : {algo.total_time:.2f} s")
            print(f"current mutation rate : {algo.agents[0].p_mutation}")

    with SCORES_FILE_LOCK:
        sauvegarder_resultats(save_file_scores, algo.scores)
    with DISTANCES_FILE_LOCK:
        sauvegarder_resultats(save_file_distances, algo.distances)

    return algo


use_multiprocess = True

runs_list = range(N_runs)
if use_multiprocess:
    with multiprocessing.Pool(processes=5, maxtasksperchild=1) as pool:
        algo_list = list(
            tqdm(
                pool.imap_unordered(partial(run, verbose=False), runs_list),
                total=len(runs_list),
            )
        )
else:
    algo_list = [run(i) for i in runs_list]


# %%

meilleur_agent = sorted(algo_list[3].agents, key=lambda agent: agent.score)[0]
print(f"meilleur score : {int(meilleur_agent.score)}")
print(meilleur_agent.parcours)
meilleur_agent.afficher_parcours()
