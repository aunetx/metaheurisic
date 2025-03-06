# %%

from utilitaires import *
from algorithm import *


# %%

instance_name = "inst2"
instance = charger_instance(f"data/{instance_name}")
dist_mat = compute_dist_mat(instance)
save_name = f"{instance_name}_avec_hybridation_70A"
save_file_scores = f"best_scores/{save_name}_scores.txt"
save_file_distances = f"best_scores/{save_name}_distances.txt"

N_agents = 70
N_batches = 10
N_iterations_par_batch = 500
N_runs = 10

for i in range(N_runs):
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
        algo.lancer_simulation(N_iterations_par_batch)
        print(f"Run {i+1} / {N_runs}")
        print(f"Batch {batch + 1} / {N_batches}")
        print(f"Batch time : {algo.start_time:.2f}")
        print(f"Total simulation time : {algo.total_time:.2f} s")
        print(f"current mutation rate : {algo.agents[0].p_mutation}")

    sauvegarder_resultats(save_file_scores, algo.scores)
    sauvegarder_resultats(save_file_distances, algo.distances)


# %%

meilleur_agent = sorted(algo.agents, key=lambda agent: agent.score)[0]
print(f"meilleur score : {int(meilleur_agent.score)}")
print(meilleur_agent.parcours)
meilleur_agent.afficher_parcours()
