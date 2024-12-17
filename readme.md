# Général

## Solution possible

Liste ordonnée des villes par lesquelles passe le voyageur.

Solution exacte ssi chaque ville est présente de manière unique, et le voyageur est dans la bonne fenêtre de temps pour passer dans chaque ville.

Solution non-exacte si le voyageur arrive dans au moins une ville vK en dehors de la fenêtre de temps, avec une différence absolue e(vK).

## Fonction de voisinage

Prend une solution possible, et retourne P voisins.

Questions :
- voisins nécessairement solutions exactes ?
- combien de voisins ?

## Fonction d'évaluation

Prend une solution (exacte ou approché) et retourne un score. But de minimiser ce score.

Fonction d'évaluation logique :

f: [v1, v2, ..., vN] -> dist(v1, v2) + ... + dist(vN-1, vN)

Fonction d'évaluation pour solutions non-exactes :

f: [v1, v2, ..., vN] -> dist(v1, v2) + ... + dist(vN-1, vN) + e(v1) + ... + e(vN)

## Minimisation

Deux phases :
- minimisation dans les solutions approchées
- minimisation dans les solutions exactes

# Algorithme génétique

## Fonction de croisement

Bipartition à une frontière entre deux nœuds, et on échange les parties droites et gauches pour faire les deux enfants.

## Mutation

Deux mutations possibles :
- on échange deux villes
- on change de place une ville

## Fonction d'évaluation

Fonction d'évaluation pour solutions non-exactes :

f: [v1, v2, ..., vN] -> dist(v1, v2) + ... + dist(vN-1, vN) + g_i(e(v1)) + ... + g_i(e(vN))

Où e(vK) est l'erreur en temps commise lors de l'arrivée du mec, et g_i est une fonction croissante des générations, qui tende vers l'infini pour i grand afin de n'obtenir que des solutions exactes à la fin.

## Variation dans la sélection

Plusieurs méthodes, pour k parents :
- soit on ne sélectionne que les k enfants
- soit on sélectionne les k meilleurs entre les k parents et les k enfants
- soit on sélectionne aléatoirement avec une meilleur probabilité pour les meilleurs
- soit on enlève les p moins bon parents, les k-p parents font k-p enfants, et on garde les k-p enfants avec les p meilleurs parents

## Choix de la solution optimale

Pendant toute la simulation, on stocke pour chaque agent son score et sa topologie lorsque l'agent respecte les contraintes temporelles s'il est meilleur que celui stocké.