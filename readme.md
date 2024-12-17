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

## Fonction d'évaluation

Prend une solution (exacte ou approché) et retourne un score.
But de minimiser ce score.

Fonction d'évaluation logique :
f: solution -> score
   [v1, v2, ..., vN] -> dist(v1, v2) + ... + dist(vN-1, vN)

Fonction d'évaluation pour solutions non-exactes :
f: solution -> score
   [v1, v2, ..., vN] -> dist(v1, v2) + ... + dist(vN-1, vN) + e(v1) + ... + e(vN)

## Minimisation

Deux phases :
- minimisation dans les solutions approchées
- minimisation dans les solutions exactes

# Algorithme génétique

## Fonction de croisement

Bipartition à une frontière entre deux nœuds, et on échange les parties droites et gauches pour faire les deux enfants

## 