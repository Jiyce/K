
"""
=====================================================================
 PLUS COURT CHEMIN ENTRE VILLES DU CAMEROUN
=====================================================================
Ce programme modélise un réseau routier simplifié du Cameroun sous
forme de graphe pondéré (sommets = villes, arêtes = routes, poids =
distance approximative en km), calcule le plus court chemin entre
deux villes avec l'algorithme de Dijkstra, puis visualise le réseau
et le trajet trouvé.

Dépendances à installer avant exécution :
    pip install networkx matplotlib

Exécution :
    python plus_court_chemin_cameroun.py
=====================================================================
"""

import heapq
import sys

import matplotlib.pyplot as plt
import networkx as nx

# ---------------------------------------------------------------
# 1. DONNÉES DU RÉSEAU ROUTIER (approximatives, à but pédagogique)
# ---------------------------------------------------------------
# Chaque tuple : (ville_A, ville_B, distance_en_km)

ROUTES = [
    ("Yaoundé", "Douala", 250),
    ("Yaoundé", "Ebolowa", 158),
    ("Yaoundé", "Bertoua", 345),
    ("Yaoundé", "Bafoussam", 290),
    ("Yaoundé", "Ngaoundéré", 625),
    ("Douala", "Bafoussam", 220),
    ("Douala", "Buea", 70),
    ("Douala", "Limbe", 70),
    ("Douala", "Edéa", 65),
    ("Douala", "Nkongsamba", 140),
    ("Edéa", "Kribi", 80),
    ("Bafoussam", "Bamenda", 70),
    ("Bafoussam", "Dschang", 35),
    ("Bafoussam", "Foumban", 80),
    ("Bafoussam", "Nkongsamba", 90),
    ("Ngaoundéré", "Garoua", 265),
    ("Garoua", "Maroua", 200),
    ("Ngaoundéré", "Bertoua", 490),
    ("Bertoua", "Garoua-Boulaï", 150),
]

# Coordonnées approximatives (longitude, latitude) pour un affichage
# qui respecte à peu près la position géographique réelle des villes.
POSITIONS = {
    "Yaoundé": (11.52, 3.87),
    "Douala": (9.70, 4.05),
    "Ebolowa": (11.15, 2.92),
    "Bertoua": (13.68, 4.58),
    "Bafoussam": (10.42, 5.48),
    "Ngaoundéré": (13.58, 7.32),
    "Buea": (9.24, 4.16),
    "Limbe": (9.21, 4.02),
    "Edéa": (10.13, 3.80),
    "Nkongsamba": (9.93, 4.95),
    "Kribi": (9.91, 2.94),
    "Bamenda": (10.17, 5.96),
    "Dschang": (10.05, 5.45),
    "Foumban": (10.90, 5.73),
    "Garoua": (13.40, 9.30),
    "Maroua": (14.32, 10.59),
    "Garoua-Boulaï": (14.55, 5.90),
}


def construire_graphe():
    """Construit le graphe (dictionnaire d'adjacence) à partir de ROUTES."""
    graphe = {}
    for a, b, d in ROUTES:
        graphe.setdefault(a, {})[b] = d
        graphe.setdefault(b, {})[a] = d  # routes bidirectionnelles
    return graphe


# ---------------------------------------------------------------
# 2. ALGORITHME DE DIJKSTRA
# ---------------------------------------------------------------
def dijkstra(graphe, depart, arrivee):
    """
    Calcule le plus court chemin entre `depart` et `arrivee`.

    Retourne (distance_totale, liste_du_chemin) ou (None, None)
    si aucun chemin n'existe.
    """
    if depart not in graphe:
        raise ValueError(f"Ville de départ inconnue : {depart}")
    if arrivee not in graphe:
        raise ValueError(f"Ville d'arrivée inconnue : {arrivee}")

    distances = {sommet: float("inf") for sommet in graphe}
    distances[depart] = 0
    predecesseurs = {sommet: None for sommet in graphe}
    visites = set()

    file_priorite = [(0, depart)]  # (distance, sommet)

    while file_priorite:
        dist_actuelle, sommet_actuel = heapq.heappop(file_priorite)

        if sommet_actuel in visites:
            continue
        visites.add(sommet_actuel)

        if sommet_actuel == arrivee:
            break

        for voisin, poids in graphe[sommet_actuel].items():
            if voisin in visites:
                continue
            nouvelle_distance = dist_actuelle + poids
            if nouvelle_distance < distances[voisin]:
                distances[voisin] = nouvelle_distance
                predecesseurs[voisin] = sommet_actuel
                heapq.heappush(file_priorite, (nouvelle_distance, voisin))

    if distances[arrivee] == float("inf"):
        return None, None

    # Reconstruction du chemin à partir des prédécesseurs
    chemin = []
    sommet = arrivee
    while sommet is not None:
        chemin.append(sommet)
        sommet = predecesseurs[sommet]
    chemin.reverse()

    return distances[arrivee], chemin


# ---------------------------------------------------------------
# 3. VISUALISATION
# ---------------------------------------------------------------
def visualiser(graphe, chemin, distance_totale, depart, arrivee):
    """Affiche le réseau routier complet et met en évidence le trajet trouvé."""
    G = nx.Graph()
    for a, b, d in ROUTES:
        G.add_edge(a, b, weight=d)

    pos = {ville: POSITIONS[ville] for ville in G.nodes if ville in POSITIONS}

    plt.figure(figsize=(11, 9))

    # Réseau complet en gris clair
    nx.draw_networkx_edges(G, pos, edge_color="lightgray", width=1.5)
    nx.draw_networkx_nodes(G, pos, node_color="lightblue",
                            node_size=600, edgecolors="black")
    nx.draw_networkx_labels(G, pos, font_size=8)

    # Étiquettes des distances sur chaque route
    labels_aretes = nx.get_edge_attributes(G, "weight")
    nx.draw_networkx_edge_labels(G, pos, edge_labels=labels_aretes, font_size=7)

    # Mise en évidence du chemin trouvé (en rouge)
    if chemin and len(chemin) > 1:
        aretes_chemin = list(zip(chemin[:-1], chemin[1:]))
        nx.draw_networkx_edges(G, pos, edgelist=aretes_chemin,
                                edge_color="red", width=3)
        nx.draw_networkx_nodes(G, pos, nodelist=chemin,
                                node_color="orange", node_size=650,
                                edgecolors="black")
        # Villes de départ et d'arrivée mises en valeur
        nx.draw_networkx_nodes(G, pos, nodelist=[depart, arrivee],
                                node_color="limegreen", node_size=750,
                                edgecolors="black")

        titre = (f"Plus court chemin : {depart} → {arrivee}\n"
                 f"Trajet : {' → '.join(chemin)}\n"
                 f"Distance totale : {distance_totale} km")
    else:
        titre = f"Aucun chemin trouvé entre {depart} et {arrivee}"

    plt.title(titre, fontsize=11)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig("resultat_plus_court_chemin.png", dpi=150)
    print("\nFigure enregistrée sous : resultat_plus_court_chemin.png")
    plt.show()


# ---------------------------------------------------------------
# 4. PROGRAMME PRINCIPAL
# ---------------------------------------------------------------
def afficher_villes_disponibles(graphe):
    print("\nVilles disponibles :")
    for ville in sorted(graphe.keys()):
        print(f"  - {ville}")


def main():
    graphe = construire_graphe()

    print("=" * 60)
    print(" PLUS COURT CHEMIN ENTRE VILLES DU CAMEROUN (Dijkstra)")
    print("=" * 60)
    afficher_villes_disponibles(graphe)

    # Saisie utilisateur (avec valeurs par défaut si entrée vide, pour test rapide)
    depart = input("\nVille de départ : ").strip() or "Yaoundé"
    arrivee = input("Ville d'arrivée : ").strip() or "Maroua"

    try:
        distance_totale, chemin = dijkstra(graphe, depart, arrivee)
    except ValueError as e:
        print(f"\nErreur : {e}")
        sys.exit(1)

    if chemin is None:
        print(f"\nAucun chemin trouvé entre {depart} et {arrivee}.")
        return

    print(f"\nTrajet trouvé : {' → '.join(chemin)}")
    print(f"Distance totale : {distance_totale} km")

    visualiser(graphe, chemin, distance_totale, depart, arrivee)


if __name__ == "__main__":
    main()