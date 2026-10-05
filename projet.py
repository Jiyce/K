"""
PLUS COURT CHEMIN ENTRE VILLES DU CAMEROUN
Application web Flask utilisant l'algorithme de Dijkstra.

Lancement local :
    pip install -r requirements.txt
    python projet.py

Puis ouvrir :
    http://127.0.0.1:5000

Pour le déploiement en ligne (Render, Railway, etc.) :
    gunicorn projet:app
"""

import heapq
from pathlib import Path

from flask import Flask, render_template, request
import matplotlib

# Backend sans interface graphique : indispensable sur un serveur en ligne.
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx


app = Flask(__name__)

# -------------------------------------------------------------------
# 1. DONNÉES DU RÉSEAU ROUTIER
# -------------------------------------------------------------------

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
    """Construit le graphe d'adjacence à partir des routes."""
    graphe = {}

    for a, b, distance in ROUTES:
        graphe.setdefault(a, {})[b] = distance
        graphe.setdefault(b, {})[a] = distance

    return graphe


def dijkstra(graphe, depart, arrivee):
    """
    Calcule le plus court chemin entre depart et arrivee.

    Retourne :
        (distance_totale, chemin)
    ou :
        (None, None) si aucun chemin n'existe.
    """
    if depart not in graphe:
        raise ValueError(f"Ville de départ inconnue : {depart}")

    if arrivee not in graphe:
        raise ValueError(f"Ville d'arrivée inconnue : {arrivee}")

    distances = {sommet: float("inf") for sommet in graphe}
    distances[depart] = 0

    predecesseurs = {sommet: None for sommet in graphe}
    visites = set()

    file_priorite = [(0, depart)]

    while file_priorite:
        distance_actuelle, sommet_actuel = heapq.heappop(file_priorite)

        if sommet_actuel in visites:
            continue

        visites.add(sommet_actuel)

        if sommet_actuel == arrivee:
            break

        for voisin, poids in graphe[sommet_actuel].items():
            if voisin in visites:
                continue

            nouvelle_distance = distance_actuelle + poids

            if nouvelle_distance < distances[voisin]:
                distances[voisin] = nouvelle_distance
                predecesseurs[voisin] = sommet_actuel
                heapq.heappush(
                    file_priorite,
                    (nouvelle_distance, voisin)
                )

    if distances[arrivee] == float("inf"):
        return None, None

    chemin = []
    sommet = arrivee

    while sommet is not None:
        chemin.append(sommet)
        sommet = predecesseurs[sommet]

    chemin.reverse()

    return distances[arrivee], chemin


def visualiser(graphe, chemin, distance_totale, depart, arrivee):
    """
    Génère une image du réseau et met en évidence le plus court chemin.

    L'image est enregistrée dans le dossier static/ afin d'être
    accessible par l'interface web.
    """
    G = nx.Graph()

    for a, b, distance in ROUTES:
        G.add_edge(a, b, weight=distance)

    pos = {
        ville: POSITIONS[ville]
        for ville in G.nodes
        if ville in POSITIONS
    }

    plt.figure(figsize=(11, 9))

    # Réseau complet
    nx.draw_networkx_edges(
        G,
        pos,
        edge_color="lightgray",
        width=1.5
    )

    nx.draw_networkx_nodes(
        G,
        pos,
        node_color="lightblue",
        node_size=600,
        edgecolors="black"
    )

    nx.draw_networkx_labels(
        G,
        pos,
        font_size=8
    )

    # Distances sur les routes
    labels_aretes = nx.get_edge_attributes(G, "weight")

    nx.draw_networkx_edge_labels(
        G,
        pos,
        edge_labels=labels_aretes,
        font_size=7
    )

    # Chemin trouvé
    if chemin and len(chemin) > 1:
        aretes_chemin = list(zip(chemin[:-1], chemin[1:]))

        nx.draw_networkx_edges(
            G,
            pos,
            edgelist=aretes_chemin,
            edge_color="red",
            width=3
        )

        nx.draw_networkx_nodes(
            G,
            pos,
            nodelist=chemin,
            node_color="orange",
            node_size=650,
            edgecolors="black"
        )

        nx.draw_networkx_nodes(
            G,
            pos,
            nodelist=[depart, arrivee],
            node_color="limegreen",
            node_size=750,
            edgecolors="black"
        )

        titre = (
            f"Plus court chemin : {depart} → {arrivee}\n"
            f"Trajet : {' → '.join(chemin)}\n"
            f"Distance totale : {distance_totale} km"
        )
    else:
        titre = f"Aucun chemin trouvé entre {depart} et {arrivee}"

    plt.title(titre, fontsize=11)
    plt.axis("off")
    plt.tight_layout()

    static_dir = Path(app.root_path) / "static"
    static_dir.mkdir(exist_ok=True)

    image_path = static_dir / "resultat_plus_court_chemin.png"

    plt.savefig(
        image_path,
        dpi=150,
        bbox_inches="tight"
    )

    plt.close()

    return "resultat_plus_court_chemin.png"


GRAPHE = construire_graphe()
VILLES = sorted(GRAPHE.keys())


# -------------------------------------------------------------------
# 2. ROUTES FLASK
# -------------------------------------------------------------------

@app.route("/", methods=["GET"])
def accueil():
    """Affiche l'interface principale."""
    return render_template(
        "index.html",
        villes=VILLES,
        depart="",
        arrivee="",
        distance=None,
        chemin=None,
        erreur=None,
        image=None
    )


@app.route("/calculer", methods=["POST"])
def calculer():
    """Récupère les villes du formulaire et lance Dijkstra."""
    depart = request.form.get("depart", "").strip()
    arrivee = request.form.get("arrivee", "").strip()

    if not depart or not arrivee:
        return render_template(
            "index.html",
            villes=VILLES,
            depart=depart,
            arrivee=arrivee,
            distance=None,
            chemin=None,
            erreur="Veuillez sélectionner une ville de départ et une ville d'arrivée.",
            image=None
        )

    if depart == arrivee:
        return render_template(
            "index.html",
            villes=VILLES,
            depart=depart,
            arrivee=arrivee,
            distance=0,
            chemin=[depart],
            erreur=None,
            image=None
        )

    try:
        distance_totale, chemin = dijkstra(
            GRAPHE,
            depart,
            arrivee
        )

        if chemin is None:
            return render_template(
                "index.html",
                villes=VILLES,
                depart=depart,
                arrivee=arrivee,
                distance=None,
                chemin=None,
                erreur=f"Aucun chemin trouvé entre {depart} et {arrivee}.",
                image=None
            )

        image = visualiser(
            GRAPHE,
            chemin,
            distance_totale,
            depart,
            arrivee
        )

        return render_template(
            "index.html",
            villes=VILLES,
            depart=depart,
            arrivee=arrivee,
            distance=distance_totale,
            chemin=chemin,
            erreur=None,
            image=image
        )

    except ValueError as erreur:
        return render_template(
            "index.html",
            villes=VILLES,
            depart=depart,
            arrivee=arrivee,
            distance=None,
            chemin=None,
            erreur=str(erreur),
            image=None
        )


if __name__ == "__main__":
    # Le port fourni par la plateforme de déploiement est utilisé
    # automatiquement lorsqu'il existe.
    import os

    port = int(os.environ.get("PORT", 5000))

    app.run(
        host="0.0.0.0",
        port=port,
        debug=False
    )
