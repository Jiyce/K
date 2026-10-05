"""
PLUS COURT CHEMIN ENTRE VILLES DU CAMEROUN
Application Streamlit utilisant l'algorithme de Dijkstra.

Lancement local :
    pip install -r requirements.txt
    streamlit run projet.py
"""

import heapq
import io

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import streamlit as st

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
    "Yaoundé":        (11.52,  3.87),
    "Douala":         ( 9.70,  4.05),
    "Ebolowa":        (11.15,  2.92),
    "Bertoua":        (13.68,  4.58),
    "Bafoussam":      (10.42,  5.48),
    "Ngaoundéré":     (13.58,  7.32),
    "Buea":           ( 9.24,  4.16),
    "Limbe":          ( 9.21,  4.02),
    "Edéa":           (10.13,  3.80),
    "Nkongsamba":     ( 9.93,  4.95),
    "Kribi":          ( 9.91,  2.94),
    "Bamenda":        (10.17,  5.96),
    "Dschang":        (10.05,  5.45),
    "Foumban":        (10.90,  5.73),
    "Garoua":         (13.40,  9.30),
    "Maroua":         (14.32, 10.59),
    "Garoua-Boulaï":  (14.55,  5.90),
}

# -------------------------------------------------------------------
# 2. LOGIQUE MÉTIER
# -------------------------------------------------------------------

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
                heapq.heappush(file_priorite, (nouvelle_distance, voisin))

    if distances[arrivee] == float("inf"):
        return None, None

    chemin = []
    sommet = arrivee
    while sommet is not None:
        chemin.append(sommet)
        sommet = predecesseurs[sommet]
    chemin.reverse()

    return distances[arrivee], chemin


def visualiser(chemin, distance_totale, depart, arrivee):
    """
    Génère la figure du réseau routier et retourne un objet BytesIO
    prêt à être affiché par Streamlit.
    """
    G = nx.Graph()
    for a, b, distance in ROUTES:
        G.add_edge(a, b, weight=distance)

    pos = {ville: POSITIONS[ville] for ville in G.nodes if ville in POSITIONS}

    fig, ax = plt.subplots(figsize=(12, 9))

    # Réseau complet
    nx.draw_networkx_edges(G, pos, edge_color="lightgray", width=1.5, ax=ax)
    nx.draw_networkx_nodes(G, pos, node_color="lightblue", node_size=600,
                           edgecolors="black", ax=ax)
    nx.draw_networkx_labels(G, pos, font_size=8, ax=ax)
    nx.draw_networkx_edge_labels(G, pos,
                                 edge_labels=nx.get_edge_attributes(G, "weight"),
                                 font_size=7, ax=ax)

    # Chemin trouvé
    if chemin and len(chemin) > 1:
        aretes_chemin = list(zip(chemin[:-1], chemin[1:]))
        nx.draw_networkx_edges(G, pos, edgelist=aretes_chemin,
                               edge_color="red", width=3, ax=ax)
        nx.draw_networkx_nodes(G, pos, nodelist=chemin,
                               node_color="orange", node_size=650,
                               edgecolors="black", ax=ax)
        nx.draw_networkx_nodes(G, pos, nodelist=[depart, arrivee],
                               node_color="limegreen", node_size=750,
                               edgecolors="black", ax=ax)
        titre = (
            f"Plus court chemin : {depart} → {arrivee}\n"
            f"Trajet : {' → '.join(chemin)}\n"
            f"Distance totale : {distance_totale} km"
        )
    else:
        titre = f"Aucun chemin trouvé entre {depart} et {arrivee}"

    ax.set_title(titre, fontsize=11)
    ax.axis("off")
    plt.tight_layout()

    buf = io.BytesIO()
    plt.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return buf


# -------------------------------------------------------------------
# 3. INTERFACE STREAMLIT
# -------------------------------------------------------------------

GRAPHE = construire_graphe()
VILLES = sorted(GRAPHE.keys())

# Configuration de la page
st.set_page_config(
    page_title="Plus court chemin – Cameroun",
    page_icon="🗺️",
    layout="centered",
)

# En-tête
st.markdown(
    """
    <div style="background:#176b3a;padding:28px 20px;border-radius:12px;
                text-align:center;color:white;margin-bottom:24px">
        <h1 style="margin:0;font-size:26px">
            🗺️ Plus court chemin entre les villes du Cameroun
        </h1>
        <p style="margin:8px 0 0 0;font-size:15px">
            Recherche d'itinéraire avec l'algorithme de Dijkstra
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

st.subheader("Calculer un itinéraire")
st.write(
    "Choisissez une ville de départ et une ville d'arrivée. "
    "Le programme calculera automatiquement le plus court chemin."
)

# Formulaire
col1, col2 = st.columns(2)

with col1:
    depart = st.selectbox(
        "🚩 Ville de départ",
        options=["-- Choisir une ville --"] + VILLES,
        index=0,
    )

with col2:
    arrivee = st.selectbox(
        "🏁 Ville d'arrivée",
        options=["-- Choisir une ville --"] + VILLES,
        index=0,
    )

calculer = st.button("Calculer le plus court chemin", use_container_width=True)

# Calcul et affichage du résultat
if calculer:
    if depart == "-- Choisir une ville --" or arrivee == "-- Choisir une ville --":
        st.error("Veuillez sélectionner une ville de départ et une ville d'arrivée.")

    elif depart == arrivee:
        st.info(f"Vous êtes déjà à **{depart}** — distance : **0 km**.")

    else:
        distance_totale, chemin = dijkstra(GRAPHE, depart, arrivee)

        if chemin is None:
            st.error(f"Aucun chemin trouvé entre **{depart}** et **{arrivee}**.")
        else:
            # Résultat textuel
            st.success("Itinéraire calculé avec succès !")

            st.markdown(
                f"""
                | | |
                |---|---|
                | **Ville de départ** | {depart} |
                | **Ville d'arrivée** | {arrivee} |
                | **Trajet optimal** | {" → ".join(chemin)} |
                | **Distance totale** | **{distance_totale} km** |
                """
            )

            # Carte
            st.subheader("Visualisation du réseau routier")
            buf = visualiser(chemin, distance_totale, depart, arrivee)
            st.image(buf, use_container_width=True)

# Pied de page
st.markdown(
    "<p style='text-align:center;color:#999;font-size:13px;margin-top:40px'>"
    "Projet pédagogique — Algorithme de Dijkstra</p>",
    unsafe_allow_html=True,
)
