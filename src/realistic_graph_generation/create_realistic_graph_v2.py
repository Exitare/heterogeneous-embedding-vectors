from argparse import ArgumentParser
import pandas as pd
from pathlib import Path
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity


def load_embeddings():
    brca_embeddings = pd.read_csv(Path("results", "embeddings", "cancer", "brca_embeddings.csv"))
    blca_embeddings = pd.read_csv(Path("results", "embeddings", "cancer", "blca_embeddings.csv"))
    laml_embeddings = pd.read_csv(Path("results", "embeddings", "cancer", "laml_embeddings.csv"))
    stad_embeddings = pd.read_csv(Path("results", "embeddings", "cancer", "stad_embeddings.csv"))
    thca_embeddings = pd.read_csv(Path("results", "embeddings", "cancer", "thca_embeddings.csv"))
    image_embeddings = pd.read_csv(Path("results", "embeddings", "image_embeddings.csv"))
    sentence_embeddings = pd.read_csv(Path("results", "embeddings", "sentence_embeddings.csv"))

    return {
        "brca": brca_embeddings,
        "blca": blca_embeddings,
        "laml": laml_embeddings,
        "stad": stad_embeddings,
        "thca": thca_embeddings,
        "image": image_embeddings,
        "text": sentence_embeddings
    }


import random


def generate_patient(patient_id, families, cancer_types):
    # Determine if the patient starts a new family or is added to an existing one
    if len(families) == 0 or np.random.rand() < 0.5:
        # Starting a new family
        has_cancer = np.random.rand() < 0.25
        cancer_type = random.choice(cancer_types) if has_cancer else None
        patient = {
            "id": patient_id,
            "has_cancer": has_cancer,
            "cancer_type": cancer_type,
            "relationships": [],
            "generation": 1
        }
        families.append([patient])
    else:
        # Joining an existing family
        family = random.choice(families)
        potential_relatives = [p for p in family if p["generation"] < 3]

        if not potential_relatives:
            # If no potential relatives, start a new family
            has_cancer = np.random.rand() < 0.25
            cancer_type = random.choice(cancer_types) if has_cancer else None
            patient = {
                "id": patient_id,
                "has_cancer": has_cancer,
                "cancer_type": cancer_type,
                "relationships": [],
                "generation": 1
            }
            families.append([patient])
        else:
            relative = random.choice(potential_relatives)
            has_cancer = np.random.rand() < 0.5 if relative["has_cancer"] else np.random.rand() < 0.25
            cancer_type = relative["cancer_type"] if relative["has_cancer"] and np.random.rand() < 0.5 else (
                random.choice(cancer_types) if has_cancer else None
            )
            patient = {
                "id": patient_id,
                "has_cancer": has_cancer,
                "cancer_type": cancer_type,
                "relationships": [{"relative": relative["id"], "type": "child"}],
                "generation": relative["generation"] + 1
            }
            relative["relationships"].append({"relative": patient_id, "type": "parent"})
            family.append(patient)

    return patient


def generate_patients(num_patients, cancer_types):
    families = []
    patients = []

    for i in range(num_patients):
        patient = generate_patient(i, families, cancer_types)
        patients.append(patient)

    return patients, families


def assign_embeddings(patients, embeddings):
    def get_random_embeddings(embedding_type, count):
        return embeddings[embedding_type].sample(n=count).to_dict(orient='records')

    for patient in patients:
        if patient['has_cancer']:
            cancer_type = patient['cancer_type']
            num_cancer_embeddings = np.random.randint(0, 11)
            num_text_embeddings = np.random.randint(0, 11 - num_cancer_embeddings)
            num_image_embeddings = 10 - num_cancer_embeddings - num_text_embeddings

            patient['embeddings'] = {
                cancer_type: get_random_embeddings(cancer_type, num_cancer_embeddings),
                'text': get_random_embeddings('text', num_text_embeddings),
                'image': get_random_embeddings('image', num_image_embeddings)
            }
        else:
            num_text_embeddings = np.random.randint(0, 11)
            num_image_embeddings = 10 - num_text_embeddings

            patient['embeddings'] = {
                'text': get_random_embeddings('text', num_text_embeddings),
                'image': get_random_embeddings('image', num_image_embeddings)
            } if num_text_embeddings + num_image_embeddings > 0 else {}


def calculate_similarity(embeddings1, embeddings2):
    total_similarity = 0
    for key in embeddings1.keys():
        if key in embeddings2 and len(embeddings1[key]) > 0 and len(embeddings2[key]) > 0:
            emb1 = np.array([list(embedding.values()) for embedding in embeddings1[key]])
            emb2 = np.array([list(embedding.values()) for embedding in embeddings2[key]])
            emb1 = emb1.reshape(1, -1) if emb1.ndim == 1 else emb1
            emb2 = emb2.reshape(1, -1) if emb2.ndim == 1 else emb2
            total_similarity += cosine_similarity(emb1, emb2).sum()
    return total_similarity


def connect_families(G):
    family_clusters = list(nx.connected_components(G))

    while len(family_clusters) > 1:
        max_similarity = -1  # Initialize to a value less than any possible similarity
        best_pair = (None, None)

        # Compare every pair of clusters
        for i in range(len(family_clusters) - 1):
            for j in range(i + 1, len(family_clusters)):
                family_a = list(family_clusters[i])
                family_b = list(family_clusters[j])

                # Find the most similar pair of nodes between the two families
                for node_a in family_a:
                    for node_b in family_b:
                        similarity = calculate_similarity(G.nodes[node_a]['embeddings'], G.nodes[node_b]['embeddings'])
                        if similarity > max_similarity:
                            max_similarity = similarity
                            best_pair = (node_a, node_b)

        # Connect the most similar pair of nodes
        if best_pair != (None, None):
            G.add_edge(best_pair[0], best_pair[1], relationship_type="relative")

        # Recalculate family clusters
        family_clusters = list(nx.connected_components(G))

    # Ensure the entire graph is connected
    components = list(nx.connected_components(G))
    while len(components) > 1:
        for i in range(len(components) - 1):
            component_a = list(components[i])
            component_b = list(components[i + 1])
            G.add_edge(component_a[0], component_b[0], relationship_type="component_link")
        components = list(nx.connected_components(G))



def aggregate_embeddings(embeddings):
    total_sum = 0
    for embedding_list in embeddings.values():
        for embedding in embedding_list:
            total_sum += sum(embedding.values())
    return total_sum


def construct_graph_with_aggregated_embeddings(patients):
    G = nx.Graph()
    for patient in patients:
        aggregated_embeddings = aggregate_embeddings(patient.get("embeddings", {}))
        G.add_node(patient["id"], has_cancer=patient["has_cancer"], cancer_prob=patient.get("cancer_prob", 0.25),
                   aggregated_embeddings=aggregated_embeddings, embeddings=patient.get("embeddings", {}))
        for relationship in patient["relationships"]:
            G.add_edge(patient["id"], relationship["relative"], relationship_type=relationship["type"])

    connect_families(G)
    return G


def visualize_graph_with_embeddings(G):
    pos = nx.spring_layout(G)
    node_colors = ['red' if G.nodes[node]['has_cancer'] else 'blue' for node in G.nodes]
    edge_colors = ['green' if G.edges[edge]['relationship_type'] in ['mother', 'father'] else 'black' for edge in
                   G.edges]
    node_sizes = [500 if G.nodes[node]['aggregated_embeddings'] else 100 for node in G.nodes]

    nx.draw(G, pos, with_labels=True, node_color=node_colors, node_size=node_sizes, font_size=10, font_color='white',
            edge_color=edge_colors)
    plt.show()


if __name__ == '__main__':
    parser = ArgumentParser(description='Arguments for graph generation')
    parser.add_argument("--nodes", "-n", type=int, help="Number of nodes in the graph")
    args = parser.parse_args()

    embeddings = load_embeddings()
    cancer_types = list(embeddings.keys())[:5]  # Extract cancer types from the embedding keys

    num_patients = args.nodes if args.nodes else 100
    patients, families = generate_patients(num_patients, cancer_types)
    assign_embeddings(patients, embeddings)
    G = construct_graph_with_aggregated_embeddings(patients)
    visualize_graph_with_embeddings(G)
