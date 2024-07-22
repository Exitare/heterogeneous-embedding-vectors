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


def generate_patient(patient_id):
    has_cancer = np.random.rand() < 0.25  # 25% probability of having cancer
    return {
        "id": patient_id,
        "has_cancer": has_cancer,
        "relationships": []
    }


def generate_relationships(patients):
    relationship_types = ["mother", "father", "daughter", "son", "brother", "sister"]
    for patient in patients:
        if np.random.rand() < 0.5:  # 50% chance of having a relationship
            relative = np.random.choice(patients)
            while relative["id"] == patient["id"]:
                relative = np.random.choice(patients)
            relationship_type = np.random.choice(relationship_types)
            patient["relationships"].append({"relative": relative["id"], "type": relationship_type})
            relative["relationships"].append({"relative": patient["id"], "type": relationship_type})


def generate_patients_and_relationships(num_patients):
    patients = [generate_patient(i) for i in range(num_patients)]
    generate_relationships(patients)
    return patients


def bayesian_update(prior, likelihood, evidence):
    return (likelihood * prior) / evidence


def update_cancer_probabilities(patients):
    base_cancer_prob = 0.25  # Initial probability of having cancer
    elevated_cancer_prob = 0.5  # Probability of having cancer if a parent has it

    for patient in patients:
        if any(rel["type"] in ["mother", "father"] and patients[rel["relative"]]["has_cancer"] for rel in
               patient["relationships"]):
            # Update probability if a parent has cancer
            patient["cancer_prob"] = bayesian_update(base_cancer_prob, elevated_cancer_prob, 1)
        else:
            patient["cancer_prob"] = base_cancer_prob


def assign_embeddings(patients, embeddings):
    def get_random_embeddings(embedding_type, count):
        return embeddings[embedding_type].sample(n=count).to_dict(orient='records')

    for patient in patients:
        if patient['has_cancer']:
            cancer_type = np.random.choice(list(embeddings.keys())[:5])  # Randomly select a cancer type
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


def aggregate_embeddings(embeddings):
    total_sum = 0
    for embedding_list in embeddings.values():
        for embedding in embedding_list:
            total_sum += sum(embedding.values())
    return total_sum


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


def connect_isolated_patients(G):
    isolated_nodes = [node for node in G.nodes if G.degree[node] == 0]
    non_isolated_nodes = [node for node in G.nodes if G.degree[node] > 0]

    for node in isolated_nodes:
        node_embeddings = G.nodes[node]['embeddings']
        similarities = []

        for other_node in non_isolated_nodes:
            other_embeddings = G.nodes[other_node]['embeddings']
            if node_embeddings and other_embeddings:
                similarity = calculate_similarity(node_embeddings, other_embeddings)
            else:
                similarity = 0
            similarities.append(similarity)

        if similarities and max(similarities) > 0:  # Ensure there are valid similarities to compare
            most_similar_node = non_isolated_nodes[np.argmax(similarities)]
            G.add_edge(node, most_similar_node, relationship_type="closest")
        else:
            # If there are no valid similarities, connect to a random non-isolated node
            random_node = np.random.choice(non_isolated_nodes)
            G.add_edge(node, random_node, relationship_type="random")

    # Final check to ensure all nodes are connected
    remaining_isolated_nodes = [node for node in G.nodes if G.degree[node] == 0]
    while remaining_isolated_nodes:
        for node in remaining_isolated_nodes:
            random_node = np.random.choice(list(G.nodes))
            if node != random_node:
                G.add_edge(node, random_node, relationship_type="random")
        remaining_isolated_nodes = [node for node in G.nodes if G.degree[node] == 0]

    # Ensure the graph is fully connected
    components = list(nx.connected_components(G))
    while len(components) > 1:
        for i in range(len(components) - 1):
            component_a = list(components[i])
            component_b = list(components[i + 1])
            G.add_edge(component_a[0], component_b[0], relationship_type="component_link")
        components = list(nx.connected_components(G))


def construct_graph_with_aggregated_embeddings(patients):
    G = nx.Graph()
    for patient in patients:
        aggregated_embeddings = aggregate_embeddings(patient.get("embeddings", {}))
        G.add_node(patient["id"], has_cancer=patient["has_cancer"], cancer_prob=patient["cancer_prob"],
                   aggregated_embeddings=aggregated_embeddings, embeddings=patient.get("embeddings", {}))
        for relationship in patient["relationships"]:
            G.add_edge(patient["id"], relationship["relative"], relationship_type=relationship["type"])
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

    num_patients = args.nodes if args.nodes else 100
    patients = generate_patients_and_relationships(num_patients)
    update_cancer_probabilities(patients)
    assign_embeddings(patients, embeddings)
    G = construct_graph_with_aggregated_embeddings(patients)
    connect_isolated_patients(G)
    visualize_graph_with_embeddings(G)
