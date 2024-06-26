import pandas as pd
from argparse import ArgumentParser
from pathlib import Path
from tqdm import tqdm
import numpy as np
import random
import networkx as nx
from sklearn.metrics.pairwise import cosine_similarity


# Query the graph: Example of getting neighbors
def get_neighbors(node_id):
    return list(G.neighbors(node_id))


# Perform a random walk
def random_walk(graph, start_node, walk_length):
    walk = [start_node]
    current_node = start_node
    for _ in range(walk_length - 1):
        neighbors = list(graph.neighbors(current_node))
        if not neighbors:
            break
        current_node = random.choice(neighbors)
        walk.append(current_node)
    return walk

def sum_embeddings_from_walk(graph, walk):
    embeddings = [graph.nodes[node]['embedding'] for node in walk]
    return np.sum(embeddings, axis=0), embeddings


def random_sum_embeddings(embeddings, max_count):
    # Randomly choose embeddings up to max_count
    n = random.randint(1, max_count)  # Ensure at least one is selected
    chosen_indices = random.sample(range(len(embeddings)), n)
    chosen_embeddings = embeddings.iloc[chosen_indices]
    summed_embeddings = chosen_embeddings.sum(axis=0)
    return summed_embeddings, n


if __name__ == '__main__':

    parser = ArgumentParser()
    parser.add_argument("--cancer", "-c", nargs='+', required=True, help="The cancer type to work with.")
    parser.add_argument("--iterations", "-i", type=int, required=False, help="Number of iterations to run.",
                        default=100)
    parser.add_argument("--embeddings", "-e", type=int, required=False, help="Number of embeddings to sum.", default=2)
    args = parser.parse_args()

    selected_cancers = args.cancer
    iterations = args.iterations
    total_embeddings = args.embeddings

    loaded_cancer_embeddings = {}
    for cancer in selected_cancers:
        loaded_cancer_embeddings[cancer] = pd.read_csv(
            Path("results", "embeddings", "cancer", f"{cancer.lower()}_embeddings.csv"), nrows=100)

    # Load the embeddings from CSV files
    image_embeddings = pd.read_csv(Path("results", "embeddings", 'image_embeddings.csv'), nrows=100)
    sentence_embeddings = pd.read_csv(Path("results", "embeddings", 'sentence_embeddings.csv'), nrows=100)

    combined_data = []

    for _ in tqdm(range(iterations)):
        # Determine random order for processing embeddings
        embeddings_list = [(sentence_embeddings, 'Text'), (image_embeddings, 'Image')]
        for cancer_type, cancer_embedding in loaded_cancer_embeddings.items():
            embeddings_list.append((cancer_embedding, cancer_type))

        random.shuffle(embeddings_list)

        combined_sum = pd.Series(np.zeros_like(embeddings_list[0][0].iloc[0]), index=embeddings_list[0][0].columns)
        remaining_embeddings = total_embeddings
        combination_counts = {}
        selected_cancer_type = None

        for embeddings, name in embeddings_list:
            if remaining_embeddings > 0:
                if name in loaded_cancer_embeddings.keys():
                    # If a specific cancer type was selected, continue using that type
                    if selected_cancer_type is None:
                        selected_cancer_type = name
                    elif name != selected_cancer_type:
                        combination_counts[name] = 0
                        continue

                current_sum, count = random_sum_embeddings(embeddings, remaining_embeddings)
                combined_sum += current_sum
                remaining_embeddings -= count
                combination_counts[name] = count
            else:
                combination_counts[name] = 0

        # Ensure there is at least one embedding selected in total (avoid all-zero entries)
        if all(v == 0 for v in combination_counts.values()):
            embeddings, name = random.choice(embeddings_list)
            current_sum, count = random_sum_embeddings(embeddings, 1)  # Force at least one selection
            combined_sum += current_sum
            combination_counts[name] = count

        # Sort the combination counts by the keys
        combination_counts = dict(sorted(combination_counts.items()))

        # Combine combined_sum and the combination_counts which are Image, Text and the cancer types
        combined_data.append(
            list(combined_sum) + [combination_counts.get('Image', 0), combination_counts.get('Text', 0)] + [
                combination_counts.get(cancer_type, 0) for cancer_type in loaded_cancer_embeddings.keys()])

        # Save the data to CSV
    column_names = list(embeddings_list[0][0].columns) + ['Image', 'Text'] + [
        cancer_type for cancer_type in loaded_cancer_embeddings.keys()]

    combined_df = pd.DataFrame(combined_data, columns=column_names)
    # create another column called RNA which is the sum of the cancer types in the dataset, defined by the loaded_cancer_embeddings
    combined_df["RNA"] = combined_df[[cancer_type for cancer_type in loaded_cancer_embeddings.keys()]].sum(axis=1)
    print(combined_df)

    # Create a graph
    G = nx.Graph()

    # Add product nodes
    for idx, embedding in enumerate(combined_df.iloc[:, :768].values):
        G.add_node(idx, embedding=embedding)

    # Compute similarity matrix and add edges
    similarity_matrix = np.corrcoef(combined_df.iloc[:, :768].values)
    similarity_threshold = 0.7

    for i in range(len(combined_df)):
        for j in range(i + 1, len(combined_df)):
            if similarity_matrix[i, j] > similarity_threshold:
                G.add_edge(i, j, weight=similarity_matrix[i, j])


    print(G)
    # Example of performing a random walk
    start_node = 0  # Starting node ID
    walk_length = 10  # Length of the walk
    walk = random_walk(G, start_node, walk_length)
    print(f"Random Walk: {walk}")

    # sum the random walk to create a new embedding
    walk_embedding = np.zeros_like(combined_df.iloc[:, :768].values[0])
    for node in walk:
        walk_embedding += G.nodes[node]['embedding']
    print(walk_embedding)

    # Perform random walks and create the dataset
    walk_length = 10  # Length of the walk
    num_walks = 10  # Number of walks to perform
    aggregated_embeddings = []
    ground_truth_embeddings = []

    for _ in tqdm(range(num_walks)):
        start_node = random.choice(list(G.nodes))
        walk = random_walk(G, start_node, walk_length)
        aggregated_embedding, individual_embeddings = sum_embeddings_from_walk(G, walk)
        aggregated_embeddings.append(aggregated_embedding)
        # Concatenate the individual embeddings to form the ground truth
        ground_truth_embeddings.append(np.concatenate(individual_embeddings))

    # Ensure all ground truth embeddings have the same length by padding with zeros
    max_length = max(len(emb) for emb in ground_truth_embeddings)
    ground_truth_embeddings_padded = [np.pad(emb, (0, max_length - len(emb)), 'constant') for emb in
                                      ground_truth_embeddings]

    aggregated_embeddings = pd.DataFrame(aggregated_embeddings)
    ground_truth_embeddings_padded = pd.DataFrame(ground_truth_embeddings_padded)


    print(aggregated_embeddings)
    print(ground_truth_embeddings_padded)

