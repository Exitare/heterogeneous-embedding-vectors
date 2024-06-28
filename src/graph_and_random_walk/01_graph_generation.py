import pandas as pd
from argparse import ArgumentParser
from pathlib import Path
from tqdm import tqdm
import numpy as np
import random
import networkx as nx
from sklearn.metrics.pairwise import cosine_similarity

results_folder = Path("results", "graph_embeddings")
embeddings_size = 768


# Query the graph: Example of getting neighbors
def get_neighbors(node_id):
    return list(G.neighbors(node_id))


# Perform a random walk
# Function to perform a random walk
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


# Function to sum embeddings from the random walk and return individual embeddings and additional info
def sum_embeddings_from_walk(graph, walk):
    individual_embeddings = [graph.nodes[node]['embedding'] for node in walk]
    additional_info = [graph.nodes[node]['additional_info'] for node in walk]
    aggregated_embedding = np.sum(individual_embeddings, axis=0)
    return aggregated_embedding, individual_embeddings, additional_info


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
    parser.add_argument("--walks", "-w", type=int, required=False, help="Number of walks to perform.", default=10)
    args = parser.parse_args()

    selected_cancers = args.cancer
    iterations = args.iterations
    cancers = "_".join(selected_cancers)
    num_walks = args.walks

    results_folder = Path(results_folder, cancers, "graph_generation")

    if not results_folder.exists():
        results_folder.mkdir(parents=True)

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
        # remaining embeddings should be a number between 2 and 10
        remaining_embeddings = random.randint(2, 10)
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
    # save combined_df to a csv file
    combined_df.to_csv(Path(results_folder, "combined_embeddings.csv"), index=False)

    # Separate the embeddings and the additional columns
    embeddings = combined_df.iloc[:, :embeddings_size].values
    additional_columns = combined_df.iloc[:, embeddings_size:].values

    # Create the graph and add nodes with embeddings
    G = nx.Graph()
    node_to_index = {}
    for idx, embedding in combined_df.iterrows():
        node_id = idx  # or some unique identifier if available
        G.add_node(node_id, embedding=embedding[:embeddings_size].values,
                   additional_info=embedding[embeddings_size:].values)
        node_to_index[node_id] = idx

    # Compute similarity matrix and add edges
    similarity_matrix = np.corrcoef(embeddings)
    similarity_threshold = 0.8

    for i in range(len(embeddings)):
        for j in range(i + 1, len(embeddings)):
            if similarity_matrix[i, j] > similarity_threshold:
                G.add_edge(i, j, weight=similarity_matrix[i, j])

    print(G)
    # Example of performing a random walk
    # start node should be random between 0 and the max
    start_node = random.randint(0, len(G.nodes) - 1)
    walk_length = random.randint(5,10)  # Length of the walk
    aggregated_walk_embeddings = []
    ground_truth_embeddings = []
    composition_details = []

    print("Performing random walks...")
    while len(aggregated_walk_embeddings) < num_walks:
        start_node = random.choice(list(G.nodes))
        walk = random_walk(G, start_node, walk_length)
        if len(walk) < walk_length:
            print(f"Walk length {len(walk)} is less than the specified length {walk_length}. Skipping...")
            continue
        aggregated_embedding, individual_embeddings, additional_info = sum_embeddings_from_walk(G, walk)
        # assert that aggregated embeddings are not just 0
        if np.sum(aggregated_embedding) == 0:
            print("Aggregated embeddings are all zero. Skipping...")
            continue
        aggregated_walk_embeddings.append(aggregated_embedding)
        # Concatenate the individual embeddings to form the ground truth
        ground_truth_embeddings.append(np.concatenate(individual_embeddings))
        # Keep track of the node composition in the walk
        composition_details.append([node_to_index[node] for node in walk])

    # Ensure all ground truth embeddings have the same length by padding with zeros
    max_length = max(len(emb) for emb in ground_truth_embeddings)
    ground_truth_embeddings_padded = [np.pad(emb, (0, max_length - len(emb)), 'constant') for emb in
                                      ground_truth_embeddings]

    aggregated_walk_embeddings = np.array(aggregated_walk_embeddings)
    ground_truth_embeddings_padded = np.array(ground_truth_embeddings_padded)
    composition_details_padded = [np.pad(comp, (0, walk_length - len(comp)), 'constant') for comp in
                                  composition_details]

    # Create DataFrames for the dataset
    aggregated_df = pd.DataFrame(aggregated_walk_embeddings)
    ground_truth_df = pd.DataFrame(ground_truth_embeddings_padded)
    composition_df = pd.DataFrame(composition_details_padded)

    # Save the dataset to CSV files (optional)
    aggregated_df.to_csv(Path(results_folder, 'aggregated_walk_embeddings.csv'), index=False)
    ground_truth_df.to_csv(Path(results_folder, 'ground_truth_embeddings.csv'), index=False)
    composition_df.to_csv(Path(results_folder, 'composition_details.csv'), index=False)
