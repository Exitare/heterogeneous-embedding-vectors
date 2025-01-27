import h5py
import numpy as np
from tqdm import tqdm
from pathlib import Path
from argparse import ArgumentParser
import logging
import random
from typing import List, Dict, Optional

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Configuration Constants
SAVE_FOLDER = Path("results", "recognizer", "summed_embeddings", "multi")
LATENT_SPACE_DIM = 767


def get_total_rows_and_columns(f: h5py.File, group: str) -> (int, int):
    """
    Returns the total number of rows and ensures columns are compatible for summation.
    """
    dataset = f[group]["embeddings"]
    total_rows = dataset.shape[0]
    if dataset.dtype.names:
        total_columns = len(dataset.dtype.names)  # Number of fields
    else:
        total_columns = dataset.shape[1] if len(dataset.shape) > 1 else 1
    total_columns = min(total_columns, LATENT_SPACE_DIM)
    return total_rows, total_columns


def filter_rows_by_cancer(dataset: h5py.Dataset, cancer_type: str) -> List[int]:
    """
    Filters rows by cancer type within the given dataset.
    Assumes there is a 'cancer' field in the dataset.
    """
    if 'cancer' not in dataset.dtype.names:
        logging.error(f"'cancer' field not found in dataset '{dataset.name}'.")
        raise KeyError("'cancer' field not found in dataset.")

    # Use vectorized operations for filtering
    cancer_bytes = dataset['cancer']
    # Decode all at once
    cancer_str = np.array([c.decode("utf-8") for c in cancer_bytes])
    indices = np.where(cancer_str == cancer_type)[0]
    return indices.tolist()


def generate_noise(embedding_length: int, scale: float = 0.1) -> np.ndarray:
    """
    Generates Gaussian noise.
    """
    return np.random.normal(loc=0, scale=scale, size=embedding_length).astype(np.float32)


def main():
    # Argument Parsing
    parser = ArgumentParser(
        description='Sum embeddings from different sources with performance improvements'
    )
    parser.add_argument("--walk_distance", "-w", type=int, help="Number of embeddings to sum", required=True)
    parser.add_argument("--amount_of_summed_embeddings", "-a", type=int, default=200000,
                        help="Amount of summed embeddings to generate")
    parser.add_argument("--selected_cancers", "-c", nargs="+", required=True,
                        help="The selected cancer identifiers to sum")
    parser.add_argument("--noise_ratio", "-n", type=float, default=0.0, help="Ratio of random noise vectors to add")
    parser.add_argument("--debug", "-d", action="store_true", help="Enable debug mode")
    parser.add_argument("--load_path", "-l", type=str, default="results/embeddings",
                        help="Path to the embeddings folder")
    args = parser.parse_args()

    amount_of_summed_embeddings = args.amount_of_summed_embeddings
    walk_distance = args.walk_distance
    noise_ratio = args.noise_ratio
    selected_cancers = args.selected_cancers

    if len(selected_cancers) == 1:
        logging.info("Selected cancers is a single string. Converting...")
        selected_cancers = selected_cancers[0].split(" ")

    cancers = "_".join(selected_cancers)

    LOAD_PATH: Path = Path(args.load_path)

    # Set logging level based on debug flag
    logging.info(f"Selected cancers: {selected_cancers}")
    logging.info(f"Walk distance: {walk_distance}")
    logging.info(f"Amount of summed embeddings: {amount_of_summed_embeddings}")
    logging.info(f"Noise ratio: {noise_ratio}")
    logging.info(f"Load path: {LOAD_PATH}")

    # Prepare Save Directory
    save_path = Path(SAVE_FOLDER, cancers, str(amount_of_summed_embeddings), str(noise_ratio))
    save_path.mkdir(parents=True, exist_ok=True)
    logging.info(f"Save path: {save_path}")

    # Initialize data structures
    combined_data = np.zeros((amount_of_summed_embeddings, LATENT_SPACE_DIM), dtype=np.float32)
    labels = ["Text", "Image", "Mutation", "RNA"] + selected_cancers
    labels_data = {label: np.zeros(amount_of_summed_embeddings, dtype=np.int32) for label in labels}

    with h5py.File(Path(LOAD_PATH, f"{cancers}.h5"), 'r') as f:
        logging.info("Loading datasets into memory...")
        # Load all datasets into memory
        text_embeddings = f['annotations']['embeddings'][:]
        image_embeddings = f['images']['embeddings'][:]
        mutation_embeddings = f['mutations']['embeddings'][:]
        rna_dataset = f['rna']['embeddings'][:]

        # Filter RNA dataset by cancer type
        cancer_indices = {
            cancer: filter_rows_by_cancer(f['rna']['embeddings'], cancer)
            for cancer in selected_cancers
        }
        logging.info(f"Filtered RNA indices: {cancer_indices}")

    logging.info("Generating summed embeddings...")
    cancer_counts = {cancer: 0 for cancer in selected_cancers}

    selected_cancer_list = []
    # Initialize cancer selection counts
    cancer_counts = {cancer: 0 for cancer in selected_cancers}

    logging.info("Generating summed embeddings...")
    # Generate summed embeddings
    for i in tqdm(range(amount_of_summed_embeddings), desc="Generating Summed Embeddings"):
        combined_sum = np.zeros(LATENT_SPACE_DIM, dtype=np.float32)

        # Dynamic probabilities for cancer selection
        total_cancer_selections = sum(cancer_counts.values())
        cancer_probs = [
            (1 - (count / total_cancer_selections)) if total_cancer_selections > 0 else 1
            for count in cancer_counts.values()
        ]
        cancer_probs = np.array(cancer_probs) / np.sum(cancer_probs)

        # Select a cancer type
        selected_cancer = np.random.choice(selected_cancers, p=cancer_probs)
        cancer_counts[selected_cancer] += 1

        # Modality weights
        modality_weights = {"Text": 0.25, "Image": 0.25, "Mutation": 0.25, selected_cancer: 0.25}
        modality_choices = list(modality_weights.keys())
        modality_probs = list(modality_weights.values())

        # Weighted sampling
        random_modalities = np.random.choice(modality_choices, size=walk_distance, p=modality_probs)
        unique, counts = np.unique(random_modalities, return_counts=True)
        modality_counts = dict(zip(unique, counts))

        for modality, count in modality_counts.items():
            for _ in range(count):
                if random.random() < noise_ratio:
                    # Add noise
                    combined_sum += generate_noise(LATENT_SPACE_DIM)
                    continue

                # Select a random embedding
                if modality == "Text":
                    embedding = text_embeddings[np.random.randint(text_embeddings.shape[0])]
                elif modality == "Image":
                    embedding = image_embeddings[np.random.randint(image_embeddings.shape[0])]
                elif modality == "Mutation":
                    embedding = mutation_embeddings[np.random.randint(mutation_embeddings.shape[0])]
                elif modality in selected_cancers:
                    random_index = np.random.choice(cancer_indices[selected_cancer])
                    embedding = rna_dataset[random_index]
                else:
                    raise ValueError(f"Unknown modality: {modality}")

                # Only use the first LATENT_SPACE_DIM fields
                combined_sum += np.array([embedding[field] for field in embedding.dtype.names[:LATENT_SPACE_DIM]])

                labels_data[modality][i] += 1
                if modality == selected_cancer:
                    labels_data["RNA"][i] += 1

        combined_data[i] = combined_sum

    # Save the data to an HDF5 file
    output_file = Path(save_path, f"{walk_distance}_embeddings.h5")
    with h5py.File(output_file, "w") as f_out:
        f_out.create_dataset("X", data=combined_data, compression="gzip")

        # Save labels
        for label, data in labels_data.items():
            f_out.create_dataset(label, data=data, compression="gzip")

    logging.info(f"Saved HDF5 file to {output_file}")


if __name__ == '__main__':
    main()
