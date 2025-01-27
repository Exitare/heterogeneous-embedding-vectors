import h5py
import numpy as np
from tqdm import tqdm
from pathlib import Path
from argparse import ArgumentParser
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Configuration
SAVE_FOLDER = Path("results", "recognizer", "summed_embeddings", "simple")
LATENT_SPACE_DIM = 767
LOAD_FOLDER = Path("results", "embeddings")

def generate_noise(embedding_length: int, scale: float = 0.1) -> np.ndarray:
    """
    Generates Gaussian noise.
    """
    return np.random.normal(loc=0, scale=scale, size=embedding_length).astype(np.float32)


def extract_numeric_embeddings(dataset, latent_space_dim):
    """
    Extract only the numeric columns (first `latent_space_dim` fields) from the dataset.
    """
    # Extract the first `latent_space_dim` numeric fields
    numeric_fields = dataset.dtype.names[:latent_space_dim]
    return np.stack([dataset[field] for field in numeric_fields], axis=1).astype(np.float32)


def main():
    # Argument Parsing
    parser = ArgumentParser(description='Sum embeddings from different sources')
    parser.add_argument("--walk_distance", "-w", type=int, help="Number of embeddings to sum", required=True)
    parser.add_argument("--amount_of_summed_embeddings", "-a", type=int, default=1000,
                        help="Amount of summed embeddings to generate")
    parser.add_argument("--noise_ratio", "-n", type=float, default=0.0, help="Ratio of random noise vectors to add")
    parser.add_argument("--selected_cancers", "-c", nargs="+", required=True, help="The cancer types to work with.")
    args = parser.parse_args()

    amount_of_summed_embeddings: int = args.amount_of_summed_embeddings
    walk_distance: int = args.walk_distance
    noise_ratio: float = args.noise_ratio
    selected_cancers = args.selected_cancers

    if len(selected_cancers) == 1:
        logging.info("Selected cancers is a single string. Converting...")
        selected_cancers = selected_cancers[0].split(" ")

    cancers = "_".join(selected_cancers)

    logging.info(
        f"Parameters: walk_distance={walk_distance}, amount_of_summed_embeddings={amount_of_summed_embeddings}, noise_ratio={noise_ratio}")

    # Prepare Save Directory
    save_path = Path(SAVE_FOLDER, str(amount_of_summed_embeddings), str(noise_ratio))
    save_path.mkdir(parents=True, exist_ok=True)
    logging.info(f"Save path: {save_path}")

    # Initialize Data Structures
    combined_data = np.zeros((amount_of_summed_embeddings, LATENT_SPACE_DIM), dtype=np.float32)
    labels_data = {label: np.zeros(amount_of_summed_embeddings, dtype=np.int32)
                   for label in ["Text", "Image", "RNA", "Mutation"]}

    with h5py.File(Path(LOAD_FOLDER, f"{cancers}.h5"), 'r') as f:
        logging.info("Loading datasets into memory...")

        # Directly load structured datasets into memory
        text_embeddings = f['annotations']['embeddings'][:]
        image_embeddings = f['images']['embeddings'][:]
        mutation_embeddings = f['mutations']['embeddings'][:]
        rna_embeddings = f['rna']['embeddings'][:]

        # Define modality mapping
        modality_to_embeddings = {
            "Text": text_embeddings,
            "Image": image_embeddings,
            "Mutation": mutation_embeddings,
            "RNA": rna_embeddings,
        }

        logging.info("Generating summed embeddings...")
        for i in tqdm(range(amount_of_summed_embeddings), desc="Generating Summed Embeddings"):
            combined_sum = np.zeros(LATENT_SPACE_DIM, dtype=np.float32)
            combination_counts = {modality: 0 for modality in modality_to_embeddings.keys()}

            # Weighted sampling of modalities
            random_modalities = np.random.choice(
                list(modality_to_embeddings.keys()),
                size=walk_distance,
                p=[0.25, 0.25, 0.25, 0.25]
            )

            for modality in random_modalities:
                embeddings = modality_to_embeddings[modality]
                random_index = np.random.randint(len(embeddings))

                if np.random.random() < noise_ratio:
                    # Add noise instead of real embedding
                    combined_sum += generate_noise(LATENT_SPACE_DIM)
                else:
                    # Extract numeric fields from the row (assume first LATENT_SPACE_DIM are numeric)
                    combined_sum += np.array([embeddings[random_index][field]
                                              for field in embeddings.dtype.names[:LATENT_SPACE_DIM]])
                    combination_counts[modality] += 1

            # Save results
            combined_data[i] = combined_sum
            for label, count in combination_counts.items():
                labels_data[label][i] = count

    # Save the data to an HDF5 file with compression for efficiency
    output_file = Path(save_path, f"{walk_distance}_embeddings.h5")
    with h5py.File(output_file, "w") as f_out:
        # Save combined embeddings
        f_out.create_dataset("X", data=combined_data, compression="gzip")

        # Save labels
        for label, data in labels_data.items():
            f_out.create_dataset(label, data=data, compression="gzip")

    logging.info(f"Saved HDF5 file to {output_file}")

if __name__ == '__main__':
    main()
