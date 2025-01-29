import h5py
import numpy as np
from tqdm import tqdm
from pathlib import Path
from argparse import ArgumentParser
import logging
import random
from typing import List

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Configuration Constants
SAVE_FOLDER = Path("results", "recognizer", "summed_embeddings", "multi")
LATENT_SPACE_DIM = 768
CHUNK_SIZE = 100000


def get_total_rows_and_columns(f: h5py.File, group: str) -> (int, int):
    """
    Returns the total number of rows and ensures columns are compatible for summation.
    """
    dataset = f[group]["embeddings"]
    total_rows = dataset.shape[0]
    total_columns = len(dataset.dtype) if dataset.dtype.names else dataset.shape[1]
    total_columns = min(total_columns, LATENT_SPACE_DIM)
    return total_rows, total_columns


def filter_rows_by_cancer(dataset: h5py.Dataset, cancer_type: str) -> List[int]:
    """
    Filters rows by cancer type within the given dataset.
    Assumes there is a 'cancer' field in the dataset.
    """

    cancer_bytes = dataset['cancer']
    cancer_str = np.array([c.decode("utf-8") for c in cancer_bytes])
    indices = np.where(cancer_str == cancer_type)[0]
    return indices.tolist()


def reload_image_chunk(f, image_chunk: int, chunk_size: int):
    total_images = f["images"]["embeddings"].shape[0]
    start_id = image_chunk * chunk_size
    end_id = start_id + chunk_size

    # Ensure start_id does not exceed total_images
    if start_id >= total_images:
        image_chunk = (image_chunk + 1) % (total_images // chunk_size)  # Cycle through chunks
        start_id = image_chunk * chunk_size
        end_id = start_id + chunk_size

    # If end_id exceeds total images, shift start_id by 40% of total images
    if end_id > total_images:
        start_id = int(total_images * 0.4)
        end_id = start_id + chunk_size

        # Ensure end_id does not exceed total_images
        if end_id > total_images:
            end_id = total_images
            start_id = max(0, total_images - chunk_size)  # Adjust start_id accordingly

    return f["images"]["embeddings"][start_id:end_id]


def generate_noise(embedding_length: int, scale: float = 0.1) -> np.ndarray:
    """
    Generates Gaussian noise.
    """
    return np.random.normal(loc=0, scale=scale, size=embedding_length).astype(np.float32)


def main():
    image_chunk: int = 0
    # Argument Parsing
    parser = ArgumentParser(description='Sum embeddings from different sources.')
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
    load_path: Path = Path(args.load_path)

    logging.info(f"Selected cancers: {selected_cancers}")
    logging.info(f"Walk distance: {walk_distance}")
    logging.info(f"Amount of summed embeddings: {amount_of_summed_embeddings}")
    logging.info(f"Noise ratio: {noise_ratio}")
    logging.info(f"Load path: {load_path}")

    save_path = Path(SAVE_FOLDER, cancers, str(amount_of_summed_embeddings), str(noise_ratio))
    save_path.mkdir(parents=True, exist_ok=True)
    logging.info(f"Save path: {save_path}")

    combined_data = np.zeros((amount_of_summed_embeddings, LATENT_SPACE_DIM), dtype=np.float32)
    labels_data = {label: np.zeros(amount_of_summed_embeddings, dtype=np.int32)
                   for label in ["Text", "Image", "Mutation", "RNA"] + selected_cancers}

    with h5py.File(Path(load_path, f"{cancers}.h5"), 'r') as f:
        rna_total_rows, rna_columns = get_total_rows_and_columns(f, "rna")
        image_total_rows, image_columns = get_total_rows_and_columns(f, "images")
        mutation_total_rows, mutation_columns = get_total_rows_and_columns(f, "mutations")
        annotation_total_rows, annotation_columns = get_total_rows_and_columns(f, "annotations")

        assert annotation_columns == image_columns == mutation_columns == LATENT_SPACE_DIM, \
            "All modalities must have the same usable dimensions."

        cancer_indices = {}
        rna_dataset = f['rna']
        print(rna_dataset)

        for cancer in selected_cancers:
            indices = filter_rows_by_cancer(rna_dataset, cancer)
            if indices:
                cancer_indices[cancer] = indices

        # Read text, image and rna data into memory
        text_data = f["annotations"]["embeddings"][:]
        mutation_data = f["mutations"]["embeddings"][:]
        # Load the first image chunk
        image_data = reload_image_chunk(f, image_chunk, CHUNK_SIZE)

        data: {} = {
            "Text": text_data,
            "Mutation": mutation_data,
            "Image": image_data
        }

        for cancer, indices in cancer_indices.items():
            data[cancer] = f["rna"]["embeddings"][indices]

        selected_cancer_list = []
        # Initialize cancer selection counts
        cancer_counts = {cancer: 0 for cancer in selected_cancers}

        for i in tqdm(range(amount_of_summed_embeddings), desc="Generating Summed Embeddings"):
            combined_sum = np.zeros(LATENT_SPACE_DIM, dtype=np.float32)

            # Calculate dynamic probabilities for cancer selection
            total_cancer_selections = sum(cancer_counts.values())
            cancer_probs = [
                (1 - (count / total_cancer_selections)) if total_cancer_selections > 0 else 1
                for count in cancer_counts.values()
            ]
            cancer_probs = np.array(cancer_probs) / np.sum(cancer_probs)  # Normalize to sum to 1

            # Select a cancer type based on dynamic probabilities
            selected_cancer = np.random.choice(selected_cancers, p=cancer_probs)
            cancer_counts[selected_cancer] += 1  # Update count

            selected_cancer_list.append(selected_cancer)

            # Update modality weights to include the selected cancer
            modality_weights = {"Text": 0.25, "Image": 0.25, "Mutation": 0.25, selected_cancer: 0.25}
            modality_choices = list(modality_weights.keys())
            modality_probs = list(modality_weights.values())
            # Perform weighted sampling for this embedding
            random_modalities = np.random.choice(modality_choices, size=walk_distance, p=modality_probs)
            unique, counts = np.unique(random_modalities, return_counts=True)
            modality_counts = dict(zip(unique, counts))

            for modality, count in modality_counts.items():
                buffer = data[modality]
                for _ in range(count):
                    if random.random() < noise_ratio:
                        # Add noise vector
                        noise_vector = generate_noise(LATENT_SPACE_DIM)
                        combined_sum += noise_vector
                        continue

                    try:
                        embedding = random.choice(buffer)
                        combined_sum += embedding
                        labels_data[modality][i] += 1

                        # If the modality is the selected cancer, increment RNA label
                        if modality == selected_cancer:
                            labels_data["RNA"][i] += 1
                    except IndexError as e:
                        logging.error(f"IndexError when accessing embedding: {e}")
                        continue
                    except Exception as e:
                        logging.error(f"Unexpected error when accessing embedding: {e}")
                        continue

            combined_data[i] = combined_sum

    # check that the unique selected_cancer_list is the same as the selected_cancers
    unique_selected_cancers = list(set(selected_cancer_list))
    assert set(unique_selected_cancers) == set(selected_cancers), \
        "The unique selected cancers do not match the selected cancers."

    logging.info("Summed embeddings generated successfully. Saving...")
    output_file = Path(save_path, f"{walk_distance}_embeddings.h5")
    with h5py.File(output_file, "w") as f_out:
        f_out.create_dataset("X", data=combined_data, compression="gzip")
        for label, data in labels_data.items():
            f_out.create_dataset(label, data=data, compression="gzip")

    logging.info(f"Saved HDF5 file to {output_file}")


if __name__ == "__main__":
    main()
