import h5py
from pathlib import Path
import argparse
import numpy as np
from tqdm import tqdm
from typing import Dict, List, Tuple
from concurrent.futures import ProcessPoolExecutor

LATENT_DIM = 767


def load_full_datasets(h5_file_path: str, exclude_group: str = "Image") -> Dict[str, np.ndarray]:
    """
    Load all datasets into memory except for the excluded group (e.g., 'Image').

    Parameters:
        h5_file_path (str): Path to the HDF5 file.
        exclude_group (str): Group name to exclude from loading.

    Returns:
        Dict[str, np.ndarray]: Dictionary with dataset names as keys and NumPy arrays as values.
    """
    full_data = {}

    with h5py.File(h5_file_path, "r") as h5_file:
        for group_name in h5_file.keys():
            if group_name == "indices" or group_name == exclude_group:
                continue  # Skip index dataset and excluded group (Image)

            # Load the entire dataset into memory
            full_data[group_name] = np.array(h5_file[group_name]["embeddings"][:])

    return full_data


def extract_submitter_data(
        h5_file_path: str, groups: List[str], submitter_id: str, chunk_size: int = 100000
) -> Tuple[Dict[str, np.ndarray], str]:
    """
    Extract submitter-specific data from all groups, loading everything except Image into memory.

    Parameters:
        h5_file_path (str): Path to the HDF5 file.
        groups (List[str]): List of group names to extract data from.
        submitter_id (str): The submitter_id to extract data for.
        chunk_size (int): Size of chunks to read image data in.

    Returns:
        Tuple[Dict[str, np.ndarray], str]:
            - Dictionary with extracted embeddings (Image chunked, others loaded in memory).
            - The submitter's cancer type.
    """
    submitter_data: Dict[str, np.ndarray] = {}
    cancer_type = None

    with h5py.File(h5_file_path, "r") as h5_file:
        for group_name in groups:
            if group_name not in h5_file:
                continue

            group = h5_file[group_name]

            # Extract cancer type from the first group it appears in
            if cancer_type is None and "cancer" in group:
                cancer_type = group["cancer"][0]
                if isinstance(cancer_type, bytes):
                    cancer_type = cancer_type.decode("utf-8")

            # Process image data in chunks
            if group_name == "Image":
                indices = h5_file["indices"][group_name].get(submitter_id.encode("utf-8"))
                if indices is None:
                    submitter_data[group_name] = np.array([])
                    continue

                results = []
                for start in range(0, len(indices), chunk_size):
                    end = min(start + chunk_size, len(indices))
                    results.append(group["embeddings"][indices[start:end]])
                submitter_data[group_name] = np.concatenate(results, axis=0) if results else np.array([])

    if cancer_type is None:
        raise ValueError(f"No cancer type found for submitter {submitter_id} in any group.")

    return submitter_data, cancer_type


def sum_random_embeddings(
        submitter_data: Dict[str, np.ndarray], walk_distance: int, amount_of_walks: int, cancer_type: str,
) -> Tuple[np.ndarray, str]:
    """
    Create summed embeddings by selecting random embeddings from random modalities.

    Parameters:
        submitter_data (Dict[str, np.ndarray]): Dictionary of numpy arrays for a submitter, keyed by modality.
        walk_distance (int): Number of embeddings to sum.
        amount_of_walks (int): Number of summed embeddings to create.

    Returns:
        Tuple[np.ndarray, str]: Concatenated summed embeddings and cancer type.
    """
    summed_embeddings: List[np.ndarray] = []

    available_modalities = [modality for modality, data in submitter_data.items() if data.size > 0]
    if not available_modalities:
        raise ValueError("No available modalities with data for the submitter.")

    for _ in range(amount_of_walks):
        selected_embeddings: List[np.ndarray] = []

        for _ in range(walk_distance):
            modality = np.random.choice(available_modalities)
            modality_data = submitter_data[modality]
            numeric_part = modality_data[0][:LATENT_DIM]
            selected_embeddings.append(numeric_part)

        summed_embedding = np.sum(selected_embeddings, axis=0)
        summed_embeddings.append(summed_embedding)

    concatenated_embeddings = np.concatenate(summed_embeddings, axis=0)
    return concatenated_embeddings, cancer_type


def process_submitter_chunk(
        h5_file_path: str, groups: List[str], submitter_ids: List[str], walk_distance: int, amount_of_walks: int
) -> Tuple[List[np.ndarray], List[str], List[str]]:
    """
    Process a chunk of submitters in parallel.

    Parameters:
        h5_file_path (str): Path to the HDF5 file.
        groups (List[str]): List of group names to process.
        submitter_ids (List[str]): List of submitter IDs in the chunk.
        walk_distance (int): Number of embeddings to sum.
        amount_of_walks (int): Number of summed embeddings to create.

    Returns:
        Tuple[List[np.ndarray], List[str], List[str]]: Summed embeddings, cancer types, and submitter IDs.
    """
    X_batch, y_batch, submitter_ids_batch = [], [], []
    for submitter_id in submitter_ids:
        try:
            submitter_data, cancer_type = extract_submitter_data(h5_file_path, groups, submitter_id)
            summed_embeddings, cancer_type = sum_random_embeddings(
                submitter_data, walk_distance, amount_of_walks, cancer_type)
            X_batch.append(summed_embeddings)
            y_batch.append(cancer_type)
            submitter_ids_batch.append(submitter_id)
        except ValueError:
            continue
    return X_batch, y_batch, submitter_ids_batch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cancer", "-c", nargs="+", required=True, help="The cancer types to work with.")
    parser.add_argument("--walk_distance", "-w", type=int, required=True, help="The walk distance.",
                        choices=[3, 4, 5], default=3)
    parser.add_argument("--amount_of_walks", "-a", type=int, required=True, help="The amount of walks.",
                        choices=[3, 4, 5], default=3)
    args = parser.parse_args()

    selected_cancers: List[str] = args.cancer
    walk_distance: int = args.walk_distance
    walk_amount: int = args.amount_of_walks
    cancers: str = "_".join(selected_cancers)

    save_folder = Path("results", "classifier", "summed_embeddings", cancers, f"{walk_distance}_{walk_amount}")
    save_folder.mkdir(parents=True, exist_ok=True)

    h5_load_path: Path = Path("results", "embeddings", f"{cancers}.h5")
    output_file = Path(save_folder, "summed_embeddings.h5")

    # Load all non-image datasets into memory
    full_data = load_full_datasets(str(h5_load_path))

    with h5py.File(h5_load_path, "r") as h5_file:
        groups = [key for key in h5_file.keys() if key != "indices"]

    process_all_submitters(
        h5_file_path=h5_load_path,
        groups=groups,
        output_path=output_file,
        walk_distance=walk_distance,
        walk_amount=walk_amount,
    )


if __name__ == "__main__":
    main()
