import h5py
from pathlib import Path
import argparse
import numpy as np
from tqdm import tqdm
from typing import Dict, List, Tuple
from concurrent.futures import ProcessPoolExecutor


def extract_submitter_data(
        h5_file_path: str, groups: List[str], submitter_id: str, chunk_size: int = 10000
) -> Dict[str, np.ndarray]:
    """
    Extract all rows for a specific submitter_id from all groups in an HDF5 file.

    Parameters:
        h5_file_path (str): Path to the HDF5 file.
        groups (List[str]): List of group names to extract data from.
        submitter_id (str): The submitter_id to extract data for.
        chunk_size (int): Size of chunks to read data in.

    Returns:
        Dict[str, np.ndarray]: A dictionary where keys are group names and values are numpy arrays for the submitter_id.
    """
    submitter_data: Dict[str, np.ndarray] = {}

    with h5py.File(h5_file_path, "r") as h5_file:
        for group_name in groups:
            if group_name not in h5_file:
                continue

            group = h5_file[group_name]
            indices = h5_file["indices"][group_name].get(submitter_id.encode("utf-8"))
            if indices is None:
                submitter_data[group_name] = np.array([])
                continue

            results = []
            for start in range(0, len(indices), chunk_size):
                end = min(start + chunk_size, len(indices))
                results.append(group["embeddings"][indices[start:end]])
            submitter_data[group_name] = np.concatenate(results, axis=0) if results else np.array([])

    return submitter_data


def sum_random_embeddings(
        submitter_data: Dict[str, np.ndarray], walk_distance: int, amount_of_walks: int
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

    if "rna" not in submitter_data or submitter_data["rna"].size == 0:
        raise ValueError("No RNA data available to extract cancer type.")

    cancer_type = submitter_data["rna"][0][768].decode("utf-8")

    for _ in range(amount_of_walks):
        selected_embeddings: List[np.ndarray] = []

        for _ in range(walk_distance):
            modality = np.random.choice(available_modalities)
            modality_data = submitter_data[modality]
            numeric_part = np.array([modality_data[0][i] for i in range(767)])
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
            submitter_data = extract_submitter_data(h5_file_path, groups, submitter_id)
            summed_embeddings, cancer_type = sum_random_embeddings(
                submitter_data, walk_distance, amount_of_walks
            )
            X_batch.append(summed_embeddings)
            y_batch.append(cancer_type)
            submitter_ids_batch.append(submitter_id)
        except ValueError:
            continue
    return X_batch, y_batch, submitter_ids_batch


def process_all_submitters(
        h5_file_path: Path, groups: List[str], output_path: Path, walk_distance: int, amount_of_walks: int,
        batch_size: int = 100, n_jobs: int = 4
) -> None:
    """
    Process all submitters using parallel processing.

    Parameters:
        h5_file_path (Path): Path to the HDF5 file.
        groups (List[str]): List of group names to process.
        output_path (Path): Path to save combined data.
        walk_distance (int): Number of embeddings to sum.
        amount_of_walks (int): Number of summed embeddings to create.
        batch_size (int): Number of submitters in a chunk.
        n_jobs (int): Number of parallel workers.
    """
    with h5py.File(h5_file_path, "r") as h5_file:
        submitter_ids: List[str] = [
            submitter_id.decode("utf-8") for submitter_id in h5_file["indices"]["submitter_ids"][:]
        ]

    with h5py.File(output_path, "w") as out_file:
        shape = 767 * amount_of_walks
        out_file.create_dataset("X", (0, shape), maxshape=(None, shape), dtype="f")
        out_file.create_dataset("y", (0,), maxshape=(None,), dtype=h5py.string_dtype())
        out_file.create_dataset("submitter_ids", (0,), maxshape=(None,), dtype=h5py.string_dtype())
        out_file.attrs["walk_distance"] = walk_distance
        out_file.attrs["amount_of_walks"] = amount_of_walks

        unique_classes = []
        chunks = [submitter_ids[i:i + batch_size] for i in range(0, len(submitter_ids), batch_size)]
        with ProcessPoolExecutor(max_workers=n_jobs) as executor:
            for X_batch, y_batch, submitter_ids_batch in tqdm(
                    executor.map(process_submitter_chunk, [str(h5_file_path)] * len(chunks), [groups] * len(chunks),
                                 chunks,
                                 [walk_distance] * len(chunks), [amount_of_walks] * len(chunks)),
                    desc="Processing Chunks"):
                current_size = out_file["X"].shape[0]
                new_size = current_size + len(X_batch)

                out_file["X"].resize(new_size, axis=0)
                out_file["X"][current_size:new_size] = np.array(X_batch)

                out_file["y"].resize(new_size, axis=0)
                out_file["y"][current_size:new_size] = y_batch

                out_file["submitter_ids"].resize(new_size, axis=0)
                out_file["submitter_ids"][current_size:new_size] = submitter_ids_batch

                # add all unique classes to the unique classes list, if not already in the list
                for y in y_batch:
                    if y not in unique_classes:
                        unique_classes.append(y)

        out_file.attrs["classes"] = unique_classes
        out_file.attrs["feature_shape"] = 767 * amount_of_walks


def load_groups(h5_file_path: Path) -> List[str]:
    """
    Load all groups from an HDF5 file.

    Parameters:
        h5_file_path (Path): Path to the HDF5 file.

    Returns:
        List[str]: List of group names in the HDF5 file.
    """
    with h5py.File(h5_file_path, "r") as h5_file:
        return [key for key in h5_file.keys() if key != "indices"]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cancer", "-c", nargs="+", required=True, help="The cancer types to work with.")
    parser.add_argument("--walk_distance", "-w", type=int, required=True, help="The walk distance.",
                        choices=[3, 4, 5], default=3)
    parser.add_argument("--amount_of_walks", "-a", type=int, required=True, help="The amount of walks.",
                        choices=[3, 4, 5], default=3)
    args = parser.parse_args()

    selected_cancers: List[str] = args.cancer
    walk_distance: int = args.walk_distance
    amount_of_walks: int = args.amount_of_walks
    cancers: str = "_".join(selected_cancers)

    save_folder = Path("results", "classifier", "summed_embeddings", cancers, f"{walk_distance}_{amount_of_walks}")
    save_folder.mkdir(parents=True, exist_ok=True)

    h5_load_path: Path = Path("results", "embeddings", f"{cancers}.h5")
    groups: List[str] = load_groups(h5_load_path)

    output_file = Path(save_folder, "summed_embeddings.h5")
    process_all_submitters(
        h5_file_path=h5_load_path,
        groups=groups,
        output_path=output_file,
        walk_distance=walk_distance,
        amount_of_walks=amount_of_walks,
    )
