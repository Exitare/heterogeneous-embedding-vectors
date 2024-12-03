import pandas as pd
import h5py
from pathlib import Path
import argparse
import numpy as np
from tqdm import tqdm
from typing import Dict, List

save_folder = Path("results", "classifier", "summed_embeddings")


def extract_submitter_data(
        h5_file: h5py.File, groups: List[str], submitter_id: str
) -> Dict[str, np.ndarray]:
    """
    Extract all rows for a specific submitter_id from all groups in an HDF5 file.

    Parameters:
        h5_file (h5py.File): Open HDF5 file object.
        groups (List[str]): List of group names to extract data from.
        submitter_id (str): The submitter_id to extract data for.

    Returns:
        Dict[str, np.ndarray]: A dictionary where keys are group names and values are numpy arrays for the submitter_id.
    """
    submitter_data: Dict[str, np.ndarray] = {}

    for group_name in groups:
        group = h5_file[group_name]
        indices = h5_file["indices"][group_name].get(submitter_id.encode("utf-8"))
        if indices is not None:
            rows = group["embeddings"][indices[:]]
            submitter_data[group_name] = rows
        else:
            submitter_data[group_name] = np.array([])  # Empty array if no data for this submitter

    return submitter_data


def sum_random_embeddings(
        submitter_data: Dict[str, np.ndarray], walk_distance: int, amount_of_walks: int
) -> np.ndarray:
    """
    Create summed embeddings by selecting random embeddings from random modalities.

    Parameters:
        submitter_data (Dict[str, np.ndarray]): Dictionary of numpy arrays for a submitter, keyed by modality.
        walk_distance (int): Number of embeddings to sum.
        amount_of_walks (int): Number of summed embeddings to create.

    Returns:
        np.ndarray: Concatenated summed embeddings with cancer type.
    """
    summed_embeddings: List[np.ndarray] = []
    cancer_type: str = ""

    for _ in range(amount_of_walks):
        selected_embeddings: List[np.ndarray] = []
        available_modalities: List[str] = [
            modality for modality, data in submitter_data.items() if data.size > 0
        ]

        if not available_modalities:
            raise ValueError("No available modalities with data for the submitter.")

        for _ in range(walk_distance):
            retries = len(available_modalities)
            while retries > 0:
                modality: str = np.random.choice(available_modalities)
                # print(f"Selected modality: {modality}")
                modality_data = submitter_data[modality]
                # Ensure modality_data is valid
                if modality_data.size > 0 and isinstance(modality_data, np.ndarray):
                    selected_row = modality_data[np.random.randint(len(modality_data))]

                    if len(selected_row) < 768:
                        raise ValueError(
                            f"Selected row for modality '{modality}' has insufficient columns: {len(selected_row)}."
                        )

                    # print(f"Selected row for modality '{modality}' has {len(selected_row)} columns.")
                    selected_row = modality_data[np.random.randint(len(modality_data))]
                    selected_embeddings.append(np.array([selected_row[i] for i in range(767)]))

                    if modality == "rna":
                        # Extract the cancer type (column 768)
                        cancer_type = selected_row[768].decode("utf-8")
                        submitter_id = selected_row[769].decode("utf-8")

                    break

                # print(f"Removing modality {modality}...")
                available_modalities.remove(modality)
                retries -= 1

            if retries == 0:
                raise ValueError(
                    "Could not find a valid row in any modality after retries."
                )

        summed_embedding = np.sum(selected_embeddings, axis=0)
        summed_embeddings.append(summed_embedding)

    concatenated_embeddings: np.ndarray = np.concatenate(summed_embeddings)
    concatenated_embeddings = np.append(concatenated_embeddings, cancer_type)

    assert len(concatenated_embeddings) == 767 * amount_of_walks + 1

    return concatenated_embeddings


def process_all_submitters(
        h5_file_path: Path, groups: List[str], output_path: Path, walk_distance: int, amount_of_walks: int
) -> None:
    """
    Process all submitters, creating summed embeddings for each.

    Parameters:
        h5_file_path (Path): Path to the HDF5 file.
        groups (List[str]): List of group names to process.
        output_path (Path): Path to save combined data.
        walk_distance (int): Number of embeddings to sum.
        amount_of_walks (int): Number of summed embeddings to create.
    """
    with h5py.File(h5_file_path, "r") as h5_file:
        submitter_ids: List[str] = [
            submitter_id.decode("utf-8") for submitter_id in h5_file["indices"]["submitter_ids"][:]
        ]

    result_rows: List[np.ndarray] = []

    for submitter_id in tqdm(submitter_ids, desc="Processing Submitters"):
        with h5py.File(h5_file_path, "r") as h5_file:
            submitter_data = extract_submitter_data(h5_file, groups, submitter_id)

        summed_embeddings = sum_random_embeddings(submitter_data, walk_distance, amount_of_walks)
        result_rows.append(summed_embeddings)

    result_df = pd.DataFrame(result_rows)
    result_df.to_csv(output_path, index=False, header=False)
    print(f"Saved summed embeddings to {output_path}")


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
    print("Selected cancers: ", selected_cancers)
    print(f"Using walk distance of {walk_distance} and {amount_of_walks} walks.")

    cancers: str = "_".join(selected_cancers)

    save_folder = Path(save_folder, cancers, f"{walk_distance}_{amount_of_walks}")
    print(f"Save folder: {save_folder}")
    if not save_folder.exists():
        save_folder.mkdir(parents=True)

    h5_load_path: Path = Path("results", "embeddings", f"{cancers}.h5")
    groups: List[str] = load_groups(h5_load_path)
    print(f"Groups: {groups}")
    # display the amount of data for each group
    for group in groups:
        with h5py.File(h5_load_path, "r") as h5_file:
            print(f"Group {group} has {len(h5_file[group]['embeddings'])} rows of data.")

    process_all_submitters(
        h5_file_path=h5_load_path,
        groups=groups,
        output_path=Path(save_folder, "summed_embeddings.csv"),
        walk_distance=walk_distance,
        amount_of_walks=amount_of_walks,
    )
