import logging
import h5py
from pathlib import Path
import argparse
import numpy as np
from tqdm import tqdm
from typing import Dict, List, Tuple, Optional

LATENT_DIM = 768
CHUNK_SIZE = 100000  # For processing large image datasets

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def sum_random_embeddings(patient_id: str, patient_data: Dict[str, np.ndarray], walk_distance: int,
                          walk_amount: int) -> np.ndarray:
    """
    Creates summed embeddings by randomly selecting embeddings from available modalities.
    """
    available_modalities = [
        key for key in ["rna", "annotations", "mutations", "images"]
        if key in patient_data and patient_data[key] is not None
    ]

    if not available_modalities:
        raise ValueError(f"No valid embeddings found for patient {patient_id}.")

    summed_embeddings = []
    for _ in range(walk_amount):
        selected_embeddings = []
        for _ in range(walk_distance):
            modality = np.random.choice(available_modalities)
            modality_data = patient_data[modality]

            if modality_data.ndim > 1:
                selected_embedding = modality_data[np.random.randint(modality_data.shape[0])]
            else:
                selected_embedding = modality_data
            selected_embeddings.append(selected_embedding)

        summed_embeddings.append(np.sum(selected_embeddings, axis=0))

    embedding = np.concatenate(summed_embeddings, axis=0)

    assert len(embedding) == LATENT_DIM * walk_amount, f"Summed embedding has incorrect shape: {embedding.shape}"
    return embedding


def load_patient_data(h5_file: h5py.File, patient_id: str) -> Tuple[Dict[str, np.ndarray], str]:
    """
    Loads patient data from the HDF5 file for all available modalities.
    """
    patient_data = {}
    cancer_type = ""
    for modality in ["rna", "annotations", "mutations", "images"]:
        if modality in h5_file and patient_id in h5_file[modality]:
            dataset = h5_file[modality][patient_id]
            # Extract cancer type BEFORE reading data into NumPy array
            if "cancer" in dataset.attrs:
                cancer_type = dataset.attrs["cancer"]
            else:
                raise ValueError(f"No cancer type found for patient {patient_id} in modality {modality}.")

            patient_data[modality] = dataset[()]  # Now safely extract data

    if not cancer_type:
        raise ValueError(f"No cancer type found for patient {patient_id} in any modality.")

    if len(patient_data) < 2:
        raise ValueError(f"Patient {patient_id} has less than 2 valid embeddings.")

    return patient_data, cancer_type


def collect_all_submitter_ids(h5_file: h5py.File, modalities: List[str]) -> List[str]:
    """
    Collects all unique submitter_ids from the specified modalities.
    """
    submitter_ids = set()
    for modality in modalities:
        if modality in h5_file:
            modality_ids = set(h5_file[modality].keys())
            submitter_ids.update(modality_ids)
            logging.info(f"Collected {len(modality_ids)} IDs from modality '{modality}'.")
        else:
            logging.warning(f"Modality '{modality}' not found in the HDF5 file.")
    logging.info(f"Total unique submitter_ids collected: {len(submitter_ids)}.")
    return list(submitter_ids)


def main():
    parser = argparse.ArgumentParser(description="Process and sum patient embeddings.")
    parser.add_argument(
        "--cancer", "-c", nargs="+", required=True,
        help="The cancer types to work with."
    )
    parser.add_argument(
        "--walk_distance", "-w", type=int, required=True,
        help="The walk distance.", choices=[3, 4, 5, 6], default=3
    )
    parser.add_argument(
        "--amount_of_walks", "-a", type=int, required=True,
        help="The amount of walks.", choices=[3, 4, 5, 6], default=3
    )
    parser.add_argument(
        "--load_path", "-l", type=str, default="results/embeddings",
        help="Path to the embeddings folder."
    )
    args = parser.parse_args()

    selected_cancers: List[str] = args.cancer
    walk_distance: int = args.walk_distance
    walk_amount: int = args.amount_of_walks

    if len(selected_cancers) == 1 and " " in selected_cancers[0]:
        logging.info("Selected cancers is a single string with spaces. Splitting into list...")
        selected_cancers = selected_cancers[0].split()

    cancers: str = "_".join(selected_cancers)
    load_path: Path = Path(args.load_path)

    save_folder = Path("results", "classifier", "summed_embeddings", cancers, f"{walk_distance}_{walk_amount}")
    save_folder.mkdir(parents=True, exist_ok=True)

    h5_load_path: Path = Path(load_path, f"{cancers}_classifier.h5")
    output_file: Path = Path(save_folder, "summed_embeddings.h5")

    logging.info(f"Loading embeddings from {h5_load_path}...")
    logging.info(f"Selected cancers: {selected_cancers}")
    logging.info(f"Walk distance: {walk_distance}")
    logging.info(f"Amount of walks: {walk_amount}")
    logging.info(f"Saving summed embeddings to {output_file}")

    with h5py.File(h5_load_path, "r") as h5_file:
        # ✅ Preload all submitter_ids from all relevant modalities into memory
        modalities = ["rna", "annotations", "mutations", "images"]
        patient_ids = collect_all_submitter_ids(h5_file, modalities)

        logging.info(f"Total unique patients to process: {len(patient_ids)}")

        summed_embeddings_data = []
        mutation_count = 0

        for patient_id in tqdm(patient_ids, desc="Processing Patients"):
            try:
                patient_data, patient_cancer = load_patient_data(h5_file, patient_id)
            except ValueError as e:
                logging.warning(f"Skipping {patient_id}: {e}")
                continue

            try:
                summed_embedding = sum_random_embeddings(
                    patient_id, patient_data, walk_distance, walk_amount
                )
                summed_embeddings_data.append((summed_embedding, patient_cancer, patient_id))
                mutation_count += 1
            except ValueError as e:
                logging.warning(f"Skipping {patient_id}: {e}")
                continue

    logging.info(f"Processed {mutation_count} patients with valid embeddings.")

    # ✅ Save summed embeddings to HDF5
    with h5py.File(output_file, "w") as out_file:
        shape = LATENT_DIM * walk_amount
        # Initialize datasets with zero rows and allow them to grow
        out_file.create_dataset("X", (0, shape), maxshape=(None, shape), dtype="float32")
        out_file.create_dataset("y", (0,), maxshape=(None,), dtype=h5py.string_dtype())
        out_file.create_dataset("submitter_ids", (0,), maxshape=(None,), dtype=h5py.string_dtype())
        out_file.attrs["classes"] = selected_cancers
        out_file.attrs["feature_shape"] = shape

        # Use batch resizing for better performance
        X_data = []
        y_data = []
        submitter_ids = []

        for summed_embedding, cancer, submitter_id in tqdm(summed_embeddings_data, desc="Saving Data"):
            X_data.append(summed_embedding)
            y_data.append(cancer.encode("utf-8"))
            submitter_ids.append(submitter_id.encode("utf-8"))

            # Write in chunks to avoid high memory usage
            if len(X_data) >= CHUNK_SIZE:
                current_size = out_file["X"].shape[0]
                out_file["X"].resize(current_size + len(X_data), axis=0)
                out_file["X"][current_size:current_size + len(X_data)] = np.array(X_data, dtype="float32")

                current_size_y = out_file["y"].shape[0]
                out_file["y"].resize(current_size_y + len(y_data), axis=0)
                out_file["y"][current_size_y:current_size_y + len(y_data)] = y_data

                current_size_id = out_file["submitter_ids"].shape[0]
                out_file["submitter_ids"].resize(current_size_id + len(submitter_ids), axis=0)
                out_file["submitter_ids"][current_size_id:current_size_id + len(submitter_ids)] = submitter_ids

                # Clear lists after writing
                X_data.clear()
                y_data.clear()
                submitter_ids.clear()

        # Write any remaining data
        if X_data:
            current_size = out_file["X"].shape[0]
            out_file["X"].resize(current_size + len(X_data), axis=0)
            out_file["X"][current_size:current_size + len(X_data)] = np.array(X_data, dtype="float32")

            current_size_y = out_file["y"].shape[0]
            out_file["y"].resize(current_size_y + len(y_data), axis=0)
            out_file["y"][current_size_y:current_size_y + len(y_data)] = y_data

            current_size_id = out_file["submitter_ids"].shape[0]
            out_file["submitter_ids"].resize(current_size_id + len(submitter_ids), axis=0)
            out_file["submitter_ids"][current_size_id:current_size_id + len(submitter_ids)] = submitter_ids

    logging.info(f"✅ Summed embeddings saved to {output_file}")
    logging.info(f"Total patients processed and saved: {len(summed_embeddings_data)}")


if __name__ == "__main__":
    main()
