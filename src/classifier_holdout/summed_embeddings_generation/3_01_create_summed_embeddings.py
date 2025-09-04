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


def write_h5_data(summed_embeddings_data: List[Tuple[np.ndarray, str, str]], output_file, walk_amount, selected_cancers):
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


def sum_random_embeddings(patient_id: str, patient_data: Dict[str, np.ndarray], walk_distance: int,
                          walk_amount: int, modalities: List[str]) -> np.ndarray:
    """
    Creates summed embeddings by randomly selecting embeddings from available modalities.
    """
    available_modalities = [
        key for key in modalities
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
    # load the submitter ids from the RNA
    if "rna" in modalities:
        submitter_ids.update(h5_file["rna"].keys())
        logging.info(f"RNA modality: {len(submitter_ids)} submitter_ids collected.")
        return list(submitter_ids)
    else:
        raise ValueError("RNA modality is required for processing.")


def main():
    parser = argparse.ArgumentParser(description="Process and sum patient embeddings.")
    parser.add_argument(
        "--cancer", "-c", nargs="+", required=False,
        help="The cancer types to work with.",
        default=["BRCA", "LUAD", "STAD", "BLCA", "COAD", "THCA"]
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
    parser.add_argument("--modalities", "-m", nargs="+", default=["rna", "annotations", "mutations", "images"],
                        help="Modalities to include in the summing process.",
                        choices=["rna", "annotations", "mutations", "images"])
    args = parser.parse_args()

    selected_cancers: List[str] = args.cancer
    walk_distance: int = args.walk_distance
    walk_amount: int = args.amount_of_walks
    selected_modalities: List[str] = args.modalities

    if len(selected_modalities) < 2:
        raise ValueError("At least two modalities must be selected for summing embeddings.")

    if len(selected_cancers) == 1:
        logging.info(f"Selected cancers {selected_cancers} is a single string with spaces. Splitting into list...")
        selected_cancers = selected_cancers[0].split(" ")

    cancers: str = "_".join(selected_cancers)
    load_path: Path = Path(args.load_path)

    save_folder = Path("results", "classifier_holdout", "summed_embeddings", cancers, '_'.join(selected_modalities),
                       f"{walk_distance}_{walk_amount}")
    save_folder.mkdir(parents=True, exist_ok=True)

    h5_load_path: Path = Path(load_path, f"{cancers}_classifier.h5")
    train_output_file: Path = Path(save_folder, "train_summed_embeddings.h5")
    test_output_file: Path = Path(save_folder, "test_summed_embeddings.h5")

    logging.info(f"Loading embeddings from {h5_load_path}...")
    logging.info(f"Selected cancers: {selected_cancers}")
    logging.info(f"Walk distance: {walk_distance}")
    logging.info(f"Amount of walks: {walk_amount}")
    logging.info(f"Saving train summed embeddings to {train_output_file}")
    logging.info(f"Saving test summed embeddings to {test_output_file}")
    logging.info(f"Using modalities: {selected_modalities}")

    with h5py.File(h5_load_path, "r") as h5_file:
        # ✅ Preload all submitter_ids from all relevant modalities into memory
        modalities = ["rna", "annotations", "mutations", "images"]
        patient_ids = collect_all_submitter_ids(h5_file, modalities)

        logging.info(f"Total unique patients to process: {len(patient_ids)}")

        summed_train_embeddings_data = []
        summed_test_embeddings_data = []
        patient_count = 0

        # extract 80% of patients for training
        np.random.seed(42)
        np.random.shuffle(patient_ids)
        train_patient_ids = patient_ids[:int(0.8 * len(patient_ids))]
        test_patient_ids = patient_ids[int(0.8 * len(patient_ids)):]

        for patient_id in tqdm(train_patient_ids, desc="Processing Patients"):
            try:
                patient_data, patient_cancer = load_patient_data(h5_file, patient_id)
            except ValueError as e:
                logging.warning(f"Skipping {patient_id}: {e}")
                continue

            try:
                summed_embedding = sum_random_embeddings(
                    patient_id, patient_data, walk_distance, walk_amount,
                    modalities=selected_modalities
                )
                summed_train_embeddings_data.append((summed_embedding, patient_cancer, patient_id))
                patient_count += 1
            except ValueError as e:
                logging.warning(f"Skipping {patient_id}: {e}")
                continue

        for patient_id in tqdm(test_patient_ids, desc="Processing Patients"):
            try:
                patient_data, patient_cancer = load_patient_data(h5_file, patient_id)
            except ValueError as e:
                logging.warning(f"Skipping {patient_id}: {e}")
                continue

            try:
                summed_embedding = sum_random_embeddings(
                    patient_id, patient_data, walk_distance, walk_amount,
                    modalities=selected_modalities
                )
                summed_test_embeddings_data.append((summed_embedding, patient_cancer, patient_id))
                patient_count += 1
            except ValueError as e:
                logging.warning(f"Skipping {patient_id}: {e}")
                continue

    logging.info(f"Processed {patient_count} patients with valid embeddings.")

    write_h5_data(summed_train_embeddings_data, train_output_file, walk_amount, selected_cancers)
    write_h5_data(summed_test_embeddings_data, test_output_file, walk_amount, selected_cancers)

    logging.info(f"✅ Summed embeddings saved to {train_output_file} and {test_output_file}")
    logging.info(f"Total patients processed and saved: {patient_count}")


if __name__ == "__main__":
    main()
