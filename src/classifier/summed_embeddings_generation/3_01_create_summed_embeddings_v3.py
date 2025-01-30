import logging

import h5py
from pathlib import Path
import argparse
import numpy as np
from tqdm import tqdm
from typing import Dict, List

LATENT_DIM = 768
CHUNK_SIZE = 100000  # For processing large image datasets

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def sum_random_embeddings(patient_id: str, patient_data: Dict[str, np.ndarray], walk_distance: int,
                          walk_amount: int) -> np.ndarray:
    """
    Creates summed embeddings by randomly selecting embeddings from available modalities.
    """
    available_modalities = [key for key in ["rna", "annotations", "mutations", "images"] if
                            key in patient_data and patient_data[key] is not None]

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


def load_patient_data(h5_file, patient_id: str) -> (Dict[str, np.ndarray], str):
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

    assert cancer_type, f"No cancer type found for patient {patient_id} in any group."

    assert len(patient_data) >= 2, f"Patient {patient_id} has less than 2 valid embeddings."

    return patient_data, cancer_type


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cancer", "-c", nargs="+", required=True, help="The cancer types to work with.")
    parser.add_argument("--walk_distance", "-w", type=int, required=True, help="The walk distance.",
                        choices=[3, 4, 5, 6],
                        default=3)
    parser.add_argument("--amount_of_walks", "-a", type=int, required=True, help="The amount of walks.",
                        choices=[3, 4, 5, 6], default=3)
    parser.add_argument("--load_path", "-l", type=str, default="results/embeddings",
                        help="Path to the embeddings folder")
    args = parser.parse_args()

    selected_cancers: List[str] = args.cancer
    walk_distance: int = args.walk_distance
    walk_amount: int = args.amount_of_walks

    if len(selected_cancers) == 1:
        logging.info("Selected cancers is a single string. Converting...")
        selected_cancers = selected_cancers[0].split(" ")

    cancers: str = "_".join(selected_cancers)
    load_path: Path = Path(args.load_path)

    save_folder = Path("results", "classifier", "summed_embeddings", cancers, f"{walk_distance}_{walk_amount}")
    save_folder.mkdir(parents=True, exist_ok=True)

    h5_load_path: Path = Path(load_path, f"{cancers}_classifier.h5")
    output_file = Path(save_folder, "summed_embeddings.h5")

    logging.info(f"Loading embeddings from {h5_load_path}...")
    logging.info(f"Selected cancers: {selected_cancers}")
    logging.info(f"Walk distance: {walk_distance}")
    logging.info(f"Amount of walks: {walk_amount}")
    logging.info(f"Saving summed embeddings to {output_file}")

    with h5py.File(h5_load_path, "r") as h5_file:
        print("✅ HDF5 Structure Loaded:", list(h5_file.keys()))

        patient_ids = list(h5_file["rna"].keys()) if "rna" in h5_file else []
        summed_embeddings_data = []
        mutation_count = 0

        for patient_id in tqdm(patient_ids, desc="Processing Patients"):
            patient_data, patient_cancer = load_patient_data(h5_file, patient_id)

            if not patient_data:
                print(f"❌ Skipping {patient_id}, no valid embeddings.")
                continue

            try:
                summed_embedding = sum_random_embeddings(patient_id, patient_data, walk_distance, walk_amount)
                summed_embeddings_data.append((summed_embedding, patient_cancer, patient_id))
                mutation_count += 1
            except ValueError:
                print(f"❌ Skipping {patient_id}, issue with summing embeddings.")
                continue

        # ✅ Save summed embeddings to HDF5
        with h5py.File(output_file, "w") as out_file:
            shape = LATENT_DIM * walk_amount
            out_file.create_dataset("X", (0, shape), maxshape=(None, shape), dtype="float32")
            out_file.create_dataset("y", (0,), maxshape=(None,), dtype=h5py.string_dtype())
            out_file.create_dataset("submitter_ids", (0,), maxshape=(None,), dtype=h5py.string_dtype())
            out_file.attrs["classes"] = selected_cancers
            out_file.attrs["feature_shape"] = LATENT_DIM * walk_amount
            for summed_embedding, cancer, submitter_id in summed_embeddings_data:
                out_file["X"].resize(out_file["X"].shape[0] + 1, axis=0)
                out_file["X"][-1] = summed_embedding
                out_file["y"].resize(out_file["y"].shape[0] + 1, axis=0)
                out_file["y"][-1] = cancer.encode("utf-8")
                out_file["submitter_ids"].resize(out_file["submitter_ids"].shape[0] + 1, axis=0)
                out_file["submitter_ids"][-1] = submitter_id.encode("utf-8")

    print(f"Processed {len(summed_embeddings_data)} patients.")
    print(f"✅ Summed embeddings saved to {output_file}")


if __name__ == "__main__":
    main()
