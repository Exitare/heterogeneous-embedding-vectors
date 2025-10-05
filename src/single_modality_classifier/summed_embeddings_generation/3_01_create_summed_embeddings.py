import logging
import h5py
from pathlib import Path
import argparse
import numpy as np
from tqdm import tqdm
from typing import List, Tuple, Optional

LATENT_DIM = 768
CHUNK_SIZE = 100000  # For processing large datasets efficiently

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
rng = np.random.default_rng(42)  # deterministic randomness for reproducibility


def normalize_embedding(arr: np.ndarray, patient_id: str, modality: str) -> Optional[np.ndarray]:
    """
    Normalize an embedding to a 1D float32 vector of length LATENT_DIM.
    - If 2D (N x D), randomly select one row (uniformly).
    - If shape mismatches or dims are unexpected, return None.
    """
    arr = np.asarray(arr)
    if arr.ndim == 2:
        if arr.shape[1] != LATENT_DIM:
            logging.warning(f"{patient_id} [{modality}]: 2D embedding width {arr.shape[1]} != {LATENT_DIM}. Skipping.")
            return None
        idx = rng.integers(0, arr.shape[0])
        arr = arr[idx]
    elif arr.ndim != 1:
        logging.warning(f"{patient_id} [{modality}]: embedding ndim={arr.ndim} not in {{1,2}}. Skipping.")
        return None

    if arr.shape[0] != LATENT_DIM:
        logging.warning(f"{patient_id} [{modality}]: 1D length {arr.shape[0]} != {LATENT_DIM}. Skipping.")
        return None

    return arr.astype("float32", copy=False)


def load_patient_embedding_and_cancer(
    h5_file: h5py.File, patient_id: str, selected_modality: str
) -> Tuple[np.ndarray, str]:
    """
    Load a single modality embedding and the cancer type for a given patient.
    Returns (embedding_array, cancer_type).
    Raises ValueError if data is missing or malformed.
    """
    if selected_modality not in h5_file:
        raise ValueError(f"Selected modality '{selected_modality}' not found in HDF5.")

    if patient_id not in h5_file[selected_modality]:
        raise ValueError(f"Patient {patient_id} not found under modality '{selected_modality}'.")

    dataset = h5_file[selected_modality][patient_id]

    # Extract cancer type and decode if bytes
    if "cancer" not in dataset.attrs:
        raise ValueError(f"No cancer type found for patient {patient_id} in modality {selected_modality}.")
    cancer_type = dataset.attrs["cancer"]
    if isinstance(cancer_type, bytes):
        cancer_type = cancer_type.decode("utf-8")

    # Load array and normalize it to 1D length LATENT_DIM
    raw = dataset[()]
    emb = normalize_embedding(raw, patient_id, selected_modality)
    if emb is None:
        raise ValueError(f"Invalid embedding shape for patient {patient_id} in modality {selected_modality}.")

    return emb, cancer_type


def collect_all_submitter_ids(h5_file: h5py.File) -> List[str]:
    """
    Collect all submitter_ids from RNA (assumed canonical list).
    If your canonical IDs live elsewhere, change this to use that group instead.
    """
    if "rna" not in h5_file:
        raise ValueError("RNA modality is required to collect submitter IDs.")
    submitter_ids = list(h5_file["rna"].keys())
    logging.info(f"RNA modality: {len(submitter_ids)} submitter_ids collected.")
    return submitter_ids


def main():
    parser = argparse.ArgumentParser(description="Create single-modality patient embedding dataset (1×768 per patient).")
    parser.add_argument(
        "--cancer", "-c", nargs="+", required=False,
        help="Cancer types to include.",
        default=["BRCA", "LUAD", "STAD", "BLCA", "COAD", "THCA"]
    )
    parser.add_argument(
        "--load_path", "-l", type=str, default="results/embeddings",
        help="Path to the embeddings folder."
    )
    parser.add_argument(
        "--selected_modality", "-sm", type=str, choices=["rna", "annotations", "mutations", "images"],
        required=True,
        help="The single modality to extract embeddings from."
    )
    args = parser.parse_args()

    selected_cancers: List[str] = args.cancer
    if len(selected_cancers) == 1:
        logging.info("Detected a single cancer string; splitting by spaces into a list...")
        selected_cancers = selected_cancers[0].split(" ")

    selected_modality: str = args.selected_modality
    cancers: str = "_".join(selected_cancers)
    load_path: Path = Path(args.load_path)

    save_folder = Path("results", "single_modality_classifier", "summed_embeddings", cancers)
    save_folder.mkdir(parents=True, exist_ok=True)

    h5_load_path: Path = Path(load_path, f"{cancers}_classifier.h5")
    output_file: Path = Path(save_folder, f"{selected_modality}_embeddings.h5")

    logging.info(f"Loading embeddings from {h5_load_path}...")
    logging.info(f"Selected cancers: {selected_cancers}")
    logging.info(f"Selected modality: {selected_modality}")
    logging.info(f"Saving single-modality embeddings to {output_file}")

    with h5py.File(h5_load_path, "r") as h5_file:
        # Canonical patient list from RNA
        patient_ids = collect_all_submitter_ids(h5_file)
        logging.info(f"Total patients to attempt: {len(patient_ids)}")

        # Accumulate rows before chunk flushes
        X_data: List[np.ndarray] = []
        y_data: List[str] = []
        submitter_ids: List[str] = []
        kept_count = 0

        # Prepare output file and datasets
        with h5py.File(output_file, "w") as out_file:
            out_file.create_dataset("X", (0, LATENT_DIM), maxshape=(None, LATENT_DIM), dtype="float32")
            out_file.create_dataset("y", (0,), maxshape=(None,), dtype=h5py.string_dtype("utf-8"))
            out_file.create_dataset("submitter_ids", (0,), maxshape=(None,), dtype=h5py.string_dtype("utf-8"))
            out_file.attrs["classes"] = np.array(selected_cancers, dtype=h5py.string_dtype("utf-8"))
            out_file.attrs["feature_shape"] = LATENT_DIM
            out_file.attrs["modality"] = selected_modality

            for patient_id in tqdm(patient_ids, desc="Processing Patients"):
                try:
                    emb, cancer = load_patient_embedding_and_cancer(h5_file, patient_id, selected_modality)
                except ValueError as e:
                    # Skip patients that lack the selected modality or have malformed data
                    logging.debug(f"Skipping {patient_id}: {e}")
                    continue

                # Optional: filter by target cancer classes if needed
                if selected_cancers and cancer not in selected_cancers:
                    continue

                X_data.append(emb)
                y_data.append(cancer)
                submitter_ids.append(patient_id)
                kept_count += 1

                # Flush in chunks
                if len(X_data) >= CHUNK_SIZE:
                    current_size = out_file["X"].shape[0]
                    X_block = np.stack(X_data, axis=0)  # ensures homogeneity
                    out_file["X"].resize(current_size + len(X_block), axis=0)
                    out_file["X"][current_size:current_size + len(X_block)] = X_block

                    y_block = np.array(y_data, dtype=h5py.string_dtype("utf-8"))
                    current_size_y = out_file["y"].shape[0]
                    out_file["y"].resize(current_size_y + len(y_block), axis=0)
                    out_file["y"][current_size_y:current_size_y + len(y_block)] = y_block

                    id_block = np.array(submitter_ids, dtype=h5py.string_dtype("utf-8"))
                    current_size_id = out_file["submitter_ids"].shape[0]
                    out_file["submitter_ids"].resize(current_size_id + len(id_block), axis=0)
                    out_file["submitter_ids"][current_size_id:current_size_id + len(id_block)] = id_block

                    X_data.clear()
                    y_data.clear()
                    submitter_ids.clear()

            # Flush any remaining rows
            if X_data:
                current_size = out_file["X"].shape[0]
                X_block = np.stack(X_data, axis=0)
                out_file["X"].resize(current_size + len(X_block), axis=0)
                out_file["X"][current_size:current_size + len(X_block)] = X_block

                y_block = np.array(y_data, dtype=h5py.string_dtype("utf-8"))
                current_size_y = out_file["y"].shape[0]
                out_file["y"].resize(current_size_y + len(y_block), axis=0)
                out_file["y"][current_size_y:current_size_y + len(y_block)] = y_block

                id_block = np.array(submitter_ids, dtype=h5py.string_dtype("utf-8"))
                current_size_id = out_file["submitter_ids"].shape[0]
                out_file["submitter_ids"].resize(current_size_id + len(id_block), axis=0)
                out_file["submitter_ids"][current_size_id:current_size_id + len(id_block)] = id_block

    logging.info(f"✅ Wrote single-modality embeddings to {output_file}")
    logging.info(f"Total patients processed and saved: {kept_count}")


if __name__ == "__main__":
    main()