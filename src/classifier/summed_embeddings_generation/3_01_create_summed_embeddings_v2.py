import h5py
from pathlib import Path
import argparse
import numpy as np
from tqdm import tqdm
from typing import Dict, List

LATENT_DIM = 768
CHUNK_SIZE = 100000  # For processing large image datasets


def sum_random_embeddings(submitter_id: str, submitter_data: Dict[str, np.ndarray], walk_distance: int,
                          walk_amount: int) -> np.ndarray:
    """
    Creates summed embeddings by randomly selecting embeddings from available modalities.

    Parameters:
        submitter_data (Dict[str, np.ndarray]): Dictionary containing embeddings for each submitter.
        walk_distance (int): Number of embeddings to sum in each walk.
        walk_amount (int): Number of summed embeddings to generate.

    Returns:
        np.ndarray: Concatenated summed embeddings.
    """

    submitter_data = submitter_data[submitter_id]
    available_modalities = [key for key in ["rna", "annotations", "mutations", "images"] if
                            submitter_data.get(key) is not None]

    if not available_modalities:
        raise ValueError("No valid embeddings found for submitter.")

    summed_embeddings = []

    for _ in range(walk_amount):
        selected_embeddings = []

        for _ in range(walk_distance):
            modality = np.random.choice(available_modalities)
            modality_data = submitter_data[modality]

            if modality_data.ndim > 1:  # Multiple embeddings exist for this submitter
                selected_embedding = modality_data[np.random.randint(modality_data.shape[0])]
            else:
                selected_embedding = modality_data  # Only a single embedding available

            selected_embeddings.append(selected_embedding)

        summed_embeddings.append(np.sum(selected_embeddings, axis=0))

    return np.concatenate(summed_embeddings, axis=0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cancer", "-c", nargs="+", required=True, help="The cancer types to work with.")
    parser.add_argument("--walk_distance", "-w", type=int, required=True, help="The walk distance.",
                        choices=[3, 4, 5, 6], default=3)
    parser.add_argument("--amount_of_walks", "-a", type=int, required=True, help="The amount of walks.",
                        choices=[3, 4, 5, 6], default=3)
    args = parser.parse_args()

    selected_cancers: List[str] = args.cancer
    walk_distance: int = args.walk_distance
    walk_amount: int = args.amount_of_walks
    cancers: str = "_".join(selected_cancers)

    save_folder = Path("results", "classifier", "summed_embeddings", cancers, f"{walk_distance}_{walk_amount}")
    save_folder.mkdir(parents=True, exist_ok=True)

    h5_load_path: Path = Path("results", "embeddings", f"{cancers}.h5")
    output_file = Path(save_folder, "summed_embeddings.h5")

    mutation_count = 0
    with h5py.File(h5_load_path, "r") as h5_file:
        print("✅ HDF5 Structure Loaded:")
        print(list(h5_file.keys()))  # Debug: Show available groups in the HDF5 file

        # Load submitter IDs from the stored dataset
        stored_submitter_ids = h5_file["rna"]["submitter_id"][:]
        submitter_ids = [sid.decode("utf-8") for sid in stored_submitter_ids]  # Convert bytes to strings

        # Build a mapping of submitter_id → row index
        submitter_id_to_index = {sid: idx for idx, sid in enumerate(submitter_ids)}

        # Load cancer types mapped to submitter IDs
        stored_cancer_types = h5_file["rna"]["cancer"][:]
        cancer_types = {sid.decode("utf-8"): cancer.decode("utf-8") for sid, cancer in
                        zip(stored_submitter_ids, stored_cancer_types)}

        # Dictionary to store extracted submitter data
        submitter_data: Dict[str, Dict[str, np.ndarray]] = {}
        summed_embeddings_data = []

        for submitter_id in tqdm(submitter_ids, desc="Processing Submitters"):
            if submitter_id not in submitter_id_to_index:
                print(f"⚠️ Skipping {submitter_id}, no matching data found!")
                continue

            idx = submitter_id_to_index[submitter_id]  # Get row index

            # ✅ Extract data by using the index safely
            submitter_rna = h5_file["rna"]["embeddings"][idx] if idx < h5_file["rna"]["embeddings"].shape[0] else None
            submitter_annotations = h5_file["annotations"]["embeddings"][idx] if (
                    "annotations" in h5_file and idx < h5_file["annotations"]["embeddings"].shape[0]) else None
            submitter_mutations = h5_file["mutations"]["embeddings"][idx] if (
                    "mutations" in h5_file and idx < h5_file["mutations"]["embeddings"].shape[0]) else None

            if submitter_mutations is not None:
                mutation_count += 1

            # ✅ Get cancer type from mapping
            patient_cancer = cancer_types.get(submitter_id, "Unknown")

            assert patient_cancer is not None, f"❌ Submitter {submitter_id} has no cancer type!"

            # ✅ Image processing: Load in chunks safely
            submitter_images = None
            if "images" in h5_file and submitter_id in h5_file["images"]:
                print(f"📸 Loading images for {submitter_id}...")

                total_images = h5_file["images"][submitter_id].shape[0]
                image_chunks = []

                for start in range(0, total_images, CHUNK_SIZE):
                    end = min(start + CHUNK_SIZE, total_images)
                    image_chunks.append(h5_file["images"][submitter_id][start:end])

                # Concatenate chunks
                submitter_images = np.concatenate(image_chunks, axis=0) if image_chunks else None

            # ✅ Store extracted data for this submitter
            submitter_data[submitter_id] = {
                "rna": submitter_rna,
                "annotations": submitter_annotations,
                "mutations": submitter_mutations,
                "cancer": patient_cancer,
                "images": submitter_images,
            }

            # ✅ Print debug info
            # print(f"✔ Submitter {submitter_id} - Cancer Type: {patient_cancer}")
            # print(f"  RNA Shape: {submitter_rna.shape if submitter_rna is not None else 'N/A'}")
            # print(f"  Annotations Shape: {submitter_annotations.shape if submitter_annotations is not None else 'N/A'}")
            # print(f"  Mutations Shape: {submitter_mutations.shape if submitter_mutations is not None else 'N/A'}")
            # print(f"  Images Shape: {submitter_images.shape if submitter_images is not None else 'N/A'}")

            # ✅ Sum embeddings for this submitter
            try:
                summed_embedding = sum_random_embeddings(submitter_id, submitter_data, walk_distance, walk_amount)
                summed_embeddings_data.append((summed_embedding, patient_cancer, submitter_id))
            except ValueError:
                print(f"❌ Skipping {submitter_id}, no valid embeddings for sum operation.")
                continue

        print(mutation_count)
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


if __name__ == "__main__":
    main()
