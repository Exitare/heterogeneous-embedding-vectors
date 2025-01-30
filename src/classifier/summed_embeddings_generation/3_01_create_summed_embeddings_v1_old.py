import h5py
from pathlib import Path
import argparse
import numpy as np
from Cython.Compiler.Future import annotations
from tqdm import tqdm
from typing import Dict, List, Tuple
from concurrent.futures import ProcessPoolExecutor

LATENT_DIM = 768
CHUNK_SIZE = 100000


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

    h5_file = h5py.File(h5_load_path, "r")

    print(h5_file["images"])

    # Load submitter IDs and cancer labels
    submitter_ids = [str(sid, "utf-8") for sid in h5_file["indices"]["submitter_ids"][:]]

    cancer_types = {
        str(submitter_id, 'utf-8'): str(cancer, "utf-8")
        for submitter_id, cancer in zip(h5_file["rna"]["submitter_id"][:], h5_file["rna"]["cancer"][:])
    }

    # Dictionary to store embeddings per submitter
    submitter_data: Dict[str, Dict[str, np.ndarray]] = {}

    for submitter_id in tqdm(submitter_ids):
        rna_mask = h5_file["rna"]["submitter_id"][:] == submitter_id.encode("utf-8")
        submitter_rna = h5_file["rna"]["embeddings"][rna_mask] if np.any(rna_mask) else None

        mutation_mask = h5_file["mutations"]["submitter_id"][:] == submitter_id.encode("utf-8")
        submitter_mutations = h5_file["mutations"]["embeddings"][mutation_mask] if np.any(mutation_mask) else None

        annotation_mask = h5_file["annotations"]["submitter_id"][:] == submitter_id.encode("utf-8")
        submitter_annotations = h5_file["annotations"]["embeddings"][annotation_mask] if np.any(
            annotation_mask) else None

        patient_cancer = cancer_types[submitter_id] if submitter_id in cancer_types else None

        assert patient_cancer is not None, f"Submitter {submitter_id} has no cancer type"

        # ✅ Filter Image embeddings exactly like the other modalities
        submitter_images = None
        if "images" in h5_file:
            image_mask = h5_file["images"]["submitter_id"][:] == submitter_id.encode("utf-8")
            if np.any(image_mask):
                print(f"Loading images for {submitter_id}...")
                total_images = np.sum(image_mask)  # Count matching images
                image_chunks = []

                for start in range(0, total_images, CHUNK_SIZE):
                    end = min(start + CHUNK_SIZE, total_images)
                    image_chunks.append(h5_file["images"]["embeddings"][image_mask][start:end])

                # Concatenate chunks after loading
                submitter_images = np.concatenate(image_chunks, axis=0) if image_chunks else None

        # ✅ Store extracted data for this submitter
        submitter_data[submitter_id] = {
            "rna": submitter_rna,
            "annotations": submitter_annotations,
            "mutations": submitter_mutations,
            "cancer": patient_cancer,
            "images": submitter_images,
        }

        # ✅ Store extracted data for this submitter
        submitter_data[submitter_id] = {
            "rna": submitter_rna,
            "annotations": submitter_annotations,
            "mutations": submitter_mutations,
            "cancer": patient_cancer,
            "images": submitter_images,
        }

        # ✅ Print debug info
        print(f"Submitter {submitter_id} - Cancer Type: {patient_cancer}")
        print(f"RNA Shape: {submitter_rna.shape if submitter_rna is not None else 'N/A'}")
        print(f"Annotations Shape: {submitter_annotations.shape if submitter_annotations is not None else 'N/A'}")
        print(f"Mutations Shape: {submitter_mutations.shape if submitter_mutations is not None else 'N/A'}")
        print(f"Images Shape: {submitter_images.shape if submitter_images is not None else 'N/A'}")


if __name__ == "__main__":
    main()
