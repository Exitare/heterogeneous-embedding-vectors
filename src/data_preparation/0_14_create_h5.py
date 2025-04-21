from pathlib import Path
from argparse import ArgumentParser
import pandas as pd
import numpy as np
import h5py
import sys
import logging

save_folder = Path("results", "embeddings")
chunk_size = 100000

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def chunked_dataframe_loader(path, chunk_size=100000, file_extension=".csv"):
    """
    Load data from a directory or a single file in chunks.
    """
    path = Path(path)

    if path.is_dir():
        for file_path in path.iterdir():
            if file_path.is_file() and file_path.suffix == file_extension:
                logging.info(f"Loading file in chunks: {file_path}")
                sep = "," if file_extension == ".csv" else "\t"
                for chunk in pd.read_csv(file_path, chunksize=chunk_size, sep=sep):
                    yield chunk
    elif path.is_file():
        logging.info(f"Loading single file in chunks: {path}")
        for chunk in pd.read_csv(path, chunksize=chunk_size):
            yield chunk
    else:
        raise ValueError(f"Provided path is neither a file nor a directory: {path}")


def chunked_image_dataframe_loader(path, chunk_size=100000):
    """
    Load image data from a directory containing cancer-specific subdirectories.

    Parameters:
        path (Path): Path to the image data directory.
        chunk_size (int): Number of rows per chunk.
        file_extension (str): Extension of image data files (default: .csv).

    Yields:
        pd.DataFrame: Chunk of data from the cancer-specific subdirectories.
    """
    logging.info(f"Loading image file: {path}...")
    for chunk in pd.read_csv(path, chunksize=chunk_size, sep='\t'):
        yield chunk


def process_and_store_in_chunks(dataset_name, loader, f, chunk_size=100000):
    """
    Process data in chunks and store embeddings in the HDF5 file, while keeping metadata separate.
    """
    logging.info(f"Processing {dataset_name} in chunks...")

    group = f.create_group(dataset_name)
    dataset = None
    current_size = 0
    indices = {}

    # Prepare lists to store metadata
    cancer_list = []
    submitter_id_list = []

    for chunk in loader:
        # Extract only numeric columns (skip 'cancer' and 'submitter_id')
        numeric_cols = [col for col in chunk.columns if
                        col not in ["cancer", "submitter_id", "cancer_type", "tile_pos"]]
        numeric_data = chunk[numeric_cols].to_numpy(dtype=np.float32)

        if dataset_name == "images":
            logging.info("Loading image cancer column...")
            cancer_values = chunk["cancer_type"].to_numpy(dtype="S")  # Store as byte strings
        else:
            logging.info("Loading other cancer columns...")
            cancer_values = chunk["cancer"].to_numpy(dtype="S")  # Store as byte strings

        # Extract metadata

        submitter_values = chunk["submitter_id"].to_numpy(dtype="S")

        if dataset is None:
            # Create the dataset dynamically on the first chunk
            dataset = group.create_dataset(
                "embeddings",
                shape=(0, numeric_data.shape[1]),  # Zero rows, fixed columns
                maxshape=(None, numeric_data.shape[1]),  # Unlimited rows, fixed columns
                dtype="float32",
                chunks=(chunk_size, numeric_data.shape[1]),  # Chunking for efficiency
            )

        # Resize and append embeddings
        new_size = current_size + numeric_data.shape[0]
        dataset.resize(new_size, axis=0)
        dataset[current_size:new_size, :] = numeric_data

        # Append metadata
        cancer_list.extend(cancer_values)
        submitter_id_list.extend(submitter_values)

        # Create indices for lookup
        for i, submitter_id in enumerate(submitter_values):
            if submitter_id not in indices:
                indices[submitter_id] = []
            indices[submitter_id].append(current_size + i)

        current_size = new_size

    try:

        submitter_id_list = ['-'.join(str(submitter_id, 'utf-8').split('-')[:3]) for submitter_id in submitter_id_list]

        # Store metadata as separate datasets
        group.create_dataset("cancer", data=np.array(cancer_list, dtype="S"), compression="gzip")
        group.create_dataset("submitter_id", data=np.array(submitter_id_list, dtype="S"), compression="gzip")

        # assert that both cancer and submitter id datasets are not empty
        assert group["cancer"].shape[0] > 0, "Cancer dataset is empty."
        assert group["submitter_id"].shape[0] > 0, "Submitter ID dataset is empty."

        logging.info(f"Completed processing {dataset_name} with {current_size} rows.")
        return indices
    except Exception as e:
        logging.info(f"Error while storing metadata for {dataset_name}: {e}")
        raise


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--cancers", "-c", nargs="+", required=True, help="The cancer types to work with."
    )
    parser.add_argument("--image_embedding_count", "-ec", type=int,
                        help="The total count of image embeddings per patient.", default=-1)

    args = parser.parse_args()
    selected_cancers = args.cancers

    if len(selected_cancers) == 1:
        logging.info("Selected cancers is a single string. Converting...")
        selected_cancers = selected_cancers[0].split(" ")

    cancers = "_".join(selected_cancers)
    image_embedding_counts: int = args.image_embedding_count

    image_file_name: str = f"combined_image_embeddings_{image_embedding_counts}.tsv" if image_embedding_counts != -1 else "combined_image_embeddings.tsv"

    logging.info(f"Selected cancers: {selected_cancers}")
    logging.info(f"Image embedding count: {image_embedding_counts}")
    logging.info(f"Image file name: {image_file_name}")

    rna_load_folder = Path("results", "embeddings", "rna", cancers)
    annotation_embedding_file = Path(
        "results", "embeddings", "annotations", cancers, "embeddings.csv"
    )
    mutation_embedding_file = Path("results", "embeddings", "mutation_embeddings.csv")
    image_embedding_file = Path("results", "embeddings", image_file_name)

    try:
        with h5py.File(Path(save_folder, f"{cancers}.h5"), "w") as f:
            # Process RNA embeddings
            rna_loader = chunked_dataframe_loader(rna_load_folder)
            rna_indices = process_and_store_in_chunks("rna", rna_loader, f, chunk_size=chunk_size)

            # Process Annotation embeddings
            annotation_loader = chunked_dataframe_loader(annotation_embedding_file)
            annotation_indices = process_and_store_in_chunks("annotations", annotation_loader, f, chunk_size=chunk_size)

            # Process Mutation embeddings
            mutation_loader = chunked_dataframe_loader(mutation_embedding_file)
            mutation_indices = process_and_store_in_chunks("mutations", mutation_loader, f, chunk_size=chunk_size)

            # Process Image embeddings
            image_loader = chunked_image_dataframe_loader(image_embedding_file, chunk_size=chunk_size)
            image_indices = process_and_store_in_chunks("images", image_loader, f, chunk_size=chunk_size)

            # Store indices
            submitter_ids = list(rna_indices.keys())
            submitter_ids = ['-'.join(str(submitter_id, 'utf-8').split('-')[:3]) for submitter_id in submitter_ids]
            submitter_ids.sort()

            index_group = f.create_group("indices")
            index_group.create_dataset("submitter_ids", data=np.array(submitter_ids, dtype="S"))
            for modality, indices in [
                ("rna", rna_indices),
                ("annotations", annotation_indices),
                ("mutations", mutation_indices),
                ("images", image_indices),
            ]:
                modality_index_group = index_group.create_group(modality)
                for submitter_id, rows in indices.items():
                    modality_index_group.create_dataset(
                        submitter_id, data=np.array(rows, dtype="int64")
                    )

            logging.info("HDF5 file with indices created successfully.")
            logging.info(f"Available groups: {f.keys()}")
            logging.info(f"Saved file to: {Path(save_folder, f'{cancers}.h5')}")

    except Exception as e:
        logging.info(e)
        logging.info(f"Error occurred: {e}")
        with open("error.txt", "w") as f:
            f.write(str(e))
        sys.exit(1)
