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
                    logging.info(f"Chunk shape: {chunk.shape}")
                    yield chunk
    elif path.is_file():
        logging.info(f"Loading single file in chunks: {path}")
        for chunk in pd.read_csv(path, chunksize=chunk_size):
            logging.info(f"Chunk shape: {chunk.shape}")
            yield chunk
    else:
        raise ValueError(f"Provided path is neither a file nor a directory: {path}")


def chunked_image_dataframe_loader(path, chunk_size=10000, file_extension=".tsv"):
    """
    Load image data from a directory containing cancer-specific subdirectories.

    Parameters:
        path (Path): Path to the image data directory.
        chunk_size (int): Number of rows per chunk.
        file_extension (str): Extension of image data files (default: .csv).

    Yields:
        pd.DataFrame: Chunk of data from the cancer-specific subdirectories.
    """
    path = Path(path)

    if path.is_dir():
        for file_path in path.iterdir():
            if file_path.is_file():
                continue  # Skip files at the top level
            for cancer_path in file_path.iterdir():
                if cancer_path.is_file() and cancer_path.suffix == file_extension:
                    logging.info(f"Loading file in chunks: {cancer_path}")
                    sep = "," if file_extension == ".csv" else "\t"
                    for chunk in pd.read_csv(cancer_path, chunksize=chunk_size, sep=sep):
                        logging.info(f"Chunk shape: {chunk.shape}")
                        yield chunk
    elif path.is_file():
        logging.info(f"Loading single file in chunks: {path}")
        for chunk in pd.read_csv(path, chunksize=chunk_size):
            yield chunk
    else:
        raise ValueError(f"Provided path is neither a file nor a directory: {path}")


def process_and_store_per_submitter(dataset_name, loader, h5_file):
    """
    Process data and store embeddings for each submitter as separate datasets.
    """
    logging.info(f"Processing {dataset_name}...")

    group = h5_file.create_group(dataset_name)

    for chunk in loader:
        # Extract submitter IDs
        submitter_ids = chunk["submitter_id"].astype(str).tolist()
        submitter_ids = ['-'.join(sid.split("-")[:3]) for sid in submitter_ids]  # Replace hyphens with underscores

        # Extract only numeric columns (skip 'cancer' and 'submitter_id')
        numeric_cols = [col for col in chunk.columns if col not in ["cancer", "submitter_id", "cancer_type", "tile_pos"]]
        numeric_data = chunk[numeric_cols].to_numpy(dtype=np.float32)

        # Extract cancer labels
        cancer_values = chunk["cancer_type"].astype(str).tolist() if dataset_name == "images" else chunk["cancer"].astype(str).tolist()

        for i, submitter_id in enumerate(submitter_ids):
            # Ensure dataset does not already exist
            if submitter_id in group:
                dataset = group[submitter_id]
                dataset.resize((dataset.shape[0] + 1), axis=0)
                dataset[-1] = numeric_data[i]
            else:
                # Create new dataset for the submitter
                dataset = group.create_dataset(
                    submitter_id,
                    data=numeric_data[i].reshape(1, -1),  # 2D array (1, features)
                    maxshape=(None, numeric_data.shape[1]),
                    dtype="float32",
                    chunks=True,  # Enable chunking for efficiency
                    compression="gzip"
                )
                # Store cancer type as an attribute
                # cancer has TCGA prefix remove it
                if cancer_values[i].startswith("TCGA-"):
                    cancer_values[i] = cancer_values[i][5:]

                dataset.attrs["cancer"] = cancer_values[i]

                assert "TCGA-" not in dataset.attrs["cancer"], f"❌ Cancer type still has TCGA prefix: {dataset.attrs['cancer']}"

    logging.info(f"✅ Completed processing {dataset_name}.")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--cancers", "-c", nargs="+", required=True, help="The cancer types to work with.")
    args = parser.parse_args()
    selected_cancers = args.cancers
    cancers = "_".join(selected_cancers)

    rna_load_folder = Path("results", "embeddings", "rna", cancers)
    annotation_embedding_file = Path("results", "embeddings", "annotations", cancers, "embeddings.csv")
    mutation_embedding_file = Path("results", "embeddings", "mutation_embeddings.csv")
    image_embedding_folder = Path("results", "embeddings", "images")

    try:
        with h5py.File(Path(save_folder, f"{cancers}_classifier.h5"), "w") as f:
            # Process RNA embeddings
            rna_loader = chunked_dataframe_loader(rna_load_folder)
            process_and_store_per_submitter("rna", rna_loader, f)

            # Process Annotation embeddings
            annotation_loader = chunked_dataframe_loader(annotation_embedding_file)
            process_and_store_per_submitter("annotations", annotation_loader, f)

            # Process Mutation embeddings
            mutation_loader = chunked_dataframe_loader(mutation_embedding_file)
            process_and_store_per_submitter("mutations", mutation_loader, f)

            # Process Image embeddings (chunked)
            image_loader = chunked_image_dataframe_loader(image_embedding_folder, chunk_size=chunk_size, file_extension=".tsv")
            process_and_store_per_submitter("images", image_loader, f)


            logging.info("✅ HDF5 file with submitter-specific datasets created successfully.")
            logging.info(f"Available groups: {list(f.keys())}")
            logging.info(f"Available submitters: {len(list(f['rna'].keys()))}")
            logging.info(f"File saved to: {Path(save_folder, f'{cancers}_classifier.h5')}")

    except Exception as e:
        logging.info(f"Error occurred: {e}")
        with open("error.txt", "w") as f:
            f.write(str(e))
        sys.exit(1)
