from pathlib import Path
from argparse import ArgumentParser
import pandas as pd
import numpy as np
import h5py

save_folder = Path("results", "embeddings")


def load_rna_embeddings(embeddings_path: Path) -> pd.DataFrame:
    """Load RNA embeddings from multiple CSV files into a single DataFrame."""
    rna_embeddings = []
    for file in embeddings_path.iterdir():
        if file.is_file() and file.name.endswith("_embeddings.csv"):
            print("Loading RNA embeddings from", file)
            rna_embeddings.append(pd.read_csv(file))
    return pd.concat(rna_embeddings, ignore_index=True)


def load_annotations_embeddings(embeddings_path: Path) -> pd.DataFrame:
    """Load annotation embeddings from a single CSV file."""
    return pd.read_csv(embeddings_path)


def load_image_embeddings(embeddings_path: Path, selected_cancers: []) -> pd.DataFrame:
    """Generator that yields image embeddings from TSV files."""
    for file in embeddings_path.iterdir():
        if file.is_file() and file.name.startswith("TCGA") and file.name.endswith(".tsv") and any(
                cancer in file.name for cancer in selected_cancers):
            print("Loading image embeddings from", file)
            yield pd.read_csv(file, sep="\t")


def load_mutation_embeddings(embeddings_path: Path) -> pd.DataFrame:
    """Load mutation embeddings from a single CSV file."""
    return pd.read_csv(embeddings_path)


def dataframe_to_structured_array(df: pd.DataFrame) -> np.ndarray:
    """Convert a Pandas DataFrame to a structured NumPy array."""
    dtype = []
    for col in df.columns:
        if df[col].dtype == 'object':  # String data
            dtype.append((col, h5py.string_dtype(encoding='utf-8')))
        else:  # Numeric data
            dtype.append((col, df[col].dtype))

    structured_array = np.zeros(len(df), dtype=dtype)
    for col in df.columns:
        structured_array[col] = df[col].astype(str) if df[col].dtype == 'object' else df[col]

    return structured_array


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--cancers", "-c", nargs="+", required=True, help="The cancer types to work with.")

    args = parser.parse_args()
    selected_cancers = args.cancers
    cancers = "_".join(selected_cancers)

    rna_load_folder = Path("results", "embeddings", "rna", cancers)
    annotation_embedding_file = Path("results", "embeddings", "annotations", cancers, "embeddings.csv")
    mutation_embedding_file = Path("results", "embeddings", "mutation_embeddings.csv")
    image_embedding_folder = Path("results", "embeddings", "images")

    with h5py.File(Path(save_folder, f"{cancers}.h5"), "w") as f:
        # RNA embeddings
        rna_group = f.create_group("rna")
        rna_data = load_rna_embeddings(rna_load_folder)
        rna_structured = dataframe_to_structured_array(rna_data)
        rna_group.create_dataset("embeddings", data=rna_structured)

        # Annotation embeddings
        annotation_group = f.create_group("annotations")
        annotation_data = load_annotations_embeddings(annotation_embedding_file)
        annotation_structured = dataframe_to_structured_array(annotation_data)
        annotation_group.create_dataset("embeddings", data=annotation_structured)

        # Mutation embeddings
        mutation_group = f.create_group("mutations")
        mutation_data = load_mutation_embeddings(mutation_embedding_file)
        mutation_structured = dataframe_to_structured_array(mutation_data)
        mutation_group.create_dataset("embeddings", data=mutation_structured)

        # Image embeddings
        images_group = f.create_group("images")
        first_file = next(load_image_embeddings(image_embedding_folder, selected_cancers))
        num_columns = first_file.shape[1]
        dtype = [
            (col, h5py.string_dtype(encoding='utf-8') if first_file[col].dtype == 'object' else 'f8')
            for col in first_file.columns
        ]

        # Create a structured dataset with an unlimited first dimension
        image_dataset = images_group.create_dataset(
            "embeddings",
            shape=(0,),  # Start with zero rows
            maxshape=(None,),  # Unlimited rows
            dtype=dtype,
            chunks=True
        )

        # Incrementally write image embeddings
        for chunk in load_image_embeddings(image_embedding_folder, selected_cancers):
            structured_chunk = dataframe_to_structured_array(chunk)
            current_size = image_dataset.shape[0]
            new_size = current_size + structured_chunk.shape[0]
            image_dataset.resize(new_size, axis=0)
            image_dataset[current_size:new_size] = structured_chunk

        print("HDF5 file created successfully.")
