from argparse import ArgumentParser
from pathlib import Path
import logging
import pandas as pd

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Dictionary to store final embeddings
cancer_embeddings = {}
# Dictionary to track the number of rows added per submitter_id
submitter_id_counts = {}

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--embedding_count", "-ec", help="The total count of embeddings per patient.", type=int, default=20)
    args = parser.parse_args()

    embedding_count: int = args.embedding_count

    file_name: str = f"combined_image_embeddings_{embedding_count}.tsv"
    logging.info(f"Combining embeddings with a total count of {embedding_count} per patient...")
    logging.info(f"File name: {file_name}")

    for file_path in Path("results", "embeddings", "images").iterdir():
        if file_path.is_file():
            continue  # Skip files at the top level

        for cancer_embeddings_file in file_path.iterdir():
            if cancer_embeddings_file.is_file() and cancer_embeddings_file.suffix == '.tsv':
                logging.info(f"Loading file: {cancer_embeddings_file}")
                df = pd.read_csv(cancer_embeddings_file, sep='\t')

                # Ensure 'submitter_id' exists in the dataframe
                if 'submitter_id' not in df.columns:
                    logging.warning(f"'submitter_id' column not found in {cancer_embeddings_file}, skipping...")
                    continue

                # Track global counts
                df["current_count"] = df["submitter_id"].map(lambda x: submitter_id_counts.get(x, 0))
                df["cumsum"] = df.groupby("submitter_id").cumcount() + 1  # Row index within each submitter_id
                df = df[df["current_count"] + df["cumsum"] <= embedding_count]  # Only keep rows where total count ≤ 20

                # Update global submitter_id counts
                new_counts = df["submitter_id"].value_counts().to_dict()
                for sid, count in new_counts.items():
                    submitter_id_counts[sid] = submitter_id_counts.get(sid, 0) + count

                # Store the filtered DataFrame
                df.drop(columns=["current_count", "cumsum"], inplace=True)  # Remove temporary columns
                if not df.empty:
                    if cancer_embeddings_file.stem not in cancer_embeddings:
                        cancer_embeddings[cancer_embeddings_file.stem] = df
                    else:
                        cancer_embeddings[cancer_embeddings_file.stem] = pd.concat(
                            [cancer_embeddings[cancer_embeddings_file.stem], df],
                            ignore_index=True
                        )

    # combine the dataframes into one dataframe and save it
    final_df = pd.concat(cancer_embeddings.values(), ignore_index=True)
    # remove TCGA- from the cancer type
    final_df["cancer_type"] = final_df["cancer_type"].str.replace("TCGA-", "")
    try:
        final_df.to_csv(Path("results", "embeddings", file_name), sep='\t', index=False)
    except PermissionError as e:
        logging.error(f"Error saving combined embeddings: {e}")
        logging.info("Saving combined embeddings to 'combined_embeddings.csv' instead...")
        final_df.to_csv(Path("results", file_name), sep='\t', index=False)
    except Exception as e:
        logging.error(f"An error occurred: {e}")
        logging.info("Saving combined embeddings to 'combined_embeddings.csv' instead...")
        final_df.to_csv(Path("results", file_name), sep='\t', index=False)
