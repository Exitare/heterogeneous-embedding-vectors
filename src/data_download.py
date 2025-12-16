import h5py
import numpy as np
from pathlib import Path
import requests


def download_h5(url: str, output_path: Path, file_name: str):
    """
    Download an HDF5 file from a given URL and save it to the specified output path.
    """

    response = requests.get(url, stream=True)
    response.raise_for_status()  # Ensure we notice bad responses

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(f"{output_path}/{file_name}", 'wb') as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)

    print(f"Downloaded HDF5 file to {output_path}")


def construct_h5(output_path: Path,
                 source_dir: Path):
    """
    Construct an HDF5 file from separate modality files and saves it to results/embeddings.
    Reverses the deconstruction process by combining split image files and other modalities.
    """
    source_dir = Path(source_dir)
    output_path = Path(output_path)

    # Create output directory if it doesn't exist
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Reconstructing H5 file to {output_path}")

    with h5py.File(output_path, 'w') as out_h5file:
        # Handle images - combine 4 parts
        print("Reconstructing images from 4 parts...")
        image_parts = sorted(source_dir.glob("images_embeddings_part*.h5"))

        if image_parts:
            # Read all parts to get total size
            all_embeddings = []
            all_cancer = []
            all_submitter_id = []

            for i, part_file in enumerate(image_parts):
                print(f"Loading {part_file.name}...")
                with h5py.File(part_file, 'r') as part_h5:
                    images_group = part_h5['images']
                    all_embeddings.append(images_group['embeddings'][:])
                    all_cancer.append(images_group['cancer'][:])
                    all_submitter_id.append(images_group['submitter_id'][:])

            # Combine all parts
            combined_embeddings = np.vstack(all_embeddings)
            combined_cancer = np.concatenate(all_cancer)
            combined_submitter_id = np.concatenate(all_submitter_id)

            # Create images group
            img_group = out_h5file.create_group('images')
            img_group.create_dataset('embeddings', data=combined_embeddings)
            img_group.create_dataset('cancer', data=combined_cancer)
            img_group.create_dataset('submitter_id', data=combined_submitter_id)

            print(f"Combined {len(image_parts)} image parts into {combined_embeddings.shape[0]} rows")

        # Handle other modalities - copy directly
        for modality in ['rna', 'annotations', 'mutations', 'indices']:
            modality_file = source_dir / f"{modality}_embeddings.h5"

            if modality_file.exists():
                print(f"Copying {modality}...")
                with h5py.File(modality_file, 'r') as modality_h5:
                    if modality in modality_h5:
                        modality_h5.copy(modality, out_h5file, name=modality)
                        print(f"Copied {modality} successfully")
                    else:
                        print(f"Warning: '{modality}' group not found in {modality_file}")
            else:
                print(f"Warning: {modality_file} not found, skipping...")

    print(f"\nReconstruction complete! File saved to {output_path}")
    print(f"Available groups: {list(h5py.File(output_path, 'r').keys())}")


def download_data(url: str, output_path: Path, file_name: str):
    """
    Download cancer sub-type data and save it to the specified output path.
    """
    response = requests.get(url)
    response.raise_for_status()

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(f"{output_path}/{file_name}", 'wb') as f:
        f.write(response.content)

    print(f"Downloaded cancer sub-type data to {output_path}")


if __name__ == "__main__":
    print("Downloading embedding files...")

    results_embeddings_save_path: Path = Path(f"results/embeddings")
    if not results_embeddings_save_path.exists():
        results_embeddings_save_path.mkdir(parents=True)

    download_h5("https://dataverse.harvard.edu/api/access/datafile/13244566", output_path=results_embeddings_save_path, file_name="rna_embeddings.h5")
    download_h5("https://dataverse.harvard.edu/api/access/datafile/13244571", output_path=results_embeddings_save_path, file_name="mutations_embeddings.h5")
    download_h5("https://dataverse.harvard.edu/api/access/datafile/13244572", output_path=results_embeddings_save_path, file_name="indices_embeddings.h5")
    download_h5("https://dataverse.harvard.edu/api/access/datafile/13244574", output_path=results_embeddings_save_path, file_name="annotations_embeddings.h5")
    download_h5("https://dataverse.harvard.edu/api/access/datafile/13244568", output_path=results_embeddings_save_path, file_name="images_embeddings_part1.h5")
    download_h5("https://dataverse.harvard.edu/api/access/datafile/13244564", output_path=results_embeddings_save_path, file_name="images_embeddings_part2.h5")
    download_h5("https://dataverse.harvard.edu/api/access/datafile/13244565", output_path=results_embeddings_save_path, file_name="images_embeddings_part3.h5")
    download_h5("https://dataverse.harvard.edu/api/access/datafile/13244575", output_path=results_embeddings_save_path, file_name="images_embeddings_part4.h5")

    download_data("https://dataverse.harvard.edu/api/access/datafile/13244573", output_path=results_embeddings_save_path, file_name="lookup.csv")

    print("\nReconstructing H5 file from downloaded parts...")
    construct_h5(output_path=Path(f"{results_embeddings_save_path}/BRCA_LUAD_STAD_BLCA_COAD_THCA.h5"),
                 source_dir=results_embeddings_save_path)

    data_folder: Path = Path("data")
    if not data_folder.exists():
        data_folder.mkdir(parents=True)

    download_data("https://dataverse.harvard.edu/api/access/datafile/13244553", output_path=data_folder, file_name="cancer2name.json")

    print("\nDownloading cancer sub-type data...")
    cancer_subtype_save_path: Path = Path("data", "cancer_sub_types")
    if not cancer_subtype_save_path.exists():
        cancer_subtype_save_path.mkdir(parents=True)

    download_data("https://dataverse.harvard.edu/api/access/datafile/13244562", output_path=cancer_subtype_save_path, file_name="BLCA.tsv")
    download_data("https://dataverse.harvard.edu/api/access/datafile/13244570", output_path=cancer_subtype_save_path, file_name="BRCA.tsv")
    download_data("https://dataverse.harvard.edu/api/access/datafile/13244576", output_path=cancer_subtype_save_path, file_name="COAD.tsv")
    download_data("https://dataverse.harvard.edu/api/access/datafile/13244569", output_path=cancer_subtype_save_path, file_name="LUAD.tsv")
    download_data("https://dataverse.harvard.edu/api/access/datafile/13244563", output_path=cancer_subtype_save_path, file_name="THCA.tsv")

    print("\nDownloading mutation data...")

    mutations_save_path: Path = Path("data", "mutations")
    if not mutations_save_path.exists():
        cancer_subtype_save_path.mkdir(parents=True)

    download_data("https://dataverse.harvard.edu/api/access/datafile/13244567", output_path=mutations_save_path, file_name="gene_mutations.tsv")

    print("\nAll downloads and reconstruction complete!")
