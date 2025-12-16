import h5py


def deconstruct_hdf5(file_path):
    """
    Deconstruct an HDF5 file into a nested dictionary.
    """


# open h5 file

with h5py.File("results_save/embeddings/BRCA_LUAD_STAD_BLCA_COAD_THCA.h5", 'r') as h5file:
    print(h5file.keys())
    # for each key create a separate h5 files
    for key in h5file.keys():
        print(f"Deconstructing key: {key}")
        group = h5file[key]

        # Special handling for images - split into 4 files
        if key == "images":
            embeddings_dataset = group['embeddings']
            cancer_dataset = group['cancer']
            submitter_id_dataset = group['submitter_id']
            
            total_rows = embeddings_dataset.shape[0]
            chunk_size = total_rows // 4
            
            print(f"Splitting images into 4 files with ~{chunk_size} rows each")
            
            for i in range(4):
                start_idx = i * chunk_size
                end_idx = (i + 1) * chunk_size if i < 3 else total_rows  # Last chunk gets remainder
                
                file_name = f"{key}_embeddings_part{i+1}.h5"
                print(f"Creating {file_name} with rows {start_idx} to {end_idx}")
                
                with h5py.File(f"results_save/embeddings/{file_name}", 'w') as out_h5file:
                    img_group = out_h5file.create_group(key)
                    img_group.create_dataset('embeddings', data=embeddings_dataset[start_idx:end_idx])
                    img_group.create_dataset('cancer', data=cancer_dataset[start_idx:end_idx])
                    img_group.create_dataset('submitter_id', data=submitter_id_dataset[start_idx:end_idx])
                
                print(f"Saved part {i+1} to results_save/embeddings/{file_name}")
        else:
            # For non-image data, save as single file
            file_name = f"{key}_embeddings.h5"
            with h5py.File(f"results_save/embeddings/{file_name}", 'w') as out_h5file:
                h5file.copy(group, out_h5file, name=key)
            
            print(f"Saved deconstructed data to results_save/embeddings/{file_name}")
