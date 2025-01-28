import h5py

with h5py.File('results/embeddings/BRCA_LUAD_STAD_BLCA_COAD_THCA.h5', 'r') as f:
    for key in f.keys():
        print(key)

    print(f["annotations"]["embeddings"].chunks)