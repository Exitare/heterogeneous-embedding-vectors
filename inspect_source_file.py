import h5py
from pathlib import Path

with h5py.File('results/embeddings/BRCA_LUAD_STAD_BLCA_COAD_THCA.h5', 'r') as f:
    for key in f.keys():
        print(key)

    print(f["annotations"]["embeddings"].chunks)
    print(len(f["rna"]["submitter_id"][:]))



with h5py.File(Path("results", "classifier", "summed_embeddings","BRCA_LUAD_STAD_BLCA_COAD_THCA","3_3", "summed_embeddings.h5")) as f:
    print("Classifier file:")
    for key in f.keys():
        print(key)

    print(len(f["y"][:]))
