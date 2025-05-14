from pathlib import Path
import pandas as pd
from collections import Counter
import numpy as np

cancers =  ["BRCA", "LUAD", "STAD", "BLCA", "COAD", "THCA"]

if __name__ == '__main__':
    submitter_ids = {}
    for cancer in cancers:
        df = pd.read_csv(Path("data","rna", cancer,"data.csv"), index_col= 0)
        print(df)
        df["submitter_id"] = df["Patient"].apply(lambda x: '-'.join(x.split('-')[:3]))
        submitter_ids[cancer] = df["submitter_id"].values

    all_submitter_ids = np.concatenate(list(submitter_ids.values()))
    print(len(list(set(all_submitter_ids))))
    input()

    annotation_embeddings = pd.read_csv(
        Path("results", "embeddings", "annotations", "BRCA_LUAD_STAD_BLCA_COAD_THCA", "embeddings.csv"))
    # print(annotation_embeddings.head())
    print(f"Annotation patient: {annotation_embeddings['submitter_id'].nunique()}")
    print(f"Annotations per cancer type: {annotation_embeddings['cancer'].value_counts()}")

    rna_embeddings = {}
    for cancer in cancers:
        df = pd.read_csv(
            Path("results", "embeddings", "rna", "BRCA_LUAD_STAD_BLCA_COAD_THCA", f"{cancer}_embeddings.csv"))

        rna_embeddings[cancer] = df["submitter_id"].nunique()

    print(f"RNA embeddings: {sum(Counter(rna_embeddings).values())}")
    print(f"RNA embeddings per cancer type: {rna_embeddings}")



    image_embeddings = pd.read_csv(
        Path("results", "embeddings", "combined_image_embeddings.tsv"), sep='\t')
    print(f"Image patient: {image_embeddings['submitter_id'].nunique()}")
    print(f"Images per cancer type: {image_embeddings['cancer_type'].value_counts()}")

    mutation_embeddings = pd.read_csv(
        Path("results", "embeddings",  "mutation_embeddings.csv"))
    print(f"Mutation patient: {mutation_embeddings['submitter_id'].nunique()}")
    print(f"Mutations per cancer type: {mutation_embeddings['cancer'].value_counts()}")



