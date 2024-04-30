from sentence_transformers import SentenceTransformer
from pathlib import Path
import pandas as pd
import argparse

#model = SentenceTransformer("all-MiniLM-L6-v2")
model = SentenceTransformer("all-mpnet-base-v2")

load_folder = Path("../../data")
save_folder = Path("../../results", "embeddings")

if not save_folder.exists():
    save_folder.mkdir(parents=True)

if __name__ == '__main__':
    # load json file
    sentences = pd.read_json(Path(load_folder, "sentences.json"))

    # convert first column to list
    sentences = sentences[sentences.columns[0]].tolist()

    # Sentences are encoded by calling model.encode()
    embeddings = model.encode(sentences)

    # save embeddings
    embeddings = pd.DataFrame(embeddings)
    embeddings.to_csv(Path(save_folder, "sentence_embeddings.csv"), index=False)
