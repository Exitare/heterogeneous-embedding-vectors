from sentence_transformers import SentenceTransformer
from pathlib import Path
import pandas as pd

model = SentenceTransformer("all-mpnet-base-v2")
save_folder = Path("results", "embeddings")

if __name__ == '__main__':
    if not save_folder.exists():
        save_folder.mkdir(parents=True)

    annotations = pd.read_csv(Path("data", "annotations", "annotations.csv"))

    # create a list from the text column
    sentences = annotations["text"].tolist()

    # Sentences are encoded by calling model.encode()
    embeddings = model.encode(sentences)

    # save embeddings
    embeddings = pd.DataFrame(embeddings)
    embeddings["submitter_id"] = annotations["submitter_id"]
    embeddings.to_csv(Path(save_folder, "annotations_embeddings.csv"), index=False)
