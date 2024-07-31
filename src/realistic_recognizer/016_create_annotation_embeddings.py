from sentence_transformers import SentenceTransformer
from pathlib import Path
import pandas as pd

model = SentenceTransformer("all-mpnet-base-v2")
save_folder = Path("results", "realistic_recognizer", "embeddings")

if __name__ == '__main__':
    if not save_folder.exists():
        save_folder.mkdir(parents=True)

    annotations = pd.read_csv(Path("data", "annotations", "annotations.csv"))

    sentences = []
    for submitter_id in annotations["submitter_id"]:
        # split the submitter text using the .
        text = annotations[annotations["submitter_id"] == submitter_id]["text"].values[0]
        texts = text.split(".")

        # add each text to the text_df
        for text in texts:
            sentences.append({
                "submitter_id": submitter_id,
                "text": text
            })

    # create a list from the text column
    sentences = [sentence["text"] for sentence in sentences]
    print(f"Number of sentences: {len(sentences)}")

    print("Encoding sentences...")
    # Sentences are encoded by calling model.encode()
    embeddings = model.encode(sentences)

    print("Saving embeddings...")
    # save embeddings
    embeddings = pd.DataFrame(embeddings)
    embeddings["submitter_id"] = annotations["submitter_id"]
    embeddings.to_csv(Path(save_folder, "annotations_embeddings.csv"), index=False)
