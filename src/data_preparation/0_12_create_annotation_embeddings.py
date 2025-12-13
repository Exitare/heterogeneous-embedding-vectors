from sentence_transformers import SentenceTransformer
from pathlib import Path
import pandas as pd
import random
import argparse

model = SentenceTransformer("all-mpnet-base-v2")
save_folder = Path("results", "embeddings", "annotations")
load_folder = Path("data", "annotations")


def sample_words_multiple_times(sentence, num_words, num_samples, submitter_id):
    if sentence is None:
        return
    # Split the sentence into a list of words
    words = sentence.split()

    # Ensure we do not sample more words than are available
    if num_words > len(words):
        words = ["No", "annotations", "available", "for", submitter_id]

    # Perform the sampling multiple times
    sampled_words_list = []
    for _ in range(num_samples):
        sampled_words = random.sample(words, num_words)
        sampled_words_list.append(sampled_words)

    return sampled_words_list


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--cancer", "-c", nargs="+", required=False, help="The cancer types to work with.",
                        default=["BRCA", "LUAD", "STAD", "BLCA", "COAD", "THCA"])
    args = parser.parse_args()

    selected_cancers = args.cancer
    # print("Selected cancers: ", selected_cancers)

    cancers = "_".join(selected_cancers)

    load_folder = Path(load_folder, cancers)
    save_folder = Path(save_folder, cancers)

    if not save_folder.exists():
        save_folder.mkdir(parents=True)

    annotations = pd.read_csv(Path(load_folder, "annotations.csv"))
    text_annotations = []

    print(annotations)

    for submitter_id in annotations["submitter_id"]:

        # split the submitter text using the .
        text = annotations[annotations["submitter_id"] == submitter_id]["text"].values[0]
        cancer = annotations[annotations["submitter_id"] == submitter_id]["cancer"].values[0]
        # count the number of dots in the text
        num_of_dots = text.count(".")

        # assert that text does not only consists of whitespace
        assert not text.isspace(), "Text should not only consist of whitespace."

        if num_of_dots == 0:
            num_words_to_sample = 5
            num_samples = 7
            texts = sample_words_multiple_times(text, num_words_to_sample, num_samples, submitter_id=submitter_id)

            # convert each list to a string
            texts = [" ".join(text) for text in texts]
        else:
            texts = text.split(".")

        # add each text to the text_df
        for text in texts:
            text_annotations.append({
                "submitter_id": submitter_id,
                "text": text,
                "cancer": cancer
            })

    if len(text_annotations) == 0:
        print("No text annotations found.")
        exit(0)

    text_annotations = pd.DataFrame(text_annotations)
    # create a list from the text column
    sentences = [sentence for sentence in text_annotations["text"]]
    # print(f"Number of sentences: {len(sentences)}")

    # print("Encoding sentences...")
    # Sentences are encoded by calling model.encode()
    print("Encoding sentences...")
    embeddings = model.encode(sentences)

    # print("Saving embeddings...")
    # save embeddings
    embeddings = pd.DataFrame(embeddings)
    embeddings["submitter_id"] = text_annotations["submitter_id"]
    embeddings["cancer"] = text_annotations["cancer"]

    # assert that submitter and cancer columns are in the embeddings df
    assert "submitter_id" in embeddings.columns, "Submitter id column not found in embeddings."
    assert "cancer" in embeddings.columns, "Cancer column not found in embeddings."

    # create df with submitter id and cancer as a lookup df
    lookup_df: pd.DataFrame = embeddings[["submitter_id", "cancer"]].drop_duplicates()
    lookup_df.to_csv(Path("results", "embeddings", "lookup.csv"), index=False)

    embeddings.to_csv(Path(save_folder, "embeddings.csv"), index=False)
