from sentence_transformers import SentenceTransformer
from pathlib import Path
import pandas as pd
import random

model = SentenceTransformer("all-mpnet-base-v2")
save_folder = Path("results", "realistic_recognizer", "embeddings")
load_folder = Path("data", "annotations")


def sample_words_multiple_times(sentence, num_words, num_samples):
    if sentence is None:
        return
    # Split the sentence into a list of words
    words = sentence.split()

    # Ensure we do not sample more words than are available
    if num_words > len(words):
        print(sentence)
        raise ValueError("The number of words to sample exceeds the number of words in the sentence.")

    # Perform the sampling multiple times
    sampled_words_list = []
    for _ in range(num_samples):
        sampled_words = random.sample(words, num_words)
        sampled_words_list.append(sampled_words)

    return sampled_words_list


if __name__ == '__main__':
    if not save_folder.exists():
        save_folder.mkdir(parents=True)

    annotations = pd.read_csv(Path(load_folder, "annotations.csv"))
    text_annotations = []

    for submitter_id in annotations["submitter_id"]:

        # split the submitter text using the .
        text = annotations[annotations["submitter_id"] == submitter_id]["text"].values[0]
        # count the number of dots in the text
        num_of_dots = text.count(".")

        # assert that text does not only consists of whitespace
        assert not text.isspace(), "Text should not only consist of whitespace."

        texts = []
        if num_of_dots == 0:
            num_words_to_sample = 10
            num_samples = 7
            texts = sample_words_multiple_times(text, num_words_to_sample, num_samples)

            # convert each list to a string
            texts = [" ".join(text) for text in texts]
        else:
            texts = text.split(".")

        # add each text to the text_df
        for text in texts:
            text_annotations.append({
                "submitter_id": submitter_id,
                "text": text
            })

    text_annotations = pd.DataFrame(text_annotations)
    # create a list from the text column
    sentences = [sentence for sentence in text_annotations["text"]]
    print(f"Number of sentences: {len(sentences)}")

    print("Encoding sentences...")
    # Sentences are encoded by calling model.encode()
    embeddings = model.encode(sentences)

    print("Saving embeddings...")
    # save embeddings
    embeddings = pd.DataFrame(embeddings)
    embeddings["submitter_id"] = text_annotations["submitter_id"]
    embeddings.to_csv(Path(save_folder, "annotations_embeddings.csv"), index=False)