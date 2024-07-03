from argparse import ArgumentParser
from pathlib import Path
from collections import defaultdict
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import mean_absolute_error

load_folder = Path("results", "graph_embeddings")
results_folder = Path("results", "graph_embeddings")
embedding_size = 768
epochs = 200


def unique_nodes(predictions: np.array, composition_details: pd.DataFrame):
    extracted_unique_nodes = {}
    for i, nodes in enumerate(predictions):
        for j, node in enumerate(nodes):
            # extract the node id from the composition details
            node_id = composition_details.iloc[i, j]
            if node_id in extracted_unique_nodes:
                continue
            df = pd.DataFrame(node).T
            df['Node Id'] = node_id
            extracted_unique_nodes[node_id] = df

    # create a dataframe from the dictionary values
    extracted_nodes_df = pd.concat(extracted_unique_nodes.values(), ignore_index=True)
    return extracted_nodes_df


if __name__ == '__main__':

    parser = ArgumentParser()
    parser.add_argument("--cancer", "-c", nargs='+', required=True, help="The cancer type to work with.")

    args = parser.parse_args()

    selected_cancers = args.cancer

    cancers = "_".join(selected_cancers)

    results_folder = Path(results_folder, cancers, "disaggregated_embeddings")
    if not results_folder.exists():
        results_folder.mkdir(parents=True)

    walk_data = pd.read_csv(Path(load_folder, cancers, "graph_generation", "aggregated_walk_embeddings.csv"))
    ground_truth_nodes = pd.read_csv(Path(load_folder, cancers, "graph_generation", "ground_truth_embeddings.csv"))
    combined_embeddings = pd.read_csv(Path(load_folder, cancers, "graph_generation", "combined_embeddings.csv"))

    # Determine the number of nodes per walk (sequence length) and embedding dimension
    max_sequence_length = ground_truth_nodes.shape[1] // embedding_size

    # Convert DataFrame to NumPy array and reshape the ground truth embeddings
    ground_truth_embeddings_array = ground_truth_nodes.values
    ground_truth_embeddings_reshaped = ground_truth_embeddings_array.reshape(-1, max_sequence_length, embedding_size)

    # Split the dataset into training and testing sets
    # Split the dataset and indices into training and testing sets
    X_train, X_test, y_train, y_test, train_idx, test_idx = train_test_split(
        walk_data.values,
        ground_truth_embeddings_reshaped,
        walk_data.index.values,
        test_size=0.2,
        random_state=42
    )

    # Define the dimensions
    input_dim = X_train.shape[1]  # Dimension of aggregated embeddings
    output_dim = y_train.shape[1]  # Number of nodes per walk
    embedding_dim = y_train.shape[2]  # Dimension of each node embedding

    # Encoder Model
    encoder_inputs = tf.keras.Input(shape=(input_dim,))
    encoder = layers.Dense(512, activation='relu')(encoder_inputs)
    encoder = layers.Dense(256, activation='relu')(encoder)
    latent_space = layers.Dense(128, activation='relu')(encoder)

    # Decoder Model
    decoder_inputs = layers.RepeatVector(output_dim)(latent_space)
    decoder = layers.LSTM(256, return_sequences=True)(decoder_inputs)
    decoder_outputs = layers.TimeDistributed(layers.Dense(embedding_dim))(decoder)

    # Combine Encoder and Decoder into a Model
    model = models.Model(inputs=encoder_inputs, outputs=decoder_outputs)

    # Compile the Model
    model.compile(optimizer='adam', loss='mse')

    # Display the model architecture
    model.summary()

    # create early stopping
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=5,
        restore_best_weights=True
    )

    # Train the model
    history = model.fit(X_train, y_train, epochs=epochs, batch_size=32, validation_split=0.2, callbacks=early_stopping)
    # save history
    history_df = pd.DataFrame(history.history)
    history_df.to_csv(Path(results_folder, "history.csv"), index=False)

    # Evaluate the model
    loss = model.evaluate(X_test, y_test)
    print(f'Test Loss: {loss}')

    # predict the embeddings
    predictions = model.predict(X_test)

    composition_details = pd.read_csv(Path(load_folder, cancers, "graph_generation", "composition_details.csv"))
    extracted_nodes = unique_nodes(predictions, composition_details)

    # calculate the mae between the extracted nodes and the ground truth embeddings based on the Node Id
    mae = []
    for idx, row in extracted_nodes.iterrows():
        extracted_node = pd.DataFrame(row).T
        # only extract the embedding columns
        extracted_node = extracted_node.iloc[:, :embedding_size].values
        node_id = row['Node Id']
        original_node = combined_embeddings.loc[combined_embeddings.index == node_id]
        if original_node.size == 0:
            continue

        # extract the embedding from the ground_truth_embedding, which are only the first 768 columns
        original_node = original_node.iloc[:, :embedding_size].values

        mae.append(mean_absolute_error(extracted_node, original_node))

    composition_details = pd.read_csv(Path(load_folder, cancers, "graph_generation", "composition_details.csv"))
    unique_nodes(predictions, composition_details)

    # compare the predictions with the ground truth embedding using cosine similarity

    # for each prediction and ground truth embedding, calculate the cosine similarity
    similarities = [cosine_similarity(predictions[i], y_test[i]) for i in range(len(predictions))]
    print(f'Cosine Similarity: {np.mean(similarities)}')
    similarity = sum(similarities) / len(similarities)

    # calculate MAE for each prediction and ground truth embedding
    mae = [np.mean(np.abs(predictions[i] - y_test[i])) for i in range(len(predictions))]
    print(f'Mean Absolute Error: {np.mean(mae)}')

    # save similarity results
    similarity_df = pd.DataFrame(similarity)
    similarity_df.to_csv(Path(results_folder, "similarity.csv"), index=False)

