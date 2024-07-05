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
from tensorflow.keras.layers import Dense, BatchNormalization, Dropout, ReLU, Input
from tensorflow.keras.models import Model

load_folder = Path("results", "graph_embeddings")
results_folder = Path("results", "graph_embeddings")
embedding_size = 768
epochs = 200


def unique_nodes(predictions: np.array, composition_details: pd.DataFrame):
    extracted_unique_nodes = {}
    for i, nodes in enumerate(predictions):
        for j, node in enumerate(nodes):
            node_id = composition_details.iloc[i, j]
            if node_id in extracted_unique_nodes:
                continue
            df = pd.DataFrame(node).T
            df['Node Id'] = node_id
            extracted_unique_nodes[node_id] = df
    extracted_nodes_df = pd.concat(extracted_unique_nodes.values(), ignore_index=True)
    return extracted_nodes_df


def build_model(input_dim, cancer_list: []):
    inputs = Input(shape=(input_dim,), name='input_layer')
    x = Dense(512, activation='relu', name='base_dense1')(inputs)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)
    x = Dense(256, activation='relu', name='base_dense2')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)
    x = Dense(128, activation='relu', name='base_dense3')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)
    x = Dense(64, activation='relu', name='base_dense4')(x)
    x = BatchNormalization()(x)

    # Increasing complexity for text data
    text_x = Dense(128, activation='relu', name='text_dense_1')(x)
    text_x = Dropout(0.2)(text_x)
    text_x = Dense(64, activation='relu', name='text_dense_2')(text_x)
    text_x = BatchNormalization()(text_x)
    text_x = Dropout(0.2)(text_x)
    text_x = Dense(32, activation='relu', name='text_dense_3')(text_x)
    text_output = Dense(1, activation=ReLU(max_value=10), name='output_text')(text_x)

    # Less complex paths for image output
    image_output = Dense(1, activation=ReLU(max_value=10), name='output_image')(x)

    # Path for RNA embeddings, including subtype classification
    rna_x = Dense(128, activation='relu', name='rna_dense_1')(x)
    rna_x = Dropout(0.2)(rna_x)
    rna_x = Dense(64, activation='relu', name='rna_dense_2')(rna_x)
    rna_x = BatchNormalization()(rna_x)
    rna_x = Dropout(0.2)(rna_x)
    rna_x = Dense(32, activation='relu', name='rna_dense_3')(rna_x)
    rna_output = Dense(1, activation=ReLU(max_value=10), name='output_rna')(rna_x)

    cancer_outputs = [Dense(1, activation=ReLU(max_value=10), name=f'output_cancer_{cancer_type}')(x) for
                      cancer_type in cancer_list]

    # Combine all outputs
    outputs = [text_output, image_output, rna_output] + cancer_outputs

    # Create model
    return Model(inputs=inputs, outputs=outputs, name='multi_output_model')


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
    composition_details = pd.read_csv(Path(load_folder, cancers, "graph_generation", "composition_details.csv"))

    max_sequence_length = ground_truth_nodes.shape[1] // embedding_size

    ground_truth_embeddings_array = ground_truth_nodes.values
    ground_truth_embeddings_reshaped = ground_truth_embeddings_array.reshape(-1, max_sequence_length, embedding_size)

    X_train, X_test, y_train, y_test, train_idx, test_idx = train_test_split(
        walk_data.values,
        ground_truth_embeddings_reshaped,
        walk_data.index.values,
        test_size=0.2,
        random_state=42
    )

    input_dim = X_train.shape[1]
    output_dim = y_train.shape[1]
    embedding_dim = y_train.shape[2]

    encoder_inputs = tf.keras.Input(shape=(input_dim,))
    encoder = layers.Dense(512, activation='relu')(encoder_inputs)
    encoder = layers.Dense(256, activation='relu')(encoder)
    latent_space = layers.Dense(128, activation='relu')(encoder)

    decoder_inputs = layers.RepeatVector(output_dim)(latent_space)
    decoder = layers.LSTM(256, return_sequences=True)(decoder_inputs)
    decoder_outputs = layers.TimeDistributed(layers.Dense(embedding_dim))(decoder)

    model = models.Model(inputs=encoder_inputs, outputs=decoder_outputs)
    model.compile(optimizer='adam', loss='mse')
    model.summary()

    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=5,
        restore_best_weights=True
    )

    history = model.fit(X_train, y_train, epochs=epochs, batch_size=32, validation_split=0.2, callbacks=early_stopping)
    history_df = pd.DataFrame(history.history)
    history_df.to_csv(Path(results_folder, "history.csv"), index=False)

    loss = model.evaluate(X_test, y_test)
    print(f'Test Loss: {loss}')

    predictions = model.predict(X_test)

    extracted_nodes = unique_nodes(predictions, composition_details)

    # TODO: CHECK THAT. Should it be nodes? is the terminology correct, we want to exclude the nodes yes, but are these the right ones?
    # create a train test split for the combined embeddings that does not include the extracted nodes in the train set
    cleaned_embeddings = combined_embeddings.loc[~combined_embeddings.index.isin(extracted_nodes['Node Id'])]

    # Random counts for demonstration; replace with actual data
    text_counts = cleaned_embeddings["Text"].values
    image_counts = cleaned_embeddings["Image"].values
    rna_counts = cleaned_embeddings["RNA"].values

    embeddings = ['Text', 'Image', 'RNA']
    # extract cancer data
    cancer_data = []
    for cancer in selected_cancers:
        cancer_data.append(cleaned_embeddings[cancer].values)
        embeddings.append(cancer)

    # get the number of subtypes
    X = cleaned_embeddings.drop(columns=embeddings).values

    # assert shape has 768 columns
    assert X.shape[1] == 768, f"Expected 768 features, got {X.shape[1]}"

    # Assuming these are the actual labels from your dataset
    y = [text_counts, image_counts, rna_counts] + cancer_data

    # Splitting the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, np.array(y).T, test_size=0.2, random_state=42)

    # Set up a list of metrics
    loss = {'output_text': 'mae', 'output_image': 'mae', 'output_rna': 'mae'}
    loss_weights = {'output_text': 3.0, 'output_image': 1., 'output_rna': 1.}
    metrics = ['mae', 'mae', 'mae']

    # Adding dynamic metrics for cancer outputs based on their number
    for i in selected_cancers:  # Assuming num_cancer_types is defined
        loss[f'output_cancer_{i}'] = 'mae'
        loss_weights[f'output_cancer_{i}'] = 1.
        metrics.append('mae')

    # create the recognizer model
    recognizer_model = build_model(embedding_size, selected_cancers)
    recognizer_model.compile(optimizer='adam', loss=loss, loss_weights=loss_weights, metrics=metrics)
    recognizer_model.summary()

    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=5,
        restore_best_weights=True
    )

    history = recognizer_model.fit(X_train, [y_train[:, i] for i in range(y_train.shape[1])], epochs=epochs,
                                   batch_size=32, validation_split=0.2,
                                   callbacks=early_stopping)
    recognizer_history = pd.DataFrame(history.history)
    recognizer_history.to_csv(Path(results_folder, "recognizer_history.csv"), index=False)

    mae_list = []
    for idx, row in extracted_nodes.iterrows():
        extracted_node = pd.DataFrame(row).T.iloc[:, :embedding_size]
        node_id = row['Node Id']
        original_node = combined_embeddings.loc[combined_embeddings.index == node_id]
        print(f'MAE: {mean_absolute_error(extracted_node, original_node.iloc[:, :embedding_size])}')
        print(f"Cosing Similarity: {cosine_similarity(extracted_node, original_node.iloc[:, :embedding_size])}")

        y_true = original_node.iloc[:, embedding_size:]
        # reorder based on embeddings
        y_true = y_true[embeddings]
        print(y_true)

        # recognize the node composition
        y_pred = recognizer_model.predict(extracted_node)
        # Convert predictions to rounded integers
        y_pred_rounded = [np.rint(pred) for pred in y_pred]

        # Convert y_pred_rounded to a DataFrame
        predictions = [pred[0][0] for pred in y_pred_rounded]  # Extract the predictions
        y_pred_rounded = pd.DataFrame([predictions], columns=y_true.columns, index=y_true.index)


        # Calculating the accuracy for each dataset
        accuracy = {col: np.mean(y_true[col] == y_pred_rounded[col]) for col in y_true.columns}

        # convert y_pred_rounded into a dataframe
        # Convert to a flat list
        y_pred_rounded = [item[0][0] for item in y_pred_rounded]

        # Convert to DataFrame
        y_pred_rounded = pd.DataFrame([y_pred_rounded], columns=embeddings)

        if original_node.size == 0:
            continue
        original_node = original_node.iloc[:, :embedding_size]
        mae_list.append(mean_absolute_error(extracted_node, original_node))

    similarities = [cosine_similarity(predictions[i], y_test[i]).mean() for i in range(len(predictions))]
    print(f'Cosine Similarity: {np.mean(similarities)}')

    # Calculate walk lengths based on composition details, where 0 indicates the end of the walk
    walk_lengths = []
    for _, row in composition_details.iterrows():
        walk_length = sum(row != 0)
        walk_lengths.append(walk_length)

    walk_length_categories = defaultdict(list)
    for i, length in enumerate(walk_lengths):
        if 5 <= length <= 10:
            walk_length_categories[length].append(i)

    walk_length_mae = defaultdict(list)
    walk_length_similarity = defaultdict(list)

    for length, indices in walk_length_categories.items():
        for i in indices:
            if i < len(mae_list) and i < len(similarities):  # Ensure index is within bounds
                walk_length_mae[length].append(mae_list[i])
                walk_length_similarity[length].append(similarities[i])

    walk_length_mae_mean = {length: np.mean(values) for length, values in walk_length_mae.items()}
    walk_length_similarity_mean = {length: np.mean(values) for length, values in walk_length_similarity.items()}
    # sort descending
    walk_length_similarity_mean = dict(sorted(walk_length_similarity_mean.items(), key=lambda x: x[1], reverse=True))

    # sort mae ascending
    walk_length_mae_mean = dict(sorted(walk_length_mae_mean.items(), key=lambda x: x[1]))

    print(f'Walk Length MAE: {walk_length_mae_mean}')
    print(f'Walk Length Cosine Similarity: {walk_length_similarity_mean}')

    similarity_df = pd.DataFrame(list(walk_length_similarity_mean.items()),
                                 columns=['Walk Length', 'Cosine Similarity'])
    similarity_df.to_csv(Path(results_folder, "similarity.csv"), index=False)

    # save the extracted nodes to a csv file
    extracted_nodes.to_csv(Path(results_folder, "unique_nodes.csv"), index=False)
