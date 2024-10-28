import tensorflow as tf
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Input, Dense, BatchNormalization, Dropout, ReLU
from tensorflow.keras.models import Model
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from argparse import ArgumentParser
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ReduceLROnPlateau
from sklearn.model_selection import StratifiedShuffleSplit

embeddings = ['Text', 'Image', 'RNA']
save_path = Path("results", "noise_recognizer", "multi_foundation")
clean_load_path = Path("results", "recognizer", "summed_embeddings", "multi")
noisy_load_path = Path("results", "noise_recognizer", "summed_embeddings", "multi")
epochs = 100


# Function to create stratified splits for multi-label data
def multilabel_stratified_split(X, y, test_size=0.2, random_state=None):
    n_samples, n_labels = y.shape
    indices = np.arange(n_samples)

    sss = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    train_idx, test_idx = next(sss.split(indices, y[:, 0]))  # Start with the first label

    for i in range(1, n_labels):
        _, new_test_idx = next(sss.split(indices[train_idx], y[train_idx, i]))
        test_idx = np.union1d(test_idx, new_test_idx)
        train_idx = np.setdiff1d(indices, test_idx)

    return train_idx, test_idx


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
    text_output = Dense(1, activation=ReLU(max_value=max_walk_distance), name='output_text')(text_x)

    # Less complex paths for image output
    image_output = Dense(1, activation=ReLU(max_value=max_walk_distance), name='output_image')(x)

    # Path for RNA embeddings, including subtype classification
    rna_x = Dense(128, activation='relu', name='rna_dense_1')(x)
    rna_x = Dropout(0.2)(rna_x)
    rna_x = Dense(64, activation='relu', name='rna_dense_2')(rna_x)
    rna_x = BatchNormalization()(rna_x)
    rna_x = Dropout(0.2)(rna_x)
    rna_x = Dense(32, activation='relu', name='rna_dense_3')(rna_x)
    rna_output = Dense(1, activation=ReLU(max_value=max_walk_distance), name='output_rna')(rna_x)

    cancer_outputs = [Dense(1, activation=ReLU(max_value=max_walk_distance), name=f'output_cancer_{cancer_type}')(x) for
                      cancer_type in cancer_list]

    # Combine all outputs
    outputs = [text_output, image_output, rna_output] + cancer_outputs

    # Create model
    return Model(inputs=inputs, outputs=outputs, name='multi_output_model')


if __name__ == '__main__':
    # python3 src/recognizer/multi_cancer_recognizer_foundation.py -c blca brca

    parser = ArgumentParser(description='Train a multi-output model for recognizing embeddings')
    parser.add_argument('--batch_size', "-bs", type=int, default=32, help='The batch size to train the model')
    parser.add_argument("--run_iteration", "-ri", type=int, required=False,
                        help="The iteration number for the run. Used for saving the results and validation.", default=1)
    parser.add_argument("--cancer", "-c", nargs="+", required=True, help="The cancer types to work with.")
    parser.add_argument("--upper_walk_distance", "-uwd", type=int, required=False, default=10)
    parser.add_argument("--summed_embedding_count", "-sec", type=int, required=True,
                        help="The size of the generated summed embeddings count.")
    args = parser.parse_args()

    batch_size = args.batch_size
    run_iteration = args.run_iteration
    selected_cancers = args.cancer
    cancers = "_".join(selected_cancers)
    upper_walk_distance = args.upper_walk_distance
    summed_embedding_count = args.summed_embedding_count

    walk_distances = range(2, upper_walk_distance + 1)

    print("Selected cancers: ", selected_cancers)
    print(f"Batch size: {batch_size}")
    print(f"Run iteration: {run_iteration}")
    print(f"Upper walk distance: {upper_walk_distance}")
    print(f"Walk distances: {walk_distances}")

    save_path = Path(save_path, cancers)
    if not save_path.exists():
        save_path.mkdir(parents=True)

    data = []
    for walk_distance in walk_distances:
        embedding_load_path = Path(clean_load_path, cancers, f"{walk_distance}_embeddings.csv")
        print(f"Loading data from {embedding_load_path}")
        data.append(pd.read_csv(embedding_load_path))

    data = pd.concat(data, axis=0)

    # find max value of embeddings for ReLU activation
    max_text = data["Text"].max().max()
    max_image = data["Image"].max().max()
    max_rna = data["RNA"].max().max()

    # Calculate the sum of each row for the selected columns
    row_sums = data[["Text", "Image", "RNA"]].sum(axis=1)

    # Find the maximum sum across all rows
    max_walk_distance = row_sums.max()

    run_name = f"run_{run_iteration}"
    save_path = Path(save_path, run_name)

    if not save_path.exists():
        save_path.mkdir(parents=True)

    print(f"Saving results to {save_path}")

    text_counts = data["Text"].values
    image_counts = data["Image"].values
    rna_counts = data["RNA"].values

    # extract cancer data
    cancer_count_data = []
    for cancer in selected_cancers:
        cancer_count_data.append(data[cancer].values)
        embeddings.append(cancer)

    # get the number of subtypes
    X = data.drop(columns=embeddings).values
    assert X.shape[1] == 768, f"Expected 768 features, got {X.shape[1]}"

    # Assuming these are the actual labels from your dataset
    y = [text_counts, image_counts, rna_counts] + cancer_count_data

    # convert all columns in y to int
    y = [y_i.astype(int) for y_i in y]

    model = build_model(X.shape[1], selected_cancers)

    # Set up a list of metrics
    loss = {'output_text': 'mae', 'output_image': 'mae', 'output_rna': 'mae'}
    loss_weights = {'output_text': 3.0, 'output_image': 1., 'output_rna': 1.}
    metrics = ['mae', 'mae', 'mae']

    # Adding dynamic metrics for cancer outputs based on their number
    for i in selected_cancers:  # Assuming num_cancer_types is defined
        loss[f'output_cancer_{i}'] = 'mae'
        loss_weights[f'output_cancer_{i}'] = 1.
        metrics.append('mae')

    model.compile(optimizer='adam',
                  loss=loss,
                  loss_weights=loss_weights,
                  metrics=metrics)
    model.summary()

    # Convert y_list to a multi-dimensional numpy array
    y = np.array(y).T  # Transpose to get shape (900, 5)

    # Perform the split
    train_index, test_index = multilabel_stratified_split(X, y, test_size=0.2, random_state=42)
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]

    # scale the data
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # create early stopping callback
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=10,
        mode='min',
        restore_best_weights=True
    )

    # Train model
    history = model.fit(X_train, [y_train[:, i] for i in range(y_train.shape[1])],
                        validation_split=0.2, epochs=epochs, batch_size=batch_size, callbacks=[early_stopping])

    # save history
    pd.DataFrame(history.history).to_csv(Path(save_path, "history.csv"), index=False)

    # fine tune the model
    # freeze all layer except the one containing text in the layer name
    for layer in model.layers:
        if 'text' not in layer.name:
            layer.trainable = False

    fine_tuning_early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=20,
        mode='min',
        restore_best_weights=True
    )
    optimizer = Adam(learning_rate=0.0001)  # Lower learning rate for fine-tuning
    reduce_lr = ReduceLROnPlateau(monitor='val_output_text_mae', factor=0.2, patience=5, min_lr=0.00001, mode='min')

    # adjust the text loss weight
    loss_weights["output_text"] = 4.0
    loss_weights["output_image"] = 0.1
    loss_weights["output_rna"] = 0.1

    model.compile(optimizer=optimizer,
                  loss=loss,
                  loss_weights=loss_weights,
                  metrics=metrics)
    model.summary()
    history = model.fit(X_train, [y_train[:, i] for i in range(y_train.shape[1])],
                        validation_split=0.2, epochs=epochs, batch_size=batch_size,
                        callbacks=[fine_tuning_early_stopping, reduce_lr])

    # Evaluate model
    results = model.evaluate(X_test, [y_test[:, i] for i in range(y_test.shape[1])])
    print("Test Loss after fine tuning:")
    print("Test MAE:", results)

    # save history
    pd.DataFrame(history.history).to_csv(Path(save_path, "fine_tune_history.csv"), index=False)

    # Predict counts
    y_pred = model.predict(X_test)

    # Convert predictions to rounded integers
    y_pred_rounded = [np.rint(pred) for pred in y_pred]

    # save numpy array
    np.save(Path(save_path, "y_pred.npy"), y_pred_rounded)

    # Ensure y_test is an integer array (necessary if y_test is not already in integer form)
    y_test_int = y_test.astype(int)

    # save y_test_int numpy array
    np.save(Path(save_path, "y_test.npy"), y_test_int)

    # save predictions for all outputs
    for i, embedding in enumerate(embeddings):
        pd.DataFrame(y_pred_rounded[i]).to_csv(Path(save_path, f"predictions_{embedding}_output.csv"),
                                               index=False)

    # save y_test for all outputs
    for i, embedding in enumerate(embeddings):
        pd.DataFrame(y_test_int[:, i]).to_csv(Path(save_path, f"true_{embedding}_output.csv"),
                                              index=False)

    metrics = []
    # Calculate accuracy for each output
    accuracy = [accuracy_score(y_true, y_pred) for y_true, y_pred in zip(y_test_int.T, y_pred_rounded)]
    # calculate f1 score for each output
    f1 = [f1_score(y_true, y_pred, average='macro') for y_true, y_pred in zip(y_test_int.T, y_pred_rounded)]
    # calculate precision for each output
    precision = [precision_score(y_true, y_pred, average='macro') for y_true, y_pred in
                 zip(y_test_int.T, y_pred_rounded)]
    # calculate recall for each output
    recall = [recall_score(y_true, y_pred, average='macro') for y_true, y_pred in zip(y_test_int.T, y_pred_rounded)]
    # calculate auc for each output
    # auc = [roc_auc_score(y_true, y_pred) for y_true, y_pred in zip(y_test_int.T, y_pred_rounded)]

    # for each output, store the metrics
    for i, embedding in enumerate(embeddings):
        metrics.append({
            "walk_distance": max_walk_distance,
            'embedding': embedding,
            'accuracy': accuracy[i],
            'precision': precision[i],
            'recall': recall[i],
            'f1': f1[i]
        })

    metrics_df = pd.DataFrame(metrics)
    metrics_df.to_csv(Path(save_path, "metrics.csv"), index=False)

    # reset index of y_test and y_pred_round
    y_test_int = pd.DataFrame(y_test_int)
    # concat y_pred_rounded
    y_pred_rounded = pd.concat([pd.DataFrame(y_pred_rounded[i]) for i in range(len(y_pred_rounded))], axis=1)

    y_test_int.reset_index(drop=True, inplace=True)
    y_pred_rounded.reset_index(drop=True, inplace=True)

    columns = ["Text", "Image", "RNA"] + selected_cancers
    y_test_int.columns = columns
    # calculate total embeddings by only using Text Image and RNa columns
    y_test_int["Walk Distance"] = y_test_int[["Text", "Image", "RNA"]].sum(axis=1)
    y_pred_rounded.columns = columns
    split_metrics = []  #

    for embedding in embeddings:
        y_test_sub = y_test_int[embedding]
        y_pred_sub = y_pred_rounded[embedding]

        # iterate over all total embeddings from y_test_int
        for walk_distance in y_test_int["Walk Distance"].unique():
            y_test_sub_total = y_test_sub[y_test_int["Walk Distance"] == walk_distance]
            y_pred_sub_total = y_pred_sub[y_test_int["Walk Distance"] == walk_distance]

            # convert to int
            y_test_sub_total = y_test_sub_total.astype(int)
            y_pred_sub_total = y_pred_sub_total.astype(int)

            accuracy = accuracy_score(y_test_sub_total, y_pred_sub_total)
            precision = precision_score(y_test_sub_total, y_pred_sub_total, average='macro')
            recall = recall_score(y_test_sub_total, y_pred_sub_total, average='macro')
            f1 = f1_score(y_test_sub_total, y_pred_sub_total, average='macro')

            split_metrics.append({
                'walk_distance': walk_distance,
                'embedding': embedding,
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1': f1
            })

    split_metrics = pd.DataFrame(split_metrics)
    split_metrics.to_csv(Path(save_path, "split_metrics.csv"), index=False)

    # Noisy data recognition
    noise_load_path = Path(noisy_load_path, str(summed_embedding_count), cancers)

    noisy_test_data = []
    for walk_distance in walk_distances:
        for noise_ratio in range(10, 110, 10):
            noise_ratio = noise_ratio / 100
            current_noise_load_path = Path(noise_load_path, f"{noise_ratio}", f"{walk_distance}_embeddings.csv")
            print(f"Loading data from {current_noise_load_path}")
            tmp_df = pd.read_csv(current_noise_load_path)
            tmp_df["Noise Ratio"] = noise_ratio
            noisy_test_data.append(tmp_df)

    noisy_test_data = pd.concat(noisy_test_data, axis=0)

    # get y_test_int
    noisy_int_truth = noisy_test_data[embeddings].values
    noisy_int_truth = noisy_int_truth.astype(int)

    noisy_int_truth = pd.DataFrame(noisy_int_truth)
    noisy_int_truth.columns = embeddings

    y_noisy_pred = model.predict(noisy_test_data.drop(columns=embeddings).values)

    noisy_split_metrics = []

    for embedding in embeddings:
        y_embedding_test_sub = noisy_int_truth[embedding]
        y_embedding_pred_sub = y_noisy_pred[embedding]

        # iterate over all total embeddings from y_test_int
        for walk_distance in y_test_int["Walk Distance"].unique():
            y_walk_test = y_embedding_test_sub[y_test_int["Walk Distance"] == walk_distance]
            y_walk_pred = y_embedding_pred_sub[y_test_int["Walk Distance"] == walk_distance]

            for noise_ratio in range(10, 110, 10):
                y_noisy_test_sub = y_walk_test[noisy_int_truth["Noise Ratio"] == noise_ratio]
                y_noisy_pred_sub = y_walk_pred[noisy_int_truth["Noise Ratio"] == noise_ratio]

                # convert to int
                y_test_sub_total = y_noisy_test_sub.astype(int)
                y_pred_sub_total = y_noisy_pred_sub.astype(int)

                accuracy = accuracy_score(y_test_sub_total, y_pred_sub_total)
                precision = precision_score(y_test_sub_total, y_pred_sub_total, average='macro')
                recall = recall_score(y_test_sub_total, y_pred_sub_total, average='macro')
                f1 = f1_score(y_test_sub_total, y_pred_sub_total, average='macro')

                noisy_split_metrics.append({
                    'walk_distance': walk_distance,
                    'embedding': embedding,
                    'noise_ratio': noise_ratio,
                    'accuracy': accuracy,
                    'precision': precision,
                    'recall': recall,
                    'f1': f1
                })

    noisy_split_metrics = pd.DataFrame(noisy_split_metrics)

    noisy_split_metrics.to_csv(Path(save_path, "noisy_split_metrics.csv"), index=False)

    print("Done!")
