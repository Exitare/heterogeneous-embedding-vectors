import h5py
import numpy as np
import tensorflow as tf
from keras.src.layers import BatchNormalization
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score, matthews_corrcoef, \
    balanced_accuracy_score
from sklearn.preprocessing import label_binarize
import pandas as pd
from pathlib import Path
import argparse
from collections import defaultdict
from collections import Counter
import logging
from typing import List



logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def create_tf_dataset_specific_indices(h5_file_path, indices, batch_size, label_encoder):
    def generator():
        with h5py.File(h5_file_path, 'r') as h5_file:
            X = h5_file["X"][:]
            y = [label.decode("utf-8") for label in h5_file["y"][:]]
        y = label_encoder.transform(y)
        for idx in indices:
            yield X[idx], y[idx]

    with h5py.File(h5_file_path, 'r') as h5_file:
        feature_dimension = h5_file.attrs["feature_shape"]

    dataset = tf.data.Dataset.from_generator(
        generator,
        output_signature=(
            tf.TensorSpec(shape=(feature_dimension,), dtype=tf.float32),
            tf.TensorSpec(shape=(), dtype=tf.int32)
        )
    ).batch(batch_size).prefetch(tf.data.AUTOTUNE)

    return dataset


def create_tf_dataset_from_array(X, y, batch_size):
    dataset = tf.data.Dataset.from_tensor_slices((X.astype(np.float32), y.astype(np.int32)))
    dataset = dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return dataset


def train_and_evaluate_model(train_ds, val_ds, test_ds, num_classes: int, save_folder: Path, iteration: int,
                             walk_distance: int, amount_of_walks: int, label_encoder, feature_dimension: int):
    input_layer = tf.keras.layers.Input(shape=(feature_dimension,))

    x = BatchNormalization()(input_layer)
    x = tf.keras.layers.Dense(256, activation='relu')(x)
    x = tf.keras.layers.Dense(128, activation='relu')(x)
    output_layer = tf.keras.layers.Dense(num_classes, activation='softmax', name="output")(x)

    classes = list(label_encoder.classes_)
    decoded_classes = label_encoder.inverse_transform(np.arange(num_classes))

    class_weights = {classes.index(cancer): 1.0 for cancer in decoded_classes}
    class_weights[classes.index("LUAD")] = 6
    class_weights[classes.index("BLCA")] = 2.5

    model = tf.keras.models.Model(inputs=input_layer, outputs=output_layer)
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    history = model.fit(train_ds,
                        epochs=50,
                        validation_data=val_ds,
                        class_weight=class_weights,
                        callbacks=[
                            tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=5, mode='max',
                                                             restore_best_weights=True)
                        ])

    loss, accuracy = model.evaluate(test_ds)
    logging.info(f"Test Loss: {loss}, Test Accuracy: {accuracy}")

    y_test = []
    y_pred = []
    y_pred_proba = []
    for X_batch, y_batch in test_ds:
        y_test.extend(y_batch.numpy())
        y_pred.extend(model.predict(X_batch).argmax(axis=1))
        y_pred_proba.extend(model.predict(X_batch))

    y_test = np.array(y_test)
    y_pred = np.array(y_pred)
    y_pred_proba = np.array(y_pred_proba)

    predictions = pd.DataFrame({
        "y_test": y_test,
        "y_test_decoded": label_encoder.inverse_transform(y_test),
        "y_pred": y_pred,
        "y_pred_decoded": label_encoder.inverse_transform(y_pred)
    })

    predictions.to_csv(Path(save_folder, f"predictions.csv"), index=False)

    results = []
    for cancer in np.unique(y_test):
        y_test_cancer = y_test[y_test == cancer]
        y_pred_cancer = y_pred[y_test == cancer]
        cancer_name = label_encoder.inverse_transform([cancer])[0]

        accuracy_cancer = (y_test_cancer == y_pred_cancer).mean()
        f1_cancer = f1_score(y_test_cancer, y_pred_cancer, average='weighted')
        precision_cancer = precision_score(y_test_cancer, y_pred_cancer, average='weighted')
        recall_cancer = recall_score(y_test_cancer, y_pred_cancer, average='weighted')

        logging.info(
            f"{cancer_name}: Accuracy: {accuracy_cancer:.4f}, F1: {f1_cancer:.4f}, Precision: {precision_cancer:.4f}, Recall: {recall_cancer:.4f}.")

        results.append({
            "cancer": cancer_name,
            "accuracy": accuracy_cancer,
            "f1": f1_cancer,
            "precision": precision_cancer,
            "recall": recall_cancer,
            "iteration": iteration,
            "walk_distance": walk_distance,
            "amount_of_walks": amount_of_walks
        })

    f1_total = f1_score(y_test, y_pred, average='weighted')
    precision_total = precision_score(y_test, y_pred, average='weighted')
    recall_total = recall_score(y_test, y_pred, average='weighted')
    accuracy_total = (y_test == y_pred).mean()
    mcc_total = matthews_corrcoef(y_test, y_pred)
    balanced_accuracy = balanced_accuracy_score(y_test, y_pred)

    num_classes = y_pred_proba.shape[1]
    y_test_one_hot = label_binarize(y_test, classes=np.arange(num_classes))
    auc_score = roc_auc_score(y_test_one_hot, y_pred_proba, multi_class='ovo', average='macro')

    logging.info(
        f"Overall: Accuracy: {accuracy_total:.4f}, F1: {f1_total:.4f}, Precision: {precision_total:.4f}, Recall: {recall_total:.4f}, AUC: {auc_score:.4f}, MCC: {mcc_total:.4f}, Balanced Accuracy: {balanced_accuracy:.4f}")

    results.append({
        "cancer": "All",
        "accuracy": accuracy_total,
        "f1": f1_total,
        "mcc": mcc_total,
        "balanced_accuracy": balanced_accuracy,
        "precision": precision_total,
        "recall": recall_total,
        "auc": auc_score,
        "iteration": iteration,
        "walk_distance": walk_distance,
        "amount_of_walks": amount_of_walks
    })

    results_df = pd.DataFrame(results)
    results_df.to_csv(Path(save_folder, f"results.csv"), index=False)
    logging.info("Results saved.")

    model.save(Path(save_folder, f"model.keras"))
    history_df = pd.DataFrame(history.history)
    history_df.to_csv(Path(save_folder, f"history.csv"), index=False)
    logging.info("Model and history saved.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", "-b", type=int, default=32)
    parser.add_argument("--cancer", "-c", nargs="+", default=["BRCA", "LUAD", "STAD", "BLCA", "COAD", "THCA"])
    parser.add_argument("--iteration", "-i", type=int, required=True)
    parser.add_argument("--walk_distance", "-w", type=int, choices=[3, 4, 5, 6], default=3)
    parser.add_argument("--amount_of_walks", "-a", type=int, choices=[3, 4, 5, 6], default=3)
    parser.add_argument("--latent_dim", "-ld", type=int,
                        choices=[50, 100, 150, 200, 250, 300, 350, 400, 450, 500, 550, 600, 650, 700], default=50)
    args = parser.parse_args()

    batch_size = args.batch_size
    walk_distance = args.walk_distance
    walk_amount = args.amount_of_walks
    iteration = args.iteration
    selected_modalities: List[str] = ["rna", "mutations"]
    latent_dim = args.latent_dim

    modalities = '_'.join(selected_modalities)
    cancers = "_".join(args.cancer)

    load_folder = Path("results", "classifier_latent_dims", "summed_embeddings", str(latent_dim))
    save_folder = Path("results", "classifier_latent_dims", "classification", str(latent_dim))

    load_folder = Path(load_folder, cancers, modalities, f"{walk_distance}_{walk_amount}", str(iteration))
    train_h5_file_path = Path(load_folder, "train_summed_embeddings.h5")
    test_h5_file_path = Path(load_folder, "test_summed_embeddings.h5")

    cancer_save_folder = Path(save_folder, cancers, modalities, f"{walk_distance}_{walk_amount}")
    iteration_save_folder = Path(cancer_save_folder, str(iteration))
    iteration_save_folder.mkdir(parents=True, exist_ok=True)

    with h5py.File(train_h5_file_path, "r") as h5_file:
        feature_dimension = h5_file.attrs["feature_shape"]
        unique_classes = h5_file.attrs["classes"]
        train_X = h5_file["X"][:]
        train_y = np.array([label.decode("utf-8") for label in h5_file["y"][:]])

    with h5py.File(test_h5_file_path, "r") as h5_file:
        test_X = h5_file["X"][:]
        test_y = np.array([label.decode("utf-8") for label in h5_file["y"][:]])

    label_encoder = LabelEncoder()
    label_encoder.fit(unique_classes)

    class_counts = Counter(train_y)
    split_indices = {"train": [], "val": []}
    split_sizes = {cls: {"train": int(count * 0.8), "val": int(count * 0.2)} for cls, count in class_counts.items()}
    allocated = defaultdict(lambda: {"train": 0, "val": 0})

    for idx, label in enumerate(train_y):
        if allocated[label]["train"] < split_sizes[label]["train"]:
            split_indices["train"].append(idx)
            allocated[label]["train"] += 1
        else:
            split_indices["val"].append(idx)
            allocated[label]["val"] += 1

    logging.info(f"Train size: {len(split_indices['train'])}, Val size: {len(split_indices['val'])}")

    train_ds = create_tf_dataset_specific_indices(train_h5_file_path, split_indices['train'], batch_size, label_encoder)
    val_ds = create_tf_dataset_specific_indices(train_h5_file_path, split_indices['val'], batch_size, label_encoder)

    test_y_encoded = label_encoder.transform(test_y)
    test_ds = create_tf_dataset_from_array(test_X, test_y_encoded, batch_size)

    train_and_evaluate_model(
        train_ds=train_ds,
        val_ds=val_ds,
        test_ds=test_ds,
        num_classes=len(unique_classes),
        save_folder=iteration_save_folder,
        iteration=args.iteration,
        walk_distance=args.walk_distance,
        amount_of_walks=args.amount_of_walks,
        label_encoder=label_encoder,
        feature_dimension=feature_dimension
    )
