import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
import argparse
from sklearn.metrics import f1_score, precision_score, recall_score

load_folder = Path("results", "classifier", "summed_embeddings")
save_folder = Path("results", "classifier", "classification")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--cancer", "-c", nargs="+", required=True, help="The cancer types to work with.")
    parser.add_argument("--iteration", "-i", type=int, required=True, help="The iteration number.")
    args = parser.parse_args()

    selected_cancers = args.cancer
    iteration = args.iteration
    print("Selected cancers: ", selected_cancers)
    print(f"Using {len(selected_cancers)} output nodes for classifier.")

    cancers = "_".join(selected_cancers)

    load_folder = Path(load_folder, cancers)
    cancer_save_folder = Path(save_folder, cancers)
    iteration_save_folder = Path(cancer_save_folder, str(iteration))

    if not cancer_save_folder.exists():
        cancer_save_folder.mkdir(parents=True)

    if not iteration_save_folder.exists():
        iteration_save_folder.mkdir(parents=True)

    # load embeddings
    summed_embeddings = pd.read_csv(Path(load_folder, "summed_embeddings.csv"))
    # split data into train and test using sklearn
    train, test = train_test_split(summed_embeddings, test_size=0.2)

    X_train = train.drop(columns=["submitter_id", "cancer"])
    y_train = train["cancer"]

    X_test = test.drop(columns=["submitter_id", "cancer"])
    y_test = test["cancer"]

    # convert y_train and y_test numbers using sklearn
    label_encoder = LabelEncoder()
    y_train = label_encoder.fit_transform(y_train)
    y_test = label_encoder.transform(y_test)

    # min max scale the data using sklearn
    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # create a classifier using tensorflow and keras
    input_layer = tf.keras.layers.Input(shape=(X_train.shape[1],))
    x = tf.keras.layers.Dense(256, activation='relu', name="Dense_1")(input_layer)
    x = tf.keras.layers.Dense(128, activation='relu', name="Dense_2")(x)
    output = tf.keras.layers.Dense(len(selected_cancers), activation='sigmoid')(x)

    # add early stopping
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='loss',
        patience=5,
        verbose=1,
        restore_best_weights=True
    )
    model = tf.keras.models.Model(inputs=input_layer, outputs=output)
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    # fit the model
    history = model.fit(X_train, y_train, epochs=100, batch_size=32, callbacks=[early_stopping])

    # evaluate the model
    loss, accuracy = model.evaluate(X_test, y_test)

    y_hat = model.predict(X_test).argmax(axis=1)
    # calculate f1, precision and recall using sklearn
    f1_score = f1_score(y_test, y_hat, average='weighted')
    precision = precision_score(y_test, y_hat, average='weighted')
    recall = recall_score(y_test, y_hat, average='weighted')

    print(f"Loss: {loss}, Accuracy: {accuracy}, F1: {f1_score}, Precision: {precision}, Recall: {recall}")

    # save history
    history_df = pd.DataFrame(history.history)
    history_df.to_csv(Path(iteration_save_folder, "history.csv"), index=False)
    print("History saved.")

    # create df with loss and accuracy
    results = pd.DataFrame(
        {"loss": [loss], "accuracy": [accuracy], "f1": [f1_score], "precision": [precision], "recall": [recall],
         "iteration": [iteration]})
    results.to_csv(Path(iteration_save_folder, "results.csv"), index=False)
    print("Results saved.")

    # check if results.csv exists in the cancer save folder
    try:
        results = pd.read_csv(Path(cancer_save_folder, "results.csv"))
        new_results = pd.DataFrame({"loss": [loss], "accuracy": [accuracy], "iteration": [iteration]})
        # if there is an iteration in the results already, override it
        if iteration in results["iteration"].values:
            results = results[results["iteration"] != iteration]
        results = pd.concat([results, new_results])
        results.to_csv(Path(cancer_save_folder, "results.csv"), index=False)
    except FileNotFoundError:
        results = pd.DataFrame({"loss": [loss], "accuracy": [accuracy], "iteration": [iteration]})
        results.to_csv(Path(cancer_save_folder, "results.csv"), index=False)
