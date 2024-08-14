import netvae
import pandas as pd
import keras
from pathlib import Path
from argparse import ArgumentParser

tf_min_size = 3
learning_rate = 1e-4
batch_size = 256
epochs = 50
latent_dim = 768
rotation_factor = 0.3

save_folder = Path("results", "classifier", "embeddings")

if __name__ == '__main__':
    if not save_folder.exists():
        save_folder.mkdir(parents=True)

    parser = ArgumentParser(description='Create mutation embeddings.')
    parser.add_argument("--data", "-d", type=Path, help="The path to the data.", required=True)
    args = parser.parse_args()

    data = args.data
    epochs = 250

    X = pd.read_csv(data)

    submitter_id = X["submitter_id"]
    X.drop(columns=["submitter_id"], inplace=True)

    # add early stopping
    early_stopping = keras.callbacks.EarlyStopping(monitor='loss', patience=10, restore_best_weights=True)

    X_encoder = netvae.build_encoder(X.shape[1], latent_dim)
    X_decoder = netvae.build_decoder(X.shape[1], latent_dim)
    X_vae = netvae.VAE(X_encoder, X_decoder)
    X_vae.compile(optimizer=keras.optimizers.Adam(learning_rate=learning_rate))
    history = X_vae.fit(X, epochs=epochs, batch_size=batch_size, shuffle=True, callbacks=[early_stopping])

    X_pred = pd.DataFrame(X_vae.encoder.predict(X)[0])

    # assert that x_pred has 768 columns
    assert X_pred.shape[1] == latent_dim, f"Expected 768 columns, got {X_pred.shape[1]} instead."
    X_pred["submitter_id"] = submitter_id
    X_pred.to_csv(Path(save_folder, "mutation_embeddings.csv"), index=False)
