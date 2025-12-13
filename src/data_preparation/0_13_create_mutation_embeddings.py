import sys
from pathlib import Path

# Add src directory to path so we can import netvae module
src_path = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(src_path))
import netvae
import pandas as pd
import keras
from argparse import ArgumentParser

# Hyperparameters
tf_min_size = 3
learning_rate = 1e-4
batch_size = 256
epochs = 50
rotation_factor = 0.3

if __name__ == '__main__':

    parser = ArgumentParser(description='Create mutation embeddings.')
    parser.add_argument("--data", "-d", type=Path, help="The path to the data.", required=True)
    parser.add_argument("--latent_dim", "-ld", type=int, default=768, help="Latent dimension size.",
                        choices=[50, 100, 150, 200, 250, 300, 350, 400, 450, 500, 550, 600, 650, 700, 768])
    args = parser.parse_args()

    data: Path = args.data
    epochs: int = 250
    latent_dim: int = args.latent_dim

    if latent_dim == 768:
        save_folder = Path("results", "embeddings")
    else:
        save_folder = Path("results", "embeddings", "mutations", str(latent_dim))

    if not save_folder.exists():
        save_folder.mkdir(parents=True)

    # Load lookup table
    lookup: pd.DataFrame = pd.read_csv(Path("results", "embeddings", "lookup.csv"))

    # Load input data
    X = pd.read_csv(data)

    # Process submitter_id (truncate to first three parts)
    X["submitter_id"] = X["submitter_id"].apply(lambda x: '-'.join(x.split("-")[:3]))

    # Keep only rows where submitter_id exists in lookup
    X = X[X["submitter_id"].isin(lookup["submitter_id"])]

    # Retrieve cancer types
    X = X.merge(lookup[["submitter_id", "cancer"]], on="submitter_id", how="left")

    # Separate submitter_id & cancer before training
    submitter_ids = X["submitter_id"].values
    cancer_labels = X["cancer"].values

    # Drop submitter_id & cancer from training data
    X.drop(columns=["submitter_id", "cancer"], inplace=True)

    # Add early stopping
    early_stopping = keras.callbacks.EarlyStopping(monitor='loss', patience=10, restore_best_weights=True)

    # Build VAE
    X_encoder = netvae.build_encoder(X.shape[1], latent_dim)  # All remaining columns are used
    X_decoder = netvae.build_decoder(X.shape[1], latent_dim)
    X_vae = netvae.VAE(encoder=X_encoder, decoder=X_decoder)
    X_vae.compile(optimizer=keras.optimizers.Adam(learning_rate=learning_rate))

    # Train VAE
    history = X_vae.fit(X, epochs=epochs, batch_size=batch_size, shuffle=True, callbacks=[early_stopping])

    # Generate embeddings
    embeddings = pd.DataFrame(X_vae.encoder.predict(X)[0])

    # Ensure correct number of columns
    assert embeddings.shape[1] == latent_dim, f"Expected {latent_dim} columns, got {embeddings.shape[1]} instead."

    # Attach metadata
    embeddings["submitter_id"] = submitter_ids
    embeddings["cancer"] = cancer_labels

    print(embeddings)
    # assert that there are 6 unique cancer types
    assert embeddings[
               "cancer"].nunique() == 6, f"Expected 6 unique cancer types, got {embeddings['cancer'].nunique()} instead."
    # Save results
    embeddings.to_csv(Path(save_folder, "mutation_embeddings.csv"), index=False)

    print("Embeddings saved successfully.")
