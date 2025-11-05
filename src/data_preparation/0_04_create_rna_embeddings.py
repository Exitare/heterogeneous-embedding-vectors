import argparse
import torch
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import MinMaxScaler
from embkit.models.vae import RNAVAE
from embkit import get_device

save_folder = Path("results", "embeddings", "rna")
cancer_load_path = Path("data", "rna")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--pre_train_epochs", "-pt", type=int, default=100)
    parser.add_argument("--fine_tune_epochs", "-ft", type=int, default=100)
    parser.add_argument("--cancer", "-c", nargs="+", required=False,default=["BRCA", "LUAD", "STAD", "BLCA", "COAD", "THCA"],
                        help="The cancer types to work with.")
    parser.add_argument("--latent_dim", "-ld", type=int, default=768,
                        help="The latent dimension for the VAE.")

    args = parser.parse_args()

    pre_train_epochs = args.pre_train_epochs
    fine_tune_epochs = args.fine_tune_epochs
    selected_cancers = args.cancer
    latent_dim: int = args.latent_dim

    cancers = "_".join(selected_cancers)
    if latent_dim == 768:
        save_folder = Path(save_folder, cancers)
    else:
        save_folder = Path(save_folder, str(latent_dim), cancers)

    if not save_folder.exists():
        save_folder.mkdir(parents=True)

    # Load cancer data
    cancer_df = []
    for cancer in selected_cancers:
        df = pd.read_csv(
            Path(cancer_load_path, cancer.upper(), f"data.csv"),
            index_col=0,
            nrows=1000
        )
        df["Cancer"] = cancer
        cancer_df.append(df)

    data = pd.concat(cancer_df, axis=0)
    data.reset_index(drop=True, inplace=True)

    # Check that all columns are float
    for column in data.columns:
        if column == "Cancer" or column == "Patient":
            continue
        if data[column].dtype != float:
            print(f"{column} is not float. Converting...")
            data[column] = data[column].astype(float)

    # Extract metadata
    cancer_types = data["Cancer"].values
    patient_ids = data["Patient"].values

    # Drop metadata columns
    data.drop(columns=["Cancer", "Patient"], inplace=True)

    # Assertions
    assert "Cancer" not in data.columns, "Cancer column should not be present"
    assert "Patient" not in data.columns, "Patient column should not be present"

    # Scale the data (MinMaxScaler to [0, 1] range)
    scaler = MinMaxScaler()
    data_scaled = pd.DataFrame(
        scaler.fit_transform(data),
        columns=data.columns,
        index=data.index
    )

    feature_dim = len(data_scaled.columns)
    batch_size = 512

    print(f"Feature dimension: {feature_dim}")
    print(f"Latent dimension: {latent_dim}")
    print(f"Number of samples: {len(data_scaled)}")
    print(f"Cancer types: {selected_cancers}")
    print(f"Save folder: {save_folder}")

    # Create RNAVAE model
    rna_vae = RNAVAE(
        features=data_scaled.columns.tolist(),
        latent_dim=latent_dim,
        lr=0.0005,
        batch_norm=True
    )

    # Train with beta warmup
    print("\nTraining RNA VAE with beta warmup...")
    print(f"Using devive: {get_device()}")
    history = rna_vae.fit(
        data_scaled,
        epochs=pre_train_epochs,
        batch_size=batch_size,
        kappa=1.0,
        early_stopping_patience=3,
        progress=True
    )

    # Generate embeddings
    print("\nGenerating embeddings...")
    rna_vae.eval()
    with torch.no_grad():
        X_tensor = torch.tensor(data_scaled.values, dtype=torch.float32)

        # Move to same device as model
        device = next(rna_vae.parameters()).device
        X_tensor = X_tensor.to(device)

        # Get latent representation (mu)
        mu, logvar, z = rna_vae.encoder(X_tensor)
        latent_space_array = mu.cpu().numpy()

    # Create latent space DataFrame
    latent_space = pd.DataFrame(latent_space_array, index=data_scaled.index)

    # Add metadata
    latent_space["cancer"] = cancer_types
    latent_space["submitter_id"] = patient_ids

    # Save embeddings for each cancer type
    print("\nSaving embeddings...")
    for cancer in selected_cancers:
        subset = latent_space[latent_space["cancer"] == cancer].copy()
        output_path = Path(save_folder, f"{cancer.lower()}_embeddings.csv")
        subset.to_csv(output_path, index=False)
        print(f"Saved {len(subset)} embeddings for {cancer} to {output_path}")

    # Optionally save the model
    model_path = Path(save_folder, "rna_vae_model")
    rna_vae.save(str(model_path))
    print(f"\nModel saved to {model_path}")

    # Print training summary
    print("\n=== Training Summary ===")
    print(f"Final loss: {history['loss'][-1]:.4f}")
    print(f"Final reconstruction loss: {history['recon'][-1]:.4f}")
    print(f"Final KL loss: {history['kl'][-1]:.4f}")
    print(f"Final beta: {history['beta'][-1]:.4f}")
    print(f"Total epochs: {len(history['loss'])}")
