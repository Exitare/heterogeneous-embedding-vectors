import os, time, argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import MinMaxScaler
import numpy as np

if torch.backends.mps.is_available():
    device = torch.device("mps")
    print(f"Using MPS (Apple Silicon GPU)")
elif torch.cuda.is_available():
    device = torch.device("cuda")
    print(f"Using CUDA GPU: {torch.cuda.get_device_name(0)}")
else:
    device = torch.device("cpu")
    print("Using CPU")

save_folder = Path("results", "embeddings", "rna")
cancer_load_path = Path("data", "rna")


class VAE(nn.Module):
    def __init__(self, feature_dim, latent_dim):
        super(VAE, self).__init__()
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(feature_dim, feature_dim // 2),
            nn.ReLU(),
            nn.Linear(feature_dim // 2, feature_dim // 3),
            nn.ReLU()
        )
        
        # Mean layer (Linear -> BatchNorm -> ReLU)
        self.fc_mu_linear = nn.Linear(feature_dim // 3, latent_dim)
        self.fc_mu_bn = nn.BatchNorm1d(latent_dim)
        
        # Log variance layer (Linear -> BatchNorm -> ReLU)
        self.fc_logvar_linear = nn.Linear(feature_dim // 3, latent_dim)
        self.fc_logvar_bn = nn.BatchNorm1d(latent_dim)
        
        # Decoder
        self.decoder_linear = nn.Linear(latent_dim, feature_dim)
        
        # Initialize weights with Xavier/Glorot uniform (matches TensorFlow default)
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize weights with glorot_uniform like TensorFlow"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def encode(self, x):
        h = self.encoder(x)
        
        # mu: Linear -> BatchNorm -> ReLU
        mu = self.fc_mu_linear(h)
        mu = self.fc_mu_bn(mu)
        mu = torch.relu(mu)
        
        # logvar: Linear -> BatchNorm -> ReLU
        logvar = self.fc_logvar_linear(h)
        logvar = self.fc_logvar_bn(logvar)
        logvar = torch.relu(logvar)
        
        return mu, logvar
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, z):
        return torch.sigmoid(self.decoder_linear(z))
    
    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar


def vae_loss(recon_x, x, mu, logvar, beta, feature_dim):
    """
    VAE loss = Reconstruction loss + beta * KL divergence
    Matches TensorFlow implementation exactly:
    - reconstruction_loss = feature_dim * binary_crossentropy (mean over features, then scaled)
    - kl_loss = sum over latent dimensions
    - total = mean over batch
    """
    # Reconstruction loss: feature_dim * binary_crossentropy
    # TF's binary_crossentropy with default reduction returns mean over last dimension (features)
    # Then multiplied by feature_dim
    BCE_per_sample = nn.functional.binary_cross_entropy(recon_x, x, reduction='none').mean(dim=1)  # Mean over features
    reconstruction_loss = feature_dim * BCE_per_sample  # Scale by feature_dim
    
    # KL divergence: sum over latent dimensions, per sample
    KLD_per_sample = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)
    
    # Total loss: mean over batch
    total_loss = torch.mean(reconstruction_loss + 5 * (beta * KLD_per_sample))
    
    return total_loss


class EarlyStopping:
    def __init__(self, patience=3, verbose=False):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
        self.best_model = None
        
    def __call__(self, loss, model):
        if self.best_loss is None:
            self.best_loss = loss
            self.best_model = model.state_dict().copy()
        elif loss > self.best_loss:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = loss
            self.best_model = model.state_dict().copy()
            self.counter = 0
        
        return self.early_stop


def train_epoch(model, dataloader, optimizer, beta, feature_dim, device):
    model.train()
    train_loss = 0
    
    for batch_idx, (data,) in enumerate(dataloader):
        data = data.to(device)
        optimizer.zero_grad()
        
        recon_batch, mu, logvar = model(data)
        loss = vae_loss(recon_batch, data, mu, logvar, beta, feature_dim)
        
        loss.backward()
        train_loss += loss.item()
        optimizer.step()
    
    return train_loss / len(dataloader)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--pre_train_epochs", "-pt", type=int, default=100)
    parser.add_argument("--fine_tune_epochs", "-ft", type=int, default=100)
    parser.add_argument("--cancer", "-c", nargs="+", required=False, help="The cancer types to work with.",
                        default=["BRCA", "LUAD", "STAD", "BLCA", "COAD", "THCA"])
    parser.add_argument("--latent_dim", "-ld", type=int, default=768, help="Latent dimension size.",
                        choices=[50, 100, 150, 200, 250, 300, 350, 400, 450, 500, 550, 600, 650, 700, 768])

    args = parser.parse_args()

    latent_dim = args.latent_dim
    pre_train_epochs = args.pre_train_epochs
    fine_tune_epochs = args.fine_tune_epochs
    selected_cancers = args.cancer

    cancers = "_".join(selected_cancers)
    if latent_dim == 768:
        save_folder = Path(save_folder, cancers)
    else:
        save_folder = Path(save_folder, str(latent_dim), cancers)

    if not save_folder.exists():
        save_folder.mkdir(parents=True)

    cancer_df = []
    for cancer in selected_cancers:
        df = pd.read_csv(Path(cancer_load_path, cancer.upper(), f"data.csv"), index_col=0, nrows=1000)
        df["Cancer"] = cancer
        cancer_df.append(df)

    data = pd.concat(cancer_df, axis=0)
    data.reset_index(drop=True, inplace=True)

    # check that all columns to be float
    for column in data.columns:
        if column == "Cancer" or column == "Patient":
            continue
        if data[column].dtype != float:
            print(f"{column} is not float. Converting...")
            data[column] = data[column].astype(float)

    cancer_types = data["Cancer"]
    patient_ids = data["Patient"]

    # drop the cancer column
    data.drop(columns=["Cancer"], inplace=True)
    data.drop(columns=["Patient"], inplace=True)

    # assert not cancer or patient columns are in data
    assert "Cancer" not in data.columns, "Cancer column should not be present"
    assert "Patient" not in data.columns, "Patient column should not be present"

    # scale the data
    data = pd.DataFrame(MinMaxScaler().fit_transform(data), columns=data.columns)

    feature_dim = len(data.columns)
    batch_size = 512
    learning_rate = 0.0005

    # Convert to PyTorch tensors
    data_tensor = torch.FloatTensor(data.values)
    dataset = TensorDataset(data_tensor)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Initialize model
    model = VAE(feature_dim, latent_dim).to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Training with warm-up (beta starts at 0, increases by kappa each epoch)
    kappa = 1
    beta = 0.0
    early_stopping = EarlyStopping(patience=3, verbose=True)

    print(f"Training VAE with {feature_dim} features, latent_dim={latent_dim}")
    fit_start = time.time()
    
    for epoch in range(pre_train_epochs):
        # Warm-up: beta increases by kappa each epoch, capped at 1
        beta = min(beta + kappa, 1.0)
        
        avg_loss = train_epoch(model, dataloader, optimizer, beta, feature_dim, device)
        
        # Print every epoch like original Keras verbose=1
        print(f'Epoch {epoch+1}/{pre_train_epochs} - loss: {avg_loss:.4f} - beta: {beta:.4f}')
        
        # Early stopping check
        if early_stopping(avg_loss, model):
            print(f'Early stopping triggered at epoch {epoch+1}')
            model.load_state_dict(early_stopping.best_model)
            break

    fit_time = time.time() - fit_start
    print(f"Training completed in {fit_time:.2f} seconds")

    # Generate embeddings
    model.eval()
    with torch.no_grad():
        data_tensor = data_tensor.to(device)
        mu, _ = model.encode(data_tensor)
        latent_space = mu.cpu().numpy()
        
        # Decode for reconstruction
        decoded = model.decode(torch.FloatTensor(latent_space).to(device)).cpu().numpy()

    decoded_df = pd.DataFrame(decoded, columns=data.columns)
    latent_space_df = pd.DataFrame(latent_space, index=data.index)

    # assign cancer types to latent space
    latent_space_df["cancer"] = cancer_types.values
    latent_space_df["submitter_id"] = patient_ids.values

    # iterate through all cancer types and save the subset of the latent space
    for cancer in selected_cancers:
        subset = latent_space_df[latent_space_df["cancer"] == cancer].copy()
        subset.to_csv(Path(save_folder, f"{cancer.lower()}_embeddings.csv"), index=False)
    
    print(f"Embeddings saved to {save_folder}")
