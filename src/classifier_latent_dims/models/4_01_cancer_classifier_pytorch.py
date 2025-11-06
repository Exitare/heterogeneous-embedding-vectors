import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
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
from tqdm import tqdm
import time

load_folder = Path("results", "classifier_latent_dims", "summed_embeddings")
save_folder = Path("results", "classifier_latent_dims", "classification")

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def get_device():
    """Get the best available device (MPS > CUDA > CPU)"""
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        logging.info("Using MPS (Apple Silicon GPU)")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        logging.info(f"Using CUDA GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        logging.info("Using CPU")
    return device


class H5Dataset(Dataset):
    """PyTorch Dataset for loading data from HDF5 files"""
    def __init__(self, h5_file_path, indices, label_encoder):
        self.h5_file_path = h5_file_path
        self.indices = indices
        self.label_encoder = label_encoder
        
        # Load data into memory for faster access
        with h5py.File(h5_file_path, 'r') as h5_file:
            self.X = h5_file["X"][indices].astype(np.float32)
            y = [label.decode("utf-8") for label in h5_file["y"][indices]]
            self.y = label_encoder.transform(y).astype(np.int64)
    
    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, idx):
        return torch.from_numpy(self.X[idx]), torch.tensor(self.y[idx], dtype=torch.long)


class ArrayDataset(Dataset):
    """PyTorch Dataset for numpy arrays"""
    def __init__(self, X, y):
        self.X = torch.from_numpy(X.astype(np.float32))
        self.y = torch.from_numpy(y.astype(np.int64))
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class CancerClassifier(nn.Module):
    """Neural network for cancer classification"""
    def __init__(self, feature_dimension, num_classes):
        super(CancerClassifier, self).__init__()
        
        self.network = nn.Sequential(
            nn.BatchNorm1d(feature_dimension),
            nn.Linear(feature_dimension, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )
    
    def forward(self, x):
        return self.network(x)


def train_epoch(model, dataloader, criterion, optimizer, device):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    for X_batch, y_batch in tqdm(dataloader, desc="Training", leave=False):
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        
        optimizer.zero_grad()
        outputs = model(X_batch)
        
        # Loss is computed with class weights already built into criterion
        loss = criterion(outputs, y_batch)
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        _, predicted = outputs.max(1)
        total += y_batch.size(0)
        correct += predicted.eq(y_batch).sum().item()
    
    return total_loss / len(dataloader), 100. * correct / total


def validate(model, dataloader, criterion, device):
    """Validate the model"""
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for X_batch, y_batch in tqdm(dataloader, desc="Validating", leave=False):
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += y_batch.size(0)
            correct += predicted.eq(y_batch).sum().item()
    
    return total_loss / len(dataloader), 100. * correct / total


def evaluate_model(model, dataloader, device, label_encoder):
    """Evaluate model and return predictions and probabilities"""
    model.eval()
    y_test = []
    y_pred = []
    y_pred_proba = []
    
    with torch.no_grad():
        for X_batch, y_batch in tqdm(dataloader, desc="Evaluating", leave=False):
            X_batch = X_batch.to(device)
            outputs = model(X_batch)
            probabilities = torch.softmax(outputs, dim=1)
            
            y_test.extend(y_batch.cpu().numpy())
            y_pred.extend(outputs.argmax(dim=1).cpu().numpy())
            y_pred_proba.extend(probabilities.cpu().numpy())
    
    return np.array(y_test), np.array(y_pred), np.array(y_pred_proba)


def train_and_evaluate_model(train_loader, val_loader, test_loader, num_classes: int, save_folder: Path, 
                             iteration: int, walk_distance: int, amount_of_walks: int, label_encoder, 
                             feature_dimension: int, device, class_weights_dict=None):
    """Train and evaluate the PyTorch model"""
    
    start_time = time.time()
    
    # Create model
    model = CancerClassifier(feature_dimension, num_classes).to(device)
    
    # Convert class weights to tensor if provided and create loss with weights
    class_weights_tensor = None
    if class_weights_dict is not None:
        weights = torch.ones(num_classes)
        for cls_idx, weight in class_weights_dict.items():
            weights[cls_idx] = weight
        class_weights_tensor = weights.to(device)
    
    # Loss and optimizer - use weight parameter of CrossEntropyLoss
    criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Training loop
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
    best_val_acc = 0
    patience = 5
    patience_counter = 0
    
    logging.info(f"Starting training on {device}...")
    training_start_time = time.time()
    
    for epoch in range(50):
        epoch_start_time = time.time()
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        epoch_time = time.time() - epoch_start_time
        
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        logging.info(f"Epoch {epoch+1}/50 - Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, "
                    f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}% - Time: {epoch_time:.2f}s")
        
        # Early stopping
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            # Save best model
            torch.save(model.state_dict(), Path(save_folder, "best_model.pt"))
        else:
            patience_counter += 1
            if patience_counter >= patience:
                logging.info(f"Early stopping at epoch {epoch+1}")
                break
    
    training_time = time.time() - training_start_time
    logging.info(f"Training completed in {training_time:.2f}s ({training_time/60:.2f} minutes)")
    
    # Load best model
    model.load_state_dict(torch.load(Path(save_folder, "best_model.pt")))
    
    # Evaluate on test set
    logging.info("Evaluating on test set...")
    eval_start_time = time.time()
    test_loss, test_acc = validate(model, test_loader, criterion, device)
    eval_time = time.time() - eval_start_time
    logging.info(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.2f}% - Evaluation time: {eval_time:.2f}s")
    
    # Get predictions
    y_test, y_pred, y_pred_proba = evaluate_model(model, test_loader, device, label_encoder)
    
    # Save predictions
    predictions = pd.DataFrame({
        "y_test": y_test,
        "y_test_decoded": label_encoder.inverse_transform(y_test),
        "y_pred": y_pred,
        "y_pred_decoded": label_encoder.inverse_transform(y_pred)
    })
    predictions.to_csv(Path(save_folder, "predictions.csv"), index=False)
    
    # Calculate metrics per cancer type
    results = []
    for cancer in np.unique(y_test):
        y_test_cancer = y_test[y_test == cancer]
        y_pred_cancer = y_pred[y_test == cancer]
        cancer_name = label_encoder.inverse_transform([cancer])[0]
        
        accuracy_cancer = (y_test_cancer == y_pred_cancer).mean()
        f1_cancer = f1_score(y_test_cancer, y_pred_cancer, average='weighted', zero_division=0)
        precision_cancer = precision_score(y_test_cancer, y_pred_cancer, average='weighted', zero_division=0)
        recall_cancer = recall_score(y_test_cancer, y_pred_cancer, average='weighted', zero_division=0)
        
        logging.info(
            f"{cancer_name}: Accuracy: {accuracy_cancer:.4f}, F1: {f1_cancer:.4f}, "
            f"Precision: {precision_cancer:.4f}, Recall: {recall_cancer:.4f}")
        
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
    
    # Calculate overall metrics
    f1_total = f1_score(y_test, y_pred, average='weighted', zero_division=0)
    precision_total = precision_score(y_test, y_pred, average='weighted', zero_division=0)
    recall_total = recall_score(y_test, y_pred, average='weighted', zero_division=0)
    accuracy_total = (y_test == y_pred).mean()
    mcc_total = matthews_corrcoef(y_test, y_pred)
    balanced_accuracy = balanced_accuracy_score(y_test, y_pred)
    
    num_classes = y_pred_proba.shape[1]
    y_test_one_hot = label_binarize(y_test, classes=np.arange(num_classes))
    auc_score = roc_auc_score(y_test_one_hot, y_pred_proba, multi_class='ovo', average='macro')
    
    logging.info(
        f"Overall: Accuracy: {accuracy_total:.4f}, F1: {f1_total:.4f}, Precision: {precision_total:.4f}, "
        f"Recall: {recall_total:.4f}, AUC: {auc_score:.4f}, MCC: {mcc_total:.4f}, "
        f"Balanced Accuracy: {balanced_accuracy:.4f}")
    
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
    
    # Save results
    results_df = pd.DataFrame(results)
    results_df.to_csv(Path(save_folder, "results.csv"), index=False)
    logging.info("Results saved.")
    
    # Save model and history
    torch.save(model.state_dict(), Path(save_folder, "model.pt"))
    history_df = pd.DataFrame(history)
    history_df.to_csv(Path(save_folder, "history.csv"), index=False)
    logging.info("Model and history saved.")
    
    # Log total execution time
    total_time = time.time() - start_time
    logging.info(f"Total execution time: {total_time:.2f}s ({total_time/60:.2f} minutes)")
    
    # Save timing information
    timing_info = pd.DataFrame([{
        'training_time_seconds': training_time,
        'evaluation_time_seconds': eval_time,
        'total_time_seconds': total_time,
        'training_time_minutes': training_time / 60,
        'evaluation_time_minutes': eval_time / 60,
        'total_time_minutes': total_time / 60
    }])
    timing_info.to_csv(Path(save_folder, "timing.csv"), index=False)
    logging.info("Timing information saved.")


if __name__ == "__main__":
    script_start_time = time.time()
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", "-b", type=int, default=32)
    parser.add_argument("--cancer", "-c", nargs="+", default=["BRCA", "LUAD", "STAD", "BLCA", "COAD", "THCA"])
    parser.add_argument("--iteration", "-i", type=int, required=True)
    parser.add_argument("--walk_distance", "-w", type=int, choices=[3, 4, 5, 6], default=3)
    parser.add_argument("--amount_of_walks", "-a", type=int, choices=[3, 4, 5, 6], default=3)
    parser.add_argument("--modalities", "-m", nargs="+", default=["annotations", "images", "mutations", "rna"],
                        choices=["rna", "annotations", "mutations", "images"])
    parser.add_argument("--latent_dim", "-ld", type=int, required=True, help="The latent dimension of the embeddings")
    args = parser.parse_args()

    batch_size = args.batch_size
    walk_distance = args.walk_distance
    walk_amount = args.amount_of_walks
    iteration = args.iteration
    selected_modalities: List[str] = args.modalities
    latent_dim = args.latent_dim

    modalities = '_'.join(selected_modalities)
    cancers = "_".join(args.cancer)

    # Get device
    device = get_device()

    # Construct paths with latent_dim
    load_folder_path = Path(load_folder, cancers, modalities, str(latent_dim), f"{walk_distance}_{walk_amount}", str(iteration))
    train_h5_file_path = Path(load_folder_path, "train_summed_embeddings.h5")
    test_h5_file_path = Path(load_folder_path, "test_summed_embeddings.h5")

    cancer_save_folder = Path(save_folder, cancers, modalities, str(latent_dim), f"{walk_distance}_{walk_amount}")
    iteration_save_folder = Path(cancer_save_folder, str(iteration))
    iteration_save_folder.mkdir(parents=True, exist_ok=True)

    # Load data
    logging.info("Loading data from HDF5 files...")
    data_load_start = time.time()
    with h5py.File(train_h5_file_path, "r") as h5_file:
        feature_dimension = h5_file.attrs["feature_shape"]
        unique_classes = h5_file.attrs["classes"]
        train_X = h5_file["X"][:]
        train_y = np.array([label.decode("utf-8") for label in h5_file["y"][:]])

    with h5py.File(test_h5_file_path, "r") as h5_file:
        test_X = h5_file["X"][:]
        test_y = np.array([label.decode("utf-8") for label in h5_file["y"][:]])
    
    data_load_time = time.time() - data_load_start
    logging.info(f"Data loaded in {data_load_time:.2f}s")

    # Encode labels
    label_encoder = LabelEncoder()
    label_encoder.fit(unique_classes)

    # Split training data into train/val
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

    # Create datasets
    train_dataset = H5Dataset(train_h5_file_path, split_indices['train'], label_encoder)
    val_dataset = H5Dataset(train_h5_file_path, split_indices['val'], label_encoder)
    
    test_y_encoded = label_encoder.transform(test_y)
    test_dataset = ArrayDataset(test_X, test_y_encoded)

    # Create dataloaders
    # Disable pin_memory for MPS devices as it's not supported
    use_pin_memory = device.type not in ['mps']
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=use_pin_memory)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=use_pin_memory)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=use_pin_memory)

    # Class weights
    classes = list(label_encoder.classes_)
    decoded_classes = label_encoder.inverse_transform(np.arange(len(unique_classes)))
    
    class_weights = {classes.index(cancer): 1.0 for cancer in decoded_classes}
    class_weights[classes.index("LUAD")] = 6
    class_weights[classes.index("BLCA")] = 2.5

    # Train and evaluate
    train_and_evaluate_model(
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        num_classes=len(unique_classes),
        save_folder=iteration_save_folder,
        iteration=args.iteration,
        walk_distance=args.walk_distance,
        amount_of_walks=args.amount_of_walks,
        label_encoder=label_encoder,
        feature_dimension=feature_dimension,
        device=device,
        class_weights_dict=class_weights
    )
    
    # Log total script execution time
    total_script_time = time.time() - script_start_time
    logging.info(f"=" * 60)
    logging.info(f"Script completed successfully!")
    logging.info(f"Total script execution time: {total_script_time:.2f}s ({total_script_time/60:.2f} minutes)")
    logging.info(f"=" * 60)
