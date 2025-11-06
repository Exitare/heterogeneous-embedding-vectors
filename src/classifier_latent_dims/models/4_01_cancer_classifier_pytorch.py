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
import time

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Device detection (MPS for Apple Silicon, CUDA for NVIDIA, CPU fallback)
if torch.backends.mps.is_available():
    device = torch.device("mps")
    print(f"Using MPS (Apple Silicon GPU)")
elif torch.cuda.is_available():
    device = torch.device("cuda")
    print(f"Using CUDA GPU: {torch.cuda.get_device_name(0)}")
else:
    device = torch.device("cpu")
    print("Using CPU")


class H5Dataset(Dataset):
    """Dataset for loading from HDF5 file with specific indices"""
    def __init__(self, h5_file_path, indices, label_encoder):
        with h5py.File(h5_file_path, 'r') as h5_file:
            self.X = h5_file["X"][:][indices]
            y = [label.decode("utf-8") for label in h5_file["y"][:]]
            self.y = label_encoder.transform([y[i] for i in indices])
        
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return torch.FloatTensor(self.X[idx]), torch.LongTensor([self.y[idx]])[0]


class ArrayDataset(Dataset):
    """Dataset for loading from numpy arrays"""
    def __init__(self, X, y):
        self.X = X.astype(np.float32)
        self.y = y.astype(np.int64)
        
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return torch.FloatTensor(self.X[idx]), torch.LongTensor([self.y[idx]])[0]


class CancerClassifier(nn.Module):
    def __init__(self, feature_dimension, num_classes):
        super(CancerClassifier, self).__init__()
        
        self.model = nn.Sequential(
            nn.BatchNorm1d(feature_dimension),
            nn.Linear(feature_dimension, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )
        
    def forward(self, x):
        return self.model(x)


def train_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    for X_batch, y_batch in dataloader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        
        optimizer.zero_grad()
        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += y_batch.size(0)
        correct += (predicted == y_batch).sum().item()
    
    return total_loss / len(dataloader), correct / total


def validate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for X_batch, y_batch in dataloader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            
            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += y_batch.size(0)
            correct += (predicted == y_batch).sum().item()
    
    return total_loss / len(dataloader), correct / total


def train_and_evaluate_model(train_loader, val_loader, test_loader, num_classes: int, save_folder: Path, 
                            iteration: int, walk_distance: int, amount_of_walks: int, label_encoder, 
                            feature_dimension: int, class_weights: dict):
    
    model = CancerClassifier(feature_dimension, num_classes).to(device)
    optimizer = optim.Adam(model.parameters())
    
    # Convert class weights to tensor
    class_weights_list = [class_weights.get(i, 1.0) for i in range(num_classes)]
    class_weights_tensor = torch.FloatTensor(class_weights_list).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)
    
    # Training
    history = {'loss': [], 'accuracy': [], 'val_loss': [], 'val_accuracy': []}
    best_val_acc = 0
    patience = 5
    patience_counter = 0
    best_model_state = None
    
    logging.info("Starting training...")
    train_start = time.time()
    
    for epoch in range(50):
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        
        history['loss'].append(train_loss)
        history['accuracy'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_accuracy'].append(val_acc)
        
        logging.info(f"Epoch {epoch+1}/50 - loss: {train_loss:.4f} - accuracy: {train_acc:.4f} - "
                    f"val_loss: {val_loss:.4f} - val_accuracy: {val_acc:.4f}")
        
        # Early stopping
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            best_model_state = model.state_dict().copy()
        else:
            patience_counter += 1
            if patience_counter >= patience:
                logging.info(f"Early stopping at epoch {epoch+1}")
                break
    
    train_time = time.time() - train_start
    logging.info(f"Training completed in {train_time:.2f} seconds")
    
    # Restore best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    
    # Evaluation
    logging.info("Evaluating on test set...")
    eval_start = time.time()
    
    model.eval()
    y_test = []
    y_pred = []
    y_pred_proba = []
    
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch = X_batch.to(device)
            outputs = model(X_batch)
            probs = torch.softmax(outputs, dim=1)
            
            y_test.extend(y_batch.numpy())
            y_pred.extend(outputs.argmax(dim=1).cpu().numpy())
            y_pred_proba.extend(probs.cpu().numpy())
    
    y_test = np.array(y_test)
    y_pred = np.array(y_pred)
    y_pred_proba = np.array(y_pred_proba)
    
    eval_time = time.time() - eval_start
    logging.info(f"Evaluation completed in {eval_time:.2f} seconds")
    
    # Save predictions
    predictions = pd.DataFrame({
        "y_test": y_test,
        "y_test_decoded": label_encoder.inverse_transform(y_test),
        "y_pred": y_pred,
        "y_pred_decoded": label_encoder.inverse_transform(y_pred)
    })
    predictions.to_csv(Path(save_folder, f"predictions.csv"), index=False)
    
    # Calculate metrics per cancer
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
            f"Precision: {precision_cancer:.4f}, Recall: {recall_cancer:.4f}.")
        
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
    
    # Overall metrics
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
    
    results_df = pd.DataFrame(results)
    results_df.to_csv(Path(save_folder, f"results.csv"), index=False)
    logging.info("Results saved.")
    
    # Save model and history
    torch.save(model.state_dict(), Path(save_folder, f"model.pt"))
    history_df = pd.DataFrame(history)
    history_df.to_csv(Path(save_folder, f"history.csv"), index=False)
    
    # Save timing benchmark
    total_time = train_time + eval_time
    benchmark_df = pd.DataFrame([{
        'framework': 'pytorch',
        'train_time': train_time,
        'eval_time': eval_time,
        'total_time': total_time,
        'iteration': iteration,
        'walk_distance': walk_distance,
        'amount_of_walks': amount_of_walks
    }])
    benchmark_df.to_csv(Path(save_folder, f"benchmark.csv"), index=False)
    
    logging.info("Model, history, and benchmark saved.")


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

    script_start = time.time()

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

    # Class weights
    classes = list(label_encoder.classes_)
    decoded_classes = label_encoder.inverse_transform(np.arange(len(unique_classes)))
    class_weights = {classes.index(cancer): 1.0 for cancer in decoded_classes}
    class_weights[classes.index("LUAD")] = 6
    class_weights[classes.index("BLCA")] = 2.5

    # Split train into train/val
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

    # Create datasets and dataloaders
    train_dataset = H5Dataset(train_h5_file_path, split_indices['train'], label_encoder)
    val_dataset = H5Dataset(train_h5_file_path, split_indices['val'], label_encoder)
    
    test_y_encoded = label_encoder.transform(test_y)
    test_dataset = ArrayDataset(test_X, test_y_encoded)

    # Disable pin_memory for MPS devices (not supported)
    pin_memory = device.type == "cuda"
    
    # Use drop_last=True to avoid batch_size=1 issues with BatchNorm
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=pin_memory, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, pin_memory=pin_memory, drop_last=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, pin_memory=pin_memory)

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
        class_weights=class_weights
    )

    total_script_time = time.time() - script_start
    logging.info(f"Total script runtime: {total_script_time:.2f} seconds")
