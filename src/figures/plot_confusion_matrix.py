
# conf_matrices = [confusion_matrix(y_true, y_pred) for y_true, y_pred in zip(y_test_binary, y_pred_binary)]

# Print confusion matrices
for i, cm in enumerate(conf_matrices):
    print(f"Confusion Matrix for output {i} (Text, Image, RNA - respective order):")
    print(cm)

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
for i, ax in enumerate(axes):
    sns.heatmap(conf_matrices[i], annot=True, fmt='d', cmap='Blues', ax=ax)
    ax.set_title(f"Confusion Matrix for output {i}")
    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')
plt.tight_layout()
plt.savefig(Path("../../results", "binary_confusion_matrices.png"), dpi=300)

fig, axes = plt.subplots(1, 3, figsize=(18, 6))  # Adjust size as needed

for i, (pred, true, ax) in enumerate(zip(y_pred_rounded, y_test, axes.flatten())):
    # Compute confusion matrix
    cm = confusion_matrix(true, pred)

    # Plot confusion matrix
    sns.heatmap(cm, annot=True, fmt='g', ax=ax, cmap='Blues', cbar=False)
    ax.set_title(f'Confusion Matrix for {embeddings[i]}')
    ax.set_xlabel('Predicted Labels')
    ax.set_ylabel('True Labels')

    # Setting tick labels if needed (remove if not necessary or adjust as needed)
    classes = np.unique(np.concatenate((true, pred)))  # Get unique classes from true and predicted
    ax.set_xticklabels(classes)
    ax.set_yticklabels(classes)

plt.tight_layout()
plt.savefig(Path("..""../../results", "confusion_matrices.png"), dpi=300)
