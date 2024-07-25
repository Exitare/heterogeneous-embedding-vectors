import os
from tensorflow.keras.layers import Input, Dense, BatchNormalization, Dropout, ReLU
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from pathlib import Path
from argparse import ArgumentParser

embeddings = ['Text', 'Image', 'RNA']
save_path = Path("results", "text_rna_recognizer")
embedding_counts = [2, 3, 4, 5, 6, 7, 8, 9, 10]
epochs = 100
total_embeddings = 10


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
    text_output = Dense(1, activation=ReLU(max_value=total_embeddings), name='output_text')(text_x)

    # Path for RNA embeddings, including subtype classification
    rna_x = Dense(128, activation='relu', name='rna_dense_1')(x)
    rna_x = Dropout(0.2)(rna_x)
    rna_x = Dense(64, activation='relu', name='rna_dense_2')(rna_x)
    rna_x = BatchNormalization()(rna_x)
    rna_x = Dropout(0.2)(rna_x)
    rna_x = Dense(32, activation='relu', name='rna_dense_3')(rna_x)
    rna_output = Dense(1, activation=ReLU(max_value=total_embeddings), name='output_rna')(rna_x)

    cancer_outputs = [Dense(1, activation=ReLU(max_value=total_embeddings), name=f'output_cancer_{cancer_type}')(x) for
                      cancer_type in cancer_list]

    # Combine all outputs
    outputs = [text_output, rna_output] + cancer_outputs

    # Create model
    return Model(inputs=inputs, outputs=outputs, name='multi_output_model')


if __name__ == '__main__':
    if not save_path.exists():
        save_path.mkdir(parents=True)

    parser = ArgumentParser(description='Train a multi-output model for recognizing embeddings')
    parser.add_argument('--batch_size', "-bs", type=int, default=32, help='The batch size to train the model')
    parser.add_argument("--run_iteration", "-ri", type=int, required=False,
                        help="The iteration number for the run. Used for saving the results and validation.", default=1)
    parser.add_argument("--cancer", "-c", nargs="+", required=True, help="The cancer types to work with.")
    args = parser.parse_args()

    batch_size = args.batch_size
    run_iteration = args.run_iteration
    selected_cancers = args.cancer

    print("Selected cancers: ", selected_cancers)

    # lower case the cancer types
    selected_cancers = [cancer.lower() for cancer in selected_cancers]
    cancer_types = "_".join(selected_cancers)

    save_path = Path(save_path, cancer_types)

    print(f"Batch size: {batch_size}")
    print(f"Run iteration: {run_iteration}")

    run_name = f"run_{run_iteration}"
    save_path = Path(save_path, run_name)

    if not save_path.exists():
        save_path.mkdir(parents=True)

    print(f"Saving results to {save_path}")
