import numpy as np
import pandas as pd
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, BatchNormalization, Dropout, ReLU
from argparse import ArgumentParser
from pathlib import Path

graph_generation_folder = Path("results", "graph_embeddings")
disaggregated_folder = Path("results", "graph_embeddings")

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

    # Less complex paths for image output
    image_output = Dense(1, activation=ReLU(max_value=total_embeddings), name='output_image')(x)

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
    outputs = [text_output, image_output, rna_output] + cancer_outputs

    # Create model
    return Model(inputs=inputs, outputs=outputs, name='multi_output_model')


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--cancer", "-c", nargs="+", required=True, help="The cancer types to work with.")

    args = parser.parse_args()

    selected_cancers = args.cancer
    cancers = "_".join(selected_cancers)

    graph_generation_folder = Path(graph_generation_folder, cancers, "graph_generation")
    disaggregated_folder = Path(disaggregated_folder, cancers, "disaggregated_embeddings")

    disaggregated_nodes: pd.DataFrame = pd.read_csv(Path(disaggregated_folder, "unique_nodes.csv"))
    combined_embeddings: pd.DataFrame = pd.read_csv(Path(graph_generation_folder, "combined_embeddings.csv"))

    print(disaggregated_nodes)
    #print(combined_embeddings)
    # select the embeddings for the unique nodes using the index from the combing embeddigs
    unique_node_embeddings = combined_embeddings.loc[disaggregated_nodes['Node']]
    print(unique_node_embeddings)


