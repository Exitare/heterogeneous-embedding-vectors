#!/bin/bash
cancer_embedding_files=$1 # the data files
# ./src/embedding_aggregation/schedule_multi_sum_embeddings.sh "./results/embeddings/cancer/brca_embeddings.csv ./results/embeddings/cancer/laml_embeddings.csv"


# iterate through 2 to 10
for i in $(seq 2 10)
do
  sbatch ./src/recognizer/1_02_create_multi_sum_embeddings.sh $i "${cancer_embedding_files}"
done