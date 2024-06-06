#!/bin/bash
cancer_embeddings=$1 # the data files
# ./src/embedding_aggregation/schedule_multi_sum_embeddings.sh "./results/embeddings/cancer/brca_embeddings.csv ./results/embeddings/cancer/stad_embeddings.csv"


# iterate through 2 to 10
for i in $(seq 2 10)
do
  sbatch ./src/embedding_aggregation/create_multi_sum_embeddings.sh $i "${cancer_embeddings}"
done