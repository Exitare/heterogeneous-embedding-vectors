
cancer_types=$1

# iterate through 2 to 10
for i in $(seq 2 10)
  sbatch ./src/embedding_aggregation/create_multi_sum_embedding.sh $i $cancer_types