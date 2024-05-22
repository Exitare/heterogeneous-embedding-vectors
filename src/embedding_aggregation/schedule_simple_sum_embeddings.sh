
# iterate through 2 to 10
for i in $(seq 2 10)
  sbatch ./src/embedding_aggregation/create_simple_sum_embeddings.sh $i