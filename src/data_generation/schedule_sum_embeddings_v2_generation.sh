
# iterate through 2 to 10
for i in $(seq 2 10)
  sbatch ./src/data_generation/schedule_sum_embeddings_v2_generation.sh $i