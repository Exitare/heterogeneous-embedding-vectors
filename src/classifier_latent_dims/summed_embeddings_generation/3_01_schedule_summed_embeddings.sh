
selected_cancers=$1
walk_distance=(5)
amount_of_walks=(5)

# iterate through all possible combinations of walk distance and walk amount
for walk_distance in "${walk_distance[@]}"
do
  for amount in "${amount_of_walks[@]}"
  do
    sbatch ./src/classifier/3_01_run_summed_embeddings -c ${selected_cancers} -w $walk_distance -a $amount
  done
done