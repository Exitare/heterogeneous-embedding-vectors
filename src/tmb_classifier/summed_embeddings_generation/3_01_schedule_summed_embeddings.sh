
selected_cancers=$1
walk_distance=(3 4 5 6)
amount_of_walks=(3 4 5 6)

# iterate through all possible combinations of walk distance and walk amount
for walk_distance in "${walk_distance[@]}"
do
  for amount in "${amount_of_walks[@]}"
  do
    sbatch ./src/classifier/3_01_run_summed_embeddings.sh -c ${selected_cancers} -w $walk_distance -a $amount
  done
done