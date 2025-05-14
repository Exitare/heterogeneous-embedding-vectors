
selected_cancers=$1
walk_distance=(3 4 5)
amount_of_walks=(3 4 5)

# iterate through all possible combinations of walk distance and walk amount
for walk_distance in "${walk_distance[@]}"
do
  for amount in "${amount_of_walks[@]}"
  do
     ./src/sub_type_classifier/summed_embeddings_generation/3_01_run_summed_embeddings.sh  "${selected_cancers}"  $walk_distance  $amount
  done
done