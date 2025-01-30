
selected_cancers=$1
walk_distance=(3 4 5 6)
amount_of_walks=(3 4 5 6)

# iterate through all possible combinations of walk distance and walk amount
for walk_distance in "${walk_distance[@]}"
do
  for amount in "${amount_of_walks[@]}"
  do
    python ./src/classifier/3_01_create_summed_embeddings_v3.py -c ${selected_cancers} -w $i -a $j
  done
done