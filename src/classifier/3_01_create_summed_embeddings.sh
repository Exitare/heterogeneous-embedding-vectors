
selected_cancers=$1
walk_distance=(3 4 5)
walk_amount=(3 4 5)

# iterate thorugh all possible combinations of walk distance and walk amount
for i in "${walk_distance[@]}"
do
  for j in "${walk_amount[@]}"
  do
    python ./src/classifier/3_01_create_summed_embeddings.py -c ${selected_cancers} -w $i -a $j
  done
done