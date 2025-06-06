#!/bin/bash
selected_cancers=$1 # cancer types
amount_of_summed_embeddings=$2
lower_bound=$3
upper_bound=$4

# ./src/recognizer/summed_embeddings_generation/1_02_schedule_multi_sum_embeddings.sh "BRCA LUAD STAD BLCA COAD THCA"

# if selected_cancers is not provided, then exit
if [ -z "$selected_cancers" ]; then
  echo "No cancer types provided!"
  exit 1
fi

# if lower bound not set, set to 3
if [ -z "$lower_bound" ]
then
  echo "Lower bound not set, setting to 3"
  lower_bound=3
fi

# if upper bound not set, set to 10
if [ -z "$upper_bound" ]
then
  echo "Upper bound not set, setting to 10"
  upper_bound=10
fi

# if no summed embeddings count is provided, set to 1000
if [ -z "$amount_of_summed_embeddings" ]
then
  echo "Amount of summed embeddings not set, setting to 1000000"
  amount_of_summed_embeddings=1000000
fi

echo $selected_cancers

# iterate through lower_bound to upper_bound
for walk_distance in $(seq $lower_bound $upper_bound)
do
  for noise_ratio in 0.0 0.1 0.2 0.3 0.4 0.5 0.6
  do
    ./src/recognizer/summed_embeddings_generation/1_02_create_multi_sum_embeddings.sh $walk_distance "${selected_cancers}" "${amount_of_summed_embeddings}" $noise_ratio
  done
done