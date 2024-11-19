#!/bin/bash
selected_cancers=$1 # the data files
lower_bound=$2
upper_bound=$3

# ./src/recognizer/1_02_schedule_multi_sum_embeddings.sh "BRCA LUAD STAD BLCA COAD THCA"

# if selected_cancers is not provided, then exit
if [ -z "$selected_cancers" ]; then
  echo "No cancer types provided!"
  exit 1
fi

# if lower bound not set, set to 2
if [ -z "$lower_bound" ]
then
  echo "Lower bound not set, setting to 2"
  lower_bound=2
fi

# if upper bound not set, set to 10
if [ -z "$upper_bound" ]
then
  echo "Upper bound not set, setting to 10"
  upper_bound=10
fi
´

# iterate through lower_bound to upper_bound
for i in $(seq $lower_bound $upper_bound)
do
  for noise_ratio in 0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0
  do
    sbatch ./src/recognizer/1_02_create_multi_sum_embeddings.sh $i "${selected_cancers}" $noise_ratio
  done
done