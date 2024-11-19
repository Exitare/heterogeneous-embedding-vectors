#!/bin/bash

# add range of values to iterate through
lower_bound=$1
upper_bound=$2
summed_embeddings_count=$3
cancer_types=$4

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

# if summed embeddings count not set, exit
if [ -z "$summed_embeddings_count" ]
then
  echo "Summed embeddings count not set, stopping."
  exit 1
fi


echo $cancer_types

# iterate through lower_bound to upper_bound
for i in $(seq $lower_bound $upper_bound)
do
  for noise_ratio in 0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0
  do
    sbatch ./src/recognizer/1_01_create_simple_sum_embeddings.sh $i $summed_embeddings_count "${cancer_types}" $noise_ratio
  done
done