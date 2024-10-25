#!/bin/bash

# add range of values to iterate through
lower_bound=$1
upper_bound=$2
summed_embeddings_count=$3

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



# iterate through lower_bound to upper_bound
for i in $(seq $lower_bound $upper_bound)
  for noise in $(seq 0 0.1 1)
    do
      sbatch ./src/noise_recognizer/1_01_create_simple_sum_embeddings.sh $i $summed_embeddings_count $noise
    done
  done
done