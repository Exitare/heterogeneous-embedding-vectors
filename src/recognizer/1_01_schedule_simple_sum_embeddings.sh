#!/bin/bash

# add range of values to iterate through
lower_bound=$1
upper_bound=$2

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


# iterate through lower_bound to upper_bound
for i in $(seq $lower_bound $upper_bound)
do
  sbatch ./src/recognizer/1_01_create_simple_sum_embeddings.sh $i
done