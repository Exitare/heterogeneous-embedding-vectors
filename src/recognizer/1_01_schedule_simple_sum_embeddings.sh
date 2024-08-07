#!/bin/bash
# iterate through 2 to 10
for i in $(seq 2 10)
do
  sbatch ./src/recognizer/1_01_create_simple_sum_embeddings.sh $i
done