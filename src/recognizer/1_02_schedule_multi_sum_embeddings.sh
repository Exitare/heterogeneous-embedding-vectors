#!/bin/bash
selected_cancers=$1 # the data files
# ./src/recognizer/1_02_schedule_multi_sum_embeddings.sh "BRCA LUAD STAD BLCA COAD THCA"

# iterate through 2 to 10
for i in $(seq 2 10)
do
  sbatch ./src/recognizer/1_02_create_multi_sum_embeddings.sh $i "${selected_cancers}"
done