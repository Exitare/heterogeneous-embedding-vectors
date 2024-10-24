#!/bin/bash
#./src/classifier/4_01_cancer_classifier.sh "BRCA BLCA LUAD STAD THCA COAD"
cancers=$1
walk_distance=$2
walk_amount=$3

# Check if cancers is empty
if [ -z "$cancers" ]; then
  echo "Error: No cancers provided. Please specify the cancers to classify."
  exit 1
fi

# if walk_distance is empty, set to 1
if [ -z "$walk_distance" ]; then
  walk_distance=3
  echo "walk_distance is empty, setting to 3"
fi

# if walk_amount is empty, set to 1
if [ -z "$walk_amount" ]; then
  walk_amount=3
  echo "walk_amount is empty, setting to 3"
fi

# iterate 30 times
for i in {1..30}
do
  # call script
  python src/classifier/4_01_cancer_classifier.py -c ${cancers} -i ${i} -w ${walk_distance} -a ${walk_amount}
done
