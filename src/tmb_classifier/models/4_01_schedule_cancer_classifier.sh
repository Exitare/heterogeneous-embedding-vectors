#!/bin/bash
#./src/tmb_classifier/models/4_01_schedule_cancer_classifier.sh "BRCA LUAD STAD BLCA COAD THCA"
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
  src/tmb_classifier/models/4_01_cancer_classifier.sh "${cancers}" ${walk_distance} ${walk_amount} ${i}
done
