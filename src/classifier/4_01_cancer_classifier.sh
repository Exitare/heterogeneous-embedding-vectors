#!/bin/bash
#./src/classifier/4_01_cancer_classifier.sh "BRCA BLCA LAML STAD THCA COAD"
cancers=$1

# Check if cancers is empty
if [ -z "$cancers" ]; then
  echo "Error: No cancers provided. Please specify the cancers to classify."
  exit 1
fi

# iterate 30 times
for i in {1..30}
do
  # call script
  python src/classifier/4_01_cancer_classifier.py -c ${cancers} -i ${i}
done
