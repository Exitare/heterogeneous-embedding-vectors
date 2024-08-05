#!/bin/bash
#./src/realistic_recognizer/041_cancer_classifier.sh "BRCA BLCA LAML STAD THCA COAD"
cancers=$1

# iterate 30 times
for i in {1..30}
do
  # call script
  python src/realistic_recognizer/041_cancer_classifier.py -c ${cancers} -i ${i}
done