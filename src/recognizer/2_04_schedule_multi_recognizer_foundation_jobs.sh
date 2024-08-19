# ./src/recognizer/2_04_schedule_multi_recognizer_foundation_jobs.sh "BRCA LUAD STAD BLCA COAD THCA"

cancer_types=$1

# if cancer_types is not provided, then exit
if [ -z "$cancer_types" ]; then
  echo "No cancer types provided!"
  exit 1
fi

# run it 30 times
for i in $(seq 1 30)
do
  sbatch ./src/recognizer/2_04_run_multi_recognizer_foundation.sh $i "${cancer_types}"
done
