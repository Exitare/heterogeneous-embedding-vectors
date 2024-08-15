# ./src/recognizer/2_03_schedule_multi_recognizer_jobs.sh "BRCA LUAD STAD BLCA COAD THCA"
cancer_types=$1

# if cancer_types is not provided, then exit
if [ -z "$cancer_types" ]; then
  echo "No cancer types provided!"
  exit 1
fi

for i in $(seq 2 9)
do
  # run it 30 times
  for j in $(seq 1 30)
  do
    sbatch ./src/recognizer/2_03_run_multi_recognizer.sh $i $j "${cancer_types}"
  done
done