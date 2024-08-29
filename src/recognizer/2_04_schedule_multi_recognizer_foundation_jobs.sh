# ./src/recognizer/2_04_schedule_multi_recognizer_foundation_jobs.sh "BRCA LUAD STAD BLCA COAD THCA"

cancer_types=$1
upper_walk_distance=$2

# if cancer_types is not provided, then exit
if [ -z "$cancer_types" ]; then
  echo "No cancer types provided!"
  exit 1
fi

# if upper_walk_distance is not provided, set to 10
if [ -z "$upper_walk_distance" ]; then
  echo "No upper walk distance provided!"
  upper_walk_distance=10
fi

# run it 30 times
for i in $(seq 1 30)
do
  sbatch ./src/recognizer/2_04_run_multi_recognizer_foundation.sh $i "${cancer_types}" "${upper_walk_distance}"
done
