# ./src/recognizer/2_03_schedule_multi_recognizer_jobs.sh "BRCA LUAD STAD BLCA COAD THCA"
cancer_types=$1
upper_walk_distance=$2
summed_embeddings_count=$3

# if cancer_types is not provided, then exit
if [ -z "$cancer_types" ]; then
  echo "No cancer types provided!"
  exit 1
fi

# if upper bound not set, set to 10
if [ -z "$upper_walk_distance" ]
then
  echo "Upper walk distance not set, setting to 10"
  upper_walk_distance=10
fi

# if no summed embeddings count is provided, set to 1000
if [ -z "$summed_embeddings_count" ]
then
  echo "Summed embeddings count not set, setting to 1000"
  summed_embeddings_count=1000
fi

for walk_distance in $(seq 2 $upper_walk_distance)
do
  # run it 30 times
  for iteration in $(seq 1 30)
  do
    sbatch ./src/recognizer/2_03_run_multi_recognizer.sh $walk_distance $iteration "${cancer_types}" $summed_embeddings_count
  done
done