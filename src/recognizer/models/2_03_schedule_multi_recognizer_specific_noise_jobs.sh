# ./src/recognizer/models/2_03_schedule_multi_recognizer_jobs.sh "BRCA LUAD STAD BLCA COAD THCA"
cancer_types=$1
amount_of_summed_embeddings=$2
upper_walk_distance=$3
noise=$4
local=$5

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

# if no summed embeddings count is provided, set to 100000
if [ -z "$amount_of_summed_embeddings" ]
then
  echo "Amount of summed embeddings not set, setting to 100000"
  amount_of_summed_embeddings=100000
fi

if [ -z "$noise" ]
then
  echo "Noise not set, aborting"
  exit 1
fi

if [ -z "$local" ]
then
  echo "Running on cluster"
else
  echo "Running locally"
fi

echo "cancer types: $cancer_types"
echo "Amount of summed embeddings: $amount_of_summed_embeddings"
echo "Upper walk distance: $upper_walk_distance"
echo "Noise: $noise"

for walk_distance in $(seq 3 $upper_walk_distance)
do
# run it 30 times
  for iteration in $(seq 1 30)
  do
    if [ -z "$local" ]
    then
      sbatch ./src/recognizer/models/2_03_run_multi_recognizer.sh $walk_distance $iteration "${cancer_types}" $amount_of_summed_embeddings $noise
    else
      ./src/recognizer/models/2_03_run_multi_recognizer.sh $walk_distance $iteration "${cancer_types}" $amount_of_summed_embeddings $noise
    fi
  done
done
