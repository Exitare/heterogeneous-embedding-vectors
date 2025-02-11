
cancer_types=$1
amount_of_summed_embeddings=$2
# upper bound is the maximum walk distance
upper_bound=$3
local=$4


# if cancer_types is not provided, then exit
if [ -z "$cancer_types" ]; then
  echo "No cancer types provided!"
  exit 1
fi

# if upper bound not set, set to 10
if [ -z "$upper_bound" ]
then
  echo "Upper bound not set, setting to 10"
  upper_bound=10
fi

# if summed embeddings count not set, set to 100000
if [ -z "$amount_of_summed_embeddings" ]
then
  echo "Amount of summed embeddings count not set, set to 100000"
  amount_of_summed_embeddings=100000
fi


echo "Upper bound: $upper_bound"
echo "Amount of summed embeddings: $amount_of_summed_embeddings"
echo "cancer types: $cancer_types"

if [ -z "$local" ]
then
  echo "Running on cluster"
else
  echo "Running locally"
fi


for walk_distance in $(seq 3 $upper_bound)
do
  # run it 30 times
  for iteration in $(seq 1 30)
  do
    if [ -z "$local" ]
    then
      sbatch ./src/recognizer/models/2_01_run_simple_recognizer.sh "${cancer_types}" $walk_distance $iteration $amount_of_summed_embeddings 0.0
    else
      ./src/recognizer/models/2_01_run_simple_recognizer.sh "${cancer_types}" $walk_distance $iteration $amount_of_summed_embeddings 0.0
    fi
  done
done