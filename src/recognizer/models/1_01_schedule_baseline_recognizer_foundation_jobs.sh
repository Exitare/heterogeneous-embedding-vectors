
cancer_types=$1
amount_of_summed_embeddings=$2
multi=$3
local=$4


# if cancer_types is not provided, then exit
if [ -z "$cancer_types" ]; then
  echo "No cancer types provided!"
  exit 1
fi


# if summed embeddings count not set, set to 100000
if [ -z "$amount_of_summed_embeddings" ]
then
  echo "Amount of summed embeddings count not set, set to 100000"
  amount_of_summed_embeddings=100000
fi

echo "Amount of summed embeddings: $amount_of_summed_embeddings"
echo "cancer types: $cancer_types"
echo "multi: $multi"

if [ -z "$local" ]
then
  echo "Running on cluster"
else
  echo "Running locally"
fi

  # run it 30 times
for iteration in $(seq 1 30)
do
  if [ -z "$local" ]
  then
    sbatch ./src/recognizer/models/1_01_run_baseline_recognizer.sh "${cancer_types}" -1 $iteration $amount_of_summed_embeddings 0.0 $multi
  else
    ./src/recognizer/models/1_01_run_baseline_recognizer.sh "${cancer_types}" -1 $iteration $amount_of_summed_embeddings 0.0 $multi
  fi
done