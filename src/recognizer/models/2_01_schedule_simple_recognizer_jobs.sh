
# upper bound is the maximum walk distance
upper_bound=$1
amount_of_summed_embeddings=$2


# if upper bound not set, set to 30
if [ -z "$upper_bound" ]
then
  echo "Upper bound not set, setting to 30"
  upper_bound=30
fi

# if summed embeddings count not set, set to 1000000
if [ -z "$amount_of_summed_embeddings" ]
then
  echo "Amount of summed embeddings count not set, setting to 1000000"
  amount_of_summed_embeddings=1000000
  exit 1
fi




for walk_distance in $(seq 3 $upper_bound)
do
  # run it 30 times
  for iteration in $(seq 1 30)
  do
    sbatch ./src/recognizer/models/2_01_run_simple_recognizer.sh $walk_distance $iteration $amount_of_summed_embeddings 0.0
  done
done