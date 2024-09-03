
# add range of values to iterate through
upper_bound=$1
summed_embeddings_count=$2


# if upper bound not set, set to 10
if [ -z "$upper_bound" ]
then
  echo "Upper bound not set, setting to 10"
  upper_bound=10
fi

# if summed embeddings count not set, exit
if [ -z "$summed_embeddings_count" ]
then
  echo "Summed embeddings count not set, stopping."
  exit 1
fi




for walk_distance in $(seq 2 $upper_bound)
do
  # run it 30 times
  for iteration in $(seq 1 30)
  do
    sbatch ./src/recognizer/2_01_run_simple_recognizer.sh $walk_distance $iteration $summed_embeddings_count
  done
done