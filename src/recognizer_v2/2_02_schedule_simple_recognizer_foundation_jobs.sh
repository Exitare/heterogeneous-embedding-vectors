upper_walk_distance=$1
summed_embeddings_count=$2

# if upper bound not set, set to 10
if [ -z "$upper_walk_distance" ]
then
  echo "Upper walk distance not set, setting to 10"
  upper_walk_distance=10
fi

# if summed embeddings count not set, exit
if [ -z "$summed_embeddings_count" ]
then
  echo "Summed embeddings count not set, stopping."
  exit 1
fi

# run it 30 times
for i in $(seq 1 30)
do
  sbatch ./src/recognizer/2_02_run_simple_recognizer_foundation.sh $i $upper_walk_distance $summed_embeddings_count
done
