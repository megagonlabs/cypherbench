set -e

all_domains="art biology company fictional_character flight_accident geography movie nba politics soccer terrorist_attack"
train_domains="biology terrorist_attack soccer art"
test_domains="geography flight_accident company movie politics fictional_character nba"
seed=42
num_revision_round=3

if [ "$#" -ne 1 ]; then
  echo "Usage: $0 <output_dir>"
  exit 1
fi

# Remove trailing slash from the output_dir if it exists
output_dir="${1%/}"

if [ -d "$output_dir" ]; then
    echo "Output directory already exists, exiting..."
    exit 1
fi

mkdir "$output_dir"
mkdir "$output_dir/logs"

for graph in $train_domains; do
  python -u -m cypherbench.taskgen.generate_benchmark --graphs $graph --output_path "${output_dir}/${graph}.json" \
    --seed $seed --sample_config_id "train" --num_threads 64 --debug 2>&1 | tee "${output_dir}/logs/${graph}.log" &
done

for graph in $test_domains; do
  python -u -m cypherbench.taskgen.generate_benchmark --graphs $graph --output_path "${output_dir}/${graph}.json" \
    --seed $seed --sample_config_id "test" --num_threads 64 --debug 2>&1 | tee "${output_dir}/logs/${graph}.log" &
done

wait

paths=""
for graph in $test_domains; do
  paths="${paths} ${output_dir}/${graph}.json"
done
python -u -m cypherbench.taskgen.filter_long_running --benchmark_path $paths --output_path "${output_dir}/test_filtered.json"
python -u -m cypherbench.taskgen.rewrite_question_llm --benchmark_path "${output_dir}/test_filtered.json" --output_path "${output_dir}/test_rewritten_0.json"

for round in $(seq 1 $num_revision_round); do
  prev_round=$((round - 1))
  python -u -m cypherbench.taskgen.verify_question_llm \
    --benchmark_path "${output_dir}/test_rewritten_${prev_round}.json" \
    --output_path "${output_dir}/test_rewritten_${round}.json" \
    --verification_output_path "${output_dir}/bad_samples_test_${round}.json"
done

cp "${output_dir}/test_rewritten_${num_revision_round}.json" "${output_dir}/test.json"
echo "Test split is saved at ${output_dir}/test.json"

paths=""
for graph in $train_domains; do
  paths="${paths} ${output_dir}/${graph}.json"
done
python -u -m cypherbench.taskgen.filter_long_running --benchmark_path $paths --output_path "${output_dir}/train_filtered.json"
python -u -m cypherbench.taskgen.rewrite_question_llm --benchmark_path "${output_dir}/train_filtered.json" --output_path "${output_dir}/train_rewritten.json"
python -u -m cypherbench.taskgen.verify_question_llm \
    --benchmark_path "${output_dir}/train_rewritten.json" \
    --output_path "${output_dir}/train.json" \
    --verification_output_path "${output_dir}/bad_samples_train.json" \
    --remove_bad

echo "Train split is saved at ${output_dir}/train.json"

python -u scripts/print_benchmark_stats.py --benchmark_path "${output_dir}/train.json"
python -u scripts/print_benchmark_stats.py --benchmark_path "${output_dir}/test.json"
