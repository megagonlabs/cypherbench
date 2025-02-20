#!/bin/bash

benchmark_dir="../benchmark"

# Check the existence of the graphs folder
if [ ! -d $benchmark_dir ]; then
  echo "Error: The folder $benchmark_dir does not exist."
  exit 1
fi

# List of expected graph JSON files
declare -a graphs=(
  "art_simplekg.json"
  "biology_simplekg.json"
  "soccer_simplekg.json"
  "terrorist_attack_simplekg.json"
)

# Check if all graph files exist
missing_files=false
for graph in "${graphs[@]}"; do
  graph_path="$benchmark_dir/graphs/simplekg/$graph"
  if [ ! -f $graph_path ]; then
    echo "Error: Missing graph file $graph_path"
    missing_files=true
  fi
done

# If any files are missing, exit the script
if [ "$missing_files" = true ]; then
  echo "One or more graph files are missing. Exiting."
  exit 1
fi

# Run docker-compose
echo "All required graph files are present. Starting Docker Compose."
docker-compose -f docker-compose-train.yml -p cypherbench_train up -d
