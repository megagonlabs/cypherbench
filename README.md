# CypherBench

[![](https://img.shields.io/badge/license-apache2.0-green.svg)](LICENSE) 
[![](https://img.shields.io/badge/🤗-HuggingFace-red.svg)](https://huggingface.co/datasets/megagonlabs/cypherbench)
[![](https://img.shields.io/badge/paper-arxiv-yellow.svg)](https://arxiv.org/pdf/2412.18702)

## 🔥 Updates

- [Feb 14, 2025] We have released the nl2cypher baseline code! See the instructions below on how to run `gpt-4o-mini` on CypherBench.
- [Feb 13, 2025] The [11 property graphs](https://huggingface.co/datasets/megagonlabs/cypherbench/tree/main/graphs) are now available on 🤗HuggingFace!
- [Dec 27, 2024] We have deployed a [demo NBA graph](https://browser.neo4j.io/?dbms=neo4j%2Bs%3A%2F%2Fneo4j@36535562.databases.neo4j.io&db=neo4j)(password: `cypherbench`) at Neo4j AuraDB! Check it out! You can run Cypher queries like `MATCH (n:Player {name: 'LeBron James'})-[r]-(m) RETURN *`.
- [Dec 27, 2024] The nl2cypher tasks are now available on [🤗HuggingFace Datasets](https://huggingface.co/datasets/megagonlabs/cypherbench)! 


## 🚀 Quickstart

### 1. Installation


```bash
conda create -n cypherbench python=3.11
conda activate cypherbench

git clone https://github.com/megagonlabs/cypherbench.git
cd cypherbench
pip install -e .
```

### 2. Download the dataset

You can easily download the dataset (both the graphs and nl2cypher tasks) by cloning the [HuggingFace dataset repository](https://huggingface.co/datasets/megagonlabs/cypherbench):

```bash
# Make sure you have git-lfs installed (https://git-lfs.com)
git lfs install

# Clone the dataset repo from HuggingFace and save it as the `benchmark` directory
git clone https://huggingface.co/datasets/megagonlabs/cypherbench benchmark
```

### 3. Deploy the graphs

Now, you can deploy all 11 property graphs with a single Docker Compose command using our [custom Neo4j Docker image](https://hub.docker.com/repository/docker/megagonlabs/neo4j-with-loader/general) and our [Docker Compose configuration](docker/docker-compose-full.yml). Run the following script (which additionally performs sanity checks to ensure required files exist) to deploy the graphs:

```bash
# Make sure you have Docker installed
cd docker/
bash start_neo4j_full.sh
```

Wait for at least 30 minutes for the graphs to be fully loaded.

To stop the Neo4j databases, run `bash stop_neo4j_full.sh`

### 4. Run `gpt-4o-mini` on CypherBench

```bash
cd .. # Go back to the root directory
python -m cypherbench.baseline.zero_shot_nl2cypher --llm gpt-4o-mini --result_dir output/gpt-4o-mini/
```


## Future Release Plan

- [x] text2cypher tasks
- [x] 11 property graphs and graph deployment docker
- [x] nl2cypher baseline code
- [ ] EX/PSJS implementation and evaluation scripts
- [ ] Wikidata RDF-to-property-graph engine
- [ ] Text2cypher task generation pipeline
