# CypherBench

[![](https://img.shields.io/badge/license-apache2.0-green.svg)](LICENSE) 
[![](https://img.shields.io/badge/🤗-HuggingFace-red.svg)](https://huggingface.co/datasets/megagonlabs/cypherbench)
[![](https://img.shields.io/badge/paper-arxiv-yellow.svg)](https://arxiv.org/pdf/2412.18702)

## 🔥 Updates

- [Feb 14, 2025] We have released the text2cypher baseline code! See the instructions below on how to run `gpt-4o-mini` on CypherBench.
- [Feb 13, 2025] The [11 property graphs](https://huggingface.co/datasets/megagonlabs/cypherbench/tree/main/graphs) are now available on 🤗HuggingFace!
- [Dec 27, 2024] We have deployed a [demo NBA graph](https://browser.neo4j.io/?dbms=neo4j%2Bs%3A%2F%2Fneo4j@36535562.databases.neo4j.io&db=neo4j)(password: `cypherbench`) at Neo4j AuraDB! Check it out! You can run Cypher queries like `MATCH (n:Player {name: 'LeBron James'})-[r]-(m) RETURN *`.
- [Dec 27, 2024] The [training and test sets](https://huggingface.co/datasets/megagonlabs/cypherbench) are now available on 🤗HuggingFace! 


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

You can easily download the dataset (both the graphs and text2cypher tasks) by cloning the [HuggingFace dataset repository](https://huggingface.co/datasets/megagonlabs/cypherbench):

```bash
# Make sure you have git-lfs installed (https://git-lfs.com)
git lfs install

# Clone the dataset repo from HuggingFace and save it as the `benchmark` directory
git clone https://huggingface.co/datasets/megagonlabs/cypherbench benchmark
```

### 3. Deploy the graphs

Now, you can deploy all 11 property graphs with a single Docker Compose command using our [custom Neo4j Docker image](https://hub.docker.com/repository/docker/megagonlabs/neo4j-with-loader/general) and our [Docker Compose configuration](docker/docker-compose-full.yml):

```bash
# Make sure you have Docker installed
cd docker/

#  This script additionally performs sanity checks to ensure required files exist before running the docker-compose command.
bash start_neo4j_full.sh
```

Run the script TODO to check if the graphs are fully loaded (it typically takes at least 15 minutes).

To stop the Neo4j databases, run `bash stop_neo4j_full.sh`

### 4. Run `gpt-4o-mini` on CypherBench

```bash
cd .. # Go back to the root directory

# Make sure you have set the OPENAI_API_KEY environment variable to use the OpenAI API
python -m cypherbench.baseline.zero_shot_nl2cypher --llm gpt-4o-mini --result_dir output/gpt-4o-mini/
```

There are two ways to fetch the graph schemas when running text2cypher:
- (default) `--load_schema_from json` loads the schema from the local JSON files stored in [the benchmark/graphs/schemas directory](benchmark/graphs/schemas). When using this option, the text2cypher code does not depend on the Neo4j databases.
- `--load_schema_from neo4j` fetches the schema from the Neo4j database by executing special Cypher queries*. This option requires the Neo4j databases to be fully loaded.

*We don't use apoc.meta.data() by default, see Appendix A.4 in the paper for details.

### 5. Evaluate execution accuracy (EX), PSJS, and other metrics

TODO (to be released soon)


## Future Release Plan

- [x] text2cypher tasks
- [x] 11 property graphs and graph deployment docker
- [x] text2cypher baseline code
- [ ] EX/PSJS implementation and evaluation scripts
- [ ] Wikidata RDF-to-property-graph engine
- [ ] Text2cypher task generation pipeline
