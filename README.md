<div align="center">
<h1 align="center">CypherBench: Towards Precise Retrieval over Full-scale Modern Knowledge Graphs in the LLM Era</h1>

[🤗 Dataset](https://huggingface.co/datasets/megagonlabs/cypherbench) &nbsp;&nbsp; [📄 Paper](https://arxiv.org/pdf/2412.18702) &nbsp;&nbsp; [💻 Code](https://github.com/megagonlabs/cypherbench) &nbsp;&nbsp; [🌐 Demo Graph](https://browser.neo4j.io/?dbms=neo4j%2Bs%3A%2F%2Fneo4j@36535562.databases.neo4j.io&db=neo4j)

<img src="assets/text2cypher.png" width="80%">

</div>

This repository contains the code and resources for the paper [CypherBench: Towards Precise Retrieval over Full-scale Modern Knowledge Graphs in the LLM Era](https://arxiv.org/pdf/2412.18702) by Yanlin Feng, Simone Papicchio, and Sajjadur Rahman.

You might find this repository useful if you are interested in:
- Building Text-to-Cypher models, including:
  - Running baselines on CypherBench [[code]](cypherbench/baseline/zero_shot_nl2cypher.py) [[doc]](#-quickstart)
  - Fetching structured schema from a Neo4j database [[code]](cypherbench/neo4j_connector.py#L96-L154)
  - Metrics for measuring Text-to-Cypher performance [[code]](cypherbench/metrics)
- Creating domain knowledge graphs from Wikidata [[code]](cypherbench/wd2neo4j) [[doc]](#-wikidata-to-property-graph-conversion-engine)
- Generating Text-to-Cypher tasks for your own Neo4j graphs [[code]](cypherbench/taskgen) [[doc]](#-text-to-cypher-task-generation-pipeline)

## 🔥 Updates

- [Jun 18, 2025] We have released the Wikidata-to-Property-Graph conversion engine! Check out the [instructions](#-wikidata-to-property-graph-conversion-engine) below!
- [Jun 17, 2025] We have released the text-to-cypher task generation pipeline! See the [instructions](#-text-to-cypher-task-generation-pipeline) below!
- [May 15, 2025] Our paper has been accepted to ACL 2025 main conference! See you in Vienna!
- [Feb 20, 2025] We updated the graph deployment configuration to reduce RAM usage.
- [Feb 19, 2025] We have released the evaluation scripts and the EX and PSJS implementations!
- [Feb 14, 2025] We have released the text-to-cypher baseline code! See the instructions below on how to run `gpt-4o-mini` on CypherBench.
- [Feb 13, 2025] The [11 property graphs](https://huggingface.co/datasets/megagonlabs/cypherbench/tree/main/graphs) are now available on 🤗HuggingFace! We also make it super easy to deploy them (see the instructions below).
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

To download the dataset (including both the graphs and text-to-cypher tasks), simply clone the [HuggingFace dataset repository](https://huggingface.co/datasets/megagonlabs/cypherbench):

```bash
# Make sure you have git-lfs installed (https://git-lfs.com)
git lfs install

# Clone the dataset repo from HuggingFace and save it as the `benchmark` directory
git clone https://huggingface.co/datasets/megagonlabs/cypherbench benchmark
```

### 3. Deploy the graphs using Docker

⚠️ Deploying the graphs requires significant memory. We recommend using a machine with at least 64GB of RAM when deploying the 7 test graphs and 128GB when deploying all 11 graphs. Additionally, ensure that Docker is installed ([Docker installation instructions](https://docs.docker.com/engine/install/)) before proceeding.

Now, you can deploy the 7 test graphs with a single Docker Compose command using our [custom Neo4j Docker image](https://hub.docker.com/repository/docker/megagonlabs/neo4j-with-loader/general) and our [Docker Compose configuration](docker/docker-compose-test.yml):

```bash
cd docker/
bash start_neo4j_test.sh  #  This script first checks if required files exist, then runs the docker-compose command
cd .. 

# check if the graphs are fully loaded (it typically takes at least 10 minutes).
python scripts/print_db_status.py
```

To stop the Neo4j databases, run `bash stop_neo4j_test.sh`.

### 4. Run `gpt-4o-mini` on CypherBench

Running `gpt-4o-mini` on the CypherBench test set costs around $0.3. First, make sure you have set the `OPENAI_API_KEY` environment variable to use the OpenAI API.

```bash
python -m cypherbench.baseline.zero_shot_nl2cypher --llm gpt-4o-mini --result_dir output/gpt-4o-mini/
```

There are two ways to fetch the graph schemas when running text-to-cypher:
- (default) `--load_schema_from json` loads the schema from the local JSON files stored in [the benchmark/graphs/schemas directory](https://huggingface.co/datasets/megagonlabs/cypherbench/tree/main/graphs/schemas). When using this option, the Neo4j databases are not used during text-to-cypher.
- `--load_schema_from neo4j` fetches the schema from the Neo4j database by executing special Cypher queries*. This option requires the Neo4j databases to be fully loaded.

*We don't use apoc.meta.data() by default, see Appendix A.4 in the paper for details.

### 5. Evaluate metrics

```bash
python -m cypherbench.evaluate --result_dir output/gpt-4o-mini/  --num_threads 8  # Adjust the number of threads as needed
```

Metric implementation:
- Execution Accuracy (EX): [execution_accuracy.py](cypherbench/metrics/execution_accuracy.py)
- Provenance Subgraph Jaccard Similarity (PSJS): [provenance_subgraph_jaccard_similarity.py](cypherbench/metrics/provenance_subgraph_jaccard_similarity.py)
- Executable Percentage: [executable.py](cypherbench/metrics/executable.py)


Reference performance for `gpt-4o-mini`:

```
{
  "overall": {
    "execution_accuracy": 0.3143,
    "psjs": 0.4591,
    "executable": 0.8739
  },
  "by_graph": {
    "flight_accident": 0.4603,
    "fictional_character": 0.3273,
...
```

## 🌐 Wikidata-to-Property-Graph Conversion Engine

We open-source our Wikidata-to-Property-Graph conversion engine in the [cypherbench.wd2neo4j](cypherbench/wd2neo4j) package. You can create a domain knowledge graph from Wikidata by just defining the graph schema!

### Quick Tutorial

The first step is to define the graph schema in a JSON file. The schema should define the entity and relation types, along with their corresponding Wikidata QID/PIDs. We provide a [sample mini NBA schema](wd2neo4j_schemas/nba_mini.json) with a single relationship `partOfDivision` between `Team` and `Division`. For complete details on the schema format, see the [WDNeo4jSchema](cypherbench/wd2neo4j/schema.py#L301) data structure.

Next, you can run the conversion engine by:

```bash
python -m cypherbench.wd2neo4j --neo4j_schema wd2neo4j_schemas/nba_mini.json --output_dir output/nba_mini/
# output graph at output/nba_mini/nba_mini-graph.json
```

The engine will automatically issue SPARQL queries to Wikidata and assemble the retrieved data into a property graph.

If your graphs are too large (e.g. > 100k entities), you might get timeout errors because the official Wikidata SPARQL endpoint has a time limit of 60 seconds per query. In this case, you can deploy your own Wikidata SPARQL endpoint (documentation coming soon!) and pass in the url using the `--sparql_url` argument.

At this point, the property graph is saved in the [WikidataKG](cypherbench/wd2neo4j/schema.py#L384) format which contains Wikidata-dependent fields like `wikidata_qid`. We recommend converting it into the [SimpleKG](cypherbench/wd2neo4j/schema.py#L102) format, the generic property graph format used by the CypherBench graphs:

```bash
python -m cypherbench.wd2neo4j.wd2simplekg --input_path output/nba_mini/nba_mini-graph.json --output_path output/nba_mini/nba_mini-graph_simplekg.json
```

The property graph can now be deployed using our custom Neo4j Docker image:

```bash
docker run -d \
  --name cypherbench-nba-mini \
  -p 15095:7687 \
  -p 7474:7474 \
  -v $(pwd)/output/nba_mini/nba_mini-graph_simplekg.json:/init/graph.json \
  -e NEO4J_AUTH="neo4j/cypherbench" \
  -e NEO4J_PLUGINS='["apoc", "graph-data-science"]' \
  megagonlabs/neo4j-with-loader:2.4
```

## 🏭 Text-to-Cypher Task Generation Pipeline

We also open-source the text-to-cypher task generation pipeline in the [cypherbench.taskgen](cypherbench/taskgen) package. You can generate as many text-to-cypher tasks as you want for your own Neo4j graphs! Simply pass in the Neo4j graph endpoint (host + port) to the [task generator](cypherbench/taskgen/generate_benchmark.py#L131). You can also create your own templates.

### Reproducing CypherBench

For the CypherBench graphs, the task generation pipeline requires a set of sampled subgraphs of the original full-scale graphs for efficient template instantiation. The graphs are already uploaded to the HuggingFace repo (if you have previously cloned the repo, run a `git pull` under `benchmark/`, otherwise, follow the instructions in the [Download the dataset](#2-download-the-dataset) section) and can be deployed using the following commands:

```
cd docker/
bash start_neo4j_sampled.sh  
cd .. 

# check if the graphs are fully loaded
python scripts/print_db_sampled_status.py
```

After the graphs have been fully loaded, you can run the task generation pipeline by:

```bash
bash scripts/run_benchmark_generation.sh output/taskgen/
```

The task generation pipeline takes the following files as input:
- [nl2cypher_generator_config.json](nl2cypher_generator_config.json) - The file that defines the question and Cypher templates (MATCH/RETURN patterns). You can create your own templates by following the syntax in the file.
- [neo4j_info.json](neo4j_info.json) - The file that specifies the host and port of the Neo4j databases, including both the full-scale graphs and sampled graphs.
- [graph_info.json](graph_info.json) - The file that specifies human-annotated characteristics (e.g. cardinality, participation, etc.) of the relations, which are used to detect semantically unrealistic questions (see Section 4.4.3 in the paper).

Under the hood, the pipeline generates the tasks by the following steps:
1. [generate_benchmark.py](cypherbench/taskgen/generate_benchmark.py) - Intantiate the templates on the sampled Neo4j graphs.
2. [filter_long_running.py](cypherbench/taskgen/filter_long_running.py) - Filter out the tasks that take more than 30 seconds to execute on the full-scale graphs.
3. [rewrite_question_llm.py](cypherbench/taskgen/rewrite_question_llm.py) - Rewrite the questions into more natural language using LLMs.
4. [verify_question_llm.py](cypherbench/taskgen/verify_question_llm.py) - Verify the rewritten questions using LLMs.



## 📅 Future Release Plan

- [x] Text-to-Cypher tasks
- [x] 11 property graphs and graph deployment docker
- [x] Text-to-Cypher baseline code
- [x] EX/PSJS implementation and evaluation scripts
- [x] Text-to-Cypher task generation pipeline
- [x] Wikidata RDF-to-property-graph engine
- [ ] Additional resources

Please open a Github issue if you have any questions, find any bugs, or need anything else that is not yet open-sourced!

## 📚 Citation

```
@article{feng2024cypherbench,
  title={CypherBench: Towards Precise Retrieval over Full-scale Modern Knowledge Graphs in the LLM Era},
  author={Feng, Yanlin and Papicchio, Simone and Rahman, Sajjadur},
  journal={arXiv preprint arXiv:2412.18702},
  year={2024}
}
```
