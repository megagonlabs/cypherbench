import json
from langchain_core.output_parsers import StrOutputParser
import random
from tqdm import trange, tqdm
import time
import argparse
import shutil
import os
from cypherbench.schema import Nl2CypherSample, PropertyGraphSchema, DataType
from cypherbench.baseline.utils import get_langchain_llm
from cypherbench.neo4j_connector import Neo4jConnector

NL2CYPHER_PROMPT_DEFAULT = """Translate the question to Cypher query based on the schema of a Neo4j knowledge graph.
- Output the Cypher query in a single line, without any additional output or explanation. Do not wrap the query with any formatting like ```.
- Perform graph pattern matching in the `MATCH` clause if possible.
- Avoid listing the same entity multiple times in the results. However, if multiple distinct entities share the same name, their names should be repeated as separate entries.
- Do not return node objects. Instead, return entity names or properties.

Graph Schema:
{schema}

Question: {question}
Cypher: """

PROMPT_MAPPING = {
    'default': NL2CYPHER_PROMPT_DEFAULT
}


def load_schema_from_json(graphs: list[str]) -> dict[str, str]:
    graph2schema = {}
    for graph in tqdm(graphs, desc='Loading schema from JSON'):
        path = f'benchmark/graphs/schemas/{graph}_schema.json'
        with open(path) as fin:
            schema = PropertyGraphSchema.from_json(
                json.load(fin),
                add_meta_properties={'name': DataType.STR}
            ).to_sorted()
        graph2schema[graph] = schema.to_str(exclude_description=True)
    return graph2schema


def load_schema_from_neo4j(neo4j_info: dict) -> dict[str, str]:
    graph2schema = {}
    for graph, info in tqdm(neo4j_info['full'].items(), desc='Loading schema from Neo4j',
                            total=len(neo4j_info['full'])):
        neo4j_conn = Neo4jConnector(name=graph, **info)

        # mode='apoc' uses apoc.meta.data() to fetch schema from Neo4j. On large graphs this might result in an incomplete schema.
        # - Reference: https://github.com/neo4j-contrib/neo4j-apoc-procedures/issues/884
        schema = neo4j_conn.get_schema(mode='direct')
        graph2schema[graph] = schema.to_str(exclude_description=True)
    return graph2schema


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--llm', default='gpt-4o')
    parser.add_argument('--prompt', default='default', choices=['default'])
    parser.add_argument('--benchmark_path', default='benchmark/test.json')
    parser.add_argument('--neo4j_info', default='neo4j_info.json')
    parser.add_argument('--load_schema_from', default='json', choices=['json', 'neo4j'])
    parser.add_argument('--batch_size', default=10, type=int)
    parser.add_argument('--wait_time_between_batches', default=0.0, type=float)
    parser.add_argument('--result_dir', default='output/gpt-4o/')
    parser.add_argument('--overwrite', action='store_true')
    parser.add_argument('--debug', action='store_true')
    args = parser.parse_args()
    print(args)
    print()

    if os.path.exists(args.result_dir):
        if not args.overwrite:
            print(f'{args.result_dir} already exists. Use --overwrite to overwrite the directory.')
            return
        else:
            shutil.rmtree(args.result_dir)

    if not os.path.exists(args.result_dir):
        os.makedirs(args.result_dir)

    with open(args.benchmark_path) as fin:
        benchmark = [Nl2CypherSample(**item) for item in json.load(fin)]
    if args.debug:
        random.seed(0)
        benchmark = random.sample(benchmark, 50)

    with open(args.neo4j_info) as fin:
        neo4j_info = json.load(fin)

    llm = get_langchain_llm(args.llm, temperature=0.0)
    chain = llm.bind(stop='\n```') | StrOutputParser()

    if args.load_schema_from == 'neo4j':
        graph2schema = load_schema_from_neo4j(neo4j_info)
    elif args.load_schema_from == 'json':
        graph2schema = load_schema_from_json(neo4j_info['domains'])
    else:
        raise ValueError(f'Invalid load_schema_from: {args.load_schema_from}')

    prompt_template = PROMPT_MAPPING[args.prompt]

    res = []
    for i in trange(0, len(benchmark), args.batch_size):
        if i > 0:
            time.sleep(args.wait_time_between_batches)

        j = min(i + args.batch_size, len(benchmark))
        prompts = [prompt_template.format(schema=graph2schema[item.graph], question=item.nl_question)
                   for item in benchmark[i:j]]
        resp = chain.batch(prompts)
        assert len(resp) == j - i
        if i == 0:
            print(f'<prompt>{prompts[0]}</prompt>')
            print(f'<response>{resp[0]}</response>')
        for item, r in zip(benchmark[i:j], resp):
            r = r.replace('```cypher', '').replace('```', '').strip()
            item.pred_cypher = r
            res.append(item)

    output_path = os.path.join(args.result_dir, f'result.json')
    with open(output_path, 'w') as fout:
        json.dump([item.model_dump(mode='json') for item in res], fout, indent=2)
    print(f'Saved result to {output_path}')


if __name__ == '__main__':
    main()
