import argparse
import json
import uuid
import neo4j
import copy
from typing import Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import random
from cypherbench.data_utils import print_benchmark_distribution
from cypherbench.schema import Nl2CypherSample
from cypherbench.neo4j_connector import Neo4jConnector


def update_num_result(item: Nl2CypherSample, neo4j_conn, timeout: int, max_records: int) -> Optional[Nl2CypherSample]:
    item = copy.deepcopy(item)
    try:
        num_records = neo4j_conn.run_query(f'{item.gold_match_cypher} RETURN count(*) as count',
                                           timeout=timeout)[0]['count']
        if num_records > max_records:
            return None
        results = neo4j_conn.run_query(item.gold_cypher, timeout=timeout)
        if len(results) == 0:
            return None
        return item
    except (
            neo4j.exceptions.CypherSyntaxError,
            neo4j.exceptions.DatabaseError,
            neo4j.exceptions.CypherTypeError,
            neo4j.exceptions.ClientError,
    ) as e:
        return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--benchmark_path', default=['output/benchmark.json'], nargs='+')
    parser.add_argument('--neo4j_info', default='neo4j_info.json')
    parser.add_argument('--output_path', default='output/benchmark_filtered.json')
    parser.add_argument('-n', '--max_num_per_type', type=int, default=2)
    parser.add_argument('--num_threads', type=int, default=128)
    parser.add_argument('--timeout', type=int, default=30)
    parser.add_argument('--max_records', type=int, default=100000)
    parser.add_argument('--seed', type=int, default=42)

    args = parser.parse_args()
    print(args)
    print()

    random.seed(args.seed)

    benchmark: list[Nl2CypherSample] = []
    for path in args.benchmark_path:
        with open(path) as fin:
            benchmark += [Nl2CypherSample(**item) for item in json.load(fin)]

    order = {item.qid: i for i, item in enumerate(benchmark)}

    print(f'Loaded benchmark with {len(benchmark)} samples')

    with open(args.neo4j_info) as fin:
        neo4j_info = json.load(fin)

    graph2conn = {graph: Neo4jConnector(
        name=graph, max_connection_pool_size=2 * args.num_threads, **info) for graph, info in
        neo4j_info['full'].items()}

    random.shuffle(benchmark)  # shuffle the graphs to reduce resource contention

    with ThreadPoolExecutor(max_workers=args.num_threads) as executor:
        futures = [executor.submit(update_num_result, item, graph2conn[item.graph], args.timeout, args.max_records)
                   for item in benchmark]
        benchmark_updated = []
        for future in tqdm(as_completed(futures), total=len(benchmark)):
            item = future.result()
            if item is not None:
                benchmark_updated.append(item)

    new_qids = {item.qid for item in benchmark_updated}
    print()
    print('Bad samples:')
    for item in benchmark:
        if item.qid not in new_qids:
            print(item.gold_cypher)

    print()
    print(f'Number of samples after removing timeout queries: {len(benchmark_updated)}')

    benchmark_updated = sorted(benchmark_updated, key=lambda x: order[x.qid])
    print_benchmark_distribution(benchmark_updated)

    with open(args.output_path, 'w') as fout:
        json.dump([item.model_dump(mode='json') for item in benchmark_updated], fout, indent=2)
    print(f'Saved balanced benchmark to {args.output_path}')


if __name__ == '__main__':
    main()
