import argparse
import copy
import json
import os
import math
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from cypherbench.metrics import *
from cypherbench.neo4j_connector import Neo4jConnector
from cypherbench.schema import Nl2CypherSample


RETURN_PATTERN_MAPPING = {
    "n_name": "n_name",
    "n_prop": "n_prop_combined",
    "n_name_prop": "n_prop_combined",
    "n_prop_distinct": "n_prop_combined",
    "n_prop_array_distinct": "n_prop_combined",
    "n_order_by": "n_order_by",
    "n_argmax": "n_argmax",
    "n_where": "n_where",
    "n_agg": "n_agg",
    "n_group_by": "n_group_by"
}

METRIC_FUNC_MAPPING = {
    'execution_accuracy': execution_accuracy,
    'psjs': provenance_subgraph_jaccard_similarity,
    'executable': executable,
}


def compute_metrics(item: Nl2CypherSample, metrics, neo4j_conn):
    item = copy.deepcopy(item)
    for m in metrics:
        pred_cypher = item.pred_cypher
        if pred_cypher.endswith('<end_of_turn>'):
            pred_cypher = pred_cypher[:-len('<end_of_turn>')].strip()
        item.metrics[m] = METRIC_FUNC_MAPPING[m](
            pred_cypher=pred_cypher,
            target_cypher=item.gold_cypher,
            neo4j_connector=neo4j_conn
        )
    return item


def avg_and_round(nums: list[float], n: int = 4):
    return round(sum(nums) / len(nums), n) if nums else math.nan


def aggregate(results: list[tuple[str, float]]):
    res = {}
    for key, value in results:
        if key not in res:
            res[key] = []
        res[key].append(value)
    for key, values in res.items():
        res[key] = avg_and_round(values)
    return res


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--neo4j_info', default='neo4j_info.json')
    parser.add_argument('--result_dir', default='output/gpt-4o')
    parser.add_argument('--num_threads', type=int, default=8)
    parser.add_argument('--metrics', nargs='+', default=['execution_accuracy', 'psjs', 'executable'])
    parser.add_argument('--metric_for_agg', default='execution_accuracy')
    args = parser.parse_args()
    print(args)
    print()

    with open(os.path.join(args.result_dir, 'result.json')) as fin:
        result = [Nl2CypherSample(**item) for item in json.load(fin)]

    with open(args.neo4j_info) as fin:
        neo4j_info = json.load(fin)

    graph2conn = {graph: Neo4jConnector(name=graph, **info) for graph, info in
                  neo4j_info['full'].items()}

    # Use ThreadPoolExecutor for multithreading
    result_with_metrics = []
    with ThreadPoolExecutor(max_workers=args.num_threads) as executor:
        futures = [executor.submit(compute_metrics, item, args.metrics, graph2conn[item.graph]) for item in result]
        for future in tqdm(as_completed(futures), total=len(result)):
            result_with_metrics.append(future.result())

    aggregated = {}
    aggregated['overall'] = {m: avg_and_round([item.metrics[m] for item in result_with_metrics]) for m in args.metrics}

    metric_for_agg = args.metric_for_agg
    aggregated['by_graph'] = aggregate([(item.graph, item.metrics[metric_for_agg]) for item in result_with_metrics])
    aggregated['by_match'] = aggregate([(item.from_template.match_category, item.metrics[metric_for_agg])
                                        for item in result_with_metrics])
    aggregated['by_return'] = aggregate(
        [(RETURN_PATTERN_MAPPING[item.from_template.return_pattern_id], item.metrics[metric_for_agg])
         for item in result_with_metrics if item.from_template.return_pattern_id in RETURN_PATTERN_MAPPING]
    )

    output_path = os.path.join(args.result_dir, f'result_with_metrics.json')
    with open(output_path, 'w') as fout:
        json.dump([item.model_dump(mode='json') for item in result_with_metrics], fout, indent=2)
    print(f'Saved result with metrics to {output_path}')

    output_path = os.path.join(args.result_dir, f'aggregated_metrics.json')
    with open(output_path, 'w') as fout:
        json.dump(aggregated, fout, indent=2)
    print(f'Saved aggregated metrics to {output_path}')

    print()
    print('Aggregated metrics:')
    print(json.dumps(aggregated, indent=2))


if __name__ == '__main__':
    main()
