from cypherbench.schema import *
import collections
import json
from tabulate import tabulate


def print_benchmark_distribution(samples: list[Nl2CypherSample], config: Nl2CypherGeneratorConfig = None):
    if config is None:
        with open('nl2cypher_generator_config.json') as fin:
            config = Nl2CypherGeneratorConfig(**json.load(fin))

    graph_counts = collections.Counter([sample.graph for sample in samples])

    mp_counts = {mp.category: 0 for mp in config.match_patterns}
    for sample in samples:
        mp_counts[sample.from_template.match_category] += 1

    basic_rp_counts = {rp.pattern_id: 0 for rp in config.return_patterns if rp.category == 'n_basic'}
    for sample in samples:
        if sample.from_template.return_pattern_id in basic_rp_counts:
            basic_rp_counts[sample.from_template.return_pattern_id] += 1

    special_rp_counts = {rp.pattern_id: 0 for rp in config.return_patterns if rp.category != 'n_basic'}
    for sample in samples:
        if sample.from_template.return_pattern_id in special_rp_counts:
            special_rp_counts[sample.from_template.return_pattern_id] += 1

    print()
    print(f'Total samples: {len(samples)}')
    print()
    print(tabulate(graph_counts.items(), headers=['Graph', 'Count'], tablefmt="github"))
    print()
    print(tabulate(mp_counts.items(), headers=['Match Category', 'Count'], tablefmt="github"))
    print()
    print(tabulate(basic_rp_counts.items(), headers=['Return Pattern', 'Count'], tablefmt="github"))
    print()
    print(tabulate(special_rp_counts.items(), headers=['Return Pattern', 'Count'], tablefmt="github"))
