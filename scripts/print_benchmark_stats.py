import argparse
import json
from cypherbench.schema import *
from cypherbench.data_utils import print_benchmark_distribution


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--benchmark_path', default='benchmark.json')
    parser.add_argument('--config', default='nl2cypher_generator_config.json')
    args = parser.parse_args()
    print(args)
    print()

    with open(args.benchmark_path) as fin:
        benchmark = [Nl2CypherSample(**item) for item in json.load(fin)]

    with open(args.config) as fin:
        config = Nl2CypherGeneratorConfig(**json.load(fin))

    print_benchmark_distribution(benchmark, config)


if __name__ == '__main__':
    main()
