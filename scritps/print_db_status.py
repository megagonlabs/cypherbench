import argparse
from tqdm import tqdm
import json
from tabulate import tabulate
from cypherbench.neo4j_connector import Neo4jConnector

graph2rels = {
    'art': 1339479,
    'biology': 7513599,
    'company': 299581,
    'fictional_character': 40548,
    'flight_accident': 2212,
    'geography': 903794,
    'movie': 1892202,
    'nba': 18991,
    'politics': 1548416,
    'soccer': 1119766,
    'terrorist_attack': 1525
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--neo4j_info', default='neo4j_info.json')
    args = parser.parse_args()
    print(args)
    print()

    with open(args.neo4j_info) as fin:
        neo4j_info = json.load(fin)

    is_all_ready = True
    df = []
    for graph, info in tqdm(neo4j_info['full'].items(), desc='Connecting to Neo4j',
                            total=len(neo4j_info['full'])):
        connection, num_entities, num_relations = 'N', float('nan'), float('nan')
        try:
            neo4j_conn = Neo4jConnector(name=graph, **info)
            connection = 'Y'
            num_entities = neo4j_conn.get_num_entities()
            num_relations = neo4j_conn.get_num_relations()
        except Exception as e:
            pass

        if num_relations != graph2rels[graph]:
            is_all_ready = False
        df.append((
            graph,
            f'{info["host"]}:{info["port"]}',
            connection,
            num_entities,
            num_relations,
            'Y' if num_relations == graph2rels[graph] else 'N'
        ))

    print()
    print(tabulate(df, headers=['Graph', 'URL', 'Connection?', 'Entities', 'Relations', 'Ready?'], tablefmt='github',
                   floatfmt='.0f'))

    print()
    if is_all_ready:
        print('All graphs are ready!')
    else:
        print('Warning: At least one graph is not ready!')


if __name__ == '__main__':
    main()
