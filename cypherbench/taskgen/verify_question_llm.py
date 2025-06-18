import argparse
import copy
import json
from tqdm import trange
from string import Template
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import JsonOutputParser
from cypherbench.data_utils import print_benchmark_distribution
from cypherbench.schema import Nl2CypherSample

REWRITE_PROMPT = """Determine if the rewritten question
  (1) preserve the same meaning of the provided Cypher query and the original question
  (2) is realistic and interesting in the real world
- The output should be a JSON dictionary with the first two keys `has_same_meaning`, `is_realistic`.
    If either of the two conditions is false, provide an explanation in `explanation` and a correct rewritten question in the `correct_rewrite` field.
- The brackets in the original question are parsing hints for the question structure to ensure it is unambiguous.
- Pay attention to the direction of the relation pattern (indicated by `->` or `<-`) in the Cypher query.
  - For example, `(n:Character)-[r0:hasFather]->(m0:Character)` indicates m0 is the father of n,
    while `(n:Character)<-[r0:hasFather]-(m0:Character)` matches n as the father of m0.
  - For example, `MATCH (n:Taxon)<-[r0:feedsOn]-(m0:Taxon) RETURN n` selects the taxon are fed on by other taxon,
    while `MATCH (n:Taxon)-[r0:feedsOn]->(m0:Taxon) RETURN n` selects the taxon that feed on other taxon.

=== Example ===
Cypher: MATCH (n:Person)<-[r0:createdBy]-(m0:Sculpture {name: 'Black and White Tipped Flower'}) WITH DISTINCT n RETURN n.name ORDER BY n.date_of_death ASC
Original Question: List the name of Person that "Black and White Tipped Flower" createdBy, ordered by date_of_death in ascending order.
Rewritten Question: Who are the individuals who created the sculpture Black and White Tipped Flower, listed by their date of death from earliest to latest?
Output:
```json
{
    "has_same_meaning": true,
    "is_realistic": true,
    "explanation": <a brief explanation if either of the two conditions is false>,
    "correct_rewrite": <a correct rewritten question f either of the two conditions is false>
}
```

Cypher: MATCH (n:Company)<-[r0:subsidiaryOf]-(m0:Company) WITH DISTINCT n WHERE n.launch_year < 1907 RETURN n.name
Original Question: For Company that Company subsidiaryOf, return the names of those that launch_year < 1907.
Rewritten Question: What are the names of companies that are subsidiaries and were launched before the year 1907?
Output:
```json
{
    "has_same_meaning": false,
    "is_realistic": true,
    "explanation": "The rewritten question implies that the companies themselves are subsidiaries, whereas the Cypher query and the original question are about companies that have subsidiaries.",
    "correct_rewrite": "What are the names of companies that have subsidiaries and were launched before the year 1907?"
}
```
=== Your task ===
Cypher: ${gold_cypher}
Original Question: ${nl_question_raw}
Rewritten Question: ${nl_question}
Output:
"""


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--llm', default='gpt-4o')
    parser.add_argument('--benchmark_path', default='output/benchmark_balanced_revised.json')
    parser.add_argument('--output_path', default='output/benchmark_balanced_revised_corrected.json')
    parser.add_argument('--verification_output_path', default='output/verification_result.json')
    parser.add_argument('--temperature', default=0, type=float)
    parser.add_argument('--batch_size', default=100, type=int)
    parser.add_argument('--remove_bad', default=False, action='store_true')
    args = parser.parse_args()
    print(args)
    print()

    with open(args.benchmark_path) as fin:
        benchmark = [Nl2CypherSample(**item) for item in json.load(fin)]

    print(f'Loaded benchmark with {len(benchmark)} samples')

    benchmark_verified = []
    samples = []
    for i in trange(0, len(benchmark), args.batch_size):
        j = min(i + args.batch_size, len(benchmark))
        prompts = [Template(REWRITE_PROMPT).substitute(
            gold_cypher=item.gold_cypher, nl_question_raw=item.nl_question_raw, nl_question=item.nl_question)
            for item in benchmark[i:j]]
        chain = ChatOpenAI(model_name=args.llm, temperature=args.temperature) | JsonOutputParser()
        resp = chain.batch(prompts)
        if i == 0:
            print(f'<prompt>{prompts[0]}</prompt>')
            print(f'<response>{resp[0]}</response>')
        for k in range(j - i):
            item_copy = copy.deepcopy(benchmark[i + k])
            if resp[k]['has_same_meaning'] and resp[k]['is_realistic']:
                benchmark_verified.append(item_copy)
            elif not args.remove_bad:
                item_copy.nl_question = resp[k]['correct_rewrite']
                benchmark_verified.append(item_copy)

            item = benchmark[i + k].model_dump(mode='json')
            item['has_same_meaning'] = resp[k]['has_same_meaning']
            item['is_realistic'] = resp[k]['is_realistic']
            item['explanation'] = resp[k].get('explanation')
            item['correct_rewrite'] = resp[k].get('correct_rewrite')
            samples.append(item)

    aggregated = {
        'both': sum(item['has_same_meaning'] and item['is_realistic'] for item in samples) / len(samples),
        'has_same_meaning': sum(item['has_same_meaning'] for item in samples) / len(samples),
        'is_realistic': sum(item['is_realistic'] for item in samples) / len(samples),
    }
    bad_samples = [item for item in samples if not item['has_same_meaning'] or not item['is_realistic']]
    output = {
        'aggregated': aggregated,
        'bad_samples': bad_samples,
    }

    print(f'Aggregated verification result: {aggregated}')

    print(f'Number of verified samples: {len(benchmark_verified)}')

    print_benchmark_distribution(benchmark_verified)

    with open(args.output_path, 'w') as fout:
        json.dump([item.model_dump(mode='json') for item in benchmark_verified], fout, indent=2)
    print(f'Saved verified benchmark to {args.output_path}')

    with open(args.verification_output_path, 'w') as fout:
        json.dump(output, fout, indent=2)
    print(f'Saved verification output to {args.verification_output_path}')


if __name__ == '__main__':
    main()
