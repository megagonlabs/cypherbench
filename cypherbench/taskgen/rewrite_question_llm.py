import argparse
import json
from tqdm import trange
from string import Template
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from cypherbench.schema import Nl2CypherSample

REWRITE_PROMPT = """Rewrite the given template-generated question in a text-to-Cypher translation task to make it sound more natural:
- Ensure the rewritten question remains semantically equivalent to the original question and the provided Cypher query. Do not remove or add any constraints.
  - Pay attention to the direction of the relation pattern (indicated by `->` or `<-`) in the Cypher query.
    For example, `(n:Character)-[r0:hasFather]->(m0:Character)` indicates m0 is the father of n,
    while `(n:Character)<-[r0:hasFather]-(m0:Character)` matches n as the father of m0.
  - Pay attention to the direction of relation in the template-generated question.
    For example `List the names of Character that "Rhaenys Targaryen" hasFather"` means "Rhaenys Targaryen" connects 
    to the Character via relation `hasFather`, thus the questions is asking for the father of Rhaenys Targaryen.
    While `List the names of Character that hasFather "Rhaenys Targaryen"` selects the Character that has Rhaenys Targaryen as father.
- Ensure the rewritten question is grammatically correct and sounds natural.
- The brackets in the question are parsing hints for the question structure to ensure it is unambiguous. Do not include them in the rewritten question.
- For relation types (e.g. hasCastMember), rewrite them to natural language and diversify the expressions. Feel free to change from passive to active voice or vice versa.
  - e.g. A hasCastMember B -> B is cast in A, B stars in A, A features B, etc.
- For entity types (e.g. Person, FlightAccident) and properties (e.g. watershed_area_km2) rewrite them to natural language and diversify the expressions.
  - e.g. Person -> individual; human; passenger; etc. 
  - e.g. TaxonRank -> taxonomic rank; etc.
  - e.g. FlightAccident -> aviation accident; plane crash; etc.
  - e.g. watershed_area_km2 -> size of the watershed in square kilometers; area covered by the watershed in km²; etc.
- For multi-hop patterns, you can simplify it if the same meaning is preserved.
  - e.g. "teams that belong to a division that belongs to Western Conference" -> "teams in the Western Conference"
  - e.g. "the father of the mother of the person" -> "the person's maternal grandfather"
  - e.g. "the children of the father of the person" -> "the person's paternal siblings"
- For quoted names and string values, remove the quotes but ensure the same text is preserved.
- For numerical values and dates, diversify the expressions, but ensure the same value is preserved.
  - e.g. 1990-07-04 -> July 4th, 1990; 4 July 1990; 07/04/1990 (US format); 4th of July, 1990; etc.
  - e.g. 2000 -> two thousand; 2000; 2,000; 2k; etc.
- For operators (e.g. >, >=, IN, NOT IN, etc.), rewrite them to natural language and diversify the expressions but ensure the meaning is preserved.
  - e.g. NOT 'France' IN n.country_of_citizenship -> "is not a citizen of France"
- For "ascending" and "descending", rewrite them to natural language and diversify the expressions.
  - e.g. "the years in descending order" -> "the years from the most recent to the oldest"
- Output only the rewritten question, without any additional explanation.

=== Example ===
Cypher: MATCH (n:Taxon)<-[r0:hasParent]-(m0:Taxon)-[r1:hasConservationStatus]->(m1:ConservationStatus {name: 'Near Threatened'}) WITH DISTINCT n RETURN n.name
question: List the names of Taxon that [some Taxon that hasConservationStatus "Near Threatened"] hasParent
rewritten_question: What are the names of parents of taxa with a conservation status of Near Threatened?

Cypher: MATCH (n:Character)<-[r0:hasFather]-(m0:Character),(n:Character)<-[r1:killedBy]-(m0:Character) WITH DISTINCT n RETURN n.name
question: List the names of Character that a Character [hasFather and killedBy].
rewritten_question: List the names of fathers who killed their children.

=== Your task ===
Cypher: ${cypher}
question: ${question}
rewritten_question: """


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--llm', default='gpt-4o')
    parser.add_argument('--benchmark_path', default='output/benchmark_balanced.json')
    parser.add_argument('--output_path', default='output/benchmark_balanced_revised.json')
    parser.add_argument('--temperature', default=0, type=float)
    parser.add_argument('--batch_size', default=100, type=int)
    args = parser.parse_args()
    print(args)
    print()

    with open(args.benchmark_path) as fin:
        benchmark = [Nl2CypherSample(**item) for item in json.load(fin)]

    print(f'Loaded benchmark with {len(benchmark)} samples')

    for i in trange(0, len(benchmark), args.batch_size):
        j = min(i + args.batch_size, len(benchmark))
        prompts = [Template(REWRITE_PROMPT).substitute(cypher=item.gold_cypher, question=item.nl_question_raw)
                   for item in benchmark[i:j]]
        chain = ChatOpenAI(model_name=args.llm, temperature=args.temperature) | StrOutputParser()
        resp = chain.batch(prompts)
        if i == 0:
            print(f'<prompt>{prompts[0]}</prompt>')
            print(f'<response>{resp[0]}</response>')
        for item in benchmark[i:j]:
            item.nl_question = resp.pop(0)
        assert not resp

    with open(args.output_path, 'w') as fout:
        json.dump([item.model_dump(mode='json') for item in benchmark], fout, indent=2)
    print(f'Saved revised benchmark to {args.output_path}')


if __name__ == '__main__':
    main()
