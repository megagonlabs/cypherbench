import argparse
import json
import copy
import collections
from typing import Union
from cypherbench.wd2neo4j.schema import *


def convert_wd_to_simple_kg(kg: WikidataKG) -> SimpleKG:
    eschemas = [
        SimpleEntitySchema(
            label=es.label,
            description=None,
            properties={p.label: p.datatype for p in es.properties},
        )
        for es in kg.schema.entities
    ]
    rschemas = [
        SimpleRelationSchema(
            label=rs.label,
            subj_label=rs.subj_label,
            obj_label=rs.obj_label,
            properties={p.label: p.datatype for p in rs.properties},
        )
        for rs in kg.schema.relations
    ]
    entities = [
        SimpleEntity(
            eid=f"{e.label}#{e.wikidata_qid}",
            label=e.label,
            name=e.name,
            aliases=e.aliases,
            description=e.description,
            properties=e.properties,
            provenance=[
                f"https://en.wikipedia.org/wiki/{e.enwiki_title.replace(' ', '_')}"
            ]
            if e.enwiki_title
            else [],
        )
        for e in kg.entities
    ]
    relations = [
        SimpleRelation(
            rid=str(rid),
            label=r.label,
            subj_id=f"{r.subj_label}#{r.subj_wikidata_qid}",
            obj_id=f"{r.obj_label}#{r.obj_wikidata_qid}",
            properties=r.properties,
        )
        for rid, r in enumerate(kg.relations)
    ]
    kg = SimpleKG(
        schema=SimpleKGSchema(
            name=kg.schema.name, entities=eschemas, relations=rschemas
        ),
        entities=entities,
        relations=relations,
    )

    relations = collections.defaultdict(list)
    for rel in kg.relations:
        rel = copy.deepcopy(rel)
        merged = False
        for r0 in relations[(rel.label, rel.subj_id, rel.obj_id)]:
            overlap = set(r0.properties) & set(rel.properties)
            non_array_keys = [
                k for k in overlap if not isinstance(rel.properties[k], list)
            ]
            # two relations can be merged if their common non-array properties have the same values

            if all(
                r0.properties[k] == rel.properties[k] for k in non_array_keys
            ):  # found a relation that can be merged
                for k in rel.properties:
                    if k not in r0.properties:
                        r0.properties[k] = rel.properties[k]
                    elif isinstance(rel.properties[k], list):
                        for v in rel.properties[k]:
                            if (
                                v not in r0.properties[k]
                            ):  # avoid adding duplicate values
                                r0.properties[k].append(v)
                        r0.properties[k] = sorted(r0.properties[k])
                print(
                    f"Merged relation {rel.label} between {rel.subj_id} and {rel.obj_id}"
                )
                merged = True
                break
        if not merged:
            relations[(rel.label, rel.subj_id, rel.obj_id)].append(rel)
    relations = [rel for rels in relations.values() for rel in rels]
    kg = SimpleKG(schema=kg.schema, entities=kg.entities, relations=relations)
    return kg


def print_kg_stats(kg: Union[SimpleKG, WikidataKG]):
    is_wd_kg = isinstance(kg, WikidataKG)
    ent_cnt = {}
    rel_cnt = {}
    ent_prop_cnt = {}
    rel_prop_cnt = {}

    if not is_wd_kg:
        eid2label = {e.eid: e.label for e in kg.entities}

    for es in kg.schema.entities:
        ent_cnt[es.label] = 0
        ent_prop_cnt[es.label] = {}
        for prop in es.properties:
            if is_wd_kg:
                prop = prop.label
            ent_prop_cnt[es.label][prop] = 0
    for rs in kg.schema.relations:
        rel_cnt[f"{rs.label}({rs.subj_label}->{rs.obj_label})"] = 0
        rel_prop_cnt[f"{rs.label}({rs.subj_label}->{rs.obj_label})"] = {}
        for prop in rs.properties:
            if is_wd_kg:
                prop = prop.label
            rel_prop_cnt[f"{rs.label}({rs.subj_label}->{rs.obj_label})"][prop] = 0

    for ent in kg.entities:
        ent_cnt[ent.label] += 1
        for prop in ent.properties:
            ent_prop_cnt[f"{ent.label}"][prop] += 1
    for rel in kg.relations:
        subj_label = rel.subj_label if is_wd_kg else eid2label[rel.subj_id]
        obj_label = rel.obj_label if is_wd_kg else eid2label[rel.obj_id]
        rel_cnt[f"{rel.label}({subj_label}->{obj_label})"] += 1
        for prop in rel.properties:
            rel_prop_cnt[f"{rel.label}({subj_label}->{obj_label})"][prop] += 1

    print()
    print(f"### Entities (total: {len(kg.entities)})")
    for elabel, num in ent_cnt.items():
        print(f"- {elabel}: {num}")
        if len(ent_prop_cnt[elabel]) > 0:
            for prop, num in ent_prop_cnt[elabel].items():
                print(f"  * {prop}: {num}")
    print()
    print(f"### Relations (total: {len(kg.relations)})")
    for rlabel, num in rel_cnt.items():
        print(f"- {rlabel}: {num}")
        if len(rel_prop_cnt[rlabel]) > 0:
            for prop, num in rel_prop_cnt[rlabel].items():
                print(f"  * {prop}: {num}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", default="output/nba_mini/nba_mini-graph.json")
    parser.add_argument(
        "--output_path",
        default="output/nba_mini/nba_mini-graph_simplekg.json",
    )
    args = parser.parse_args()
    print(args)
    print()

    with open(args.input_path) as f:
        kg = WikidataKG(**json.load(f))

    kg_simple = convert_wd_to_simple_kg(kg)

    print_kg_stats(kg_simple)

    with open(args.output_path, "w") as f:
        json.dump(kg_simple.model_dump(mode="json"), f, indent=2)

    print(f"Saved SimpleKG to {args.output_path}")


if __name__ == "__main__":
    main()
