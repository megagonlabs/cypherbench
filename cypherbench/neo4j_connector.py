from neo4j import GraphDatabase, Query
from typing import Literal
import time
from cypherbench.schema import *

APOC_META_NODE_PROPERTIES_QUERY = """
CALL apoc.meta.data()
YIELD label, other, elementType, type, property
WHERE NOT type = "RELATIONSHIP" AND elementType = "node"
WITH label AS label, apoc.coll.sortMaps(collect({property:property, type:type}), 'property') AS properties
RETURN label, properties ORDER BY label
""".strip()

APOC_META_REL_QUERY = """
CALL apoc.meta.data()
YIELD label, other, elementType, type, property
WHERE type = "RELATIONSHIP" AND elementType = "node"
UNWIND other AS other_node
RETURN label as start, property as type, other_node as end ORDER BY type, start, end
""".strip()

APOC_META_REL_PROPERTIES_QUERY = """
CALL apoc.meta.data()
YIELD label, other, elementType, type, property
WHERE NOT type = "RELATIONSHIP" AND elementType = "relationship"
RETURN label AS type, apoc.coll.sortMaps(collect({property:property, type:type}), 'property') AS properties ORDER BY type
""".strip()

DIRECT_NODE_PROPERTIES_QUERY = """
MATCH (n)
UNWIND labels(n) AS label
WITH label, keys(n) AS propertyKeys, n
UNWIND propertyKeys AS property
WITH DISTINCT label, property, apoc.meta.cypher.type(n[property]) as type
WITH label AS label, apoc.coll.sortMaps(collect({property:property, type:type}), 'property') AS properties
RETURN label, properties ORDER BY label
""".strip()

DIRECT_REL_QUERY = """
MATCH (n)-[r]->(m)
UNWIND labels(n) AS start UNWIND labels(m) AS end
RETURN DISTINCT start, type(r) as type, end
ORDER BY type, start, end
""".strip()

DIRECT_REL_PROPERTIES_QUERY = """
MATCH ()-[r]-()
WITH type(r) AS type, keys(r) AS propertyKeys, r
UNWIND propertyKeys AS property
WITH DISTINCT type, property, apoc.meta.cypher.type(r[property]) AS propType
WITH type, apoc.coll.sortMaps(collect({property:property, type:propType}), 'property') AS properties
RETURN type, properties ORDER BY type
""".strip()

VALUE_COUNT_QUERY = "MATCH (n:%s) RETURN count(DISTINCT n.%s) as prop_cnt, count(n) as node_cnt"


class Neo4jConnector:
    def __init__(self, name, host, port, username, password,
                 max_connection_pool_size: int = 100, database='neo4j', debug=False):
        self.name = name
        self.database = database
        self.debug = debug
        self.driver = GraphDatabase.driver(f'bolt://{host}:{port}', auth=(username, password),
                                           max_connection_pool_size=max_connection_pool_size)
        self.driver.verify_connectivity()

    def run_query(self, cypher, timeout=None, convert_func: Literal['data', 'graph'] = 'data', **kwargs):
        if self.debug:
            t0 = time.time()
            print(f'Running Cypher:\n```\n{cypher}\n```')

        with self.driver.session(database=self.database) as session:
            query = Query(cypher, timeout=timeout)
            result = session.run(query, **kwargs)
            if convert_func == 'data':
                result = result.data()
            elif convert_func == 'graph':
                result = result.graph()
            else:
                raise ValueError(f"Invalid convert_func: {convert_func}")

        if self.debug:
            print(f'Cypher finished in {time.time() - t0:.2f}s')
        return result

    def get_num_entities(self):
        return self.run_query("MATCH (n) RETURN count(n) as num")[0]['num']

    def get_num_enwiki_links(self):
        return self.run_query("MATCH (n) UNWIND n.provenance AS wiki RETURN count(DISTINCT wiki) as num")[0]['num']

    def get_num_relations(self):
        return self.run_query("MATCH ()-[r]->() RETURN count(r) as num")[0]['num']

    def get_schema(
            self,
            exclude_properties=['eid', 'description', 'aliases', 'provenance'],
            map_to_categorical=False,
            categorical_threshold=0.1,
            mode: Literal['direct', 'apoc'] = 'direct'
    ) -> PropertyGraphSchema:
        entities = {}
        relations = {}  # key is "label#subj_label#obj_label"

        if mode == 'direct':
            rel_query = DIRECT_REL_QUERY
            node_properties_query = DIRECT_NODE_PROPERTIES_QUERY
            rel_properties_query = DIRECT_REL_PROPERTIES_QUERY
        elif mode == 'apoc':
            rel_query = APOC_META_REL_QUERY
            node_properties_query = APOC_META_NODE_PROPERTIES_QUERY
            rel_properties_query = APOC_META_REL_PROPERTIES_QUERY
        else:
            raise ValueError(f"Invalid schema fetching mode: {mode}")

        for record in self.run_query(node_properties_query):
            label = record['label']
            entities[label] = EntitySchema(label=label, properties={})
            for prop in record['properties']:
                if prop['property'] in exclude_properties:
                    continue
                if prop['type'] == 'STRING' and map_to_categorical:
                    d = self.run_query(VALUE_COUNT_QUERY % (label, prop['property']))[0]
                    if d['prop_cnt'] / d['node_cnt'] < categorical_threshold:
                        dtype = DataType.CATEGORICAL
                    else:
                        dtype = DataType.STR
                else:
                    dtype = DataType.from_neo4j_type(prop['type'])
                entities[label].properties[prop['property']] = dtype

        for record in self.run_query(rel_query):
            label = record['type']
            subj_label = record['start']
            obj_label = record['end']
            relations[(label, subj_label, obj_label)] = RelationSchema(
                label=label,
                subj_label=subj_label,
                obj_label=obj_label,
                properties={}
            )

        for record in self.run_query(rel_properties_query):
            rel_type = record['type']
            for key, rel in relations.items():
                if key[0] == rel_type:
                    for prop in record['properties']:
                        if prop['property'] in exclude_properties:
                            continue
                        relations[key].properties[prop['property']] = DataType.from_neo4j_type(prop['type'])

        return PropertyGraphSchema(name=self.name, entities=list(entities.values()),
                                   relations=list(relations.values())).to_sorted()
