import argparse
import os
import time
import json
import datetime
import requests
from pint import UnitRegistry
import urllib.parse
from cypherbench.wd2neo4j.resolver import *
from cypherbench.wd2neo4j.schema import *

SPARQL_WAIT_TIME_SECONDS = 1.0

# ?rqid acts as a pseudo-randomized ID which is used by ORDER BY and LIMIT to fetch a random but consistent subset of entities
RQID_BINDING = "\n    BIND(SUBSTR(STR(FLOOR(xsd:integer(SUBSTR(STR(?item), 33)) / 307)), 2) AS ?rqid)."

ENTITY_SPARQL = """
SELECT DISTINCT ?item ?itemLabel ?itemDescription ?siteLink WHERE {<rqid_clause>
    <entity_sparql>
    OPTIONAL {?siteLink schema:about ?item; schema:isPartOf <https://en.wikipedia.org/>. }
    OPTIONAL {?item schema:description ?itemDescription. FILTER(LANG(?itemDescription) = "en") }
    ?item rdfs:label ?itemLabel. FILTER(LANG(?itemLabel) = "en")
}
""".strip('\n')

ENTITY_ALIAS_SPARQL = """
SELECT DISTINCT ?item ?itemAltLabel WHERE {<rqid_clause>
    <entity_sparql>
    ?item skos:altLabel ?itemAltLabel. FILTER(LANG(?itemAltLabel) = "en")
}
""".strip('\n')

ENTITY_PROPERTY_SPARQL = """
SELECT DISTINCT ?item ?property ?propertyLabel ?value ?unit ?unitLabel ?date ?datePrecision WHERE {<rqid_clause>
    <entity_sparql>
    ?item wdt:<wd_pid> ?property1. ?item p:<wd_pid> ?statement. ?statement ps:<wd_pid> ?property. FILTER(?property1 = ?property)<wd_constraints> 
    ?item rdfs:label ?itemLabel. FILTER(LANG(?itemLabel) = "en").
    OPTIONAL { ?property rdfs:label ?propertyLabel. FILTER(LANG(?propertyLabel) = "en"). }
    OPTIONAL {
      ?statement psv:<wd_pid> ?valuenode.
      ?valuenode wikibase:quantityAmount ?value.
      ?valuenode wikibase:quantityUnit ?unit.
      ?unit rdfs:label ?unitLabel. FILTER(LANG(?unitLabel) = "en")
    }
    OPTIONAL {
        ?statement psv:<wd_pid> ?valuenode.
        ?valuenode wikibase:timeValue ?date.
        ?valuenode wikibase:timePrecision ?datePrecision.
    }
}
""".strip('\n')

ENTITY_PROPERTY_LINKED_TO_SPARQL = """
SELECT DISTINCT ?item (BOUND(?link) AS ?isLinked) WHERE {<rqid_clause>
    <entity_sparql>
    OPTIONAL { ?item wdt:<wd_pid> wd:<wd_qid>. BIND(TRUE AS ?link) }
}
""".strip('\n')

RELATION_SPARQL = """
SELECT DISTINCT ?subj ?obj ?statement <relation_property_variables>
WHERE {<rqid_clause>
    <relation_sparql>
<relation_property_clauses>
}
""".strip('\n')

RELATION_PROPERTY_VARIABLES = "?property0 ?property0Label ?value0 ?unit0 ?unit0Label ?date0 ?datePrecision0"

RELATION_PROPERTY_CLAUSE = """
    OPTIONAL { 
        ?statement <wd_prefix>:<wd_pid> ?property0. 
        OPTIONAL { ?property0 rdfs:label ?property0Label. FILTER (LANG(?property0Label) = "en") }
        OPTIONAL { 
            ?statement <wd_prefix>v:<wd_pid> [ 
                wikibase:quantityAmount ?value0; 
                wikibase:quantityUnit ?unit0 
            ]. 
            FILTER(?value0 = ?property0) 
            OPTIONAL { ?unit0 rdfs:label ?unit0Label. FILTER (LANG(?unitLabel) = "en") } 
        }   
        OPTIONAL { 
            ?statement <wd_prefix>v:<wd_pid> [ 
                wikibase:timeValue ?date0; 
                wikibase:timePrecision ?datePrecision0 
            ]. 
            FILTER(?date0 = ?property0) 
        } 
    }
""".strip('\n')

UNIT_REGISTRY = UnitRegistry()

EXTRA_UNITS = {
    'earth radius': (6371.0, 'kilometre'),
    'jupiter radius': (69911.0, 'kilometre'),
    'solar radius': (695700.0, 'kilometre'),
    'light-year': (9460730472580.8, 'kilometre'),
    'nautical mile': (1.852, 'kilometre'),
    'astronomical unit': (149597870.7, 'kilometre'),
    'parsec': (30856775814913.672789, 'kilometre'),
    'earth mass': (5.972e24, 'kilogram'),
    'jupiter mass': (1.89813e27, 'kilogram'),
    'solar mass': (1.9884e30, 'kilogram'),
}


def convert_unit(value, src_unit, trg_unit):
    if src_unit in EXTRA_UNITS:
        ratio, middle_unit = EXTRA_UNITS[src_unit]
        value *= ratio
        src_unit = middle_unit

    if trg_unit in EXTRA_UNITS:
        ratio, middle_unit = EXTRA_UNITS[trg_unit]
        return UNIT_REGISTRY.Quantity(value, src_unit).to(middle_unit).magnitude / ratio
    else:
        return UNIT_REGISTRY.Quantity(value, src_unit).to(trg_unit).magnitude


class WD2Neo4j:
    def __init__(self,
                 sparql_url='https://query.wikidata.org/sparql',
                 sparql_wait_time_seconds=SPARQL_WAIT_TIME_SECONDS,
                 max_entities_per_root: Optional[int] = None,
                 no_property_mode=False,
                 debug=False
                 ):
        self.sparql_url = sparql_url
        self.sparql_wait_time_seconds = sparql_wait_time_seconds
        self.max_entities_per_root = None if max_entities_per_root is None or max_entities_per_root <= 0 else max_entities_per_root
        self.no_property_mode = no_property_mode
        self.debug = debug

        self.max_rqid_per_class: Dict[str, str] = {}

    def execute_sparql(self, query: str):
        try:
            if self.debug:
                print(f'<query_start>{query}<query_end>')
            raw_resp = requests.get(self.sparql_url, params={'format': 'json', 'query': query})
            resp = raw_resp.json()
            time.sleep(self.sparql_wait_time_seconds)
        except Exception as e:
            print(f'<query_start>{query}<query_end>')
            print(f'<resp_start>{raw_resp}<resp_end>')
            if raw_resp.status_code == 429:
                retry_after = raw_resp.headers.get('Retry-After')
                print(f'Too many requests, please retry after {retry_after} seconds')
                exit(9)
            raise e
        return resp

    @staticmethod
    def process_date_with_precision(date_str, precision, return_date=False):
        """
        Process a Wikidata date string based on its precision.
        Returns a string indicating the precise part of the date, formatted as "30 December 1984",
        "December 1984", etc., according to Wikidata's format.
        """
        if date_str.startswith('-'):
            return None
        date_object = datetime.datetime.strptime(date_str, '%Y-%m-%dT%H:%M:%SZ')

        if return_date:
            if precision >= 11:
                return date_object.date()
            else:
                return None

        if precision == 11:  # Day precision
            return date_object.strftime('%d %B %Y')
        elif precision == 10:  # Month precision
            return date_object.strftime('%B %Y')
        elif precision == 9:  # Year precision
            return date_object.strftime('%Y')
        elif precision == 8:  # Decade precision
            decade_start = (date_object.year // 10) * 10
            return f"{decade_start}s"
        elif precision == 7:  # Century precision
            century = (date_object.year // 100) + 1
            return f"{century}th century" if century != 21 else f"{century}st century"
        else:
            # print('Warning: unsupported date precision', precision)
            return None

    def get_entities(self, entity_schema: WDEntitySchema) -> List[WDEntity]:
        """
        Returns a mapping from Wikidata QID to entity data.
        """
        print(f'Fetching {entity_schema.label}')

        t0 = time.time()
        sparql = ENTITY_SPARQL.replace('<entity_sparql>', entity_schema.wd_sparql)

        if self.max_entities_per_root is not None:
            if len(entity_schema.wd_sparql_roots) == 0:
                sparql = sparql.replace('<rqid_clause>', RQID_BINDING)
                sparql = sparql.replace('SELECT DISTINCT ?item', 'SELECT DISTINCT ?rqid ?item')
                sparql += f' ORDER BY ?rqid LIMIT {self.max_entities_per_root}'
            else:
                root_max_rqid = max(self.max_rqid_per_class[root] for root in entity_schema.wd_sparql_roots)
                rqid_clause = f'{RQID_BINDING} FILTER(?rqid <= "{root_max_rqid}")'
                rqid_clause = rqid_clause.replace('?item', '?root').replace('?rqid', '?rootrqid')
                sparql = sparql.replace('<rqid_clause>', rqid_clause)
        else:
            sparql = sparql.replace('<rqid_clause>', '')

        resp = self.execute_sparql(sparql)
        print(f'SPARQL query took {time.time() - t0:.2f} seconds')
        print(f'Fetched {len(resp["results"]["bindings"])} entities')

        res: Dict[str, WDEntity] = {}
        # for wd_qid, wd_label, enwiki_url in self.run_wikidata_sparql_entity(sparql):
        for d in resp['results']['bindings']:
            wd_qid = d['item']['value'].rsplit('/', 1)[-1]
            name = d['itemLabel']['value'] if d['itemLabel']['value'] not in (wd_qid, '') else None
            enwiki_title = urllib.parse.unquote(
                d['siteLink']['value'].split('en.wikipedia.org/wiki/')[-1].replace('_',
                                                                                   ' ')) if 'siteLink' in d else None
            description = d['itemDescription']['value'] if 'itemDescription' in d else None

            if name is None:
                continue

            res[wd_qid] = WDEntity(label=entity_schema.label, wikidata_qid=wd_qid, name=name,
                                   description=description, aliases=[], enwiki_title=enwiki_title, properties={})

        if self.max_entities_per_root and len(entity_schema.wd_sparql_roots) == 0:
            max_rqid = resp['results']['bindings'][-1]['rqid']['value']
            self.max_rqid_per_class[entity_schema.label] = max_rqid

        if self.max_entities_per_root is not None:
            if len(entity_schema.wd_sparql_roots) == 0:
                rqid_clause = f'{RQID_BINDING} FILTER(?rqid <= "{max_rqid}")'
            else:
                root_max_rqid = max(self.max_rqid_per_class[root] for root in entity_schema.wd_sparql_roots)
                rqid_clause = f'{RQID_BINDING} FILTER(?rqid <= "{root_max_rqid}")'
                rqid_clause = rqid_clause.replace('?item', '?root').replace('?rqid', '?rootrqid')
        else:
            rqid_clause = ''

        t0 = time.time()
        sparql = ENTITY_ALIAS_SPARQL.replace('<entity_sparql>', entity_schema.wd_sparql)
        sparql = sparql.replace('<rqid_clause>', rqid_clause)
        resp = self.execute_sparql(sparql)
        print(f'SPARQL query took {time.time() - t0:.2f} seconds')
        for d in resp['results']['bindings']:
            wd_qid = d['item']['value'].rsplit('/', 1)[-1]
            alias = d['itemAltLabel']['value'].strip()
            if wd_qid in res and alias and alias not in res[wd_qid].aliases:
                res[wd_qid].aliases.append(alias)

        if self.no_property_mode:
            return list(res.values())

        for property_schema in entity_schema.properties:
            property_label = property_schema.label
            print(f'Fetching {entity_schema.label}.{property_label}')

            if re.match(Op.LINKED_TO, property_schema.wd_source):
                _, (wd_pid, wd_qid) = Op.parse(property_schema.wd_source)
                sparql = ENTITY_PROPERTY_LINKED_TO_SPARQL
                sparql = sparql.replace('<entity_sparql>', entity_schema.wd_sparql)
                sparql = sparql.replace('<wd_pid>', wd_pid)
                sparql = sparql.replace('<wd_qid>', wd_qid)
            else:
                assert re.match(r'^P\d+$', property_schema.wd_source)
                wd_pid = property_schema.wd_source
                constraints = []
                for constraint in property_schema.wd_constraints:
                    if re.match(Op.LINKED_TO, constraint):
                        _, (pid, qid) = Op.parse(constraint)
                        constraints.append(f'?statement pq:{pid} wd:{qid}.')
                wd_constraints = (' ' + ' '.join(constraints)) if constraints else ''

                sparql = ENTITY_PROPERTY_SPARQL
                sparql = sparql.replace('<entity_sparql>', entity_schema.wd_sparql)
                sparql = sparql.replace('<wd_pid>', wd_pid)
                sparql = sparql.replace('<wd_constraints>', wd_constraints)

            sparql = sparql.replace('<rqid_clause>', rqid_clause)

            t0 = time.time()
            resp = self.run_wikidata_sparql_property(sparql, property_schema)
            print(f'SPARQL query took {time.time() - t0:.2f} seconds')
            for wd_qid, value in resp:
                if wd_qid not in res:
                    # print(f'Warning: {wd_qid} not found in {entity_schema.label}')
                    continue
                if value is None:
                    continue
                if property_schema.datatype.is_array():
                    if property_label not in res[wd_qid].properties:
                        res[wd_qid].properties[property_label] = []
                    if value not in res[wd_qid].properties[property_label]:
                        res[wd_qid].properties[property_label].append(value)
                        res[wd_qid].properties[property_label] = sorted(res[wd_qid].properties[property_label])
                else:
                    if property_label not in res[wd_qid].properties:
                        res[wd_qid].properties[property_label] = value
                    else:
                        # print(f'Warning: overwriting {wd_qid}.{property_label}')
                        existing_value = res[wd_qid].properties[property_label]
                        if (len(str(value)), value) < (len(str(existing_value)), existing_value):
                            res[wd_qid].properties[property_label] = value

        entities = list(res.values())
        entities = sorted(entities, key=lambda e: e.wikidata_qid)
        for e in entities:
            e.aliases = sorted(e.aliases)
        return entities

    def run_wikidata_sparql_property(self, query, property_schema: WDEntityPropertySchema):
        # print(f'<start>{query}<end>')
        resp = self.execute_sparql(query)
        results = []
        datatype = property_schema.datatype.primitive_type()
        for d in resp['results']['bindings']:
            subj = d['item']['value'].rsplit('/', 1)[-1]
            try:
                if property_schema.quantity_unit is not None and 'value' not in d:
                    print(f'Warning: {subj}.{property_schema.label} is not a quantity')
                    property_value = None
                elif property_schema.is_year and 'date' not in d:
                    print(f'Warning: {subj}.{property_schema.label} is not a date')
                    property_value = None
                elif 'isLinked' in d:
                    assert d['isLinked']['value'] in ('true', 'false')
                    property_value = d['isLinked']['value'] == 'true'
                elif 'date' in d:  # date
                    if property_schema.is_year:
                        precision = int(d['datePrecision']['value'])
                        if precision >= 9:
                            property_value = self.process_date_with_precision(d['date']['value'], 9)
                            if not re.fullmatch(r'[1-9]\d{3}', property_value):
                                property_value = None
                        else:
                            property_value = None
                    else:
                        property_value = self.process_date_with_precision(
                            d['date']['value'],
                            int(d['datePrecision']['value']),
                            return_date=datatype == datetime.date
                        )
                elif 'value' in d:  # quantities
                    if property_schema.quantity_unit is None:
                        property_value = d['value']['value']
                        if 'unitLabel' in d and d['unitLabel']['value'] != '1':
                            property_value += ' ' + d['unitLabel']['value']
                    elif not property_schema.quantity_convert_unit:
                        if d['unitLabel']['value'].lower() == property_schema.quantity_unit.lower():
                            property_value = float(d['value']['value'])
                        else:
                            property_value = None
                    else:
                        value = float(d['value']['value'])
                        src_unit = d['unitLabel']['value'].lower()
                        trg_unit = property_schema.quantity_unit.lower()
                        property_value = convert_unit(value, src_unit, trg_unit)
                elif d['property']['value'].startswith('http://www.wikidata.org/entity/'):  # Wikidata entity
                    property_value = d['propertyLabel']['value'] if 'propertyLabel' in d else None
                else:  # literal values that are not dates or quantities (include strings, numbers that aren't quantities, etc.)
                    property_value = d['property']['value']
                if property_value is not None and not isinstance(property_value, datatype):
                    property_value = datatype(property_value)
            except Exception as e:
                print(f'Error: {e} for {subj}.{property_schema.label}')
                property_value = None
            if property_value is not None:
                results.append((subj, property_value))
        return results

    def get_all_relations(self, schema: WDNeo4jSchema) -> List[WDRelation]:
        res = []
        eschemas = {e.label: e for e in schema.entities}
        for relation_schema in schema.relations:
            print(
                f'Fetching {relation_schema.label} ({relation_schema.subj_label} -> {relation_schema.obj_label})')
            property_variables = RELATION_PROPERTY_VARIABLES.split()
            property_conditions = []
            # construct the OPTIONAL blocks for fetching properties
            for idx, prop in enumerate(relation_schema.properties):
                if self.no_property_mode:
                    continue

                wd_pid = prop.wd_source
                prop_sparql = RELATION_PROPERTY_CLAUSE.replace('<wd_pid>', wd_pid)
                if prop.is_qualifier:
                    prop_sparql = prop_sparql.replace('<wd_prefix>', 'pq')
                else:
                    prop_sparql = prop_sparql.replace('<wd_prefix>', 'ps')
                for var in property_variables:
                    prop_sparql = prop_sparql.replace(var, var.replace('0', str(idx)))
                property_conditions.append(prop_sparql)

            property_variables = ' '.join(
                RELATION_PROPERTY_VARIABLES.replace('0', str(idx)) for idx in range(len(relation_schema.properties))
            )

            sparql = RELATION_SPARQL
            sparql = sparql.replace('<relation_sparql>', relation_schema.wd_sparql)
            sparql = sparql.replace('<relation_property_variables>', property_variables)
            sparql = sparql.replace('<relation_property_clauses>', '\n'.join(property_conditions))
            if self.max_entities_per_root is not None:
                if 'subj' in relation_schema.wd_sparql_heads:
                    if len(eschemas[relation_schema.subj_label].wd_sparql_roots) == 0:
                        subj_max_rqid = self.max_rqid_per_class[relation_schema.subj_label]
                        subj_rqid_clause = f'{RQID_BINDING} FILTER(?rqid <= "{subj_max_rqid}")'
                        subj_rqid_clause = subj_rqid_clause.replace('?rqid', '?subjrqid').replace('?item', '?subj')
                    else:
                        root1_max_rqid = max(self.max_rqid_per_class[root] for root in
                                             eschemas[relation_schema.subj_label].wd_sparql_roots)
                        subj_rqid_clause = f'{RQID_BINDING} FILTER(?rqid <= "{root1_max_rqid}")'
                        subj_rqid_clause = subj_rqid_clause.replace('?rqid', '?root1rqid').replace('?item', '?root1')
                else:
                    subj_rqid_clause = ''

                if 'obj' in relation_schema.wd_sparql_heads:
                    if len(eschemas[relation_schema.obj_label].wd_sparql_roots) == 0:
                        obj_max_rqid = self.max_rqid_per_class[relation_schema.obj_label]
                        obj_rqid_clause = f'{RQID_BINDING} FILTER(?rqid <= "{obj_max_rqid}")'
                        obj_rqid_clause = obj_rqid_clause.replace('?rqid', '?objrqid').replace('?item', '?obj')
                    else:
                        root2_max_rqid = max(self.max_rqid_per_class[root] for root in
                                             eschemas[relation_schema.obj_label].wd_sparql_roots)
                        obj_rqid_clause = f'{RQID_BINDING} FILTER(?rqid <= "{root2_max_rqid}")'
                        obj_rqid_clause = obj_rqid_clause.replace('?rqid', '?root2rqid').replace('?item', '?root2')
                else:
                    obj_rqid_clause = ''

                sparql = sparql.replace('<rqid_clause>', f'{subj_rqid_clause}{obj_rqid_clause}')
            else:
                sparql = sparql.replace('<rqid_clause>', '')

            res += self.run_wikidata_sparql_relation(sparql, relation_schema)

        return res

    def run_wikidata_sparql_relation(self, query, relation_schema) -> List[WDRelation]:
        # print(f'<start>{query}<end>')
        t0 = time.time()
        resp = self.execute_sparql(query)
        print(f'SPARQL query took {time.time() - t0:.2f} seconds')
        res = {}  # (subj, obj) -> Dict[statement, properties]
        for d in resp['results']['bindings']:
            subj = d['subj']['value'].rsplit('/', 1)[-1]
            obj = d['obj']['value'].rsplit('/', 1)[-1]
            if (subj, obj) not in res:
                res[(subj, obj)] = {}
            statement = d['statement']['value']
            if statement not in res[(subj, obj)]:
                res[(subj, obj)][statement] = {}
            statement_properties = res[(subj, obj)][statement]
            for idx, prop_schema in enumerate(relation_schema.properties):
                datatype = prop_schema.datatype.primitive_type()
                key = f'property{idx}'
                if key not in d:
                    continue
                try:
                    if prop_schema.quantity_unit is not None and 'value' not in d:
                        print(f'Warning: {relation_schema.label}({subj}, {obj}).{prop_schema.label} is not a quantity')
                        prop_value = None
                    elif prop_schema.is_year and f'date{idx}' not in d:
                        print(f'Warning: {relation_schema.label}({subj}, {obj}).{prop_schema.label} is not a date')
                        prop_value = None
                    elif f'date{idx}' in d:  # date
                        if prop_schema.is_year:
                            precision = int(d[f'datePrecision{idx}']['value'])
                            if precision >= 9:
                                prop_value = self.process_date_with_precision(d[f'date{idx}']['value'], 9)
                                if not re.fullmatch(r'[1-9]\d{3}', prop_value):
                                    prop_value = None
                            else:
                                prop_value = None
                        else:
                            prop_value = self.process_date_with_precision(
                                d[f'date{idx}']['value'],
                                int(d[f'datePrecision{idx}']['value']),
                                return_date=datatype == datetime.date
                            )
                    elif f'value{idx}' in d:  # quantities
                        if prop_schema.quantity_unit is None:
                            prop_value = d[f'value{idx}']['value'] + (
                                (' ' + d[f'unitLabel{idx}']['value']) if f'unitLabel{idx}' in d else '')
                        elif not prop_schema.quantity_convert_unit:
                            if d[f'unitLabel{idx}']['value'].lower() == prop_schema.quantity_unit.lower():
                                prop_value = float(d[f'value{idx}']['value'])
                            else:
                                prop_value = None
                        else:
                            value = float(d[f'value{idx}']['value'])
                            src_unit = d[f'unitLabel{idx}']['value'].lower()
                            trg_unit = prop_schema.quantity_unit.lower()
                            prop_value = convert_unit(value, src_unit, trg_unit)
                    elif d[key]['value'].startswith('http://www.wikidata.org/entity/'):  # Wikidata entity
                        prop_value = d[f'{key}Label']['value'] if f'{key}Label' in d else None
                    else:  # literal values that are not dates or quantities (include strings, numbers that aren't quantities, etc.)
                        prop_value = d[key]['value']
                    if prop_value is not None and not isinstance(prop_value, datatype):
                        prop_value = datatype(prop_value)
                except Exception as e:
                    print(f'Error: {e} for {relation_schema.label}({subj}, {obj}).{prop_schema.label}')
                    prop_value = None
                if prop_value is None:
                    continue
                if prop_schema.datatype.is_array():
                    if prop_schema.label not in statement_properties:
                        statement_properties[prop_schema.label] = []
                    if prop_value not in statement_properties[prop_schema.label]:
                        statement_properties[prop_schema.label].append(prop_value)
                        statement_properties[prop_schema.label] = sorted(statement_properties[prop_schema.label])
                else:
                    if prop_schema.label not in statement_properties:
                        statement_properties[prop_schema.label] = prop_value
                    else:
                        # print(f'Warning: overwriting {relation_schema.label}({subj}, {obj}).{prop_schema.label}')
                        exiting_value = statement_properties[prop_schema.label]
                        if (len(str(prop_value)), prop_value) < (len(str(exiting_value)), exiting_value):
                            statement_properties[prop_schema.label] = prop_value

        relations = []
        for subj, obj in sorted(res.keys()):
            for statement in sorted(res[(subj, obj)].keys()):
                properties = res[(subj, obj)][statement]
                relations.append(WDRelation(
                    label=relation_schema.label, subj_label=relation_schema.subj_label,
                    obj_label=relation_schema.obj_label,
                    subj_wikidata_qid=subj, obj_wikidata_qid=obj, properties=properties
                ))
        return relations

    def transform(self, schema: WDNeo4jSchema) -> WikidataKG:
        root_entities = [e for e in schema.entities if len(e.wd_sparql_roots) == 0]
        print()
        print(f'Root entities: {", ".join(e.label for e in root_entities)}')
        print()

        all_entities = []
        for entity_schema in schema.entities:
            all_entities += self.get_entities(entity_schema)

        all_relations = self.get_all_relations(schema)

        # Postprocessing - Remove relations whose subj or obj is not in the entities
        elabel2qids = collections.defaultdict(set)
        for e in all_entities:
            elabel2qids[e.label].add(e.wikidata_qid)

        rschema2r = {f'{r.label}#{r.subj_label}#{r.obj_label}': r for r in schema.relations}
        filtered_relations = []
        for r in all_relations:
            rs = rschema2r[f'{r.label}#{r.subj_label}#{r.obj_label}']
            subj_label = rs.subj_label
            obj_label = rs.obj_label
            if r.subj_wikidata_qid in elabel2qids[subj_label] and r.obj_wikidata_qid in elabel2qids[obj_label]:
                filtered_relations.append(r)

        print(f'Removed {len(all_relations) - len(filtered_relations)} relations with out-of-scope entities')

        return WikidataKG(schema=schema, entities=all_entities, relations=filtered_relations)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--neo4j_schema', default='schemas/sport.nba.json',
                        help='path to Neo4j schema file')
    parser.add_argument('--output_dir', default='output/graphs/', help='output directory')
    parser.add_argument('--overwrite', default=False, action='store_true')
    parser.add_argument('-e', '--endpoint', default='wikidata', choices=['local', 'wikidata'],
                        help='endpoint to fetch data from')
    parser.add_argument('--sparql_url', default='https://query.wikidata.org/sparql',
                        choices=['http://localhost:9999/bigdata/namespace/wdq/sparql',
                                 'https://query.wikidata.org/sparql'], help='Wikidata SPARQL endpoint')
    parser.add_argument('--sparql_wait_time_seconds', default=1., type=float,
                        help='Wait time between SPARQL queries')
    parser.add_argument('--max_entities_per_root', default=None, type=int)
    parser.add_argument('--no_properties', default=False, action='store_true', help='Do not fetch properties')
    parser.add_argument('--debug', default=False, action='store_true', help='Debug mode')
    parser.add_argument('--resolve_only', default=False, action='store_true', help='Resolve only')
    parser.add_argument('--run_sparql', default=None, type=str)
    args = parser.parse_args()
    if args.endpoint == 'local':
        parser.set_defaults(sparql_url='http://localhost:9999/bigdata/namespace/wdq/sparql',
                            sparql_wait_time_seconds=0.)
    elif args.endpoint == 'wikidata':
        parser.set_defaults(sparql_url='https://query.wikidata.org/sparql',
                            sparql_wait_time_seconds=1.)
    # if "SPARQL_URL" in os.environ:
    #     parser.set_defaults(sparql_url=os.environ["SPARQL_URL"])
    # if "SPARQL_WAIT_TIME_SECONDS" in os.environ:
    #     parser.set_defaults(sparql_wait_time_seconds=float(os.environ["SPARQL_WAIT_TIME_SECONDS"]))
    args = parser.parse_args()
    print(args)
    print()

    if args.run_sparql is not None:
        t0 = time.time()
        raw_resp = requests.get(args.sparql_url, params={'format': 'json', 'query': args.run_sparql})
        try:
            resp = raw_resp.json()
            print(json.dumps(resp, indent=2))
            print(f'SPARQL query took {time.time() - t0:.2f} seconds')
            print(f'Number of results: {len(resp["results"]["bindings"])}')
        except:
            print(raw_resp)
        return

    output_path = os.path.join(args.output_dir, os.path.basename(args.neo4j_schema).replace('.json', '-graph.json'))

    if os.path.exists(output_path) and not args.overwrite:
        print(f'File {output_path} already exists. Use --overwrite to overwrite.')
        return

    with open(args.neo4j_schema, 'r') as f:
        schema = json.load(f)

    schema = WDNeo4jSchema(**schema)

    resolver = SPARQLResolver()
    schema = resolver.resolve(schema)
    resolved_output_path = os.path.join(args.output_dir,
                                        os.path.basename(args.neo4j_schema).replace('.json', '-schema.json'))
    with open(resolved_output_path, 'w') as f:
        json.dump(schema.model_dump(mode='json'), f, indent=2)
    print(f'Resolved schema saved to {resolved_output_path}')

    if args.resolve_only:
        return

    wd2neo4j = WD2Neo4j(
        sparql_url=args.sparql_url,
        sparql_wait_time_seconds=args.sparql_wait_time_seconds,
        max_entities_per_root=args.max_entities_per_root,
        no_property_mode=args.no_properties,
        debug=args.debug
    )

    t0 = time.time()
    kg = wd2neo4j.transform(schema)
    print()
    print(f'KG transformation took {time.time() - t0:.2f} seconds')

    with open(output_path, 'w') as f:
        json.dump(kg.model_dump(mode='json'), f, indent=2)

    print(f'Output saved to {output_path}')

    kg.print_stats()


if __name__ == '__main__':
    main()
