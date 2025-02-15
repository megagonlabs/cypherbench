import copy

from pydantic import BaseModel
from typing import Optional, Dict, List, Literal, Any
import re
import json
from enum import Enum


class TemplateInfo(BaseModel):
    match_category: str
    match_cypher: str
    return_pattern_id: str
    return_cypher: str


class Nl2CypherSample(BaseModel):
    qid: str
    graph: str
    gold_cypher: str
    gold_match_cypher: Optional[str] = None
    nl_question: Optional[str]
    nl_question_raw: Optional[str] = None
    answer_json: Optional[str] = None
    from_template: TemplateInfo
    pred_cypher: Optional[str] = None
    metrics: Dict[str, float] = {}


class DataType(Enum):
    CATEGORICAL = 'categorical'
    STR = 'str'
    INT = 'int'
    FLOAT = 'float'
    BOOL = 'bool'
    DATE = 'date'
    STR_ARRAY = 'list[str]'

    @classmethod
    def from_neo4j_type(cls, t: str):
        mapping = {
            'STRING': cls.STR,
            'INTEGER': cls.INT,
            'FLOAT': cls.FLOAT,
            'BOOLEAN': cls.BOOL,
            'DATE': cls.DATE,
            'LIST': cls.STR_ARRAY,
            'LIST OF STRING': cls.STR_ARRAY
        }
        return mapping[t]

    @classmethod
    def from_simplekg_type(cls, t: str):
        mapping = {
            'str': cls.STR,
            'int': cls.INT,
            'float': cls.FLOAT,
            'bool': cls.BOOL,
            'date': cls.DATE,
            'list[str]': cls.STR_ARRAY
        }
        return mapping[t]


class EntitySchema(BaseModel):
    label: str
    description: Optional[str] = None
    properties: dict[str, DataType]


class RelationSchema(BaseModel):
    label: str
    subj_label: str
    obj_label: str
    properties: dict[str, DataType]


class PropertyGraphSchema(BaseModel):
    name: str
    entities: list[EntitySchema]
    relations: list[RelationSchema]

    def to_json(self, exclude_description=False):
        res = self.model_dump(mode='json')
        if exclude_description:
            for ent in res['entities']:
                ent.pop('description')
        return res

    def to_str(self, exclude_description=False):
        return json.dumps(self.to_json(exclude_description), indent=2)

    def to_sorted(self) -> 'PropertyGraphSchema':
        schema = copy.deepcopy(self)
        schema.entities = sorted(schema.entities, key=lambda x: x.label)
        schema.relations = sorted(schema.relations, key=lambda x: (x.label, x.subj_label, x.obj_label))
        for x in schema.entities + schema.relations:
            x.properties = dict(sorted(x.properties.items()))
        return PropertyGraphSchema(**schema.model_dump(mode='json'))

    @classmethod
    def from_json(cls, data, add_meta_properties={'name': DataType.STR}):
        schema = cls(**data)
        for ent in schema.entities:
            if add_meta_properties:
                ent.properties = dict(**add_meta_properties, **ent.properties)
        schema = cls(**schema.model_dump(mode='json'))  # Validate again
        return schema


def deduplicate(input_list):
    seen = set()
    return [x for x in input_list if not (x in seen or seen.add(x))]


class RelationInfo(BaseModel):
    label: str
    variants: List[str]
    is_symmetric: bool
    is_time_sensitive: bool
    is_mandatory_subj: bool
    is_mandatory_obj: bool
    subj_cardinality: Literal['one', 'many']
    obj_cardinality: Literal['one', 'many']
    implied_relations: List[str] = []


class GraphInfo(BaseModel):
    relations: List[RelationInfo]


class MatchPattern(BaseModel):
    category: str
    cypher: str
    nl_args: Dict[str, str]
    return_category: str

    @property
    def node_vars(self) -> list[str]:
        if not hasattr(self, '_node_vars'):
            self._node_vars = deduplicate(re.findall(r'\((\w+)(?:<.*?>)?\)', self.cypher))
        return self._node_vars

    @property
    def singletons(self) -> list[str]:
        if not hasattr(self, '_singletons'):
            seen = set()
            for r, subj, obj in self.relations:
                seen.add(subj)
                seen.add(obj)
            self._singletons = [n for n in self.node_vars if n not in seen]
        return self._singletons

    @property
    def node_vars_with_name(self) -> list[str]:
        if not hasattr(self, '_node_vars_with_name'):
            self._node_vars_with_name = [n for n in self.node_vars if f'{n}.name' in self.property_constraints]
        return self._node_vars_with_name

    @property
    def rel_vars(self) -> list[str]:
        if not hasattr(self, '_rel_vars'):
            self._rel_vars = deduplicate(re.findall(r'-\[(\w+)(?:<.*?>)?\]-', self.cypher))
        return self._rel_vars

    @property
    def relations(self) -> list[tuple[str, str, str]]:
        if not hasattr(self, '_relations'):
            self._relations = []
            for subj, rel, obj in re.findall(r'(?=\((\w+)(?:<.*?>)?\)-\[(\w+)(?:<.*?>)?\]->\((\w+)(?:<.*?>)?\))',
                                             self.cypher):
                self._relations.append((rel, subj, obj))
            for obj, rel, subj in re.findall(r'(?=\((\w+)(?:<.*?>)?\)<-\[(\w+)(?:<.*?>)?\]-\((\w+)(?:<.*?>)?\))',
                                             self.cypher):
                self._relations.append((rel, subj, obj))
            self._relations = deduplicate(self._relations)
        return self._relations

    @property
    def property_constraints(self) -> dict[str, str]:
        if not hasattr(self, '_var2props'):
            self._properties = {}
            for var, props in re.findall(r'[\(\[](\w+)<(.*?)>[\)\]]', self.cypher):
                for label in props.split(','):
                    if label.endswith('^'):
                        key = f'{var}.{label[:-1]}'
                        constraint = f'{key} IS NULL'
                    elif label.endswith('?'):
                        key = f'{var}.{label[:-1]}'
                        constraint = None
                    else:
                        key = f'{var}.{label}'
                        constraint = f'{key} IS NOT NULL'
                    self._properties[key] = constraint
        return self._properties

    @property
    def cypher_clean(self):
        """
        `MATCH (n) OPTIONAL MATCH (n)-[r0<start_year>]->(m0<name>)` -> `MATCH (n) OPTIONAL MATCH (n)-[r0]->(m0)`
        """
        cypher = re.sub(r'\((\w+)(?:<.*?>)?\)', r'(\1)', self.cypher)
        cypher = re.sub(r'\[(\w+)(?:<.*?>)?\]', r'[\1]', cypher)
        return cypher

    @property
    def cypher_match_pattern(self):
        """
        `MATCH (n) OPTIONAL MATCH (n)-[r0<start_year>]->(m0<name>)` -> `MATCH (n)-[r0]->(m0)`
        """
        patterns = [f'({n})' for n in self.singletons] + \
                   [f'({subj})-[{rel}]->({obj})' for rel, subj, obj in self.relations]
        return 'MATCH ' + ', '.join(patterns)


class ReturnPattern(BaseModel):
    pattern_id: str
    category: str
    cypher: str
    nl_template: str
    return_vars: List[str]


class MatchClause(BaseModel):
    pattern: MatchPattern
    n_assignment: Dict[str, str]
    r_assignment: Dict[str, str]
    prop_values: Dict[str, Any]
    cypher: str
    nl_args: Dict[str, str]
    num_matched_node: int


class ReturnClause(BaseModel):
    cypher: str
    nl_question: str


class Nl2CypherGeneratorConfig(BaseModel):
    match_patterns: List[MatchPattern]
    return_patterns: List[ReturnPattern]
