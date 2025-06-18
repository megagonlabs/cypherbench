from pydantic import (
    BaseModel,
    Field,
    model_validator,
    ValidationError,
    conlist,
    constr,
    ConfigDict,
)
from enum import Enum
import re
import collections
import json
import datetime
from typing import List, Any, Union, Dict, Optional, Tuple, Literal

LABEL_STR = constr(min_length=1, pattern=r"^[a-zA-Z0-9_]+$")


class DataType(Enum):
    STR = "str"
    INT = "int"
    FLOAT = "float"
    BOOL = "bool"
    DATE = "date"
    STRLIST = "list[str]"
    INTLIST = "list[int]"
    FLOATLIST = "list[float]"
    DATELIST = "list[date]"

    def is_array(self):
        return self in [
            DataType.STRLIST,
            DataType.INTLIST,
            DataType.FLOATLIST,
            DataType.DATELIST,
        ]

    def primitive_type(self):
        return {
            DataType.STR: str,
            DataType.INT: int,
            DataType.FLOAT: float,
            DataType.BOOL: bool,
            DataType.DATE: datetime.date,
            DataType.STRLIST: str,
            DataType.INTLIST: int,
            DataType.FLOATLIST: float,
            DataType.DATELIST: datetime.date,
        }[self]

    def validate(self, value) -> bool:
        if not self.is_array():
            return isinstance(value, self.primitive_type())
        else:
            primitive = self.primitive_type()
            return (
                isinstance(value, list)
                and all(isinstance(v, primitive) for v in value)
                and len(value) > 0
            )


class SimpleEntitySchema(BaseModel):
    label: LABEL_STR
    description: Optional[str] = None
    properties: Dict[str, DataType] = {}


class SimpleRelationSchema(BaseModel):
    label: LABEL_STR
    subj_label: str
    obj_label: str
    properties: Dict[str, DataType] = {}


class SimpleKGSchema(BaseModel):
    name: str
    entities: List[SimpleEntitySchema]
    relations: List[SimpleRelationSchema]


class SimpleEntity(BaseModel):
    eid: str
    label: LABEL_STR
    name: str
    aliases: List[str] = []
    description: Optional[str] = None
    properties: Dict[str, Any]
    provenance: List[str] = []


class SimpleRelation(BaseModel):
    rid: str
    label: LABEL_STR
    subj_id: str
    obj_id: str
    properties: Dict[str, Any]
    provenance: List[str] = []


class SimpleKG(BaseModel):
    schema: SimpleKGSchema
    entities: List[SimpleEntity]
    relations: List[SimpleRelation]

    @model_validator(mode="after")
    def validate_schema(self):
        all_eids = [e.eid for e in self.entities]
        if len(all_eids) != len(set(all_eids)):
            raise ValueError("Duplicate entity ids found")
        all_rids = [r.rid for r in self.relations]
        if len(all_rids) != len(set(all_rids)):
            raise ValueError("Duplicate relation ids found")

        eschemas = {e.label: e for e in self.schema.entities}
        rschemas = {
            f"{r.label}#{r.subj_label}#{r.obj_label}": r for r in self.schema.relations
        }

        # Validate relation schemas
        eid2label = {e.eid: e.label for e in self.entities}
        for r in self.relations:
            subj_label = eid2label[r.subj_id]
            obj_label = eid2label[r.obj_id]
            if f"{r.label}#{subj_label}#{obj_label}" not in rschemas:
                raise ValueError(
                    f'Schema of relation "{r.label}({r.subj_id}, {r.obj_id})" is not Valid'
                )

            # convert date strings to datetime.date
            rschema = rschemas[f"{r.label}#{subj_label}#{obj_label}"]
            for key, datatype in rschema.properties.items():
                if datatype == DataType.DATE and isinstance(r.properties.get(key), str):
                    r.properties[key] = datetime.datetime.strptime(
                        r.properties[key], "%Y-%m-%d"
                    ).date()

            # validate relation properties
            for key, value in r.properties.items():
                if key not in rschema.properties:
                    raise ValueError(f'Entity "{r.rid}" has unknown property "{key}"')
                if not rschema.properties[key].validate(value):
                    raise ValueError(
                        f'Property "{key}" of relation "{r.rid}" has wrong datatype, expected {rschema.properties[key].value} but got {type(value)}'
                    )

        # Validate entities
        for ent in self.entities:
            if ent.label not in eschemas:
                raise ValueError(f'Entity type "{ent.label}" not found in schema')
            eschema = eschemas[ent.label]

            # convert dates to datetime.date
            for key, datatype in eschema.properties.items():
                if datatype == DataType.DATE and isinstance(
                    ent.properties.get(key), str
                ):
                    ent.properties[key] = datetime.datetime.strptime(
                        ent.properties[key], "%Y-%m-%d"
                    ).date()

            # validate entity properties
            for key, value in ent.properties.items():
                if key not in eschema.properties:
                    raise ValueError(f'Entity "{ent.eid}" has unknown property "{key}"')
                if not eschema.properties[key].validate(value):
                    raise ValueError(
                        f'Property "{key}" of entity "{ent.eid}" has wrong datatype, expected {eschema.properties[key].value} but got {type(value)}'
                    )
        return self


class Op:
    LINKED_TO = r"^\$LINKED_TO\((P\d+),(Q\d+)\)$"
    HAS_PROPERTY = r"^\$HAS_PROPERTY\((P\d+)\)$"
    RAW_SPARQL = r"^\$RAW_SPARQL\((.+)\)$"
    IDENTICAL = r"^\$IDENTICAL$"
    FROM_RELATION = r"^\$FROM_RELATION\(([a-zA-Z0-9_#]+(,[a-zA-Z0-9_#]+)*)\)$"

    @classmethod
    def parse(cls, s: str) -> Tuple[str, Tuple[str]]:
        """
        Return operator and arguments
        """
        for op in [
            "LINKED_TO",
            "IDENTICAL",
            "FROM_RELATION",
            "HAS_PROPERTY",
            "RAW_SPARQL",
        ]:
            m = re.match(getattr(cls, op), s)
            if m:
                if op == "FROM_RELATION":
                    args = m.group(1).split(",")
                else:
                    args = m.groups()
                return op, args
        return None, ()


LABEL_STR = constr(min_length=1, pattern=r"^[a-zA-Z0-9_]+$")

EntityWDSource = constr(pattern="^Q\d+$")
RelationWDSource = constr(pattern="|".join([r"^P\d+$", Op.IDENTICAL]))
EntityPropertyWDSource = constr(pattern="|".join([r"^P\d+$", Op.LINKED_TO]))
RelationPropertyWDSource = constr(pattern=r"^P\d+$")

EntityWDConstraint = constr(
    pattern="|".join([Op.LINKED_TO, Op.FROM_RELATION, Op.HAS_PROPERTY, Op.RAW_SPARQL])
)
EntityPropertyWDConstraint = constr(pattern="|".join([Op.LINKED_TO]))


class WDEntityPropertySchema(BaseModel):
    label: LABEL_STR
    wd_source: EntityPropertyWDSource
    wd_constraints: List[EntityPropertyWDConstraint] = []
    datatype: DataType = DataType.STR
    quantity_unit: Optional[str] = None
    quantity_convert_unit: bool = False
    is_year: bool = False
    disable: bool = False

    @model_validator(mode="after")
    def validate(self):
        if self.quantity_unit is not None and self.datatype != DataType.FLOAT:
            raise ValueError("quantity_unit is only allowed for datatype float")
        if self.is_year and self.datatype not in (DataType.STR, DataType.INT):
            raise ValueError("is_year is only allowed for datatype str or int")
        return self


class WDEntitySchema(BaseModel):
    label: LABEL_STR
    wd_source: EntityWDSource
    wd_is_instance: bool = True
    wd_include_subclasses: bool = True
    wd_constraints: List[EntityWDConstraint] = []
    properties: List[WDEntityPropertySchema] = []
    wd_sparql: Optional[str] = None
    wd_sparql_roots: List[str] = []
    disable: bool = False

    @model_validator(mode="after")
    def validate_include_subclasses(self):
        if self.wd_source == "Q5" and self.wd_include_subclasses:
            raise ValueError(f"wd_include_subclasses=True is not allowed for entity Q5")
        return self


class WDRelationPropertySchema(BaseModel):
    label: LABEL_STR
    wd_source: RelationPropertyWDSource
    allow_multiple: bool = True
    is_qualifier: bool = True
    datatype: DataType = DataType.STR
    quantity_unit: Optional[str] = None
    quantity_convert_unit: bool = False
    is_year: bool = False
    disable: bool = False

    @model_validator(mode="after")
    def validate(self):
        if self.quantity_unit is not None and self.datatype != DataType.FLOAT:
            raise ValueError(f"quantity_unit is only allowed for datatype=float")
        if self.is_year and self.datatype not in (DataType.STR, DataType.INT):
            raise ValueError("is_year is only allowed for datatype str or int")
        return self


class WDRelationSchema(BaseModel):
    label: LABEL_STR
    wd_source: RelationWDSource
    wd_source_suffix: Optional[str] = None
    wd_qualifier_pid: Optional[str] = None
    wd_is_transitive: bool = False
    wd_rank_filter: Literal["highest", "non_deprecated"] = "highest"
    subj_label: str
    obj_label: str
    properties: List[WDRelationPropertySchema] = []
    wd_sparql: Optional[str] = None
    wd_sparql_heads: List[str] = []
    disable: bool = False

    @model_validator(mode="after")
    def validate(self):
        if self.wd_is_transitive:
            if self.wd_qualifier_pid is not None:
                raise ValueError(
                    f"wd_qualifier_pid is not allowed when wd_is_transitive=True"
                )
            if len(self.properties) > 0:
                raise ValueError(
                    f"Relation properties are not allowed when wd_is_transitive=True"
                )
        return self


class WDNeo4jSchema(BaseModel):
    model_config = ConfigDict(extra="forbid", validate_default=True)

    name: str
    entities: List[WDEntitySchema]
    relations: List[WDRelationSchema]

    @model_validator(mode="after")
    def validate_relation_subj_obj(self):
        for relation in self.relations:
            if not any(entity.label == relation.subj_label for entity in self.entities):
                raise ValueError(
                    f'Entity "{relation.subj_label}" not found in entities'
                )
            if not any(entity.label == relation.obj_label for entity in self.entities):
                raise ValueError(f'Entity "{relation.obj_label}" not found in entities')
        return self

    @model_validator(mode="after")
    def validate_relation_uniqueness(self):
        seen = set()
        for i, relation in enumerate(self.relations):
            s = f"{relation.label}#{relation.subj_label}#{relation.obj_label}"
            if s in seen:
                raise ValueError(
                    f'Duplicate relation "{relation.label}"({relation.subj_label}->{relation.obj_label}) found'
                )
            seen.add(s)
        return self

    @model_validator(mode="after")
    def validate_entity_uniqueness(self):
        for i, entity in enumerate(self.entities):
            if any(entity.label == e.label for e in self.entities[:i]):
                raise ValueError(f'Duplicate entity "{entity.label}" found')
        return self

    @model_validator(mode="after")
    def validate_FROM_RELATION(self):
        rlabel2r = {r.label: r for r in self.relations}
        rschema2r = {
            f"{r.label}#{r.subj_label}#{r.obj_label}": r for r in self.relations
        }
        for entity in self.entities:
            for constraint in entity.wd_constraints:
                if re.match(Op.FROM_RELATION, constraint):
                    _, relations = Op.parse(constraint)
                    for rlabel in relations:
                        if rlabel not in rlabel2r and rlabel not in rschema2r:
                            raise ValueError(
                                f'Relation "{rlabel}" not found in relations'
                            )
                        r = (
                            rschema2r[rlabel]
                            if rlabel in rschema2r
                            else rlabel2r[rlabel]
                        )
                        if entity.label not in [r.subj_label, r.obj_label]:
                            raise ValueError(
                                f'Entity "{entity.label}" not found in relation "{rlabel}"'
                            )
        return self


class WDEntity(BaseModel):
    label: LABEL_STR
    wikidata_qid: str
    name: str
    description: Optional[str]
    aliases: List[str]
    enwiki_title: Optional[str]
    properties: Dict[str, Any]


class WDRelation(BaseModel):
    label: LABEL_STR
    subj_label: str
    obj_label: str
    subj_wikidata_qid: str
    obj_wikidata_qid: str
    properties: Dict[str, Any]


class WikidataKG(BaseModel):
    schema: WDNeo4jSchema
    entities: List[WDEntity]
    relations: List[WDRelation]

    @model_validator(mode="after")
    def validate_schema(self):
        label2entities = collections.defaultdict(dict)
        schema2relations = collections.defaultdict(list)
        for entity in self.entities:
            label2entities[entity.label][entity.wikidata_qid] = entity
        for relation in self.relations:
            schema2relations[
                f"{relation.label}#{relation.subj_label}#{relation.obj_label}"
            ].append(relation)
        label2eschema = {e.label: e for e in self.schema.entities}
        schema2rschema = {
            f"{r.label}#{r.subj_label}#{r.obj_label}": r for r in self.schema.relations
        }

        # Validate relation schemas
        for rschema, relations in schema2relations.items():
            if rschema not in schema2rschema:
                raise ValueError(f'Relation "{rschema}" not found in schema')
            rschema = schema2rschema[rschema]
            for r in relations:
                if r.subj_wikidata_qid not in label2entities[rschema.subj_label]:
                    raise ValueError(
                        f"Entity {r.subj_wikidata_qid} of {r.label} not found in entities with label {rschema.subj_label}"
                    )
                if r.obj_wikidata_qid not in label2entities[rschema.obj_label]:
                    raise ValueError(
                        f"Entity {r.obj_wikidata_qid} of {r.label} not found in entities with label {rschema.obj_label}"
                    )

        # Validate entities
        for elabel, entities in label2entities.items():
            if elabel not in label2eschema:
                raise ValueError(f'Entity "{elabel}" not found in schema')
            eschema = label2eschema[elabel]
            label2pschema = {
                prop_schema.label: prop_schema for prop_schema in eschema.properties
            }
            for ent in entities.values():
                # convert dates to datetime.date
                for key, pschema in label2pschema.items():
                    if pschema.datatype == DataType.DATE and isinstance(
                        ent.properties.get(key), str
                    ):
                        ent.properties[key] = datetime.datetime.strptime(
                            ent.properties[key], "%Y-%m-%d"
                        ).date()
                for key, value in ent.properties.items():
                    if key not in label2pschema:
                        raise ValueError(
                            f'Entity "{ent.wikidata_qid}" has unknown property "{key}"'
                        )
                    pschema = label2pschema[key]
                    if not pschema.datatype.validate(value):
                        raise ValueError(
                            f'Property "{pschema.label}" of entity "{ent.wikidata_qid}" has wrong datatype'
                        )
        return self

    def print_stats(self):
        ent_cnt = {}
        rel_cnt = {}
        ent_prop_cnt = {}
        rel_prop_cnt = {}
        for es in self.schema.entities:
            ent_cnt[es.label] = 0
            ent_prop_cnt[es.label] = {}
            for prop in es.properties:
                ent_prop_cnt[es.label][prop.label] = 0
        for rs in self.schema.relations:
            rel_cnt[f"{rs.label}({rs.subj_label}->{rs.obj_label})"] = 0
            rel_prop_cnt[f"{rs.label}({rs.subj_label}->{rs.obj_label})"] = {}
            for prop in rs.properties:
                rel_prop_cnt[f"{rs.label}({rs.subj_label}->{rs.obj_label})"][
                    prop.label
                ] = 0

        for ent in self.entities:
            ent_cnt[ent.label] += 1
            for prop in ent.properties:
                ent_prop_cnt[f"{ent.label}"][prop] += 1
        for rel in self.relations:
            rel_cnt[f"{rel.label}({rel.subj_label}->{rel.obj_label})"] += 1
            for prop in rel.properties:
                rel_prop_cnt[f"{rel.label}({rel.subj_label}->{rel.obj_label})"][
                    prop
                ] += 1

        print()
        print(f"### Entities (total: {len(self.entities)})")
        for elabel, num in ent_cnt.items():
            print(f"- {elabel}: {num}")
            if len(ent_prop_cnt[elabel]) > 0:
                for prop, num in ent_prop_cnt[elabel].items():
                    print(f"  * {prop}: {num}")
        print()
        print(f"### Relations (total: {len(self.relations)})")
        for rlabel, num in rel_cnt.items():
            print(f"- {rlabel}: {num}")
            if len(rel_prop_cnt[rlabel]) > 0:
                for prop, num in rel_prop_cnt[rlabel].items():
                    print(f"  * {prop}: {num}")


if __name__ == "__main__":
    testing_schemas = [
        # 'wd2neo4j/sample_schemas/nba_mini.json',
        "wd2neo4j/sample_schemas/movie_mini.json",
    ]
    for schema_path in testing_schemas:
        with open(schema_path) as f:
            schema = WDNeo4jSchema(**json.load(f))
        print(schema.entities[0].properties)
