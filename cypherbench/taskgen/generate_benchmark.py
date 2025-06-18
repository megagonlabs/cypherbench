import argparse
import collections
from concurrent.futures import ThreadPoolExecutor, as_completed
import math
import time
import hashlib
import os
from tqdm import tqdm
import threading
import random
import uuid
import json
import datetime
from string import Template
import neo4j
from cypherbench.schema import *
from cypherbench.neo4j_connector import Neo4jConnector
from cypherbench.data_utils import print_benchmark_distribution

MATCH_INSTANTIATION_CYPHER = """
${match_clause}${where_clause}
${with_clause}
WITH ${all_vars}, [row IN all_properties | {names: row, hash: apoc.util.md5(row + [${all_vars}, '${category}', ${seed}])}] AS all_properties
WITH ${all_vars}, apoc.coll.sortMaps(all_properties, 'hash') AS all_properties
WITH ${all_vars}, [row IN all_properties | row.names][..${num_samples}] AS all_properties
RETURN *
""".strip()

COMPARISON_MATCH_INSTANTIATION_CYPHER = "MATCH (n:${etype}) WHERE n.${prop} IS NOT NULL RETURN n.name, n.${prop} ORDER BY apoc.util.md5([n.name, '${etype}', '${prop}', ${seed}]) LIMIT ${limit}"


def escape_and_quote(value):
    if isinstance(value, (datetime.date, neo4j.time.Date)):
        value = f"date('{value}')"
    elif isinstance(value, str):
        value = value.replace("'", "\\'")
        value = f"'{value}'"
    else:
        value = str(value)
    return value


def round_number(value, mode: Literal["floor", "ceil"], top_n_digits=2):
    """
    mode=floor: 232623 -> 230000, 26123.2 -> 26000, -324 -> -330
    mode=ceil: 232623 -> 240000, 26123.2 -> 27000, -324 -> -320
    """
    assert isinstance(value, (int, float))
    if value == 0:
        return value
    else:
        sign = 1 if value > 0 else -1
        abs_value = abs(value)
        exponent = math.floor(math.log10(abs_value)) + 1 - top_n_digits
        shifted = abs_value / (10**exponent)
        if mode == "floor" and sign == 1 or mode == "ceil" and sign == -1:
            rounded = math.floor(shifted)
        else:
            rounded = math.ceil(shifted)
        result = int(sign * rounded * (10**exponent))
        return result


def sample_op_value(ops: List[str], ordered_values: List[str], do_round=True, rng=None):
    """
    Sample an operator and a value from the given list of operators and ordered values,
    such that the condition "X <op> <value>" yields a non-empty result.
    """
    assert len(ordered_values) >= 2
    assert all(op in ["<", ">", "<=", ">=", "=", "<>"] for op in ops)
    assert rng is not None
    # rng = rng if rng is not None else random
    op = rng.sample(ops, 1)[0]
    if op in ("<", "<="):
        value = rng.sample(ordered_values[1:], 1)[0]
        if do_round and isinstance(value, (int, float)):
            value = round_number(value, "ceil", 2)
    elif op in (">", ">="):
        value = rng.sample(ordered_values[:-1], 1)[0]
        if do_round and isinstance(value, (int, float)):
            value = round_number(value, "floor", 2)
    elif op == "<>":
        value = rng.sample(ordered_values, 1)[0]
    else:
        value = rng.sample(ordered_values, 1)[0]
    return op, value


SAMPLE_CONFIG = {
    "train": {
        "max_sample_per_match_category": 10000,
        "num_name_per_sample": collections.defaultdict(
            lambda: 10,
            {
                "basic_(n)-(m0)-(m1*)": 5,
                "basic_(n)-(m0*),(n)-(m1*)": 5,
                "special_union": 5,
            },
        ),
        "max_return_per_match": collections.defaultdict(
            lambda: 10,
            {
                "basic_(n)-(m0)-(m1*)": 5,
                "basic_(n)-(m0*),(n)-(m1*)": 5,
                "special_union": 5,
            },
        ),
    },
    "test": {
        "max_sample_per_match_category": 60,
        "num_name_per_sample": collections.defaultdict(
            lambda: 2,
            {
                "basic_(n)-(m0)-(m1*)": 1,
                "basic_(n)-(m0*),(n)-(m1*)": 1,
                "special_union": 1,
            },
        ),
        "max_return_per_match": collections.defaultdict(
            lambda: 2,
            {
                "basic_(n)-(m0)-(m1*)": 1,
                "basic_(n)-(m0*),(n)-(m1*)": 1,
                "special_union": 1,
            },
        ),
    },
}


class Nl2CypherBenchmarkGenerator:
    def __init__(
        self,
        neo4j_conn: Neo4jConnector,
        graph_info: GraphInfo,
        config: Nl2CypherGeneratorConfig,
        random_seed: int = 42,
        categorical_threshold: float = 0.1,
        sample_config_id: str = "test",
        num_threads: Optional[int] = None,
        debug=False,
    ):
        self.config = config
        self.random_seed = random_seed
        self.neo4j_conn = neo4j_conn
        self.graph_info = graph_info
        self.max_sample_per_match_category = SAMPLE_CONFIG[sample_config_id][
            "max_sample_per_match_category"
        ]
        self.num_name_samples = SAMPLE_CONFIG[sample_config_id]["num_name_per_sample"]
        self.max_return_per_match = SAMPLE_CONFIG[sample_config_id][
            "max_return_per_match"
        ]
        self.num_threads = num_threads
        self.debug = debug

        schema = self.neo4j_conn.get_schema(
            exclude_properties=["eid", "description", "aliases", "provenance"],
            map_to_categorical=True,
            categorical_threshold=categorical_threshold,
        )
        self.etype2props = {
            e.label: e.properties for e in schema.entities
        }  # etype -> prop -> dtype

        self.rel_info: Dict[str, RelationInfo] = {
            r.label: r for r in self.graph_info.relations
        }

        self.return_func_mapping = {
            "n_name": self._return_n_name,
            "n_prop": self._return_n_prop,
            "n_name_prop": self._return_n_name_prop,
            "n_prop_distinct": self._return_n_prop_distinct,
            "n_prop_array_distinct": self._return_n_prop_array_distinct,
            "n_order_by": self._return_n_order_by,
            "n_argmax": self._return_n_argmax,
            "n_where": self._return_n_where,
            "n_agg": self._return_n_agg,
            "n_group_by": self._return_n_group_by,
            "n_name_special": self._return_n_name,
            "n_m0_group_by_count": self._return_n_m0_group_by_count,
            "n_m0_group_by_count_where": self._return_n_m0_group_by_count_where,
            "n_m0_comparison_argmax": self._return_n_m0_comparison,
            "n_m0_comparison_boolean": self._return_n_m0_comparison,
            "n_m0_comparison_arithmetic": self._return_n_m0_comparison,
            "n_union_count": self._return_n_union,
            "n_union_name": self._return_n_union,
        }
        self._thread_local = threading.local()

    @property
    def rng(self):
        """
        The random module is not thread-safe, so we need to create a separate random instance for each thread
        The random generator will be stored in _thread_local.rng
        This will be initialized in _instantiate_match and _instantiate_return
        """
        if not hasattr(self._thread_local, "rng"):
            self._thread_local.rng = random.Random()
        return self._thread_local.rng

    def _instantiate_return(self, match_clause: MatchClause) -> List[Nl2CypherSample]:
        # Note that simply self.rng.seed(self.random_seed) doesn't work and
        # will result in the same distribution across instantiation for different match clauses
        self.rng.seed(
            self.random_seed
            + int(hashlib.md5(match_clause.cypher.encode()).hexdigest(), 16)
        )
        res = []
        for rp in self.config.return_patterns:
            if rp.category == match_clause.pattern.return_category:
                if self._has_unnecessary_leaf(match_clause, rp):
                    if self.debug:
                        print(
                            f"Skipping {Template(rp.cypher).safe_substitute(match_cypher=match_clause.cypher)} (unnecessary leaf)"
                        )
                    continue
                func = self.return_func_mapping[rp.pattern_id]
                res += func(match_clause, rp)

        max_return_num = self.max_return_per_match[match_clause.pattern.category]
        if len(res) > max_return_num:
            sampled = []
            pattern2samples = collections.defaultdict(list)
            self.rng.shuffle(res)
            for item in res:
                pattern2samples[item.from_template.return_pattern_id].append(item)
            while len(sampled) < max_return_num:
                pattern_id = self.rng.choice(
                    [k for k, vs in pattern2samples.items() if len(vs) > 0]
                )
                sampled.append(pattern2samples[pattern_id].pop(0))
            return sampled
        return res

    def _get_num_result(self, cypher: str) -> int:
        return len(self.neo4j_conn.run_query(cypher))

    def _to_sample(self, cypher, nl_question, mc: MatchClause, rp: ReturnPattern):
        return Nl2CypherSample(
            qid="",  # qid will affect the sampling result, so we set it in the end
            graph=self.neo4j_conn.name,
            gold_cypher=cypher,
            gold_match_cypher=mc.cypher,
            nl_question=None,
            nl_question_raw=nl_question,
            from_template=TemplateInfo(
                match_category=mc.pattern.category,
                match_cypher=mc.pattern.cypher,
                return_pattern_id=rp.pattern_id,
                return_cypher=rp.cypher,
            ),
            num_result=self._get_num_result(cypher),
        )

    def _return_n_name(
        self, mc: MatchClause, rp: ReturnPattern
    ) -> List[Nl2CypherSample]:
        if "n.name" in mc.pattern.property_constraints:
            return []
        return [
            self._to_sample(
                Template(rp.cypher).substitute(match_cypher=mc.cypher),
                Template(rp.nl_template).substitute(**mc.nl_args),
                mc,
                rp,
            )
        ]

    def _return_n_prop(
        self, mc: MatchClause, rp: ReturnPattern
    ) -> List[Nl2CypherSample]:
        if "n.name" not in mc.pattern.property_constraints:
            return []
        res = []
        for prop, dtype in self.etype2props[mc.n_assignment["n"]].items():
            if prop == "name" or f"n.{prop}" in mc.pattern.property_constraints:
                continue
            cypher = Template(rp.cypher).substitute(match_cypher=mc.cypher, prop=prop)
            nl_question = Template(rp.nl_template).substitute(**mc.nl_args, prop=prop)
            res.append(self._to_sample(cypher, nl_question, mc, rp))
        return res

    def _return_n_name_prop(
        self, mc: MatchClause, rp: ReturnPattern
    ) -> List[Nl2CypherSample]:
        if "n.name" in mc.pattern.property_constraints:
            return []
        res = []
        for prop, dtype in self.etype2props[mc.n_assignment["n"]].items():
            if prop == "name" or f"n.{prop}" in mc.pattern.property_constraints:
                continue
            cypher = Template(rp.cypher).substitute(match_cypher=mc.cypher, prop=prop)
            nl_question = Template(rp.nl_template).substitute(**mc.nl_args, prop=prop)
            res.append(self._to_sample(cypher, nl_question, mc, rp))
        return res

    def _return_n_prop_distinct(
        self, mc: MatchClause, rp: ReturnPattern
    ) -> List[Nl2CypherSample]:
        if mc.num_matched_node < 2:
            return []
        res = []
        for prop, dtype in self.etype2props[mc.n_assignment["n"]].items():
            if prop == "name" or f"n.{prop}" in mc.pattern.property_constraints:
                continue
            if dtype != DataType.CATEGORICAL:
                continue
            cypher = Template(rp.cypher).substitute(match_cypher=mc.cypher, prop=prop)
            nl_question = Template(rp.nl_template).substitute(**mc.nl_args, prop=prop)
            res.append(self._to_sample(cypher, nl_question, mc, rp))
        return res

    def _return_n_prop_array_distinct(
        self, mc: MatchClause, rp: ReturnPattern
    ) -> List[Nl2CypherSample]:
        if "n.name" in mc.pattern.property_constraints or mc.num_matched_node < 2:
            return []
        res = []
        for prop, dtype in self.etype2props[mc.n_assignment["n"]].items():
            if prop == "name" or f"n.{prop}" in mc.pattern.property_constraints:
                continue
            if dtype != DataType.STR_ARRAY:
                continue
            cypher = Template(rp.cypher).substitute(match_cypher=mc.cypher, prop=prop)
            nl_question = Template(rp.nl_template).substitute(**mc.nl_args, prop=prop)
            res.append(self._to_sample(cypher, nl_question, mc, rp))
        return res

    def _return_n_order_by(
        self, mc: MatchClause, rp: ReturnPattern
    ) -> List[Nl2CypherSample]:
        if "n.name" in mc.pattern.property_constraints:
            return []
        if mc.num_matched_node < 2:
            return []
        etype = mc.n_assignment["n"]
        res = []
        for prop, dtype in self.etype2props[etype].items():
            if dtype in [DataType.INT, DataType.FLOAT, DataType.DATE]:
                order = self.rng.sample(["ASC", "DESC"], 1)[0]
                cypher = Template(rp.cypher).substitute(
                    match_cypher=mc.cypher, prop=prop, order=order
                )
                nl_question = Template(rp.nl_template).substitute(
                    **mc.nl_args,
                    prop=prop,
                    order_nl="ascending" if order == "ASC" else "descending",
                )
                res.append(self._to_sample(cypher, nl_question, mc, rp))
        return res

    def _return_n_argmax(
        self, mc: MatchClause, rp: ReturnPattern
    ) -> List[Nl2CypherSample]:
        if "n.name" in mc.pattern.property_constraints:
            return []
        if mc.num_matched_node < 2:
            return []
        etype = mc.n_assignment["n"]
        res = []
        for prop, dtype in self.etype2props[etype].items():
            if dtype in [DataType.INT, DataType.FLOAT, DataType.DATE]:
                order = self.rng.sample(["ASC", "DESC"], 1)[0]
                cypher = Template(rp.cypher).substitute(
                    match_cypher=mc.cypher, prop=prop, order=order
                )
                nl_question = Template(rp.nl_template).substitute(
                    **mc.nl_args,
                    prop=prop,
                    order_nl="lowest" if order == "ASC" else "highest",
                )
                res.append(self._to_sample(cypher, nl_question, mc, rp))
        return res

    def _get_distinct_values(self, match_cypher: str, var: str, is_array=False):
        if is_array:
            cypher = f"{match_cypher} UNWIND {var} as value RETURN DISTINCT value ORDER BY value"
        else:
            cypher = f"{match_cypher} WITH {var} as value WHERE value IS NOT NULL RETURN DISTINCT value ORDER BY value"
        return [record["value"] for record in self.neo4j_conn.run_query(cypher)]

    def _return_n_where(
        self, mc: MatchClause, rp: ReturnPattern
    ) -> List[Nl2CypherSample]:
        # We consider properties with the following types valid for WHERE clause:
        # - int, float, date
        # - str that is categorical (i.e. has less than 50 unique values)
        # TODO: support bool
        if "n.name" in mc.pattern.property_constraints:
            return []
        if mc.num_matched_node < 2:
            return []
        res = []
        for prop, dtype in self.etype2props[mc.n_assignment["n"]].items():
            if prop == "name" or f"n.{prop}" in mc.pattern.property_constraints:
                continue

            distinct_values = self._get_distinct_values(
                mc.cypher, f"n.{prop}", is_array=dtype == DataType.STR_ARRAY
            )
            if dtype == DataType.INT:
                distinct_values = [v for v in distinct_values if v > 2]
            if len(distinct_values) < 2:
                continue

            condition, condition_nl = None, None
            if dtype == DataType.INT:
                if prop.endswith("year") or prop.endswith("time"):
                    op, value = sample_op_value(
                        ["<", ">", "="], distinct_values, rng=self.rng, do_round=False
                    )
                else:
                    op, value = sample_op_value(
                        ["<", ">", "<=", ">=", "=", "<>"],
                        distinct_values,
                        rng=self.rng,
                        do_round=True,
                    )
            elif dtype == DataType.FLOAT:
                op, value = sample_op_value(
                    ["<", ">", "<=", ">="], distinct_values, rng=self.rng, do_round=True
                )
            elif dtype == DataType.DATE:
                distinct_years = self._get_distinct_values(mc.cypher, f"n.{prop}.year")
                if len(distinct_years) >= 2 and self.rng.random() < 0.5:
                    prop = f"{prop}.year"
                    op, value = sample_op_value(
                        ["<", ">", "="], distinct_years, rng=self.rng, do_round=False
                    )
                else:
                    op, value = sample_op_value(
                        ["<", ">", "="], distinct_values, rng=self.rng
                    )
            elif dtype == DataType.STR:
                op, value = sample_op_value(["=", "<>"], distinct_values, rng=self.rng)
                condition = f"n.{prop} {op} {escape_and_quote(value)}"
                condition_nl = (
                    f'{prop} is "{value}"' if op == "=" else f'{prop} is not "{value}"'
                )
            elif dtype == DataType.STR_ARRAY:
                op, value = sample_op_value(["=", "<>"], distinct_values, rng=self.rng)
                if op == "=":
                    condition = f"{escape_and_quote(value)} IN n.{prop}"
                    condition_nl = f'have "{value}" as {prop}'
                else:
                    condition = f"NOT {escape_and_quote(value)} IN n.{prop}"
                    condition_nl = f'{prop} is not "{value}"'
            else:
                continue

            if condition is None:
                condition = f"n.{prop} {op} {escape_and_quote(value)}"
                condition_nl = f"{prop} {op} {value}"

            cypher = Template(rp.cypher).substitute(
                match_cypher=mc.cypher, condition=condition
            )
            nl_question = Template(rp.nl_template).substitute(
                **mc.nl_args, condition_nl=condition_nl
            )
            res.append(self._to_sample(cypher, nl_question, mc, rp))
        return res

    def _return_n_agg(
        self, mc: MatchClause, rp: ReturnPattern
    ) -> List[Nl2CypherSample]:
        if "n.name" in mc.pattern.property_constraints:
            return []
        if mc.num_matched_node < 2:
            return []
        etype = mc.n_assignment["n"]
        res = []
        for prop, dtype in self.etype2props[etype].items():
            if dtype == DataType.INT:
                agg_op = self.rng.sample(["min", "max"], 1)[0]
                agg_clause = f"{agg_op}(n.{prop})"
            elif dtype == DataType.FLOAT:
                agg_op = self.rng.sample(["min", "max", "avg"], 1)[0]
                agg_clause = f"{agg_op}(n.{prop})"
            elif dtype == DataType.DATE:
                agg_op = self.rng.sample(["min", "max"], 1)[0]
                if self.rng.random() < 0.5:
                    prop = f"{prop}.year"
                agg_clause = f"{agg_op}(n.{prop})"
            elif prop == "name":
                agg_op = "count"
                agg_clause = f"count(DISTINCT n)"
            elif dtype == DataType.CATEGORICAL and prop != "name":
                # agg_op = self.rng.sample(['count', 'collect'], 1)[0]
                agg_op = self.rng.sample(["count"], 1)[0]
                if agg_op == "collect":
                    agg_clause = f"collect(DISTINCT n.{prop})"
                else:
                    agg_clause = f"count(DISTINCT n.{prop})"
            else:  # TODO: support string array
                continue
            agg_op2nl = {
                "min": "minimum",
                "max": "maximum",
                "avg": "average",
                "count": "number",
                "collect": "list",
            }
            if prop == "name":
                agg_clause_nl = "number"
            else:
                agg_clause_nl = f"{agg_op2nl[agg_op]} of {prop}"
            cypher = Template(rp.cypher).substitute(
                match_cypher=mc.cypher, agg_clause=agg_clause
            )
            nl_question = Template(rp.nl_template).substitute(
                **mc.nl_args, agg_clause_nl=agg_clause_nl
            )
            res.append(self._to_sample(cypher, nl_question, mc, rp))

        return res

    def _return_n_group_by(
        self, mc: MatchClause, rp: ReturnPattern
    ) -> List[Nl2CypherSample]:
        if "n.name" in mc.pattern.property_constraints:
            return []
        if mc.num_matched_node < 2:
            return []
        etype = mc.n_assignment["n"]
        res = []
        for group_by_key, dtype in self.etype2props[etype].items():
            if dtype != DataType.CATEGORICAL:
                continue
            if len(self._get_distinct_values(mc.cypher, f"n.{group_by_key}")) < 2:
                continue
            for prop, dtype in self.etype2props[etype].items():
                if prop == group_by_key:
                    continue
                if dtype in (DataType.INT, DataType.FLOAT):
                    agg_op = self.rng.sample(["min", "max", "avg"], 1)[0]
                    agg_clause = f"{agg_op}(n.{prop})"
                elif dtype == DataType.DATE:
                    agg_op = self.rng.sample(["min", "max"], 1)[0]
                    if self.rng.random() < 0.5:
                        prop = f"{prop}.year"
                    agg_clause = f"{agg_op}(n.{prop})"
                elif prop == "name":
                    agg_op = "count"
                    agg_clause = f"count(DISTINCT n)"
                    prop = etype
                elif dtype == DataType.CATEGORICAL and prop != "name":
                    # agg_op = self.rng.sample(['count', 'collect'], 1)[0]
                    agg_op = self.rng.sample(["count"], 1)[0]
                    if agg_op == "collect":
                        agg_clause = f"collect(DISTINCT n.{prop})"
                    else:
                        agg_clause = f"count(DISTINCT n.{prop})"
                else:  # TODO: support string array
                    continue
                agg_op2nl = {
                    "min": "minimum",
                    "max": "maximum",
                    "avg": "average",
                    "count": "number",
                    "collect": "list",
                }
                agg_clause_nl = f"{agg_op2nl[agg_op]} of {prop}"
                cypher = Template(rp.cypher).substitute(
                    match_cypher=mc.cypher, key=group_by_key, agg_clause=agg_clause
                )
                nl_question = Template(rp.nl_template).substitute(
                    **mc.nl_args, key=group_by_key, agg_clause_nl=agg_clause_nl
                )
                res.append(self._to_sample(cypher, nl_question, mc, rp))
        return res

    def _has_unnecessary_leaf(self, mc: MatchClause, rp: ReturnPattern) -> bool:
        """
        We check whether the MATCH pattern has an unnecessary leaf node
        For example, in `MATCH (n:Company)-[r0:foundedBy]->(m0:Person)` m0 is unnecessary since foundedBy is mandatory for Company
        However, `MATCH (n:Person)<-[r0:foundedBy]-(m0:Company)` is acceptable
        Also, `MATCH (n:Company)-[r0:foundedBy]->(m0:Person)<-[r0:hasCEO]-(m1:Company)` is acceptable since the relation at the end is good
        The criteria of unnecessary relation is defined as follows:
            1. The relation is attached to a leaf node (i.e. a node that has only one edge)
            2. The leaf node is not constrained by name
            3. The leaf node is not in the RETURN clause
            4. The relation is mandatory for the other node

        Note that one subtle case is when a node is connected to only one node via two relations. In such
        cases, we don't consider it as a leaf node, thus the relations remain valid.
            Example: `MATCH (n:Company)-[r0:foundedBy]->(m0:Person), (n)-[r1:hasCEO]->(m1:Person)`
        """
        assert all(n in mc.pattern.node_vars for n in rp.return_vars)

        g = collections.defaultdict(list)
        for r, subj, obj in mc.pattern.relations:
            g[subj].append((mc.r_assignment[r], obj, "out"))
            g[obj].append((mc.r_assignment[r], subj, "in"))

        for n, edges in g.items():
            if (
                len(edges) == 1
                and n not in mc.pattern.node_vars_with_name
                and n not in rp.return_vars
            ):
                rlabel, other, direction = edges[0]
                if (
                    direction == "out"
                    and self.rel_info[rlabel].is_mandatory_obj
                    or direction == "in"
                    and self.rel_info[rlabel].is_mandatory_subj
                ):
                    return True
        return False

    def _has_invalid_group_by(self, mc: MatchClause, rp: ReturnPattern):
        if (
            "(n)-[r0]->(m0)" not in mc.pattern.cypher
            and "(n)<-[r0]-(m0)" not in mc.pattern.cypher
        ):
            raise ValueError("n and m0 must be connected by relation r0")
        if (
            "(n)-[r0]->(m0)" in mc.pattern.cypher
            and self.rel_info[mc.r_assignment["r0"]].obj_cardinality == "one"
        ) or (
            "(n)<-[r0]-(m0)" in mc.pattern.cypher
            and self.rel_info[mc.r_assignment["r0"]].subj_cardinality == "one"
        ):
            return True
        return False

    def _return_n_m0_group_by_count(
        self, mc: MatchClause, rp: ReturnPattern
    ) -> List[Nl2CypherSample]:
        if mc.num_matched_node < 2:
            return []
        if self._has_invalid_group_by(mc, rp):
            if self.debug:
                print(
                    f"Skipping {Template(rp.cypher).safe_substitute(match_cypher=mc.cypher)} (invalid group-by)"
                )
            return []
        return [
            self._to_sample(
                Template(rp.cypher).substitute(match_cypher=mc.cypher),
                Template(rp.nl_template).substitute(**mc.nl_args),
                mc,
                rp,
            )
        ]

    def _return_n_m0_group_by_count_where(
        self, mc: MatchClause, rp: ReturnPattern
    ) -> List[Nl2CypherSample]:
        if mc.num_matched_node < 2:
            return []
        if self._has_invalid_group_by(mc, rp):
            if self.debug:
                print(
                    f"Skipping {Template(rp.cypher).safe_substitute(match_cypher=mc.cypher)} (invalid group-by)"
                )
            return []
        distinct_values = self._get_distinct_values(
            f"{mc.cypher}  WITH n as n, count(DISTINCT m0) as num", "num"
        )
        distinct_values = [v for v in distinct_values if v > 2]
        if len(distinct_values) < 2:
            return []

        op, value = sample_op_value(
            ["<", ">", "<=", ">="], distinct_values, rng=self.rng, do_round=True
        )
        return [
            self._to_sample(
                Template(rp.cypher).substitute(
                    match_cypher=mc.cypher, op=op, value=value
                ),
                Template(rp.nl_template).substitute(**mc.nl_args, op=op, value=value),
                mc,
                rp,
            )
        ]

    def _return_n_m0_comparison(
        self, mc: MatchClause, rp: ReturnPattern
    ) -> List[Nl2CypherSample]:
        allowed_datatypes = {
            "n_m0_comparison_argmax": (DataType.INT, DataType.FLOAT, DataType.DATE),
            "n_m0_comparison_boolean": (DataType.CATEGORICAL, DataType.INT),
            "n_m0_comparison_arithmetic": (DataType.INT, DataType.FLOAT),
        }
        prop = mc.nl_args["prop"]
        if (
            self.etype2props[mc.n_assignment["m0"]][prop]
            not in allowed_datatypes[rp.pattern_id]
        ):
            return []
        if (
            rp.pattern_id == "n_m0_comparison_argmax"
            and mc.prop_values[f"n_{prop}"] == mc.prop_values[f"m0_{prop}"]
        ):
            assert mc.prop_values[f"n_{prop}"] is not None
            if self.debug:
                print(
                    f"Skipping {Template(rp.cypher).safe_substitute(match_cypher=mc.cypher, prop=mc.nl_args['prop'])}"
                    f" (identical property value)"
                )
            return []
        return [
            self._to_sample(
                Template(rp.cypher).substitute(match_cypher=mc.cypher, prop=prop),
                Template(rp.nl_template).substitute(**mc.nl_args),
                mc,
                rp,
            )
        ]

    def _return_n_union(
        self, mc: MatchClause, rp: ReturnPattern
    ) -> List[Nl2CypherSample]:
        return [
            self._to_sample(
                Template(rp.cypher).substitute(match_cypher=mc.cypher),
                Template(rp.nl_template).substitute(**mc.nl_args),
                mc,
                rp,
            )
        ]

    def _substitute_cypher(
        self, mp: MatchPattern, n_assignment, r_assignment, prop_values
    ) -> str:
        cypher = (
            mp.cypher_clean
        )  # `MATCH (n)-[r0]->(m0<name>)` -> `MATCH (n)-[r0]->(m0)`
        quoted_values = {
            key: escape_and_quote(value) for key, value in prop_values.items()
        }
        for key, value in quoted_values.items():
            if key.endswith("_name"):
                n = key.split("_")[0]
                cypher = cypher.replace(
                    f"({n})", f"({n}:{n_assignment[n]} {{name: {value}}})"
                )
        for n in n_assignment:
            cypher = cypher.replace(f"({n})", f"({n}:{n_assignment[n]})")
        for r in r_assignment:
            cypher = cypher.replace(f"-[{r}]-", f"-[{r}:{r_assignment[r]}]-")
        cypher = Template(cypher).substitute(**quoted_values)
        return cypher

    def _substitute_nl(
        self, mp: MatchPattern, n_assignment, r_assignment, prop_values
    ) -> dict[str, str]:
        res = {}
        for key, nl_template in mp.nl_args.items():
            params = (
                {
                    k: f'"{v}"' if isinstance(v, str) else str(v)
                    for k, v in prop_values.items()
                }
                | {f"{n}_LABEL": n_assignment[n] for n in n_assignment}
                | {f"{r}_LABEL": r_assignment[r] for r in r_assignment}
            )
            for r, rlabel in r_assignment.items():
                if (
                    all(
                        key not in prop_values
                        for key in (
                            f"{r}_start_year",
                            f"{r}_end_year",
                            f"{r}_start_time",
                            f"{r}_end_time",
                        )
                    )
                    and self.rel_info[rlabel].is_time_sensitive
                ):
                    params[f"{r}_LABEL"] += " (at some point, past or present)"
            res[key] = Template(nl_template).substitute(**params)
        return res

    def _has_redundant_relation(
        self, mp: MatchPattern, r_assignment: dict[str, str]
    ) -> bool:
        """
        We check whether the MATCH pattern has two relations between the same nodes
        where either they are the same or one implies the other
            Example: `MATCH (n:Player)-[r0:draftedBy]->(m0:Team), (n)-[r1:playsFor]->(m0)`
        In the above case, the `playsFor` relation is redundant thus the pattern should be skipped
        """
        triples = {(r_assignment[r], subj, obj) for r, subj, obj in mp.relations}
        if len(triples) != len(mp.relations):
            return True
        for r, subj, obj in triples:
            # TODO: this is an initial solution, we haven't considered transitivity yet
            if any(
                (implied_r, subj, obj) in triples
                for implied_r in self.rel_info[r].implied_relations
            ):
                return True
        return False

    def _has_unnecessary_one_to_x_pair(
        self, mp: MatchPattern, r_assignment: dict[str, str]
    ) -> bool:
        """
        We check whether the MATCH pattern with two consecutive relations with type A
        followed by the inverse of A, where A is a one-to-one or one-to-many
        relation. We also consider symmetry here.
            Example: `MATCH (n:Character)-[r0:hasSpouse]->(m0:Character)-[r1:hasSpouse]->(m1:Character {name: 'Rhaenyra Targaryen'})`
                - We use the symmetry of `hasSpouse` here.
            Example: `MATCH (n:Character)<-[r0:hasMother]-(m0:Character)-[r1:hasMother]->(m1:Character)`
                - However, `MATCH (n:Character)-[r0:hasMother]->(m0:Character)<-[r1:hasMother]-(m1:Character)`
                  which selects the maternal siblings of n is valid
        """
        g = collections.defaultdict(list)
        for r, subj, obj in mp.relations:
            g[subj].append((r_assignment[r], obj, "out"))
            g[obj].append((r_assignment[r], subj, "in"))

        for n, edges in g.items():
            for rlabel, other, direction in edges:
                if (
                    direction == "out"
                    and self.rel_info[rlabel].obj_cardinality == "one"
                ):
                    for d in (
                        ("out", "in")
                        if self.rel_info[rlabel].is_symmetric
                        else ("out",)
                    ):
                        if any(
                            (rlabel, n1, d) in edges
                            for n1 in mp.node_vars
                            if n1 != other
                        ):
                            return True
                if (
                    direction == "in"
                    and self.rel_info[rlabel].subj_cardinality == "one"
                ):
                    for d in (
                        ("out", "in") if self.rel_info[rlabel].is_symmetric else ("in",)
                    ):
                        if any(
                            (rlabel, n1, d) in edges
                            for n1 in mp.node_vars
                            if n1 != other
                        ):
                            return True
        return False

    def _instantiate_comparison_match(self, mp: MatchPattern) -> list[MatchClause]:
        assert mp.cypher == "MATCH (n<name,$prop>),(m0<name,$prop>)"
        assert set(mp.nl_args) == {"n_desc", "m0_desc", "prop"}
        res = []
        for etype in self.etype2props:
            for prop in self.etype2props[etype]:
                if prop == "name":
                    continue
                cypher = Template(COMPARISON_MATCH_INSTANTIATION_CYPHER).substitute(
                    etype=etype,
                    prop=prop,
                    seed=self.random_seed,
                    limit=self.num_name_samples[mp.category] * 2,
                )
                result = self.neo4j_conn.run_query(cypher)
                for i in range(0, len(result), 2):
                    if i + 1 >= len(result):
                        break
                    n_name = result[i]["n.name"]
                    n_prop = result[i][f"n.{prop}"]
                    m0_name = result[i + 1]["n.name"]
                    m0_prop = result[i + 1][f"n.{prop}"]
                    cypher = Template(
                        "MATCH (n:${etype} {name: ${n_name}}), (m0:${etype} {name: ${m0_name}})"
                    ).substitute(
                        etype=etype,
                        n_name=escape_and_quote(n_name),
                        m0_name=escape_and_quote(m0_name),
                    )
                    nl_args = {
                        "n_desc": f'"{n_name}"',
                        "m0_desc": f'"{m0_name}"',
                        "prop": prop,
                    }
                    res.append(
                        MatchClause(
                            pattern=mp,
                            n_assignment={"n": etype, "m0": etype},
                            r_assignment={},
                            prop_values={
                                "n_name": n_name,
                                "m0_name": m0_name,
                                f"n_{prop}": n_prop,
                                f"m0_{prop}": m0_prop,
                            },
                            cypher=cypher,
                            nl_args=nl_args,
                            num_matched_node=2,
                        )
                    )
        return res

    def _instantiate_match(self, mp: MatchPattern) -> list[MatchClause]:
        self.rng.seed(self.random_seed)

        if mp.category == "special_comparison":
            return self._instantiate_comparison_match(mp)

        params = {}
        params["match_clause"] = mp.cypher_match_pattern
        constraints = [
            constraint
            for constraint in mp.property_constraints.values()
            if constraint is not None
        ]
        if len(constraints) > 0:
            params["where_clause"] = " WHERE " + " AND ".join(constraints)
        else:
            params["where_clause"] = ""
        params["with_clause"] = "WITH " + ", ".join(
            [f"HEAD(LABELS({n})) as {n}" for n in mp.node_vars]
            + [f"TYPE({r}) as {r}" for r in mp.rel_vars]
            + [
                f"collect(DISTINCT [{', '.join(list(mp.property_constraints))}]) as all_properties"
            ]
        )
        params["all_vars"] = ", ".join(mp.node_vars + mp.rel_vars)
        params["num_samples"] = self.num_name_samples[mp.category]
        params["category"] = mp.category
        params["seed"] = self.random_seed

        cypher = Template(MATCH_INSTANTIATION_CYPHER).substitute(**params)
        # print(cypher)
        result = self.neo4j_conn.run_query(cypher)
        res = []
        for record in result:
            n_assignment = {n: record[n] for n in mp.node_vars}
            r_assignment = {r: record[r] for r in mp.rel_vars}
            assert all(v is not None for v in n_assignment.values())
            assert all(v is not None for v in r_assignment.values())

            if self._has_redundant_relation(mp, r_assignment):
                if self.debug:
                    print(
                        f"Skipping {self._substitute_cypher(mp, n_assignment, r_assignment, {})}"
                        f" (redundant relations found)"
                    )
                continue

            if self._has_unnecessary_one_to_x_pair(mp, r_assignment):
                if self.debug:
                    print(
                        f"Skipping {self._substitute_cypher(mp, n_assignment, r_assignment, {})}"
                        f" (unnecessary one-to-x relation pair found)"
                    )
                continue

            for prop_values in record["all_properties"]:
                prop_values = {
                    key.replace(".", "_"): val
                    for key, val in zip(mp.property_constraints, prop_values)
                }
                cypher = self._substitute_cypher(
                    mp, n_assignment, r_assignment, prop_values
                )
                nl_args = self._substitute_nl(
                    mp, n_assignment, r_assignment, prop_values
                )
                num_matched_node = self.neo4j_conn.run_query(
                    f"{cypher} RETURN count(DISTINCT n) as num"
                )[0]["num"]
                if num_matched_node == 0:
                    if self.debug:
                        print(f"Skipping {cypher} (no matched node)")
                    continue
                res.append(
                    MatchClause(
                        pattern=mp,
                        n_assignment=n_assignment,
                        r_assignment=r_assignment,
                        prop_values=prop_values,
                        cypher=cypher,
                        nl_args=nl_args,
                        num_matched_node=num_matched_node,
                    )
                )
        return res

    def generate_benchmark(
        self, show_progress_bar: bool = True
    ) -> list[Nl2CypherSample]:
        for rp in self.config.return_patterns:
            if rp.pattern_id not in self.return_func_mapping:
                raise ValueError(f"Return pattern id {rp.pattern_id} is not supported")

        mp_order = {
            mp.category: idx for idx, mp in enumerate(self.config.match_patterns)
        }
        rp_order = {
            rp.pattern_id: idx for idx, rp in enumerate(self.config.return_patterns)
        }

        with ThreadPoolExecutor(max_workers=self.num_threads) as executor:
            match_futures = [
                executor.submit(self._instantiate_match, mp)
                for mp in self.config.match_patterns
            ]

            # Collect match clauses results as they complete
            match_clauses = []
            for future in tqdm(
                as_completed(match_futures),
                desc="Instantiating MATCH",
                total=len(self.config.match_patterns),
                disable=not show_progress_bar,
            ):
                match_clauses += future.result()

        # Instantiate RETURN clauses in parallel
        res = []
        with ThreadPoolExecutor(max_workers=self.num_threads) as executor:
            return_futures = [
                executor.submit(self._instantiate_return, mc) for mc in match_clauses
            ]

            # Collect return clauses results as they complete
            for future in tqdm(
                as_completed(return_futures),
                desc="Instantiating RETURN",
                total=len(match_clauses),
                disable=not show_progress_bar,
            ):
                res += future.result()

        # Ensure deterministic ordering
        res = sorted(
            res,
            key=lambda x: (
                x.graph,
                mp_order[x.from_template.match_category],
                rp_order[x.from_template.return_pattern_id],
                x.gold_cypher,
            ),
        )

        # Downsample if necessary
        random.seed(self.random_seed)
        random.shuffle(res)
        category2samples = collections.defaultdict(list)
        for item in res:
            category2samples[item.from_template.match_category].append(item)
        for category, samples in category2samples.items():
            if len(samples) > self.max_sample_per_match_category:
                category2samples[category] = random.sample(
                    samples, self.max_sample_per_match_category
                )
        res = [item for samples in category2samples.values() for item in samples]
        res = sorted(
            res,
            key=lambda x: (
                x.graph,
                mp_order[x.from_template.match_category],
                rp_order[x.from_template.return_pattern_id],
                x.gold_cypher,
            ),
        )

        # Check for duplicate Cyphers (typically there shouldn't be any, but just in case)
        seen = set()
        res_filtered = []
        for item in res:
            if item.gold_cypher in seen:
                print(f"Warning: Duplicate Cypher found: {item.gold_cypher}")
                continue
            seen.add(item.gold_cypher)
            res_filtered.append(item)

        # Add qid
        for item in res:
            item.qid = str(uuid.uuid4())
        return res


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--neo4j_info", default="neo4j_info.json")
    parser.add_argument("--graph_info", default="graph_info.json")
    parser.add_argument("--config", default="nl2cypher_generator_config.json")
    parser.add_argument("--graphs", default=None, nargs="+")
    parser.add_argument("--output_path", default="output/benchmark.json")
    # parser.add_argument('--num_name_samples', default=2, type=int)
    # parser.add_argument('--max_return_per_match', default=2, type=int)
    parser.add_argument("--sample_config_id", default="test")
    parser.add_argument("--num_threads", default=None, type=int)
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()
    print(args)
    print()

    if not os.path.exists(os.path.dirname(args.output_path)):
        os.makedirs(os.path.dirname(args.output_path))

    with open(args.neo4j_info) as fin:
        neo4j_info = json.load(fin)

    with open(args.graph_info) as fin:
        graph_info = json.load(fin)
        graph_info = {graph: GraphInfo(**info) for graph, info in graph_info.items()}

    with open(args.config) as fin:
        config = Nl2CypherGeneratorConfig(**json.load(fin))

    domains = (
        (neo4j_info["train_domains"] + neo4j_info["test_domains"])
        if args.graphs is None
        else args.graphs
    )

    t0 = time.time()

    benchmark = []
    for graph in domains:
        info = neo4j_info["sampled"][graph]

        print()
        print(f"Generating examples for {graph}")
        neo4j_conn = Neo4jConnector(
            name=graph,
            host=info["host"],
            port=info["port"],
            username=info["username"],
            password=info["password"],
            debug=False,
        )
        generator = Nl2CypherBenchmarkGenerator(
            neo4j_conn=neo4j_conn,
            graph_info=graph_info[graph],
            config=config,
            sample_config_id=args.sample_config_id,
            num_threads=args.num_threads,
            random_seed=args.seed,
            debug=args.debug,
        )
        examples = generator.generate_benchmark()
        print(f"Generated {len(examples)} examples for {graph}")
        benchmark += examples

    print()
    print(f"Benchmark generation finished in {time.time() - t0:.2f} seconds")
    print()
    print(f"Benchmark statistics:")
    print(f"Total number of examples: {len(benchmark)}")
    print()
    print_benchmark_distribution(benchmark, config)

    with open(args.output_path, "w") as fout:
        json.dump([ex.model_dump(mode="json") for ex in benchmark], fout, indent=2)
    print(f"Saved examples to {args.output_path}")


if __name__ == "__main__":
    main()
