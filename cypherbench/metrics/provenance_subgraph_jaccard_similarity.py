import re
from itertools import chain
from itertools import product
from collections import defaultdict
import random
import math
import logging
import time
import neo4j
import neo4j.graph
from pydantic import BaseModel
from typing import List, Tuple, Dict, Set, Optional, Any, Literal
from cypherbench.neo4j_connector import Neo4jConnector

logger = logging.getLogger(__name__)


def split_by_union(cypher: str) -> List[str]:
    # Regex pattern to match UNION and split the cypher
    pattern = r'\bUNION\b'

    # Check if the cypher starts with CALL and has UNION inside it
    if cypher.strip().startswith("CALL"):
        # Extract the portion inside the CALL {} and handle UNION inside
        inner_query_match = re.search(r'CALL\s*\{(.*?)\}\s*(WITH|RETURN|WHERE|UNWIND)', cypher, re.DOTALL)
        if inner_query_match:
            inner_query = inner_query_match.group(1)
            # Split the inner queries by UNION
            split_inner_queries = re.split(pattern, inner_query)
            return [q.strip() for q in split_inner_queries]
        else:
            # If no UNION found, return the entire cypher
            return [cypher.strip()]
    else:
        # If no CALL block, split by UNION directly
        split_queries = re.split(pattern, cypher)
        return [q.strip() for q in split_queries]


def split_cypher_into_clauses(cypher_query: str) -> list:
    """
    Splits a Cypher query into its constituent clauses.

    Args:
        cypher_query (str): The Cypher query to split.

    Returns:
        list: A list of clauses as strings.

    Example:
        Input:  "MATCH (p:Player)-[:playsFor]->(t:Team) WHERE t.name = 'Chicago Bulls' RETURN p.name, t.name"
        Output: ['MATCH (p:Player)-[:playsFor]->(t:Team)', 'WHERE t.name = 'Chicago Bulls'', 'RETURN p.name, t.name']
    """
    # Define a regex pattern for Cypher clauses
    clause_pattern = r'\b(MATCH|OPTIONAL MATCH|WHERE|RETURN|UNION|WITH|CREATE|SET|DELETE|MERGE|UNWIND|ORDER BY|LIMIT|SKIP|FOREACH|CALL|YIELD)\b'

    # Find clause headers and their positions
    matches = list(re.finditer(clause_pattern, cypher_query))

    # Extract clauses based on their positions
    clauses = []
    for i, match in enumerate(matches):
        start = match.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(cypher_query)
        clauses.append(cypher_query[start:end].strip())

    return clauses


def extract_match_cypher(cypher: str) -> Optional[str]:
    if not cypher.startswith('MATCH'):
        return None

    clauses = split_cypher_into_clauses(cypher)

    match_clauses = []
    for clause in clauses:
        if not any(clause.startswith(keyword) for keyword in ['MATCH', 'OPTIONAL MATCH', 'WITH', 'WHERE']):
            break
        if clause.startswith('WITH'):
            if ' as ' in clause.lower():
                break
            else:  # We allow non-aliased WITH clauses between MATCH and WHERE
                match_clauses.append('WITH *')
        else:
            match_clauses.append(clause)

    # Remove trailing WITH clauses
    while match_clauses and match_clauses[-1].startswith('WITH'):
        match_clauses.pop()

    return ' '.join(match_clauses)


def add_variables(match_cypher: str) -> str:
    """
    Adds temporary variables to relationships and nodes in a Cypher MATCH query.

    Args:
        match_cypher (str): Input Cypher MATCH query.

    Returns:
        str: Updated Cypher MATCH query with temporary variables.

    Example:
        Input:  MATCH (p:Player)-[:playsFor]-(:Team {name:"Sacramento Kings"})
        Output: MATCH (p:Player)-[rtmp0:playsFor]-(ntmp0:Team {name:"Sacramento Kings"})
    """
    node_counter = 0
    relationship_counter = 0

    # Function to replace nodes with unique variables
    def replace_node(match):
        nonlocal node_counter
        replacement = f"(ntmp{node_counter}:{match.group(2)}{match.group(3) or ''})"
        node_counter += 1
        return replacement

    # Function to replace relationships with unique variables
    def replace_relationship(match):
        nonlocal relationship_counter
        replacement = f"[rtmp{relationship_counter}{match.group(2)}]"
        relationship_counter += 1
        return replacement

    clauses = split_cypher_into_clauses(match_cypher)
    for i, clause in enumerate(clauses):
        if clause.startswith('MATCH') or clause.startswith('OPTIONAL MATCH'):
            # Replace relationships first
            clause = re.sub(r'(\[)(:.*?)(\])', replace_relationship, clause)

            # Replace nodes with unique variables
            clauses[i] = re.sub(r'(\(:)([A-Za-z]+)(\s*\{.*?\})?\)', replace_node, clause)

    return ' '.join(clauses)


def extract_node_variables(match_cypher: str) -> List[str]:
    # Replace property blocks with dummy text
    match_cypher = re.sub(r'\{[^}]*\}', '{dummy}', match_cypher)

    pattern = r'\((\w+)(?::[^\)]*|\))'
    vars = []
    clauses = split_cypher_into_clauses(match_cypher)
    for clause in clauses:
        if clause.startswith('MATCH') or clause.startswith('OPTIONAL MATCH'):
            vars += re.findall(pattern, clause)
    return sorted(list(set(vars)))


def extract_relationship_variables(match_cypher: str) -> List[str]:
    pattern = r'-\[(\w+)(?::|\])'
    vars = []
    clauses = split_cypher_into_clauses(match_cypher)
    for clause in clauses:
        if clause.startswith('MATCH') or clause.startswith('OPTIONAL MATCH'):
            vars += re.findall(pattern, clause)
    return sorted(list(set(vars)))


def get_ps_cypher(cypher: str, return_var='elemId', node_element_id_only=False) -> str:
    """
    Get the Cypher for fetching the provenance subgraph of the given Cypher query.

    node_element_id_only:
        If True, return only the elementIds of the nodes in the provenance subgraph (used for computing PSJS efficiently)
        If False, return the entire provenance subgraph as a neo4j.graph.Graph object (used for visualization)
    """
    logger.debug(f'Getting provenance subgraph for cypher: {cypher}')

    cyphers = split_by_union(cypher)
    ps_cyphers = []
    for sub_cypher in cyphers:
        match_cypher = extract_match_cypher(sub_cypher)
        if match_cypher:
            match_cypher = add_variables(match_cypher)
            logger.debug(f'match_cypher: {match_cypher}')
            node_vars = extract_node_variables(match_cypher)
            rel_vars = extract_relationship_variables(match_cypher)
            if node_element_id_only:
                node_expr = ' + '.join(f'collect(distinct elementId({var}))' for var in node_vars)
                node_expr = node_expr if node_expr else "[]"
                ps_cyphers.append(
                    f'{match_cypher} WITH {node_expr} AS elemIds UNWIND elemIds AS elemId RETURN elemId AS {return_var}')
            else:
                node_expr = ' + '.join(f'collect(distinct {var})' for var in node_vars)
                node_expr = node_expr if node_expr else "[]"
                rel_expr = ' + '.join(f'collect(distinct {var})' for var in rel_vars)
                rel_expr = rel_expr if rel_expr else "[]"
                ps_cyphers.append(f'{match_cypher} RETURN {node_expr} AS nodes, {rel_expr} AS relationships')

    if len(ps_cyphers) == 0:
        print(f'No MATCH clause found in cypher: {cypher}')
        if node_element_id_only:
            ps_cypher = f'UNWIND [] AS elemId RETURN elemId AS {return_var}'
        else:
            ps_cypher = 'RETURN [] AS nodes, [] AS relationships'
    else:
        ps_cypher = ' UNION '.join(ps_cyphers)
    logger.debug(f'Provenance subgraph cypher: {ps_cypher}')
    return ps_cypher


def provenance_subgraph_jaccard_similarity(pred_cypher: str,
                                           target_cypher: str,
                                           neo4j_connector: Neo4jConnector,
                                           timeout: int = 120) -> float:
    if pred_cypher == target_cypher:
        return 1.0

    target_ps_cypher = get_ps_cypher(target_cypher, node_element_id_only=True, return_var='elemId1')
    pred_ps_cypher = get_ps_cypher(pred_cypher, node_element_id_only=True, return_var='elemId2')

    # psjs_cypher = f'CALL {{ {target_ps_cypher} }} CALL {{ {pred_ps_cypher} }} ' \
    #               'WITH collect(DISTINCT elemId1) as target_ps, collect(DISTINCT elemId2) as pred_ps ' \
    #               'WITH size(apoc.coll.intersection(target_ps, pred_ps)) as I, size(apoc.coll.union(target_ps, pred_ps)) as U ' \
    #               'RETURN CASE WHEN U = 0 THEN 0.0 ELSE 1.0 * I / U END as PSJS'
    # logger.debug(f'PSJS cypher: {psjs_cypher}')

    try:
        result = neo4j_connector.run_query(target_ps_cypher, timeout=None)
        target_ps = set(record['elemId1'] for record in result)
        result = neo4j_connector.run_query(pred_ps_cypher, timeout=timeout)
        pred_ps = set(record['elemId2'] for record in result)
        I = len(target_ps.intersection(pred_ps))
        U = len(target_ps.union(pred_ps))
        psjs = I / U if U > 0 else 0.0
    except Exception as e:
        print(f'When evaluating PSJS encountered exception: {e}')
        return 0.0
    return psjs
