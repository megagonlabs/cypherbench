from typing import List, Any, Dict, Tuple
import re
import collections
import copy
from cypherbench.wd2neo4j.schema import *


class SPARQLResolver:
    @staticmethod
    def topo_sort(nodes: List[Any], g: Dict[Any, List[Tuple[str, Any]]]) -> List[Any]:
        """
        Topological sort of a graph represented by a dictionary of dependencies.
        Ensure all nodes are accounted for in the dependencies dictionary.
        """
        # Ensure every node is in the dictionary even if it has no outgoing edges
        g1 = {node: g.get(node, []) for node in nodes}

        # Calculate in-degrees
        indegree = {node: 0 for node in nodes}
        for deps in g1.values():
            for _, dep in deps:
                indegree[dep] += 1

        # Initialize the queue with nodes having zero in-degree
        queue = [node for node, deg in indegree.items() if deg == 0]
        order = []

        while queue:
            node = queue.pop(0)
            order.append(node)
            for _, dep in g1[node]:  # Safely access dependencies
                indegree[dep] -= 1
                if indegree[dep] == 0:
                    queue.append(dep)

        # Check if topological sort is possible (i.e., no cycles)
        if len(order) != len(nodes):
            raise ValueError('Graph contains a cycle')

        return order

    def resolve(self, schema: WDNeo4jSchema) -> WDNeo4jSchema:
        schema = copy.deepcopy(schema)
        elabel2e = {e.label: e for e in schema.entities}
        rlabel2rs = collections.defaultdict(list)
        for r in schema.relations:
            rlabel2rs[r.label].append(r)
        rschema2r = {f'{r.label}#{r.subj_label}#{r.obj_label}': r for r in schema.relations}

        # 1. Resolve $FROM_RELATION
        from_relation = collections.defaultdict(list)  # entity label -> relation objects
        for e in schema.entities:
            for constraint in e.wd_constraints:
                if re.match(Op.FROM_RELATION, constraint):
                    _, rlabels = Op.parse(constraint)
                    from_relation[e.label] = []
                    for r in rlabels:
                        if r in rschema2r:
                            from_relation[e.label].append(rschema2r[r])
                        else:
                            if len(rlabel2rs[r]) > 1:
                                raise ValueError(f'Found multiple relations with label {r}')
                            from_relation[e.label].append(rlabel2rs[r][0])

        # 2. Construct dependency graph
        # note: A -> (predicate, B) means A needs to be resolved before B
        # sample dependency graph:
        #     "Film" -> ("P123", "Award")
        #     "Award" -> ("^P456", "Person")
        g = collections.defaultdict(list)  # We use list instead of set for reproducibility
        g_reverse = collections.defaultdict(list)
        for elabel, rs in from_relation.items():
            for r in rs:
                assert re.match(r'^P\d+$', r.wd_source) or r.wd_source == '$IDENTICAL'
                if r.wd_source == '$IDENTICAL':
                    predicate = None
                else:
                    if r.wd_qualifier_pid is not None:
                        if r.obj_label == elabel:
                            predicate = f'(p:{r.wd_source}/pq:{r.wd_qualifier_pid})'
                        elif r.subj_label == elabel:
                            predicate = f'((^pq:{r.wd_qualifier_pid})/(^p:{r.wd_source}))'
                    else:
                        if r.obj_label == elabel:
                            if r.wd_rank_filter == 'highest':
                                predicate = f'wdt:{r.wd_source}'
                            else:
                                predicate = f'(p:{r.wd_source}/ps:{r.wd_source})'
                        elif r.subj_label == elabel:
                            if r.wd_rank_filter == 'highest':
                                predicate = f'(^wdt:{r.wd_source})'
                            else:
                                predicate = f'((^ps:{r.wd_source})/(^p:{r.wd_source}))'
                    if r.wd_is_transitive:
                        predicate += '*'
                parent = r.subj_label if r.obj_label == elabel else r.obj_label
                if (predicate, elabel) not in g[parent]:
                    g[parent].append((predicate, elabel))
                if (predicate, parent) not in g_reverse[elabel]:
                    g_reverse[elabel].append((predicate, parent))

        topo_order = self.topo_sort([e.label for e in schema.entities], g)  # ensure the graph is a DAG
        label2eschema = {e.label: e for e in schema.entities}
        schema.entities = [label2eschema[label] for label in topo_order]  # topologically sorted entities

        # Utility function to traverse a DAG graph
        def dfs(g, node, visited=None, leaf_only=False):
            if visited is None:
                visited = set()
            if node in visited:
                return
            if not leaf_only or len(g[node]) == 0:
                yield node
            visited.add(node)
            for _, dep in g[node]:
                yield from dfs(g, dep, visited, leaf_only)

        # Utility function to get paths from src to trg
        def get_predicate_paths(g, src, trg, path=[]):
            if src == trg:
                yield path.copy()
            else:
                for predicate, dep in g[src]:
                    yield from get_predicate_paths(g, dep, trg, path + [predicate])

        # 3. Resolve entity SPARQL
        for e in schema.entities:
            if e.wd_is_instance:
                base_predicate = 'wdt:P31'
            else:
                base_predicate = 'wdt:P279'
            if e.wd_include_subclasses:
                e.wd_sparql = f'?item {base_predicate}/wdt:P279* wd:{e.wd_source}.'
            else:
                e.wd_sparql = f'?item {base_predicate} wd:{e.wd_source}.'

        for e in schema.entities:
            for constraint in e.wd_constraints:
                if re.match(Op.LINKED_TO, constraint):
                    if len(g[e.label]) > 0 and len(g_reverse[e.label]) > 0:
                        raise ValueError(
                            f'$LINKED_TO constraint cannot be applied on entities with incoming and outgoing dependencies: {e.label}')
                    _, (pid, qid) = Op.parse(constraint)
                    e.wd_sparql += f' ?item wdt:{pid} wd:{qid}.'
                elif re.match(Op.HAS_PROPERTY, constraint):
                    if len(g[e.label]) > 0 and len(g_reverse[e.label]) > 0:
                        raise ValueError(
                            f'$HAS_PROPERTY constraint cannot be applied on entities with incoming and outgoing dependencies: {e.label}')
                    _, (pid,) = Op.parse(constraint)
                    e.wd_sparql += f' ?item wdt:{pid} [].'
                elif re.match(Op.RAW_SPARQL, constraint):
                    if len(g[e.label]) > 0 and len(g_reverse[e.label]) > 0:
                        raise ValueError(
                            f'$RAW_SPARQL constraint cannot be applied on entities with incoming and outgoing dependencies: {e.label}')
                    _, (sparql,) = Op.parse(constraint)
                    sparql = sparql.strip()
                    if not sparql.endswith('.'):
                        sparql += '.'
                    e.wd_sparql += ' ' + sparql

        for e in schema.entities:

            if len(from_relation[e.label]) == 0:
                continue
            # First, we need to find all roots of the dependency graph
            roots = list(dfs(g_reverse, e.label, leaf_only=True))
            e.wd_sparql_roots = roots
            if len(roots) > 1:
                paths = list(get_predicate_paths(g, roots[0], e.label))
                root_sparql = elabel2e[roots[0]].wd_sparql
                root_qid = elabel2e[roots[0]].wd_source
                root_source_qids = [root_qid]
                for root1 in roots[1:]:
                    paths1 = list(get_predicate_paths(g, root1, e.label))
                    root1_sparql = elabel2e[root1].wd_sparql
                    root1_qid = elabel2e[root1].wd_source
                    if paths1 != paths or root1_sparql != re.sub(rf'wd:{root_qid}\b', f'wd:{root1_qid}', root_sparql):
                        raise ValueError(f'Entity {e.label} has multiple roots with different paths: {roots}')
                    root_source_qids.append(root1_qid)
                root_sparql = root_sparql.replace('?item', '?root')
                root_sparql = re.sub(rf'wd:{root_qid}\b', f'?rootsource', root_sparql)
                root_sparql += ' VALUES ?rootsource { ' + ' '.join(f'wd:{qid}' for qid in root_source_qids) + ' }.'
                e.wd_sparql += ' ' + root_sparql
                predicate = '|'.join([('(' + '/'.join(p) + ')') if len(p) > 1 else p[0] for p in paths])
                e.wd_sparql += f' ?root {predicate} ?item.'
            else:
                root, = roots
                e.wd_sparql += ' ' + elabel2e[root].wd_sparql.replace('?item', '?root')
                # Then, we need to find all predicate paths from the root to the entity
                paths = list(get_predicate_paths(g, root, e.label))
                # remove $IDENTICAL from the paths
                paths = [[n for n in path if n is not None] for path in paths]
                if any(len(path) == 0 for path in paths):
                    raise ValueError(f'There is a path with length 0 from {root} to {e.label}')
                predicate = '|'.join([('(' + '/'.join(p) + ')') if len(p) > 1 else p[0] for p in paths])
                e.wd_sparql += f' ?root {predicate} ?item.'

        # 4. Resolve relation SPARQL
        for r in schema.relations:
            subj_is_parent_of_obj = r.obj_label in set(dfs(g, r.subj_label))
            obj_is_parent_of_subj = r.subj_label in set(dfs(g, r.obj_label))
            conditions = []
            subj_sparql = elabel2e[r.subj_label].wd_sparql.replace('?item', '?subj').replace('?root', '?root1')
            obj_sparql = elabel2e[r.obj_label].wd_sparql.replace('?item', '?obj').replace('?root', '?root2')
            if not subj_is_parent_of_obj and not obj_is_parent_of_subj:
                r.wd_sparql_heads = ['subj', 'obj']
                conditions.append(subj_sparql)
                conditions.append(obj_sparql)
            elif subj_is_parent_of_obj and not obj_is_parent_of_subj:
                r.wd_sparql_heads = ['subj']
                conditions.append(subj_sparql)
            elif not subj_is_parent_of_obj and obj_is_parent_of_subj:
                r.wd_sparql_heads = ['obj']
                conditions.append(obj_sparql)
            else:
                assert r.subj_label == r.obj_label
                r.wd_sparql_heads = ['subj']
                conditions.append(subj_sparql)

            if re.match(Op.IDENTICAL, r.wd_source):
                conditions.append('FILTER(?subj = ?obj) BIND("IDENTICAL" AS ?statement)')
            elif re.match(r'^P\d+$', r.wd_source):
                if r.wd_is_transitive:
                    conditions.append(f'?subj wdt:{r.wd_source}* ?obj. BIND("TRANSITIVE" AS ?statement)')
                    if len(r.properties) > 0:
                        raise ValueError(f'Transitive relation {r.label} cannot have properties')
                    if r.wd_rank_filter == 'non_deprecated':
                        raise ValueError('Transitive relation cannot have wd_rank_filter="non_deprecated"')
                    if r.wd_qualifier_pid is not None:
                        raise ValueError('Transitive relation cannot have wd_qualifier_pid')
                elif r.wd_source_suffix:
                    conditions.append(
                        f'?subj wdt:{r.wd_source}/{r.wd_source_suffix} ?obj. BIND("WITH_SUFFIX" AS ?statement)')
                    if len(r.properties) > 0:
                        raise ValueError(f'Relation {r.label} with suffix cannot have properties')
                    if r.wd_rank_filter == 'non_deprecated':
                        raise ValueError('Relation with suffix cannot have wd_rank_filter="non_deprecated"')
                    if r.wd_qualifier_pid is not None:
                        raise ValueError('Relation with suffix cannot have wd_qualifier_pid')
                else:
                    if r.wd_qualifier_pid is not None and r.wd_rank_filter == 'highest':
                        raise ValueError(
                            'wd_qualifier_pid=true and wd_rank_filter="highest" cannot be used together')
                    if r.wd_qualifier_pid is not None:
                        pred = f'pq:{r.wd_qualifier_pid}'
                    else:
                        pred = f'ps:{r.wd_source}'
                    if r.wd_rank_filter == 'highest':
                        cons = f'?subj wdt:{r.wd_source} ?obj1. FILTER(?obj1 = ?obj)'
                    elif r.wd_rank_filter == 'non_deprecated':
                        cons = '?statement wikibase:rank ?rank. FILTER(?rank != wikibase:DeprecatedRank)'
                    else:
                        raise ValueError(f'Unknown rank constraint: {r.wd_rank_filter}')
                    conditions.append(
                        f'?subj p:{r.wd_source} ?statement. ?statement {pred} ?obj. {cons}')
            else:
                raise ValueError(f'Unsupported relation source {r.wd_source}')

            r.wd_sparql = ' '.join(conditions)

        # 5. Remove disabled elements
        entities = [e for e in schema.entities if not e.disable]
        for e in entities:
            e.properties = [p for p in e.properties if not p.disable]
        elabels = {e.label for e in entities}
        relations = [r for r in schema.relations if
                     not r.disable and r.subj_label in elabels and r.obj_label in elabels]
        for r in relations:
            r.properties = [p for p in r.properties if not p.disable]
        return WDNeo4jSchema(name=schema.name, entities=entities, relations=relations)
